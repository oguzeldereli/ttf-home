// ttf::Model: load a .ttf file; trits rows5 -> TQ1_P (K padded to 160); side tensors and float weights in D.

#include "ttf.h"

#include "ggml-cpu.h"
#ifdef TTF_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdlib>
#include <tuple>
#include <cstring>
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <stdexcept>

namespace ttf {

namespace {

constexpr int64_t QK = 160;  // TQ1_P block

float bf16_round(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        return v;
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

std::string gguf_str(const gguf_context * g, const std::string & key, const std::string & dflt) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) {
        return dflt;
    }
    return gguf_get_val_str(g, id);
}

// the subparts of one ttf.linear node, with their K
std::vector<std::pair<std::string, int64_t>> linear_parts(const json & node) {
    const json &                                 a = node.at("attr");
    std::vector<std::pair<std::string, int64_t>> out;
    const std::string                            p = node.at("p");
    std::vector<std::string>                     subs;
    if (a.value("adapter", false)) {
        subs = { ".lin", ".ad.0", ".ad.1" };
    } else {
        subs = { "" };
    }
    for (const auto & s : subs) {
        int64_t K = a.contains("Ks") ? a.at("Ks").at(s).get<int64_t>() : a.at("K").get<int64_t>();
        out.emplace_back(p + s, K);
    }
    return out;
}

}  // namespace

void Model::use_threads(int n_threads) const {
    if (device != "cpu") {
        return;
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    if (getenv("TTF_NO_POOL")) {
        return;
    }
    auto it = pools.find(n_threads);
    if (it == pools.end()) {
        ggml_threadpool_params tp = ggml_threadpool_params_default(n_threads);
        if (const char * e = getenv("TTF_POLL")) {
            tp.poll = (uint32_t) atoi(e);
        }
        it = pools.emplace(n_threads, ggml_threadpool_new(&tp)).first;
    }
    ggml_backend_cpu_set_threadpool(backend, it->second);
}

Model::~Model() {
    if (backend && device == "cpu") {
        ggml_backend_cpu_set_threadpool(backend, nullptr);
    }
    for (auto & p : pools) {
        ggml_threadpool_free(p.second);
    }
    if (dctx) {
        ggml_free(dctx);
    }
    if (wctx) {
        ggml_free(wctx);
    }
    if (meta) {
        ggml_free(meta);
    }
    if (gguf) {
        gguf_free(gguf);
    }
    for (auto * b : bufs) {
        ggml_backend_buffer_free(b);
    }
    if (backend) {
        ggml_backend_free(backend);
    }
}

std::string Model::kv_str(const std::string & key, const std::string & dflt) const {
    return gguf_str(gguf, key, dflt);
}

ggml_tensor * Model::get(const std::string & name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

std::unique_ptr<Model> Model::load(const std::string & path, std::string * err, const std::string & device) {
    auto m = std::make_unique<Model>();
    try {
        m->device = device;
        if (device == "cpu") {
            m->backend = ggml_backend_cpu_init();
        } else if (device == "cuda") {
#ifdef TTF_CUDA
            m->backend = ggml_backend_cuda_init(0);
#endif
            if (!m->backend) {
                throw std::runtime_error("no CUDA backend (build with GGML_CUDA=ON)");
            }
        } else {
            throw std::runtime_error("unknown device " + device);
        }
        gguf_init_params gp = { /*no_alloc =*/false, /*ctx =*/&m->meta };
        m->gguf             = gguf_init_from_file(path.c_str(), gp);
        if (!m->gguf) {
            throw std::runtime_error("cannot read " + path);
        }
        if (m->kv_str("general.architecture") != "ttf") {
            throw std::runtime_error("not a .ttf file (general.architecture != ttf)");
        }
        m->family = m->kv_str("ttf.family");
        m->ir     = json::parse(m->kv_str("ttf.ir", "{}"));
        m->config = json::parse(m->kv_str("ttf.config", "{}"));
        const std::string D = m->kv_str("ttf.compute_dtype", "f32");
        if (D != "bf16" && D != "f32") {
            throw std::runtime_error("compute_dtype " + D + ": not supported yet");
        }
        m->d_bf16 = D == "bf16";
        const std::string layout = m->kv_str("ttf.layout.trits", "rows5");
        if (layout != "rows5") {
            throw std::runtime_error("trit layout " + layout + ": not supported yet");
        }
        for (ggml_tensor * t = ggml_get_first_tensor(m->meta); t; t = ggml_get_next_tensor(m->meta, t)) {
            m->tensors[ggml_get_name(t)] = t;
        }
        // runtime trit layout: Q2P on the CPU (default), TQ1_P with TTF_TRITS=tq1p
        // TQ1_P on CUDA: memory-bound there, 1.6 vs 2 bits per trit
        const char * lt = std::getenv("TTF_TRITS");
        m->q2p          = lt ? std::string(lt) == "q2p" : device == "cpu";
        const int64_t QB = m->q2p ? 128 : QK;

        // pass 1: shapes of the derived tensors
        struct Pending {
            std::string   part;
            int64_t       K;
            ggml_tensor * packed;
        };
        std::vector<Pending>                                       pend;
        std::vector<std::pair<std::string, const ggml_tensor *>> heads;
        std::vector<std::pair<std::string, std::string>>          fder;  // (source weight, derived suffix)
        std::vector<std::tuple<std::string, json, bool>>          intk;  // int-k embeddings / heads: (p, attr, head)
        std::vector<std::string>                                  snakes;
        for (const auto & [gname, g] : m->ir.at("graphs").items()) {
            for (const auto & n : g.at("nodes")) {
                const std::string op = n.at("op");
                if (op == "ttf.linear") {
                    for (const auto & [p, K] : linear_parts(n)) {
                        if (m->parts.count(p)) {
                            continue;
                        }
                        ggml_tensor * wp = m->get(p + ".wpacked");
                        if (!wp) {
                            throw std::runtime_error("missing " + p + ".wpacked");
                        }
                        if (wp->ne[0] != (K + 4) / 5) {
                            throw std::runtime_error(p + ".wpacked: ne0 " + std::to_string(wp->ne[0]) + " != ceil(K/5)");
                        }
                        Part & P = m->parts[p];
                        P.K      = K;
                        P.Kp     = (K + QB - 1) / QB * QB;
                        P.N      = wp->ne[1];
                        pend.push_back({ p, K, wp });
                    }
                } else if (op == "conv_transpose1d_causal") {  // weight as [Cin, k Cout] for one mul_mat + overlap-add
                    fder.emplace_back(n.at("p").get<std::string>() + ".weight", ".weight.ct");
                } else if (op == "matmul" || op == "conv1d_causal" ||
                           (op == "rms_norm" && n.contains("p") && n.at("attr").value("offset", false))) {
                    const std::string p    = n.at("p");
                    const std::string kind = op == "matmul" ? ".weight.F32" : op == "rms_norm" ? ".weight.off1" : ".weight.conv";
                    if (!m->get(p + ".weight")) {
                        throw std::runtime_error("missing " + p + ".weight");
                    }
                    fder.emplace_back(p + ".weight", kind);
                } else if ((op == "ttf.embed" || op == "ttf.head") && n.value("attr", json::object()).value("kind", std::string("float")) == "intk") {
                    const std::string p = n.at("p");
                    intk.emplace_back(p, n.at("attr"), op == "ttf.head");
                } else if (op == "ttf.head") {
                    const std::string p = n.at("p");
                    if (!m->get(p + ".weight")) {
                        throw std::runtime_error("missing " + p + ".weight");
                    }
                    heads.emplace_back(p, m->get(p + ".weight"));
                } else if (op == "snake") {
                    snakes.push_back(n.at("p").get<std::string>());
                }
            }
        }
        const ggml_type     dt = m->d_bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        size_t n_meta = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(m->meta); t; t = ggml_get_next_tensor(m->meta, t)) {
            ++n_meta;
        }
        const size_t        nt = pend.size() * 8 + heads.size() + fder.size() + n_meta + intk.size() + snakes.size() + 8;
        ggml_init_params    ip = { ggml_tensor_overhead() * nt, nullptr, /*no_alloc =*/true };
        m->wctx                = ggml_init(ip);
        std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> dvec;   // (D copy, source) of side vectors
        std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> dmat;   // (D-typed copy, source), maybe transposed
        std::vector<bool>                                          dmat_t;
        auto side = [&](const std::string & name) -> ggml_tensor * {
            ggml_tensor * s = m->get(name);
            if (!s) {
                return nullptr;
            }
            if (s->type != GGML_TYPE_F32 && s->type != GGML_TYPE_BF16 && s->type != GGML_TYPE_F16) {
                throw std::runtime_error(name + ": side tensor type");
            }
            if (!m->d_bf16 && s->type == GGML_TYPE_F32) {
                return s;
            }
            ggml_tensor * d = ggml_new_tensor(m->wctx, GGML_TYPE_F32, GGML_MAX_DIMS, s->ne);
            ggml_set_name(d, (name + ".D").c_str());
            dvec.emplace_back(d, s);
            return d;
        };
        for (auto & pe : pend) {
            Part & P = m->parts[pe.part];
            P.w = m->q2p ? ggml_new_tensor_2d(m->wctx, GGML_TYPE_I8, P.Kp / 4, P.N)
                         : ggml_new_tensor_2d(m->wctx, GGML_TYPE_TQ1_P, P.Kp, P.N);
            ggml_set_name(P.w, (pe.part + (m->q2p ? ".q2p" : ".tq1p")).c_str());
            P.beta = m->get(pe.part + ".beta");
            if (!P.beta || P.beta->type != GGML_TYPE_F32) {
                throw std::runtime_error(pe.part + ".beta: missing or not F32");
            }
            P.mu = side(pe.part + ".in_mean");
            P.c  = side(pe.part + ".col_scale");
            P.r  = side(pe.part + ".row_scale");
            P.b  = side(pe.part + ".out_bias");
            const ggml_tensor * mA = m->get(pe.part + ".mag_A");
            const ggml_tensor * mB = m->get(pe.part + ".mag_B");
            if (mA && mB) {
                // file: mag_A torch [N, rank] = ggml [rank, N]; mag_B torch [K, rank] = ggml [rank, K]
                // F32 holding D values: the fused ops read them as floats
                P.mag_A = ggml_new_tensor_2d(m->wctx, GGML_TYPE_F32, mA->ne[0], mA->ne[1]);
                P.mag_B = ggml_new_tensor_2d(m->wctx, GGML_TYPE_F32, mB->ne[1], mB->ne[0]);  // transposed: [K, rank]
                ggml_set_name(P.mag_A, (pe.part + ".mag_A.D").c_str());
                ggml_set_name(P.mag_B, (pe.part + ".mag_B.DT").c_str());
                dmat.emplace_back(P.mag_A, mA);
                dmat_t.push_back(false);
                dmat.emplace_back(P.mag_B, mB);
                dmat_t.push_back(true);
            }
        }
        for (auto & [p, w] : heads) {
            if (m->head_w.count(p)) {
                continue;
            }
            if (w->type == dt) {  // already in D's type
                m->head_w[p] = const_cast<ggml_tensor *>(w);
                continue;
            }
            ggml_tensor * h = ggml_new_tensor_2d(m->wctx, dt, w->ne[0], w->ne[1]);
            ggml_set_name(h, (p + ".head.D").c_str());
            m->head_w[p] = h;
            dmat.emplace_back(h, w);
            dmat_t.push_back(false);
        }
        // float weights the graph wants as F32 in another form: matmul (computed in f32), rms_norm offset (1 + w),
        // depthwise conv [k, C]
        std::vector<std::tuple<ggml_tensor *, const ggml_tensor *, std::string>> fd;
        {
            std::map<std::string, bool> done;
            for (auto & [src, kind] : fder) {
                const std::string name = src.substr(0, src.size() - 7) + kind;
                if (done[name]) {
                    continue;
                }
                done[name]            = true;
                const ggml_tensor * w = m->get(src);
                if (kind == ".weight.F32" && w->type == GGML_TYPE_F32) {
                    continue;
                }
                ggml_tensor * d;
                if (kind == ".weight.conv") {  // depthwise: torch [C, 1, k] = ggml [k, 1, C] -> [k, C]
                    if (w->ne[1] != 1) {
                        continue;  // dense conv reads the weight as stored (F32 copy below)
                    }
                    d = ggml_new_tensor_2d(m->wctx, GGML_TYPE_F32, w->ne[0], w->ne[2]);
                } else if (kind == ".weight.ct") {  // torch [Cin, Cout, k] = ggml [k, Cout, Cin] -> ggml [Cin, k Cout]
                    d = ggml_new_tensor_2d(m->wctx, GGML_TYPE_F32, w->ne[2], w->ne[0] * w->ne[1]);
                } else {
                    d = ggml_new_tensor(m->wctx, GGML_TYPE_F32, GGML_MAX_DIMS, w->ne);
                }
                ggml_set_name(d, name.c_str());
                fd.emplace_back(d, w, kind);
            }
        }
        // small 16-bit float tensors (norm weights, biases, gains) as F32: ggml's elementwise ops want F32 operands.
        // Large ones (embeddings, heads) stay as stored: get_rows / mul_mat read them directly.
        std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> f32c;
        for (ggml_tensor * t = ggml_get_first_tensor(m->meta); t; t = ggml_get_next_tensor(m->meta, t)) {
            if ((t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F16) && ggml_nelements(t) <= (1 << 24)) {
                ggml_tensor * d = ggml_new_tensor(m->wctx, GGML_TYPE_F32, GGML_MAX_DIMS, t->ne);
                ggml_set_name(d, ggml_get_name(t));
                f32c.emplace_back(d, t);
            }
        }
        // int-k (nib2 codes, per-group lo / s): the weight D(lo + s q) in D's type, exact (fp32 mul, then add, then D)
        std::vector<std::tuple<ggml_tensor *, std::string, int64_t>> ikd;
        for (auto & [p, a, head] : intk) {
            if (m->kv_str("ttf.layout.intk", "nib2") != "nib2" || a.at("bits").get<int>() > 4) {
                throw std::runtime_error(p + ": int-k layout / bits not yet");
            }
            const int64_t K = a.at("K"), V = a.at("V"), g = a.at("g");
            ggml_tensor * d = ggml_new_tensor_2d(m->wctx, dt, K, V);
            ggml_set_name(d, (p + ".weight").c_str());
            ikd.emplace_back(d, p, g);
            if (head) {
                m->head_w[p] = d;
            }
        }
        std::vector<std::pair<ggml_tensor *, std::string>> snd;
        for (auto & p : snakes) {
            const ggml_tensor * al = m->get(p + ".alpha");
            if (!al) {
                throw std::runtime_error("missing " + p + ".alpha");
            }
            ggml_tensor * d = ggml_new_tensor(m->wctx, GGML_TYPE_F32, GGML_MAX_DIMS, al->ne);
            ggml_set_name(d, (p + ".alpha.inv").c_str());
            snd.emplace_back(d, p);
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(m->wctx, ggml_backend_cpu_buffer_type());
        if (!buf) {
            throw std::runtime_error("cannot allocate the weights");
        }
        m->bufs.push_back(buf);
#if defined(__linux__)
        {
            // back the weights with transparent huge pages before they are first written: with THP defrag=madvise the
            // fault path only compacts for madvised ranges, and decode streams the weights each token (TLB misses)
            const uintptr_t a = (uintptr_t) ggml_backend_buffer_get_base(buf), e = a + ggml_backend_buffer_get_size(buf);
            const uintptr_t h = 2u << 20, a2 = (a + h - 1) & ~(h - 1), e2 = e & ~(h - 1);
            if (e2 > a2 && !getenv("TTF_NO_THP")) {
                madvise((void *) a2, e2 - a2, MADV_HUGEPAGE);
            }
        }
#endif

        // pass 2: fill
        std::vector<float> row;
        for (auto & pe : pend) {
            Part &         P   = m->parts[pe.part];
            const uint8_t * pk = (const uint8_t *) pe.packed->data;
            const int64_t   nb = pe.packed->ne[0];
            row.assign(P.Kp, 0.0f);
            const size_t rs = m->q2p ? (size_t) P.Kp / 4 : ggml_row_size(GGML_TYPE_TQ1_P, P.Kp);
            for (int64_t n = 0; n < P.N; ++n) {
                const uint8_t * r = pk + n * pe.packed->nb[1];
                for (int64_t i = 0; i < nb; ++i) {
                    int v = r[i];
                    for (int j = 0; j < 5; ++j) {
                        const int64_t k = 5 * i + j;
                        if (k < P.K) {
                            const int d = v % 3;
                            if (d > 2) {
                                throw std::runtime_error(pe.part + ": bad trit byte");
                            }
                            row[k] = (float) (d - 1);
                        }
                        v /= 3;
                    }
                }
                uint8_t * dst = (uint8_t *) P.w->data + n * rs;
                if (m->q2p) {  // element 32 n + b of block i = bits 2n of byte 32 i + b, code = trit + 1
                    std::memset(dst, 0, rs);
                    for (int64_t k = 0; k < P.Kp; ++k) {
                        const int code = (int) row[k] + 1;
                        const int64_t i = k / 128, r = k % 128;
                        dst[32 * i + r % 32] |= (uint8_t) (code << (2 * (r / 32)));
                    }
                } else {
                    ggml_quantize_chunk(GGML_TYPE_TQ1_P, row.data(), dst, 0, 1, P.Kp, nullptr);
                }
            }
        }
        auto f32of = [&](const std::string & name) {  // a float tensor of the file as f32
            const ggml_tensor * t = m->get(name);
            if (!t) {
                throw std::runtime_error("missing " + name);
            }
            std::vector<float> f(ggml_nelements(t));
            if (t->type == GGML_TYPE_F32) {
                std::memcpy(f.data(), t->data, f.size() * 4);
            } else if (t->type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) t->data, f.data(), f.size());
            } else if (t->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, f.data(), f.size());
            } else {
                throw std::runtime_error(name + ": not a float tensor");
            }
            return f;
        };
        for (auto & [d, p, g] : ikd) {
            const ggml_tensor * c  = m->get(p + ".codes");
            if (!c) {
                throw std::runtime_error("missing " + p + ".codes");
            }
            const std::vector<float> lo = f32of(p + ".lo"), sc = f32of(p + ".s");
            const int64_t            K = d->ne[0], V = d->ne[1], G = K / g;
            std::vector<float>       row(K);
            for (int64_t v = 0; v < V; ++v) {
                const uint8_t * cr = (const uint8_t *) c->data + v * c->nb[1];
                for (int64_t k = 0; k < K; ++k) {
                    const int   q  = (k & 1) ? (cr[k / 2] >> 4) : (cr[k / 2] & 15);
                    const float pq = sc[v * G + k / g] * (float) q;
                    row[k]         = lo[v * G + k / g] + pq;
                }
                if (d->type == GGML_TYPE_BF16) {
                    ggml_fp32_to_bf16_row_ref(row.data(), (ggml_bf16_t *) d->data + v * K, K);
                } else {
                    std::memcpy((float *) d->data + v * K, row.data(), K * 4);
                }
            }
        }
        for (auto & [d, p] : snd) {  // torch: (alpha + 1e-9).reciprocal()
            std::vector<float> al = f32of(p + ".alpha");
            for (size_t i = 0; i < al.size(); ++i) {
                ((float *) d->data)[i] = 1.0f / (al[i] + 1e-9f);
            }
        }
        for (auto & [d, t] : f32c) {
            if (t->type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) t->data, (float *) d->data, ggml_nelements(t));
            } else {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, (float *) d->data, ggml_nelements(t));
            }
        }
        for (auto & [d, w, kind] : fd) {
            const int64_t      n = ggml_nelements(w);
            std::vector<float> f(n);
            if (w->type == GGML_TYPE_F32) {
                std::memcpy(f.data(), w->data, n * sizeof(float));
            } else if (w->type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) w->data, f.data(), n);
            } else if (w->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) w->data, f.data(), n);
            } else {
                throw std::runtime_error(std::string(ggml_get_name(w)) + ": float weight type");
            }
            if (kind == ".weight.off1") {
                for (auto & v : f) {
                    v = 1.0f + v;
                }
            }
            if (kind == ".weight.ct") {
                const int64_t      kc = w->ne[0] * w->ne[1], ci = w->ne[2];
                std::vector<float> t(n);
                for (int64_t i = 0; i < ci; ++i) {
                    for (int64_t j = 0; j < kc; ++j) {
                        t[i + ci * j] = f[j + kc * i];
                    }
                }
                f.swap(t);
            }
            std::memcpy(d->data, f.data(), n * sizeof(float));
        }
        for (auto & [d, s] : dvec) {
            float *       dv = (float *) d->data;
            const int64_t n  = ggml_nelements(s);
            if (s->type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) s->data, dv, n);
            } else if (s->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) s->data, dv, n);
            } else {
                std::memcpy(dv, s->data, n * sizeof(float));
            }
            if (m->d_bf16) {
                for (int64_t i = 0; i < n; ++i) {
                    dv[i] = bf16_round(dv[i]);
                }
            }
        }
        for (size_t i = 0; i < dmat.size(); ++i) {
            ggml_tensor *       d  = dmat[i].first;
            const ggml_tensor * s  = dmat[i].second;
            const bool          tr = dmat_t[i];
            GGML_ASSERT(ggml_is_contiguous(s));
            std::vector<float> sf;
            const float *      sv = (const float *) s->data;
            if (s->type == GGML_TYPE_BF16) {
                sf.resize(ggml_nelements(s));
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) s->data, sf.data(), sf.size());
                sv = sf.data();
            } else if (s->type != GGML_TYPE_F32) {
                throw std::runtime_error(std::string(ggml_get_name(s)) + ": weight type");
            }
            const int64_t n0 = d->ne[0], n1 = d->ne[1];
            std::vector<float> tmp;
            if (tr) {  // d [n0, n1] = s^T, s [n1, n0]
                tmp.resize(n0 * n1);
                for (int64_t a = 0; a < n1; ++a) {
                    for (int64_t b = 0; b < n0; ++b) {
                        tmp[a * n0 + b] = sv[b * n1 + a];
                    }
                }
                sv = tmp.data();
            }
            if (d->type == GGML_TYPE_BF16) {
                ggml_fp32_to_bf16_row_ref(sv, (ggml_bf16_t *) d->data, n0 * n1);
            } else if (m->d_bf16) {
                float * dv = (float *) d->data;
                for (int64_t j = 0; j < n0 * n1; ++j) {
                    dv[j] = bf16_round(sv[j]);
                }
            } else {
                std::memcpy(d->data, sv, n0 * n1 * sizeof(float));
            }
        }
        for (ggml_tensor * t = ggml_get_first_tensor(m->wctx); t; t = ggml_get_next_tensor(m->wctx, t)) {
            m->tensors[ggml_get_name(t)] = t;
        }
        if (device != "cpu") {  // everything the graphs read, copied to the device
            std::map<const ggml_tensor *, ggml_tensor *> to;
            ggml_init_params dp = { ggml_tensor_overhead() * (m->tensors.size() + 1), nullptr, true };
            m->dctx             = ggml_init(dp);
            for (auto & [name, t] : m->tensors) {
                if (name.size() > 8 && name.compare(name.size() - 8, 8, ".wpacked") == 0) {
                    continue;
                }
                ggml_tensor * d = ggml_dup_tensor(m->dctx, t);
                ggml_set_name(d, name.c_str());
                to[t] = d;
            }
            ggml_backend_buffer_t db =
                ggml_backend_alloc_ctx_tensors_from_buft(m->dctx, ggml_backend_get_default_buffer_type(m->backend));
            if (!db) {
                throw std::runtime_error("cannot allocate the weights on " + device);
            }
            m->bufs.push_back(db);
            for (auto & [h, d] : to) {
                ggml_backend_tensor_set(d, h->data, 0, ggml_nbytes(h));
            }
            auto mv = [&](ggml_tensor *& p) {
                if (p) {
                    p = to.at(p);
                }
            };
            for (auto & [name, t] : m->tensors) {
                if (to.count(t)) {
                    t = to[t];
                }
            }
            for (auto & [p, P] : m->parts) {
                mv(P.w); mv(P.beta); mv(P.mu); mv(P.c); mv(P.r); mv(P.b); mv(P.mag_A); mv(P.mag_B);
            }
            for (auto & [p, h] : m->head_w) {
                mv(h);
            }
        }
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return nullptr;
    }
    return m;
}

}  // namespace ttf
