// ttf::Model: load a .ttf file; trits rows5 -> TQ1_P (K padded to 160); side tensors and float weights in D.

#include "ttf.h"

#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
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

Model::~Model() {
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
}

std::string Model::kv_str(const std::string & key, const std::string & dflt) const {
    return gguf_str(gguf, key, dflt);
}

ggml_tensor * Model::get(const std::string & name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

std::unique_ptr<Model> Model::load(const std::string & path, std::string * err) {
    auto m = std::make_unique<Model>();
    try {
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
        const char * lt = std::getenv("TTF_TRITS");
        m->q2p          = !(lt && std::string(lt) == "tq1p");
        const int64_t QB = m->q2p ? 128 : QK;

        // pass 1: shapes of the derived tensors
        struct Pending {
            std::string   part;
            int64_t       K;
            ggml_tensor * packed;
        };
        std::vector<Pending>                                       pend;
        std::vector<std::pair<std::string, const ggml_tensor *>> heads;
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
                } else if (op == "ttf.head") {
                    const std::string p = n.at("p");
                    if (!m->get(p + ".weight")) {
                        throw std::runtime_error("missing " + p + ".weight");
                    }
                    heads.emplace_back(p, m->get(p + ".weight"));
                }
            }
        }
        const ggml_type     dt = m->d_bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        const size_t        nt = pend.size() * 8 + heads.size() + 8;
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
            if (s->type != GGML_TYPE_F32) {
                throw std::runtime_error(name + ": side tensor not F32");
            }
            if (!m->d_bf16) {
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
            if (!P.beta) {
                throw std::runtime_error("missing " + pe.part + ".beta");
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
            ggml_tensor * h = ggml_new_tensor_2d(m->wctx, dt, w->ne[0], w->ne[1]);
            ggml_set_name(h, (p + ".head.D").c_str());
            m->head_w[p] = h;
            dmat.emplace_back(h, w);
            dmat_t.push_back(false);
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(m->wctx, ggml_backend_cpu_buffer_type());
        if (!buf) {
            throw std::runtime_error("cannot allocate the weights");
        }
        m->bufs.push_back(buf);

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
        for (auto & [d, s] : dvec) {
            const float * sv = (const float *) s->data;
            float *       dv = (float *) d->data;
            for (int64_t i = 0; i < ggml_nelements(s); ++i) {
                dv[i] = bf16_round(sv[i]);
            }
        }
        for (size_t i = 0; i < dmat.size(); ++i) {
            ggml_tensor *       d  = dmat[i].first;
            const ggml_tensor * s  = dmat[i].second;
            const bool          tr = dmat_t[i];
            GGML_ASSERT(s->type == GGML_TYPE_F32 && ggml_is_contiguous(s));
            const float * sv = (const float *) s->data;
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
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return nullptr;
    }
    return m;
}

}  // namespace ttf
