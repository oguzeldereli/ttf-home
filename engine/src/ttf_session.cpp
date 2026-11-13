// ttf::Session: state (KV caches by IR state id) and the ggml graph built from the IR, one mapping per op.
// Shapes: the IR's are innermost-last (torch order); ggml's are reversed ([K, T] for an IR [T, K]).

#include "ttf.h"

#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"  // ggml_graph_view (TTF_PROF_OPS)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ttf {

namespace {

constexpr size_t GRAPH_SIZE = 32768;

struct Builder {
    const Model &                        m;
    Session &                            s;
    ggml_context *                       ctx;
    std::map<std::string, ggml_tensor *> val;
    ggml_tensor *                        pos   = nullptr;
    ggml_tensor *                        mask  = nullptr;  // F32 [S, T], shared by every causal attn
    int64_t                              S     = 0;        // key length this call
    int64_t                              T     = 0;
    bool                                 last_only = false;
    bool                                 bf16;

    ggml_tensor * rd(ggml_tensor * t) {  // round to D
        if (!bf16) {
            return t;
        }
        return ggml_cast(ctx, ggml_cast(ctx, t, GGML_TYPE_BF16), GGML_TYPE_F32);
    }

    ggml_tensor * cont(ggml_tensor * t) {
        return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
    }

    ggml_tensor * in(const json & n, size_t i) {
        const auto & ins = n.at("in");
        if (i >= ins.size() || ins[i].is_null()) {
            return nullptr;
        }
        auto it = val.find(ins[i].get<std::string>());
        if (it == val.end()) {
            throw std::runtime_error("IR: undefined value " + ins[i].get<std::string>());
        }
        return it->second;
    }

    ggml_tensor * weight(const std::string & name) {
        ggml_tensor * t = m.get(name);
        if (!t) {
            throw std::runtime_error("missing tensor " + name);
        }
        return t;
    }

    // one part on x2 F32 [K, rows] -> F32 [N, rows] (Arithmetic steps 1-8; acc_in: + step)
    ggml_tensor * part(const std::string & p, ggml_tensor * x2, ggml_tensor * acc_in) {
        auto it = m.parts.find(p);
        if (it == m.parts.end()) {
            throw std::runtime_error("unknown part " + p);
        }
        const Part & P = it->second;
        if (x2->ne[0] != P.K) {
            throw std::runtime_error(p + ": input K " + std::to_string(x2->ne[0]) + " != " + std::to_string(P.K));
        }
        // step 6 (legacy adapter) fused: the down projection in ttf_quant, the up projection in ttf_mm
        ggml_tensor * qa = ggml_ttf_quant(ctx, x2, P.mu, P.c, P.mag_A ? P.mag_B : nullptr, P.Kp, bf16);
        return ggml_ttf_mm(ctx, P.w, qa, P.beta, P.mag_B ? P.mag_A : nullptr, P.r, P.b, acc_in, bf16);
    }

    ggml_tensor * op_linear(const json & n) {
        ggml_tensor * x   = in(n, 0);
        ggml_tensor * acc = in(n, 1);
        const json &  a   = n.at("attr");
        const std::string p = n.at("p");
        const int64_t K     = x->ne[0];
        const int64_t rows  = ggml_nelements(x) / K;
        ggml_tensor * x2    = ggml_reshape_2d(ctx, cont(x), K, rows);
        ggml_tensor * y;
        if (a.value("adapter", false)) {  // QTad: rD(lin + up(down(x)))
            ggml_tensor * z  = part(p + ".ad.0", x2, nullptr);
            ggml_tensor * up = part(p + ".ad.1", z, nullptr);
            y                = part(p + ".lin", x2, up);
        } else {
            y = part(p, x2, nullptr);
        }
        if (acc) {
            ggml_tensor * acc2 = ggml_reshape_2d(ctx, cont(acc), y->ne[0], rows);
            y                  = rd(ggml_add(ctx, rd(acc2), y));
        }
        int64_t ne[GGML_MAX_DIMS] = { y->ne[0], x->ne[1], x->ne[2], x->ne[3] };
        return ggml_reshape(ctx, y, ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(x) > 1 ? ggml_n_dims(x) : 1, ne));
    }

    ggml_tensor * op_reshape(const json & n) {
        ggml_tensor * x   = cont(in(n, 0));
        const auto &  shp = n.at("attr").at("shape");
        std::vector<int64_t> ne;
        for (auto it = shp.rbegin(); it != shp.rend(); ++it) {
            ne.push_back(it->is_string() ? T : it->get<int64_t>());
        }
        switch (ne.size()) {
            case 1: return ggml_reshape_1d(ctx, x, ne[0]);
            case 2: return ggml_reshape_2d(ctx, x, ne[0], ne[1]);
            case 3: return ggml_reshape_3d(ctx, x, ne[0], ne[1], ne[2]);
            case 4: return ggml_reshape_4d(ctx, x, ne[0], ne[1], ne[2], ne[3]);
        }
        throw std::runtime_error("reshape: rank");
    }

    ggml_tensor * operand2(const json & n) {
        const json & a = n.value("attr", json::object());
        if (a.contains("w")) {
            return weight(a.at("w"));
        }
        return in(n, 1);
    }

    ggml_tensor * op_rope(const json & n) {
        ggml_tensor * x  = in(n, 0);
        ggml_tensor * ps = in(n, 1);
        const json &  a  = n.at("attr");
        const int     mode = a.value("style", std::string("interleaved")) == "interleaved" ? GGML_ROPE_TYPE_NORMAL
                                                                                           : GGML_ROPE_TYPE_NEOX;
        const int     hd   = (int) x->ne[0];
        return ggml_rope_ext(ctx, x, ps, nullptr, hd, mode, 0, a.at("theta").get<float>(), 1.0f, 0.0f, 1.0f, 0.0f,
                             0.0f);
    }

    ggml_tensor * op_attn(const json & n) {
        ggml_tensor * q = in(n, 0);  // [hd, H, T]
        ggml_tensor * k = in(n, 1);  // [hd, Hkv, T]
        ggml_tensor * v = in(n, 2);
        const json &  a = n.at("attr");
        if (!a.value("causal", true)) {
            throw std::runtime_error("attn: non-causal not yet");
        }
        const std::string sid = a.at("state");
        auto              it  = s.kv.find(sid);
        if (it == s.kv.end()) {
            throw std::runtime_error("attn: unknown state " + sid);
        }
        Session::Kv & kv  = it->second;
        const int64_t hd  = q->ne[0];
        const int64_t H   = q->ne[1];
        const float   scl = a.contains("scale") ? a.at("scale").get<float>() : 1.0f / std::sqrt((float) hd);
        if (k->ne[0] != kv.hd || k->ne[1] != kv.hkv) {
            throw std::runtime_error("attn: k shape vs state " + sid);
        }
        ggml_tensor * k2 = ggml_reshape_2d(ctx, cont(k), kv.hd * kv.hkv, T);
        ggml_tensor * v2 = ggml_reshape_2d(ctx, cont(v), kv.hd * kv.hkv, T);
        ggml_tensor * kc = ggml_set_rows(ctx, kv.k, k2, pos);
        ggml_tensor * vc = ggml_set_rows(ctx, kv.v, v2, pos);
        ggml_tensor * K  = ggml_view_3d(ctx, kc, kv.hd, kv.hkv, S, kv.hd * sizeof(float), kc->nb[1], 0);
        ggml_tensor * V  = ggml_view_3d(ctx, vc, kv.hd, kv.hkv, S, kv.hd * sizeof(float), vc->nb[1], 0);
        ggml_tensor * Q  = ggml_permute(ctx, q, 0, 2, 1, 3);                              // [hd, T, H]
        ggml_tensor * Kp = ggml_permute(ctx, K, 0, 2, 1, 3);                              // [hd, S, Hkv]
        ggml_tensor * kq = ggml_mul_mat(ctx, Kp, Q);                                      // [S, T, H]
        kq               = ggml_soft_max_ext(ctx, kq, mask, scl, 0.0f);
        ggml_tensor * Vt = cont(ggml_permute(ctx, V, 1, 2, 0, 3));              // [S, hd, Hkv]
        ggml_tensor * o  = ggml_mul_mat(ctx, Vt, kq);                                     // [hd, T, H]
        return cont(ggml_permute(ctx, o, 0, 2, 1, 3));                          // [hd, H, T]
        (void) H;
    }

    ggml_tensor * op_head(const json & n) {
        ggml_tensor * x = in(n, 0);
        auto          it = m.head_w.find(n.at("p").get<std::string>());
        if (it == m.head_w.end()) {
            throw std::runtime_error("head weight");
        }
        ggml_tensor * x2 = ggml_reshape_2d(ctx, cont(x), x->ne[0], ggml_nelements(x) / x->ne[0]);
        if (last_only && x2->ne[1] > 1) {
            x2 = ggml_view_2d(ctx, x2, x2->ne[0], 1, x2->nb[1], (x2->ne[1] - 1) * x2->nb[1]);
        }
        // x.to(D) @ w.to(D).T, output in D (a BF16 weight rounds the activations itself)
        return rd(ggml_mul_mat(ctx, it->second, bf16 ? x2 : rd(x2)));
    }

    void node(const json & n) {
        const std::string op = n.at("op");
        ggml_tensor *     y  = nullptr;
        if (op == "ttf.linear") {
            y = op_linear(n);
        } else if (op == "ttf.embed") {
            const json & a = n.value("attr", json::object());
            if (a.value("kind", std::string("float")) != "float") {
                throw std::runtime_error("ttf.embed: int-k not yet");
            }
            y = ggml_get_rows(ctx, weight(n.at("p").get<std::string>() + ".weight"), in(n, 0));
        } else if (op == "ttf.head") {
            y = op_head(n);
        } else if (op == "rms_norm") {
            y = ggml_rms_norm(ctx, in(n, 0), n.at("attr").at("eps").get<float>());
            if (n.contains("p")) {
                y = ggml_mul(ctx, y, weight(n.at("p").get<std::string>() + ".weight"));
            }
        } else if (op == "rope") {
            y = op_rope(n);
        } else if (op == "attn") {
            y = op_attn(n);
        } else if (op == "glu") {
            const std::string act = n.value("attr", json::object()).value("act", std::string("silu"));
            ggml_tensor *     g   = in(n, 0);
            y = ggml_mul(ctx, act == "silu" ? ggml_silu(ctx, g) : ggml_gelu(ctx, g), in(n, 1));
        } else if (op == "reshape") {
            y = op_reshape(n);
        } else if (op == "add") {
            y = ggml_add(ctx, in(n, 0), operand2(n));
        } else if (op == "mul") {
            y = ggml_mul(ctx, in(n, 0), operand2(n));
        } else if (op == "sub") {
            y = ggml_sub(ctx, in(n, 0), operand2(n));
        } else if (op == "div") {
            y = ggml_div(ctx, in(n, 0), operand2(n));
        } else {
            throw std::runtime_error("IR op not supported yet: " + op);
        }
        const auto & outs = n.at("out");
        if (outs.size() != 1) {
            throw std::runtime_error(op + ": multi-output not yet");
        }
        val[outs[0].get<std::string>()] = y;
    }
};

}  // namespace

Session::Session(const Model & m_, int n_threads_, int64_t max_ctx_) : m(m_), n_threads(n_threads_), max_ctx(max_ctx_) {
    backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    const json & st = m.ir.value("state", json::array());
    ggml_init_params ip = { ggml_tensor_overhead() * (st.size() * 2 + 4), nullptr, true };
    sctx                = ggml_init(ip);
    for (const auto & e : st) {
        const std::string kind = e.at("kind");
        if (kind != "kv") {
            throw std::runtime_error("state kind " + kind + ": not yet");
        }
        Kv kvs;
        kvs.hd  = e.at("head_dim");
        kvs.hkv = e.at("heads_kv");
        kvs.k   = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, kvs.hd * kvs.hkv, max_ctx);
        kvs.v   = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, kvs.hd * kvs.hkv, max_ctx);
        kvs.filled.assign(max_ctx, false);
        kv[e.at("id").get<std::string>()] = kvs;
    }
    sbuf = ggml_backend_alloc_ctx_tensors(sctx, backend);
    if (sbuf) {
        ggml_backend_buffer_clear(sbuf, 0);
    }
    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    gbuf.resize(ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false));
}

Session::~Session() {
    if (galloc) {
        ggml_gallocr_free(galloc);
    }
    if (sbuf) {
        ggml_backend_buffer_free(sbuf);
    }
    if (sctx) {
        ggml_free(sctx);
    }
    if (backend) {
        ggml_backend_free(backend);
    }
}

void Session::reset() {
    for (auto & [id, k] : kv) {
        k.filled.assign(max_ctx, false);
    }
}

StepOut Session::step(const std::string & graph, const std::vector<int32_t> & ids, const std::vector<int32_t> & pos,
                      bool last_only) {
    const auto t0 = std::chrono::steady_clock::now();
    const json & g = m.ir.at("graphs").at(graph);
    const int64_t T = (int64_t) ids.size();
    if (T == 0 || (int64_t) pos.size() != T) {
        throw std::runtime_error("step: ids / pos");
    }
    for (int32_t p : pos) {
        if (p < 0 || p >= max_ctx) {
            throw std::runtime_error("step: position out of context");
        }
    }
    // state update first: a token sees itself and the earlier tokens of its own call
    std::vector<bool> filled = kv.empty() ? std::vector<bool>() : kv.begin()->second.filled;
    for (auto & [id, k] : kv) {
        for (int32_t p : pos) {
            k.filled[p] = true;
        }
    }
    int64_t S = 0;
    if (!kv.empty()) {
        const auto & f = kv.begin()->second.filled;
        for (int64_t j = max_ctx - 1; j >= 0; --j) {
            if (f[j]) {
                S = j + 1;
                break;
            }
        }
    }

    ggml_init_params ip = { gbuf.size(), gbuf.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    Builder          b{ m, *this, ctx, {}, nullptr, nullptr, S, T, last_only, m.d_bf16 };
    ggml_tensor *    tid = nullptr;
    for (const auto & inp : g.at("inputs")) {
        const std::string name = inp.at("name");
        ggml_tensor *     t    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_name(t, name.c_str());
        ggml_set_input(t);
        b.val[name] = t;
        if (name == "ids") {
            tid = t;
        } else if (name == "pos") {
            b.pos = t;
        } else {
            throw std::runtime_error("graph input " + name + ": not yet");
        }
    }
    if (!kv.empty()) {
        b.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, T);
        ggml_set_name(b.mask, "kq_mask");
        ggml_set_input(b.mask);
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    for (const auto & n : g.at("nodes")) {
        b.node(n);
    }
    // TTF_DUMP=DIR: every IR value of this call to DIR/<name>.bin (f32 / i32 raw) + DIR/index.json (ggml ne)
    const char * dump_dir = std::getenv("TTF_DUMP");
    if (dump_dir) {
        for (auto & [name, t] : b.val) {
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
        }
    }
    std::vector<ggml_tensor *> outs;
    for (const auto & o : g.at("outputs")) {
        ggml_tensor * t = b.val.at(o.at("of").get<std::string>());
        ggml_set_output(t);
        outs.push_back(t);
        ggml_build_forward_expand(gf, t);
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_free(ctx);
        throw std::runtime_error("graph allocation failed");
    }
    ggml_backend_tensor_set(tid, ids.data(), 0, T * sizeof(int32_t));
    ggml_backend_tensor_set(b.pos, pos.data(), 0, T * sizeof(int32_t));
    if (b.mask) {
        const auto &       f = kv.begin()->second.filled;
        std::vector<float> mk(S * T);
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t j = 0; j < S; ++j) {
                mk[t * S + j] = (f[j] && j <= pos[t]) ? 0.0f : -std::numeric_limits<float>::infinity();
            }
        }
        ggml_backend_tensor_set(b.mask, mk.data(), 0, mk.size() * sizeof(float));
    }
    const auto t2 = std::chrono::steady_clock::now();
    if (std::getenv("TTF_PROF_OPS")) {  // node by node, time per op (+ part shape for TTF_MM)
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_cgraph  gv = ggml_graph_view(gf, i, i + 1);
            const auto   a  = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(backend, &gv);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
            ggml_tensor * nd = ggml_graph_node(gf, i);
            std::string   k  = ggml_op_desc(nd);
            if (nd->op == GGML_OP_TTF_MM || nd->op == GGML_OP_MUL_MAT) {
                k += " " + std::to_string(nd->src[0]->ne[0]) + "x" + std::to_string(nd->src[0]->ne[1]) + " " +
                     ggml_type_name(nd->src[0]->type);
            }
            prof_ops[k].first += ms;
            prof_ops[k].second += 1;
        }
    } else if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("graph compute failed");
    }
    if (dump_dir) {
        json idx = json::object();
        for (auto & [name, t] : b.val) {
            if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32)) {
                continue;
            }
            std::vector<char> buf(ggml_nbytes(t));
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
            std::ofstream f(std::string(dump_dir) + "/" + name + ".bin", std::ios::binary);
            f.write(buf.data(), (std::streamsize) buf.size());
            idx[name] = { { "ne", { t->ne[0], t->ne[1], t->ne[2], t->ne[3] } },
                          { "type", t->type == GGML_TYPE_F32 ? "f32" : "i32" } };
        }
        std::ofstream(std::string(dump_dir) + "/index.json") << idx.dump();
    }
    StepOut r;
    ggml_tensor * lo = outs.at(0);
    r.n_vocab        = lo->ne[0];
    r.T_out          = ggml_nelements(lo) / lo->ne[0];
    r.logits.resize(ggml_nelements(lo));
    ggml_backend_tensor_get(lo, r.logits.data(), 0, ggml_nbytes(lo));
    ggml_free(ctx);
    const auto t3 = std::chrono::steady_clock::now();
    last_ms       = std::chrono::duration<double, std::milli>(t3 - t0).count();
    prof_build += std::chrono::duration<double, std::milli>(t1 - t0).count();
    prof_alloc += std::chrono::duration<double, std::milli>(t2 - t1).count();
    prof_compute += std::chrono::duration<double, std::milli>(t3 - t2).count();
    prof_nodes = ggml_graph_n_nodes(gf);
    (void) filled;
    return r;
}

}  // namespace ttf
