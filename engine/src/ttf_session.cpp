// ttf::Session: state (KV caches by IR state id) and the ggml graph built from the IR, one mapping per op.
// Shapes: the IR's are innermost-last (torch order); ggml's are reversed ([K, T] for an IR [T, K]).

#include "ttf.h"

#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"  // ggml_graph_view (TTF_PROF_OPS)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace ttf {

namespace {

constexpr size_t GRAPH_SIZE = 32768;


// TTF_MM nodes after the quant grouping: a later TTF_MM whose inputs are all computed by the first one's position
// (its activations from the grouped quants, the rest leaves) moves next to it (up to 3), so the CUDA backend runs a
// decode step's q / k / v and gate / up as one launch each. Same rule: only independent nodes change order.
static void group_mms(ggml_cgraph * gf) {
    constexpr int SCAN = 96, GMAX = 3;
    ggml_tensor ** nodes = gf->nodes;
    const int      n     = gf->n_nodes;
    std::unordered_map<const ggml_tensor *, int> at;
    auto index = [&]() {
        at.clear();
        for (int k = 0; k < n; ++k) {
            at[nodes[k]] = k;
        }
    };
    auto base = [](const ggml_tensor * t) { return t->view_src ? t->view_src : t; };
    index();
    for (int i = 0; i < n; ++i) {
        ggml_tensor * a = nodes[i];
        if (a->op != GGML_OP_TTF_MM) {
            continue;
        }
        int pos = i + 1;
        int G   = 1;
        for (int j = pos; j < n && j < pos + SCAN && G < GMAX; ++j) {
            ggml_tensor * t = nodes[j];
            if (t->op != GGML_OP_TTF_MM || t->src[0]->type != a->src[0]->type ||
                    ggml_get_op_params_i32(t, 0) != ggml_get_op_params_i32(a, 0)) {
                continue;
            }
            bool ok = true;
            for (int si = 0; si < GGML_MAX_SRC && ok; ++si) {
                const ggml_tensor * s = t->src[si];
                if (!s) {
                    continue;
                }
                for (const ggml_tensor * u : { s, base(s) }) {
                    auto it = at.find(u);
                    ok = ok && (it == at.end() || it->second < pos);
                }
                for (int k = pos; k < j && ok; ++k) {  // nothing between writes an input in place
                    const ggml_tensor * v = nodes[k];
                    ok = !(v->view_src && v->view_src == base(s) && v->op != GGML_OP_RESHAPE && v->op != GGML_OP_VIEW &&
                           v->op != GGML_OP_PERMUTE && v->op != GGML_OP_TRANSPOSE && v->op != GGML_OP_NONE);
                }
            }
            if (!ok) {
                continue;
            }
            std::rotate(nodes + pos, nodes + j, nodes + j + 1);
            index();
            ++pos;
            ++G;
        }
        i = pos - 1;
    }
}

// TTF_QUANT nodes on the same input (parts with their own mu / c / adapter: q / k / v, gate / up) moved next to the
// first of them, before allocation, so the CUDA backend runs them as one launch. A quant depends on its input (views
// of it) and leaves only, so it can move up to any point after the first; a node between that writes the input in
// place stops the scan. The values do not change (only the order of independent nodes).
static void group_quants(ggml_cgraph * gf) {
    constexpr int SCAN = 96, GMAX = 4;
    ggml_tensor ** nodes = gf->nodes;
    const int      n     = gf->n_nodes;
    auto base = [](const ggml_tensor * t) { return t->view_src ? t->view_src : t; };
    auto offs = [](const ggml_tensor * t) { return t->view_src ? t->view_offs : (size_t) 0; };
    auto same = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const ggml_tensor * x = a->src[0];
        const ggml_tensor * y = b->src[0];
        return base(x) == base(y) && offs(x) == offs(y) && ggml_are_same_shape(x, y) && x->nb[1] == y->nb[1] &&
               ggml_is_contiguous(x) && ggml_is_contiguous(y) &&
               ggml_get_op_params_i32(a, 1) == ggml_get_op_params_i32(b, 1);
    };
    auto pure_view = [](const ggml_tensor * t) {
        return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
               t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
    };
    std::vector<ggml_tensor *> moved;
    for (int i = 0; i < n; ++i) {
        ggml_tensor * a = nodes[i];
        if (a->op != GGML_OP_TTF_QUANT) {
            continue;
        }
        const ggml_tensor * xb  = base(a->src[0]);
        int                 pos = i + 1;
        int                 G   = 1;
        for (int j = pos; j < n && j < pos + SCAN && G < GMAX; ++j) {
            ggml_tensor * t = nodes[j];
            if (t->view_src == xb && !pure_view(t)) {
                break;  // writes the input in place
            }
            if (t->op != GGML_OP_TTF_QUANT || !same(a, t)) {
                continue;
            }
            bool movable = true;  // mu / c / adapter: leaves (weights), never nodes between
            for (int k = pos; k < j && movable; ++k) {
                for (int si = 1; si < GGML_MAX_SRC; ++si) {
                    movable = movable && (t->src[si] == nullptr || t->src[si] != nodes[k]);
                }
            }
            if (!movable) {
                continue;
            }
            // t's view chain of the input (graph nodes between pos and j) moves with it, in order
            std::vector<int> mv;
            for (const ggml_tensor * v = t->src[0]; v && v->view_src; v = v->src[0]) {
                for (int k = pos; k < j; ++k) {
                    if (nodes[k] == v) {
                        mv.push_back(k);
                    }
                }
            }
            std::sort(mv.begin(), mv.end());
            mv.push_back(j);
            moved.clear();
            for (int k : mv) {
                moved.push_back(nodes[k]);
            }
            int w = j;  // compact [pos, j] without the moved nodes, from the back
            for (int k = j, m = (int) mv.size() - 1; k >= pos; --k) {
                if (m >= 0 && mv[m] == k) {
                    --m;
                    continue;
                }
                nodes[w--] = nodes[k];
            }
            for (size_t k = 0; k < moved.size(); ++k) {
                nodes[pos + (int) k] = moved[k];
            }
            pos += (int) moved.size();
            j = pos - 1;
            ++G;
        }
        i = pos - 1;
    }
}

struct Builder {
    const Model &                        m;
    Session &                            s;
    ggml_context *                       ctx;
    std::map<std::string, ggml_tensor *> val;
    std::map<std::string, int>           nd;      // each value's torch rank (ggml drops trailing 1s: T = 1)
    std::vector<ggml_tensor *>           writes;  // state updates (conv / rec), part of the graph

    // a constant made on the host (rope factors, frequencies, aranges, zero context): one device copy per distinct
    // content for the session's life, uploaded once (not per call: each upload is a synchronous copy on CUDA)
    ggml_tensor * host_const(const Tensor & h) {
        std::string key(h.type == GGML_TYPE_F32 ? (const char *) h.f.data() : (const char *) h.i.data(),
                        (h.type == GGML_TYPE_F32 ? h.f.size() : h.i.size()) * 4);
        key += '|' + std::to_string((int) h.type);
        for (int64_t d : h.shape) {
            key += ',' + std::to_string(d);
        }
        auto it = s.consts.find(key);
        if (it != s.consts.end()) {
            return it->second.t;
        }
        int64_t   ne[4] = { 1, 1, 1, 1 };
        const int r     = (int) h.shape.size();
        for (int i = 0; i < r; ++i) {
            ne[r - 1 - i] = h.shape[i];
        }
        Session::Const c;
        ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
        c.ctx               = ggml_init(ip);
        c.t                 = ggml_new_tensor(c.ctx, h.type, std::max(r, 1), ne);
        c.buf               = ggml_backend_alloc_ctx_tensors(c.ctx, s.backend);
        ggml_backend_tensor_set(c.t, h.type == GGML_TYPE_F32 ? (const void *) h.f.data() : (const void *) h.i.data(), 0,
                                ggml_nbytes(c.t));
        s.consts[key] = c;
        return c.t;
    }
    ggml_tensor *                        pos   = nullptr;
    ggml_tensor *                        mask  = nullptr;  // F32 [S, T], shared by every causal attn
    int64_t                              S     = 0;        // key length this call
    int64_t                              T     = 0;
    bool                                 last_only = false;
    bool                                 bf16;
    // ttf_quant results by (data, offset, shape, mu, c, mag_B, Kp): parts on one input (q / k / v, gate / up) share
    // the quantised activations (identical by construction: the same op on the same bytes)
    std::map<std::vector<uintptr_t>, ggml_tensor *> qcache = {};
    std::map<int64_t, ggml_tensor *>               vidx   = {};  // per V width C: I32 [C T], c * max_ctx + pos[t]

    ggml_tensor * vidx_for(int64_t C) {
        ggml_tensor *& t = vidx[C];
        if (!t) {
            t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, C * T);
            ggml_set_name(t, ("vidx" + std::to_string(C)).c_str());
            ggml_set_input(t);
        }
        return t;
    }

    ggml_tensor * rd(ggml_tensor * t) {  // round to D
        if (!bf16) {
            return t;
        }
        return ggml_cast(ctx, ggml_cast(ctx, t, GGML_TYPE_BF16), GGML_TYPE_F32);
    }

    ggml_tensor * cont(ggml_tensor * t) {
        return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
    }

    int rank_of(const json & n, size_t i) {
        return nd.at(n.at("in")[i].get<std::string>());
    }

    // torch dim d of a rank-r value -> ggml axis
    static int ax(int d, int r) {
        if (d < 0) {
            d += r;
        }
        return r - 1 - d;
    }

    ggml_tensor * reshape_ne(ggml_tensor * x, const std::vector<int64_t> & ne) {
        x = cont(x);
        switch (ne.size()) {
            case 1: return ggml_reshape_1d(ctx, x, ne[0]);
            case 2: return ggml_reshape_2d(ctx, x, ne[0], ne[1]);
            case 3: return ggml_reshape_3d(ctx, x, ne[0], ne[1], ne[2]);
            case 4: return ggml_reshape_4d(ctx, x, ne[0], ne[1], ne[2], ne[3]);
        }
        throw std::runtime_error("reshape: rank");
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
        ggml_tensor *                qb  = P.mag_A ? P.mag_B : nullptr;
        const ggml_tensor *          src = x2;
        size_t                       off = 0;
        while (src->view_src && (src->op == GGML_OP_RESHAPE || src->op == GGML_OP_VIEW) && ggml_is_contiguous(src)) {
            off += src->view_offs;
            src = src->view_src;
        }
        const std::vector<uintptr_t> qk  = { (uintptr_t) src, off, (uintptr_t) x2->ne[0], (uintptr_t) x2->ne[1],
                                             (uintptr_t) x2->ne[2], (uintptr_t) x2->ne[3], (uintptr_t) P.mu,
                                             (uintptr_t) P.c, (uintptr_t) qb, (uintptr_t) P.Kp, ggml_is_contiguous(x2) };
        ggml_tensor *& qa = qcache[qk];
        if (!qa) {
            qa = ggml_ttf_quant(ctx, x2, P.mu, P.c, qb, P.Kp, bf16);
        }
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
        ggml_tensor * x   = in(n, 0);
        const int     r   = rank_of(n, 0);
        const auto &  shp = n.at("attr").at("shape");
        std::vector<int64_t> tdim;  // torch order
        int64_t              known = 1, infer = -1;
        for (size_t i = 0; i < shp.size(); ++i) {
            int64_t d;
            if (shp[i].is_string()) {  // a name takes x's own dim at that position
                d = x->ne[ax((int) i, r)];
            } else {
                d = shp[i].get<int64_t>();
            }
            if (d == -1) {
                infer = (int64_t) i;
            } else {
                known *= d;
            }
            tdim.push_back(d);
        }
        if (infer >= 0) {
            tdim[infer] = ggml_nelements(x) / known;
        }
        return reshape_ne(x, std::vector<int64_t>(tdim.rbegin(), tdim.rend()));
    }

    ggml_tensor * operand2(const json & n) {
        const json & a = n.value("attr", json::object());
        if (a.contains("w")) {
            return weight(a.at("w"));
        }
        if (a.contains("c")) {
            ggml_tensor * x = in(n, 0);
            return ggml_scale_bias(ctx, ggml_view_1d(ctx, x, 1, 0), 0.0f, a.at("c").get<float>());
        }
        return in(n, 1);
    }

    ggml_tensor * op_rope(const json & n) {
        ggml_tensor * x  = in(n, 0);
        ggml_tensor * ps = in(n, 1);
        const json &  a  = n.at("attr");
        const int     mode = a.value("style", std::string("interleaved")) == "interleaved" ? GGML_ROPE_TYPE_NORMAL
                                                                                           : GGML_ROPE_TYPE_NEOX;
        const int     hd   = a.contains("dims") ? a.at("dims").get<int>() : (int) x->ne[0];
        ggml_tensor * ff     = nullptr;  // longrope: theta_j / factor_j, cos / sin x mscale
        float         mscale = 1.0f;
        if (a.contains("scaling")) {
            const json & sc = a.at("scaling");
            if (sc.at("type") != "longrope") {
                throw std::runtime_error("rope scaling " + sc.at("type").get<std::string>() + ": not yet");
            }
            std::vector<float> f = sc.at("factors").get<std::vector<float>>();
            ff                   = host_const(Tensor::f32({ (int64_t) f.size() }, f));
            mscale               = sc.value("mscale", 1.0f);
        }
        x = cont(x);
        // ggml rope: [hd, H, S, B] with pos [S]; the IR's x is [..., S, H, hd]
        const int64_t S_ = ps->ne[0];
        x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1], S_, ggml_nelements(x) / (x->ne[0] * x->ne[1] * S_));
        if (x->ne[3] == 1) {
            return ggml_rope_ext(ctx, x, ps, ff, hd, mode, 0, a.at("theta").get<float>(), 1.0f, 0.0f, mscale, 0.0f, 0.0f);
        }
        // leading batch: rope wants [hd, H, S] per call with pos [S]; fold B into H (same pos for every batch row)
        ggml_tensor * xb = ggml_cont(ctx, ggml_permute(ctx, x, 0, 1, 3, 2));                          // [hd, H, B, S]
        xb = ggml_reshape_3d(ctx, xb, xb->ne[0], xb->ne[1] * xb->ne[2], S_);
        xb = ggml_rope_ext(ctx, xb, ps, ff, hd, mode, 0, a.at("theta").get<float>(), 1.0f, 0.0f, mscale, 0.0f, 0.0f);
        xb = ggml_reshape_4d(ctx, xb, x->ne[0], x->ne[1], x->ne[3], S_);
        return ggml_cont(ctx, ggml_permute(ctx, xb, 0, 1, 3, 2));                                     // [hd, H, S, B]
    }

    ggml_tensor * op_attn(const json & n) {
        ggml_tensor * q = in(n, 0);  // [hd, H, T]
        ggml_tensor * k = in(n, 1);  // [hd, Hkv, T]
        ggml_tensor * v = in(n, 2);
        const json &  a = n.at("attr");
        if (!a.value("causal", true)) {  // q [..., S, H, hd], k / v [..., S, Hkv, hd]: ggml [hd, H, S, B]
            const int64_t hd = q->ne[0];
            const float   sc = a.contains("scale") ? a.at("scale").get<float>() : 1.0f / std::sqrt((float) hd);
            auto as4 = [&](ggml_tensor * t) {
                t = cont(t);
                return ggml_reshape_4d(ctx, t, t->ne[0], t->ne[1], t->ne[2], ggml_nelements(t) / (t->ne[0] * t->ne[1] * t->ne[2]));
            };
            ggml_tensor * Q  = ggml_permute(ctx, as4(q), 0, 2, 1, 3);                  // [hd, S, H, B]
            ggml_tensor * K  = ggml_permute(ctx, as4(k), 0, 2, 1, 3);
            ggml_tensor * kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, K, Q), nullptr, sc, 0.0f);  // [Sk, Sq, H, B]
            ggml_tensor * Vt = ggml_cont(ctx, ggml_permute(ctx, as4(v), 1, 2, 0, 3));  // [S, hd, Hkv, B]
            ggml_tensor * o  = ggml_mul_mat(ctx, Vt, kq);                              // [hd, Sq, H, B]
            return ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));                   // [hd, H, Sq, B]
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
        if (s.fa) {  // V cached like K; Q [hd, T, H], K / V [hd, S, Hkv] views; out [hd, H, T]
            ggml_tensor * vc = ggml_set_rows(ctx, kv.v, v2, pos);
            ggml_tensor * K  = ggml_view_3d(ctx, kc, kv.hd, S, kv.hkv, kc->nb[1], kv.hd * sizeof(float), 0);
            ggml_tensor * V  = ggml_view_3d(ctx, vc, kv.hd, S, kv.hkv, vc->nb[1], kv.hd * sizeof(float), 0);
            ggml_tensor * o  = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), K, V, mask, scl, 0.0f, 0.0f);
            if (s.fa_exact) {  // the vec path for any T (no tiled prefill): decode == prefill exactly
                ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
            }
            return o;
        }
        if (s.dfa) {  // CUDA: V cached like K
            if (kv.v->type != GGML_TYPE_F32) {  // a BF16 cache holds V exactly only if V is a D-rounded ttf_mm output
                const ggml_tensor * vs = v;
                while (vs->view_src || vs->op == GGML_OP_CONT || vs->op == GGML_OP_RESHAPE || vs->op == GGML_OP_VIEW ||
                       vs->op == GGML_OP_PERMUTE) {
                    vs = vs->view_src ? vs->view_src : vs->src[0];
                }
                if (vs->op != GGML_OP_TTF_MM || ggml_get_op_params_i32(vs, 0) == 0) {
                    throw std::runtime_error("attn " + sid + ": bf16 V cache but V is not a D-rounded linear output");
                }
            }
            ggml_tensor * vc = ggml_set_rows(ctx, kv.v, v2, pos);
            ggml_tensor * K  = ggml_view_3d(ctx, kc, kv.hd, S, kv.hkv, kc->nb[1], kv.hd * sizeof(float), 0);
            ggml_tensor * V  = ggml_view_3d(ctx, vc, kv.hd, S, kv.hkv, vc->nb[1], kv.hd * ggml_type_size(vc->type), 0);
            if (T <= 16) {  // decode: one kernel reading K and V once per kv head
                return ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), K, V, mask, scl, 0.0f, 0.0f);
            }
            ggml_tensor * Q  = ggml_permute(ctx, q, 0, 2, 1, 3);                         // [hd, T, H]
            ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);                                  // [S, T, H]
            kq               = ggml_soft_max_ext(ctx, kq, mask, scl, 0.0f);
            ggml_tensor * Vp = ggml_permute(ctx, V, 1, 0, 2, 3);                         // [S, hd, Hkv]
            ggml_tensor * Vt = V->type == GGML_TYPE_F32
                                   ? ggml_cont(ctx, Vp)
                                   : ggml_cast(ctx, Vp, GGML_TYPE_F32);
            ggml_tensor * o  = ggml_mul_mat(ctx, Vt, kq);                                // [hd, T, H]
            return cont(ggml_permute(ctx, o, 0, 2, 1, 3));                     // [hd, H, T]
        }
        // V is cached transposed ([max_ctx, C]: position fastest), so the second product reads it without a copy;
        // written element-wise by set_rows on its one-float rows at c * max_ctx + pos (index input vidx, per C)
        const int64_t C  = kv.hd * kv.hkv;
        ggml_tensor * vi = vidx_for(C);
        ggml_tensor * vf = ggml_reshape_2d(ctx, kv.v, 1, C * s.max_ctx);
        ggml_tensor * vc = ggml_set_rows(ctx, vf, ggml_reshape_2d(ctx, v2, 1, C * T), vi);
        ggml_tensor * vt = ggml_reshape_2d(ctx, vc, s.max_ctx, C);
        ggml_tensor * K  = ggml_view_3d(ctx, kc, kv.hd, kv.hkv, S, kv.hd * sizeof(float), kc->nb[1], 0);
        ggml_tensor * Q  = ggml_permute(ctx, q, 0, 2, 1, 3);                              // [hd, T, H]
        ggml_tensor * Kp = ggml_permute(ctx, K, 0, 2, 1, 3);                              // [hd, S, Hkv]
        ggml_tensor * kq = ggml_mul_mat(ctx, Kp, Q);                                      // [S, T, H]
        kq               = ggml_soft_max_ext(ctx, kq, mask, scl, 0.0f);
        ggml_tensor * Vt = ggml_view_3d(ctx, vt, S, kv.hd, kv.hkv, vt->nb[1], kv.hd * vt->nb[1], 0);  // [S, hd, Hkv]
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

    ggml_tensor * op_permute(const json & n) {
        ggml_tensor * x    = in(n, 0);
        const int     r    = rank_of(n, 0);
        const auto &  dims = n.at("attr").at("dims");
        int           axs[4] = { 0, 1, 2, 3 };
        for (int i = 0; i < r; ++i) {  // output torch dim i <- input torch dim dims[i]
            axs[ax(dims[i].get<int>(), r)] = ax(i, r);
        }
        return ggml_permute(ctx, x, axs[0], axs[1], axs[2], axs[3]);
    }

    ggml_tensor * op_slice(const json & n) {
        ggml_tensor * x     = in(n, 0);
        const json &  a     = n.at("attr");
        const int     r     = rank_of(n, 0);
        const int     g     = ax(a.at("dim").get<int>(), r);
        const int64_t start = a.at("start");
        const int64_t len   = a.contains("len") && !a.at("len").is_null() ? a.at("len").get<int64_t>() : x->ne[g] - start;
        int64_t       ne[4] = { x->ne[0], x->ne[1], x->ne[2], x->ne[3] };
        ne[g]               = len;
        return ggml_view_4d(ctx, x, ne[0], ne[1], ne[2], ne[3], x->nb[1], x->nb[2], x->nb[3], start * x->nb[g]);
    }

    ggml_tensor * op_matmul(const json & n) {
        const std::string p = n.at("p");
        ggml_tensor *     w = m.get(p + ".weight.F32");
        if (!w) {
            w = weight(p + ".weight");
        }
        ggml_tensor * y = ggml_mul_mat(ctx, w, in(n, 0));
        if (n.value("attr", json::object()).value("bias", false)) {
            y = ggml_add(ctx, y, weight(p + ".bias"));
        }
        return y;
    }

    // x torch [C, T] (ggml [T, C], a view is fine) -> torch [C, T]; depthwise (groups = C) with a stream state
    ggml_tensor * op_conv1d(const json & n) {
        const json & a = n.at("attr");
        if (a.value("stride", 1) != 1 || a.value("dilation", 1) != 1) {
            throw std::runtime_error("conv1d_causal: stride / dilation != 1 not yet");
        }
        ggml_tensor * x = in(n, 0);
        const int64_t C = x->ne[1];
        if (a.at("groups").get<int64_t>() != C) {
            throw std::runtime_error("conv1d_causal: only depthwise yet");
        }
        const int     ctxn = a.at("ctx");
        const std::string p = n.at("p");
        ggml_tensor * w    = weight(p + ".weight.conv");  // F32 [kernel, C]
        if (w->ne[0] != ctxn + 1) {
            throw std::runtime_error("conv1d_causal: kernel != ctx + 1");
        }
        ggml_tensor * xt = ggml_reshape_2d(ctx, cont(x), x->ne[0], C);  // [T, C], time fastest
        ggml_tensor * prev = nullptr;
        ggml_tensor * st   = nullptr;
        if (a.contains("state") && !a.at("state").is_null()) {
            st   = s.conv.at(a.at("state").get<std::string>());
            prev = st;
        } else {
            prev = ggml_scale(ctx, ggml_view_2d(ctx, xt, ctxn, C, xt->nb[1], 0), 0.0f);  // zeros
        }
        ggml_tensor * sx = ggml_concat(ctx, prev, xt, 0);                                // [ctx + T, C]
        if (st) {  // the new state: the last ctx frames
            ggml_tensor * tail = ggml_view_2d(ctx, sx, ctxn, C, sx->nb[1], (sx->ne[0] - ctxn) * sx->nb[0]);
            writes.push_back(ggml_cpy(ctx, tail, st));
        }
        ggml_tensor * y = ggml_ssm_conv(ctx, ggml_reshape_3d(ctx, sx, sx->ne[0], C, 1), w);  // [C, T, 1]
        if (n.value("attr", json::object()).value("bias", false)) {
            y = ggml_add(ctx, y, weight(p + ".bias"));
        }
        if (a.value("act", std::string("none")) == "silu") {
            y = ggml_silu(ctx, y);
        }
        return ggml_transpose(ctx, ggml_reshape_2d(ctx, y, C, y->ne[1]));  // torch [C, T] = ggml [T, C]
    }

    // gated delta rule (recurrent): q / k [T, Hk, dk], v [T, Hv, dv], g / b [T, Hv]; state rec [Hv, dk, dv]
    ggml_tensor * op_delta_rule(const json & n) {
        const json &  a  = n.at("attr");
        ggml_tensor * q  = cont(in(n, 0));
        ggml_tensor * k  = cont(in(n, 1));
        ggml_tensor * v  = cont(in(n, 2));
        ggml_tensor * g  = cont(in(n, 3));
        ggml_tensor * be = cont(in(n, 4));
        const int64_t dk = q->ne[0], Hk = q->ne[1], dv = v->ne[0], Hv = v->ne[1];
        if (dk != dv) {
            throw std::runtime_error("delta_rule: dk != dv not yet");
        }
        q = ggml_reshape_3d(ctx, q, dk, Hk, T);
        k = ggml_reshape_3d(ctx, k, dk, Hk, T);
        v = ggml_reshape_3d(ctx, v, dv, Hv, T);
        // q, k <- x / sqrt(|x|^2 + 1e-6) = rms_norm(x, 1e-6 / dk) / sqrt(dk); q *= scale (the op scales by dv^-1/2)
        const float scale = a.contains("scale") ? a.at("scale").get<float>() : 1.0f / std::sqrt((float) dk);
        if (a.value("qk_l2norm", true)) {
            q = ggml_scale(ctx, ggml_rms_norm(ctx, q, 1e-6f / (float) dk), scale * std::sqrt((float) dv) / std::sqrt((float) dk));
            k = ggml_scale(ctx, ggml_rms_norm(ctx, k, 1e-6f / (float) dk), 1.0f / std::sqrt((float) dk));
        } else {
            q = ggml_scale(ctx, q, scale * std::sqrt((float) dv));
        }
        if (Hv != Hk) {  // v head h reads q / k head h // (Hv / Hk): interleave (the op broadcasts tiled)
            const int64_t r = Hv / Hk;
            q = ggml_reshape_3d(ctx, ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, q, dk, 1, Hk, T), dk, r, Hk, T), dk, Hv, T);
            k = ggml_reshape_3d(ctx, ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, k, dk, 1, Hk, T), dk, r, Hk, T), dk, Hv, T);
        }
        g  = ggml_reshape_4d(ctx, g, 1, Hv, T, 1);
        be = ggml_reshape_4d(ctx, be, 1, Hv, T, 1);
        ggml_tensor * st  = s.rec.at(a.at("state").get<std::string>());  // [dv, dk, Hv, 1] (transposed S)
        ggml_tensor * out = ggml_gated_delta_net(ctx, ggml_reshape_4d(ctx, q, dk, Hv, T, 1),
                                                 ggml_reshape_4d(ctx, k, dk, Hv, T, 1),
                                                 ggml_reshape_4d(ctx, v, dv, Hv, T, 1), g, be, st, 1);
        const int64_t na = dv * Hv * T;
        writes.push_back(ggml_cpy(ctx, ggml_view_1d(ctx, out, dv * dv * Hv, na * sizeof(float)),
                                  ggml_reshape_1d(ctx, st, ggml_nelements(st))));
        return ggml_view_3d(ctx, out, dv, Hv, T, dv * sizeof(float), dv * Hv * sizeof(float), 0);
    }

    // causal conv over (left context, x): x torch [C, T] = ggml [T, C]; the context is the stream state or zeros
    ggml_tensor * with_ctx(const json & a, ggml_tensor * xt, int ctxn) {
        if (ctxn == 0) {
            return xt;
        }
        const int64_t C = xt->ne[1];
        ggml_tensor * prev;
        ggml_tensor * st = nullptr;
        if (a.contains("state") && !a.at("state").is_null()) {
            st   = s.conv.at(a.at("state").get<std::string>());
            prev = st;
        } else {
            prev = host_const(Tensor::f32({ C, (int64_t) ctxn }, std::vector<float>(C * ctxn, 0.0f)));
        }
        ggml_tensor * full = ggml_concat(ctx, prev, xt, 0);
        if (st) {
            ggml_tensor * tail = ggml_view_2d(ctx, full, ctxn, C, full->nb[1], (full->ne[0] - ctxn) * full->nb[0]);
            writes.push_back(ggml_cpy(ctx, tail, st));
        }
        return full;
    }

    ggml_tensor * conv_bias_act(const json & n, ggml_tensor * y) {  // y [T, Cout]
        const json &      a = n.at("attr");
        const std::string p = n.at("p");
        if (a.value("bias", false)) {
            ggml_tensor * bb = weight(p + ".bias");
            y                = ggml_add(ctx, y, ggml_reshape_2d(ctx, bb, 1, ggml_nelements(bb)));
        }
        if (a.value("act", std::string("none")) == "silu") {
            y = ggml_silu(ctx, y);
        }
        return y;
    }

    ggml_tensor * op_conv1d_general(const json & n) {
        const json &      a  = n.at("attr");
        const std::string p  = n.at("p");
        ggml_tensor *     xt = cont(in(n, 0));
        xt                   = ggml_reshape_2d(ctx, xt, xt->ne[0], ggml_nelements(xt) / xt->ne[0]);  // [T, C]
        const int64_t C = xt->ne[1];
        const int     st = a.value("stride", 1), dl = a.value("dilation", 1), gr = a.value("groups", 1);
        ggml_tensor * w  = weight(p + ".weight");  // ggml [k, Cin / g, Cout]
        ggml_tensor * full = with_ctx(a, xt, a.at("ctx").get<int>());
        ggml_tensor * y;
        if (gr == 1) {
            ggml_tensor * col = ggml_im2col(ctx, w, full, st, 0, 0, 0, dl, 0, false, GGML_TYPE_F32);   // [Cin k, Tout, 1]
            y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
                             ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));                    // [Tout, Cout]
        } else if (gr == C && w->ne[1] == 1) {  // depthwise
            ggml_tensor * fb  = ggml_reshape_4d(ctx, full, full->ne[0], 1, C, 1);
            ggml_tensor * col = ggml_im2col(ctx, w, fb, st, 0, 0, 0, dl, 0, false, GGML_TYPE_F32);       // [k, Tout, C]
            y = ggml_mul_mat(ctx, col, w);                                                                // [Tout, 1, C]
            y = ggml_reshape_2d(ctx, y, y->ne[0], C);
        } else {
            throw std::runtime_error("conv1d_causal " + p + ": groups not 1 / C");
        }
        return conv_bias_act(n, y);
    }

    ggml_tensor * op_conv_t1d(const json & n) {
        const json &      a  = n.at("attr");
        const std::string p  = n.at("p");
        if (a.value("groups", 1) != 1) {
            throw std::runtime_error("conv_transpose1d_causal: groups");
        }
        ggml_tensor * xt = cont(in(n, 0));
        xt               = ggml_reshape_2d(ctx, xt, xt->ne[0], ggml_nelements(xt) / xt->ne[0]);
        const int     c = a.at("ctx"), st = a.at("stride"), trim = a.at("trim");
        ggml_tensor * full = with_ctx(a, xt, c);
        ggml_tensor * w    = weight(p + ".weight");  // ggml [k, Cout, Cin]
        const int64_t k = w->ne[0], Co = w->ne[1], Lf = full->ne[0];
        ggml_tensor * y;
        if (k % st == 0) {
            // cols [k Cout, L] = W^T x, then y[o, (l + q) s + r] += cols[q s + r + k o, l] for q < k / s (overlap-add)
            ggml_tensor * cols = ggml_mul_mat(ctx, weight(p + ".weight.ct"), ggml_cont(ctx, ggml_transpose(ctx, full)));
            const int     mq   = (int) (k / st);
            y                  = nullptr;
            for (int q = 0; q < mq; ++q) {
                ggml_tensor * pc = ggml_view_3d(ctx, cols, st, Co, Lf, k * sizeof(float), cols->nb[1], q * st * sizeof(float));
                pc = ggml_cont(ctx, ggml_permute(ctx, pc, 0, 2, 1, 3));  // [s, L, Cout]
                pc = ggml_reshape_2d(ctx, pc, st * Lf, Co);
                pc = ggml_pad_ext(ctx, pc, q * st, (mq - 1 - q) * st, 0, 0, 0, 0, 0, 0);
                y  = y ? ggml_add(ctx, y, pc) : pc;
            }
        } else {
            y = ggml_conv_transpose_1d(ctx, w, full, st, 0, 1);  // [L, Cout]
        }
        const int64_t L    = y->ne[0] - (int64_t) c * st - trim;
        y = ggml_view_2d(ctx, y, L, y->ne[1], y->nb[1], (int64_t) c * st * y->nb[0]);
        return conv_bias_act(n, cont(y));
    }

    ggml_tensor * op_timestep(const json & n) {  // t [B] -> [B, dim] = cat(sin(scale t f), cos(scale t f))
        const json &  a    = n.at("attr");
        const int64_t half = a.at("dim").get<int64_t>() / 2;
        const float   mp   = a.value("max_period", 10000.0f);
        std::vector<float> f(half);
        for (int64_t j = 0; j < half; ++j) {  // torch: exp(arange(half) * -(log(mp) / (half - 1))), fp32
            f[j] = std::exp((float) j * -(std::log(mp) / (float) (half - 1)));
        }
        ggml_tensor * t  = in(n, 0);
        ggml_tensor * tb = ggml_scale(ctx, ggml_reshape_2d(ctx, cont(t), 1, ggml_nelements(t)), a.value("scale", 1.0f));
        ggml_tensor * fc = host_const(Tensor::f32({ half }, f));
        ggml_tensor * e  = ggml_mul(ctx, ggml_repeat_4d(ctx, tb, half, tb->ne[1], 1, 1), fc);  // [half, B]
        return ggml_concat(ctx, ggml_sin(ctx, e), ggml_cos(ctx, e), 0);
    }

    ggml_tensor * op_snake(const json & n) {  // x [C, T] torch: x + inv sin^2(alpha x)
        const std::string p  = n.at("p");
        ggml_tensor *     x  = cont(in(n, 0));
        ggml_tensor *     al = weight(p + ".alpha");
        ggml_tensor *     iv = weight(p + ".alpha.inv");
        const int64_t     C  = ggml_nelements(al);
        al = ggml_reshape_2d(ctx, al, 1, C);
        iv = ggml_reshape_2d(ctx, iv, 1, C);
        ggml_tensor * sn = ggml_sin(ctx, ggml_mul(ctx, x, al));
        return ggml_add(ctx, x, ggml_mul(ctx, ggml_sqr(ctx, sn), iv));
    }

    // ggml-cuda's SSM_CONV: channels a multiple of 128, kernel 3 / 4 / 5 / 7 / 9 / 15
    static bool ssm_conv_cuda_ok(int64_t C, int k) {
        return C % 128 == 0 && (k == 3 || k == 4 || k == 5 || k == 7 || k == 9 || k == 15);
    }

    void node(const json & n) {
        const std::string op = n.at("op");
        ggml_tensor *     y  = nullptr;
        int               r  = n.at("in").size() > 0 && !n.at("in")[0].is_null() ? rank_of(n, 0) : 1;
        if (op == "ttf.linear") {
            y = op_linear(n);
        } else if (op == "ttf.embed") {
            const json & a = n.value("attr", json::object());
            const std::string kind = a.value("kind", std::string("float"));
            if (kind != "float" && kind != "intk") {
                throw std::runtime_error("ttf.embed: kind " + kind);
            }
            y = ggml_get_rows(ctx, weight(n.at("p").get<std::string>() + ".weight"), in(n, 0));
            r += 1;
        } else if (op == "ttf.head") {
            y = op_head(n);
        } else if (op == "rms_norm") {
            y = ggml_rms_norm(ctx, cont(in(n, 0)), n.at("attr").at("eps").get<float>());
            if (n.contains("p")) {  // offset: x (1 + w), the (1 + w) made at load
                const bool off = n.at("attr").value("offset", false);
                y = ggml_mul(ctx, y, weight(n.at("p").get<std::string>() + (off ? ".weight.off1" : ".weight")));
            }
        } else if (op == "rope") {
            y = op_rope(n);
        } else if (op == "attn") {
            y = op_attn(n);  // q's rank: [T, H, hd] causal, [..., S, H, hd] batched
        } else if (op == "glu") {
            const std::string act = n.value("attr", json::object()).value("act", std::string("silu"));
            ggml_tensor *     g   = in(n, 0);
            ggml_tensor *     ga;
            if (act == "silu") {
                ga = ggml_silu(ctx, g);
            } else if (act == "gelu") {
                ga = ggml_gelu(ctx, g);
            } else if (act == "relu2") {  // relu(g)^2 (BitNet b1.58)
                ga = ggml_sqr(ctx, ggml_relu(ctx, g));
            } else {
                throw std::runtime_error("glu act " + act + ": not implemented");
            }
            y = ggml_mul(ctx, ga, in(n, 1));
        } else if (op == "reshape") {
            y = op_reshape(n);
            r = (int) n.at("attr").at("shape").size();
        } else if (op == "permute") {
            y = op_permute(n);
        } else if (op == "slice") {
            y = op_slice(n);
        } else if (op == "matmul") {
            y = op_matmul(n);
        } else if (op == "conv1d_causal") {
            const json & a = n.at("attr");
            ggml_tensor * x0 = in(n, 0);
            const bool fast = a.value("stride", 1) == 1 && a.value("dilation", 1) == 1 && a.value("groups", 1) == x0->ne[1] &&
                              a.value("kernel", 0) == a.at("ctx").get<int>() + 1 && m.get(n.at("p").get<std::string>() + ".weight.conv") &&
                              m.get(n.at("p").get<std::string>() + ".weight")->ne[2] == x0->ne[1] &&  // depthwise, Cout = C
                              (m.device == "cpu" || ssm_conv_cuda_ok(x0->ne[1], a.value("kernel", 0)));
            y = fast ? op_conv1d(n) : op_conv1d_general(n);  // both: ggml [T, C] = torch [C, T]
            r = 2;
        } else if (op == "conv_transpose1d_causal") {
            y = op_conv_t1d(n);
            r = 2;
        } else if (op == "snake") {
            y = op_snake(n);
        } else if (op == "timestep_embed") {
            y = op_timestep(n);
            r = 2;
        } else if (op == "const") {
            const json & a = n.at("attr");
            y              = weight(a.at("w"));
            if (a.contains("shape")) {
                std::vector<int64_t> sh = a.at("shape").get<std::vector<int64_t>>();
                y                       = reshape_ne(y, std::vector<int64_t>(sh.rbegin(), sh.rend()));
                r                       = (int) sh.size();
            } else {
                r = ggml_n_dims(y);
            }
        } else if (op == "repeat") {
            y = ggml_repeat(ctx, in(n, 0), in(n, 1));
            r = rank_of(n, 1);
        } else if (op == "concat") {
            const int d = ax(n.at("attr").at("dim").get<int>(), r);
            y           = cont(in(n, 0));
            for (size_t i = 1; i < n.at("in").size(); ++i) {
                y = ggml_concat(ctx, y, cont(in(n, i)), d);
            }
        } else if (op == "arange") {
            const json &         a  = n.at("attr");
            const int32_t        st = a.value("start", 0), en = a.at("n");
            std::vector<int32_t> v;
            for (int32_t i = st; i < en; ++i) {
                v.push_back(i);
            }
            y = host_const(Tensor::i32({ (int64_t) v.size() }, v));
            r = 1;
        } else if (op == "round") {
            y = ggml_round(ctx, in(n, 0));
        } else if (op == "scale") {
            y = ggml_scale(ctx, in(n, 0), n.at("attr").at("s").get<float>());
        } else if (op == "delta_rule") {
            y = op_delta_rule(n);
            r = 3;
        } else if (op == "sigmoid") {
            y = ggml_sigmoid(ctx, in(n, 0));
        } else if (op == "softplus") {
            y = ggml_softplus(ctx, in(n, 0));
        } else if (op == "silu") {
            y = ggml_silu(ctx, in(n, 0));
        } else if (op == "gelu") {
            y = ggml_gelu(ctx, in(n, 0));
        } else if (op == "tanh") {
            y = ggml_tanh(ctx, in(n, 0));
        } else if (op == "exp") {
            y = ggml_exp(ctx, in(n, 0));
        } else if (op == "add" || op == "mul") {
            ggml_tensor * a0 = in(n, 0);
            ggml_tensor * a1 = operand2(n);
            if (!ggml_can_repeat(a1, a0) && ggml_can_repeat(a0, a1)) {
                std::swap(a0, a1);
            }
            y = op == "add" ? ggml_add(ctx, a0, a1) : ggml_mul(ctx, a0, a1);
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
        if ((op == "add" || op == "mul" || op == "sub" || op == "div") && n.at("in").size() > 1) {
            r = std::max(r, rank_of(n, 1));
        }
        nd[outs[0].get<std::string>()] = r;
    }
};

}  // namespace

// every attn node on state sid takes its V from a ttf.linear (through reshape / view / permute only) and the model's
// D is bf16: the V values are bf16 numbers (ttf_mm rounds its output to D)
static bool v_is_d_rounded(const Model & m, const std::string & sid) {
    if (!m.d_bf16 || std::getenv("TTF_KV_V_F32")) {
        return false;
    }
    bool any = false;
    for (const auto & [gname, g] : m.ir.at("graphs").items()) {
        std::map<std::string, const json *> prod;
        for (const auto & n : g.at("nodes")) {
            for (const auto & o : n.value("out", json::array())) {
                prod[o.get<std::string>()] = &n;
            }
        }
        for (const auto & n : g.at("nodes")) {
            if (n.at("op") != "attn" || n.value("attr", json::object()).value("state", "") != sid) {
                continue;
            }
            any = true;
            const auto & ins = n.at("in");
            if (ins.size() < 3 || ins[2].is_null()) {
                return false;
            }
            std::string v = ins[2];
            for (int hop = 0; hop < 16; ++hop) {
                auto it = prod.find(v);
                if (it == prod.end()) {
                    return false;
                }
                const std::string op = it->second->at("op");
                if (op == "ttf.linear") {
                    break;
                }
                if (op != "reshape" && op != "view" && op != "permute") {
                    return false;
                }
                v = it->second->at("in")[0];
            }
            if (prod.count(v) == 0 || prod.at(v)->at("op") != "ttf.linear") {
                return false;
            }
        }
    }
    return any;
}

Session::Session(const Model & m_, int n_threads_, int64_t max_ctx_) : m(m_), n_threads(n_threads_), max_ctx(max_ctx_) {
    backend = m.backend;
    m.use_threads(n_threads);
    fa       = m.device == "cpu" && !getenv("TTF_NO_FA");
    fa_exact = getenv("TTF_FA_EXACT") != nullptr;
    dfa      = m.device != "cpu" && !getenv("TTF_NO_CUDA_FA");
    const json & st = m.ir.value("state", json::array());
    ggml_init_params ip = { ggml_tensor_overhead() * (st.size() * 2 + 4), nullptr, true };
    sctx                = ggml_init(ip);
    for (const auto & e : st) {
        const std::string kind = e.at("kind");
        const std::string id   = e.at("id");
        if (kind == "conv") {  // the last `len` input frames, [len, C] time fastest
            conv[id] = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, e.at("len").get<int64_t>(), e.at("channels").get<int64_t>());
            continue;
        }
        if (kind == "rec") {  // delta-rule state, ggml [dv, dk, H, 1] = S^T per head
            const int64_t dk = e.at("hd_k"), dv = e.at("hd_v");
            rec[id] = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, dv, dk, e.at("heads").get<int64_t>(), 1);
            continue;
        }
        if (kind != "kv") {
            throw std::runtime_error("state kind " + kind + ": not yet");
        }
        Kv kvs;
        kvs.hd  = e.at("head_dim");
        kvs.hkv = e.at("heads_kv");
        kvs.k   = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, kvs.hd * kvs.hkv, max_ctx);
        // CUDA: V in BF16 when every attn on this state takes V straight from a ttf.linear in a D = bf16 model
        // (its output is rounded to bf16, so the cache is lossless; op_attn checks it). TTF_KV_V_F32=1: F32.
        const ggml_type vt = dfa && v_is_d_rounded(m, id) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        kvs.v   = fa || dfa ? ggml_new_tensor_2d(sctx, vt, kvs.hd * kvs.hkv, max_ctx)
                     : ggml_new_tensor_2d(sctx, GGML_TYPE_F32, max_ctx, kvs.hd * kvs.hkv);  // transposed (see op_attn)
        kvs.filled.assign(max_ctx, false);
        kv[e.at("id").get<std::string>()] = kvs;
    }
    sbuf = ggml_backend_alloc_ctx_tensors(sctx, backend);
    if (sbuf) {
        ggml_backend_buffer_clear(sbuf, 0);
    }
}

Session::~Session() {
    built.clear();
    for (auto & [k, g] : gmem) {
        ggml_gallocr_free(g.galloc);
    }
    for (auto & [k, c] : consts) {
        ggml_backend_buffer_free(c.buf);
        ggml_free(c.ctx);
    }
    if (sbuf) {
        ggml_backend_buffer_free(sbuf);
    }
    if (sctx) {
        ggml_free(sctx);
    }
}

void Session::reset() {
    for (auto & [id, k] : kv) {
        k.filled.assign(max_ctx, false);
    }
    if (sbuf) {
        ggml_backend_buffer_clear(sbuf, 0);
    }
}

std::vector<uint8_t> Session::save_recurrent() const {
    size_t n = 0;
    for (const auto * mp : { &conv, &rec }) {
        for (const auto & [id, t] : *mp) {
            n += ggml_nbytes(t);
        }
    }
    std::vector<uint8_t> b(n);
    size_t               o = 0;
    for (const auto * mp : { &conv, &rec }) {
        for (const auto & [id, t] : *mp) {
            ggml_backend_tensor_get(t, b.data() + o, 0, ggml_nbytes(t));
            o += ggml_nbytes(t);
        }
    }
    return b;
}

void Session::load_recurrent(const std::vector<uint8_t> & b) {
    size_t o = 0;
    for (const auto * mp : { &conv, &rec }) {
        for (const auto & [id, t] : *mp) {
            if (o + ggml_nbytes(t) > b.size()) {
                throw std::runtime_error("load_recurrent: size mismatch");
            }
            ggml_backend_tensor_set(t, b.data() + o, 0, ggml_nbytes(t));
            o += ggml_nbytes(t);
        }
    }
    if (o != b.size()) {
        throw std::runtime_error("load_recurrent: size mismatch");
    }
}

void Session::truncate(int32_t n) {
    for (auto & [id, k] : kv) {
        for (int64_t j = std::max<int32_t>(n, 0); j < max_ctx; ++j) {
            k.filled[j] = false;
        }
    }
}

StepOut Session::step(const std::string & graph, const std::vector<int32_t> & ids, const std::vector<int32_t> & pos,
                      bool last_only) {
    Values in;
    in["ids"] = Tensor::i32({ (int64_t) ids.size() }, ids);
    in["pos"] = Tensor::i32({ (int64_t) pos.size() }, pos);
    Values  o = run(graph, in, last_only);
    StepOut r;
    Tensor & lo = o.begin()->second;
    r.n_vocab   = lo.shape.back();
    r.T_out     = lo.numel() / r.n_vocab;
    r.logits    = std::move(lo.f);
    return r;
}

Values Session::run(const std::string & graph, const Values & in, bool last_only) {
    m.use_threads(n_threads);  // sessions share the backend: each runs with its own thread count
    const auto t0 = std::chrono::steady_clock::now();
    const json & g = m.ir.at("graphs").at(graph);
    // T: the positions' length, else the first input's leading dim
    const Tensor * pin = in.count("pos") ? &in.at("pos") : nullptr;
    int64_t        T   = 1;
    if (pin) {
        T = pin->numel();
    } else if (!g.at("inputs").empty()) {
        const Tensor & f0 = in.at(g.at("inputs")[0].at("name").get<std::string>());
        T                 = f0.shape.empty() ? 1 : f0.shape[0];
    }
    // the causal kv states this graph uses; filled first: a token sees itself and the earlier tokens of its call
    std::vector<Kv *> used;
    for (const auto & n : g.at("nodes")) {
        if (n.at("op") == "attn" && n.value("attr", json::object()).value("causal", true)) {
            used.push_back(&kv.at(n.at("attr").at("state").get<std::string>()));
        }
    }
    const std::vector<int32_t> pos = pin ? pin->i : std::vector<int32_t>();
    if (!used.empty() && !pin) {
        throw std::runtime_error("graph " + graph + ": causal attn needs a pos input");
    }
    for (int32_t p : pos) {
        if (p < 0 || p >= max_ctx) {
            throw std::runtime_error("run: position out of context");
        }
    }
    for (Kv * k : used) {
        for (int32_t p : pos) {
            k->filled[p] = true;
        }
    }
    int64_t S = 0;
    if (!used.empty()) {
        const auto & f = used[0]->filled;
        for (Kv * k : used) {
            if (k->filled != f) {
                throw std::runtime_error("graph " + graph + ": kv states filled differently");
            }
        }
        for (int64_t j = max_ctx - 1; j >= 0; --j) {
            if (f[j]) {
                S = j + 1;
                break;
            }
        }
        // stable shapes across decode steps (CUDA graph reuse, no allocator re-plans): the key length padded to 256;
        // the masked keys get probability exactly 0 (the cache is zeroed at start / reset)
        S = std::min<int64_t>(max_ctx, (S + 255) / 256 * 256);
    }

    // a built graph is kept per (graph, T, S, last_only, input shapes) and reused: inputs and mask are uploaded, nothing
    // is rebuilt or re-allocated (decode: one per S bucket). Not with TTF_DUMP / TTF_PROF_OPS, nor for T > 64 (prefill
    // lengths vary; their compute buffers are large)
    const char * dump_dir   = std::getenv("TTF_DUMP");
    const bool   prof_by_op = std::getenv("TTF_PROF_OPS") != nullptr;
    static const bool no_cache = std::getenv("TTF_NO_GRAPH_CACHE") != nullptr;
    const bool   cacheable  = !dump_dir && !prof_by_op && !no_cache && T <= 64;
    std::string  key = graph + "|" + std::to_string(T) + "|" + std::to_string(S) + "|" + (last_only ? "1" : "0");
    for (const auto & inp : g.at("inputs")) {
        const std::string name = inp.at("name");
        auto              it   = in.find(name);
        if (it == in.end()) {
            throw std::runtime_error("graph " + graph + ": missing input " + name);
        }
        key += "|" + name + ":" + std::to_string((int) it->second.type);
        for (int64_t d : it->second.shape) {
            key += "," + std::to_string(d);
        }
    }
    GraphMem & gm = gmem[graph + (T == 1 ? "/1" : "/n")];  // scratch: metadata for a build, allocator if not kept
    if (!gm.galloc) {
        gm.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        gm.gbuf.resize(ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false));
    }
    std::map<std::string, ggml_tensor *> dumped;  // TTF_DUMP: views made contiguous for the dump
    std::map<std::string, int>           dump_nd;
    // build the graph into ctx (graph capacity gsize): inputs, mask, outputs into e
    auto build = [&](ggml_context * ctx, size_t gsize, Built & e) {
        Builder b{ m, *this, ctx, {}, {}, {}, nullptr, nullptr, S, T, last_only, m.d_bf16 };
        for (const auto & inp : g.at("inputs")) {
            const std::string name = inp.at("name");
            const Tensor &    h    = in.at(name);
            int64_t           ne[4] = { 1, 1, 1, 1 };
            const int         r     = (int) h.shape.size();
            if (r > 4) {
                throw std::runtime_error("input rank > 4");
            }
            for (int i = 0; i < r; ++i) {
                ne[r - 1 - i] = h.shape[i];
            }
            ggml_tensor * t = ggml_new_tensor(ctx, h.type, std::max(r, 1), ne);
            ggml_set_name(t, name.c_str());
            ggml_set_input(t);
            b.val[name] = t;
            b.nd[name]  = std::max(r, 1);
            if (name == "pos") {
                b.pos = t;
            }
            e.inputs.emplace_back(name, t);
        }
        if (!used.empty()) {
            b.mask = ggml_new_tensor_2d(ctx, fa || dfa ? GGML_TYPE_F16 : GGML_TYPE_F32, S, T);  // flash attn: F16
            ggml_set_name(b.mask, "kq_mask");
            ggml_set_input(b.mask);
        }
        e.mask = b.mask;
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, gsize, false);
        for (const auto & n : g.at("nodes")) {
            b.node(n);
        }
        for (auto & [C, t] : b.vidx) {
            e.vidx.emplace_back(C, t);
        }
        // TTF_DUMP=DIR: every IR value of this call to DIR/<name>.bin (f32 / i32 raw) + DIR/index.json (ggml ne)
        if (dump_dir) {
            for (auto & [name, t] : b.val) {
                ggml_tensor * d = ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
                ggml_set_output(d);
                ggml_build_forward_expand(gf, d);
                dumped[name] = d;
            }
            dump_nd = b.nd;
        }
        for (ggml_tensor * w : b.writes) {
            ggml_build_forward_expand(gf, w);
        }
        for (const auto & o : g.at("outputs")) {
            ggml_tensor * t = b.val.at(o.at("of").get<std::string>());
            if (!ggml_is_contiguous(t)) {
                t = ggml_cont(ctx, t);
            }
            ggml_set_output(t);
            e.outs.push_back(t);
            e.out_nd.push_back(b.nd.at(o.at("of").get<std::string>()));
            ggml_build_forward_expand(gf, t);
        }
        if (m.device != "cpu" && !std::getenv("TTF_NO_QGROUP")) {
            group_quants(gf);
            group_mms(gf);
        }
        if (m.device != "cpu") {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor * nd = ggml_graph_node(gf, i);
                if (!ggml_backend_supports_op(backend, nd)) {
                    throw std::runtime_error(std::string("op not supported on ") + m.device + ": " + ggml_op_desc(nd) +
                                             " (" + nd->name + ")");
                }
            }
        }
        e.gf = gf;
    };
    auto scratch_ctx = [&]() {
        ggml_init_params ip = { gm.gbuf.size(), gm.gbuf.data(), true };
        return ggml_init(ip);
    };
    Built   tmp;  // the not-kept case: scratch metadata, the scratch allocator
    Built * e   = nullptr;
    bool    fresh = false;
    auto    hit = cacheable ? built.find(key) : built.end();
    // kept from the second call with a key on (one-off shapes such as a vae_decode length are never kept); one-token
    // decode graphs at once (every step repeats them; a 27B build is ~70 ms)
    bool keep = false;
    if (cacheable && hit == built.end()) {
        if (seen_keys.size() > 4096) {
            seen_keys.clear();
        }
        keep = !seen_keys.insert(key).second || T == 1;
    }
    if (hit != built.end()) {
        e = hit->second.get();
    } else if (keep) {
        // measure in the scratch, then build again into exactly sized metadata kept with the entry
        Built probe;
        probe.ctx = scratch_ctx();
        size_t used_mem = 0, gsize = 0;
        try {
            build(probe.ctx, GRAPH_SIZE, probe);
            const size_t n = (size_t) ggml_graph_n_nodes(probe.gf) + (size_t) probe.gf->n_leafs;
            gsize          = 64;
            while (gsize < n + n / 4 + 16) {
                gsize *= 2;
            }
            used_mem = ggml_used_mem(probe.ctx) - ggml_graph_overhead_custom(GRAPH_SIZE, false) +
                       ggml_graph_overhead_custom(gsize, false) + 64 * 1024;
        } catch (...) {
            ggml_free(probe.ctx);
            probe.ctx = nullptr;
            throw;
        }
        ggml_free(probe.ctx);
        probe.ctx = nullptr;
        if (built.size() >= 48) {  // drop the least recently used
            auto lru = built.begin();
            for (auto i = built.begin(); i != built.end(); ++i) {
                lru = i->second->use < lru->second->use ? i : lru;
            }
            built.erase(lru);
        }
        auto ne  = std::make_unique<Built>();
        ne->gbuf.resize(used_mem);
        ggml_init_params ip = { ne->gbuf.size(), ne->gbuf.data(), true };
        ne->ctx = ggml_init(ip);
        build(ne->ctx, gsize, *ne);
        ne->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        e          = ne.get();
        built[key] = std::move(ne);
        fresh      = true;
    } else {
        tmp.ctx = scratch_ctx();
        try {
            build(tmp.ctx, GRAPH_SIZE, tmp);
        } catch (...) {
            ggml_free(tmp.ctx);
            tmp.ctx = nullptr;
            throw;
        }
        e     = &tmp;
        fresh = true;
    }
    e->use               = ++use_clock;
    ggml_cgraph * gf     = e->gf;
    const auto    t1     = std::chrono::steady_clock::now();
    ggml_gallocr_t ga    = e == &tmp ? gm.galloc : e->galloc;
    if (fresh && !ggml_gallocr_alloc_graph(ga, gf)) {
        throw std::runtime_error("graph allocation failed");
    }
    if (fresh && e != &tmp) {  // kept compute buffers capped (TTF_GRAPH_CACHE_MB, default 1024): drop the LRU others
        static const size_t cap = (size_t) (std::getenv("TTF_GRAPH_CACHE_MB") ? std::atoll(std::getenv("TTF_GRAPH_CACHE_MB")) : 1024) << 20;
        for (;;) {
            size_t tot = 0;
            auto   lru = built.end();
            for (auto i = built.begin(); i != built.end(); ++i) {
                tot += ggml_gallocr_get_buffer_size(i->second->galloc, 0);
                if (i->second.get() != e && (lru == built.end() || i->second->use < lru->second->use)) {
                    lru = i;
                }
            }
            if (tot <= cap || lru == built.end()) {
                break;
            }
            built.erase(lru);
        }
    }
    prof_galloc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    for (auto & [name, t] : e->inputs) {
        const Tensor & h = in.at(name);
        if (h.numel() != ggml_nelements(t) || (h.type == GGML_TYPE_F32 ? h.f.size() : h.i.size()) != (size_t) h.numel()) {
            throw std::runtime_error(std::string("input ") + t->name + ": size");
        }
        // async on the compute stream: no per-input synchronisation (the host data outlives the call)
        ggml_backend_tensor_set_async(backend, t,
                                      h.type == GGML_TYPE_F32 ? (const void *) h.f.data() : (const void *) h.i.data(), 0,
                                      ggml_nbytes(t));
    }
    std::vector<std::vector<int32_t>> vix(e->vidx.size());
    for (size_t i = 0; i < e->vidx.size(); ++i) {
        const int64_t C = e->vidx[i].first;
        vix[i].resize(C * T);
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t c = 0; c < C; ++c) {
                vix[i][t * C + c] = (int32_t) (c * max_ctx + pos[t]);
            }
        }
        ggml_backend_tensor_set_async(backend, e->vidx[i].second, vix[i].data(), 0, vix[i].size() * sizeof(int32_t));
    }
    std::vector<float>       mk;
    std::vector<ggml_fp16_t> mk16;
    if (e->mask) {
        const auto & f = used[0]->filled;
        mk.resize(S * T);
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t j = 0; j < S; ++j) {
                mk[t * S + j] = (f[j] && j <= pos[t]) ? 0.0f : -std::numeric_limits<float>::infinity();
            }
        }
        if (e->mask->type == GGML_TYPE_F16) {
            mk16.resize(mk.size());
            ggml_fp32_to_fp16_row(mk.data(), mk16.data(), (int64_t) mk.size());
            ggml_backend_tensor_set_async(backend, e->mask, mk16.data(), 0, mk16.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set_async(backend, e->mask, mk.data(), 0, mk.size() * sizeof(float));
        }
    }
    if (fresh) {  // bytes read per call (the roofline numerator)
        std::map<const ggml_tensor *, std::pair<int, int64_t>> seen;  // per tensor: kind, the most any op reads of it
        e->wbytes = e->sbytes = 0;
        auto in_model = [&](const ggml_tensor * t) {
            const ggml_tensor * o = t->view_src ? t->view_src : t;
            if (!o->buffer) {  // CPU: the file's tensors (a weight already in D's type, e.g. a bf16 head) live in
                return o->data ? 1 : 0;  // the gguf context's own memory
            }
            if (o->buffer == sbuf) {
                return 2;
            }
            for (ggml_backend_buffer_t mb : m.bufs) {
                if (o->buffer == mb) {
                    return 1;
                }
            }
            return 0;
        };
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            const ggml_tensor * nd = ggml_graph_node(gf, i);
            if (nd->op == GGML_OP_VIEW || nd->op == GGML_OP_RESHAPE || nd->op == GGML_OP_PERMUTE ||
                nd->op == GGML_OP_TRANSPOSE || nd->op == GGML_OP_NONE) {
                continue;  // metadata only; the op that reads the view counts the view's bytes
            }
            for (int k = 0; k < GGML_MAX_SRC && nd->src[k]; ++k) {
                const ggml_tensor * t = nd->src[k];
                const int           w = in_model(t);
                if (!w) {
                    continue;
                }
                // a get_rows source: the rows gathered; a set_rows / cpy destination: the bytes written
                // a strided view (the transposed V cache: S of max_ctx per row) reads its elements, not its span
                int64_t nb = ggml_is_contiguous(t) ? (int64_t) ggml_nbytes(t)
                                                   : ggml_nelements(t) * (int64_t) ggml_type_size(t->type) / ggml_blck_size(t->type);
                if (nd->op == GGML_OP_GET_ROWS && k == 0) {
                    nb = ggml_nelements(nd->src[1]) * t->nb[1];
                } else if (nd->op == GGML_OP_SET_ROWS && k == 2) {  // src 0 the rows, 2 the destination
                    nb = ggml_nelements(nd->src[0]) * (int64_t) ggml_type_size(t->type) / ggml_blck_size(t->type);
                } else if (nd->op == GGML_OP_CPY && k == 1) {
                    nb = ggml_nelements(nd->src[0]) * (int64_t) ggml_type_size(t->type) / ggml_blck_size(t->type);
                }
                auto & se = seen[t];  // a tied embedding: one row by get_rows, all of it by the head
                se        = { w, std::max(se.second, nb) };
            }
        }
        for (const auto & [t, se] : seen) {
            (se.first == 1 ? e->wbytes : e->sbytes) += se.second;
        }
    }
    last_wbytes   = e->wbytes;
    last_sbytes   = e->sbytes;
    const auto t2 = std::chrono::steady_clock::now();
    if (prof_by_op) {  // node by node, time per op (+ part shape for TTF_MM)
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
        throw std::runtime_error("graph compute failed");
    }
    if (dump_dir) {
        json idx = json::object();
        for (auto & [name, t] : dumped) {
            if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) {
                continue;
            }
            std::vector<char> buf(ggml_nbytes(t));
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
            std::ofstream f(std::string(dump_dir) + "/" + name + ".bin", std::ios::binary);
            f.write(buf.data(), (std::streamsize) buf.size());
            idx[name] = { { "ne", { t->ne[0], t->ne[1], t->ne[2], t->ne[3] } }, { "nd", dump_nd[name] },
                          { "type", t->type == GGML_TYPE_F32 ? "f32" : "i32" } };
        }
        std::ofstream(std::string(dump_dir) + "/index.json") << idx.dump();
    }
    Values r;
    size_t oi = 0;
    for (const auto & o : g.at("outputs")) {
        ggml_tensor * t  = e->outs.at(oi);
        const int     nd = e->out_nd.at(oi++);
        Tensor        h;
        h.type = GGML_TYPE_F32;
        for (int i = nd - 1; i >= 0; --i) {
            h.shape.push_back(t->ne[i]);
        }
        if (t->type != GGML_TYPE_F32) {
            throw std::runtime_error("output " + o.at("name").get<std::string>() + ": not f32");
        }
        h.f.resize(ggml_nelements(t));
        if (ggml_is_contiguous(t)) {
            ggml_backend_tensor_get(t, h.f.data(), 0, ggml_nbytes(t));
        } else {
            throw std::runtime_error("output " + o.at("name").get<std::string>() + ": not contiguous");
        }
        r[o.at("name").get<std::string>()] = std::move(h);
    }
    const auto t3 = std::chrono::steady_clock::now();
    last_ms       = std::chrono::duration<double, std::milli>(t3 - t0).count();
    prof_build += std::chrono::duration<double, std::milli>(t1 - t0).count();
    prof_alloc += std::chrono::duration<double, std::milli>(t2 - t1).count();
    prof_compute += std::chrono::duration<double, std::milli>(t3 - t2).count();
    prof_nodes = ggml_graph_n_nodes(gf);
    return r;
}

Session::Built::~Built() {
    if (galloc) {
        ggml_gallocr_free(galloc);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

}  // namespace ttf
