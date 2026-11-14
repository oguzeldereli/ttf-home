// test-ttf-ops: GGML_OP_TTF_QUANT + GGML_OP_TTF_MM on random data, CPU (Q2P and TQ1_P) vs CUDA (TQ1_P and Q2P):
// every output must be bit-identical. Also TTF_I4 (int-k) GET_ROWS / MUL_MAT against a BF16 copy of the weight. Cases cover T 1 / 3 / 17, K with padding, adapter ranks 0 / 16 / 24 / 104 (F32, and BF16 for bf16 D), acc / r / b.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#ifdef TTF_CUDA
#    include "ggml-cuda.h"
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <random>
#include <vector>

namespace {

struct Case {
    int64_t K, N, T;
    int     rank;
    bool    side, bf16;
};

float bf16r(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u;
    std::memcpy(&v, &u, 4);
    return v;
}

// adapters stored as BF16 (bf16 cases: their values are bf16, so the outputs must not change)
bool g_adp_bf16 = false;

// runs the case on one backend with one trit layout; returns the output
std::vector<float> run(ggml_backend_t be, bool q2p, const Case & cs, const std::vector<int8_t> & trits,
                       const std::vector<float> & x, const std::vector<float> & mu, const std::vector<float> & c,
                       const std::vector<float> & mB, const std::vector<float> & mA, const std::vector<float> & r,
                       const std::vector<float> & b, const std::vector<float> & acc, float beta, int reps = 0) {
    const int64_t Kp = q2p ? (cs.K + 127) / 128 * 128 : (cs.K + 159) / 160 * 160;
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_tensor *    w   = q2p ? ggml_new_tensor_2d(ctx, GGML_TYPE_I8, Kp / 4, cs.N) : ggml_new_tensor_2d(ctx, GGML_TYPE_TQ1_P, Kp, cs.N);
    ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cs.K, cs.T);
    ggml_tensor * tmu = cs.side ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cs.K) : nullptr;
    ggml_tensor * tc  = cs.side ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cs.K) : nullptr;
    const ggml_type at = g_adp_bf16 && cs.bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    ggml_tensor * tB  = cs.rank ? ggml_new_tensor_2d(ctx, at, cs.K, cs.rank) : nullptr;
    ggml_tensor * tA  = cs.rank ? ggml_new_tensor_2d(ctx, at, cs.rank, cs.N) : nullptr;
    ggml_tensor * tr  = cs.side ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cs.N) : nullptr;
    ggml_tensor * tb  = cs.side ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cs.N) : nullptr;
    ggml_tensor * tac = cs.side ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cs.N, cs.T) : nullptr;
    ggml_tensor * tbe = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * qa  = ggml_ttf_quant(ctx, tx, tmu, tc, tB, Kp, cs.bf16);
    ggml_tensor * y   = ggml_ttf_mm(ctx, w, qa, tbe, tA, tr, tb, tac, cs.bf16);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    // weights: trits [N, K] -> layout rows of Kp
    std::vector<uint8_t> wb(ggml_nbytes(w), 0);
    const size_t         rs = w->nb[1];
    for (int64_t n = 0; n < cs.N; ++n) {
        if (q2p) {
            uint8_t * o = wb.data() + n * rs;
            for (int64_t k = 0; k < Kp; ++k) {
                const int t    = k < cs.K ? trits[n * cs.K + k] : 0;
                const int64_t blk = k / 128, e = k % 128;
                o[blk * 32 + e % 32] |= (uint8_t) ((t + 1) << (2 * (e / 32)));
            }
        } else {
            std::vector<float> row(Kp, 0.0f);
            for (int64_t k = 0; k < cs.K; ++k) {
                row[k] = (float) trits[n * cs.K + k];
            }
            ggml_quantize_chunk(GGML_TYPE_TQ1_P, row.data(), wb.data() + n * rs, 0, 1, Kp, nullptr);
        }
    }
    ggml_backend_tensor_set(w, wb.data(), 0, wb.size());
    ggml_backend_tensor_set(tx, x.data(), 0, ggml_nbytes(tx));
    ggml_backend_tensor_set(tbe, &beta, 0, 4);
    if (cs.side) {
        ggml_backend_tensor_set(tmu, mu.data(), 0, ggml_nbytes(tmu));
        ggml_backend_tensor_set(tc, c.data(), 0, ggml_nbytes(tc));
        ggml_backend_tensor_set(tr, r.data(), 0, ggml_nbytes(tr));
        ggml_backend_tensor_set(tb, b.data(), 0, ggml_nbytes(tb));
        ggml_backend_tensor_set(tac, acc.data(), 0, ggml_nbytes(tac));
    }
    if (cs.rank && at == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> hB(mB.size()), hA(mA.size());
        ggml_fp32_to_bf16_row_ref(mB.data(), hB.data(), (int64_t) hB.size());
        ggml_fp32_to_bf16_row_ref(mA.data(), hA.data(), (int64_t) hA.size());
        ggml_backend_tensor_set(tB, hB.data(), 0, ggml_nbytes(tB));
        ggml_backend_tensor_set(tA, hA.data(), 0, ggml_nbytes(tA));
    } else if (cs.rank) {
        ggml_backend_tensor_set(tB, mB.data(), 0, ggml_nbytes(tB));
        ggml_backend_tensor_set(tA, mA.data(), 0, ggml_nbytes(tA));
    }
    ggml_backend_graph_compute(be, gf);
    if (reps > 0) {  // bench: quant + mm, repeated (one sync each)
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            ggml_backend_graph_compute(be, gf);
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("bench K %5lld N %6lld T %3lld rank %3d %s: %8.1f us per quant+mm, weights %.1f GB/s\n", (long long) cs.K,
                    (long long) cs.N, (long long) cs.T, cs.rank, q2p ? "q2p " : "tq1p", us, ggml_nbytes(w) / us / 1e3);
    }
    std::vector<float> out(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

// TTF_I4 (int-k): GET_ROWS and MUL_MAT on one backend; w as raw blocks (160 bytes per 256: f32 lo[4], s[4], codes)
struct I4Out {
    std::vector<float> rows, y;
};
I4Out run_i4(ggml_backend_t be, int64_t K, int64_t V, int64_t T, const std::vector<uint8_t> & wb,
             const std::vector<int32_t> & ids, const std::vector<float> & x, ggml_type wt = GGML_TYPE_TTF_I4,
             const std::vector<uint16_t> * w16 = nullptr) {
    ggml_init_params ip  = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_tensor *    w   = ggml_new_tensor_2d(ctx, wt, K, V);
    ggml_tensor *    ti  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) ids.size());
    ggml_tensor *    tx  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_tensor *    gr  = ggml_get_rows(ctx, w, ti);
    ggml_tensor *    y   = ggml_mul_mat(ctx, w, tx);
    ggml_cgraph *    gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, gr);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(w, w16 ? (const void *) w16->data() : (const void *) wb.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(ti, ids.data(), 0, ggml_nbytes(ti));
    ggml_backend_tensor_set(tx, x.data(), 0, ggml_nbytes(tx));
    ggml_backend_graph_compute(be, gf);
    I4Out o;
    o.rows.resize(ggml_nelements(gr));
    o.y.resize(ggml_nelements(y));
    ggml_backend_tensor_get(gr, o.rows.data(), 0, ggml_nbytes(gr));
    ggml_backend_tensor_get(y, o.y.data(), 0, ggml_nbytes(y));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return o;
}

// CPU TTF_I4 = CPU BF16 copy (bit-identical, rows and products); CUDA rows bit-identical, CUDA products (own
// summation order) within 1e-5 of the largest output
int test_i4(ggml_backend_t cpu, ggml_backend_t gpu) {
    std::mt19937                    rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int                             fails = 0;
    for (const auto & [K, V, T] : std::vector<std::tuple<int64_t, int64_t, int64_t>>{ { 512, 300, 1 }, { 1280, 77, 3 }, { 5120, 1000, 1 }, { 256, 64, 9 } }) {
        const int64_t        nb = K / 256;
        std::vector<uint8_t> wb(V * nb * 160);
        for (int64_t i = 0; i < V * nb; ++i) {
            float lo[4], sc[4];
            for (int j = 0; j < 4; ++j) {
                lo[j] = 0.05f * nd(rng);
                sc[j] = 0.01f * std::fabs(nd(rng));
            }
            std::memcpy(&wb[i * 160], lo, 16);
            std::memcpy(&wb[i * 160 + 16], sc, 16);
            for (int b = 0; b < 128; ++b) {
                wb[i * 160 + 32 + b] = (uint8_t) (rng() & 255);
            }
        }
        // the reference BF16 weight: bf16(lo + s q), product and sum rounded to f32
        std::vector<uint16_t> w16(V * K);
        for (int64_t v = 0; v < V; ++v) {
            for (int64_t k = 0; k < K; ++k) {
                const uint8_t * bl = &wb[(v * nb + k / 256) * 160];
                const int       e  = (int) (k % 256);
                float           lo, sc;
                std::memcpy(&lo, bl + 4 * (e / 64), 4);
                std::memcpy(&sc, bl + 16 + 4 * (e / 64), 4);
                const int      q = (e & 1) ? (bl[32 + e / 2] >> 4) : (bl[32 + e / 2] & 15);
                volatile float p = sc * (float) q;
                volatile float f = lo + p;
                uint32_t       u;
                float          fv = f;
                std::memcpy(&u, &fv, 4);
                w16[v * K + k] = (uint16_t) ((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
            }
        }
        std::vector<int32_t> ids;
        for (int i = 0; i < 7; ++i) {
            ids.push_back((int32_t) (rng() % V));
        }
        std::vector<float> x(K * T);
        for (auto & e : x) {
            e = nd(rng);
        }
        const I4Out ref = run_i4(cpu, K, V, T, wb, ids, x, GGML_TYPE_BF16, &w16);
        auto        cmp = [&](const char * name, const I4Out & o, bool exact_y) {
            int64_t dr = 0, dy = 0;
            double  worst = 0.0;
            for (size_t i = 0; i < o.rows.size(); ++i) {
                dr += std::memcmp(&o.rows[i], &ref.rows[i], 4) != 0;
            }
            double ymax = 1e-30;  // error relative to the largest output (sums near 0 have no relative scale)
            for (size_t i = 0; i < o.y.size(); ++i) {
                ymax = std::max(ymax, (double) std::fabs(ref.y[i]));
            }
            for (size_t i = 0; i < o.y.size(); ++i) {
                dy += std::memcmp(&o.y[i], &ref.y[i], 4) != 0;
                worst = std::max(worst, (double) std::fabs(o.y[i] - ref.y[i]) / ymax);
            }
            const bool bad = dr != 0 || (exact_y ? dy != 0 : worst > 1e-5);
            std::printf("ttf_i4 K %5lld V %4lld T %lld  %-8s rows %lld / %zu differ, y %lld / %zu differ (worst rel %.2e)%s\n",
                        (long long) K, (long long) V, (long long) T, name, (long long) dr, o.rows.size(), (long long) dy,
                        o.y.size(), worst, bad ? "  FAIL" : "");
            fails += bad;
        };
        cmp("cpu", run_i4(cpu, K, V, T, wb, ids, x), true);
        if (gpu) {
            cmp("cuda", run_i4(gpu, K, V, T, wb, ids, x), false);
        }
    }
    return fails;
}

}  // namespace

// ARGMAX on CUDA vs the first index of the max (the host's argmax_first, the greedy pick): rows with planted ties
// (within a block, across the many-block kernel's chunks, at the ends), all equal, all negative; one-block and
// many-block paths (ncols < / >= 16384). Returns the failures.
int test_argmax(ggml_backend_t gpu) {
    if (!gpu) {
        return 0;
    }
    int          fails = 0;
    std::mt19937 rng(7);
    for (int64_t n : { 1000, 5000, 16384, 32000, 131072, 151936, 248320 }) {
        for (int kind = 0; kind < 6; ++kind) {
            std::vector<float> x(n);
            std::normal_distribution<float> nd(0.0f, 3.0f);
            for (auto & v : x) {
                v = kind == 4 ? 1.5f : kind == 5 ? -std::fabs(nd(rng)) - 1.0f : bf16r(nd(rng));
            }
            const float M = 40.0f;
            if (kind == 1) {  // ties far apart: the later one in an earlier-scanned lane / chunk
                x[n - 1] = M; x[n / 2] = M; x[n / 3 + 7] = M;
            } else if (kind == 2) {  // ties next to each other (same warp) and at a chunk edge
                x[2049] = M; x[2048] = M; x[2047 + (n > 4096 ? 2048 : 0)] = M;
            } else if (kind == 3) {  // the first element ties a late one
                x[0] = M; x[n - 2] = M;
            }
            int64_t ref = 0;
            for (int64_t j = 1; j < n; ++j) {
                ref = x[j] > x[ref] ? j : ref;
            }
            ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_tensor *    tx  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
            ggml_tensor *    am  = ggml_argmax(ctx, tx);
            ggml_cgraph *    gf  = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, am);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
            ggml_backend_tensor_set(tx, x.data(), 0, n * 4);
            ggml_backend_graph_compute(gpu, gf);
            int32_t got = -2;
            ggml_backend_tensor_get(am, &got, 0, 4);
            if (got != ref) {
                std::printf("argmax n %lld kind %d: cuda %d, first max %lld\n", (long long) n, kind, got, (long long) ref);
                ++fails;
            }
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    std::printf("argmax (ties to the first index): %d failures\n", fails);
    return fails;
}

// TTF_TOPK on CUDA vs the CPU op (the host sampler's list: value descending, ties by the lower index), bit for bit:
// random rows, planted ties across blocks, many equal values, -0 / +0, k 1 / 20 / 40 / 256. Returns the failures.
int test_topk(ggml_backend_t cpu, ggml_backend_t gpu) {
    if (!gpu) {
        return 0;
    }
    int          fails = 0;
    std::mt19937 rng(11);
    auto run1 = [](ggml_backend_t be, const std::vector<float> & x, int k) {
        ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_tensor *    tx  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) x.size());
        ggml_tensor *    tk  = ggml_ttf_top_k(ctx, tx, k);
        ggml_cgraph *    gf  = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, tk);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(tx, x.data(), 0, x.size() * 4);
        ggml_backend_graph_compute(be, gf);
        std::vector<float> o(2 * k);
        ggml_backend_tensor_get(tk, o.data(), 0, o.size() * 4);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return o;
    };
    for (int64_t n : { 300, 2048, 5000, 32000, 131072, 151936, 248320 }) {
        for (int kind = 0; kind < 4; ++kind) {
            for (int k : { 1, 20, 40, 256 }) {
                if (k >= n) {
                    continue;
                }
                std::vector<float> x(n);
                std::normal_distribution<float> nd(0.0f, 3.0f);
                for (auto & v : x) {
                    v = kind == 2 ? (float) (rng() % 7) : bf16r(nd(rng));  // kind 2: 7 values only (ties everywhere)
                }
                if (kind == 1) {  // equal top values far apart, in different blocks
                    for (int64_t j = 0; j < n; j += n / 37 + 1) {
                        x[j] = 30.0f;
                    }
                } else if (kind == 3) {  // -0 / +0 at the top
                    for (auto & v : x) {
                        v = -std::fabs(v) - 1.0f;
                    }
                    for (int64_t j = 0; j < n; j += n / 50 + 1) {
                        x[j] = (j / (n / 50 + 1)) % 2 ? 0.0f : -0.0f;
                    }
                }
                const auto a = run1(cpu, x, k), b = run1(gpu, x, k);
                if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
                    std::printf("top_k n %lld kind %d k %d: cuda != cpu (first: %g @%g vs %g @%g)\n", (long long) n, kind, k, b[0],
                                b[k], a[0], a[k]);
                    ++fails;
                }
            }
        }
    }
    std::printf("ttf_top_k cuda vs cpu: %d failures\n", fails);
    return fails;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench-cpu") {  // decode-shaped, Q2P, weights past the L3 (~70 MB)
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(cpu, argc > 2 ? std::atoi(argv[2]) : 8);
        for (const Case & cs : std::vector<Case>{ { 2560, 110592, 1, 0, false, true }, { 6912, 40960, 1, 0, false, true },
                                                 { 2560, 110592, 4, 0, false, true } }) {
            std::vector<int8_t> tr(cs.N * cs.K, 1);
            std::vector<float>  x(cs.K * cs.T, 0.5f), mu, c, mB, mA, r, b, acc;
            run(cpu, true, cs, tr, x, mu, c, mB, mA, r, b, acc, 0.01f, 20);
        }
        return 0;
    }
    if (argc > 5 && std::string(argv[1]) == "--bench-cold") {  // --bench-cold K N T rank: NC weight copies (> L2), CUDA
#ifdef TTF_CUDA
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        const int64_t  K = std::atoll(argv[2]), N = std::atoll(argv[3]), T = std::atoll(argv[4]);
        const int      rank = std::atoi(argv[5]);
        const int64_t  Kp   = (K + 159) / 160 * 160;
        // TTF_BENCH_MB: the weight copies' total (default 400 MB, over the L2; the 27B reads ~6300 MB per token)
        const double   wmb  = std::getenv("TTF_BENCH_MB") ? std::atof(std::getenv("TTF_BENCH_MB")) : 400.0;
        const int      NC   = (int) std::max<int64_t>(4, (int64_t) (wmb * 1e6 / (double) (N * Kp / 5)));
        // TTF_BENCH_NODES: products per graph, cycling over the NC copies (default NC; more with few copies keeps
        // the per-graph overhead out of the time)
        const int      NN   = std::getenv("TTF_BENCH_NODES") ? std::max(NC, std::atoi(std::getenv("TTF_BENCH_NODES"))) : NC;
        ggml_init_params ip = { (size_t) (16 * NN + 64) * ggml_tensor_overhead() + ggml_graph_overhead_custom(16 * NN + 64, false), nullptr, true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_tensor * tx  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
        ggml_tensor * tmu = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K), * tc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
        ggml_tensor * tB  = rank ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, rank) : nullptr;
        ggml_tensor * tA  = rank ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, N) : nullptr;
        ggml_tensor * tr  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N), * tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_tensor * tbe = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_tensor * qa  = ggml_ttf_quant(ctx, tx, tmu, tc, tB, Kp, true);
        ggml_cgraph * gf  = ggml_new_graph_custom(ctx, 16 * NN + 64, false);
        std::vector<ggml_tensor *> ws;
        // TTF_BENCH_CHAIN=1: each product accumulates the previous one (a dependency chain, as a model's layers)
        const bool    chain = std::getenv("TTF_BENCH_CHAIN") != nullptr;
        ggml_tensor * prev  = nullptr;
        // TTF_BENCH_GGML=type name (tq1_0, tq2_0, q4_K, ...): ggml's MUL_MAT on that type instead of the TTF linear (the
        // fork's kernels at the same shape; K a multiple of the type's block)
        ggml_type gt = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("TTF_BENCH_GGML")) {
            for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                if (ggml_get_type_traits((ggml_type) t)->type_name && std::string(ggml_type_name((ggml_type) t)) == e) {
                    gt = (ggml_type) t;
                }
            }
            GGML_ASSERT(gt != GGML_TYPE_COUNT && K % ggml_blck_size(gt) == 0);
        }
        for (int i = 0; i < NN && gt != GGML_TYPE_COUNT; ++i) {
            if (i < NC) {
                ws.push_back(ggml_new_tensor_2d(ctx, gt, K, N));
            }
            ggml_build_forward_expand(gf, ggml_mul_mat(ctx, ws[i % NC], tx));
        }
        for (int i = 0; i < NN && gt == GGML_TYPE_COUNT; ++i) {
            if (i < NC) {  // TTF_BENCH_Q2=1: Q2P weights (I8 [Kp / 4, N], 2-bit codes; K % 640 == 0 so Kp fits both)
                ws.push_back(std::getenv("TTF_BENCH_Q2") ? ggml_new_tensor_2d(ctx, GGML_TYPE_I8, Kp / 4, N)
                                                         : ggml_new_tensor_2d(ctx, GGML_TYPE_TQ1_P, Kp, N));
            }
            ggml_tensor * w = ws[i % NC];
            ggml_tensor * mm = ggml_ttf_mm(ctx, w, qa, tbe, tA, tr, tb, chain ? prev : nullptr, true);
            prev             = mm;
            if (std::getenv("TTF_MM_R")) {  // the session's rows-per-warp hint (0 = shape rule)
                ((int32_t *) mm->op_params)[1] = std::atoi(std::getenv("TTF_MM_R"));
            }
            ggml_build_forward_expand(gf, mm);
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        ggml_backend_buffer_clear(buf, 0);
        std::mt19937 rng(3);
        // TTF_BENCH_P0: fraction of zero trits (default: uniform over -1, 0, 1)
        const double p0 = std::getenv("TTF_BENCH_P0") ? std::atof(std::getenv("TTF_BENCH_P0")) : -1.0;
        std::vector<float>   row(Kp, 0.0f);
        std::vector<uint8_t> wb(ggml_nbytes(ws[0]));
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < K; ++k) {
                row[k] = p0 >= 0 ? ((rng() % 1000) < p0 * 1000 ? 0.0f : (rng() % 2 ? 1.0f : -1.0f))
                                 : (float) ((int) (rng() % 3) - 1);
            }
            if (gt != GGML_TYPE_COUNT) {
                ggml_quantize_chunk(gt, row.data(), wb.data() + n * ws[0]->nb[1], 0, 1, K, nullptr);
            } else if (ws[0]->type == GGML_TYPE_I8) {
                for (int64_t k = 0; k < Kp / 4; ++k) {
                    wb[n * ws[0]->nb[1] + k] = (uint8_t) (rng() % 3 | (rng() % 3) << 2 | (rng() % 3) << 4 | (rng() % 3) << 6);
                }
            } else {
                ggml_quantize_chunk(GGML_TYPE_TQ1_P, row.data(), wb.data() + n * ws[0]->nb[1], 0, 1, Kp, nullptr);
            }
        }
        for (auto * w : ws) {
            ggml_backend_tensor_set(w, wb.data(), 0, wb.size());
        }
        std::vector<float> ones(std::max<int64_t>(K, N), 1.0f), xs(K * T, 0.3f), sm(std::max<int64_t>(K, N) * std::max(rank, 1), 0.01f);
        ggml_backend_tensor_set(tx, xs.data(), 0, ggml_nbytes(tx));
        ggml_backend_tensor_set(tc, ones.data(), 0, ggml_nbytes(tc));
        ggml_backend_tensor_set(tr, ones.data(), 0, ggml_nbytes(tr));
        if (rank) {
            ggml_backend_tensor_set(tB, sm.data(), 0, ggml_nbytes(tB));
            ggml_backend_tensor_set(tA, sm.data(), 0, ggml_nbytes(tA));
        }
        float beta = 0.01f;
        ggml_backend_tensor_set(tbe, &beta, 0, 4);
        ggml_backend_graph_compute(gpu, gf);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) {
            ggml_backend_graph_compute(gpu, gf);
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 10 / NN;
        std::printf("cold K %lld N %lld T %lld rank %d: %d copies, %.1f us per mm (incl. 1/%d quant), %.0f GB/s\n", (long long) K,
                    (long long) N, (long long) T, rank, NC, us, NN, ggml_nbytes(ws[0]) / us / 1e3);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
#endif
        return 0;
    }
    if (argc > 5 && std::string(argv[1]) == "--bench-case") {  // --bench-case K N T rank: CUDA, side vectors, bf16
#ifdef TTF_CUDA
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        const Case     cs  = { std::atoll(argv[2]), std::atoll(argv[3]), std::atoll(argv[4]), std::atoi(argv[5]), true, true };
        std::vector<int8_t> tr(cs.N * cs.K, 1);
        std::vector<float>  x(cs.K * cs.T, 0.5f), mu(cs.K, 0.0f), c(cs.K, 1.0f), mB(cs.K * cs.rank, 0.01f),
            mA(cs.rank * cs.N, 0.01f), r(cs.N, 1.0f), b(cs.N, 0.0f), acc(cs.N * cs.T, 0.0f);
        run(gpu, false, cs, tr, x, mu, c, mB, mA, r, b, acc, 0.01f, 300);
#endif
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--bench") {
#ifdef TTF_CUDA
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        std::mt19937   rng(1);
        for (const Case & cs : std::vector<Case>{ { 1024, 4096, 1, 0, true, true },  { 1024, 4096, 4, 0, true, true },
                                                 { 1024, 4096, 8, 0, true, true },  { 1024, 4096, 24, 0, true, true },
                                                 { 1024, 4096, 24, 104, true, true }, { 4096, 1024, 24, 0, true, true },
                                                 { 1024, 104, 24, 0, true, true },  { 1024, 3584, 1, 0, true, true },
                                                 { 1024, 4096, 512, 0, true, true }, { 4096, 1024, 512, 0, true, true } }) {
            std::vector<int8_t> tr(cs.N * cs.K, 1);
            std::vector<float>  x(cs.K * cs.T, 0.5f), mu(cs.K, 0.0f), c(cs.K, 1.0f), mB(cs.K * cs.rank, 0.01f),
                mA(cs.rank * cs.N, 0.01f), r(cs.N, 1.0f), b(cs.N, 0.0f), acc(cs.N * cs.T, 0.0f);
            run(gpu, false, cs, tr, x, mu, c, mB, mA, r, b, acc, 0.01f, 300);
        }
#endif
        return 0;
    }
    std::vector<Case> cases = {
        { 1024, 300, 1, 0, false, true },   { 1024, 300, 1, 24, true, true },  { 2816, 257, 3, 104, true, true },
        { 1000, 130, 17, 64, true, true },  { 2048, 64, 5, 104, true, false }, { 4096, 96, 1, 104, true, true },
        { 1024, 300, 40, 24, true, true },  { 2816, 257, 33, 0, true, true },  { 1000, 130, 64, 64, false, false },
        { 2048, 70, 8, 104, true, true },
        // rank 16 (the 27B's): preloaded adapter rows; decode split quant (K > 1024), odd K, the MMA path
        { 5120, 600, 1, 16, true, true },   { 17408, 300, 1, 16, true, true }, { 2050, 100, 1, 16, true, true },
        { 1024, 4096, 64, 16, true, true },
        // no adapters, decode split (one launch): K > 1024, rows x parts <= 8
        { 5120, 600, 1, 0, true, true },    { 17408, 300, 2, 0, true, false }, { 2050, 100, 3, 0, false, true },
    };
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
#ifdef TTF_CUDA
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
#endif
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> tri(-1, 1);
    int fails = 0;
    for (const Case & cs : cases) {
        auto D = [&](float v) { return cs.bf16 ? bf16r(v) : v; };
        std::vector<int8_t> trits(cs.N * cs.K);
        for (auto & t : trits) {
            t = (int8_t) tri(rng);
        }
        auto vec = [&](int64_t n, float sc, float off, bool d) {
            std::vector<float> v(n);
            for (auto & e : v) {
                e = off + sc * nd(rng);
                e = d ? D(e) : e;
            }
            return v;
        };
        auto x  = vec(cs.K * cs.T, 1.0f, 0.0f, false);
        auto mu = vec(cs.K, 0.1f, 0.0f, true), c = vec(cs.K, 0.2f, 1.0f, true);
        auto mB = vec(cs.K * cs.rank, 0.05f, 0.0f, true), mA = vec(cs.rank * cs.N, 0.05f, 0.0f, true);
        auto r = vec(cs.N, 0.1f, 1.0f, true), b = vec(cs.N, 0.1f, 0.0f, true), acc = vec(cs.N * cs.T, 1.0f, 0.0f, false);
        const float beta = 0.0123f;
        auto ref = run(cpu, true, cs, trits, x, mu, c, mB, mA, r, b, acc, beta);
        struct V {
            const char *   name;
            ggml_backend_t be;
            bool           q2p;
            bool           ah;  // BF16 adapters
        };
        std::vector<V> vs = { { "cpu tq1p", cpu, false, false } };
#ifdef TTF_CUDA
        vs.push_back({ "cuda tq1p", gpu, false, false });
        vs.push_back({ "cuda q2p", gpu, true, false });
#endif
        if (cs.bf16 && cs.rank) {
            vs.push_back({ "cpu q2p ah", cpu, true, true });
#ifdef TTF_CUDA
            vs.push_back({ "cuda tq1p ah", gpu, false, true });
            vs.push_back({ "cuda q2p ah", gpu, true, true });
#endif
        }
        for (const V & v : vs) {
            g_adp_bf16   = v.ah;
            auto    o    = run(v.be, v.q2p, cs, trits, x, mu, c, mB, mA, r, b, acc, beta);
            g_adp_bf16   = false;
            int64_t diff = 0;
            for (size_t i = 0; i < o.size(); ++i) {
                diff += std::memcmp(&o[i], &ref[i], 4) != 0;
            }
            std::printf("K %5lld N %4lld T %2lld rank %3d side %d bf16 %d  %-12s vs cpu q2p: %lld / %zu differ\n", (long long) cs.K,
                        (long long) cs.N, (long long) cs.T, cs.rank, cs.side, cs.bf16, v.name, (long long) diff, o.size());
            fails += diff != 0;
        }
    }
#ifdef TTF_CUDA
    fails += test_argmax(gpu);
    fails += test_topk(cpu, gpu);
    fails += test_i4(cpu, gpu);
#else
    fails += test_i4(cpu, nullptr);
#endif
    std::printf(fails ? "FAIL\n" : "OK: ttf ops bit-identical; ttf_i4 rows exact, products within tolerance\n");
    return fails != 0;
}
