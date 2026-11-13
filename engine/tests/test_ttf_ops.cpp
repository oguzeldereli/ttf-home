// test-ttf-ops: GGML_OP_TTF_QUANT + GGML_OP_TTF_MM on random data, CPU (Q2P and TQ1_P) vs CUDA (TQ1_P and Q2P):
// every output must be bit-identical. Cases cover T 1 / 3 / 17, K with padding, adapter ranks 0 / 24 / 104, acc / r / b.
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
    ggml_tensor * tB  = cs.rank ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cs.K, cs.rank) : nullptr;
    ggml_tensor * tA  = cs.rank ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cs.rank, cs.N) : nullptr;
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
    if (cs.rank) {
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

}  // namespace

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
        };
        std::vector<V> vs = { { "cpu tq1p", cpu, false } };
#ifdef TTF_CUDA
        vs.push_back({ "cuda tq1p", gpu, false });
        vs.push_back({ "cuda q2p", gpu, true });
#endif
        for (const V & v : vs) {
            auto    o    = run(v.be, v.q2p, cs, trits, x, mu, c, mB, mA, r, b, acc, beta);
            int64_t diff = 0;
            for (size_t i = 0; i < o.size(); ++i) {
                diff += std::memcmp(&o[i], &ref[i], 4) != 0;
            }
            std::printf("K %5lld N %4lld T %2lld rank %3d side %d bf16 %d  %-9s vs cpu q2p: %lld / %zu differ\n", (long long) cs.K,
                        (long long) cs.N, (long long) cs.T, cs.rank, cs.side, cs.bf16, v.name, (long long) diff, o.size());
            fails += diff != 0;
        }
    }
    std::printf(fails ? "FAIL\n" : "OK: all bit-identical\n");
    return fails != 0;
}
