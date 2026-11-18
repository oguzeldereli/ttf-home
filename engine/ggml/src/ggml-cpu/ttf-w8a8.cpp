// W8A8 CPU matmul for small batches of rows (ttf-home, 17 Nov 2026): a Q8_0 weight (blocks of 32, one fp16 scale
// each) is requantized once to int8 with one scale per output row, the activations to int8 with one scale per row
// (token), and every dot product runs over the whole K in integers (AVX-VNNI vpdpbusd, or AVX2 maddubs). ggml's Q8_0
// path converts to float after every 32-block, which bounds it at ~5 vector ops per 32 multiply-adds; this runs at ~1.
// VoxCPM2's DiT on a laptop i9-14900HX (M = 22 rows, AVX-VNNI): 3.5x ggml's Q8_0 per core (tile 4 x 3, 12 accumulators
// in named registers). The numerics differ from Q8_0 (row instead of block scales): on by TTF_W8A8=1.
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
#    define TTF_W8_VNNI 1
#    if defined(__AVXVNNI__)
#        define TTF_DPBUSD(acc, a, b) _mm256_dpbusd_avx_epi32(acc, a, b)
#    else
#        define TTF_DPBUSD(acc, a, b) _mm256_dpbusd_epi32(acc, a, b)
#    endif
#elif defined(__AVX2__)
#    define TTF_W8_VNNI 0
#endif

#if defined(TTF_W8_VNNI)

#ifndef TTF_W8_PD
#    define TTF_W8_PD 1
#endif

namespace {

// q holds the rows in panels of NT = 3 (18 Nov 2026): for panel p (rows 3p .. 3p + 2, zero rows past N) the 32-byte
// chunks of the three rows interleaved, [K / 32][3][32], so a tile reads ONE sequential stream (hardware prefetcher
// friendly; i7-9800X cold weights: +10% over three strided rows) and the software prefetch runs ahead in it.
constexpr int64_t W8_NT = 3;

inline size_t w8_off(int64_t n, int64_t k, int64_t K) {  // byte of element (n, k) in the panel layout
    return (size_t) (n / W8_NT) * W8_NT * K + (size_t) (k / 32) * (32 * W8_NT) + (size_t) (n % W8_NT) * 32 + k % 32;
}

struct W8 {
    int64_t              K = 0, N = 0;
    bool                 fresh     = true;   // rows not converted yet: the first call converts them, each thread its own
    bool                 conv_call = false;  // the converting call has run (thread 0's flag)
    std::vector<int8_t>  q;   // [N3, K] in panels (w8_off), N3 = N rounded up to 3
    std::vector<float>   s;   // [N]
    std::vector<int32_t> rs;  // [N] row sums (VNNI: the u8 offset's correction)
};

std::mutex                                   g_mu;
std::unordered_map<const void *, W8 *>       g_cache;  // by the weight's data pointer (weights do not move)

W8 * alloc(const ggml_tensor * w) {
    W8 * c = new W8;
    c->K   = w->ne[0];
    c->N   = w->ne[1];
    c->q.assign((size_t) ((c->N + W8_NT - 1) / W8_NT * W8_NT) * c->K, 0);   // (the pad rows stay 0)
    c->s.resize(c->N);
    c->rs.resize(c->N);
    return c;
}

// rows [n0, n1) of the Q8_0 weight into c (each thread converts the rows it multiplies, on the first call)
void convert_rows(const ggml_tensor * w, W8 * c, int64_t n0, int64_t n1) {
    const int64_t      K = w->ne[0], KB = K / 32;
    std::vector<float> f(K);
    for (int64_t n = n0; n < n1; ++n) {
        const block_q8_0 * b    = (const block_q8_0 *) ((const char *) w->data + n * w->nb[1]);
        float              amax = 0;
        for (int64_t j = 0; j < KB; ++j) {
            const float d = GGML_FP16_TO_FP32(b[j].d);
            for (int i = 0; i < 32; ++i) {
                f[j * 32 + i] = d * b[j].qs[i];
                amax          = std::fmax(amax, std::fabs(f[j * 32 + i]));
            }
        }
        const float s = amax / 127.f, is = s > 0 ? 1.f / s : 0.f;
        int32_t     sum = 0;
        for (int64_t k = 0; k < K; ++k) {
            const int q              = (int) std::lrintf(f[k] * is);
            c->q[w8_off(n, k, K)] = (int8_t) q;
            sum += q;
        }
        c->s[n]  = s;
        c->rs[n] = sum;
    }
}

// AVX2 without VNNI: 8-bit activations need vpsignb per product (maddubs wants one unsigned operand and |a| |w| 2
// <= 32767); TTF_W8A7=1 quantizes the activations to 7 bits instead (|a| <= 63, stored + 64 as u8: maddubs(u8, s8) is
// exact, 127 * 127 * 2 < 32767) and drops the vpsignb: 1.4x on the DiT shapes (cold weights). Off by default.
#if !TTF_W8_VNNI
const bool g_a7 = [] {
    const char * e = std::getenv("TTF_W8A7");
    return e && std::atoi(e) != 0;
}();
#else
constexpr bool g_a7 = false;
#endif

inline int hsum8i(__m256i v) {
    __m128i a = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    a         = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4e));
    a         = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xb1));
    return _mm_cvtsi128_si32(a);
}

// one activation row to int8 codes with one scale (round to nearest even, as lrintf); VNNI: stored + 128 (u8)
void quant_row(const float * p, int64_t K, uint8_t * q, float * sd) {
    const __m256 sgn = _mm256_set1_ps(-0.0f);
    __m256       mx  = _mm256_setzero_ps();
    for (int64_t k = 0; k < K; k += 8) {
        mx = _mm256_max_ps(mx, _mm256_andnot_ps(sgn, _mm256_loadu_ps(p + k)));
    }
    __m128 h = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
    h        = _mm_max_ps(h, _mm_movehl_ps(h, h));
    h        = _mm_max_ss(h, _mm_movehdup_ps(h));
    const float d = _mm_cvtss_f32(h) / (g_a7 ? 63.f : 127.f), id = d > 0 ? 1.f / d : 0.f;
    *sd           = d;
    const __m256  vi  = _mm256_set1_ps(id);
    const __m256i prm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t k = 0; k < K; k += 32) {
        const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + k), vi));
        const __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + k + 8), vi));
        const __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + k + 16), vi));
        const __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + k + 24), vi));
        __m256i       b  = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        b                = _mm256_permutevar8x32_epi32(b, prm);  // the packs interleave the 128-bit lanes
#if TTF_W8_VNNI
        b = _mm256_xor_si256(b, _mm256_set1_epi8((char) 0x80));  // + 128
#else
        if (g_a7) {
            b = _mm256_add_epi8(b, _mm256_set1_epi8(64));        // |a| <= 63 (scale amax / 63): + 64 in [1, 127]
        }
#endif
        _mm256_storeu_si256((__m256i *) (q + k), b);
    }
}

// rows [n0, n1) of W x all M activation rows; tile 4 x 3 (12 accumulators, 3 weight + 1 activation vector: 16 ymm);
// n0 a multiple of 3 (a panel's first row). A7: the a7 kernel (u8 a + 64), else the vpsignb one (AVX2) / VNNI.
template <bool A7>
void gemm_t(const W8 * w, int64_t K, int64_t n0, int64_t n1, const uint8_t * aq, const float * sa, int64_t M, float * y,
            size_t ldy) {
    constexpr int MT = 4, NT = (int) W8_NT;
#if TTF_W8_VNNI
#    define TTF_W8_DOT(acc, a, w) TTF_DPBUSD(acc, a, w)
#else  // A8: a holds signed codes: |w| x sign(a, w) stays within int16 pairs (|w|, |a| <= 127); A7: a in [1, 127]
    const __m256i ones = _mm256_set1_epi16(1);
#    define TTF_W8_DOT(acc, a, w)                                                                                  \
        (A7 ? _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), ones))                           \
            : _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(a, w)), ones)))
#endif
    constexpr int64_t PF = 1024;  // software prefetch distance in the panel stream (bytes)
    for (int64_t n = n0; n < n1; n += NT) {
        const int      nn = (int) std::min<int64_t>(NT, n1 - n);
        const int8_t * pn = w->q.data() + (size_t) n * K;   // the panel of rows n .. n + 2
        for (int64_t m = 0; m < M; m += MT) {
            const int       mm = (int) std::min<int64_t>(MT, M - m);
            const uint8_t * ar[MT];
            for (int i = 0; i < MT; ++i) {
                ar[i] = aq + (size_t) (m + (i < mm ? i : 0)) * K;
            }
            // named accumulators: as an array GCC keeps them on the stack across the loop
            __m256i c00 = _mm256_setzero_si256(), c01 = c00, c02 = c00, c10 = c00, c11 = c00, c12 = c00, c20 = c00, c21 = c00,
                    c22 = c00, c30 = c00, c31 = c00, c32 = c00;
            const uint8_t * a0 = ar[0], * a1 = ar[1], * a2 = ar[2], * a3 = ar[3];
            const int8_t *  q  = pn;
            const bool      pf = m == 0;   // the first m-tile streams the panel from memory; the rest hit the cache
            for (int64_t k = 0; k < K; k += 32, q += 32 * NT) {
                if (pf) {
                    _mm_prefetch((const char *) (q + PF), _MM_HINT_T0);
                    _mm_prefetch((const char *) (q + PF + 64), _MM_HINT_T0);
                }
                const __m256i v0 = _mm256_loadu_si256((const __m256i *) (q));
                const __m256i v1 = _mm256_loadu_si256((const __m256i *) (q + 32));
                const __m256i v2 = _mm256_loadu_si256((const __m256i *) (q + 64));
                __m256i       a  = _mm256_loadu_si256((const __m256i *) (a0 + k));
                c00 = TTF_W8_DOT(c00, a, v0);
                c01 = TTF_W8_DOT(c01, a, v1);
                c02 = TTF_W8_DOT(c02, a, v2);
                a   = _mm256_loadu_si256((const __m256i *) (a1 + k));
                c10 = TTF_W8_DOT(c10, a, v0);
                c11 = TTF_W8_DOT(c11, a, v1);
                c12 = TTF_W8_DOT(c12, a, v2);
                a   = _mm256_loadu_si256((const __m256i *) (a2 + k));
                c20 = TTF_W8_DOT(c20, a, v0);
                c21 = TTF_W8_DOT(c21, a, v1);
                c22 = TTF_W8_DOT(c22, a, v2);
                a   = _mm256_loadu_si256((const __m256i *) (a3 + k));
                c30 = TTF_W8_DOT(c30, a, v0);
                c31 = TTF_W8_DOT(c31, a, v1);
                c32 = TTF_W8_DOT(c32, a, v2);
            }
            const __m256i acc[MT][NT] = { { c00, c01, c02 }, { c10, c11, c12 }, { c20, c21, c22 }, { c30, c31, c32 } };
            for (int i = 0; i < mm; ++i) {
                for (int j = 0; j < nn; ++j) {
#if TTF_W8_VNNI
                    const int32_t d = hsum8i(acc[i][j]) - 128 * w->rs[n + j];
#else
                    const int32_t d = hsum8i(acc[i][j]) - (A7 ? 64 * w->rs[n + j] : 0);
#endif
                    y[(size_t) (m + i) * ldy + n + j] = w->s[n + j] * sa[m + i] * (float) d;
                }
            }
        }
    }
#undef TTF_W8_DOT
}

void gemm(const W8 * w, int64_t K, int64_t n0, int64_t n1, const uint8_t * aq, const float * sa, int64_t M, float * y,
          size_t ldy) {
    if (g_a7) {
        gemm_t<true>(w, K, n0, n1, aq, sa, M, y, ldy);
    } else {
        gemm_t<false>(w, K, n0, n1, aq, sa, M, y, ldy);
    }
}

}  // namespace

// on / off: TTF_W8A8=0|1 when set, else what the runtime chose (ggml_ttf_w8a8_enable; speech models turn it on: the
// VoxCPM2 DiT A/B showed no measurable loss, 17 Nov 2026), else off
namespace {
std::atomic<int> g_on{ [] {
    const char * e = std::getenv("TTF_W8A8");
    return e ? std::atoi(e) : 0;
}() };
}  // namespace

extern "C" void ggml_ttf_w8a8_enable(int on);
extern "C" void ggml_ttf_w8a8_enable(int on) {
    if (!std::getenv("TTF_W8A8")) {
        g_on.store(on, std::memory_order_relaxed);
    }
}

extern "C" bool ggml_ttf_w8a8_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
extern "C" bool ggml_ttf_w8a8_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const int on = g_on.load(std::memory_order_relaxed);
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    if (!on || params->use_ref || w->type != GGML_TYPE_Q8_0 || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = w->ne[0], N = w->ne[1], M = x->ne[1] * x->ne[2] * x->ne[3];  // x [K, T, B]: B T rows, one weight
    if (K % 32 || M > 64 || w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(x) || !ggml_is_contiguous(w) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }
    // work buffer: codes [M, K] then scales [M]; the plan reserves the Q8_0 copy of x (M K 34/32 bytes), enough for K >= 64
    if (params->wsize < (size_t) (M * K + 4 * M + 64)) {
        return false;
    }
    const int ith = params->ith, nth = params->nth;
    // the weight in W8: allocated by thread 0; on the first call every thread converts the rows it multiplies (the
    // splits of all threads cover [0, N), so the next call, after at least one barrier, finds every row converted)
    W8 * c = nullptr;
    if (ith == 0) {
        std::lock_guard<std::mutex> lk(g_mu);
        auto                        it = g_cache.find(w->data);
        if (it == g_cache.end() || it->second->K != K || it->second->N != N) {  // a freed weight's address reused
            if (it != g_cache.end()) {
                delete it->second;
                g_cache.erase(it);
            }
            g_cache.emplace(w->data, alloc(w));
        } else if (it->second->fresh && it->second->conv_call) {
            it->second->fresh = false;  // converted in an earlier call
        }
    }
    uint8_t * aq = (uint8_t *) params->wdata;
    float *   sa = (float *) (aq + GGML_PAD(M * K, 32));
    for (int64_t m = ith; m < M; m += nth) {  // activations per row
        quant_row((const float *) x->data + m * K, K, aq + (size_t) m * K, sa + m);
    }
    ggml_barrier(params->threadpool);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        c = g_cache.at(w->data);
    }
    const int64_t per = ((N + nth - 1) / nth + 2) / 3 * 3;
    const int64_t n0 = std::min<int64_t>(N, ith * per), n1 = std::min<int64_t>(N, n0 + per);
    if (c->fresh && n0 < n1) {
        convert_rows(w, c, n0, n1);
    }
    if (n0 < n1) {
        gemm(c, K, n0, n1, aq, sa, M, (float *) dst->data, (size_t) N);
    }
    if (ith == 0) {
        c->conv_call = true;  // read by thread 0 only, at its next call
    }
    return true;
}

#else

extern "C" bool ggml_ttf_w8a8_mul_mat(const struct ggml_compute_params *, struct ggml_tensor *) {
    return false;
}

extern "C" void ggml_ttf_w8a8_enable(int) {}

#endif
