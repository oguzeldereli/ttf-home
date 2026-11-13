#pragma once

// CPU kernels of the exact TTF product on TQ1_P weights (no ggml dependency; ttf-ops.cpp wraps them as ggml ops).
//   Q(v)   the per-token int8 quantiser a TTF model is trained with (ttf.kernel act_quant_i8 / _pro_kernel), on the
//          scaled and centred input v = c x + m (zero past K): s = max(max|v|, 1e-5) * fp32(1 / 127),
//          q = clamp(rint(v / s), -128, 127) (round half to even, as torch). An fp32 model quantises v as it is (the
//          default); bf16 = true first rounds v to bf16, as a bf16 model's activations are.
//   TQ1_P  160 trits in 32 bytes, no scale: element 32 n + b of a block is digit n (n = 0 most significant) of byte b,
//          digit n of byte v = ((uint8_t) (v * 3^n) * 3) >> 8 in {0, 1, 2} = trit + 1
// One row of W is decoded once to its codes c in {0, 1, 2} (in element order: digit plane n of block i is elements
// 160 i + 32 n .. + 31), then for every token sum_k (c_k - 1) q_k = sum_k c_k q_k - sum_k q_k in int32
// (|.| <= 2 * 128 * K: exact for K < 2^23), four tokens per pass over the codes.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ttf_kern {

constexpr int QK_TQ1P = 160;

inline float bf16_round(float v) {
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

#if defined(__AVX2__)
inline __m256 bf16_round8(__m256 v) {  // round to nearest even, as bf16_round (inputs finite)
    const __m256i u   = _mm256_castps_si256(v);
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    const __m256i r   = _mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7fff)), lsb);
    return _mm256_castsi256_ps(_mm256_and_si256(r, _mm256_set1_epi32((int) 0xffff0000u)));
}
#endif

// one token, the spec's steps 1-3 (TTF_FORMAT_IR_SPEC Arithmetic): v_k = rD(rD(x_k - mu_k) * c_k) + m_k (k < K; mu / c / m
// optional; rD = bf16 rounding when bf16, else none; mu / c already in D), v_k = 0 for K <= k < Kp; q [Kp] int8 codes,
// *qsum = sum_k q_k; returns the scale s = max(amax, floor) * fp32(1 / 127). v [Kp] is scratch.
// (m: the older side-tensor form m = -c mu, added after c; the spec's form is mu.)
inline float quant_row(const float * x, int64_t K, int64_t Kp, const float * mu, const float * c, const float * m,
                       bool bf16, float floor, float * v, int8_t * q, int32_t * qsum) {
    int64_t k0 = 0;
#if defined(__AVX2__)
    if (!m) {
        for (; k0 + 8 <= K; k0 += 8) {
            __m256 t = _mm256_loadu_ps(x + k0);
            if (bf16) {
                t = bf16_round8(t);
            }
            if (mu) {
                t = _mm256_sub_ps(t, _mm256_loadu_ps(mu + k0));
                if (bf16) {
                    t = bf16_round8(t);
                }
            }
            if (c) {
                t = _mm256_mul_ps(t, _mm256_loadu_ps(c + k0));
                if (bf16) {
                    t = bf16_round8(t);
                }
            }
            _mm256_storeu_ps(v + k0, t);
        }
    }
#endif
    for (int64_t k = k0; k < K; ++k) {
        float t = bf16 ? bf16_round(x[k]) : x[k];  // the input in D at node entry
        if (mu) {
            t = t - mu[k];
            if (bf16) {
                t = bf16_round(t);
            }
        }
        if (c) {
            t = t * c[k];
            if (bf16) {
                t = bf16_round(t);
            }
        }
        if (m) {
            t = t + m[k];
            if (bf16) {
                t = bf16_round(t);
            }
        }
        v[k] = t;
    }
    for (int64_t k = K; k < Kp; ++k) {
        v[k] = 0.0f;
    }
    float   amax = 0.0f;
    int64_t k    = 0;
#if defined(__AVX2__)
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256       am   = _mm256_setzero_ps();
    for (; k + 8 <= Kp; k += 8) {
        am = _mm256_max_ps(am, _mm256_andnot_ps(sign, _mm256_loadu_ps(v + k)));
    }
    float a8[8];
    _mm256_storeu_ps(a8, am);
    for (float a : a8) {
        amax = std::max(amax, a);
    }
#endif
    for (; k < Kp; ++k) {
        amax = std::max(amax, std::fabs(v[k]));
    }
    const float s   = std::max(amax, floor) * (1.0f / 127.0f);
    int32_t     sum = 0;
    k               = 0;
#if defined(__AVX2__)
    const __m256  vs  = _mm256_set1_ps(s);
    const __m256  lo  = _mm256_set1_ps(-128.0f);
    const __m256  hi  = _mm256_set1_ps(127.0f);
    __m256i       acc = _mm256_setzero_si256();
    for (; k + 32 <= Kp; k += 32) {
        __m256i i32[4];
        for (int j = 0; j < 4; ++j) {
            __m256 t = _mm256_div_ps(_mm256_loadu_ps(v + k + 8 * j), vs);
            t        = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            t        = _mm256_min_ps(_mm256_max_ps(t, lo), hi);
            i32[j]   = _mm256_cvtps_epi32(t);
            acc      = _mm256_add_epi32(acc, i32[j]);
        }
        // int32 x 32 -> int8 x 32 in order (packs work per 128-bit lane: fix the order with a permute)
        const __m256i p16 = _mm256_packs_epi32(i32[0], i32[1]);  // lanes: 0lo 1lo | 0hi 1hi
        const __m256i q16 = _mm256_packs_epi32(i32[2], i32[3]);
        const __m256i p8  = _mm256_packs_epi16(p16, q16);         // per lane: 0 1 2 3 (4 each)
        const __m256i o   = _mm256_permutevar8x32_epi32(p8, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i *) (q + k), o);
    }
    int32_t a32[8];
    _mm256_storeu_si256((__m256i *) a32, acc);
    for (int32_t a : a32) {
        sum += a;
    }
#endif
    for (; k < Kp; ++k) {
        const float t = std::nearbyint(v[k] / s);
        q[k]          = (int8_t) std::min(127.0f, std::max(-128.0f, t));
        sum += q[k];
    }
    *qsum = sum;
    return s;
}

inline uint8_t tq1p_digit(uint8_t v, int n) {
    static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
    const uint8_t        p       = (uint8_t) (v * pow3[n]);
    return (uint8_t) (((uint16_t) p * 3) >> 8);
}

// nb TQ1_P blocks (32 bytes each) -> 160 nb codes in {0, 1, 2}
inline void tq1p_decode_row(const uint8_t * qs, int64_t nb, uint8_t * codes) {
#if defined(__AVX2__)
    const __m256i m8   = _mm256_set1_epi8((char) 0xF8);
    const __m256i m3   = _mm256_set1_epi8(3);
    const __m256i one8 = _mm256_set1_epi8(1);
    for (int64_t i = 0; i < nb; ++i) {
        __m256i p[5];
        p[0] = _mm256_loadu_si256((const __m256i *) (qs + 32 * i));
        p[1] = _mm256_add_epi8(p[0], _mm256_add_epi8(p[0], p[0]));                          // 3 v
        p[2] = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(p[0], 3), m8), p[0]);     // 9 v
        p[3] = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(p[1], 3), m8), p[1]);     // 27 v
        p[4] = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(p[2], 3), m8), p[2]);     // 81 v
        for (int n = 0; n < 5; ++n) {
            // (p * 3) >> 8 by halving adds (ggml's tq1_0 decode)
            __m256i t = _mm256_subs_epu8(p[n], one8);
            t         = _mm256_avg_epu8(t, _mm256_avg_epu8(t, _mm256_setzero_si256()));
            t         = _mm256_and_si256(_mm256_srli_epi16(t, 6), m3);
            _mm256_storeu_si256((__m256i *) (codes + QK_TQ1P * i + 32 * n), t);
        }
    }
#else
    for (int64_t i = 0; i < nb; ++i) {
        for (int n = 0; n < 5; ++n) {
            for (int b = 0; b < 32; ++b) {
                codes[QK_TQ1P * i + 32 * n + b] = tq1p_digit(qs[32 * i + b], n);
            }
        }
    }
#endif
}

// out[j] = sum_k c_k q_j[k] for nt <= 4 token rows q_j = q + j * ldq, K % 32 == 0
inline void dot_codes4(const uint8_t * c, const int8_t * q, int64_t ldq, int nt, int64_t K, int32_t * out) {
#if defined(__AVX2__)
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i       acc[4] = { _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                             _mm256_setzero_si256() };
    // u8 x s8 -> s16 pairs, |.| <= 2 * 2 * 128: no saturation
#define TTF_DOT_STEP(J)                                                                                    \
    acc[J] = _mm256_add_epi32(                                                                             \
        acc[J], _mm256_madd_epi16(                                                                         \
                    _mm256_maddubs_epi16(vc, _mm256_loadu_si256((const __m256i *) (q + (J) * ldq + k))), ones))
    if (nt == 4) {
        for (int64_t k = 0; k < K; k += 32) {
            const __m256i vc = _mm256_loadu_si256((const __m256i *) (c + k));
            TTF_DOT_STEP(0);
            TTF_DOT_STEP(1);
            TTF_DOT_STEP(2);
            TTF_DOT_STEP(3);
        }
    } else {
        for (int64_t k = 0; k < K; k += 32) {
            const __m256i vc = _mm256_loadu_si256((const __m256i *) (c + k));
            for (int j = 0; j < nt; ++j) {
                TTF_DOT_STEP(j);
            }
        }
    }
#undef TTF_DOT_STEP
    // four 8-lane sums at once: hadd pairs, then the two 128-bit halves
    const __m256i h01 = _mm256_hadd_epi32(acc[0], acc[1]);
    const __m256i h23 = _mm256_hadd_epi32(acc[2], acc[3]);
    const __m256i h   = _mm256_hadd_epi32(h01, h23);  // per lane: s0 s1 s2 s3 (partial)
    const __m128i s   = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
    int32_t       r[4];
    _mm_storeu_si128((__m128i *) r, s);
    for (int j = 0; j < nt; ++j) {
        out[j] = r[j];
    }
#else
    for (int j = 0; j < nt; ++j) {
        int32_t acc = 0;
        for (int64_t k = 0; k < K; ++k) {
            acc += (int32_t) c[k] * q[j * ldq + k];
        }
        out[j] = acc;
    }
#endif
}

// out[j] = sum_k c_k q_j[k] straight from nb TQ1_P blocks (digits decoded in registers, no code buffer), nt <= 4
// token rows q_j = q + j * ldq; for decode (few tokens), where decoding a row to codes would cost more than the dot
inline void dot_tq1p4(const uint8_t * qs, int64_t nb, const int8_t * q, int64_t ldq, int nt, int32_t * out) {
#if defined(__AVX2__)
    const __m256i m8   = _mm256_set1_epi8((char) 0xF8);
    const __m256i m3   = _mm256_set1_epi8(3);
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i       acc[4] = { _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                             _mm256_setzero_si256() };
    for (int64_t i = 0; i < nb; ++i) {
        __m256i p[5];
        p[0] = _mm256_loadu_si256((const __m256i *) (qs + 32 * i));
        p[1] = _mm256_add_epi8(p[0], _mm256_add_epi8(p[0], p[0]));
        p[2] = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(p[0], 3), m8), p[0]);
        p[3] = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(p[1], 3), m8), p[1]);
        p[4] = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(p[2], 3), m8), p[2]);
        for (int n = 0; n < 5; ++n) {
            __m256i c = _mm256_subs_epu8(p[n], one8);
            c         = _mm256_avg_epu8(c, _mm256_avg_epu8(c, _mm256_setzero_si256()));
            c         = _mm256_and_si256(_mm256_srli_epi16(c, 6), m3);
            const int64_t k = QK_TQ1P * i + 32 * n;
            for (int j = 0; j < nt; ++j) {
                const __m256i qv = _mm256_loadu_si256((const __m256i *) (q + j * ldq + k));
                acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(_mm256_maddubs_epi16(c, qv), ones));
            }
        }
    }
    const __m256i h01 = _mm256_hadd_epi32(acc[0], acc[1]);
    const __m256i h23 = _mm256_hadd_epi32(acc[2], acc[3]);
    const __m256i h   = _mm256_hadd_epi32(h01, h23);
    const __m128i s   = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
    int32_t       r[4];
    _mm_storeu_si128((__m128i *) r, s);
    for (int j = 0; j < nt; ++j) {
        out[j] = r[j];
    }
#else
    for (int j = 0; j < nt; ++j) {
        int32_t a = 0;
        for (int64_t i = 0; i < nb; ++i) {
            for (int n = 0; n < 5; ++n) {
                for (int b = 0; b < 32; ++b) {
                    a += (int32_t) tq1p_digit(qs[32 * i + b], n) * q[j * ldq + QK_TQ1P * i + 32 * n + b];
                }
            }
        }
        out[j] = a;
    }
#endif
}

// Q2P: 128 trits in 32 bytes, element 32 n + b of a block = bits 2n, 2n+1 of byte b (code = trit + 1, in {0, 1, 2}).
// The CPU runtime layout: 2.5x faster to decode than TQ1_P at 1.25x the bytes.
// out[j] = sum_k c_k q_j[k] for nt <= 4 token rows q_j = q + j * ldq; nb blocks
inline void dot_q2p4(const uint8_t * qs, int64_t nb, const int8_t * q, int64_t ldq, int nt, int32_t * out) {
#if defined(__AVX2__)
    const __m256i m3   = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i       acc[4] = { _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                             _mm256_setzero_si256() };
    for (int64_t i = 0; i < nb; ++i) {
        const __m256i p  = _mm256_loadu_si256((const __m256i *) (qs + 32 * i));
        const __m256i c0 = _mm256_and_si256(p, m3);
        const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(p, 2), m3);
        const __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(p, 4), m3);
        const __m256i c3 = _mm256_and_si256(_mm256_srli_epi16(p, 6), m3);
        for (int j = 0; j < nt; ++j) {
            const int8_t * qq = q + j * ldq + 128 * i;
            // u8 {0,1,2} x s8 pairs: |.| <= 2 * 2 * 128 per lane, two summed <= 1024: no int16 overflow
            const __m256i s0 = _mm256_add_epi16(_mm256_maddubs_epi16(c0, _mm256_loadu_si256((const __m256i *) qq)),
                                                _mm256_maddubs_epi16(c1, _mm256_loadu_si256((const __m256i *) (qq + 32))));
            const __m256i s1 = _mm256_add_epi16(_mm256_maddubs_epi16(c2, _mm256_loadu_si256((const __m256i *) (qq + 64))),
                                                _mm256_maddubs_epi16(c3, _mm256_loadu_si256((const __m256i *) (qq + 96))));
            acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(_mm256_add_epi16(s0, s1), ones));
        }
    }
    const __m256i h01 = _mm256_hadd_epi32(acc[0], acc[1]);
    const __m256i h23 = _mm256_hadd_epi32(acc[2], acc[3]);
    const __m256i h   = _mm256_hadd_epi32(h01, h23);
    const __m128i s   = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
    int32_t       r[4];
    _mm_storeu_si128((__m128i *) r, s);
    for (int j = 0; j < nt; ++j) {
        out[j] = r[j];
    }
#else
    for (int j = 0; j < nt; ++j) {
        int32_t a = 0;
        for (int64_t i = 0; i < nb; ++i) {
            for (int n = 0; n < 4; ++n) {
                for (int b = 0; b < 32; ++b) {
                    a += (int32_t) ((qs[32 * i + b] >> (2 * n)) & 3) * q[j * ldq + 128 * i + 32 * n + b];
                }
            }
        }
        out[j] = a;
    }
#endif
}

// fp32 dot of n floats (the legacy adapter's rank-r projections), sequential in 8 lanes then the lanes in order
inline float dot_f32(const float * a, const float * b, int64_t n) {
    int64_t k = 0;
    float   r = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps();
    for (; k + 8 <= n; k += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
    }
    float l[8];
    _mm256_storeu_ps(l, acc);
    for (float v : l) {
        r += v;
    }
#endif
    for (; k < n; ++k) {
        r += a[k] * b[k];
    }
    return r;
}

// fp32 dot of n floats in 64 lanes (the legacy adapter's K-length down projection; one CUDA warp per dot):
// lane j accumulates k = j mod 64 by fma in increasing k; then l[i] += l[i + 32], then the halves 16, 8, 4, 2, 1
inline float dot_f32_64(const float * a, const float * b, int64_t n) {
    alignas(32) float l[64] = {};
    int64_t           k     = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc[8];
    for (int i = 0; i < 8; ++i) {
        acc[i] = _mm256_setzero_ps();
    }
    for (; k + 64 <= n; k += 64) {
        for (int i = 0; i < 8; ++i) {
            acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8 * i), _mm256_loadu_ps(b + k + 8 * i), acc[i]);
        }
    }
    for (int i = 0; i < 8; ++i) {
        _mm256_store_ps(l + 8 * i, acc[i]);
    }
#endif
    for (; k < n; k += 64) {
        for (int j = 0; j < 64 && k + j < n; ++j) {
            l[j] = std::fma(a[k + j], b[k + j], l[j]);
        }
    }
    for (int off = 32; off > 0; off >>= 1) {
        for (int i = 0; i < off; ++i) {
            l[i] = l[i] + l[i + off];
        }
    }
    return l[0];
}

}  // namespace ttf_kern
