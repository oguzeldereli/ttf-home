// GGML_OP_TTF_QUANT / GGML_OP_TTF_MM on the CPU: the TTF linear as the trainer computes it
// (TTF_FORMAT_IR_SPEC.md, Arithmetic steps 1-8; kernels in ttf-kern.h, verified bit-exact against the trainer's parts).

#include "ops.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ttf-kern.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t QROW_TAIL = 32;  // per quantised row: Kp codes, then f32 s at Kp, int32 qsum at Kp + 4

inline float rd(float v, bool bf16) {
    return bf16 ? ttf_kern::bf16_round(v) : v;
}

}  // namespace

void ggml_compute_forward_ttf_quant(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * mu = dst->src[1];
    const ggml_tensor * c  = dst->src[2];
    const ggml_tensor * mB = dst->src[3];
    const int64_t       Kp   = ggml_get_op_params_i32(dst, 0);
    const bool          bf16 = ggml_get_op_params_i32(dst, 1) != 0;
    const int64_t       K    = x->ne[0];
    const int64_t       rank = mB ? mB->ne[1] : 0;
    const float *       mBv  = mB ? (const float *) mB->data : nullptr;
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I8 && dst->ne[0] == Kp + QROW_TAIL + 4 * rank);
    GGML_ASSERT(x->nb[0] == sizeof(float));
    const float floor_ = rd(1e-5f, bf16);
    const float * muv  = mu ? (const float *) mu->data : nullptr;
    const float * cv   = c ? (const float *) c->data : nullptr;
    thread_local std::vector<float> v;
    v.resize(Kp);
    const int64_t      nr = x->ne[1] * x->ne[2] * x->ne[3];
    for (int64_t ir = params->ith; ir < nr; ir += params->nth) {
        const int64_t i3 = ir / (x->ne[1] * x->ne[2]);
        const int64_t i2 = (ir / x->ne[1]) % x->ne[2];
        const int64_t i1 = ir % x->ne[1];
        const float * xr = (const float *) ((const char *) x->data + i1 * x->nb[1] + i2 * x->nb[2] + i3 * x->nb[3]);
        char *        o  = (char *) dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3];
        int32_t       qs;
        const float   s = ttf_kern::quant_row(xr, K, Kp, muv, cv, nullptr, bf16, floor_, v.data(), (int8_t *) o, &qs);
        std::memcpy(o + Kp, &s, 4);
        std::memcpy(o + Kp + 4, &qs, 4);
        // step 6, first half: u = rD(xc @ mag_B), xc = v (in D)
        for (int64_t j = 0; j < rank; ++j) {
            const float u = rd(ttf_kern::dot_f32_64(v.data(), mBv + j * K, K), bf16);
            std::memcpy(o + Kp + QROW_TAIL + 4 * j, &u, 4);
        }
    }
}

static constexpr int64_t CHMAX = 256;

void ggml_compute_forward_ttf_mm(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * qa  = dst->src[1];
    const ggml_tensor * be  = dst->src[2];
    const ggml_tensor * mA  = dst->src[3];
    const ggml_tensor * r   = dst->src[4];
    const ggml_tensor * b   = dst->src[5];
    const ggml_tensor * acc = dst->src[6];
    const bool          bf16 = ggml_get_op_params_i32(dst, 0) != 0;
    const bool q2 = w->type == GGML_TYPE_I8;  // Q2P layout
    GGML_ASSERT((q2 || w->type == GGML_TYPE_TQ1_P) && qa->type == GGML_TYPE_I8 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(qa) && ggml_is_contiguous(dst));
    GGML_ASSERT(!acc || ggml_is_contiguous(acc));
    const int64_t Kp  = q2 ? 4 * w->ne[0] : w->ne[0];
    const int64_t N   = w->ne[1];
    const int64_t T   = ggml_nrows(qa);
    const int64_t ldq = qa->nb[1];
    const int64_t rank = mA ? mA->ne[0] : 0;
    GGML_ASSERT(Kp % (q2 ? 128 : ttf_kern::QK_TQ1P) == 0 && qa->ne[0] == Kp + QROW_TAIL + 4 * rank && dst->ne[0] == N);
    const float   bt  = ((const float *) be->data)[0];
    const float * rv  = r ? (const float *) r->data : nullptr;
    const float * bv  = b ? (const float *) b->data : nullptr;
    const float * mAv = mA ? (const float *) mA->data : nullptr;
    const float * av  = acc ? (const float *) acc->data : nullptr;
    float *       out = (float *) dst->data;

    std::vector<float>   s(T);
    std::vector<int32_t> qsum(T);
    for (int64_t t = 0; t < T; ++t) {
        const char * row = (const char *) qa->data + t * ldq;
        std::memcpy(&s[t], row + Kp, 4);
        std::memcpy(&qsum[t], row + Kp + 4, 4);
    }
    const int8_t *       q = (const int8_t *) qa->data;
    std::vector<uint8_t> codes(T > 4 && !q2 ? Kp : 0);  // TQ1_P prefill: a row decoded once, reused over the tokens
    // rows in chunks handed out dynamically (a shared counter, as ggml's mul_mat): on a hybrid CPU the slower cores
    // take fewer chunks instead of holding up the barrier. ~16 chunks per thread, at least 8 rows: chunks of a few
    // KB restart the hardware prefetcher too often (BitNet 2B decode, 8 threads: +4%)
    const int64_t CH = std::min<int64_t>(CHMAX, std::max<int64_t>(8, N / (16 * params->nth) / 8 * 8));
    const int64_t nc = (N + CH - 1) / CH;
#ifdef GGML_USE_OPENMP
    if (params->ith == 0) {
        ggml_threadpool_chunk_set(params->threadpool, params->nth);
    }
    ggml_barrier(params->threadpool);
#endif  // else ggml_barrier left the counter at nth when this node started
    for (int64_t ci = params->ith; ci < nc; ci = ggml_threadpool_chunk_add(params->threadpool, 1)) {
        int32_t       a4[CHMAX];  // decode, Q2P: the chunk's full groups of 4 rows in one pass each
        const int64_t n0 = ci * CH, n1 = std::min(N, (ci + 1) * CH);
        const int64_t n4 = (T == 1 && q2) ? n0 + (n1 - n0) / 4 * 4 : n0;
        for (int64_t n = n0; n < n4; n += 4) {
            ttf_kern::dot_q2p_r4((const uint8_t *) w->data + n * w->nb[1], w->nb[1], Kp / 128, q, a4 + (n - n0));
        }
        for (int64_t n = n0; n < n1; ++n) {
            const uint8_t * wr = (const uint8_t *) w->data + n * w->nb[1];
            if (T > 4 && !q2) {
                ttf_kern::tq1p_decode_row(wr, Kp / ttf_kern::QK_TQ1P, codes.data());
            }
            const float rn = rv ? rd(rv[n], bf16) : 0.0f;
            const float bn = bv ? rd(bv[n], bf16) : 0.0f;
            for (int64_t t = 0; t < T; t += 4) {
                const int nt = (int) std::min<int64_t>(4, T - t);
                int32_t   a[4];
                if (n < n4) {
                    a[0] = a4[n - n0];
                } else if (q2) {
                    ttf_kern::dot_q2p4(wr, Kp / 128, q + t * ldq, ldq, nt, a);
                } else if (T > 4) {
                    ttf_kern::dot_codes4(codes.data(), q + t * ldq, ldq, nt, Kp, a);
                } else {
                    ttf_kern::dot_tq1p4(wr, Kp / ttf_kern::QK_TQ1P, q + t * ldq, ldq, nt, a);
                }
                for (int j = 0; j < nt; ++j) {
                    const int64_t i = (t + j) * N + n;
                    // step 5: the trainer's int8 GEMM returns bf16 for any D
                    float y = ttf_kern::bf16_round((s[t + j] * (float) (a[j] - qsum[t + j])) * bt);
                    y       = rd(y, bf16);
                    if (mAv) {  // step 6, second half: y = rD(y + rD(u @ mag_A^T))
                        const float * u = (const float *) (q + (t + j) * ldq + Kp + QROW_TAIL);
                        y = rd(y + rd(ttf_kern::dot_f32(u, mAv + n * rank, rank), bf16), bf16);
                    }
                    if (rv) {  // step 7
                        y = rd(y * rn, bf16);
                    }
                    if (bv) {  // step 8
                        y = rd(y + bn, bf16);
                    }
                    if (av) {  // IR acc / QTad sum
                        y = rd(av[i] + y, bf16);
                    }
                    out[i] = y;
                }
            }
        }
    }
}
