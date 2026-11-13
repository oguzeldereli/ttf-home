// GGML_OP_TTF_QUANT / GGML_OP_TTF_MM on CUDA: the TTF linear as the trainer computes it (TTF_FORMAT_IR_SPEC.md,
// Arithmetic steps 1-8), bit-identical to the CPU op (ggml-cpu/ttf-ops.cpp):
//  - every rounding is an explicit _rn intrinsic (the CUDA build uses -use_fast_math; nothing may contract or approximate);
//  - the legacy adapter's float dots follow the CPU's order: 8 lanes, lane l accumulating k = l mod 8 by fma in increasing
//    k, the lanes summed in order, then the tail (ttf_kern::dot_f32 with AVX2 + FMA);
//  - the trit product is exact in int32 (dp4a), so its order does not matter.
// Weights: TQ1_P (decoded by a 256-entry shared table, as mmvq-tq1p.cuh) or Q2P (2-bit planes).

#include "ttf.cuh"

#include <cuda_bf16.h>

#define TTF_QROW_TAIL 32
#define TTF_MM_WARPS  4   // rows per block, one warp each
#define TTF_MM_TOK    4   // tokens per warp pass

static __device__ __forceinline__ float ttf_rbf(const float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
}

static __device__ __forceinline__ float ttf_rd(const float v, const bool bf16) {
    return bf16 ? ttf_rbf(v) : v;
}

// xc_k as the CPU computes it (steps 0-2)
static __device__ __forceinline__ float ttf_xc(const float * x, const float * mu, const float * c, const int64_t k,
                                               const bool bf16) {
    float t = ttf_rd(x[k], bf16);
    if (mu) {
        t = ttf_rd(__fsub_rn(t, mu[k]), bf16);
    }
    if (c) {
        t = ttf_rd(__fmul_rn(t, c[k]), bf16);
    }
    return t;
}

// dot_f32's order for n floats a . b (a, b contiguous): 8 lanes by fma, lanes summed in order, then the tail
static __device__ __forceinline__ float ttf_dot8(const float * a, const float * b, const int64_t n) {
    float         l[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    const int64_t n8   = n / 8 * 8;
    for (int64_t k = 0; k < n8; k += 8) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            l[i] = __fmaf_rn(a[k + i], b[k + i], l[i]);
        }
    }
    float r = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        r = __fadd_rn(r, l[i]);
    }
    for (int64_t k = n8; k < n; ++k) {
        r = __fadd_rn(r, __fmul_rn(a[k], b[k]));
    }
    return r;
}

static __device__ __forceinline__ float ttf_block_max(float v, float * sh) {
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    }
    const int w = threadIdx.x / WARP_SIZE, l = threadIdx.x % WARP_SIZE;
    if (l == 0) {
        sh[w] = v;
    }
    __syncthreads();
    if (w == 0) {
        v = l < (int) (blockDim.x / WARP_SIZE) ? sh[l] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        }
        if (l == 0) {
            sh[0] = v;
        }
    }
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}

static __device__ __forceinline__ int ttf_block_sum_i(int v, int * sh) {
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    const int w = threadIdx.x / WARP_SIZE, l = threadIdx.x % WARP_SIZE;
    if (l == 0) {
        sh[w] = v;
    }
    __syncthreads();
    if (w == 0) {
        v = l < (int) (blockDim.x / WARP_SIZE) ? sh[l] : 0;
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        if (l == 0) {
            sh[0] = v;
        }
    }
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}

// one block per row; 256 threads
static __global__ void ttf_quant_kernel(const char * x, const int64_t nbx1, const float * mu, const float * c,
                                        const float * mB, const int64_t K, const int64_t Kp, const int rank,
                                        const bool bf16, char * dst, const int64_t nbd1) {
    __shared__ float sf[32];
    __shared__ int   si[32];
    __shared__ float lanes[64 * 8];  // adapter dot lanes, rank <= 64
    const int64_t    row = blockIdx.x;
    const float *    xr  = (const float *) (x + row * nbx1);
    char *           o   = dst + row * nbd1;
    float            am  = 0.0f;
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        am = fmaxf(am, fabsf(ttf_xc(xr, mu, c, k, bf16)));
    }
    am             = ttf_block_max(am, sf);
    const float fl = ttf_rd(1e-5f, bf16);
    const float s  = __fmul_rn(fmaxf(am, fl), 1.0f / 127.0f);
    int         qs = 0;
    for (int64_t k = threadIdx.x; k < Kp; k += blockDim.x) {
        int q = 0;
        if (k < K) {
            const float t = rintf(__fdiv_rn(ttf_xc(xr, mu, c, k, bf16), s));
            q             = (int) fminf(127.0f, fmaxf(-128.0f, t));
        }
        o[k] = (char) q;
        qs += q;
    }
    qs = ttf_block_sum_i(qs, si);
    if (threadIdx.x == 0) {
        *(float *) (o + Kp)  = s;
        *(int *) (o + Kp + 4) = qs;
    }
    // adapter down projection u_j = rD(xc . mB_j) in dot_f32's order: thread (j, lane) runs lane l of row j
    if (rank > 0) {
        const int64_t n8 = K / 8 * 8;
        for (int e = threadIdx.x; e < rank * 8; e += blockDim.x) {
            const int     j  = e / 8, l = e % 8;
            const float * bj = mB + (int64_t) j * K;
            float         a  = 0.0f;
            for (int64_t k = l; k < n8; k += 8) {
                a = __fmaf_rn(ttf_xc(xr, mu, c, k, bf16), bj[k], a);
            }
            lanes[e] = a;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < rank; j += blockDim.x) {
            const float * bj = mB + (int64_t) j * K;
            float         r  = 0.0f;
            for (int l = 0; l < 8; ++l) {
                r = __fadd_rn(r, lanes[j * 8 + l]);
            }
            for (int64_t k = n8; k < K; ++k) {
                r = __fadd_rn(r, __fmul_rn(ttf_xc(xr, mu, c, k, bf16), bj[k]));
            }
            *(float *) (o + Kp + TTF_QROW_TAIL + 4 * j) = ttf_rd(r, bf16);
        }
    }
}

void ggml_cuda_op_ttf_quant(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * mu = dst->src[1];
    const ggml_tensor * c  = dst->src[2];
    const ggml_tensor * mB = dst->src[3];
    const int64_t       Kp   = ggml_get_op_params_i32(dst, 0);
    const bool          bf16 = ggml_get_op_params_i32(dst, 1) != 0;
    const int           rank = mB ? (int) mB->ne[1] : 0;
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_nrows(x) == x->ne[1] && ggml_is_contiguous(dst));  // 2-d input (the builder reshapes)
    GGML_ASSERT(rank <= 64);
    const int64_t rows = x->ne[1];
    if (rows == 0) {
        return;
    }
    ttf_quant_kernel<<<rows, 256, 0, ctx.stream()>>>(
        (const char *) x->data, x->nb[1], mu ? (const float *) mu->data : nullptr,
        c ? (const float *) c->data : nullptr, mB ? (const float *) mB->data : nullptr, x->ne[0], Kp, rank, bf16,
        (char *) dst->data, dst->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}

static __device__ __forceinline__ uint32_t ttf_tq1p_lut_entry(const uint32_t b) {
    uint32_t lo = 0;
    uint32_t p  = 1;
#pragma unroll
    for (int n = 0; n < 4; ++n, p *= 3) {
        lo |= ((((b * p) & 0xFF) * 3) >> 8) << (2 * n);
    }
    const uint32_t hi = ((((b * 81) & 0xFF) * 3) >> 8);
    return lo | (hi << 8);
}

// grid (ceil(N / WARPS), ceil(T / TOK)); block WARPS warps, one row each, TOK tokens per pass
template <bool Q2>
static __global__ void ttf_mm_kernel(const uint8_t * w, const int64_t nbw1, const char * qa, const int64_t ldq,
                                     const float * beta, const float * mA, const int rank, const float * r,
                                     const float * b, const float * acc, const bool bf16, float * out, const int64_t N,
                                     const int64_t T, const int64_t Kp) {
    __shared__ uint32_t lut[256];
    if (!Q2) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            lut[i] = ttf_tq1p_lut_entry(i);
        }
        __syncthreads();
    }
    const int     lane = threadIdx.x % WARP_SIZE;
    const int64_t n    = (int64_t) blockIdx.x * TTF_MM_WARPS + threadIdx.x / WARP_SIZE;
    const int64_t t0   = (int64_t) blockIdx.y * TTF_MM_TOK;
    if (n >= N) {
        return;
    }
    const int      nt = (int) min((int64_t) TTF_MM_TOK, T - t0);
    const int *    wr = (const int *) (w + n * nbw1);
    int            a[TTF_MM_TOK] = { 0, 0, 0, 0 };
    if (Q2) {
        // Q2P: block = 8 ints; lane handles int m = lane % 8 of block 4 it + lane / 8: elements 32 j + 4 m .. + 3
        const int64_t nb = Kp / 128;
        for (int64_t bi = lane / 8; bi < nb; bi += 4) {
            const int m = lane % 8;
            const int v = wr[bi * 8 + m];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int cj = (v >> (2 * j)) & 0x03030303;
#pragma unroll
                for (int tt = 0; tt < TTF_MM_TOK; ++tt) {
                    if (tt < nt) {
                        const int q4 = *(const int *) (qa + (t0 + tt) * ldq + 128 * bi + 32 * j + 4 * m);
                        a[tt]        = __dp4a(cj, q4, a[tt]);
                    }
                }
            }
        }
    } else {
        const int64_t nb = Kp / 160;
        for (int64_t bi = lane / 8; bi < nb; bi += 4) {
            const int      m  = lane % 8;
            const int      v  = wr[bi * 8 + m];
            const uint32_t e0 = lut[(v) & 0xFF], e1 = lut[(v >> 8) & 0xFF], e2 = lut[(v >> 16) & 0xFF],
                           e3 = lut[(uint32_t) v >> 24];
            const uint32_t p01 = __byte_perm(e0, e1, 0x5140);
            const uint32_t p23 = __byte_perm(e2, e3, 0x5140);
            const int      wa  = (int) __byte_perm(p01, p23, 0x5410);
            const int      wb  = (int) __byte_perm(p01, p23, 0x7632);
#pragma unroll
            for (int j = 0; j < 5; ++j) {
                const int cj = j < 4 ? (wa >> (2 * j)) & 0x03030303 : wb & 0x03030303;
#pragma unroll
                for (int tt = 0; tt < TTF_MM_TOK; ++tt) {
                    if (tt < nt) {
                        const int q4 = *(const int *) (qa + (t0 + tt) * ldq + 160 * bi + 32 * j + 4 * m);
                        a[tt]        = __dp4a(cj, q4, a[tt]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int tt = 0; tt < TTF_MM_TOK; ++tt) {
        for (int o = 16; o > 0; o >>= 1) {
            a[tt] += __shfl_xor_sync(0xffffffff, a[tt], o);
        }
    }
    if (lane >= nt) {
        return;
    }
    // lane tt finishes token t0 + tt
    int at = a[0];
#pragma unroll
    for (int tt = 1; tt < TTF_MM_TOK; ++tt) {
        if (lane == tt) {
            at = a[tt];
        }
    }
    const int64_t t   = t0 + lane;
    const char *  row = qa + t * ldq;
    const float   s   = *(const float *) (row + Kp);
    const int     qs  = *(const int *) (row + Kp + 4);
    float         y   = ttf_rbf(__fmul_rn(__fmul_rn(s, (float) (at - qs)), beta[0]));
    y                 = ttf_rd(y, bf16);
    if (mA) {
        const float * u = (const float *) (row + Kp + TTF_QROW_TAIL);
        y               = ttf_rd(__fadd_rn(y, ttf_rd(ttf_dot8(u, mA + n * rank, rank), bf16)), bf16);
    }
    if (r) {
        y = ttf_rd(__fmul_rn(y, ttf_rd(r[n], bf16)), bf16);
    }
    if (b) {
        y = ttf_rd(__fadd_rn(y, ttf_rd(b[n], bf16)), bf16);
    }
    const int64_t i = t * N + n;
    if (acc) {
        y = ttf_rd(__fadd_rn(acc[i], y), bf16);
    }
    out[i] = y;
}

void ggml_cuda_op_ttf_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * qa  = dst->src[1];
    const ggml_tensor * be  = dst->src[2];
    const ggml_tensor * mA  = dst->src[3];
    const ggml_tensor * r   = dst->src[4];
    const ggml_tensor * b   = dst->src[5];
    const ggml_tensor * acc = dst->src[6];
    const bool          bf16 = ggml_get_op_params_i32(dst, 0) != 0;
    const bool          q2   = w->type == GGML_TYPE_I8;
    GGML_ASSERT(ggml_is_contiguous(qa) && ggml_is_contiguous(dst) && (!acc || ggml_is_contiguous(acc)));
    const int64_t Kp   = q2 ? 4 * w->ne[0] : w->ne[0];
    const int64_t N    = w->ne[1];
    const int64_t T    = ggml_nrows(qa);
    const int     rank = mA ? (int) mA->ne[0] : 0;
    if (T == 0) {
        return;
    }
    const dim3 grid((unsigned) ((N + TTF_MM_WARPS - 1) / TTF_MM_WARPS), (unsigned) ((T + TTF_MM_TOK - 1) / TTF_MM_TOK));
    const dim3 block(TTF_MM_WARPS * WARP_SIZE);
    auto       go = [&](auto kern) {
        kern<<<grid, block, 0, ctx.stream()>>>(
            (const uint8_t *) w->data, w->nb[1], (const char *) qa->data, qa->nb[1], (const float *) be->data,
            mA ? (const float *) mA->data : nullptr, rank, r ? (const float *) r->data : nullptr,
            b ? (const float *) b->data : nullptr, acc ? (const float *) acc->data : nullptr, bf16,
            (float *) dst->data, N, T, Kp);
    };
    if (q2) {
        go(ttf_mm_kernel<true>);
    } else {
        go(ttf_mm_kernel<false>);
    }
    CUDA_CHECK(cudaGetLastError());
}
