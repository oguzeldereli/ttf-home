// GGML_OP_TTF_QUANT / GGML_OP_TTF_MM on CUDA: the TTF linear as the trainer computes it (TTF_FORMAT_IR_SPEC.md,
// Arithmetic steps 1-8), bit-identical to the CPU op (ggml-cpu/ttf-ops.cpp):
//  - every rounding is an explicit _rn intrinsic (the CUDA build uses -use_fast_math; nothing may contract or approximate);
//  - the legacy adapter's float dots follow the CPU's order: the K-length down projection in 64 lanes (lane j accumulates
//    k = j mod 64 by fma, then a fixed halving tree; ttf_kern::dot_f32_64, one warp per dot here), the rank-length up
//    projection in 8 lanes summed in order, then the tail (ttf_kern::dot_f32 with AVX2 + FMA);
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
// grid (rows, 1 + ceil(rank / 8)), 256 threads: blockIdx.y 0 makes the codes, s and qsum; blockIdx.y > 0 makes 8 of
// the adapter's down projections u_j = rD(xc . mB_j), one warp each, in dot_f32_64's order
static __global__ void ttf_quant_kernel(const char * x, const int64_t nbx1, const float * mu, const float * c,
                                        const float * mB, const int64_t K, const int64_t Kp, const int rank,
                                        const bool bf16, char * dst, const int64_t nbd1) {
    __shared__ float sf[32];
    __shared__ int   si[32];
    const int64_t    row = blockIdx.x;
    const float *    xr  = (const float *) (x + row * nbx1);
    char *           o   = dst + row * nbd1;
    if (blockIdx.y > 0) {
        const int lane = threadIdx.x % WARP_SIZE;
        const int j    = (blockIdx.y - 1) * 8 + threadIdx.x / WARP_SIZE;
        if (j >= rank) {
            return;
        }
        const float * bj = mB + (int64_t) j * K;
        float         a0 = 0.0f, a1 = 0.0f;
        const int64_t K64 = K / 64 * 64;
        int64_t       k0  = 0;
        // full blocks, 4 at a time: the loads of 4 blocks issue before the (ordered) fma chains
        for (; k0 + 256 <= K64; k0 += 256) {
            float xa[8], wa[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int64_t k = k0 + 32 * u + lane;
                xa[u]           = ttf_xc(xr, mu, c, k, bf16);
                wa[u]           = bj[k];
            }
#pragma unroll
            for (int u = 0; u < 8; u += 2) {
                a0 = __fmaf_rn(xa[u], wa[u], a0);
                a1 = __fmaf_rn(xa[u + 1], wa[u + 1], a1);
            }
        }
        for (; k0 < K; k0 += 64) {
            const int64_t k = k0 + lane;
            if (k < K) {
                a0 = __fmaf_rn(ttf_xc(xr, mu, c, k, bf16), bj[k], a0);
            }
            if (k + 32 < K) {
                a1 = __fmaf_rn(ttf_xc(xr, mu, c, k + 32, bf16), bj[k + 32], a1);
            }
        }
        float s = __fadd_rn(a0, a1);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s = __fadd_rn(s, __shfl_down_sync(0xffffffff, s, off));
        }
        if (lane == 0) {
            *(float *) (o + Kp + TTF_QROW_TAIL + 4 * j) = ttf_rd(s, bf16);
        }
        return;
    }
    float am = 0.0f;
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
    const int64_t rows = x->ne[1];
    if (rows == 0) {
        return;
    }
    const dim3 grid((unsigned) rows, 1 + (rank + 7) / 8);
    ttf_quant_kernel<<<grid, 256, 0, ctx.stream()>>>(
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

// grid (ceil(N / (WARPS R)), <= ceil(T / TOK)); block WARPS warps, R rows each (R = 4 when T > TOK: each activation load
// then feeds 4 rows), TOK tokens per pass; the block strides over its token chunks
template <bool Q2, int R>
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
    const int64_t n0   = ((int64_t) blockIdx.x * TTF_MM_WARPS + threadIdx.x / WARP_SIZE) * R;
    if (n0 >= N) {
        return;
    }
    const int nr = (int) min((int64_t) R, N - n0);
    const int m  = lane % 8;
    for (int64_t t0 = (int64_t) blockIdx.y * TTF_MM_TOK; t0 < T; t0 += (int64_t) gridDim.y * TTF_MM_TOK) {
        const int nt = (int) min((int64_t) TTF_MM_TOK, T - t0);
        int       a[R][TTF_MM_TOK];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
#pragma unroll
            for (int tt = 0; tt < TTF_MM_TOK; ++tt) {
                a[rr][tt] = 0;
            }
        }
        const int64_t QB = Q2 ? 128 : 160;
        const int64_t nb = Kp / QB;
        for (int64_t bi = lane / 8; bi < nb; bi += 4) {
            int wa[R], wb[R];
#pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                const int v = rr < nr ? ((const int *) (w + (n0 + rr) * nbw1))[bi * 8 + m] : 0;
                if (Q2) {
                    wa[rr] = v;
                    wb[rr] = 0;
                } else {
                    const uint32_t e0 = lut[(v) & 0xFF], e1 = lut[(v >> 8) & 0xFF], e2 = lut[(v >> 16) & 0xFF],
                                   e3 = lut[(uint32_t) v >> 24];
                    const uint32_t p01 = __byte_perm(e0, e1, 0x5140);
                    const uint32_t p23 = __byte_perm(e2, e3, 0x5140);
                    wa[rr]             = (int) __byte_perm(p01, p23, 0x5410);
                    wb[rr]             = (int) __byte_perm(p01, p23, 0x7632);
                }
            }
#pragma unroll
            for (int j = 0; j < (Q2 ? 4 : 5); ++j) {
#pragma unroll
                for (int tt = 0; tt < TTF_MM_TOK; ++tt) {
                    if (tt < nt) {
                        const int q4 = *(const int *) (qa + (t0 + tt) * ldq + QB * bi + 32 * j + 4 * m);
#pragma unroll
                        for (int rr = 0; rr < R; ++rr) {
                            const int cj = j < 4 ? (wa[rr] >> (2 * j)) & 0x03030303 : wb[rr] & 0x03030303;
                            a[rr][tt]    = __dp4a(cj, q4, a[rr][tt]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
#pragma unroll
            for (int tt = 0; tt < TTF_MM_TOK; ++tt) {
                for (int o = 16; o > 0; o >>= 1) {
                    a[rr][tt] += __shfl_xor_sync(0xffffffff, a[rr][tt], o);
                }
            }
        }
        // adapter up projection in ttf_dot8's order, spread over the warp: lane 8 tt + l runs lane l of token tt's dot;
        // lane tt then sums its 8 partials in order and adds the tail (per row; rr < nr is warp-uniform)
        float ad[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            ad[rr] = 0.0f;
            if (mA && rr < nr) {
                const int     tt = lane / 8, l = lane % 8;
                const int64_t n8 = rank / 8 * 8;
                const float * ar = mA + (n0 + rr) * rank;
                float         pl = 0.0f;
                if (tt < nt) {
                    const float * u = (const float *) (qa + (t0 + tt) * ldq + Kp + TTF_QROW_TAIL);
                    for (int64_t k = l; k < n8; k += 8) {
                        pl = __fmaf_rn(u[k], ar[k], pl);
                    }
                }
                float rs = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    rs = __fadd_rn(rs, __shfl_sync(0xffffffff, pl, (lane & 3) * 8 + i));
                }
                if (lane < nt) {
                    const float * u = (const float *) (qa + (t0 + lane) * ldq + Kp + TTF_QROW_TAIL);
                    for (int64_t k = n8; k < rank; ++k) {
                        rs = __fadd_rn(rs, __fmul_rn(u[k], ar[k]));
                    }
                }
                ad[rr] = ttf_rd(rs, bf16);
            }
        }
        if (lane >= nt) {
            continue;
        }
        // lane tt finishes token t0 + tt for each row
        const int64_t t   = t0 + lane;
        const char *  row = qa + t * ldq;
        const float   s   = *(const float *) (row + Kp);
        const int     qs  = *(const int *) (row + Kp + 4);
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            if (rr >= nr) {
                break;
            }
            const int64_t n  = n0 + rr;
            int           at = a[rr][0];
#pragma unroll
            for (int tt = 1; tt < TTF_MM_TOK; ++tt) {
                if (lane == tt) {
                    at = a[rr][tt];
                }
            }
            float y = ttf_rbf(__fmul_rn(__fmul_rn(s, (float) (at - qs)), beta[0]));
            y       = ttf_rd(y, bf16);
            if (mA) {
                y = ttf_rd(__fadd_rn(y, ad[rr]), bf16);
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
    }
}

// int8 tensor-core path for T >= TTF_MMA_MIN_T (sm_80+): mma.sync m16n8k32 s8.s8.s32. Each 32-trit plane of a weight
// block is one k-tile: plane j of block bi holds trits k = QB bi + 32 j + p at byte p (both layouts), so the A fragment
// of row g, k = 4 tq + e is plane j of int tq (k < 16) or int tq + 4 (k >= 16) of the row's block. B (activation codes)
// is staged per block bi in shared memory, rows padded to QB / 4 + 4 ints (no bank conflicts). The int32 sums are exact,
// so the result equals the dp4a kernel's; the epilogue is the same (adapter dot per output in ttf_dot8's order).
// block: 4 warps x 16 rows; NT n-tiles of 8 tokens per block.
#define TTF_MMA_MIN_T 8

static __device__ __forceinline__ void ttf_mma_s8(int (&c)[4], const int a0, const int a1, const int a2, const int a3,
                                                  const int b0, const int b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    NO_DEVICE_CODE;
#endif
}

template <bool Q2, int NT>
static __global__ void ttf_mma_kernel(const uint8_t * w, const int64_t nbw1, const char * qa, const int64_t ldq,
                                      const float * beta, const float * mA, const int rank, const float * r,
                                      const float * b, const float * acc, const bool bf16, float * out,
                                      const int64_t N, const int64_t T, const int64_t Kp) {
    constexpr int QB = Q2 ? 128 : 160, NP = Q2 ? 4 : 5, LDB = QB / 4 + 4, TB = 8 * NT;
    __shared__ uint32_t lut[Q2 ? 1 : 256];
    __shared__ int      sb[TB * LDB];
    if (!Q2) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            lut[i] = ttf_tq1p_lut_entry(i);
        }
    }
    const int     lane = threadIdx.x % WARP_SIZE, g = lane / 4, tq = lane % 4;
    const int64_t n0 = ((int64_t) blockIdx.x * TTF_MM_WARPS + threadIdx.x / WARP_SIZE) * 16;
    const int64_t t0 = (int64_t) blockIdx.y * TB;
    const int64_t ra = n0 + g, rb = n0 + g + 8;
    const int *   wa = ra < N ? (const int *) (w + ra * nbw1) : nullptr;
    const int *   wb = rb < N ? (const int *) (w + rb * nbw1) : nullptr;
    int           c[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0;
    }
    const int64_t nb = Kp / QB;
    for (int64_t bi = 0; bi < nb; ++bi) {
        __syncthreads();
        for (int i = threadIdx.x; i < TB * (QB / 4); i += blockDim.x) {
            const int     tt = i / (QB / 4), q = i % (QB / 4);
            const int64_t t  = t0 + tt;
            sb[tt * LDB + q] = t < T ? *(const int *) (qa + t * ldq + QB * bi + 4 * q) : 0;
        }
        __syncthreads();
        // the 4 ints of this thread's A fragments: rows g / g + 8, ints tq / tq + 4 of block bi
        int v[4];
        v[0] = wa ? wa[bi * 8 + tq] : 0;
        v[1] = wb ? wb[bi * 8 + tq] : 0;
        v[2] = wa ? wa[bi * 8 + tq + 4] : 0;
        v[3] = wb ? wb[bi * 8 + tq + 4] : 0;
        int hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (Q2) {
                hi[i] = 0;
            } else {
                const uint32_t e0 = lut[(v[i]) & 0xFF], e1 = lut[(v[i] >> 8) & 0xFF], e2 = lut[(v[i] >> 16) & 0xFF],
                               e3 = lut[(uint32_t) v[i] >> 24];
                const uint32_t p01 = __byte_perm(e0, e1, 0x5140);
                const uint32_t p23 = __byte_perm(e2, e3, 0x5140);
                v[i]               = (int) __byte_perm(p01, p23, 0x5410);
                hi[i]              = (int) __byte_perm(p01, p23, 0x7632);
            }
        }
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            int a[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = j < 4 ? (v[i] >> (2 * j)) & 0x03030303 : hi[i] & 0x03030303;
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int * br = sb + (8 * nt + g) * LDB + 8 * j + tq;
                ttf_mma_s8(c[nt], a[0], a[1], a[2], a[3], br[0], br[4]);
            }
        }
    }
    // c[nt]: (row g, token 2 tq), (g, 2 tq + 1), (g + 8, 2 tq), (g + 8, 2 tq + 1) of n-tile nt
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int64_t n = e < 2 ? ra : rb;
            const int64_t t = t0 + 8 * nt + 2 * tq + (e & 1);
            if (n >= N || t >= T) {
                continue;
            }
            const char * row = qa + t * ldq;
            const float  s   = *(const float *) (row + Kp);
            const int    qs  = *(const int *) (row + Kp + 4);
            float        y   = ttf_rbf(__fmul_rn(__fmul_rn(s, (float) (c[nt][e] - qs)), beta[0]));
            y                = ttf_rd(y, bf16);
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
    }
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
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t mgx = (N + TTF_MM_WARPS * 16 - 1) / (TTF_MM_WARPS * 16);
    const int     nsm = ggml_cuda_info().devices[ctx.device].nsm;
    // MMA only when its grid (one block per 64 rows x 8 tokens at least) can fill the GPU; else the SIMT kernel, which
    // splits rows finer
    if (T >= TTF_MMA_MIN_T && ampere_mma_available(cc) && mgx * ((T + 7) / 8) >= nsm && !getenv("TTF_NO_MMA")) {
        const int64_t gx = mgx;
        auto          go = [&](auto kern, const int tb) {
            const dim3 grid((unsigned) gx, (unsigned) ((T + tb - 1) / tb));
            kern<<<grid, TTF_MM_WARPS * WARP_SIZE, 0, ctx.stream()>>>(
                (const uint8_t *) w->data, w->nb[1], (const char *) qa->data, qa->nb[1], (const float *) be->data,
                mA ? (const float *) mA->data : nullptr, rank, r ? (const float *) r->data : nullptr,
                b ? (const float *) b->data : nullptr, acc ? (const float *) acc->data : nullptr, bf16,
                (float *) dst->data, N, T, Kp);
        };
        // widest token tile that still gives >= 1 block per SM (wider tiles reuse each A fragment more)
        const int nt = gx * ((T + 31) / 32) >= nsm ? 4 : gx * ((T + 15) / 16) >= nsm ? 2 : 1;
        if (q2) {
            nt == 4 ? go(ttf_mma_kernel<true, 4>, 32) : nt == 2 ? go(ttf_mma_kernel<true, 2>, 16) : go(ttf_mma_kernel<true, 1>, 8);
        } else {
            nt == 4 ? go(ttf_mma_kernel<false, 4>, 32) : nt == 2 ? go(ttf_mma_kernel<false, 2>, 16) : go(ttf_mma_kernel<false, 1>, 8);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // grid.y: token chunks split over blocks only as far as needed to fill the GPU (~2048 blocks); each block loops
    // over its chunks
    const int     R  = T > TTF_MM_TOK ? 4 : 1;
    const int64_t gx = (N + TTF_MM_WARPS * R - 1) / (TTF_MM_WARPS * R), nch = (T + TTF_MM_TOK - 1) / TTF_MM_TOK;
    const int64_t gy = std::max<int64_t>(1, std::min<int64_t>(nch, 2048 / gx));
    const dim3    grid((unsigned) gx, (unsigned) gy);
    const dim3 block(TTF_MM_WARPS * WARP_SIZE);
    auto       go = [&](auto kern) {
        kern<<<grid, block, 0, ctx.stream()>>>(
            (const uint8_t *) w->data, w->nb[1], (const char *) qa->data, qa->nb[1], (const float *) be->data,
            mA ? (const float *) mA->data : nullptr, rank, r ? (const float *) r->data : nullptr,
            b ? (const float *) b->data : nullptr, acc ? (const float *) acc->data : nullptr, bf16,
            (float *) dst->data, N, T, Kp);
    };
    if (q2) {
        R == 4 ? go(ttf_mm_kernel<true, 4>) : go(ttf_mm_kernel<true, 1>);
    } else {
        R == 4 ? go(ttf_mm_kernel<false, 4>) : go(ttf_mm_kernel<false, 1>);
    }
    CUDA_CHECK(cudaGetLastError());
}
