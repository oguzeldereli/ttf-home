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
#include <climits>

#define TTF_QROW_TAIL 32
#define TTF_MM_WARPS  4   // rows per block, one warp each
#define TTF_MM_TOK    4   // tokens per warp pass
#define TTF_MM_AP     2   // adapter values per lane kept in registers by the decode kernel (ranks <= 16)

static __device__ __forceinline__ float ttf_rbf(const float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
}

static __device__ __forceinline__ float ttf_rd(const float v, const bool bf16) {
    return bf16 ? ttf_rbf(v) : v;
}

// the input x_k: x itself, or act(g_k) * u_k with ggml-cuda's unary formulas (op_silu, op_relu_sqr), as the unfused
// UNARY [SQR] MUL nodes compute it (a fused MLP gate in front of a quantisation)
#define TTF_ACT_NONE  0
#define TTF_ACT_SILU  1
#define TTF_ACT_RELU2 2
static __device__ __forceinline__ float ttf_act(const float x, const int act) {
    if (act == TTF_ACT_SILU) {
        return x / (1.0f + expf(-x));
    }
    const float r = fmaxf(x, 0.0f);
    return r * r;
}

static __device__ __forceinline__ float ttf_xin(const float * g, const float * u, const int act, const int64_t k) {
    return act == TTF_ACT_NONE ? g[k] : ttf_act(g[k], act) * u[k];
}

// xc_k as the CPU computes it (steps 0-2) from the input value v = x_k
static __device__ __forceinline__ float ttf_xc_v(const float v, const float * mu, const float * c, const int64_t k,
                                                 const bool bf16) {
    float t = ttf_rd(v, bf16);
    if (mu) {
        t = ttf_rd(__fsub_rn(t, mu[k]), bf16);
    }
    if (c) {
        t = ttf_rd(__fmul_rn(t, c[k]), bf16);
    }
    return t;
}

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

// adapter factors (mag_A / mag_B): F32, or BF16 (a bf16-D model's values are bf16: the same floats, half the bytes).
// Kernels take the type as a template flag (BH; a runtime select in the loops cost registers and load batching) and
// read element i with ttf_ad_ld<BH>
struct ttf_adp {
    const void * p;
    bool         bh;
    __host__ __device__ explicit operator bool() const { return p != nullptr; }
};
template <bool BH> static __device__ __forceinline__ float ttf_ad_ld(const void * p, const int64_t i) {
    return BH ? __bfloat162float(((const nv_bfloat16 *) p)[i]) : ((const float *) p)[i];
}
static ttf_adp ttf_adp_of(const ggml_tensor * t) {
    GGML_ASSERT(!t || t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16);
    return { t ? t->data : nullptr, t && t->type == GGML_TYPE_BF16 };
}

// dot_f32's order for n floats a . b (a, b contiguous): 8 lanes by fma, lanes summed in order, then the tail
template <bool BH>
static __device__ __forceinline__ float ttf_dot8(const float * a, const void * b, const int64_t bo, const int64_t n) {
    float         l[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    const int64_t n8   = n / 8 * 8;
    for (int64_t k = 0; k < n8; k += 8) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            l[i] = __fmaf_rn(a[k + i], ttf_ad_ld<BH>(b, bo + k + i), l[i]);
        }
    }
    float r = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        r = __fadd_rn(r, l[i]);
    }
    for (int64_t k = n8; k < n; ++k) {
        r = __fadd_rn(r, __fmul_rn(a[k], ttf_ad_ld<BH>(b, bo + k)));
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
// the adapter's down projection u_j = xc . bj by one warp (lane-strided, two chains, then a shuffle tree; lane 0
// holds the sum); xk(k) gives xc[k]. One order for every kernel that makes u (bit-identical results)
template <bool BH, class XK>
static __device__ __forceinline__ float ttf_adapter_dot_t(XK xk, const void * mB, const int64_t bo, const int64_t K,
                                                         const int lane) {
    float         a0 = 0.0f, a1 = 0.0f;
    const int64_t K64 = K / 64 * 64;
    int64_t       k0  = 0;
    // full blocks, 4 at a time: the loads of 4 blocks issue before the (ordered) fma chains
#pragma unroll 2
    for (; k0 + 256 <= K64; k0 += 256) {
        float xa[8], wa[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int64_t k = k0 + 32 * u + lane;
            xa[u]           = xk(k);
            wa[u]           = ttf_ad_ld<BH>(mB, bo + k);
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
            a0 = __fmaf_rn(xk(k), ttf_ad_ld<BH>(mB, bo + k), a0);
        }
        if (k + 32 < K) {
            a1 = __fmaf_rn(xk(k + 32), ttf_ad_ld<BH>(mB, bo + k + 32), a1);
        }
    }
    float s = __fadd_rn(a0, a1);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s = __fadd_rn(s, __shfl_down_sync(0xffffffff, s, off));
    }
    return s;
}
template <class XK>
static __device__ __forceinline__ float ttf_adapter_dot(XK xk, const ttf_adp mB, const int64_t bo, const int64_t K,
                                                       const int lane) {
    return mB.bh ? ttf_adapter_dot_t<true>(xk, mB.p, bo, K, lane) : ttf_adapter_dot_t<false>(xk, mB.p, bo, K, lane);
}

// up to TTF_QMAX quantisations of one input in one launch (parts on one input: own mu / c / adapter), blockIdx.z
struct ttf_qpart {
    const float * mu;
    const float * c;
    const void *  mB;  // F32, or BF16 when mbh (ttf_adp)
    char *        dst;
    int64_t       nbd1;
    int64_t       Kp;
    int           rank;
    int           mbh;
};
#define TTF_QMAX 4
struct ttf_qparts {
    ttf_qpart p[TTF_QMAX];
};
#define TTF_QCH 4096  // decode adapter blocks: xc / bj chunk staged in shared memory (2 x 16 KB)

static __global__ void __launch_bounds__(1024) ttf_quant_kernel(const char * x, const int64_t nbx1, const int64_t K, const bool bf16,
                                        const int xsz, const ttf_qparts P, const char * ux, const int64_t nbu1,
                                        const int act, const int one_j) {
    __shared__ float sf[32];
    __shared__ int   si[32];
    const int        z    = blockIdx.z;  // static indices: the parameters stay in the parameter bank
    const ttf_qpart  q    = z == 0 ? P.p[0] : z == 1 ? P.p[1] : z == 2 ? P.p[2] : P.p[3];
    const float *    mu   = q.mu;
    const float *    c    = q.c;
    const ttf_adp    mB   = { q.mB, q.mbh != 0 };
    char *           dst  = q.dst;
    const int64_t    nbd1 = q.nbd1;
    const int64_t    Kp   = q.Kp;
    const int        rank = q.rank;
    const int64_t    row = blockIdx.x;
    const float *    xr  = (const float *) (x + row * nbx1);
    const float *    ur  = act != TTF_ACT_NONE ? (const float *) (ux + row * nbu1) : nullptr;
    char *           o   = dst + row * nbd1;
    extern __shared__ float xs[];
    if (blockIdx.y > 0 && one_j) {
        // few rows (decode): one block per down projection j, so the rank's reads spread over the SMs. The block
        // stages TTF_QCH of xc and of bj in shared memory, and warp 0 runs ttf_adapter_dot's two chains over them in
        // its order (chunks are whole 64-blocks), so u is bit-identical
        const int j = blockIdx.y - 1;
        if (j >= rank) {
            return;
        }
        const int     lane = threadIdx.x % WARP_SIZE;
        const int64_t bo   = (int64_t) j * K;
        float *       ws   = xs + TTF_QCH;
        float         a0 = 0.0f, a1 = 0.0f;
        for (int64_t c0 = 0; c0 < K; c0 += TTF_QCH) {
            const int n = (int) min((int64_t) TTF_QCH, K - c0);
            for (int k = threadIdx.x; k < n; k += blockDim.x) {
                xs[k] = ttf_xc_v(ttf_xin(xr, ur, act, c0 + k), mu, c, c0 + k, bf16);
                ws[k] = mB.bh ? ttf_ad_ld<true>(mB.p, bo + c0 + k) : ttf_ad_ld<false>(mB.p, bo + c0 + k);
            }
            __syncthreads();
            if (threadIdx.x < WARP_SIZE) {
                int k = lane;
                // 4 steps of 64 per round: the shared loads issue ahead of the (ordered) fma chains
                for (; k + 192 + 32 < n; k += 256) {
                    float xv[8], wv[8];
#pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        xv[u] = xs[k + 32 * u];
                        wv[u] = ws[k + 32 * u];
                    }
#pragma unroll
                    for (int u = 0; u < 8; u += 2) {
                        a0 = __fmaf_rn(xv[u], wv[u], a0);
                        a1 = __fmaf_rn(xv[u + 1], wv[u + 1], a1);
                    }
                }
                for (; k < n; k += 64) {
                    a0 = __fmaf_rn(xs[k], ws[k], a0);
                    if (k + 32 < n) {
                        a1 = __fmaf_rn(xs[k + 32], ws[k + 32], a1);
                    }
                }
            }
            __syncthreads();
        }
        if (threadIdx.x < WARP_SIZE) {
            float sum = __fadd_rn(a0, a1);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, off));
            }
            if (lane == 0) {
                *(float *) (o + Kp + TTF_QROW_TAIL + 4 * j) = ttf_rd(sum, bf16);
            }
        }
        return;
    }
    if (blockIdx.y > 0) {
        const int lane = threadIdx.x % WARP_SIZE;
        const int j    = (blockIdx.y - 1) * (int) (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
        // xc staged once per block in shared memory (the warps' dots all read it); dynamic smem: K floats or 0
        const bool staged = xsz > 0;
        if (staged) {
            for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
                xs[k] = ttf_xc_v(ttf_xin(xr, ur, act, k), mu, c, k, bf16);
            }
            __syncthreads();
        }
        if (j >= rank) {
            return;
        }
        const float   s  = ttf_adapter_dot([&](int64_t k) { return staged ? xs[k] : ttf_xc_v(ttf_xin(xr, ur, act, k), mu, c, k, bf16); }, mB, (int64_t) j * K, K, lane);
        if (lane == 0) {
            *(float *) (o + Kp + TTF_QROW_TAIL + 4 * j) = ttf_rd(s, bf16);
        }
        return;
    }
    // xc kept in registers between the two passes when the row fits (K <= TTF_QV blockDim); max and the integer sum
    // do not depend on the order
    constexpr int TTF_QV = 20;  // decode (1024 threads): rows up to 20480 held (27B down: 17408)
    const bool    held   = K <= (int64_t) TTF_QV * blockDim.x;
    float         xv[TTF_QV];
    float         am = 0.0f;
    if (held) {
#pragma unroll
        for (int u = 0; u < TTF_QV; ++u) {
            const int64_t k = threadIdx.x + (int64_t) u * blockDim.x;
            xv[u]           = k < K ? ttf_xc_v(ttf_xin(xr, ur, act, k), mu, c, k, bf16) : 0.0f;
            am              = fmaxf(am, fabsf(xv[u]));
        }
    } else {
        for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
            am = fmaxf(am, fabsf(ttf_xc_v(ttf_xin(xr, ur, act, k), mu, c, k, bf16)));
        }
    }
    am             = ttf_block_max(am, sf);
    const float fl = ttf_rd(1e-5f, bf16);
    const float s  = __fmul_rn(fmaxf(am, fl), 1.0f / 127.0f);
    int         qs = 0;
    if (held) {
#pragma unroll
        for (int u = 0; u < TTF_QV; ++u) {
            const int64_t k = threadIdx.x + (int64_t) u * blockDim.x;
            if (k < Kp) {
                int q = 0;
                if (k < K) {
                    q = (int) fminf(127.0f, fmaxf(-128.0f, rintf(__fdiv_rn(xv[u], s))));
                }
                o[k] = (char) q;
                qs += q;
            }
        }
        for (int64_t k = threadIdx.x + (int64_t) TTF_QV * blockDim.x; k < Kp; k += blockDim.x) {
            o[k] = 0;  // padding past K beyond the held part
        }
    } else
    for (int64_t k = threadIdx.x; k < Kp; k += blockDim.x) {
        int q = 0;
        if (k < K) {
            const float t = rintf(__fdiv_rn(ttf_xc_v(ttf_xin(xr, ur, act, k), mu, c, k, bf16), s));
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

// decode split: a row over several blocks (one block per row put all of a 17408-wide row's work on one SM: 16 us).
// Kernel 1: block b of part z makes xc of its slice into xcb [z][row][K] and the slice's max |xc| into pm [z][row][b];
// block 0 zeroes the row's qsum. Kernel 2: slice blocks read the B maxima (max is order-free), quantise their slice
// and add their integer sums atomically (order-free); blocks B.. make the adapter's down projections from xcb, in
// ttf_adapter_dot's order (as the one-block-per-j path). Bit-identical to ttf_quant_kernel
#define TTF_QSPLIT_EL 1024  // elements per slice block
static __global__ void __launch_bounds__(256) ttf_quant_xc_kernel(const char * x, const int64_t nbx1, const int64_t K,
                                                                  const bool bf16, const ttf_qparts P, const char * ux,
                                                                  const int64_t nbu1, const int act, float * xcb,
                                                                  float * pm, const int B) {
    __shared__ float sf[32];
    const int       z  = blockIdx.z;
    const ttf_qpart q  = z == 0 ? P.p[0] : z == 1 ? P.p[1] : z == 2 ? P.p[2] : P.p[3];
    const int64_t   row = blockIdx.y, rows = gridDim.y;
    const float *   xr  = (const float *) (x + row * nbx1);
    const float *   ur  = act != TTF_ACT_NONE ? (const float *) (ux + row * nbu1) : nullptr;
    float *         xo  = xcb + (z * rows + row) * K;
    const int64_t   k0  = (int64_t) blockIdx.x * TTF_QSPLIT_EL, k1 = min(K, k0 + TTF_QSPLIT_EL);
    float           am  = 0.0f;
    for (int64_t k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
        const float v = ttf_xc_v(ttf_xin(xr, ur, act, k), q.mu, q.c, k, bf16);
        xo[k]         = v;
        am            = fmaxf(am, fabsf(v));
    }
    am = ttf_block_max(am, sf);
    if (threadIdx.x == 0) {
        pm[(z * rows + row) * B + blockIdx.x] = am;
        if (blockIdx.x == 0) {
            *(int *) (q.dst + row * q.nbd1 + q.Kp + 4) = 0;
        }
    }
}

#define TTF_QSPLIT_TH 512   // threads per kernel-2 block
#define TTF_QSPLIT_CH 8192  // adapter chunk (xc and mB_j: 2 x 32 KB shared)
// adapter block of the decode split: u_j = rD(xc . mB_j) from the row's xc in global memory, in ttf_adapter_dot's
// order (chunks of CH, a multiple of 64, staged in xs: 2 CH floats)
template <int CH>
static __device__ __forceinline__ void ttf_quant_adapter_block(const ttf_qpart & q, const float * xc, const int64_t K,
                                                               const bool bf16, const int j, char * o, float * xs) {
    const int     lane = threadIdx.x % WARP_SIZE;
    const int64_t bo   = (int64_t) j * K;
    float *       ws   = xs + CH;
    float         a0 = 0.0f, a1 = 0.0f;
    const bool    v4 = (K % 4) == 0 && ((uintptr_t) q.mB % 16) == 0 && ((uintptr_t) xc % 16) == 0;
    for (int64_t c0 = 0; c0 < K; c0 += CH) {
        const int n = (int) min((int64_t) CH, K - c0);
        if (v4) {  // float4 copies, 4 + 4 per thread in flight
            const int n4 = n / 4;
#pragma unroll 4
            for (int k = threadIdx.x; k < n4; k += blockDim.x) {
                ((float4 *) xs)[k] = ((const float4 *) (xc + c0))[k];
                if (q.mbh) {  // 4 bf16 (8 bytes): a bf16's float is its bits shifted up
                    const uint2 h = ((const uint2 *) ((const nv_bfloat16 *) q.mB + bo + c0))[k];
                    ((float4 *) ws)[k] = make_float4(__uint_as_float(h.x << 16), __uint_as_float(h.x & 0xffff0000u),
                                                     __uint_as_float(h.y << 16), __uint_as_float(h.y & 0xffff0000u));
                } else {
                    ((float4 *) ws)[k] = ((const float4 *) ((const float *) q.mB + bo + c0))[k];
                }
            }
        } else {
            for (int k = threadIdx.x; k < n; k += blockDim.x) {
                xs[k] = xc[c0 + k];
                ws[k] = q.mbh ? ttf_ad_ld<true>(q.mB, bo + c0 + k) : ttf_ad_ld<false>(q.mB, bo + c0 + k);
            }
        }
        __syncthreads();
        // the two chains of ttf_adapter_dot (a0: k = lane + 64 i, a1: k = lane + 32 + 64 i) are independent
        // until their final sum: warp 0 runs a0, warp 1 a1
        if (threadIdx.x < 2 * WARP_SIZE) {
            const int w = threadIdx.x / WARP_SIZE;
            float     a = w ? a1 : a0;
            int       k = lane + 32 * w;
            for (; k + 7 * 64 < n; k += 8 * 64) {
                float xv[8], wv[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    xv[u] = xs[k + 64 * u];
                    wv[u] = ws[k + 64 * u];
                }
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    a = __fmaf_rn(xv[u], wv[u], a);
                }
            }
            for (; k < n; k += 64) {
                a = __fmaf_rn(xs[k], ws[k], a);
            }
            if (w) {
                a1 = a;
            } else {
                a0 = a;
            }
        }
        __syncthreads();
    }
    __shared__ float sa1[WARP_SIZE];
    if (threadIdx.x >= WARP_SIZE && threadIdx.x < 2 * WARP_SIZE) {
        sa1[lane] = a1;
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        float sum = __fadd_rn(a0, sa1[lane]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, off));
        }
        if (lane == 0) {
            *(float *) (o + q.Kp + TTF_QROW_TAIL + 4 * j) = ttf_rd(sum, bf16);
        }
    }
}

static __global__ void __launch_bounds__(TTF_QSPLIT_TH) ttf_quant_q_kernel(const int64_t K, const bool bf16, const ttf_qparts P,
                                                                 const float * xcb, const float * pm, const int B) {
    __shared__ int  si[32];
    const int       z  = blockIdx.z;
    const ttf_qpart q  = z == 0 ? P.p[0] : z == 1 ? P.p[1] : z == 2 ? P.p[2] : P.p[3];
    const int64_t   row = blockIdx.y, rows = gridDim.y;
    const float *   xc  = xcb + (z * rows + row) * K;
    char *          o   = q.dst + row * q.nbd1;
    if ((int) blockIdx.x >= B) {  // adapter: u_j, j = blockIdx.x - B
        extern __shared__ float xs[];
        const int j = blockIdx.x - B;
        if (j < q.rank) {
            ttf_quant_adapter_block<TTF_QSPLIT_CH>(q, xc, K, bf16, j, o, xs);
        }
        return;
    }
    float am = 0.0f;
    for (int b = 0; b < B; ++b) {
        am = fmaxf(am, pm[(z * rows + row) * B + b]);
    }
    const float   fl = ttf_rd(1e-5f, bf16);
    const float   s  = __fmul_rn(fmaxf(am, fl), 1.0f / 127.0f);
    // slice b writes codes [b EL, (b + 1) EL), the last one also the padding up to Kp
    const int64_t k0 = (int64_t) blockIdx.x * TTF_QSPLIT_EL;
    const int64_t k1 = (int) blockIdx.x == B - 1 ? q.Kp : min(q.Kp, k0 + TTF_QSPLIT_EL);
    int           qs = 0;
    for (int64_t k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
        int v = 0;
        if (k < K) {
            v = (int) fminf(127.0f, fmaxf(-128.0f, rintf(__fdiv_rn(xc[k], s))));
        }
        o[k] = (char) v;
        qs += v;
    }
    qs = ttf_block_sum_i(qs, si);
    if (threadIdx.x == 0) {
        if (qs != 0) {
            atomicAdd((int *) (o + q.Kp + 4), qs);
        }
        if (blockIdx.x == 0) {
            *(float *) (o + q.Kp) = s;
        }
    }
}

// the decode split without adapters in one launch (TTF_QUANT_TWO_LAUNCH=1: the two kernels above). Grid (B, rows,
// parts), TTF_QSPLIT_TH threads: slice block b makes xc of its slice (kept in registers), adds its max |xc| to the
// row's maximum and arrives; when the row's B blocks have arrived, each quantises its slice (sums atomically, order-
// free). Waiting needs the whole grid resident at once: the host checks the occupancy, else two launches. The row's
// counters live in device globals (zero at load) and the grid's last block out resets them, so launches on one
// stream reuse them; one stream per device uses this kernel (the session's). Bit-identical to the two kernels. (With
// adapters, their blocks waiting in the same launch for xc cost more than the second launch: 27B -0.4%)
#define TTF_QONE_SLOT 8  // rows x parts (the split's rows * np <= 8)
__device__ unsigned int ttf_qone_arrive[TTF_QONE_SLOT];
__device__ unsigned int ttf_qone_depart[TTF_QONE_SLOT];
__device__ unsigned int ttf_qone_max[TTF_QONE_SLOT];  // max |xc| as float bits (>= 0: unsigned order is float order)

static __global__ void __launch_bounds__(TTF_QSPLIT_TH) ttf_quant_one_kernel(const char * x, const int64_t nbx1, const int64_t K,
                                                                   const bool bf16, const ttf_qparts P, const char * ux,
                                                                   const int64_t nbu1, const int act, const int B) {
    constexpr int   PT = TTF_QSPLIT_EL / TTF_QSPLIT_TH;  // slice elements held per thread
    __shared__ float sf[32];
    __shared__ int   si[32];
    __shared__ float sam;
    const int        z    = blockIdx.z;
    const ttf_qpart  q    = z == 0 ? P.p[0] : z == 1 ? P.p[1] : z == 2 ? P.p[2] : P.p[3];
    const int64_t    row  = blockIdx.y, rows = gridDim.y;
    const int        slot = (int) (z * rows + row);
    char *           o    = q.dst + row * q.nbd1;
    const int64_t    k0   = (int64_t) blockIdx.x * TTF_QSPLIT_EL;
    const float *    xr   = (const float *) (x + row * nbx1);
    const float *    ur   = act != TTF_ACT_NONE ? (const float *) (ux + row * nbu1) : nullptr;
    float            xv[PT];
    float            am = 0.0f;
#pragma unroll
    for (int u = 0; u < PT; ++u) {
        const int64_t k = k0 + threadIdx.x + u * TTF_QSPLIT_TH;
        xv[u]           = k < K ? ttf_xc_v(ttf_xin(xr, ur, act, k), q.mu, q.c, k, bf16) : 0.0f;
        am              = fmaxf(am, fabsf(xv[u]));
    }
    am = ttf_block_max(am, sf);
    if (threadIdx.x == 0) {
        atomicMax(&ttf_qone_max[slot], __float_as_uint(am));
        if (blockIdx.x == 0) {
            *(int *) (o + q.Kp + 4) = 0;
        }
        __threadfence();  // the max and the zeroed sum before the arrival
        atomicAdd(&ttf_qone_arrive[slot], 1u);
        while (*(volatile unsigned int *) &ttf_qone_arrive[slot] < (unsigned int) B) {
        }
        __threadfence();
        sam = __uint_as_float(*(volatile unsigned int *) &ttf_qone_max[slot]);
        // the last block out resets the row's counters (each has read the max by now)
        if (atomicAdd(&ttf_qone_depart[slot], 1u) == gridDim.x - 1) {
            ttf_qone_arrive[slot] = 0;
            ttf_qone_max[slot]    = 0;
            ttf_qone_depart[slot] = 0;
        }
    }
    __syncthreads();
    const float   s  = __fmul_rn(fmaxf(sam, ttf_rd(1e-5f, bf16)), 1.0f / 127.0f);
    // slice b writes codes [b EL, (b + 1) EL), the last one also the padding up to Kp (< 160 past K)
    const int64_t k1 = (int) blockIdx.x == B - 1 ? q.Kp : min(q.Kp, k0 + TTF_QSPLIT_EL);
    int           qs = 0;
#pragma unroll
    for (int u = 0; u < PT + 1; ++u) {
        const int64_t k = k0 + threadIdx.x + u * TTF_QSPLIT_TH;
        if (k < k1) {
            int v = 0;
            if (u < PT && k < K) {
                v = (int) fminf(127.0f, fmaxf(-128.0f, rintf(__fdiv_rn(xv[u < PT ? u : 0], s))));
            }
            o[k] = (char) v;
            qs += v;
        }
    }
    qs = ttf_block_sum_i(qs, si);
    if (threadIdx.x == 0) {
        if (qs != 0) {
            atomicAdd((int *) (o + q.Kp + 4), qs);
        }
        if (blockIdx.x == 0) {
            *(float *) (o + q.Kp) = s;
        }
    }
}

// the TTF_QUANT nodes qs[0..np) all on the same x (same shape and rounding), one launch
// act: the quants' input is act(g) * u (g, u: same shape as the input), the product never written
static void ttf_quant_launch(ggml_backend_cuda_context & ctx, ggml_tensor * const * qs, const int np,
                             const ggml_tensor * g = nullptr, const ggml_tensor * u = nullptr, const int act = 0) {
    const ggml_tensor * x    = act ? g : qs[0]->src[0];
    const bool          bf16 = ggml_get_op_params_i32(qs[0], 1) != 0;
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_nrows(x) == x->ne[1]);  // 2-d input (the builder reshapes)
    GGML_ASSERT(np >= 1 && np <= TTF_QMAX);
    const int64_t rows = x->ne[1];
    if (rows == 0) {
        return;
    }
    ttf_qparts P = {};
    int        rmax = 0;
    for (int p = 0; p < np; ++p) {
        const ggml_tensor * d  = qs[p];
        const ggml_tensor * mB = d->src[3];
        GGML_ASSERT(ggml_is_contiguous(d));
        P.p[p].mu   = d->src[1] ? (const float *) d->src[1]->data : nullptr;
        P.p[p].c    = d->src[2] ? (const float *) d->src[2]->data : nullptr;
        P.p[p].mB   = ttf_adp_of(mB).p;
        P.p[p].mbh  = ttf_adp_of(mB).bh;
        P.p[p].dst  = (char *) d->data;
        P.p[p].nbd1 = d->nb[1];
        P.p[p].Kp   = ggml_get_op_params_i32(d, 0);
        P.p[p].rank = mB ? (int) mB->ne[1] : 0;
        rmax        = std::max(rmax, P.p[p].rank);
    }
    // few rows (decode): 1024 threads per row, the row held in registers; many rows: 256 (more blocks per SM)
    const int  nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const int  nth = rows < nsm ? 1024 : 256;
    // decode: the row split over blocks (TTF_QUANT_ONE_BLOCK=1: one block per row, as before)
    static const bool one_block = getenv("TTF_QUANT_ONE_BLOCK") != nullptr;
    if (!one_block && rows * np <= 8 && x->ne[0] > TTF_QSPLIT_EL) {
        const int64_t K = x->ne[0];
        const int     B = (int) ((K + TTF_QSPLIT_EL - 1) / TTF_QSPLIT_EL);
        // no adapters: one launch when the whole grid fits on the GPU at once (its blocks wait for each other)
        static const bool two = getenv("TTF_QUANT_TWO_LAUNCH") != nullptr;
        static int        one_per_sm = -1;
        if (one_per_sm < 0) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&one_per_sm, ttf_quant_one_kernel, TTF_QSPLIT_TH, 0));
        }
        static_assert(TTF_QSPLIT_EL % TTF_QSPLIT_TH == 0, "ttf_quant_one_kernel");
        // (not with concurrent streams, GGML_CUDA_GRAPH_OPT=1: the counters are one set per device)
        if (!two && rmax == 0 && rows * np <= TTF_QONE_SLOT && (int64_t) B * rows * np <= (int64_t) one_per_sm * nsm &&
                ctx.stream_context().concurrent_events.empty()) {
            ttf_quant_one_kernel<<<dim3((unsigned) B, (unsigned) rows, (unsigned) np), TTF_QSPLIT_TH, 0, ctx.stream()>>>(
                (const char *) x->data, x->nb[1], K, bf16, P, act ? (const char *) u->data : nullptr, act ? u->nb[1] : 0,
                act, B);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        ggml_cuda_pool_alloc<float> xcb(ctx.pool(), (size_t) (np * rows * K));
        ggml_cuda_pool_alloc<float> pm(ctx.pool(), (size_t) (np * rows * B));
        ttf_quant_xc_kernel<<<dim3((unsigned) B, (unsigned) rows, (unsigned) np), 256, 0, ctx.stream()>>>(
            (const char *) x->data, x->nb[1], K, bf16, P, act ? (const char *) u->data : nullptr, act ? u->nb[1] : 0, act,
            xcb.get(), pm.get(), B);
        const size_t smem = rmax > 0 ? 2 * TTF_QSPLIT_CH * sizeof(float) : 0;
        static bool  attr = false;
        if (!attr) {
            CUDA_CHECK(cudaFuncSetAttribute(ttf_quant_q_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int) (2 * TTF_QSPLIT_CH * sizeof(float))));
            attr = true;
        }
        ttf_quant_q_kernel<<<dim3((unsigned) (B + rmax), (unsigned) rows, (unsigned) np), TTF_QSPLIT_TH, smem,
                             ctx.stream()>>>(K, bf16, P, xcb.get(), pm.get(), B);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // few rows: one adapter block per down projection (TTF_QUANT_WARP_J=1: one warp each, as for many rows)
    static const bool warp_j = getenv("TTF_QUANT_WARP_J") != nullptr;
    const int  one_j = rmax > 0 && rows * np * 4 <= nsm && !warp_j;
    const dim3 grid((unsigned) rows, 1 + (one_j ? rmax : (rmax + nth / WARP_SIZE - 1) / (nth / WARP_SIZE)), (unsigned) np);
    // adapter blocks stage xc in shared memory when it fits (48 KB)
    const int xsz = rmax > 0 && x->ne[0] * (int64_t) sizeof(float) <= 48 * 1024 ? (int) (x->ne[0] * sizeof(float)) : 0;
    const int smem = one_j ? 2 * TTF_QCH * (int) sizeof(float) : xsz;
    ttf_quant_kernel<<<grid, nth, smem, ctx.stream()>>>((const char *) x->data, x->nb[1], x->ne[0], bf16, xsz, P,
                                                        act ? (const char *) u->data : nullptr, act ? u->nb[1] : 0, act,
                                                        one_j);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_ttf_quant(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ttf_quant_launch(ctx, &dst, 1);
}

bool ggml_cuda_ttf_act_quant(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u, int act,
                             ggml_tensor * const * qs, int np) {
    const ggml_tensor * x = qs[0]->src[0];
    if (np < 1 || np > TTF_QMAX || g->type != GGML_TYPE_F32 || u->type != GGML_TYPE_F32 || g->ne[0] != x->ne[0] ||
            ggml_nrows(g) != x->ne[1] || ggml_nrows(u) != x->ne[1] || u->ne[0] != x->ne[0] || g->nb[0] != 4 ||
            u->nb[0] != 4 || ggml_nrows(g) != g->ne[1] || ggml_nrows(u) != u->ne[1]) {
        return false;
    }
    for (int p = 1; p < np; ++p) {
        if (ggml_get_op_params_i32(qs[p], 1) != ggml_get_op_params_i32(qs[0], 1)) {
            return false;
        }
    }
    ttf_quant_launch(ctx, qs, np, g, u, act);
    return true;
}

bool ggml_cuda_ttf_quant_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * qs, int np) {
    if (np < 2 || np > TTF_QMAX) {
        return false;
    }
    for (int p = 1; p < np; ++p) {
        if (qs[p]->src[0]->data != qs[0]->src[0]->data || !ggml_are_same_shape(qs[p]->src[0], qs[0]->src[0]) ||
                qs[p]->src[0]->nb[1] != qs[0]->src[0]->nb[1] ||
                ggml_get_op_params_i32(qs[p], 1) != ggml_get_op_params_i32(qs[0], 1)) {
            return false;
        }
    }
    ttf_quant_launch(ctx, qs, np);
    return true;
}

// RMS_NORM -> MUL (weight row) -> TTF_QUANT (no adapter) as one kernel, one block per row. The norm is ggml's
// rms_norm_f32 vec path (same block size per width, same per-thread order, same block_reduce), so the values equal
// the unfused graph's; they stay in registers for the quantisation (steps 0-4, as ttf_quant_kernel).
// grid (rows, 1 + adapter blocks, parts): blockIdx.y > 0 recomputes the row's norm (same reduction), stages xc in
// shared memory and makes BS / 32 of the part's adapter down projections (ttf_adapter_dot, as ttf_quant_kernel)
template <int BS, int VPT>
static __global__ void ttf_norm_quant_kernel(const float * x, const int64_t sx, const float * mul, const float eps,
                                             const int K, const bool bf16, const ttf_qparts P, const float * ux,
                                             const int64_t su, const int act) {
    __shared__ float sf[32];
    __shared__ int   si[32];
    const int        z    = blockIdx.z;
    const ttf_qpart  qp   = z == 0 ? P.p[0] : z == 1 ? P.p[1] : z == 2 ? P.p[2] : P.p[3];
    const float *    mu   = qp.mu;
    const float *    c    = qp.c;
    const int64_t    Kp   = qp.Kp;
    const int        tid = threadIdx.x;
    const float4 *   x4  = (const float4 *) (x + blockIdx.x * sx);
    const float4 *   m4  = (const float4 *) mul;
    char *           o   = qp.dst + blockIdx.x * qp.nbd1;
    const int        n4  = K / 4;
    float4           xs[VPT];
    float            tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < VPT; k++) {
        const int i = tid + k * BS;
        if (i < n4) {
            xs[k] = x4[i];
            if (act != TTF_ACT_NONE) {  // the norm's input is act(g) * u
                const float4 u4 = ((const float4 *) (ux + blockIdx.x * su))[i];
                xs[k].x         = ttf_act(xs[k].x, act) * u4.x;
                xs[k].y         = ttf_act(xs[k].y, act) * u4.y;
                xs[k].z         = ttf_act(xs[k].z, act) * u4.z;
                xs[k].w         = ttf_act(xs[k].w, act) * u4.w;
            }
            tmp += xs[k].x * xs[k].x + xs[k].y * xs[k].y + xs[k].z * xs[k].z + xs[k].w * xs[k].w;
        }
    }
    tmp               = block_reduce<block_reduce_method::SUM, BS>(tmp, sf);
    const float scale = rsqrtf(tmp / K + eps);
    __syncthreads();  // sf reused below
    float am = 0.0f;
#pragma unroll
    for (int k = 0; k < VPT; k++) {
        const int i = tid + k * BS;
        if (i < n4) {
            float4       r = make_float4(scale * xs[k].x, scale * xs[k].y, scale * xs[k].z, scale * xs[k].w);
            const float4 m = m4[i];
            r.x *= m.x;
            r.y *= m.y;
            r.z *= m.z;
            r.w *= m.w;
            float * rv = &r.x;
#pragma unroll
            for (int e = 0; e < 4; ++e) {  // xc (steps 0-2)
                const int64_t col = 4 * (int64_t) i + e;
                float         t   = ttf_rd(rv[e], bf16);
                if (mu) {
                    t = ttf_rd(__fsub_rn(t, mu[col]), bf16);
                }
                if (c) {
                    t = ttf_rd(__fmul_rn(t, c[col]), bf16);
                }
                rv[e] = t;
                am    = fmaxf(am, fabsf(t));
            }
            xs[k] = r;
        }
    }
    if (blockIdx.y > 0) {
        extern __shared__ float xsh[];
#pragma unroll
        for (int k = 0; k < VPT; k++) {
            const int i = tid + k * BS;
            if (i < n4) {
                *(float4 *) (xsh + 4 * i) = xs[k];
            }
        }
        __syncthreads();
        const int lane = tid % WARP_SIZE;
        const int j    = (blockIdx.y - 1) * (BS / WARP_SIZE) + tid / WARP_SIZE;
        if (j < qp.rank) {
            const float s = ttf_adapter_dot([&](int64_t k) { return xsh[k]; }, ttf_adp{ qp.mB, qp.mbh != 0 }, (int64_t) j * K, K, lane);
            if (lane == 0) {
                *(float *) (o + Kp + TTF_QROW_TAIL + 4 * j) = ttf_rd(s, bf16);
            }
        }
        return;
    }
    am             = ttf_block_max(am, sf);
    const float fl = ttf_rd(1e-5f, bf16);
    const float s  = __fmul_rn(fmaxf(am, fl), 1.0f / 127.0f);
    int         qs = 0;
#pragma unroll
    for (int k = 0; k < VPT; k++) {
        const int i = tid + k * BS;
        if (i < n4) {
            const float * rv = &xs[k].x;
            char4         q4;
            signed char * qp = &q4.x;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int q = (int) fminf(127.0f, fmaxf(-128.0f, rintf(__fdiv_rn(rv[e], s))));
                qp[e]       = (signed char) q;
                qs += q;
            }
            *(char4 *) (o + 4 * (int64_t) i) = q4;
        }
    }
    for (int64_t k = K + tid; k < Kp; k += BS) {
        o[k] = 0;
    }
    qs = ttf_block_sum_i(qs, si);
    if (tid == 0) {
        *(float *) (o + Kp)   = s;
        *(int *) (o + Kp + 4) = qs;
    }
}

// rms: RMS_NORM on x [K, rows] F32 (rows contiguous); mul: its MUL by a weight row; q: TTF_QUANT of the product (or
// of a reshape of it), no adapter. False when the shapes do not fit the kernel (the caller then runs the nodes).
bool ggml_cuda_ttf_norm_quant(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul,
                              ggml_tensor * const * qs, const int np, const ggml_tensor * g, const ggml_tensor * u,
                              const int act) {
    const ggml_tensor * x = act ? g : rms->src[0];
    if (act && (u->type != GGML_TYPE_F32 || !ggml_are_same_shape(u, g) || u->nb[0] != 4 || u->nb[1] % 16 != 0 ||
                ((uintptr_t) u->data % 16) != 0 || ggml_nrows(u) != u->ne[1] || !ggml_are_same_shape(g, rms->src[0]))) {
        return false;
    }
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const int64_t       K = x->ne[0], rows = ggml_nrows(x);
    if (np < 1 || np > TTF_QMAX || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || K % 4 != 0 || K > 16384 ||
        x->nb[0] != sizeof(float) || x->nb[1] % 16 != 0 || ((uintptr_t) x->data % 16) != 0 || ggml_nrows(x) != x->ne[1] ||
        w->ne[0] != K || ggml_nelements(w) != K || !ggml_is_contiguous(w) || ((uintptr_t) w->data % 16) != 0) {
        return false;
    }
    ttf_qparts P = {};
    int        rmax = 0;
    for (int p = 0; p < np; ++p) {
        const ggml_tensor * q  = qs[p];
        const ggml_tensor * mB = q->src[3];
        if (ggml_nrows(q) != rows || q->ne[0] < ggml_get_op_params_i32(q, 0) || !ggml_is_contiguous(q) ||
                q->src[0]->ne[0] != K || (q->src[1] && !ggml_is_contiguous(q->src[1])) ||
                (q->src[2] && !ggml_is_contiguous(q->src[2])) || (mB && (!ggml_is_contiguous(mB) || mB->ne[0] != K)) ||
                ggml_get_op_params_i32(q, 1) != ggml_get_op_params_i32(qs[0], 1)) {
            return false;
        }
        P.p[p].mu   = q->src[1] ? (const float *) q->src[1]->data : nullptr;
        P.p[p].c    = q->src[2] ? (const float *) q->src[2]->data : nullptr;
        P.p[p].mB   = ttf_adp_of(mB).p;
        P.p[p].mbh  = ttf_adp_of(mB).bh;
        P.p[p].dst  = (char *) q->data;
        P.p[p].nbd1 = q->nb[1];
        P.p[p].Kp   = ggml_get_op_params_i32(q, 0);
        P.p[p].rank = mB ? (int) mB->ne[1] : 0;
        rmax        = std::max(rmax, P.p[p].rank);
    }
    if (rmax > 0 && K * (int64_t) sizeof(float) > 48 * 1024) {
        return false;
    }
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    const bool    bf16 = ggml_get_op_params_i32(qs[0], 1) != 0;
    const int64_t n4   = K / 4;
    const int     xsz  = rmax > 0 ? (int) (K * sizeof(float)) : 0;
    auto go = [&](auto kern, const int bs) {
        const dim3 grid((unsigned) rows, 1 + (rmax + bs / WARP_SIZE - 1) / (bs / WARP_SIZE), (unsigned) np);
        kern<<<grid, bs, xsz, ctx.stream()>>>((const float *) x->data, x->nb[1] / 4, (const float *) w->data, eps,
                                              (int) K, bf16, P, act ? (const float *) u->data : nullptr,
                                              act ? u->nb[1] / 4 : 0, act);
    };
    // rms_norm_f32_dispatch's vec choices
    if (n4 <= 32) {
        go(ttf_norm_quant_kernel<32, 1>, 32);
    } else if (n4 <= 128) {
        go(ttf_norm_quant_kernel<128, 1>, 128);
    } else if (n4 <= 256) {
        go(ttf_norm_quant_kernel<256, 1>, 256);
    } else if (n4 <= 512) {
        go(ttf_norm_quant_kernel<512, 1>, 512);
    } else if (n4 <= 1024) {
        go(ttf_norm_quant_kernel<1024, 1>, 1024);
    } else {
        go(ttf_norm_quant_kernel<1024, 4>, 1024);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
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
// TK: tokens per warp pass (1: the decode build); BH: mag_A in BF16 (else F32)
template <bool Q2, int R, bool H, int TK = TTF_MM_TOK, bool BH = false>
static __device__ __forceinline__ void ttf_mm_body(const uint8_t * w, const int64_t nbw1, const char * qa,
                                                   const int64_t ldq, const float * beta, const void * mA,
                                                   const int rank, const float * r, const float * b, const float * acc,
                                                   const bool bf16, float * out, const int64_t N, const int64_t T,
                                                   const int64_t Kp, const int64_t bx, const float * res) {
    __shared__ uint32_t lut[256];  // (a precomputed global table, read through L1, was slower: 168 vs 173 tok/s)
    if (!Q2) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            lut[i] = ttf_tq1p_lut_entry(i);
        }
        __syncthreads();
    }
    const int     lane = threadIdx.x % WARP_SIZE;
    const int64_t n0   = (bx * TTF_MM_WARPS + threadIdx.x / WARP_SIZE) * R;
    if (n0 >= N) {
        return;
    }
    const int nr = (int) min((int64_t) R, N - n0);
    const int m  = lane % 8;
    // the epilogue's per-row operands, loaded before the weight stream so that their latency hides behind it (as
    // mmvq-tq1p.cuh): row scale, bias, and the adapter row's values k = lane % 8 + 8 i (ranks up to 8 TTF_MM_AP)
    float rv[R], bv[R], av[R][TTF_MM_AP];
    const int64_t n8a = rank / 8 * 8;
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
        const bool live = rr < nr;
        rv[rr]          = r && live ? r[n0 + rr] : 0.0f;
        bv[rr]          = b && live ? b[n0 + rr] : 0.0f;
#pragma unroll
        for (int i = 0; i < TTF_MM_AP; ++i) {
            const int64_t k = m + 8 * i;
            av[rr][i]       = mA && live && k < n8a ? ttf_ad_ld<BH>(mA, (n0 + rr) * rank + k) : 0.0f;
        }
    }
    for (int64_t t0 = (int64_t) blockIdx.y * TK; t0 < T; t0 += (int64_t) gridDim.y * TK) {
        const int nt = (int) min((int64_t) TK, T - t0);
        // lane tt < nt: the residual of token t0 + tt per row (decode: before the weight stream)
        float resv[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            resv[rr] = res && rr < nr && lane < nt ? res[(t0 + lane) * N + n0 + rr] : 0.0f;
        }
        int       a[R][TK];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
#pragma unroll
            for (int tt = 0; tt < TK; ++tt) {
                a[rr][tt] = 0;
            }
        }
        const int64_t QB = Q2 ? 128 : 160;
        const int64_t nb = Kp / QB;
        const bool v16 = ldq % 16 == 0 && nbw1 % 16 == 0 && ((uintptr_t) qa | (uintptr_t) w) % 16 == 0;
        constexpr int RH = R >= 2 ? R / 2 : 1;  // rows per half warp (H)
        if (H && v16 && 2 * nb <= 16) {
            // short rows (<= 16 chunks, e.g. K 1024 = 14): half-warp h takes rows 2 q + h, lane l of the half chunk
            // l; the other half's sums for a row stay 0, so the full-warp reduction below gives the same integers
            const int     h  = lane >> 4;
            const int64_t c  = lane & 15;
            int           a2[RH][TK];
#pragma unroll
            for (int qq = 0; qq < RH; ++qq) {
#pragma unroll
                for (int tt = 0; tt < TK; ++tt) {
                    a2[qq][tt] = 0;
                }
            }
            if (c < 2 * nb) {
                const int64_t bi = c / 2;
                const int     p0 = 16 * (int) (c & 1);
                int           wa4[RH][4], wb4[RH][4];
#pragma unroll
                for (int qq = 0; qq < RH; ++qq) {
                    const int  rr    = 2 * qq + h;
                    const int4 v4    = rr < nr ? ((const int4 *) (w + (n0 + rr) * nbw1))[c] : make_int4(0, 0, 0, 0);
                    const int  vv[4] = { v4.x, v4.y, v4.z, v4.w };
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (Q2) {
                            wa4[qq][i] = vv[i];
                            wb4[qq][i] = 0;
                        } else {
                            const int      v = vv[i];
                            const uint32_t e0 = lut[(v) & 0xFF], e1 = lut[(v >> 8) & 0xFF], e2 = lut[(v >> 16) & 0xFF],
                                           e3 = lut[(uint32_t) v >> 24];
                            const uint32_t p01 = __byte_perm(e0, e1, 0x5140);
                            const uint32_t p23 = __byte_perm(e2, e3, 0x5140);
                            wa4[qq][i]         = (int) __byte_perm(p01, p23, 0x5410);
                            wb4[qq][i]         = (int) __byte_perm(p01, p23, 0x7632);
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < (Q2 ? 4 : 5); ++j) {
#pragma unroll
                    for (int tt = 0; tt < TK; ++tt) {
                        if (tt < nt) {
                            const int4 q     = *(const int4 *) (qa + (t0 + tt) * ldq + QB * bi + 32 * j + p0);
                            const int  qv[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
                            for (int qq = 0; qq < RH; ++qq) {
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    const int cj = j < 4 ? (wa4[qq][i] >> (2 * j)) & 0x03030303 : wb4[qq][i] & 0x03030303;
                                    a2[qq][tt]   = __dp4a(cj, qv[i], a2[qq][tt]);
                                }
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int qq = 0; qq < RH; ++qq) {
#pragma unroll
                for (int tt = 0; tt < TK; ++tt) {
                    if (2 * qq + 1 < R) {
                        a[2 * qq][tt]     = h ? 0 : a2[qq][tt];
                        a[2 * qq + 1][tt] = h ? a2[qq][tt] : 0;
                    }
                }
            }
        } else if (v16) {
            // lanes take the rows' 16-byte chunks (a TQ1_P / Q2P block is 32 bytes = 2 chunks; byte p of the block
            // holds plane j's trit k = QB bi + 32 j + p): per step a warp has 512 bytes of each of its R rows in
            // flight; the activations for plane j are the 16 bytes at QB bi + 32 j + p0, loaded once for the R rows.
            // Integer sums: the same result in any order.
#pragma unroll 2
            for (int64_t c = lane; c < 2 * nb; c += WARP_SIZE) {
                const int64_t bi = c / 2;
                const int     p0 = 16 * (int) (c & 1);
                int           wa4[R][4], wb4[R][4];
#pragma unroll
                for (int rr = 0; rr < R; ++rr) {
                    const int4 v4 = rr < nr ? ((const int4 *) (w + (n0 + rr) * nbw1))[c] : make_int4(0, 0, 0, 0);
                    const int  vv[4] = { v4.x, v4.y, v4.z, v4.w };
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (Q2) {
                            wa4[rr][i] = vv[i];
                            wb4[rr][i] = 0;
                        } else {
                            const int      v = vv[i];
                            const uint32_t e0 = lut[(v) & 0xFF], e1 = lut[(v >> 8) & 0xFF], e2 = lut[(v >> 16) & 0xFF],
                                           e3 = lut[(uint32_t) v >> 24];
                            const uint32_t p01 = __byte_perm(e0, e1, 0x5140);
                            const uint32_t p23 = __byte_perm(e2, e3, 0x5140);
                            wa4[rr][i]         = (int) __byte_perm(p01, p23, 0x5410);
                            wb4[rr][i]         = (int) __byte_perm(p01, p23, 0x7632);
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < (Q2 ? 4 : 5); ++j) {
#pragma unroll
                    for (int tt = 0; tt < TK; ++tt) {
                        if (tt < nt) {
                            const int4 q = *(const int4 *) (qa + (t0 + tt) * ldq + QB * bi + 32 * j + p0);
                            const int  qq[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
                            for (int rr = 0; rr < R; ++rr) {
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    const int cj = j < 4 ? (wa4[rr][i] >> (2 * j)) & 0x03030303 : wb4[rr][i] & 0x03030303;
                                    a[rr][tt]    = __dp4a(cj, qq[i], a[rr][tt]);
                                }
                            }
                        }
                    }
                }
            }
        } else
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
                for (int tt = 0; tt < TK; ++tt) {
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
            for (int tt = 0; tt < TK; ++tt) {
                if (tt < nt && rr < nr) {  // warp-uniform: only the live tokens and rows
                    for (int o = 16; o > 0; o >>= 1) {
                        a[rr][tt] += __shfl_xor_sync(0xffffffff, a[rr][tt], o);
                    }
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
                const int64_t ao = (n0 + rr) * rank;
                float         pl = 0.0f;
                if (tt < nt) {
                    const float * u = (const float *) (qa + (t0 + tt) * ldq + Kp + TTF_QROW_TAIL);
                    if (n8 <= 8 * TTF_MM_AP) {
#pragma unroll
                        for (int i = 0; i < TTF_MM_AP; ++i) {
                            if (l + 8 * i < n8) {
                                pl = __fmaf_rn(u[l + 8 * i], av[rr][i], pl);
                            }
                        }
                    } else {
                        for (int64_t k = l; k < n8; k += 8) {
                            pl = __fmaf_rn(u[k], ttf_ad_ld<BH>(mA, ao + k), pl);
                        }
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
                        rs = __fadd_rn(rs, __fmul_rn(u[k], ttf_ad_ld<BH>(mA, ao + k)));
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
            for (int tt = 1; tt < TK; ++tt) {
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
                y = ttf_rd(__fmul_rn(y, ttf_rd(rv[rr], bf16)), bf16);
            }
            if (b) {
                y = ttf_rd(__fadd_rn(y, ttf_rd(bv[rr], bf16)), bf16);
            }
            const int64_t i = t * N + n;
            if (acc) {
                y = ttf_rd(__fadd_rn(acc[i], y), bf16);
            }
            if (res) {  // a following ADD of the product (residual), as ggml's add: not rounded
                y = __fadd_rn(resv[rr], y);
            }
            out[i] = y;
        }
    }
}

template <bool Q2, int R, bool H = false, int TK = TTF_MM_TOK, bool BH = false>
static __global__ void __launch_bounds__(TTF_MM_WARPS * WARP_SIZE, TK == 1 ? (R == 1 ? 10 : 5) : (R == 1 ? 8 : 4)) ttf_mm_kernel(const uint8_t * w, const int64_t nbw1, const char * qa, const int64_t ldq,
                                     const float * beta, const void * mA, const int rank, const float * r,
                                     const float * b, const float * acc, const bool bf16, float * out, const int64_t N,
                                     const int64_t T, const int64_t Kp, const float * res = nullptr) {
    ttf_mm_body<Q2, R, H, TK, BH>(w, nbw1, qa, ldq, beta, mA, rank, r, b, acc, bf16, out, N, T, Kp, blockIdx.x, res);
}

// up to TTF_MMMAX parts (each its own weights, activations, epilogue) in one launch: part z owns blocks
// [bx0, next bx0) of grid.x
struct ttf_mmpart {
    const uint8_t * w;
    int64_t         nbw1;
    const char *    qa;
    int64_t         ldq;
    const float *   beta;
    const void *    mA;  // F32, or BF16 (the kernel's BH; one type for all parts)
    const float *   r;
    const float *   b;
    const float *   acc;
    float *         out;
    int64_t         N;
    int64_t         Kp;
    int             rank;
    int             bx0;
};
#define TTF_MMMAX 3
struct ttf_mmparts {
    ttf_mmpart p[TTF_MMMAX];
};

// register cap as the single-part kernel gets: the part selection otherwise costs occupancy
template <bool Q2, int R, bool H = false, int TK = TTF_MM_TOK, bool BH = false>
static __global__ void __launch_bounds__(TTF_MM_WARPS * WARP_SIZE, TK == 1 ? (R == 1 ? 10 : 5) : (R == 1 ? 8 : 4)) ttf_mm_multi_kernel(const ttf_mmparts P, const bool bf16, const int64_t T) {
    const int        bx = blockIdx.x;
    const int        z  = bx >= P.p[2].bx0 ? 2 : bx >= P.p[1].bx0 ? 1 : 0;  // unused parts: bx0 = INT_MAX
    const ttf_mmpart q  = z == 0 ? P.p[0] : z == 1 ? P.p[1] : P.p[2];
    ttf_mm_body<Q2, R, H, TK, BH>(q.w, q.nbw1, q.qa, q.ldq, q.beta, q.mA, q.rank, q.r, q.b, q.acc, bf16, q.out, q.N, T, q.Kp,
                          bx - q.bx0, nullptr);
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

template <bool Q2, int NT, bool BH>  // BH: mag_A in BF16 (else F32)
static __global__ void ttf_mma_kernel(const uint8_t * w, const int64_t nbw1, const char * qa, const int64_t ldq,
                                      const float * beta, const void * mA, const int rank, const float * r,
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
                y               = ttf_rd(__fadd_rn(y, ttf_rd(ttf_dot8<BH>(u, mA, n * rank, rank), bf16)), bf16);
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

static bool ttf_mm_no_half() {
    static const bool d = getenv("TTF_MM_NO_HALF") != nullptr;
    return d;
}

// res / out: the product plus res (a following ADD) written to out instead of dst (decode-sized only)
static void ttf_mm_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float * res, float * out_o) {
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
    if (!res && T >= TTF_MMA_MIN_T && ampere_mma_available(cc) && mgx * ((T + 7) / 8) >= nsm && !getenv("TTF_NO_MMA")) {
        const int64_t gx = mgx;
        auto          go = [&](auto kern, const int tb) {
            const dim3 grid((unsigned) gx, (unsigned) ((T + tb - 1) / tb));
            kern<<<grid, TTF_MM_WARPS * WARP_SIZE, 0, ctx.stream()>>>(
                (const uint8_t *) w->data, w->nb[1], (const char *) qa->data, qa->nb[1], (const float *) be->data,
                mA ? mA->data : nullptr, rank, r ? (const float *) r->data : nullptr,
                b ? (const float *) b->data : nullptr, acc ? (const float *) acc->data : nullptr, bf16,
                (float *) dst->data, N, T, Kp);
        };
        const bool bh  = ttf_adp_of(mA).bh;
        auto       gob = [&](auto kf, auto kb, const int tb) { bh ? go(kb, tb) : go(kf, tb); };
        // widest token tile that still gives >= 1 block per SM (wider tiles reuse each A fragment more)
        const int nt = gx * ((T + 31) / 32) >= nsm ? 4 : gx * ((T + 15) / 16) >= nsm ? 2 : 1;
        if (q2) {
            nt == 4 ? gob(ttf_mma_kernel<true, 4, false>, ttf_mma_kernel<true, 4, true>, 32) : nt == 2 ? gob(ttf_mma_kernel<true, 2, false>, ttf_mma_kernel<true, 2, true>, 16) : gob(ttf_mma_kernel<true, 1, false>, ttf_mma_kernel<true, 1, true>, 8);
        } else {
            nt == 4 ? gob(ttf_mma_kernel<false, 4, false>, ttf_mma_kernel<false, 4, true>, 32) : nt == 2 ? gob(ttf_mma_kernel<false, 2, false>, ttf_mma_kernel<false, 2, true>, 16) : gob(ttf_mma_kernel<false, 1, false>, ttf_mma_kernel<false, 1, true>, 8);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // grid.y: token chunks split over blocks only as far as needed to fill the GPU (~2048 blocks); each block loops
    // over its chunks
    // 4 rows per warp for prefill (each activation load feeds 4 rows) and for decode on tall matrices (4 rows'
    // loads in flight per lane, one epilogue latency per 4 rows); 1 row per warp when N is too small to fill the GPU
    static const int64_t r4_min_n = getenv("TTF_MM_R4_N") ? atoll(getenv("TTF_MM_R4_N")) : 2048;
    // (op param 1: the caller's choice, 1 or 4; ttf::Session times both on its first decode step)
    const int     hint = ggml_get_op_params_i32(dst, 1);
    const int     R  = T > TTF_MM_TOK ? 4 : hint == 1 || hint == 4 ? hint : N >= r4_min_n ? 4 : 1;
    // short rows (<= 16 chunks of 16 bytes) with 4 rows per warp: half warps (H), 2 rows each. (With 1 row per warp,
    // H's 2 rows per warp cost more than the idle lanes: their epilogues run one after the other; 340M o: 5.6 vs 4.8 us)
    const bool    H  = R == 4 && 2 * (Kp / (q2 ? 128 : 160)) <= 16 && !ttf_mm_no_half();
    const int64_t gx = (N + TTF_MM_WARPS * R - 1) / (TTF_MM_WARPS * R), nch = (T + TTF_MM_TOK - 1) / TTF_MM_TOK;
    const int64_t gy = std::max<int64_t>(1, std::min<int64_t>(nch, 2048 / gx));
    const dim3    grid((unsigned) gx, (unsigned) gy);
    const dim3 block(TTF_MM_WARPS * WARP_SIZE);
    auto       go = [&](auto kern) {
        kern<<<grid, block, 0, ctx.stream()>>>(
            (const uint8_t *) w->data, w->nb[1], (const char *) qa->data, qa->nb[1], (const float *) be->data,
            mA ? mA->data : nullptr, rank, r ? (const float *) r->data : nullptr,
            b ? (const float *) b->data : nullptr, acc ? (const float *) acc->data : nullptr, bf16,
            out_o ? out_o : (float *) dst->data, N, T, Kp, res);
    };
    const bool bh  = ttf_adp_of(mA).bh;
    auto       gob = [&](auto kf, auto kb) { bh ? go(kb) : go(kf); };
    if (q2) {
        H ? gob(ttf_mm_kernel<true, 4, true, TTF_MM_TOK, false>, ttf_mm_kernel<true, 4, true, TTF_MM_TOK, true>)
          : (R == 4 ? gob(ttf_mm_kernel<true, 4, false, TTF_MM_TOK, false>, ttf_mm_kernel<true, 4, false, TTF_MM_TOK, true>) : gob(ttf_mm_kernel<true, 1, false, TTF_MM_TOK, false>, ttf_mm_kernel<true, 1, false, TTF_MM_TOK, true>));
    } else if (T == 1) {  // one token: the kernel built for one (fewer registers, no token selection)
        H ? gob(ttf_mm_kernel<false, 4, true, 1, false>, ttf_mm_kernel<false, 4, true, 1, true>)
          : (R == 4 ? gob(ttf_mm_kernel<false, 4, false, 1, false>, ttf_mm_kernel<false, 4, false, 1, true>) : gob(ttf_mm_kernel<false, 1, false, 1, false>, ttf_mm_kernel<false, 1, false, 1, true>));
    } else {
        H ? gob(ttf_mm_kernel<false, 4, true, TTF_MM_TOK, false>, ttf_mm_kernel<false, 4, true, TTF_MM_TOK, true>)
          : (R == 4 ? gob(ttf_mm_kernel<false, 4, false, TTF_MM_TOK, false>, ttf_mm_kernel<false, 4, false, TTF_MM_TOK, true>) : gob(ttf_mm_kernel<false, 1, false, TTF_MM_TOK, false>, ttf_mm_kernel<false, 1, false, TTF_MM_TOK, true>));
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_ttf_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ttf_mm_launch(ctx, dst, nullptr, nullptr);
}

bool ggml_cuda_ttf_mm_add(ggml_backend_cuda_context & ctx, ggml_tensor * mm, const ggml_tensor * res, ggml_tensor * add) {
    if (ggml_nrows(mm->src[1]) > TTF_MM_TOK || res->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(res) || !ggml_is_contiguous(add) || ggml_nelements(res) != ggml_nelements(mm) ||
            !ggml_are_same_shape(res, add) || !ggml_is_contiguous(mm)) {
        return false;
    }
    ttf_mm_launch(ctx, mm, (const float *) res->data, (float *) add->data);
    return true;
}

// TTF_MM nodes ds[0..np) of one decode step (T <= TTF_MM_TOK, the SIMT kernel, same weight layout and rounding) in
// one launch; false if they do not qualify
bool ggml_cuda_ttf_mm_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * ds, int np) {
    if (np < 2 || np > TTF_MMMAX) {
        return false;
    }
    const ggml_type wt   = ds[0]->src[0]->type;
    const int64_t   T    = ggml_nrows(ds[0]->src[1]);
    const int32_t   bf16 = ggml_get_op_params_i32(ds[0], 0);
    if (T == 0 || T > TTF_MM_TOK) {
        return false;
    }
    ttf_mmparts P = {};
    int64_t     Ntot = 0;
    for (int p = 0; p < TTF_MMMAX; ++p) {
        P.p[p].bx0 = INT_MAX;
    }
    bool bh = false;  // the parts' adapters in BF16 (one type for all: the kernel's BH)
    for (int p = 0; p < np; ++p) {
        if (ds[p]->src[3]) {
            bh = ttf_adp_of(ds[p]->src[3]).bh;
        }
    }
    for (int p = 0; p < np; ++p) {
        const ggml_tensor * d   = ds[p];
        const ggml_tensor * w   = d->src[0];
        const ggml_tensor * qa  = d->src[1];
        const ggml_tensor * acc = d->src[6];
        if (w->type != wt || ggml_nrows(qa) != T || ggml_get_op_params_i32(d, 0) != bf16 || !ggml_is_contiguous(qa) ||
                !ggml_is_contiguous(d) || (acc && !ggml_is_contiguous(acc)) ||
                (d->src[3] && ttf_adp_of(d->src[3]).bh != bh)) {
            return false;
        }
        Ntot += w->ne[1];
    }
    static const int64_t r4_min_n = getenv("TTF_MM_R4_N") ? atoll(getenv("TTF_MM_R4_N")) : 2048;
    static const int64_t r4_tot_n = getenv("TTF_MM_R4_TOT") ? atoll(getenv("TTF_MM_R4_TOT")) : (int64_t) 1 << 40;
    // rows per warp as each part alone would use (all parts must agree), or 4 when the group is tall enough
    const bool q2h = wt == GGML_TYPE_I8;
    bool       H   = !ttf_mm_no_half();
    for (int p = 0; p < np; ++p) {
        const int64_t Kp = q2h ? 4 * ds[p]->src[0]->ne[0] : ds[p]->src[0]->ne[0];
        H = H && 2 * (Kp / (q2h ? 128 : 160)) <= 16;
    }
    int R = 0;
    for (int p = 0; p < np; ++p) {
        const int hint = ggml_get_op_params_i32(ds[p], 1);
        const int Rp   = hint == 1 || hint == 4 ? hint : ds[p]->src[0]->ne[1] >= r4_min_n || Ntot >= r4_tot_n ? 4 : 1;
        if (R != 0 && Rp != R) {
            return false;
        }
        R = Rp;
    }
    H             = H && R == 4;  // as the single-part kernel
    const bool q2 = wt == GGML_TYPE_I8;
    int        bx = 0;
    for (int p = 0; p < np; ++p) {
        const ggml_tensor * d  = ds[p];
        const ggml_tensor * w  = d->src[0];
        const ggml_tensor * qa = d->src[1];
        const ggml_tensor * mA = d->src[3];
        ttf_mmpart &        q  = P.p[p];
        q.w    = (const uint8_t *) w->data;
        q.nbw1 = w->nb[1];
        q.qa   = (const char *) qa->data;
        q.ldq  = qa->nb[1];
        q.beta = (const float *) d->src[2]->data;
        q.mA   = mA ? mA->data : nullptr;
        q.rank = mA ? (int) mA->ne[0] : 0;
        q.r    = d->src[4] ? (const float *) d->src[4]->data : nullptr;
        q.b    = d->src[5] ? (const float *) d->src[5]->data : nullptr;
        q.acc  = d->src[6] ? (const float *) d->src[6]->data : nullptr;
        q.out  = (float *) d->data;
        q.N    = w->ne[1];
        q.Kp   = q2 ? 4 * w->ne[0] : w->ne[0];
        q.bx0  = bx;
        bx += (int) ((q.N + TTF_MM_WARPS * R - 1) / (TTF_MM_WARPS * R));
    }
    const dim3 grid((unsigned) bx, 1);
    const dim3 block(TTF_MM_WARPS * WARP_SIZE);
    auto       go = [&](auto kern) { kern<<<grid, block, 0, ctx.stream()>>>(P, bf16 != 0, T); };
    auto       gob = [&](auto kf, auto kb) { bh ? go(kb) : go(kf); };
    if (q2) {
        H ? gob(ttf_mm_multi_kernel<true, 4, true, TTF_MM_TOK, false>, ttf_mm_multi_kernel<true, 4, true, TTF_MM_TOK, true>)
          : (R == 4 ? gob(ttf_mm_multi_kernel<true, 4, false, TTF_MM_TOK, false>, ttf_mm_multi_kernel<true, 4, false, TTF_MM_TOK, true>) : gob(ttf_mm_multi_kernel<true, 1, false, TTF_MM_TOK, false>, ttf_mm_multi_kernel<true, 1, false, TTF_MM_TOK, true>));
    } else if (T == 1) {
        H ? gob(ttf_mm_multi_kernel<false, 4, true, 1, false>, ttf_mm_multi_kernel<false, 4, true, 1, true>)
          : (R == 4 ? gob(ttf_mm_multi_kernel<false, 4, false, 1, false>, ttf_mm_multi_kernel<false, 4, false, 1, true>) : gob(ttf_mm_multi_kernel<false, 1, false, 1, false>, ttf_mm_multi_kernel<false, 1, false, 1, true>));
    } else {
        H ? gob(ttf_mm_multi_kernel<false, 4, true, TTF_MM_TOK, false>, ttf_mm_multi_kernel<false, 4, true, TTF_MM_TOK, true>)
          : (R == 4 ? gob(ttf_mm_multi_kernel<false, 4, false, TTF_MM_TOK, false>, ttf_mm_multi_kernel<false, 4, false, TTF_MM_TOK, true>) : gob(ttf_mm_multi_kernel<false, 1, false, TTF_MM_TOK, false>, ttf_mm_multi_kernel<false, 1, false, TTF_MM_TOK, true>));
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ---- decode attention on an f32 (or bf16 V) cache: GGML_OP_FLASH_ATTN_EXT with K F32 (the session's CUDA decode
// path; prefill keeps mul_mat + soft_max). Flash decoding: grid (Hkv, splits, T), 4 warps; a block takes one kv head's
// G = H / Hkv query heads over one range of positions, so K and V are read once per kv head. A warp takes a position
// per step: each lane holds HD / 32 elements of the K and V rows; scores by a shuffle tree; online softmax in fp32.
// Splits write (m, l, o) partials that ttf_attn_combine merges (one split: written directly).
#define TTF_ATTN_WARPS 4

template <typename TV> static __device__ __forceinline__ float ttf_attn_ld(const TV * p) { return (float) *p; }
template <> __device__ __forceinline__ float ttf_attn_ld<nv_bfloat16>(const nv_bfloat16 * p) { return __bfloat162float(*p); }

template <int HD, int G, typename TV>
static __global__ void __launch_bounds__(TTF_ATTN_WARPS * WARP_SIZE)
ttf_attn_dec_kernel(const char * q, const int64_t q_nb1, const int64_t q_nb2,     // Q [HD, T, H]
                    const char * k, const int64_t k_nb1, const int64_t k_nb2,     // K [HD, S, Hkv]
                    const char * v, const int64_t v_nb1, const int64_t v_nb2,     // V [HD, S, Hkv]
                    const half * mask, const int64_t m_nb1,                       // [S, >= T] or null
                    const int64_t S, const int H, const float scale, const int chunk,
                    float * dst,                                                  // [HD, H, T]
                    float * part) {                                               // splits > 1: [T, H, splits, HD + 2]
    constexpr int EL = HD / WARP_SIZE;
    const int hk = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const int lane = threadIdx.x % WARP_SIZE, wid = threadIdx.x / WARP_SIZE;
    const int nsp = gridDim.y;
    float qv[G][EL];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float * qr = (const float *) (q + t * q_nb1 + (int64_t) (hk * G + g) * q_nb2);
#pragma unroll
        for (int e = 0; e < EL; ++e) {
            qv[g][e] = qr[lane * EL + e] * scale;
        }
    }
    float m[G], l[G], o[G][EL];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
#pragma unroll
        for (int e = 0; e < EL; ++e) {
            o[g][e] = 0.0f;
        }
    }
    const int64_t s0 = (int64_t) sp * chunk, s1 = min(S, s0 + chunk);
    const half *  mr = mask ? (const half *) ((const char *) mask + t * m_nb1) : nullptr;
    for (int64_t s = s0 + wid; s < s1; s += TTF_ATTN_WARPS) {
        const float mk = mr ? __half2float(mr[s]) : 0.0f;
        if (mk == -INFINITY) {
            continue;  // warp-uniform
        }
        const float * kr = (const float *) (k + s * k_nb1 + hk * k_nb2) + lane * EL;
        const TV *    vr = (const TV *) (v + s * v_nb1 + hk * v_nb2) + lane * EL;
        float         kk[EL], vv[EL];
#pragma unroll
        for (int e = 0; e < EL; ++e) {
            kk[e] = kr[e];
            vv[e] = ttf_attn_ld(vr + e);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float d = 0.0f;
#pragma unroll
            for (int e = 0; e < EL; ++e) {
                d = fmaf(qv[g][e], kk[e], d);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                d += __shfl_xor_sync(0xffffffff, d, off);
            }
            const float sc = d + mk;
            const float mn = fmaxf(m[g], sc);
            const float cr = expf(m[g] - mn);  // m = -inf at first: 0
            const float p  = expf(sc - mn);
            l[g]           = l[g] * cr + p;
#pragma unroll
            for (int e = 0; e < EL; ++e) {
                o[g][e] = fmaf(p, vv[e], o[g][e] * cr);
            }
            m[g] = mn;
        }
    }
    // merge the block's warps
    __shared__ float sm[TTF_ATTN_WARPS][G][2];
    __shared__ float so[TTF_ATTN_WARPS][G][HD];
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            sm[wid][g][0] = m[g];
            sm[wid][g][1] = l[g];
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int e = 0; e < EL; ++e) {
            so[wid][g][lane * EL + e] = o[g][e];
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < G * HD; i += blockDim.x) {
        const int g = i / HD, d = i % HD;
        float     M = -INFINITY;
#pragma unroll
        for (int w = 0; w < TTF_ATTN_WARPS; ++w) {
            M = fmaxf(M, sm[w][g][0]);
        }
        float L = 0.0f, O = 0.0f;
#pragma unroll
        for (int w = 0; w < TTF_ATTN_WARPS; ++w) {
            const float f = sm[w][g][0] == -INFINITY ? 0.0f : expf(sm[w][g][0] - M);
            L += sm[w][g][1] * f;
            O += so[w][g][d] * f;
        }
        const int h = hk * G + g;
        if (nsp == 1) {
            dst[((int64_t) t * H + h) * HD + d] = L > 0.0f ? O / L : 0.0f;
        } else {
            float * pr = part + (((int64_t) t * H + h) * nsp + sp) * (HD + 2);
            pr[2 + d]  = O;
            if (d == 0) {
                pr[0] = M;
                pr[1] = L;
            }
        }
    }
}

// grid (H, T), HD threads: merge the splits' partials
template <int HD>
static __global__ void ttf_attn_combine(const float * part, const int nsp, const int H, float * dst) {
    const int     h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const float * pr = part + ((int64_t) t * H + h) * nsp * (HD + 2);
    float         M  = -INFINITY;
    for (int i = 0; i < nsp; ++i) {
        M = fmaxf(M, pr[i * (HD + 2)]);
    }
    float L = 0.0f, O = 0.0f;
    for (int i = 0; i < nsp; ++i) {
        const float mi = pr[i * (HD + 2)];
        const float f  = mi == -INFINITY ? 0.0f : expf(mi - M);
        L += pr[i * (HD + 2) + 1] * f;
        O += pr[i * (HD + 2) + 2 + d] * f;
    }
    dst[((int64_t) t * H + h) * HD + d] = L > 0.0f ? O / L : 0.0f;
}

bool ggml_cuda_ttf_attn_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (K->type != GGML_TYPE_F32 || Q->type != GGML_TYPE_F32 || (V->type != GGML_TYPE_F32 && V->type != GGML_TYPE_BF16)) {
        return false;
    }
    const int64_t HD = Q->ne[0], H = Q->ne[2], Hkv = K->ne[2];
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int64_t G = Hkv > 0 && H % Hkv == 0 ? H / Hkv : 0;
    return (HD == 64 || HD == 128 || HD == 256) && K->ne[0] == HD && V->ne[0] == HD && V->ne[2] == Hkv &&
           (G == 1 || G == 2 || G == 3 || G == 4 || G == 6 || G == 8) && Q->ne[3] == 1 && K->ne[3] == 1 && max_bias == 0.0f &&
           softcap == 0.0f && dst->src[5] == nullptr && Q->ne[1] <= 16 && Q->nb[0] == 4 && K->nb[0] == 4 &&
           V->nb[0] == ggml_type_size(V->type) && ggml_is_contiguous(dst) &&
           (!dst->src[3] || dst->src[3]->ne[2] == 1);
}

void ggml_cuda_ttf_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int64_t HD = Q->ne[0], T = Q->ne[1], H = Q->ne[2], S = K->ne[1], Hkv = K->ne[2];
    const int     G  = (int) (H / Hkv);
    float         scale;
    memcpy(&scale, dst->op_params, sizeof(float));
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    // splits: about 4 blocks per SM, at least 32 positions each (TTF_ATTN_BLOCKS_PER_SM, TTF_ATTN_MIN_CHUNK). More splits
    // won at every depth tried (the warps' position loop is latency-bound; two positions per step did not help)
    static const int64_t min_chunk = getenv("TTF_ATTN_MIN_CHUNK") ? atoll(getenv("TTF_ATTN_MIN_CHUNK")) : 32;
    static const int per_sm = getenv("TTF_ATTN_BLOCKS_PER_SM") ? atoi(getenv("TTF_ATTN_BLOCKS_PER_SM")) : 4;
    int64_t nsp = std::max<int64_t>(1, (per_sm * nsm + Hkv * T - 1) / (Hkv * T));
    nsp         = std::min<int64_t>(nsp, std::max<int64_t>(1, S / min_chunk));
    const int chunk = (int) ((S + nsp - 1) / nsp);
    nsp             = (S + chunk - 1) / chunk;
    ggml_cuda_pool_alloc<float> part(ctx.pool(), nsp > 1 ? (size_t) (T * H * nsp * (HD + 2)) : 1);
    const dim3 grid((unsigned) Hkv, (unsigned) nsp, (unsigned) T);
    auto go = [&](auto kern) {
        kern<<<grid, TTF_ATTN_WARPS * WARP_SIZE, 0, ctx.stream()>>>(
            (const char *) Q->data, Q->nb[1], Q->nb[2], (const char *) K->data, K->nb[1], K->nb[2],
            (const char *) V->data, V->nb[1], V->nb[2], mask ? (const half *) mask->data : nullptr,
            mask ? mask->nb[1] : 0, S, (int) H, scale, chunk, (float *) dst->data, part.get());
    };
    auto pick = [&](auto hd, auto tv) {
        constexpr int D = decltype(hd)::value;
        using TV        = typename decltype(tv)::type;
        switch (G) {
            case 1: go(ttf_attn_dec_kernel<D, 1, TV>); break;
            case 2: go(ttf_attn_dec_kernel<D, 2, TV>); break;
            case 3: go(ttf_attn_dec_kernel<D, 3, TV>); break;
            case 4: go(ttf_attn_dec_kernel<D, 4, TV>); break;
            case 6: go(ttf_attn_dec_kernel<D, 6, TV>); break;
            default: go(ttf_attn_dec_kernel<D, 8, TV>); break;
        }
        if (nsp > 1) {
            ttf_attn_combine<D><<<dim3((unsigned) H, (unsigned) T), D, 0, ctx.stream()>>>(part.get(), (int) nsp, (int) H,
                                                                                         (float *) dst->data);
        }
    };
    struct f32_t { using type = float; };
    struct bf16_t { using type = nv_bfloat16; };
    const bool vb = V->type == GGML_TYPE_BF16;
    if (HD == 64) {
        vb ? pick(std::integral_constant<int, 64>{}, bf16_t{}) : pick(std::integral_constant<int, 64>{}, f32_t{});
    } else if (HD == 128) {
        vb ? pick(std::integral_constant<int, 128>{}, bf16_t{}) : pick(std::integral_constant<int, 128>{}, f32_t{});
    } else {
        vb ? pick(std::integral_constant<int, 256>{}, bf16_t{}) : pick(std::integral_constant<int, 256>{}, f32_t{});
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---- TTF_I4: int-k weights (block_ttf_i4: 256 elements, groups of 64 with f32 lo / s, 4-bit codes) ----
// element value bf16(lo + s q): the product and the sum each rounded to f32, then to bf16 (nearest even), as the CPU's
// dequantize_row_ttf_i4 and the loader's expansion

static __device__ __forceinline__ float ttf_i4_val(const float lo, const float s, const uint32_t q) {
    return __bfloat162float(__float2bfloat16_rn(__fadd_rn(lo, __fmul_rn(s, (float) q))));
}

// one output row per block, the row's elements over the threads; ids I32 [n, B]
static __global__ void ttf_i4_get_rows_kernel(const block_ttf_i4 * w, const int64_t nbw1, const int32_t * ids,
                                              const int64_t nbi1, const int64_t n, const int64_t K, float * dst,
                                              const int64_t nbd1, const int64_t nbd2) {
    const int64_t i = blockIdx.x % n, b = blockIdx.x / n;
    const int64_t r = *(const int32_t *) ((const char *) ids + b * nbi1 + i * sizeof(int32_t));
    const block_ttf_i4 * wr = (const block_ttf_i4 *) ((const char *) w + r * nbw1);
    float * o = (float *) ((char *) dst + b * nbd2 + i * nbd1);
    for (int64_t e = threadIdx.x; e < K; e += blockDim.x) {
        const block_ttf_i4 & bl = wr[e / QK_TTF_I4];
        const int             k = (int) (e % QK_TTF_I4);
        const uint32_t        q = (bl.qs[k / 2] >> (4 * (k & 1))) & 15;
        o[e]                    = ttf_i4_val(bl.lo[k / 64], bl.s[k / 64], q);
    }
}

void ggml_cuda_ttf_i4_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    GGML_ASSERT(w->type == GGML_TYPE_TTF_I4 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(w->ne[2] == 1 && w->ne[3] == 1 && ids->ne[2] == 1 && ids->ne[3] == 1);
    const int64_t n = ids->ne[0], B = ids->ne[1];
    if (n * B == 0) {
        return;
    }
    ttf_i4_get_rows_kernel<<<(unsigned) (n * B), 256, 0, ctx.stream()>>>(
        (const block_ttf_i4 *) w->data, w->nb[1], (const int32_t *) ids->data, ids->nb[1], n, w->ne[0],
        (float *) dst->data, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

// y [V, T] = w [K, V]^T x [K, T], x rounded to bf16. Blocks stride over the rows, one warp per row; a lane takes 16-byte
// code chunks (32 elements of one group) and keeps NT column sums. x of the NT columns is staged in shared memory as
// bf16 (K NT 2 bytes)
#define TTF_I4_WARPS 8
template <int NT>
static __global__ void __launch_bounds__(TTF_I4_WARPS * WARP_SIZE)
ttf_i4_mm_kernel(const block_ttf_i4 * w, const int64_t nbw1, const int64_t V, const int64_t K, const float * x,
                 const int64_t nbx1, const int nt, float * dst, const int64_t nbd1) {
    extern __shared__ __align__(16) nv_bfloat16 xs16[];  // [NT][K]
    for (int t = 0; t < NT; ++t) {
        for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
            xs16[t * K + k] = t < nt ? __float2bfloat16_rn(((const float *) ((const char *) x + t * nbx1))[k])
                                     : __float2bfloat16_rn(0.0f);
        }
    }
    __syncthreads();
    const int     lane = threadIdx.x % WARP_SIZE;
    const int64_t nch  = K / 32;  // 16-byte code chunks per row
    for (int64_t row = (int64_t) blockIdx.x * TTF_I4_WARPS + threadIdx.x / WARP_SIZE; row < V;
         row += (int64_t) gridDim.x * TTF_I4_WARPS) {
        const block_ttf_i4 * wr = (const block_ttf_i4 *) ((const char *) w + row * nbw1);
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            acc[t] = 0.0f;
        }
        for (int64_t ci = lane; ci < nch; ci += WARP_SIZE) {
            const block_ttf_i4 & bl = wr[ci / 8];
            const int            c  = (int) (ci % 8);
            const uint4          qv = *(const uint4 *) (bl.qs + 16 * c);
            const float          lo = bl.lo[c / 2], s = bl.s[c / 2];
            const uint32_t       qw[4] = { qv.x, qv.y, qv.z, qv.w };
            const int64_t        k0 = ci * 32;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                float wv[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    wv[e] = ttf_i4_val(lo, s, (qw[u] >> (4 * e)) & 15);
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const uint4 xv = *(const uint4 *) (xs16 + t * K + k0 + 8 * u);
                    const nv_bfloat162 * xp = (const nv_bfloat162 *) &xv;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 xf = __bfloat1622float2(xp[e]);
                        acc[t] = fmaf(wv[2 * e], xf.x, acc[t]);
                        acc[t] = fmaf(wv[2 * e + 1], xf.y, acc[t]);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = acc[t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, off);
            }
            if (lane == 0 && t < nt) {
                *(float *) ((char *) dst + t * nbd1 + row * sizeof(float)) = v;
            }
        }
    }
}

bool ggml_cuda_ttf_i4_supported(const ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    return w->type == GGML_TYPE_TTF_I4 && x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           w->ne[2] == 1 && w->ne[3] == 1 && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) &&
           w->ne[0] % QK_TTF_I4 == 0 && w->ne[0] * 2 * 4 <= 96 * 1024;
}

void ggml_cuda_ttf_i4_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    GGML_ASSERT(ggml_cuda_ttf_i4_supported(dst));
    const int64_t K = w->ne[0], V = w->ne[1], T = x->ne[1];
    const int     nsm  = ggml_cuda_info().devices[ctx.device].nsm;
    const int     grid = (int) std::min<int64_t>((V + TTF_I4_WARPS - 1) / TTF_I4_WARPS, (int64_t) nsm * 4);
    for (int64_t t0 = 0; t0 < T; t0 += 4) {
        const int nt = (int) std::min<int64_t>(4, T - t0);
        const float * xp = (const float *) ((const char *) x->data + t0 * x->nb[1]);
        float *       dp = (float *) ((char *) dst->data + t0 * dst->nb[1]);
        auto go = [&](auto kern, int NT) {
            const size_t smem = (size_t) NT * K * sizeof(nv_bfloat16);
            if (smem > 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
            }
            kern<<<grid, TTF_I4_WARPS * WARP_SIZE, smem, ctx.stream()>>>((const block_ttf_i4 *) w->data, w->nb[1], V, K,
                                                                         xp, x->nb[1], nt, dp, dst->nb[1]);
        };
        if (nt == 1) {
            go(ttf_i4_mm_kernel<1>, 1);
        } else if (nt == 2) {
            go(ttf_i4_mm_kernel<2>, 2);
        } else {
            go(ttf_i4_mm_kernel<4>, 4);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
