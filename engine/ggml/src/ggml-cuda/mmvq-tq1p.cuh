#pragma once

#include "unary.cuh"

// TQ1_P matrix-vector product for a few tokens (ncols_dst <= 4), table decoded.
//
// A TQ1_P block holds 160 trits in 32 bytes: digit n (0..4) of byte b is element 32n + b, digit n of byte b being
// ((b * 3^n mod 256) * 3) >> 8 in {0, 1, 2} (trit = digit - 1). Digit plane n of a block is q8_1 block n of the five
// under it. The arithmetic decode costs ~14 integer ops per digit and makes the generic kernel compute-bound at about
// half the memory bandwidth; here a 256-entry shared table gives each byte's five digits as 2-bit codes, and byte
// permutes gather the codes of four bytes into the int8x4 lanes of dp4a.
//
// lut[b]: byte 0 = digits 0..3 (2 bits each, digit n at bits 2n), byte 1 = digit 4.
// For four code bytes b0..b3 (one int of the block), with T_k = lut[b_k]:
//   W_a = [T0.B0, T1.B0, T2.B0, T3.B0]   ->  digit n codes (n < 4): (W_a >> 2n) & 0x03030303
//   W_b = [T0.B1, T1.B1, T2.B1, T3.B1]   ->  digit 4 codes
// sum_k (code_k - 1) * u_k = dp4a(codes, u) - dp4a(1, u). The second term does not depend on the row: it is computed
// once per step for the TQ1P_ROWS rows of the warp and enters as the dp4a accumulator's start value, so each q8_1
// block's sum is the exact integer the generic kernel forms (the same rounding, only the float summation order differs).
//
// Each warp handles TQ1P_ROWS rows; each thread reads two ints (8 bytes) of one block per step, 4 threads per block,
// 8 blocks per warp step. The epilogue applies the TTF row scale and bias, with R > 0 a float low-rank adapter's
// up layer: + sum_j U[row, j] h[j], h = the sum over the prologue's chunks of its partial down products, and the
// residual add that follows the linear in the graph.

#define TQ1P_ROWS  4
#define TQ1P_WARPS 4

static __device__ __forceinline__ uint32_t tq1p_lut_entry(const uint32_t b) {
    uint32_t lo = 0;
    uint32_t p  = 1;
#pragma unroll
    for (int n = 0; n < 4; ++n, p *= 3) {
        lo |= ((((b * p) & 0xFF) * 3) >> 8) << (2*n);
    }
    const uint32_t hi = ((((b * 81) & 0xFF) * 3) >> 8);
    return lo | (hi << 8);
}

// the digit-4 code vector and the packed digit 0..3 code vector of one int of codes
static __device__ __forceinline__ void tq1p_decode(const uint32_t * lut, const int v, int & wa, int & wb) {
    const uint32_t t0 = lut[(v      ) & 0xFF];
    const uint32_t t1 = lut[(v >>  8) & 0xFF];
    const uint32_t t2 = lut[(v >> 16) & 0xFF];
    const uint32_t t3 = lut[(uint32_t) v >> 24];
    const uint32_t p01 = __byte_perm(t0, t1, 0x5140);
    const uint32_t p23 = __byte_perm(t2, t3, 0x5140);
    wa = (int) __byte_perm(p01, p23, 0x5410);
    wb = (int) __byte_perm(p01, p23, 0x7632);
}

#define TQ1P_MAX_GROUP GGML_CUDA_TTF_MAX_GROUP

// the prologue of the TTF linear, per linear g: xc = c x + m (x zero beyond K_in: a K-padded weight), quantized to
// q8_1 exactly as quantize_q8_1 does (zeros beyond K up to K_q), and with R > 0 a float low-rank adapter's down
// product h = D xc (D F16 [R, K]): each block (TTF_PRO_CHUNK elements of one token) writes its chunk's partial sums,
// and the last block of a (linear, token) to finish adds them up in chunk order (deterministic) into hfin.
#define TTF_PRO_CHUNK (TQ1P_WARPS*WARP_SIZE)
struct ttf_pro_lin {
    const float * c;
    const float * m;
    const half  * D;      // F16 [R, K] or null
    block_q8_1  * y;      // [ntok][K_q/32]
    float       * hp;     // [ntok][nchunk][R]
    float       * hfin;   // [ntok][R]
};
struct ttf_pro_group {
    ttf_pro_lin l[TQ1P_MAX_GROUP];
    const float * x;
    int64_t       sx;
    int           K_in;
    int           K;
    int           K_q;
    int           nchunk;
    int           ntok;
};

// per graph node slot: the prologue's arrivals per (linear, token), back to zero by the end of the kernel
#define TTF_LIN_SLOTS 8192
#define TTF_CNT_STRIDE 16
static __device__ unsigned int ttf_lin_counters[TTF_LIN_SLOTS*TTF_CNT_STRIDE];

template <int R>
__launch_bounds__(TQ1P_WARPS*WARP_SIZE, 1)
static __global__ void k_ttf_prologue(const ttf_pro_group pg, const int slot) {
    unsigned int * cnt = ttf_lin_counters + slot*TTF_CNT_STRIDE;
    const int pb  = blockIdx.x;
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int ch  = pb % pg.nchunk;
    const int tok = (pb / pg.nchunk) % pg.ntok;
    const int g   = pb / (pg.nchunk*pg.ntok);
    const ttf_pro_lin & L = pg.l[g];
    const int K = pg.K;
    const int k = ch*TTF_PRO_CHUNK + tid;
    float xi = 0.0f;
    if (k < K) {
        xi = __fadd_rn(__fmul_rn(k < pg.K_in ? pg.x[tok*pg.sx + k] : 0.0f, L.c[k]), L.m[k]);
    }
    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const int sum = warp_reduce_sum<QK8_1>((int) q);
    block_q8_1 * yt = L.y + (int64_t) tok*(pg.K_q/QK8_1);
    yt[k / QK8_1].qs[k % QK8_1] = q;
    if (k % QK8_1 == 0) {
        yt[k / QK8_1].ds = make_half2(d, d * sum);
    }
    if constexpr (R > 0) {
        static_assert(R == 16, "the butterfly below reduces 16 values");
        __shared__ float part[TQ1P_WARPS][R];
        const int warp = threadIdx.y;
        const int lane = threadIdx.x;
        float v[R];
#pragma unroll
        for (int j = 0; j < R; ++j) {
            v[j] = k < K ? xi * __half2float(L.D[(int64_t) j*K + k]) : 0.0f;
        }
        // butterfly over the warp reducing 16 values in 16 shuffles: each step a lane keeps half of its values and
        // adds its partner's copy of that half; at the end lane l holds value j = (bits 4..1 of l), summed over lanes
#pragma unroll
        for (int w = 8, off = 16; w >= 1; w /= 2, off /= 2) {
            const bool hi = lane & off;
#pragma unroll
            for (int q = 0; q < w; ++q) {
                const float keep = hi ? v[q + w] : v[q];
                const float send = hi ? v[q] : v[q + w];
                v[q] = keep + __shfl_xor_sync(0xFFFFFFFF, send, off, WARP_SIZE);
            }
        }
        v[0] += __shfl_xor_sync(0xFFFFFFFF, v[0], 1, WARP_SIZE);
        if ((lane & 1) == 0) {
            part[warp][lane >> 1] = v[0];
        }
        __syncthreads();
        if (tid < R) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < TQ1P_WARPS; ++w) {
                s += part[w][tid];
            }
            L.hp[((int64_t) tok*pg.nchunk + ch)*R + tid] = s;
        }
        __threadfence();
        __shared__ bool last;
        __syncthreads();
        unsigned int * arr = cnt + g*4 + tok;
        if (tid == 0) {
            last = atomicAdd(arr, 1u) == (unsigned) pg.nchunk - 1;
        }
        __syncthreads();
        if (last) {
            // TTF_PRO_CHUNK/R groups of R threads: group p sums chunks p, p + NP, ...; then the groups in order
            constexpr int NP = TTF_PRO_CHUNK/R;
            const int qq = tid % R;
            const int p  = tid / R;
            float s = 0.0f;
            for (int c2 = p; c2 < pg.nchunk; c2 += NP) {
                s += __ldcg(L.hp + ((int64_t) tok*pg.nchunk + c2)*R + qq);
            }
            __shared__ float red[NP][R];
            red[p][qq] = s;
            __syncthreads();
            if (tid < R) {
                float t = 0.0f;
#pragma unroll
                for (int pp = 0; pp < NP; ++pp) {
                    t += red[pp][tid];
                }
                L.hfin[tok*R + tid] = t;
            }
            if (tid == 0) {
                *arr = 0;
            }
        }
    }
}

// One product of a group sharing the input (the grouped TTF linear) or alone (the plain MUL_MAT).
struct tq1p_lin {
    const void       * W;          // TQ1_P [K, N]
    const block_q8_1 * y;          // its quantized input, ncols columns of stride scy blocks
    float            * dst;        // [N, ncols], column stride sdst
    const float      * rscale;     // [N] or null
    const float      * bias;       // [N] or null
    const float      * hfin;       // [ncols, R] the adapter's down product (R > 0)
    const half       * U;          // F16 [R, N] the adapter's up layer (R > 0)
    const float      * resid;      // [N, ncols] column stride sres, added last, or null
    int64_t            stride_w;   // row stride of W in blocks
    int                N;
    int                scy;
    int                sdst;
    int                sres;
    int                blk0;       // first thread block of this product
};

struct tq1p_group {
    tq1p_lin l[TQ1P_MAX_GROUP];
    int      G;
    int      K;
    float  * glu_dst;     // GLU mode (G == 2): glu_dst = silu(l[0]) * l[1], [N, ncols] column stride glu_sdst
    int      glu_sdst;
    int      nblocks;
};

// one warp's TQ1P_ROWS rows from row0 of product L: the result of row row0 + lane for lanes < TQ1P_ROWS (row scale,
// bias, adapter and residual applied), per column
template <int ncols_dst, int R>
static __device__ __forceinline__ void tq1p_rows(const uint32_t * lut, const tq1p_lin & L, const int K, const int row0,
        float (&res)[ncols_dst]) {
    const int nrows_x = L.N;
    const int lane = threadIdx.x;

    // the epilogue's operands, loaded now so that their latency hides behind the weight stream: lane q < R holds
    // h[j][q]; lane < TQ1P_ROWS holds its row's scale, bias and U row (as half2)
    const bool own = lane < TQ1P_ROWS && row0 + lane < nrows_x;
    const int  orow = own ? row0 + lane : 0;
    const float rs_own = own && L.rscale ? L.rscale[orow] : 1.0f;
    const float b_own  = own && L.bias   ? L.bias[orow]   : 0.0f;
    float res_own[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        res_own[j] = own && L.resid ? L.resid[j*L.sres + orow] : 0.0f;
    }
    uint4 ureg[R > 0 ? R/8 : 1];
    if constexpr (R > 0) {
#pragma unroll
        for (int q = 0; q < R/8; ++q) {
            ureg[q] = own ? ((const uint4 *) (L.U + (int64_t) orow*R))[q] : make_uint4(0, 0, 0, 0);
        }
    }

    const int nb  = K / QK_TQ1_P;
    const int sub = lane % 4;              // ints 2*sub, 2*sub + 1 of the block
    const block_tq1_p * x = (const block_tq1_p *) L.W;
    const block_q8_1  * y = L.y;
    const bool active = row0 < nrows_x;

    float hreg[ncols_dst];
    if constexpr (R > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            hreg[j] = lane < R ? L.hfin[j*R + lane] : 0.0f;
        }
    }

    float acc[ncols_dst][TQ1P_ROWS];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < TQ1P_ROWS; ++r) {
            acc[j][r] = 0.0f;
        }
    }

    for (int kb = lane / 4; active && kb < nb; kb += WARP_SIZE / 4) {
        int   u0[ncols_dst][5];
        int   u1[ncols_dst][5];
        float d [ncols_dst][5];
        int   su[ncols_dst][5];   // -sum of the 8 activations this thread holds in q8_1 block n
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int n = 0; n < 5; ++n) {
                const block_q8_1 * yb = y + j*L.scy + kb*5 + n;
                u0[j][n] = get_int_b4(yb->qs, 2*sub);     // q8_1 qs sits at byte 4 of a 36-byte block: 4-byte loads
                u1[j][n] = get_int_b4(yb->qs, 2*sub + 1);
                d [j][n] = __low2float(yb->ds);
                su[j][n] = -ggml_cuda_dp4a(0x01010101, u0[j][n], ggml_cuda_dp4a(0x01010101, u1[j][n], 0));
            }
        }

        // all rows' loads first (several in flight per thread); a row past the end re-reads the last row, unused
        int2 v[TQ1P_ROWS];
#pragma unroll
        for (int r = 0; r < TQ1P_ROWS; ++r) {
            const int row = min(row0 + r, nrows_x - 1);
            v[r] = *((const int2 *) x[(int64_t) row*L.stride_w + kb].qs + sub);
        }
#pragma unroll
        for (int r = 0; r < TQ1P_ROWS; ++r) {
            int wa0, wb0, wa1, wb1;
            tq1p_decode(lut, v[r].x, wa0, wb0);
            tq1p_decode(lut, v[r].y, wa1, wb1);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                float s = 0.0f;
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int c0 = (wa0 >> (2*n)) & 0x03030303;
                    const int c1 = (wa1 >> (2*n)) & 0x03030303;
                    s += d[j][n] * (float) ggml_cuda_dp4a(c0, u0[j][n], ggml_cuda_dp4a(c1, u1[j][n], su[j][n]));
                }
                s += d[j][4] * (float) ggml_cuda_dp4a(wb0, u0[j][4], ggml_cuda_dp4a(wb1, u1[j][4], su[j][4]));
                acc[j][r] += s;
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < TQ1P_ROWS; ++r) {
            acc[j][r] = warp_reduce_sum<WARP_SIZE>(acc[j][r]);
        }
    }

    // adapter up layer: ad[j] = sum_q U[row, q] h[j][q] for the row of lane (< TQ1P_ROWS), all lanes shuffling
    float ad[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        ad[j] = 0.0f;
    }
    if constexpr (R > 0) {
        const half2 * u = (const half2 *) ureg;
#pragma unroll
        for (int q = 0; q < R/2; ++q) {
            const float2 uq = __half22float2(u[q]);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                ad[j] += uq.x*__shfl_sync(0xFFFFFFFF, hreg[j], 2*q,     WARP_SIZE) +
                         uq.y*__shfl_sync(0xFFFFFFFF, hreg[j], 2*q + 1, WARP_SIZE);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        float a = 0.0f;
#pragma unroll
        for (int r = 0; r < TQ1P_ROWS; ++r) {
            a = r == lane ? acc[j][r] : a;
        }
        if (L.rscale) {
            a = __fmul_rn(a, rs_own);
        }
        if (L.bias) {
            a = __fadd_rn(a, b_own);
        }
        if constexpr (R > 0) {
            a = __fadd_rn(a, ad[j]);
        }
        if (L.resid) {
            a = __fadd_rn(a, res_own[j]);
        }
        res[j] = a;
    }
}

// GLU: G == 2 and the warp computes the same rows of both (gate l[0], up l[1]) and writes silu(gate) * up.
template <int ncols_dst, int R, bool GLU>
__launch_bounds__(TQ1P_WARPS*WARP_SIZE, 1)
static __global__ void mul_mat_vec_tq1_p(const tq1p_group grp) {
    __shared__ uint32_t lut[256];
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    for (int t = tid; t < 256; t += TQ1P_WARPS*WARP_SIZE) {
        lut[t] = tq1p_lut_entry(t);
    }
    __syncthreads();
    const int lane = threadIdx.x;
    const int mb = blockIdx.x;

    if constexpr (GLU) {
        const int row0 = (mb*TQ1P_WARPS + threadIdx.y)*TQ1P_ROWS;
        float rg[ncols_dst];
        float ru[ncols_dst];
        tq1p_rows<ncols_dst, R>(lut, grp.l[0], grp.K, row0, rg);
        tq1p_rows<ncols_dst, R>(lut, grp.l[1], grp.K, row0, ru);
        if (lane < TQ1P_ROWS && row0 + lane < grp.l[0].N) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                grp.glu_dst[j*grp.glu_sdst + row0 + lane] = ggml_cuda_op_silu_single(rg[j]) * ru[j];
            }
        }
        return;
    }

    int g = 0;
#pragma unroll
    for (int k = 1; k < TQ1P_MAX_GROUP; ++k) {
        if (k < grp.G && mb >= grp.l[k].blk0) {
            g = k;
        }
    }
    const tq1p_lin & L = grp.l[g];
    const int row0 = ((mb - L.blk0)*TQ1P_WARPS + threadIdx.y)*TQ1P_ROWS;
    float res[ncols_dst];
    tq1p_rows<ncols_dst, R>(lut, L, grp.K, row0, res);
    if (lane < TQ1P_ROWS && row0 + lane < L.N) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            L.dst[j*L.sdst + row0 + lane] = res[j];
        }
    }
}

static int tq1p_blocks(const int N) {
    return (N + TQ1P_WARPS*TQ1P_ROWS - 1) / (TQ1P_WARPS*TQ1P_ROWS);
}

template <int R, bool GLU>
static void tq1p_launch(const tq1p_group & grp, const int ncols, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, TQ1P_WARPS, 1);
    switch (ncols) {
        case 1: mul_mat_vec_tq1_p<1, R, GLU><<<grp.nblocks, block, 0, stream>>>(grp); break;
        case 2: mul_mat_vec_tq1_p<2, R, GLU><<<grp.nblocks, block, 0, stream>>>(grp); break;
        case 3: mul_mat_vec_tq1_p<3, R, GLU><<<grp.nblocks, block, 0, stream>>>(grp); break;
        case 4: mul_mat_vec_tq1_p<4, R, GLU><<<grp.nblocks, block, 0, stream>>>(grp); break;
        default: GGML_ABORT("tq1p: %d columns", ncols);
    }
}

// true if the table kernel handled the product
static bool mul_mat_vec_tq1_p_cuda(
        const void * vx, const void * vy, float * dst, const ggml_cuda_mm_fusion_args_device & fusion,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    if (ncols_dst < 1 || ncols_dst > 4 || fusion.gate != nullptr || fusion.gate_bias != nullptr ||
            fusion.x_scale != nullptr || fusion.shared_up != nullptr || (ncols_x % QK_TQ1_P) != 0) {
        return false;
    }
    tq1p_group grp{};
    grp.G = 1;
    grp.K = ncols_x;
    tq1p_lin & l = grp.l[0];
    l.W = vx;
    l.y = (const block_q8_1 *) vy;
    l.dst = dst;
    l.rscale = (const float *) fusion.x_rscale;
    l.bias = (const float *) fusion.x_bias;
    l.stride_w = stride_row_x;
    l.N = nrows_x;
    l.scy = stride_col_y;
    l.sdst = stride_col_dst;
    l.blk0 = 0;
    grp.nblocks = tq1p_blocks(nrows_x);
    tq1p_launch<0, false>(grp, ncols_dst, stream);
    return true;
}
