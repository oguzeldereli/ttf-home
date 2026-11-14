#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval || (val == maxval && col >= 0 && (argmax < 0 || col < argmax))) {  // ttf: ties to the first index
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval || (val == maxval && col >= 0 && (argmax < 0 || col < argmax))) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax < 0 ? 0 : argmax;  // ttf: all -inf / NaN: the first (as argmax_first)
    }
}

// ttf: one long row (a decode step's logits, 32k-256k) over many blocks: each block the first index of the max of its
// chunk, then one block over the partials (ties to the first index throughout, as the one-block kernel). The one-block
// kernel reads a 248k row on one SM (~24 us on a 4080 Laptop for 131k); this ~4 us.
static __device__ __forceinline__ void argmax_merge(float & m, int & a, const float v, const int c) {
    if (v > m || (v == m && c >= 0 && (a < 0 || c < a))) {
        m = v;
        a = c;
    }
}

template <int NT>
static __device__ __forceinline__ void argmax_block(float & m, int & a) {
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        argmax_merge(m, a, __shfl_xor_sync(0xFFFFFFFF, m, offset, WARP_SIZE), __shfl_xor_sync(0xFFFFFFFF, a, offset, WARP_SIZE));
    }
    constexpr int NW = NT / WARP_SIZE;
    if (NW > 1) {
        __shared__ float sm[NW];
        __shared__ int   sa[NW];
        const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
        if (lane == 0) {
            sm[warp] = m;
            sa[warp] = a;
        }
        __syncthreads();
        if (warp == 0) {
            m = lane < NW ? sm[lane] : -FLT_MAX;
            a = lane < NW ? sa[lane] : -1;
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                argmax_merge(m, a, __shfl_xor_sync(0xFFFFFFFF, m, offset, WARP_SIZE), __shfl_xor_sync(0xFFFFFFFF, a, offset, WARP_SIZE));
            }
        }
    }
}

template <int NT>
static __global__ void argmax_f32_part(const float * __restrict__ x, float * __restrict__ pm, int * __restrict__ pa,
                                       const int64_t ncols, const int64_t chunk) {
    const int64_t c0 = (int64_t) blockIdx.x * chunk;
    const int64_t c1 = min(ncols, c0 + chunk);
    float m = -FLT_MAX;
    int   a = -1;
    for (int64_t c = c0 + threadIdx.x; c < c1; c += NT) {
        const float v = x[c];
        if (v > m) {
            m = v;
            a = (int) c;
        }
    }
    argmax_block<NT>(m, a);
    if (threadIdx.x == 0) {
        pm[blockIdx.x] = m;
        pa[blockIdx.x] = a;
    }
}

template <int NT>
static __global__ void argmax_f32_final(const float * __restrict__ pm, const int * __restrict__ pa, int32_t * __restrict__ dst,
                                        const int n) {
    float m = -FLT_MAX;
    int   a = -1;
    for (int i = threadIdx.x; i < n; i += NT) {
        argmax_merge(m, a, pm[i], pa[i]);
    }
    argmax_block<NT>(m, a);
    if (threadIdx.x == 0) {
        dst[0] = a < 0 ? 0 : a;  // ttf: all -inf / NaN: the first (as argmax_first; -1 would index row -1)
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    if (nrows == 1 && ne00 >= 16384 && !getenv("GGML_CUDA_ARGMAX_ONE_BLOCK")) {  // ttf: one long row over many blocks
        constexpr int NT    = 256;
        const int64_t nblk  = std::min<int64_t>(128, (ne00 + 2047) / 2048);
        const int64_t chunk = (ne00 + nblk - 1) / nblk;
        ggml_cuda_pool_alloc<float> pm(ctx.pool(), nblk);
        ggml_cuda_pool_alloc<int>   pa(ctx.pool(), nblk);
        argmax_f32_part<NT><<<nblk, NT, 0, stream>>>(src0_d, pm.get(), pa.get(), ne00, chunk);
        argmax_f32_final<128><<<1, 128, 0, stream>>>(pm.get(), pa.get(), dst_d, (int) nblk);
        return;
    }

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
