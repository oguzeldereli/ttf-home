// ttf: GGML_OP_TTF_TOPK on CUDA, the k largest of one row in the host sampler's order (value descending, ties by the
// lower index), deterministic. Each element becomes a unique 64-bit key (ordered value bits << 32 | ~index), so the
// order is a plain unsigned one.
#include "ttf-topk.cuh"

static __device__ __forceinline__ uint64_t ttf_tk_key(float v, uint32_t idx) {
    v                   = v == 0.0f ? 0.0f : v;  // -0 as +0 (the host compares by value)
    const uint32_t bits = __float_as_uint(v);
    const uint32_t ord  = bits ^ ((uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000u);
    return ((uint64_t) ord << 32) | (uint64_t) (0xFFFFFFFFu - idx);
}

static __device__ __forceinline__ float ttf_tk_val(uint64_t key) {
    const uint32_t ord  = (uint32_t) (key >> 32);
    const uint32_t bits = (ord & 0x80000000u) ? (ord ^ 0x80000000u) : ~ord;
    return __uint_as_float(bits);
}

// threshold: T = the k-th largest of nb >= k block maxima (k distinct keys >= T exist, so the top k are all >= T);
// every key >= T compacted (atomics: any order, the keys are unique) and only those sorted. A flood of keys >= T (more
// than TTF_TK_CAP: many equal values) takes the exact slow path (k rounds over the row in one block)
constexpr int TTF_TK_NB  = 256;   // at most this many block maxima (>= k)
constexpr int TTF_TK_CAP = 2048;  // candidates sorted in shared memory

struct ttf_tk_state {
    uint64_t thr;
    int      cnt;
};

static __device__ __forceinline__ uint64_t ttf_tk_umax(uint64_t a, uint64_t b) {
    return a > b ? a : b;
}

// block reduce (max) over blockDim.x threads (a multiple of 32, <= 1024); the result in every thread
static __device__ uint64_t ttf_tk_block_max(uint64_t m) {
    __shared__ uint64_t w[32];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = ttf_tk_umax(m, __shfl_xor_sync(0xFFFFFFFF, m, off, 32));
    }
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32, nw = blockDim.x / 32;
    __syncthreads();
    if (lane == 0) {
        w[wid] = m;
    }
    __syncthreads();
    m = lane < nw ? w[lane] : 0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = ttf_tk_umax(m, __shfl_xor_sync(0xFFFFFFFF, m, off, 32));
    }
    return m;
}

// bitonic sort, descending, of sk[0, P) (P a power of 2), by all threads of the block
static __device__ void ttf_tk_sort(uint64_t * sk, const int P) {
    for (int size = 2; size <= P; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int t = threadIdx.x; t < P / 2; t += blockDim.x) {
                const int      i = 2 * t - (t & (stride - 1)), j = i + stride;
                const uint64_t a = sk[i], b = sk[j];
                if ((i & size) == 0 ? a < b : a > b) {
                    sk[i] = b;
                    sk[j] = a;
                }
            }
            __syncthreads();
        }
    }
}

static __global__ void ttf_tk_max(const float * __restrict__ x, const int64_t n, const int64_t chunk, uint64_t * __restrict__ mx) {
    const int64_t c0 = (int64_t) blockIdx.x * chunk, c1 = min(n, c0 + chunk);
    uint64_t      m  = 0;
    for (int64_t i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
        m = ttf_tk_umax(m, ttf_tk_key(x[i], (uint32_t) i));
    }
    m = ttf_tk_block_max(m);
    if (threadIdx.x == 0) {
        mx[blockIdx.x] = m;
    }
}

static __global__ void ttf_tk_thr(const uint64_t * __restrict__ mx, const int nb, const int k, ttf_tk_state * st) {
    __shared__ uint64_t sk[TTF_TK_NB];
    for (int i = threadIdx.x; i < TTF_TK_NB; i += blockDim.x) {
        sk[i] = i < nb ? mx[i] : 0;
    }
    __syncthreads();
    ttf_tk_sort(sk, TTF_TK_NB);
    if (threadIdx.x == 0) {
        st->thr = sk[k - 1];
        st->cnt = 0;
    }
}

static __global__ void ttf_tk_pick(const float * __restrict__ x, const int64_t n, ttf_tk_state * st, uint64_t * __restrict__ cand) {
    const uint64_t T = st->thr;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const uint64_t key = ttf_tk_key(x[i], (uint32_t) i);
        if (key >= T) {
            const int c = atomicAdd(&st->cnt, 1);
            if (c < TTF_TK_CAP) {
                cand[c] = key;
            }
        }
    }
}

static __global__ void __launch_bounds__(1024) ttf_tk_final(const float * __restrict__ x, const int64_t n, const int k,
                                                            const ttf_tk_state * st, const uint64_t * __restrict__ cand,
                                                            float * __restrict__ dst) {
    __shared__ uint64_t sk[TTF_TK_CAP];
    const int c = st->cnt;
    if (c <= TTF_TK_CAP) {
        int P = 1;
        while (P < c) {
            P <<= 1;
        }
        for (int i = threadIdx.x; i < P; i += blockDim.x) {
            sk[i] = i < c ? cand[i] : 0;
        }
        __syncthreads();
        ttf_tk_sort(sk, P);
        for (int r = threadIdx.x; r < k; r += blockDim.x) {
            dst[r]     = ttf_tk_val(sk[r]);
            dst[k + r] = (float) (0xFFFFFFFFu - (uint32_t) sk[r]);
        }
        return;
    }
    uint64_t last = ~(uint64_t) 0;  // the flood: k rounds, each the largest key below the last
    for (int r = 0; r < k; ++r) {
        uint64_t m = 0;
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const uint64_t key = ttf_tk_key(x[i], (uint32_t) i);
            m                  = key < last ? ttf_tk_umax(m, key) : m;
        }
        m    = ttf_tk_block_max(m);
        last = m;
        if (threadIdx.x == 0) {
            dst[r]     = ttf_tk_val(m);
            dst[k + r] = (float) (0xFFFFFFFFu - (uint32_t) m);
        }
    }
}

void ggml_cuda_ttf_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    GGML_ASSERT(a->type == GGML_TYPE_F32 && ggml_is_contiguous(a) && ggml_nrows(a) == 1);
    const int64_t n = a->ne[0];
    const int     k = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(k > 0 && k <= 256 && k < n);
    cudaStream_t  stream = ctx.stream();
    const float * x      = (const float *) a->data;
    // nb block maxima, k <= nb <= 256 (chunks of ~1024 where the row allows)
    const int64_t nb     = std::min<int64_t>(n, std::min<int64_t>(TTF_TK_NB, std::max<int64_t>(k, (n + 1023) / 1024)));
    const int64_t chunk  = (n + nb - 1) / nb;
    ggml_cuda_pool_alloc<uint64_t>     mx(ctx.pool(), nb), cand(ctx.pool(), TTF_TK_CAP);
    ggml_cuda_pool_alloc<ttf_tk_state> st(ctx.pool(), 1);
    ttf_tk_max<<<nb, 256, 0, stream>>>(x, n, chunk, mx.get());
    ttf_tk_thr<<<1, 128, 0, stream>>>(mx.get(), (int) nb, k, st.get());
    ttf_tk_pick<<<(int) std::min<int64_t>(256, (n + 1023) / 1024), 256, 0, stream>>>(x, n, st.get(), cand.get());
    ttf_tk_final<<<1, 1024, 0, stream>>>(x, n, k, st.get(), cand.get(), (float *) dst->data);
}
