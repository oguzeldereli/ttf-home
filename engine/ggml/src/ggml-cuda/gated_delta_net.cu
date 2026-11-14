#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>

constexpr int gdn_cols_per_warp = 4;

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * state_ids,
                                     int64_t       state_row,
                                     int           l2,
                                     float         l2_eps,
                                     float         l2_scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    // with state_ids the state of sequence s is row state_ids[s] of the cache (read before this thread writes the
    // same elements, so a row read and written in place is safe)
    curr_state += (state_ids ? (int64_t) state_ids[sequence] * state_row + h_idx * S_v * S_v : state_in_offset) + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }
        if (l2) {
            // q, k <- x * rsqrt(mean(x^2) + eps) * scale (the graph's l2 norm), per head; each column segment holds
            // the whole head
            float ssq = 0.0f, ssk = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                ssq += q_reg[r] * q_reg[r];
                ssk += k_reg[r] * k_reg[r];
            }
            ssq = warp_reduce_sum<lanes_per_col>(ssq);
            ssk = warp_reduce_sum<lanes_per_col>(ssk);
            const float fq = rsqrtf(ssq / S_v + l2_eps);
            const float fk = rsqrtf(ssk / S_v + l2_eps);
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                q_reg[r] = (q_reg[r] * fq) * l2_scale;
                k_reg[r] = (k_reg[r] * fk) * l2_scale;
            }
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * state_ids, int64_t state_row,
        int l2, float l2_eps, float l2_scale, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row,
                l2, l2_eps, l2_scale);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row,
                l2, l2_eps, l2_scale);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row,
                l2, l2_eps, l2_scale);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row,
                l2, l2_eps, l2_scale);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache,
        const ggml_cuda_gated_delta_net_state_gather * gather) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const bool  l2  = gather && gather->q_raw;
    const float * q_d = l2 ? gather->q_raw : (const float *) src_q->data;
    const float * k_d = l2 ? gather->k_raw : (const float *) src_k->data;
    const float l2_eps   = l2 ? gather->l2_eps : 0.0f;
    const float l2_scale = l2 ? gather->l2_scale : 1.0f;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = gather ? gather->base : (const float *) src_state->data;
    const int32_t * s_ids = gather ? gather->ids : nullptr;
    const int64_t   s_row = gather ? gather->row : 0;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = l2 ? gather->sq1 : (int64_t) (nbq1 / sizeof(float));
    const int64_t sq2 = l2 ? gather->sq2 : (int64_t) (nbq2 / sizeof(float));
    const int64_t sq3 = l2 ? gather->sq3 : (int64_t) (nbq3 / sizeof(float));
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (gather == nullptr) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row, l2, l2_eps, l2_scale, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row, l2, l2_eps, l2_scale, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row, l2, l2_eps, l2_scale, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row, l2, l2_eps, l2_scale, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache,
        const ggml_cuda_gated_delta_net_state_gather * gather) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache, gather);
}

// the gates of a gated delta-net layer for one token, one kernel: beta = sigmoid(Wb x [+ bb]) and
// gate = softplus(Wa x + dt) * a (Wb, Wa F16 [H rows of K]; x F32 [K]). One block per output row, rows [0, H) beta,
// [H, 2H) gate; fp32 accumulation (a different summation order than mul_mat_vec_f: rounding level)
template <int WARPS>
__launch_bounds__(WARPS * WARP_SIZE, 1)
static __global__ void k_gdn_gates(const ggml_cuda_gdn_gates p) {
    const int row = blockIdx.x;
    const bool is_b = row < p.H;
    const int  h    = is_b ? row : row - p.H;
    const uint4  * w = (const uint4 *) (is_b ? (const half *) p.wb + h * p.swb : (const half *) p.wa + h * p.swa);
    const float4 * x = (const float4 *) p.x;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    float s = 0.0f;
    for (int k = tid; k < p.K / 8; k += WARPS * WARP_SIZE) {
        const uint4  wv = w[k];
        const float4 x0 = x[2 * k];
        const float4 x1 = x[2 * k + 1];
        const float2 w0 = __half22float2(*(const half2 *) &wv.x);
        const float2 w1 = __half22float2(*(const half2 *) &wv.y);
        const float2 w2 = __half22float2(*(const half2 *) &wv.z);
        const float2 w3 = __half22float2(*(const half2 *) &wv.w);
        s += w0.x * x0.x + w0.y * x0.y + w1.x * x0.z + w1.y * x0.w + w2.x * x1.x + w2.y * x1.y + w3.x * x1.z + w3.y * x1.w;
    }
    s = warp_reduce_sum(s);
    __shared__ float part[WARPS];
    if (threadIdx.x == 0) {
        part[threadIdx.y] = s;
    }
    __syncthreads();
    if (tid != 0) {
        return;
    }
    s = 0.0f;
#pragma unroll
    for (int i = 0; i < WARPS; ++i) {
        s += part[i];
    }
    if (is_b) {
        if (p.bb) {
            s += p.bb[h];
        }
        p.beta[h] = 1.0f / (1.0f + expf(-s));
    } else {
        s += p.dt[h];
        p.gate[h] = ((s > 20.0f) ? s : log1pf(expf(s))) * p.a[h];
    }
}

// F32 weights: per row the sums of mul_mat_vec_f<float, float, 1, 256> (float2 pairs strided by the block, each
// thread's fma chain, a warp reduce, then the warps' partials (the rest zero) reduced by warp 0), then the epilogues of
// its bias fusion and of the separate sigmoid / softplus-times-a kernels: the same bits as the unfused graph
static __global__ void __launch_bounds__(256, 1) k_gdn_gates_f32(const ggml_cuda_gdn_gates p) {
    constexpr int BS  = 256;
    const int     row = blockIdx.x;
    const bool    is_b = row < p.H;
    const int     h    = is_b ? row : row - p.H;
    const float2 * w   = (const float2 *) (is_b ? (const float *) p.wb + h * p.swb : (const float *) p.wa + h * p.swa);
    const float2 * y   = (const float2 *) p.x;
    const int     tid  = threadIdx.x;
    __shared__ float buf[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf[tid] = 0.0f;
    }
    __syncthreads();
    float s = 0.0f;
    for (int c = tid; c < p.K / 2; c += BS) {
        const float2 wv = w[c];
        const float2 yv = y[c];
        ggml_cuda_mad(s, wv.x, yv.x);
        ggml_cuda_mad(s, wv.y, yv.y);
    }
    s = warp_reduce_sum<WARP_SIZE>(s);
    buf[tid / WARP_SIZE] = s;
    __syncthreads();
    if (tid >= WARP_SIZE) {
        return;
    }
    s = buf[tid];
    s = warp_reduce_sum<WARP_SIZE>(s);
    if (tid != 0) {
        return;
    }
    if (is_b) {
        if (p.bb) {
            s += p.bb[h];
        }
        p.beta[h] = 1.0f / (1.0f + expf(-s));
    } else {
        s += p.dt[h];
        p.gate[h] = ((s > 20.0f) ? s : log1pf(expf(s))) * p.a[h];
    }
}

void ggml_cuda_gdn_gates_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_gates & p) {
    if (p.f32) {
        k_gdn_gates_f32<<<2 * p.H, 256, 0, ctx.stream()>>>(p);
        return;
    }
    constexpr int WARPS = 4;
    const dim3 block(WARP_SIZE, WARPS);
    k_gdn_gates<WARPS><<<2 * p.H, block, 0, ctx.stream()>>>(p);
}

// the gated output norm of a gated delta-net layer: dst = rms_norm(x, eps) * w * silu(z), one warp per row of D
// (D a multiple of 32, at most 1024); replaces rms_norm + mul (fused) and silu(z) * (fused)
static __global__ void k_gdn_out_norm(const ggml_cuda_gdn_out_norm p) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= p.rows) {
        return;
    }
    const float * x = p.x + row * p.sx;
    const float * z = p.z + row * p.sz;
    float       * d = p.dst + row * p.sd;
    constexpr int MAXV = 1024 / WARP_SIZE;
    float v[MAXV];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXV; ++i) {
        const int c = i * WARP_SIZE + threadIdx.x;
        v[i] = c < p.D ? x[c] : 0.0f;
        ss += v[i] * v[i];
    }
    ss = warp_reduce_sum(ss);
    const float sc = rsqrtf(ss / p.D + p.eps);
#pragma unroll
    for (int i = 0; i < MAXV; ++i) {
        const int c = i * WARP_SIZE + threadIdx.x;
        if (c < p.D) {
            const float zz = z[c];
            d[c] = ((v[i] * sc) * p.w[c]) * (zz / (1.0f + expf(-zz)));
        }
    }
}

void ggml_cuda_gdn_out_norm_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_out_norm & p) {
    constexpr int ROWS = 4;
    const dim3 block(WARP_SIZE, ROWS);
    k_gdn_out_norm<<<(p.rows + ROWS - 1) / ROWS, block, 0, ctx.stream()>>>(p);
}
