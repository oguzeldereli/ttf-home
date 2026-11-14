#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// the initial state read from the cache rows the gather would copy: row ids[seq] of base (stride row, elements),
// in place of the GET_ROWS output (see ggml_cuda_gdn_state_gather)
struct ggml_cuda_gated_delta_net_state_gather {
    const float   * base;
    const int32_t * ids;
    int64_t         row;
    // optional: q / k read before their l2 norm (scale(rms_norm(x, l2_eps), l2_scale), ggml_cuda_gdn_l2) and
    // normalized in the kernel; null when q / k are the op's own sources
    const float   * q_raw = nullptr;
    const float   * k_raw = nullptr;
    int64_t         sq1 = 0, sq2 = 0, sq3 = 0;
    float           l2_eps = 0.0f, l2_scale = 1.0f;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache,
                                              const ggml_cuda_gated_delta_net_state_gather * gather = nullptr);

// beta = sigmoid(Wb x [+ bb]) and gate = softplus(Wa x + dt) * a for one token (ggml-cuda.cu matches the graph)
struct ggml_cuda_gdn_gates {
    const void  * wb;       // F16, or F32 (f32)
    const void  * wa;
    int64_t       swb;      // row strides in elements
    int64_t       swa;
    bool          f32;      // F32 weights: mul_mat_vec_f's sums (block 256), the separate kernels' bits
    const float * x;
    const float * bb;       // may be null
    const float * dt;
    const float * a;
    float       * beta;
    float       * gate;
    int           H;
    int           K;
};

void ggml_cuda_gdn_gates_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_gates & p);

// dst = rms_norm(x, eps) * w * silu(z) per row of D (ggml-cuda.cu matches the graph after the gated delta-net)
struct ggml_cuda_gdn_out_norm {
    const float * x;
    const float * w;
    const float * z;
    float       * dst;
    int64_t       sx, sz, sd;   // row strides in floats
    int           rows;
    int           D;
    float         eps;
};

void ggml_cuda_gdn_out_norm_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_out_norm & p);
