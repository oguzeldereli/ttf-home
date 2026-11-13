#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// decode (one token, one sequence) of a conv layer whose state is gathered from the recurrent cache: reads the state row
// ids[0] of rows and the new column x, writes silu(conv) to dst and the shifted state to cell (may be the same row)
struct ggml_cuda_conv_state {
    const float * rows; const int32_t * ids; int64_t row_stride;   // floats
    const float * x; const float * w; float * cell; float * dst; int C; int KC;
    bool one_block;   // dst overlaps x at another offset: read everything before writing
};
void ggml_cuda_conv_state_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_conv_state & p);
