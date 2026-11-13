#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// TTF linear on a TQ1_P weight W [K, N] for 1..4 tokens: dst = r * (W Q8(c x + m)) + b [+ U (D (c x + m))] [+ res];
// x [K_in <= K, ntok] (zero beyond K_in), D F16 [K, 16], U F16 [16, N] (both or neither).
// slot: a number distinct among the products that may run concurrently (the first graph node's index).
#define GGML_CUDA_TTF_MAX_GROUP 4

struct ggml_cuda_ttf_lin {
    const ggml_tensor * c = nullptr;
    const ggml_tensor * m = nullptr;
    const ggml_tensor * W = nullptr;
    const ggml_tensor * r = nullptr;
    const ggml_tensor * b = nullptr;
    const ggml_tensor * D = nullptr;
    const ggml_tensor * U = nullptr;
    const ggml_tensor * res = nullptr;   // a residual added last: dst = linear + res
    ggml_tensor       * dst = nullptr;
    int                 slot = 0;
};

bool ggml_cuda_ttf_linear_tq1p_supported(const ggml_tensor * x, const ggml_cuda_ttf_lin & l);

// G <= 4 linears reading the same x (the same K, all with or all without an adapter), in two kernels.
// false: the shapes are not covered, nothing was launched.
// glu: G == 2 and glu = silu(l[0].dst) * l[1].dst is written instead of the two products (which are not).
bool ggml_cuda_ttf_linear_tq1p(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_cuda_ttf_lin * l, int G,
    ggml_tensor * glu = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
