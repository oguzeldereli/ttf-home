#pragma once

#include "common.cuh"

void ggml_cuda_op_ttf_quant(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// TTF_QUANT nodes on the same input (2..4) in one launch; false if they do not qualify
// TTF_MM nodes of one decode step (2..3) in one launch; false if they do not qualify
bool ggml_cuda_ttf_mm_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * ds, int np);
bool ggml_cuda_ttf_quant_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * qs, int np);
void ggml_cuda_op_ttf_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_ttf_norm_quant(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul,
                              ggml_tensor * q);
