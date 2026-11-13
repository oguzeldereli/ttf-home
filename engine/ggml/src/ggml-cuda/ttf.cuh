#pragma once

#include "common.cuh"

void ggml_cuda_op_ttf_quant(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ttf_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_ttf_norm_quant(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul,
                              ggml_tensor * q);
