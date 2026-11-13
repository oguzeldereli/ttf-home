#pragma once

#include "common.cuh"

void ggml_cuda_op_ttf_quant(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// TTF_QUANT nodes on the same input (2..4) in one launch; false if they do not qualify
// TTF_MM nodes of one decode step (2..3) in one launch; false if they do not qualify
bool ggml_cuda_ttf_mm_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * ds, int np);
bool ggml_cuda_ttf_quant_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * qs, int np);
void ggml_cuda_op_ttf_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// RMS_NORM, MUL (weight row), then 1..4 TTF_QUANTs of the product (with or without adapters) as one kernel
// g / u / act: the norm's input is act(g) * u (act 1 silu, 2 relu^2), the product never written
bool ggml_cuda_ttf_norm_quant(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul,
                              ggml_tensor * const * qs, int np, const ggml_tensor * g = nullptr,
                              const ggml_tensor * u = nullptr, int act = 0);
// TTF_QUANTs (1..4) of act(g) * u (act 1 silu, 2 relu^2) in one launch, the product never written
bool ggml_cuda_ttf_act_quant(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u, int act,
                             ggml_tensor * const * qs, int np);
// GGML_OP_FLASH_ATTN_EXT with an f32 K (and f32 / bf16 V), decode-sized Q: ttf's flash-decoding kernel
bool ggml_cuda_ttf_attn_supported(const ggml_tensor * dst);
void ggml_cuda_ttf_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
