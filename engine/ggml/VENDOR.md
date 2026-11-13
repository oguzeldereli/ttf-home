# Vendored ggml

Copied from our llama.cpp fork, branch `ttf-q38-speed` at 4ef45134d (upstream llama.cpp ca89968a2 + our ggml commits:
TQ2_0 / TQ1_0 CUDA matvec and MMQ, TQ1_P (160 trits in 32 bytes), the TQ1_P table-decode matvec and fused TTF linear
(df70b4200), DeltaNet decode fusions (99e2016ab, 4eaa6c06b)) on 13 Nov 2026, `git archive 4ef45134d ggml`.

Removed backends we do not build: cann, et, hexagon, musa, opencl, openvino, rpc, sycl, virtgpu, webgpu, zdnn, zendnn.
Kept: cpu, cuda, hip, metal, vulkan, blas.

ggml is MIT licensed (LICENSE.llama.cpp, the llama.cpp repository's licence file, which covers ggml). Our changes in this
tree are listed in the engine's git history; new TTF ops: GGML_OP_TTF_QUANT, GGML_OP_TTF_MM.
