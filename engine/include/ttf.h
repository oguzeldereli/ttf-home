#pragma once

// ttf runtime: a .ttf file (GGUF: config, graph IR, trits, side tensors, tokenizer) run on ggml.
//   ttf::Model    the file: KV, the IR, the weights (trits repacked to TQ1_P at load, K padded to 160)
//   ttf::Session  one stream of state (KV caches by IR state id) and the graph built from the IR per call
// Spec: the .ttf format and IR (Format, Arithmetic, IR) as documented with the exporter in ttf-ternary.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../third_party/nlohmann/json.hpp"

namespace ttf {

using json = nlohmann::json;

// one ternary part (KernelTernaryLinear): trits + side tensors, ready for ggml_ttf_quant / ggml_ttf_mm
struct Part {
    int64_t       K = 0, Kp = 0, N = 0;
    ggml_tensor * w     = nullptr;  // Q2P (I8 [Kp / 4, N]) or TQ1_P [Kp, N]
    ggml_tensor * beta  = nullptr;  // F32 [1]
    ggml_tensor * mu    = nullptr;  // F32 [K] in D (in_mean)
    ggml_tensor * c     = nullptr;  // F32 [K] in D (col_scale)
    ggml_tensor * r     = nullptr;  // F32 [N] in D (row_scale)
    ggml_tensor * b     = nullptr;  // F32 [N] in D (out_bias)
    ggml_tensor * mag_B = nullptr;  // F32 [K, rank] in D (transposed at load: rank rows of K)
    ggml_tensor * mag_A = nullptr;  // F32 [rank, N] in D
};

struct Model {
    ~Model();
    // device: "cpu" or "cuda" (CUDA0; built with TTF_CUDA). Weights live in that device's buffer.
    static std::unique_ptr<Model> load(const std::string & path, std::string * err, const std::string & device = "cpu");

    std::string kv_str(const std::string & key, const std::string & dflt = "") const;

    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;  // the file's tensors, as stored
    ggml_context * wctx = nullptr;  // derived tensors: TQ1_P trits, D copies
    json           ir, config;
    bool           d_bf16 = false;  // compute dtype D
    bool           q2p    = true;   // runtime trit layout: Q2P (CPU default) or TQ1_P (CUDA default)
    std::string    device = "cpu";
    ggml_backend_t backend = nullptr;  // the device's backend, shared by sessions
    ggml_context * dctx    = nullptr;  // device copies (non-CPU)
    std::string    family;

    std::map<std::string, Part>          parts;    // by part name ("layers.0.attn.wq", "<p>.lin", ...)
    std::map<std::string, ggml_tensor *> tensors;  // by name: the file's floats (F32) and derived ones
    std::map<std::string, ggml_tensor *> head_w;   // ttf.head weights in D's type, by p
    std::vector<ggml_backend_buffer_t>   bufs;

    ggml_tensor * get(const std::string & name) const;  // nullptr when absent
};

struct StepOut {
    std::vector<float> logits;  // [T_out, n_vocab]
    int64_t            n_vocab = 0;
    int64_t            T_out   = 0;
};

struct Session {
    explicit Session(const Model & m, int n_threads = 4, int64_t max_ctx = 2048);
    ~Session();

    // run one graph of the IR on ids at positions pos (explicit), updating the state;
    // last_only: the head only on the last token
    StepOut step(const std::string & graph, const std::vector<int32_t> & ids, const std::vector<int32_t> & pos,
                 bool last_only = false);

    void reset();

    const Model &  m;
    int            n_threads;
    int64_t        max_ctx;
    ggml_backend_t backend = nullptr;
    ggml_context * sctx    = nullptr;  // state tensors
    ggml_backend_buffer_t sbuf = nullptr;
    struct Kv {
        ggml_tensor *     k = nullptr, *v = nullptr;  // F32 [head_dim * heads_kv, max_ctx]
        int64_t           hd = 0, hkv = 0;
        std::vector<bool> filled;
    };
    std::map<std::string, Kv> kv;
    std::map<std::string, ggml_tensor *> conv;  // conv stream state [len, C]
    std::map<std::string, ggml_tensor *> rec;   // delta-rule state [dv, dk, H, 1]
    ggml_gallocr_t            galloc = nullptr;
    std::vector<uint8_t>      gbuf;  // graph context memory
    double                    last_ms = 0;
    double                    prof_build = 0, prof_alloc = 0, prof_compute = 0;  // ms, summed over calls
    int                       prof_nodes = 0;                                    // last graph
    std::map<std::string, std::pair<double, int>> prof_ops;                      // TTF_PROF_OPS: ms, count
};

}  // namespace ttf
