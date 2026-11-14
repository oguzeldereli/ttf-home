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
#include <set>
#include <unordered_map>
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
    // CPU: one persistent worker pool per thread count (ggml otherwise spawns and joins its threads on every graph)
    mutable std::map<int, struct ggml_threadpool *> pools;
    void use_threads(int n_threads) const;
    ggml_context * dctx    = nullptr;  // device copies (non-CPU)
    void *         map      = nullptr;  // the file, mapped (CPU: the file's tensors point into it)
    size_t         map_size = 0;
    std::string    family;

    std::map<std::string, Part>          parts;    // by part name ("layers.0.attn.wq", "<p>.lin", ...)
    std::map<std::string, ggml_tensor *> tensors;  // by name: the file's floats (F32) and derived ones
    std::map<std::string, ggml_tensor *> head_w;   // ttf.head weights in D's type, by p
    std::vector<ggml_backend_buffer_t>   bufs;

    ggml_tensor * get(const std::string & name) const;  // nullptr when absent
};

// a host tensor: torch-order shape, f32 or i32 data
struct Tensor {
    ggml_type            type = GGML_TYPE_F32;
    std::vector<int64_t> shape;
    std::vector<float>   f;
    std::vector<int32_t> i;
    int64_t              numel() const {
        int64_t n = 1;
        for (int64_t d : shape) {
            n *= d;
        }
        return n;
    }
    static Tensor f32(std::vector<int64_t> s, std::vector<float> v) {
        Tensor t;
        t.shape = std::move(s);
        t.f     = std::move(v);
        return t;
    }
    static Tensor i32(std::vector<int64_t> s, std::vector<int32_t> v) {
        Tensor t;
        t.type  = GGML_TYPE_I32;
        t.shape = std::move(s);
        t.i     = std::move(v);
        return t;
    }
};
using Values = std::map<std::string, Tensor>;

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

    // run any graph of the IR on named inputs (torch-order shapes); returns its named outputs
    Values run(const std::string & graph, const Values & in, bool last_only = false);

    void reset();

    // prefix reuse: the recurrent state (conv + delta-rule) copied out / back, and the kv cache cut to positions < n
    // (later positions are masked; their stale values get probability exactly 0)
    std::vector<uint8_t> save_recurrent() const;
    void                 load_recurrent(const std::vector<uint8_t> & b);
    void                 truncate(int32_t n);

  private:
    int                 mm_mode = -1;  // -1 not decided, 0 fixed (env / CPU / no ternary products), 1 online
    int64_t             mm_n    = 0;   // one-token steps so far
    int                 mm_best = 0;
    std::vector<double> mm_ms[2];      // this trial's step times (by shape, 1)

  public:

    const Model &  m;
    int            n_threads;
    int64_t        max_ctx;
    // causal attention by ggml_flash_attn_ext (CPU; V cached like K). Else two mul_mats with V cached transposed
    // (CUDA). TTF_NO_FA=1: the mul_mat path on the CPU too.
    bool           fa = false;
    // flash attn: the per-row vec path also for prefill (T >= 32 otherwise tiles: ~40% faster at T 1024, but a row's
    // result then differs from decode's in the last bits). Set by --check-decode / TTF_FA_EXACT=1. Before the first call.
    bool           fa_exact = false;
    // CUDA: V cached like K; decode-sized calls (T <= 16) by ggml_flash_attn_ext on the f32 cache (ttf's flash-decoding
    // kernel), prefill by the two mul_mats (V copied transposed). TTF_NO_CUDA_FA=1: the transposed cache and mul_mats.
    bool           dfa = false;
    // CUDA decode: rows per warp of the ternary products (TTF_MM op param 1): 0 = by shape (1 below N 2048, else 4), or
    // 1 / 4 for all (the same results, bit for bit). Chosen online from the one-token steps themselves: trials of 32
    // steps alternating by shape / 1 (at step 0, 256, then every 8192), the steps between with 1 when its median step time is
    // over 5% faster. 1 wins where the GPU has power to spare (4070 Ti Super 27B: 55.9 vs 49.9 tok/s), the shape rule under a
    // power cap (4080 Laptop 27B: 33.0 vs 31.7), which a short trial at start cannot see. TTF_MM_R=0|1|4 fixes it.
    int            mm_r = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * sctx    = nullptr;  // state tensors
    ggml_backend_buffer_t sbuf = nullptr;
    struct Kv {
        ggml_tensor *     k = nullptr, *v = nullptr;  // F32 k [head_dim * heads_kv, max_ctx]; v like k (fa / dfa) or transposed [max_ctx, C]
        int64_t           hd = 0, hkv = 0;
        std::vector<bool> filled;
    };
    std::map<std::string, Kv> kv;
    std::map<std::string, ggml_tensor *> conv;  // conv stream state [len, C]
    std::map<std::string, ggml_tensor *> rec;   // delta-rule state [dv, dk, H, 1]
    // per graph (and prefill / single-token call): its own context memory and allocator, so that alternating graphs
    // neither re-plan the allocator nor share node addresses (ggml-cuda keys its CUDA graphs by the first node)
    struct GraphMem {
        std::vector<uint8_t> gbuf;
        ggml_gallocr_t       galloc = nullptr;
    };
    std::map<std::string, GraphMem> gmem;
    // a graph built once and reused (see run): its own exactly sized metadata and allocator
    struct Built {
        std::vector<uint8_t>                               gbuf;
        ggml_context *                                     ctx    = nullptr;
        ggml_cgraph *                                      gf     = nullptr;
        ggml_gallocr_t                                     galloc = nullptr;
        std::vector<std::pair<std::string, ggml_tensor *>> inputs;
        ggml_tensor *                                      mask = nullptr;
        std::vector<std::pair<int64_t, ggml_tensor *>>     vidx;  // transposed-V write indices per width
        std::vector<ggml_tensor *>                         outs;
        std::vector<int>                                   out_nd;
        int64_t                                            wbytes = 0, sbytes = 0;
        uint64_t                                           use    = 0;
        Built()                                            = default;
        Built(const Built &)                               = delete;
        Built & operator=(const Built &)                   = delete;
        ~Built();
    };
    std::map<std::string, std::unique_ptr<Built>> built;
    uint64_t                                      use_clock = 0;
    std::set<std::string>                         seen_keys;  // keys called once (a second call keeps the graph)
    struct Const {
        ggml_context *        ctx = nullptr;
        ggml_tensor *         t   = nullptr;
        ggml_backend_buffer_t buf = nullptr;
    };
    std::map<std::string, Const> consts;  // host-made constants by content
    double                    last_ms = 0;
    double                    prof_build = 0, prof_alloc = 0, prof_compute = 0;  // ms, summed over calls
    double                    prof_galloc = 0;                                   // ms, the allocator part of alloc
    int                       prof_nodes = 0;                                    // last graph
    // last graph: bytes of model weights it reads (a get_rows source counts only the rows it gathers; each tensor once)
    // and of session state (KV cache, conv / delta-rule states) it reads or writes (writes: the bytes written)
    int64_t                   last_wbytes = 0, last_sbytes = 0;
    std::map<std::string, std::pair<double, int>> prof_ops;                      // TTF_PROF_OPS: ms, count
};

// Tokenizer from a HF tokenizer.json (the file's ttf.tokenizer.json, verbatim). Implemented: BPE (merges, byte fallback,
// fuse_unk, ignore_merges), normalizers NFC / Prepend / Replace (string) / Sequence, pre-tokenizers none or Split by
// the Llama-3 / Qwen3.5 regex (GPT-4 style, matched by hand) then ByteLevel (no regex, no prefix space), the
// TemplateProcessing post-processor's special tokens (post_pre / post_post), added tokens split first.
// Anything else throws at load (never a silent mismatch).
struct Tokenizer {
    static std::unique_ptr<Tokenizer> from_json(const std::string & tokenizer_json, std::string * err = nullptr);
    // tokens as strings and ids, no BOS / EOS (HF tokenize + convert_tokens_to_ids)
    void encode(const std::string & text, std::vector<std::string> & toks, std::vector<int32_t> & ids) const;
    std::vector<int32_t> encode(const std::string & text) const;
    // VoxCPM2's post-step: a token whose text without U+2581 is >= 2 CJK ideographs (U+4E00..U+9FFF) and in the
    // vocabulary becomes its single characters (the U+2581 dropped), as mask_multichar_chinese_tokens
    std::vector<int32_t> encode_multichar_cjk(const std::string & text) const;
    int32_t id_of(const std::string & tok) const;
    // ids -> text: byte-fallback tokens become their bytes, the Replace normalizers are undone (b -> a), a leading
    // Prepend is dropped once (at the start of the text only when strip_prepend)
    std::string decode(const std::vector<int32_t> & ids, bool strip_prepend = true) const;

    std::unordered_map<std::string, int32_t> vocab;
    std::vector<std::string>                 id_to_tok;
    std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges;  // (id a, id b) -> (rank, merged id)
    std::vector<std::pair<std::string, int32_t>> added;       // content, id; matched longest first on the raw text
    std::vector<std::pair<std::string, int32_t>> added_norm;  // normalized: true ones, by normalized content, matched
                                                              // in each normalized piece (as HF's AddedVocabulary)
    std::vector<int32_t>                         special_ids; // added tokens with special: true (decode drops them)
    struct NormStep {
        std::string kind, a, b;  // Prepend: a; Replace: a -> b
    };
    std::vector<NormStep> norm;
    int32_t unk = -1;
    bool    byte_fallback = false, fuse_unk = false, ignore_merges = false;
    int     split      = 0;      // 0 none, 1 the Llama-3 regex, 2 the Qwen3.5 regex, 3 GPT-2's (ByteLevel use_regex)
    bool    byte_level = false;  // GPT-2 byte -> char map before the model; its inverse in decode
    bool    punct      = false;  // Punctuation pre-tokenizer (Contiguous), before the split
    bool    digits     = false;  // Digits pre-tokenizer (individual), after the byte-level map
    std::vector<int32_t> post_pre, post_post;  // the post-processor's special tokens around a single sequence
    int32_t byte_id[256];

    std::string normalize(const std::string & text) const;  // the normalizer steps in order

  private:
    void bpe_word(const std::string & w, std::vector<int32_t> & out) const;
    void model_piece(const std::string & normalized, std::vector<int32_t> & out) const;
};

// lm_diffusion driver (VoxCPM2): text ids -> waveform. Defaults (-1 / empty) come from the IR's driver attr.
struct SpeakParams {
    int      timesteps   = -1;
    float    cfg         = -1;
    int      min_len     = -1;
    int      max_len     = -1;
    float    temperature = -1;
    uint32_t seed        = 0;
    const std::vector<float> * noise = nullptr;  // fed noise, [patches, P, D] in order (parity runs); else mt19937
    bool     verbose     = false;
    // VoxCPM2.generate: patches capped at text tokens * badcase_ratio + 10; retry_badcase attempts (1 = no retry; the
    // VoxCPM API default is on: 3) while an attempt reaches text tokens * badcase_ratio, seed + 1 each (not with noise)
    int      retry_badcase = 3;
    float    badcase_ratio = 6.0f;
};

struct SpeakOut {
    std::vector<float> wav;          // mono, out_sample_rate
    int                sample_rate = 0;
    std::vector<float> feat;         // generated latents [patches, P, D]
    int                n_patches = 0;
    double             ms_prefill = 0, ms_lm = 0, ms_dit = 0, ms_enc = 0, ms_vae = 0;
    int                attempts = 1;  // retry_badcase
    uint32_t           seed     = 0;  // the kept attempt's seed
};

// text_ids: the tokenizer's output (the driver appends audio_start)
SpeakOut speak(Session & s, const std::vector<int32_t> & text_ids, const SpeakParams & p = {});

}  // namespace ttf
