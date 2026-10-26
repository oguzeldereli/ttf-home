// ttf_score: next-token NLL of a GGUF model on raw token ids (uint32, the trainer's .bin format), in chunks of
// CTX tokens, each scored on positions [FIRST, CTX-1) (the llama-perplexity protocol when FIRST = CTX/2), with no
// re-tokenization. Prints the mean NLL and writes one float per scored token to OUT (optional).
//   ttf_score MODEL.gguf TOKENS.bin [-c 512] [-n chunks] [-f first] [-ngl 99] [-t threads] [-o out.f32] [-u16] [-ub 1] [-s stride]
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKENS.bin [-c 512] [-n chunks] [-f first] [-ngl 99] [-t threads] [-o out]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * tok_path   = argv[2];
    int ctx = 512, n_chunks = 10, first = -1, ngl = 0, threads = 8, ub = 0, stride = 0;
    const char * out_path = nullptr;
    bool u16 = false;   // -u16: uint16 token ids (the 32k-vocab data files)
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc && std::string(argv[i]) != "-u16") break;
        std::string a = argv[i];
        if      (a == "-c")   ctx      = atoi(argv[i + 1]);
        else if (a == "-n")   n_chunks = atoi(argv[i + 1]);
        else if (a == "-f")   first    = atoi(argv[i + 1]);
        else if (a == "-ngl") ngl      = atoi(argv[i + 1]);
        else if (a == "-t")   threads  = atoi(argv[i + 1]);
        else if (a == "-ub")  ub       = atoi(argv[i + 1]);
        else if (a == "-s")   stride   = atoi(argv[i + 1]);
        else if (a == "-o")   out_path = argv[i + 1];
        else if (a == "-u16") { u16 = true; --i; }
    }
    if (first < 0) first = ctx / 2;
    if (stride <= 0) stride = ctx;     // -s 2048 -c 2049 -f 0: the eval windows (2049 tokens every 2048, 2048 scored)

    FILE * f = fopen(tok_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", tok_path); return 1; }
    std::vector<uint32_t> toks((size_t) stride * (n_chunks - 1) + ctx);
    size_t got;
    if (u16) {
        std::vector<uint16_t> t16(toks.size());
        got = fread(t16.data(), sizeof(uint16_t), t16.size(), f);
        for (size_t i = 0; i < got; ++i) toks[i] = t16[i];
    } else {
        got = fread(toks.data(), sizeof(uint32_t), toks.size(), f);
    }
    fclose(f);
    n_chunks = got < (size_t) ctx ? 0 : (int) ((got - ctx) / stride + 1);

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = ctx; cp.n_batch = ctx; cp.n_ubatch = ub > 0 ? ub : ctx;   // -ub 1: the decode (matrix-vector) path cp.n_threads = threads; cp.n_threads_batch = threads;
    llama_context * lctx = llama_init_from_model(model, cp);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    FILE * fo = out_path ? fopen(out_path, "wb") : nullptr;
    double nll = 0.0; long cnt = 0;
    llama_batch batch = llama_batch_init(ctx, 0, 1);
    for (int c = 0; c < n_chunks; ++c) {
        llama_memory_clear(llama_get_memory(lctx), true);
        batch.n_tokens = ctx;
        for (int j = 0; j < ctx; ++j) {
            batch.token[j] = (llama_token) toks[(size_t) c * stride + j];
            batch.pos[j] = j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
            batch.logits[j] = j >= first && j < ctx - 1;
        }
        if (llama_decode(lctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        double cn = 0.0;
        for (int j = first; j < ctx - 1; ++j) {
            const float * lg = llama_get_logits_ith(lctx, j);
            double mx = -1e30; for (int v = 0; v < n_vocab; ++v) mx = lg[v] > mx ? lg[v] : mx;
            double s = 0.0;    for (int v = 0; v < n_vocab; ++v) s += exp(lg[v] - mx);
            const float l = (float) (log(s) + mx - lg[toks[(size_t) c * stride + j + 1]]);
            cn += l;
            if (fo) fwrite(&l, sizeof(float), 1, fo);
        }
        nll += cn; cnt += ctx - 1 - first;
        fprintf(stderr, "chunk %d: %.4f\n", c, cn / (ctx - 1 - first));
    }
    if (fo) fclose(fo);
    printf("{\"model\": \"%s\", \"chunks\": %d, \"ctx\": %d, \"nll\": %.6f, \"ppl\": %.4f}\n", model_path, n_chunks, ctx, nll / cnt, exp(nll / cnt));
    llama_batch_free(batch);
    llama_free(lctx);
    llama_model_free(model);
    return 0;
}
