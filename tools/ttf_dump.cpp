// ttf_dump: the residual stream after every layer (the graph's "l_out-<il>" tensors) of a GGUF model on raw token ids,
// for comparing two models' representations of the same text. Writes OUT as int32 n_layer, n_tokens, n_embd, then
// fp16 [n_layer][n_tokens][n_embd]; prints the top-k next tokens at the last -k positions (k = 10).
//   ttf_dump MODEL.gguf TOKENS.u32 OUT.f16 [-ngl 99] [-ub 512] [-t threads] [-last 8]
//   ttf_dump MODEL.gguf -list LIST [...]      LIST lines: TOKENS.u32 OUT.f16 LAST (one model load; top-k to OUT.top)
//   -layers 8,16,24: only these layers, ascending (the header's n_layer counts them)
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct dump_state {
    int n_embd = 0;
    std::vector<std::vector<ggml_fp16_t>> layers;   // [il] -> tokens * n_embd, appended per ubatch
    std::vector<float> tmp;
};

static std::vector<int> g_keep;   // -layers a,b,c: dump only these layers (in this order of il); empty: all

static bool dump_cb(struct ggml_tensor * t, bool ask, void * ud) {
    int il = -1;
    if (strncmp(t->name, "l_out-", 6) != 0 || sscanf(t->name + 6, "%d", &il) != 1 ||
        (!g_keep.empty() && std::find(g_keep.begin(), g_keep.end(), il) == g_keep.end())) {
        return !ask;   // ask: not wanted; otherwise continue
    }
    if (ask) {
        return true;
    }
    auto * st = (dump_state *) ud;
    GGML_ASSERT(t->type == GGML_TYPE_F32 && t->ne[0] == st->n_embd && ggml_is_contiguous(t));
    const size_t n = ggml_nelements(t);
    st->tmp.resize(n);
    ggml_backend_tensor_get(t, st->tmp.data(), 0, n * sizeof(float));
    if ((int) st->layers.size() <= il) {
        st->layers.resize(il + 1);
    }
    auto & v = st->layers[il];
    const size_t o = v.size();
    v.resize(o + n);
    ggml_fp32_to_fp16_row(st->tmp.data(), v.data() + o, (int64_t) n);
    return true;
}

static int run_one(llama_model * model, const char * tok_path, const char * out_path, int ub, int threads, int last,
                   FILE * ftop) {
    FILE * f = fopen(tok_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", tok_path); return 1; }
    std::vector<uint32_t> toks;
    uint32_t x;
    while (fread(&x, sizeof(x), 1, f) == 1) toks.push_back(x);
    fclose(f);
    const int T = (int) toks.size();
    last = std::min(last, T);

    dump_state st;
    st.n_embd = llama_model_n_embd(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max(T, 64); cp.n_batch = T; cp.n_ubatch = std::min(ub, T);
    cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.cb_eval = dump_cb; cp.cb_eval_user_data = &st;
    llama_context * lctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_batch batch = llama_batch_init(T, 0, 1);
    batch.n_tokens = T;
    for (int j = 0; j < T; ++j) {
        batch.token[j] = (llama_token) toks[j];
        batch.pos[j] = j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
        batch.logits[j] = j >= T - last;
    }
    if (llama_decode(lctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    int L = 0;
    for (const auto & v : st.layers) L += !v.empty();
    FILE * fo = fopen(out_path, "wb");
    const int32_t hdr[3] = {L, T, st.n_embd};
    fwrite(hdr, sizeof(int32_t), 3, fo);
    for (const auto & v : st.layers) {
        if (v.empty()) continue;
        GGML_ASSERT(v.size() == (size_t) T * st.n_embd);
        fwrite(v.data(), sizeof(ggml_fp16_t), v.size(), fo);
    }
    fclose(fo);
    fprintf(stderr, "wrote %d layers x %d tokens x %d to %s\n", L, T, st.n_embd, out_path);

    char piece[256];
    for (int j = T - last; j < T; ++j) {
        const float * lg = llama_get_logits_ith(lctx, j);
        std::vector<int> idx(n_vocab);
        for (int v = 0; v < n_vocab; ++v) idx[v] = v;
        std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        fprintf(ftop, "pos %d top:", j);
        for (int r = 0; r < 10; ++r) {
            const int n = llama_token_to_piece(vocab, idx[r], piece, sizeof(piece) - 1, 0, true);
            piece[n > 0 ? n : 0] = 0;
            fprintf(ftop, " [%s] %.2f", piece, lg[idx[r]]);
        }
        fprintf(ftop, " ids:");
        for (int r = 0; r < 10; ++r) fprintf(ftop, " %d", idx[r]);
        fprintf(ftop, "\n");
    }
    fflush(ftop);
    llama_batch_free(batch);
    llama_free(lctx);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKENS.u32 OUT.f16 [-ngl 99] [-ub 512] [-t threads] [-last 8]\n"
                        "       %s MODEL.gguf -list LIST [-ngl 99] [-ub 512] [-t threads]\n", argv[0], argv[0]);
        return 1;
    }
    const bool list = std::string(argv[2]) == "-list";
    int ngl = 99, ub = 512, threads = 8, last = 8;
    for (int i = 4; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if      (a == "-ngl")  ngl     = atoi(argv[i + 1]);
        else if (a == "-ub")   ub      = atoi(argv[i + 1]);
        else if (a == "-t")    threads = atoi(argv[i + 1]);
        else if (a == "-last") last    = atoi(argv[i + 1]);
        else if (a == "-layers") {
            for (const char * p = argv[i + 1]; *p; ) {
                g_keep.push_back(atoi(p));
                const char * q = strchr(p, ',');
                if (!q) break;
                p = q + 1;
            }
        }
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    int rc = 0;
    if (!list) {
        rc = run_one(model, argv[2], argv[3], ub, threads, last, stdout);
    } else {
        FILE * fl = fopen(argv[3], "r");
        if (!fl) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
        char tp[1024], op[1024];
        int lst;
        while (fscanf(fl, "%1023s %1023s %d", tp, op, &lst) == 3) {
            std::string top = std::string(op) + ".top";
            FILE * ft = fopen(top.c_str(), "w");
            rc |= run_one(model, tp, op, ub, threads, lst, ft);
            fclose(ft);
        }
        fclose(fl);
    }
    llama_model_free(model);
    return rc;
}
