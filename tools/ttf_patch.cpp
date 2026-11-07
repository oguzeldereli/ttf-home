// ttf_patch: activation patching on a GGUF model. Runs TOKENS once per condition in PATCH, adding a delta to the residual
// stream after one layer ("l_out-<il>") at chosen positions, and prints the last position's logits of the TARGET token ids
// with their ranks and the top 5. PATCH: int32 n_cond, then per condition int32 il (-1: no patch), int32 npos,
// int32 pos[npos], float delta[npos][n_embd].
//   ttf_patch MODEL.gguf TOKENS.u32 PATCH.bin TARGET_ID[,TARGET_ID...] [-ngl 99]
//   ttf_patch MODEL.gguf -list LIST [-ngl 99]    LIST lines: TOKENS.u32 PATCH.bin TARGETS OUT.txt (one model load)
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct patch_cond {
    int il = -1;
    std::vector<int> pos;
    std::vector<float> delta;   // npos * n_embd
};

struct patch_state {
    int n_embd = 0;
    const patch_cond * cur = nullptr;
    std::vector<float> buf;
};

static bool patch_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (patch_state *) ud;
    int il = -1;
    const bool hit = st->cur && st->cur->il >= 0 && strncmp(t->name, "l_out-", 6) == 0 &&
                     sscanf(t->name + 6, "%d", &il) == 1 && il == st->cur->il;
    if (ask) {
        return hit;
    }
    if (!hit) {
        return true;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32 && t->ne[0] == st->n_embd && ggml_is_contiguous(t));
    const size_t n = ggml_nelements(t);
    st->buf.resize(n);
    ggml_backend_tensor_get(t, st->buf.data(), 0, n * sizeof(float));
    const auto & c = *st->cur;
    for (size_t k = 0; k < c.pos.size(); ++k) {
        GGML_ASSERT(c.pos[k] < t->ne[1]);
        float * row = st->buf.data() + (size_t) c.pos[k] * st->n_embd;
        const float * d = c.delta.data() + k * st->n_embd;
        for (int i = 0; i < st->n_embd; ++i) row[i] += d[i];
    }
    ggml_backend_tensor_set(t, st->buf.data(), 0, n * sizeof(float));
    return true;
}

static int run_one(llama_model * model, const char * tok_path, const char * patch_path, const char * tgt, FILE * fout) {
    std::vector<uint32_t> toks;
    {
        FILE * f = fopen(tok_path, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", tok_path); return 1; }
        uint32_t x;
        while (fread(&x, sizeof(x), 1, f) == 1) toks.push_back(x);
        fclose(f);
    }
    std::vector<int> targets;
    for (const char * p = tgt; *p; ) {
        targets.push_back(atoi(p));
        const char * q = strchr(p, ',');
        if (!q) break;
        p = q + 1;
    }
    const int T = (int) toks.size();
    patch_state st;
    st.n_embd = llama_model_n_embd(model);

    std::vector<patch_cond> conds;
    {
        FILE * f = fopen(patch_path, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", patch_path); return 1; }
        int32_t nc = 0;
        GGML_ASSERT(fread(&nc, 4, 1, f) == 1);
        conds.resize(nc);
        for (auto & c : conds) {
            int32_t il, np;
            GGML_ASSERT(fread(&il, 4, 1, f) == 1 && fread(&np, 4, 1, f) == 1);
            c.il = il;
            c.pos.resize(np);
            c.delta.resize((size_t) np * st.n_embd);
            GGML_ASSERT(fread(c.pos.data(), 4, np, f) == (size_t) np);
            GGML_ASSERT(fread(c.delta.data(), 4, c.delta.size(), f) == c.delta.size());
        }
        fclose(f);
    }

    auto cp = llama_context_default_params();
    cp.n_ctx = std::max(T, 256); cp.n_batch = T; cp.n_ubatch = T;
    cp.cb_eval = patch_cb; cp.cb_eval_user_data = &st;
    llama_context * lctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_batch batch = llama_batch_init(T, 0, 1);
    char piece[256];
    for (size_t ci = 0; ci < conds.size(); ++ci) {
        st.cur = &conds[ci];
        llama_memory_clear(llama_get_memory(lctx), true);
        batch.n_tokens = T;
        for (int j = 0; j < T; ++j) {
            batch.token[j] = (llama_token) toks[j];
            batch.pos[j] = j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
            batch.logits[j] = j == T - 1;
        }
        if (llama_decode(lctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        const float * lg = llama_get_logits_ith(lctx, T - 1);
        fprintf(fout, "cond %zu", ci);
        for (int tg : targets) {
            int rank = 0;
            for (int v = 0; v < n_vocab; ++v) rank += lg[v] > lg[tg];
            fprintf(fout, " t%d %.2f r%d", tg, lg[tg], rank);
        }
        std::vector<int> idx(n_vocab);
        for (int v = 0; v < n_vocab; ++v) idx[v] = v;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        fprintf(fout, " top:");
        for (int r = 0; r < 5; ++r) {
            const int n = llama_token_to_piece(vocab, idx[r], piece, sizeof(piece) - 1, 0, true);
            piece[n > 0 ? n : 0] = 0;
            fprintf(fout, " [%s] %.2f", piece, lg[idx[r]]);
        }
        fprintf(fout, "\n");
    }
    fflush(fout);
    st.cur = nullptr;
    llama_batch_free(batch);
    llama_free(lctx);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 4 || (std::string(argv[2]) != "-list" && argc < 5)) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKENS.u32 PATCH.bin TARGET_ID[,...] [-ngl 99]\n"
                        "       %s MODEL.gguf -list LIST [-ngl 99]\n", argv[0], argv[0]);
        return 1;
    }
    const bool list = std::string(argv[2]) == "-list";
    int ngl = 99;
    for (int i = list ? 4 : 5; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "-ngl") ngl = atoi(argv[i + 1]);
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    int rc = 0;
    if (!list) {
        rc = run_one(model, argv[2], argv[3], argv[4], stdout);
    } else {
        FILE * fl = fopen(argv[3], "r");
        if (!fl) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
        char tp[1024], pp[1024], tg[1024], op[1024];
        while (fscanf(fl, "%1023s %1023s %1023s %1023s", tp, pp, tg, op) == 4) {
            FILE * fo = fopen(op, "w");
            rc |= run_one(model, tp, pp, tg, fo);
            fclose(fo);
        }
        fclose(fl);
    }
    llama_model_free(model);
    return rc;
}
