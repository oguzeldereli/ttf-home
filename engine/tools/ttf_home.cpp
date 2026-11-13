// ttf-home: run any .ttf model. The IR's driver decides what the model does:
//   lm            text in, text out (greedy, or sampled with --temp)
//   lm_diffusion  text in, speech out (a 16-bit WAV)
//
//   ttf-home MODEL "text" [-n N] [--temp T] [--top-k K] [--seed S]      text model: the continuation on stdout
//   ttf-home MODEL "text" [-o out.wav] [--seed S] [--timesteps N] [--cfg F] [--max-len N]   speech model
//   ttf-home MODEL --list FILE           speech model: one load, then each line "out.wav<TAB>text[<TAB>seed]"
//   ttf-home MODEL --bench [--pp 512] [--tg 128] [--reps 3]
//            text model: prefill tok/s at T = pp, decode tok/s at batch 1 over tg tokens, and the bytes one decode
//            token reads (weights, state) with the bandwidth that implies; speech model: real-time factor
//   ttf-home MODEL --tokenize            stdin lines -> comma-separated ids (no weights loaded)
//   common: --ids 1,2,3 (in place of text), --device cpu|cuda (default cuda when built with it), --threads N,
//           --ctx N, -v
//   checks: --dump FILE (prefill logits, f32 [T, n_vocab]), --check-decode N (decode vs prefill on the last N),
//           --graph NAME --in NAME=FILE ... [--out DIR] (one call of one graph), --noise FILE / --feat FILE (speech)
#include "ttf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct Args {
    std::string                        model, text, list, out = "out.wav", device, dump, graph, out_dir = ".";
    std::string                        noise, feat;
    std::vector<int32_t>               ids;
    std::map<std::string, std::string> gin;
    int                                n_gen = 128, threads = 4, top_k = 40, check_dec = 0, pp = 512, tg = 128, reps = 3;
    int64_t                            ctx = 0;
    float                              temp = 0.0f;
    bool                               bench = false, tokenize = false, verbose = false, ids_given = false;
    ttf::SpeakParams                   sp;
    uint32_t                           seed = 0;
};

std::vector<int32_t> parse_ids(const std::string & s) {
    std::vector<int32_t> out;
    std::stringstream    ss(s);
    for (std::string tok; std::getline(ss, tok, ',');) {
        if (!tok.empty()) {
            out.push_back((int32_t) std::strtol(tok.c_str(), nullptr, 10));
        }
    }
    return out;
}

bool write_wav(const std::string & path, const std::vector<float> & x, int sr) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const uint32_t n = (uint32_t) x.size(), data = n * 2, riff = 36 + data, fmt = 16, rate = (uint32_t) sr, br = rate * 2;
    const uint16_t pcm = 1, ch = 1, ba = 2, bits = 16;
    std::fwrite("RIFF", 1, 4, f);
    std::fwrite(&riff, 4, 1, f);
    std::fwrite("WAVEfmt ", 1, 8, f);
    std::fwrite(&fmt, 4, 1, f);
    std::fwrite(&pcm, 2, 1, f);
    std::fwrite(&ch, 2, 1, f);
    std::fwrite(&rate, 4, 1, f);
    std::fwrite(&br, 4, 1, f);
    std::fwrite(&ba, 2, 1, f);
    std::fwrite(&bits, 2, 1, f);
    std::fwrite("data", 1, 4, f);
    std::fwrite(&data, 4, 1, f);
    for (float v : x) {
        v           = v > 1.0f ? 1.0f : v < -1.0f ? -1.0f : v;
        int16_t s16 = (int16_t) std::lrintf(v * 32767.0f);
        std::fwrite(&s16, 2, 1, f);
    }
    std::fclose(f);
    return true;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the tokenizer config's special token (a plain string or an AddedToken object)
std::string special(const json & cfg, const char * key) {
    if (!cfg.contains(key) || cfg.at(key).is_null()) {
        return "";
    }
    const json & v = cfg.at(key);
    return v.is_string() ? v.get<std::string>() : v.value("content", std::string());
}

struct Text {
    std::unique_ptr<ttf::Tokenizer> tok;
    std::string                     post;    // the driver's tokenizer_post
    int32_t                         bos = -1;
    std::vector<int32_t>            eos;
};

// tokenizer + BOS / EOS from the file (IR driver attr eos first, else the tokenizer config's eos_token)
Text load_text(const std::string & tj, const std::string & cfgs, const json & drv) {
    Text        t;
    std::string err;
    t.tok = ttf::Tokenizer::from_json(tj, &err);
    if (!t.tok) {
        throw std::runtime_error("tokenizer: " + err + " (give token ids with --ids)");
    }
    t.post = drv.value("attr", json::object()).value("tokenizer_post", std::string());
    if (t.post != "" && t.post != "mask_multichar_cjk") {
        throw std::runtime_error("tokenizer_post " + t.post + ": not implemented");
    }
    const json cfg = cfgs.empty() ? json::object() : json::parse(cfgs);
    if (cfg.value("add_bos_token", false)) {
        t.bos = t.tok->id_of(special(cfg, "bos_token"));
    }
    for (const auto & e : drv.value("attr", json::object()).value("eos", json::array())) {
        t.eos.push_back(e.get<int32_t>());
    }
    if (t.eos.empty() && t.tok->id_of(special(cfg, "eos_token")) >= 0) {
        t.eos.push_back(t.tok->id_of(special(cfg, "eos_token")));
    }
    return t;
}

std::vector<int32_t> encode(const Text & t, const std::string & s, bool bos) {
    std::vector<int32_t> ids = t.post == "mask_multichar_cjk" ? t.tok->encode_multichar_cjk(s) : t.tok->encode(s);
    if (bos && t.bos >= 0) {
        ids.insert(ids.begin(), t.bos);
    }
    return ids;
}

int32_t pick(const float * l, int64_t n, float temp, int top_k, std::mt19937 & rng) {
    if (temp <= 0.0f) {
        return (int32_t) (std::max_element(l, l + n) - l);
    }
    std::vector<std::pair<float, int32_t>> c(n);
    for (int64_t j = 0; j < n; ++j) {
        c[j] = { l[j], (int32_t) j };
    }
    const int k = top_k > 0 ? std::min<int64_t>(top_k, n) : n;
    std::partial_sort(c.begin(), c.begin() + k, c.end(), [](auto & a, auto & b) { return a.first > b.first; });
    std::vector<double> p(k);
    for (int i = 0; i < k; ++i) {
        p[i] = std::exp((c[i].first - c[0].first) / temp);
    }
    std::discrete_distribution<int> d(p.begin(), p.end());
    return c[d(rng)].second;
}

std::vector<int32_t> positions(int64_t from, int64_t n) {
    std::vector<int32_t> p(n);
    for (int64_t i = 0; i < n; ++i) {
        p[i] = (int32_t) (from + i);
    }
    return p;
}

void print_ops(const ttf::Session & s, int n) {
    std::vector<std::pair<double, std::string>> ops;
    double                                      tot = 0;
    for (auto & [k, v] : s.prof_ops) {
        ops.emplace_back(v.first, k);
        tot += v.first;
    }
    std::sort(ops.rbegin(), ops.rend());
    for (size_t i = 0; i < ops.size() && i < 25; ++i) {
        std::fprintf(stderr, "  %7.3f ms/call %5.1f%%  %s\n", ops[i].first / n, 100 * ops[i].first / tot, ops[i].second.c_str());
    }
}

// ---- one graph on raw inputs (checks)
int run_graph(const ttf::Model & m, const Args & a) {
    ttf::Values in;
    for (const auto & spec : m.ir.at("graphs").at(a.graph).at("inputs")) {
        const std::string nm = spec.at("name");
        if (!a.gin.count(nm)) {
            std::fprintf(stderr, "--in %s=FILE missing\n", nm.c_str());
            return 1;
        }
        std::ifstream        f(a.gin.at(nm), std::ios::binary | std::ios::ate);
        const int64_t        nb  = f.tellg();
        const bool           i32 = spec.value("dtype", std::string("f32")) == "i32";
        int64_t              known = 1, dyn = -1;
        std::vector<int64_t> sh;
        for (const auto & d : spec.at("shape")) {
            if (d.is_number()) {
                sh.push_back(d.get<int64_t>());
                known *= sh.back();
            } else {
                dyn = (int64_t) sh.size();
                sh.push_back(0);
            }
        }
        const int64_t n = nb / 4;
        if (dyn >= 0) {
            sh[dyn] = n / known;
        }
        f.seekg(0);
        ttf::Tensor t;
        t.shape = sh;
        if (i32) {
            t.type = GGML_TYPE_I32;
            t.i.resize(n);
            f.read((char *) t.i.data(), nb);
        } else {
            t.f.resize(n);
            f.read((char *) t.f.data(), nb);
        }
        if (t.numel() != n) {
            std::fprintf(stderr, "%s: %lld values do not fit the shape\n", nm.c_str(), (long long) n);
            return 1;
        }
        in[nm] = std::move(t);
    }
    ttf::Session s(m, a.threads, a.ctx ? a.ctx : 2048);
    ttf::Values  o = s.run(a.graph, in, false);
    std::fprintf(stderr, "%s: %.1f ms\n", a.graph.c_str(), s.last_ms);
    for (auto & [nm, t] : o) {
        const std::string fn = a.out_dir + "/" + nm + ".bin";
        FILE *            f  = std::fopen(fn.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", fn.c_str());
            return 1;
        }
        if (t.type == GGML_TYPE_I32) {
            std::fwrite(t.i.data(), 4, t.i.size(), f);
        } else {
            std::fwrite(t.f.data(), 4, t.f.size(), f);
        }
        std::fclose(f);
        std::string shs;
        for (int64_t d : t.shape) {
            shs += std::to_string(d) + " ";
        }
        std::fprintf(stderr, "out %s [ %s] -> %s\n", nm.c_str(), shs.c_str(), fn.c_str());
    }
    return 0;
}

// ---- lm driver
int run_lm(const ttf::Model & m, const Args & a) {
    const std::string graph = m.ir.at("driver").at("graphs").at("step");
    const int64_t     ctx   = a.ctx ? a.ctx : std::max<int64_t>(2048, a.pp + a.tg + 8);
    ttf::Session      s(m, a.threads, ctx);
    if (a.bench) {
        const int64_t        nv = m.config.value("vocab_size", (int64_t) 1000);
        std::mt19937         rng(1);
        std::vector<int32_t> ids(a.pp);
        for (auto & v : ids) {
            v = (int32_t) (rng() % std::min<int64_t>(nv, 32000));
        }
        // prefill: one warm-up, then reps (state reset each time); the head on the last token only
        s.step(graph, ids, positions(0, a.pp), true);
        double best = 1e30;
        for (int r = 0; r < a.reps; ++r) {
            s.reset();
            const double t0 = now_ms();
            s.step(graph, ids, positions(0, a.pp), true);
            best = std::min(best, now_ms() - t0);
        }
        std::printf("prefill  T %5d: %8.1f ms  %9.1f tok/s\n", a.pp, best, 1000.0 * a.pp / best);
        // decode: after a 16-token prompt, tg single-token steps (greedy on random ids is irrelevant: fixed ids)
        double dbest = 1e30;
        for (int r = 0; r < a.reps; ++r) {
            s.reset();
            s.step(graph, std::vector<int32_t>(ids.begin(), ids.begin() + 16), positions(0, 16), true);
            s.prof_build = s.prof_alloc = s.prof_compute = 0;
            const double t0 = now_ms();
            for (int i = 0; i < a.tg; ++i) {
                s.step(graph, { ids[16 + i % (a.pp - 16)] }, { (int32_t) (16 + i) }, true);
            }
            dbest = std::min(dbest, (now_ms() - t0) / a.tg);
        }
        const double gb = (double) (s.last_wbytes + s.last_sbytes) / 1e9;
        std::printf("decode   B 1  : %8.2f ms  %9.1f tok/s  (%d tokens)\n", dbest, 1000.0 / dbest, a.tg);
        std::printf("per token: weights %.1f MB, state %.1f MB at position %d; %.1f GB/s\n", s.last_wbytes / 1e6,
                    s.last_sbytes / 1e6, 16 + a.tg - 1, gb / (dbest / 1000.0));
        std::printf("decode split (last rep, per token): build %.2f ms, alloc %.2f ms, compute %.2f ms; %d nodes\n",
                    s.prof_build / a.tg, s.prof_alloc / a.tg, s.prof_compute / a.tg, s.prof_nodes);
        if (!s.prof_ops.empty()) {
            print_ops(s, a.tg);
        }
        return 0;
    }
    std::vector<int32_t> ids = a.ids;
    std::unique_ptr<Text> txt;
    std::string           err;
    const std::string     tj = m.kv_str("ttf.tokenizer.json");
    if (!a.ids_given || !tj.empty()) {
        try {
            txt = std::make_unique<Text>(load_text(tj, m.kv_str("ttf.tokenizer.config"), m.ir.at("driver")));
        } catch (const std::exception & e) {
            if (!a.ids_given) {
                throw;
            }
            // ids given: run without text output
        }
    }
    if (!a.ids_given) {
        ids = encode(*txt, a.text, true);
    }
    if (ids.empty()) {
        std::fprintf(stderr, "no tokens\n");
        return 1;
    }
    if (a.check_dec > 0) {  // prefill all vs prefill all but N, then N decode steps: the state path
        const int64_t L = (int64_t) ids.size(), P0 = L - a.check_dec;
        ttf::StepOut  full = s.step(graph, ids, positions(0, L), false);
        s.reset();
        s.step(graph, std::vector<int32_t>(ids.begin(), ids.begin() + P0), positions(0, P0), true);
        for (int64_t t = P0; t < L; ++t) {
            ttf::StepOut  o  = s.step(graph, { ids[t] }, { (int32_t) t }, true);
            const float * x  = full.logits.data() + t * full.n_vocab;
            const float * y  = o.logits.data();
            double        dn = 0, rn = 0;
            for (int64_t j = 0; j < full.n_vocab; ++j) {
                dn += (double) (x[j] - y[j]) * (x[j] - y[j]);
                rn += (double) x[j] * x[j];
            }
            std::printf("pos %lld: decode vs prefill rel %.3e, argmax %d / %d\n", (long long) t, std::sqrt(dn / rn),
                        (int) (std::max_element(x, x + full.n_vocab) - x), (int) (std::max_element(y, y + o.n_vocab) - y));
        }
        return 0;
    }
    ttf::StepOut r = s.step(graph, ids, positions(0, (int64_t) ids.size()), a.dump.empty());
    if (a.verbose) {
        std::fprintf(stderr, "prefill %zu tokens: %.1f ms\n", ids.size(), s.last_ms);
    }
    if (!a.dump.empty()) {
        FILE * f = std::fopen(a.dump.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", a.dump.c_str());
            return 1;
        }
        std::fwrite(r.logits.data(), sizeof(float), r.logits.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "dumped logits [%lld, %lld] to %s\n", (long long) r.T_out, (long long) r.n_vocab, a.dump.c_str());
    }
    std::mt19937         rng(a.seed);
    int32_t              next = pick(r.logits.data() + (r.T_out - 1) * r.n_vocab, r.n_vocab, a.temp, a.top_k, rng);
    std::vector<int32_t> gen;
    std::string          shown;
    double               ms = 0;
    int32_t              p  = (int32_t) ids.size();
    s.prof_build = s.prof_alloc = s.prof_compute = 0;
    if (txt && !a.ids_given) {
        std::printf("%s", a.text.c_str());
    }
    for (int i = 0; i < a.n_gen; ++i) {
        if (txt && std::find(txt->eos.begin(), txt->eos.end(), next) != txt->eos.end()) {
            break;
        }
        gen.push_back(next);
        if (txt) {  // stream: print what the decoded continuation added (no BOS-style strip mid-text)
            const std::string all = txt->tok->decode(gen, false);
            if (all.size() > shown.size() && all.compare(0, shown.size(), shown) == 0) {
                std::printf("%s", all.c_str() + shown.size());
                std::fflush(stdout);
                shown = all;
            }
        }
        if (i + 1 == a.n_gen) {
            break;
        }
        ttf::StepOut o = s.step(graph, { next }, { p++ }, true);
        ms += s.last_ms;
        next = pick(o.logits.data(), o.n_vocab, a.temp, a.top_k, rng);
    }
    if (txt) {
        std::printf("\n");
    }
    if (!txt || a.verbose) {
        std::printf("ids:");
        for (int32_t t : gen) {
            std::printf(" %d", t);
        }
        std::printf("\n");
    }
    const int nd = std::max<int>(1, (int) gen.size() - 1);
    if (a.verbose && gen.size() > 1) {
        std::fprintf(stderr, "decode %d tokens: %.2f ms/token, %.1f tok/s\n", nd, ms / nd, 1000.0 * nd / ms);
        std::fprintf(stderr, "  per token: build %.2f ms, alloc %.2f ms, compute %.2f ms; %d graph nodes\n",
                     s.prof_build / nd, s.prof_alloc / nd, s.prof_compute / nd, s.prof_nodes);
        if (!s.prof_ops.empty()) {
            print_ops(s, nd + 1);
        }
    }
    return 0;
}

// ---- lm_diffusion driver (speech)
int run_speech(const ttf::Model & m, Args a) {
    std::vector<float> noise;
    if (!a.noise.empty()) {
        std::ifstream f(a.noise, std::ios::binary | std::ios::ate);
        noise.resize((size_t) f.tellg() / 4);
        f.seekg(0);
        f.read((char *) noise.data(), noise.size() * 4);
        a.sp.noise = &noise;
    }
    a.sp.verbose = a.verbose;
    a.sp.seed    = a.seed;
    struct Job {
        std::string out, text;  // text "" = --ids
        uint32_t    seed;
    };
    std::vector<Job> jobs;
    if (!a.list.empty()) {
        std::ifstream f(a.list);
        if (!f) {
            std::fprintf(stderr, "cannot read %s\n", a.list.c_str());
            return 1;
        }
        for (std::string line; std::getline(f, line);) {
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                continue;
            }
            std::string  rest = line.substr(tab + 1);
            uint32_t     sd   = a.seed;
            const size_t t2   = rest.find('\t');
            if (t2 != std::string::npos) {
                sd   = (uint32_t) std::strtoul(rest.c_str() + t2 + 1, nullptr, 10);
                rest = rest.substr(0, t2);
            }
            jobs.push_back({ line.substr(0, tab), rest, sd });
        }
    } else if (a.bench) {
        const char * t = "The quick brown fox jumps over the lazy dog, and then it runs back home before the rain.";
        for (int r = 0; r <= a.reps; ++r) {  // the first is the warm-up
            jobs.push_back({ "", t, 0 });
        }
    } else {
        jobs.push_back({ a.out, a.text, a.seed });
    }
    std::unique_ptr<Text> txt;
    for (const Job & j : jobs) {
        if (!j.text.empty() && !txt) {
            txt = std::make_unique<Text>(load_text(m.kv_str("ttf.tokenizer.json"), "", m.ir.at("driver")));
        }
    }
    ttf::Session s(m, a.threads, a.ctx ? a.ctx : 4096);
    double       bsec = 0, bms = 0;
    for (size_t k = 0; k < jobs.size(); ++k) {
        const Job &          jb  = jobs[k];
        std::vector<int32_t> ids = jb.text.empty() ? a.ids : encode(*txt, jb.text, false);
        if (ids.empty()) {
            std::fprintf(stderr, "%s: no tokens\n", jb.out.c_str());
            return 1;
        }
        a.sp.seed = jb.seed;
        s.prof_build = s.prof_alloc = s.prof_galloc = s.prof_compute = 0;
        const double  t0  = now_ms();
        ttf::SpeakOut o   = ttf::speak(s, ids, a.sp);
        const double  ms  = now_ms() - t0;
        const double  sec = (double) o.wav.size() / o.sample_rate;
        if (a.bench) {
            if (k > 0) {
                bsec += sec;
                bms += ms;
            }
            continue;
        }
        std::fprintf(stderr, "%s: %.2f s audio in %.2f s (RTF %.3f)\n", jb.out.c_str(), sec, ms / 1000, ms / 1000 / sec);
        if (a.verbose) {
            std::fprintf(stderr, "  %d patches; prefill %.0f ms, dit %.0f ms, lm %.0f ms (locenc %.0f), vae %.0f ms\n",
                         o.n_patches, o.ms_prefill, o.ms_dit, o.ms_lm, o.ms_enc, o.ms_vae);
            std::fprintf(stderr, "  session: build %.0f ms, alloc %.0f ms (allocator %.0f), compute %.0f ms\n", s.prof_build,
                         s.prof_alloc, s.prof_galloc, s.prof_compute);
        }
        if (!write_wav(jb.out, o.wav, o.sample_rate)) {
            std::fprintf(stderr, "cannot write %s\n", jb.out.c_str());
            return 1;
        }
        if (!a.feat.empty()) {
            FILE * f = std::fopen(a.feat.c_str(), "wb");
            if (f) {
                std::fwrite(o.feat.data(), 4, o.feat.size(), f);
                std::fclose(f);
            }
        }
    }
    if (a.bench) {
        std::printf("speech: %.2f s audio in %.2f s per run (%d runs after a warm-up), RTF %.3f\n", bsec / a.reps,
                    bms / 1000 / a.reps, a.reps, bms / 1000 / bsec);
    }
    return 0;
}

// ---- tokenizer only: no weights loaded
int run_tokenize(const std::string & path) {
    gguf_init_params gp = { true, nullptr };
    gguf_context *   g  = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        return 1;
    }
    auto kv = [&](const char * k) -> std::string {
        const int64_t i = gguf_find_key(g, k);
        return i < 0 ? std::string() : std::string(gguf_get_val_str(g, i));
    };
    const std::string irs = kv("ttf.ir");
    const json        ir  = irs.empty() ? json::object() : json::parse(irs);
    const Text        t   = load_text(kv("ttf.tokenizer.json"), "", ir.value("driver", json::object()));
    gguf_free(g);
    for (std::string line; std::getline(std::cin, line);) {
        const auto ids = encode(t, line, false);
        for (size_t k = 0; k < ids.size(); ++k) {
            std::printf(k ? ",%d" : "%d", ids[k]);
        }
        std::printf("\n");
    }
    return 0;
}

const char * USAGE =
    "usage: ttf-home MODEL \"text\" [options]     run the model (text model: continuation; speech model: -o out.wav)\n"
    "       ttf-home MODEL --bench [--pp 512] [--tg 128] [--reps 3]\n"
    "       ttf-home MODEL --list FILE           speech model, one job per line: out.wav<TAB>text[<TAB>seed]\n"
    "       ttf-home MODEL --tokenize < lines\n"
    "options: -n N  --temp T  --top-k K  --seed S  -o FILE  --timesteps N  --cfg F  --max-len N\n"
    "         --ids 1,2,3  --device cpu|cuda  --threads N  --ctx N  -v\n"
    "checks:  --dump FILE  --check-decode N  --graph NAME --in NAME=FILE [--out DIR]  --noise FILE  --feat FILE\n";

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        std::fputs(USAGE, stderr);
        return argc < 2;
    }
    Args a;
    a.model = argv[1];
#ifdef TTF_CUDA
    a.device = "cuda";
#else
    a.device = "cpu";
#endif
    std::vector<std::string> pos;
    for (int i = 2; i < argc; ++i) {
        const std::string o    = argv[i];
        auto              next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: needs a value\n", o.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (o == "--ids") {
            a.ids       = parse_ids(next());
            a.ids_given = true;
        } else if (o == "--text") {
            a.text = next();
        } else if (o == "-n" || o == "--gen") {
            a.n_gen = std::atoi(next().c_str());
        } else if (o == "--temp") {
            a.temp = (float) std::atof(next().c_str());
        } else if (o == "--top-k") {
            a.top_k = std::atoi(next().c_str());
        } else if (o == "--seed") {
            a.seed = (uint32_t) std::strtoul(next().c_str(), nullptr, 10);
        } else if (o == "-o") {
            a.out = next();
        } else if (o == "--out") {  // a directory with --graph, the wav otherwise
            a.out = a.out_dir = next();
        } else if (o == "--list") {
            a.list = next();
        } else if (o == "--timesteps") {
            a.sp.timesteps = std::atoi(next().c_str());
        } else if (o == "--cfg") {
            a.sp.cfg = (float) std::atof(next().c_str());
        } else if (o == "--max-len") {
            a.sp.max_len = std::atoi(next().c_str());
        } else if (o == "--threads") {
            a.threads = std::atoi(next().c_str());
        } else if (o == "--ctx") {
            a.ctx = std::atoll(next().c_str());
        } else if (o == "--device") {
            a.device = next();
        } else if (o == "--bench") {
            a.bench = true;
        } else if (o == "--pp") {
            a.pp = std::max(32, std::atoi(next().c_str()));
        } else if (o == "--tg") {
            a.tg = std::max(1, std::atoi(next().c_str()));
        } else if (o == "--reps") {
            a.reps = std::max(1, std::atoi(next().c_str()));
        } else if (o == "--tokenize") {
            a.tokenize = true;
        } else if (o == "--dump") {
            a.dump = next();
        } else if (o == "--check-decode") {
            a.check_dec = std::atoi(next().c_str());
        } else if (o == "--graph") {
            a.graph = next();
        } else if (o == "--in") {
            const std::string v = next();
            const size_t      e = v.find('=');
            a.gin[v.substr(0, e)] = v.substr(e + 1);
        } else if (o == "--noise") {
            a.noise = next();
        } else if (o == "--feat") {
            a.feat = next();
        } else if (o == "-v") {
            a.verbose = true;
        } else if (!o.empty() && o[0] == '-') {
            std::fprintf(stderr, "unknown option %s\n%s", o.c_str(), USAGE);
            return 1;
        } else {
            pos.push_back(o);
        }
    }
    if (!pos.empty()) {
        std::string t;
        for (const auto & p : pos) {
            t += (t.empty() ? "" : " ") + p;
        }
        a.text = a.text.empty() ? t : a.text + " " + t;
    }
    try {
        if (a.tokenize) {
            return run_tokenize(a.model);
        }
        if (a.graph.empty() && !a.bench && a.text.empty() && a.ids.empty() && a.list.empty()) {
            std::fputs(USAGE, stderr);
            return 1;
        }
        std::string err;
        auto        m = ttf::Model::load(a.model, &err, a.device);
        if (!m) {
            std::fprintf(stderr, "load: %s\n", err.c_str());
            return 1;
        }
        if (a.verbose || a.bench) {
            std::fprintf(stderr, "%s: family %s, driver %s, D %s, %s, trits %s\n", a.model.c_str(), m->family.c_str(),
                         m->ir.at("driver").at("name").get<std::string>().c_str(), m->d_bf16 ? "bf16" : "f32",
                         m->device.c_str(), m->q2p ? "q2p" : "tq1p");
        }
        if (!a.graph.empty()) {
            return run_graph(*m, a);
        }
        const std::string drv = m->ir.at("driver").at("name");
        if (drv == "lm") {
            return run_lm(*m, a);
        }
        if (drv == "lm_diffusion") {
            return run_speech(*m, a);
        }
        std::fprintf(stderr, "driver %s: not implemented\n", drv.c_str());
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
