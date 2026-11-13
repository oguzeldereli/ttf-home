// ttf-home: run any .ttf model. The IR's driver decides what the model does:
//   lm            text in, text out (greedy, or sampled with --temp)
//   lm_diffusion  text in, speech out (a 16-bit WAV)
//
//   ttf-home MODEL "text" [-n N] [--temp T] [--top-k K] [--seed S]      text model: the continuation on stdout
//   ttf-home MODEL "text" [-o out.wav] [--seed S] [--timesteps N] [--cfg F] [--max-len N]   speech model
//   ttf-home MODEL --list FILE           speech model: one load, then each line "out.wav<TAB>text[<TAB>seed]"
//   ttf-home MODEL --chat [--system S] [--think | --no-think] [--effort xhigh|medium|low]
//            text model with a chat template: one turn per stdin line, the reply streamed; the KV state is kept
//            across turns (/reset clears it). Sampling defaults --temp 0.7 --top-k 20; -n: the most reply tokens
//   ttf-home MODEL --bench [--pp 512] [--tg 128] [--reps 3] [--depth 16]
//            text model: prefill tok/s at T = pp, decode tok/s at batch 1 over tg tokens, and the bytes one decode
//            token reads (weights, state) with the bandwidth that implies; speech model: real-time factor
//   ttf-home MODEL --tokenize [--specials] [--escapes]   stdin lines -> comma-separated ids (no weights loaded);
//            --specials adds the special tokens (BOS ...) as HF encode with add_special_tokens; --escapes reads
//            \n \r \t \\ in a line as those characters
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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <ctime>

using json = nlohmann::json;

namespace {

struct Args {
    std::string                        model, text, list, out = "out.wav", device, dump, graph, out_dir = ".";
    std::string                        noise, feat;
    std::vector<int32_t>               ids;
    std::map<std::string, std::string> gin;
    int                                n_gen = 128, threads = 4, top_k = 40, check_dec = 0, pp = 512, tg = 128, reps = 3,
                                       depth = 16;
    int64_t                            ctx = 0;
    float                              temp = 0.0f;
    bool                               bench = false, tokenize = false, specials = false, escapes = false, verbose = false, ids_given = false;
    bool                               chat = false, temp_set = false, top_k_set = false;
    int                                think = -1;  // --think 1 / --no-think 0 / the template's default -1
    std::string                        system, effort = "xhigh", host = "127.0.0.1";
    bool                               serve = false;
    int                                port  = 8080;
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
    std::string                     post_kind;  // the driver's tokenizer_post
    std::vector<int32_t>            pre, post;  // special tokens around the text (HF add_special_tokens)
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
    t.post_kind = drv.value("attr", json::object()).value("tokenizer_post", std::string());
    if (t.post_kind != "" && t.post_kind != "mask_multichar_cjk") {
        throw std::runtime_error("tokenizer_post " + t.post_kind + ": not implemented");
    }
    const json cfg = cfgs.empty() ? json::object() : json::parse(cfgs);
    // the post-processor's template first (as HF encode); else the tokenizer config's add_bos_token
    t.pre  = t.tok->post_pre;
    t.post = t.tok->post_post;
    if (t.pre.empty() && cfg.value("add_bos_token", json(false)).is_boolean() && cfg.value("add_bos_token", false) &&
        t.tok->id_of(special(cfg, "bos_token")) >= 0) {
        t.pre.push_back(t.tok->id_of(special(cfg, "bos_token")));
    }
    for (const auto & e : drv.value("attr", json::object()).value("eos", json::array())) {
        t.eos.push_back(e.get<int32_t>());
    }
    if (t.eos.empty() && t.tok->id_of(special(cfg, "eos_token")) >= 0) {
        t.eos.push_back(t.tok->id_of(special(cfg, "eos_token")));
    }
    return t;
}

std::vector<int32_t> encode(const Text & t, const std::string & s, bool specials) {
    std::vector<int32_t> ids = t.post_kind == "mask_multichar_cjk" ? t.tok->encode_multichar_cjk(s) : t.tok->encode(s);
    if (specials) {
        ids.insert(ids.begin(), t.pre.begin(), t.pre.end());
        ids.insert(ids.end(), t.post.begin(), t.post.end());
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
        std::fprintf(stderr, "  %7.3f ms/call %5.1f%%  %5.0f nodes/call  %s\n", ops[i].first / n, 100 * ops[i].first / tot,
                     (double) s.prof_ops.at(ops[i].second).second / n, ops[i].second.c_str());
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
    const int64_t     ctx   = a.ctx ? a.ctx : std::max<int64_t>(2048, std::max(a.pp, a.depth) + a.tg + 8);
    ttf::Session      s(m, a.threads, ctx);
    s.fa_exact = s.fa_exact || (a.check_dec > 0 && !std::getenv("TTF_CHECK_FAST"));  // TTF_CHECK_FAST: check the fast paths  // the state path check compares bit for bit
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
            const int D = a.depth;  // decode after a D-token prefill (in chunks of up to pp)
            for (int p0 = 0; p0 < D; p0 += a.pp) {
                const int n = std::min(a.pp, D - p0);
                std::vector<int32_t> c(n);
                for (int i = 0; i < n; ++i) {
                    c[i] = ids[(p0 + i) % a.pp];
                }
                s.step(graph, c, positions(p0, n), true);
            }
            s.prof_build = s.prof_alloc = s.prof_compute = 0;
            const double t0 = now_ms();
            for (int i = 0; i < a.tg; ++i) {
                s.step(graph, { ids[(D + i) % a.pp] }, { (int32_t) (D + i) }, true);
            }
            dbest = std::min(dbest, (now_ms() - t0) / a.tg);
        }
        const double gb = (double) (s.last_wbytes + s.last_sbytes) / 1e9;
        std::printf("decode   B 1  : %8.2f ms  %9.1f tok/s  (%d tokens)\n", dbest, 1000.0 / dbest, a.tg);
        std::printf("per token: weights %.1f MB, state %.1f MB at position %d; %.1f GB/s\n", s.last_wbytes / 1e6,
                    s.last_sbytes / 1e6, a.depth + a.tg - 1, gb / (dbest / 1000.0));
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
        if (P0 < 1) {
            throw std::runtime_error("--check-decode N needs N < the prompt's " + std::to_string(L) + " tokens");
        }
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


// ---- chat: the file's chat template, rendered per family (no Jinja engine): ChatML (Qwen: system / user /
// assistant, the thinking switch and reasoning-effort text read from the template) and "Role: text<|eot_id|>"
// (BitNet). The KV state is kept across turns: each turn feeds only its new tokens.
struct ChatFmt {
    enum Kind { NONE, CHATML, ROLE_EOT } kind = NONE;
    bool                               think_on = false;     // the template's default for enable_thinking
    bool                               effort_sys = false;   // reasoning-effort text in the system turn when thinking
    bool                               keep_think = false;   // earlier assistant turns keep their think block
    std::map<std::string, std::string> effort_text;          // effort -> instruction (none for "medium")
    std::vector<int32_t>               stop;
};

std::string trim(const std::string & x) {
    const size_t a = x.find_first_not_of(" \t\r\n"), b = x.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : x.substr(a, b - a + 1);
}

ChatFmt chat_fmt(const Text & t, const std::string & cfgs) {
    ChatFmt          f;
    const json       cfg = cfgs.empty() ? json::object() : json::parse(cfgs);
    const std::string tpl = cfg.value("chat_template", json("")).is_string() ? cfg.value("chat_template", std::string()) : "";
    auto              add_stop = [&](const std::string & tok) {
        const int32_t id = t.tok->id_of(tok);
        if (id >= 0 && std::find(f.stop.begin(), f.stop.end(), id) == f.stop.end()) {
            f.stop.push_back(id);
        }
    };
    if (tpl.find("<|im_start|>") != std::string::npos) {
        f.kind = ChatFmt::CHATML;
        // generation prompt: '<think>\n' when thinking, '<think>\n\n</think>\n\n' when not; which one is the default
        f.think_on   = tpl.find("enable_thinking is defined and enable_thinking is false") != std::string::npos;
        f.effort_sys = tpl.find("reasoning_instructions") != std::string::npos;
        f.keep_think = tpl.find("preserve_thinking is undefined or preserve_thinking is true") != std::string::npos;
        for (const char * e : { "xhigh", "low" }) {  // 'Reasoning effort is set to E. ...' literals
            const std::string key = std::string("'Reasoning effort is set to ") + e;
            const size_t      i   = tpl.find(key);
            if (i != std::string::npos) {
                const size_t j = tpl.find('\'', i + 1);
                f.effort_text[e] = tpl.substr(i + 1, j - i - 1);
            }
        }
        add_stop("<|im_end|>");
        add_stop("<|endoftext|>");
    } else if (tpl.find("<|eot_id|>") != std::string::npos && tpl.find("capitalize") != std::string::npos) {
        f.kind = ChatFmt::ROLE_EOT;
        add_stop("<|eot_id|>");
    }
    for (int32_t e : t.eos) {
        if (std::find(f.stop.begin(), f.stop.end(), e) == f.stop.end()) {
            f.stop.push_back(e);
        }
    }
    return f;
}

int run_chat(const ttf::Model & m, const Args & a) {
    const std::string graph = m.ir.at("driver").at("graphs").at("step");
    const Text        txt   = load_text(m.kv_str("ttf.tokenizer.json"), m.kv_str("ttf.tokenizer.config"), m.ir.at("driver"));
    const ChatFmt     f     = chat_fmt(txt, m.kv_str("ttf.tokenizer.config"));
    if (f.kind == ChatFmt::NONE) {
        std::fprintf(stderr, "chat: the file has no chat template this program knows (ChatML, Role: text<|eot_id|>); "
                             "run it as a text continuation instead\n");
        return 1;
    }
    const bool think = a.think < 0 ? f.think_on : a.think == 1;
    if (a.effort != "xhigh" && a.effort != "medium" && a.effort != "low") {
        throw std::runtime_error("--effort: xhigh, medium or low");
    }
    const float   temp  = a.temp_set ? a.temp : 0.7f;
    const int     top_k = a.top_k_set ? a.top_k : 20;
    const int     n_max = a.n_gen == 128 ? 4096 : a.n_gen;  // -n: the most tokens per reply (default 4096)
    const int64_t ctx   = a.ctx ? a.ctx : 16384;
    ttf::Session  s(m, a.threads, ctx);
    std::mt19937  rng(a.seed);
    const bool    tty = isatty(0);
    int32_t       pos = 0;
    int32_t       pending = -1;  // the last reply's stop token: not fed yet
    std::string   prefix;        // text before the next user turn (the system turn on the first)
    auto          start = [&]() {
        s.reset();
        pos     = 0;
        pending = -1;
        prefix.clear();
        if (f.kind == ChatFmt::CHATML) {
            std::string instr;
            if (f.effort_sys && think && f.effort_text.count(a.effort)) {
                instr = f.effort_text.at(a.effort);
            }
            const std::string sys = trim(a.system);
            if (!sys.empty()) {
                prefix = "<|im_start|>system\n" + (instr.empty() ? "" : instr + "\n\n") + sys + "<|im_end|>\n";
            } else if (!instr.empty()) {
                prefix = "<|im_start|>system\n" + instr + "<|im_end|>\n";
            }
        } else if (!trim(a.system).empty()) {
            prefix = "System: " + trim(a.system) + "<|eot_id|>";
        }
    };
    start();
    if (tty) {
        std::fprintf(stderr, "chat (%s, thinking %s); /reset clears the conversation, /exit or Ctrl-D quits\n",
                     f.kind == ChatFmt::CHATML ? "ChatML" : "Role: text", think ? "on" : "off");
    }
    for (;;) {
        if (tty) {
            std::fprintf(stderr, "\n> ");
        }
        std::string line;
        if (!std::getline(std::cin, line)) {
            break;
        }
        const std::string u = trim(line);
        if (u.empty()) {
            continue;
        }
        if (u == "/exit" || u == "/quit") {
            break;
        }
        if (u == "/reset") {
            start();
            continue;
        }
        std::string turn = prefix;
        if (f.kind == ChatFmt::CHATML) {
            turn += (pending >= 0 ? "\n" : "") + std::string("<|im_start|>user\n") + u + "<|im_end|>\n<|im_start|>assistant\n" +
                    (think ? "<think>\n" : "<think>\n\n</think>\n\n");
        } else {
            turn += "User: " + u + "<|eot_id|>Assistant: ";
        }
        std::vector<int32_t> ids = txt.tok->encode(turn);
        if (pos == 0) {
            ids.insert(ids.begin(), txt.pre.begin(), txt.pre.end());
        }
        if (pending >= 0) {
            ids.insert(ids.begin(), pending);
        }
        if (pos + (int64_t) ids.size() + 8 > ctx) {
            std::fprintf(stderr, "[context full (%lld tokens): conversation reset]\n", (long long) ctx);
            start();
            ids = txt.tok->encode(prefix + turn.substr(prefix.size()));
            ids.insert(ids.begin(), txt.pre.begin(), txt.pre.end());
        }
        prefix.clear();
        static const bool show_ids = std::getenv("TTF_CHAT_IDS") != nullptr;  // the fed and generated ids, for checks
        if (show_ids) {
            std::fprintf(stderr, "[fed");
            for (int32_t v : ids) {
                std::fprintf(stderr, " %d", v);
            }
            std::fprintf(stderr, "]\n");
        }
        const double t0 = now_ms();
        ttf::StepOut r  = s.step(graph, ids, positions(pos, (int64_t) ids.size()), true);
        const double tp = now_ms() - t0;
        pos += (int32_t) ids.size();
        int32_t              next = pick(r.logits.data() + (r.T_out - 1) * r.n_vocab, r.n_vocab, temp, top_k, rng);
        std::vector<int32_t> gen;
        std::string          shown;
        const double         t1 = now_ms();
        if (f.kind == ChatFmt::CHATML && think) {  // the prompt opened the reasoning block
            std::printf("<think>\n");
        }
        pending                 = -1;
        for (int i = 0; i < n_max; ++i) {
            if (std::find(f.stop.begin(), f.stop.end(), next) != f.stop.end()) {
                pending = next;
                break;
            }
            gen.push_back(next);
            const std::string all = txt.tok->decode(gen, false);
            if (all.size() > shown.size() && all.compare(0, shown.size(), shown) == 0) {
                std::printf("%s", all.c_str() + shown.size());
                std::fflush(stdout);
                shown = all;
            }
            if (pos + 2 > ctx) {
                break;
            }
            ttf::StepOut o = s.step(graph, { next }, { pos++ }, true);
            next           = pick(o.logits.data(), o.n_vocab, temp, top_k, rng);
        }
        if (show_ids) {
            std::fprintf(stderr, "[gen");
            for (int32_t v : gen) {
                std::fprintf(stderr, " %d", v);
            }
            std::fprintf(stderr, " | stop %d]\n", pending);
        }
        if (pending < 0) {  // cut off (length or context): close the turn as the template would
            pending = f.stop.empty() ? -1 : f.stop[0];
        }
        std::printf("\n");
        std::fflush(stdout);
        if (a.verbose) {
            const double td = now_ms() - t1;
            std::fprintf(stderr, "[prompt %zu tokens %.0f ms (%.0f tok/s); reply %zu tokens %.1f tok/s; context %d / %lld]\n",
                         ids.size(), tp, 1000.0 * ids.size() / tp, gen.size(), gen.empty() ? 0.0 : 1000.0 * gen.size() / td,
                         pos, (long long) ctx);
        }
    }
    return 0;
}


// ---- serve: an OpenAI-style HTTP endpoint (POST /v1/chat/completions, GET /v1/models), one request at a time.
// Messages render as the template does for the families above (no tools); reasoning goes out as reasoning_content.
// The state is reused across requests in two ways, both exact in tokens: (1) when the new prompt extends the last
// reply as the client sends it back (templates that keep the reasoning in history), the state as the reply left it;
// (2) else, when it extends the last request's history (without the generation prompt), the recurrent state saved
// there and the kv cache cut back to it. Otherwise a reset. TTF_CHAT_IDS=1 prints each prompt's ids.
struct Msg {
    std::string role, content, reasoning;
};

std::string render_chat(const ChatFmt & f, const std::vector<Msg> & ms, bool gen, bool think, const std::string & effort) {
    std::string o;
    size_t      i0 = 0;
    if (f.kind == ChatFmt::CHATML) {
        std::string instr;
        if (f.effort_sys && think && f.effort_text.count(effort)) {
            instr = f.effort_text.at(effort);
        }
        if (!ms.empty() && ms[0].role == "system") {
            const std::string c = trim(ms[0].content);
            if (!c.empty()) {
                o += "<|im_start|>system\n" + (instr.empty() ? "" : instr + "\n\n") + c + "<|im_end|>\n";
            } else if (!instr.empty()) {
                o += "<|im_start|>system\n" + instr + "<|im_end|>\n";
            }
            i0 = 1;
        } else if (!instr.empty()) {
            o += "<|im_start|>system\n" + instr + "<|im_end|>\n";
        }
        size_t last_user = 0;
        for (size_t i = 0; i < ms.size(); ++i) {
            if (ms[i].role == "user") {
                last_user = i;
            }
        }
        for (size_t i = i0; i < ms.size(); ++i) {
            const Msg & m = ms[i];
            std::string c = trim(m.content), rc = trim(m.reasoning);
            if (m.role == "user") {
                o += "<|im_start|>user\n" + c + "<|im_end|>\n";
            } else if (m.role == "assistant") {
                const size_t e = c.find("</think>");
                if (rc.empty() && e != std::string::npos) {  // reasoning inline in the content
                    const size_t b = c.rfind("<think>", e);
                    rc             = trim(c.substr(b == std::string::npos ? 0 : b + 7, e - (b == std::string::npos ? 0 : b + 7)));
                    c              = trim(c.substr(e + 8));
                }
                if (f.keep_think || i > last_user) {
                    o += "<|im_start|>assistant\n<think>\n" + rc + "\n</think>\n\n" + c + "<|im_end|>\n";
                } else {
                    o += "<|im_start|>assistant\n" + c + "<|im_end|>\n";
                }
            } else {
                throw std::runtime_error("role " + m.role + ": not supported");
            }
        }
        if (gen) {
            o += "<|im_start|>assistant\n" + std::string(think ? "<think>\n" : "<think>\n\n</think>\n\n");
        }
    } else {
        for (const Msg & m : ms) {
            if (m.role != "system" && m.role != "user" && m.role != "assistant") {
                throw std::runtime_error("role " + m.role + ": not supported");
            }
            std::string r = m.role;
            r[0]          = (char) std::toupper(r[0]);
            o += r + ": " + trim(m.content) + "<|eot_id|>";
        }
        if (gen) {
            o += "Assistant: ";
        }
    }
    return o;
}

bool send_all(int fd, const std::string & d) {
    size_t o = 0;
    while (o < d.size()) {
        const ssize_t n = ::send(fd, d.data() + o, d.size() - o, MSG_NOSIGNAL);
        if (n <= 0) {
            return false;
        }
        o += (size_t) n;
    }
    return true;
}

void http_reply(int fd, int code, const std::string & type, const std::string & body) {
    const char * st = code == 200 ? "OK" : code == 204 ? "No Content" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" : "Error";
    send_all(fd, "HTTP/1.1 " + std::to_string(code) + " " + st + "\r\nContent-Type: " + type +
                     "\r\nContent-Length: " + std::to_string(body.size()) +
                     "\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n" + body);
}

int run_serve(const ttf::Model & m, const Args & a) {
    const std::string graph = m.ir.at("driver").at("graphs").at("step");
    const Text        txt   = load_text(m.kv_str("ttf.tokenizer.json"), m.kv_str("ttf.tokenizer.config"), m.ir.at("driver"));
    const ChatFmt     f     = chat_fmt(txt, m.kv_str("ttf.tokenizer.config"));
    if (f.kind == ChatFmt::NONE) {
        std::fprintf(stderr, "serve: the file has no chat template this program knows\n");
        return 1;
    }
    const int64_t ctx = a.ctx ? a.ctx : 16384;
    ttf::Session  s(m, a.threads, ctx);
    std::string   name = a.model.substr(a.model.find_last_of('/') + 1);
    std::vector<int32_t> kv_view;  // the client view of what the state holds (empty: nothing reusable)
    std::vector<int32_t> snap_ids;  // the history (no generation prompt) whose recurrent state is in snap
    std::vector<uint8_t> snap;
    int32_t              pos = 0;
    const int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    int       one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) a.port);
    if (inet_pton(AF_INET, a.host.c_str(), &addr.sin_addr) != 1 || ::bind(lfd, (sockaddr *) &addr, sizeof(addr)) != 0 ||
        ::listen(lfd, 8) != 0) {
        std::fprintf(stderr, "serve: cannot listen on %s:%d\n", a.host.c_str(), a.port);
        return 1;
    }
    std::fprintf(stderr, "serving %s on http://%s:%d/v1 (POST /v1/chat/completions, GET /v1/models)\n", name.c_str(),
                 a.host.c_str(), a.port);
    for (;;) {
        const int fd = ::accept(lfd, nullptr, nullptr);
        if (fd < 0) {
            continue;
        }
        timeval tv{ 30, 0 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        std::string req;
        char        buf[65536];
        size_t      hend = std::string::npos, need = 0;
        bool        ok   = true;
        while (ok) {
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                ok = false;
                break;
            }
            req.append(buf, (size_t) n);
            if (hend == std::string::npos && (hend = req.find("\r\n\r\n")) != std::string::npos) {
                std::string h = req.substr(0, hend);
                std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) { return (char) std::tolower(c); });
                const size_t cl = h.find("content-length:");
                need            = hend + 4 + (cl == std::string::npos ? 0 : std::strtoull(h.c_str() + cl + 15, nullptr, 10));
            }
            if (hend != std::string::npos && req.size() >= need) {
                break;
            }
            if (req.size() > (64u << 20)) {
                ok = false;
            }
        }
        if (!ok || hend == std::string::npos) {
            ::close(fd);
            continue;
        }
        const std::string line   = req.substr(0, req.find("\r\n"));
        const std::string method = line.substr(0, line.find(' '));
        std::string       path   = line.substr(line.find(' ') + 1);
        path                     = path.substr(0, path.find(' '));
        const std::string body   = req.substr(hend + 4);
        try {
            if (method == "OPTIONS") {
                http_reply(fd, 204, "text/plain", "");
            } else if (method == "GET" && path == "/v1/models") {
                http_reply(fd, 200, "application/json",
                           json{ { "object", "list" }, { "data", json::array({ json{ { "id", name }, { "object", "model" }, { "owned_by", "local" } } }) } }.dump());
            } else if (method == "POST" && path == "/v1/chat/completions") {
                const json       q = json::parse(body);
                if (q.contains("tools") && !q.at("tools").empty()) {
                    throw std::runtime_error("tools: not supported");
                }
                std::vector<Msg> ms;
                for (const auto & jm : q.at("messages")) {
                    Msg mm;
                    mm.role = jm.at("role");
                    if (jm.contains("content") && jm.at("content").is_string()) {
                        mm.content = jm.at("content");
                    } else if (jm.contains("content") && jm.at("content").is_array()) {
                        for (const auto & part : jm.at("content")) {
                            if (part.value("type", std::string()) == "text") {
                                mm.content += part.value("text", std::string());
                            } else {
                                throw std::runtime_error("content part " + part.value("type", std::string("?")) + ": not supported");
                            }
                        }
                    }
                    if (jm.contains("reasoning_content") && jm.at("reasoning_content").is_string()) {
                        mm.reasoning = jm.at("reasoning_content");
                    }
                    ms.push_back(mm);
                }
                const bool  stream = q.value("stream", false);
                const float temp   = q.contains("temperature") && q.at("temperature").is_number() ? q.at("temperature").get<float>() : (a.temp_set ? a.temp : 0.7f);
                const int   top_k  = q.contains("top_k") && q.at("top_k").is_number() ? q.at("top_k").get<int>() : (a.top_k_set ? a.top_k : 20);
                int         n_max  = a.n_gen == 128 ? 4096 : a.n_gen;
                for (const char * k : { "max_tokens", "max_completion_tokens" }) {
                    if (q.contains(k) && q.at(k).is_number()) {
                        n_max = q.at(k).get<int>();
                    }
                }
                bool think = a.think < 0 ? f.think_on : a.think == 1;
                if (q.contains("chat_template_kwargs") && q.at("chat_template_kwargs").contains("enable_thinking")) {
                    think = q.at("chat_template_kwargs").at("enable_thinking").get<bool>();
                }
                std::string effort = q.value("reasoning_effort", a.effort);
                if (effort == "high") {
                    effort = "xhigh";
                }
                std::mt19937 rng(q.contains("seed") && q.at("seed").is_number() ? q.at("seed").get<uint32_t>() : (uint32_t) std::random_device{}());
                // the prompt; reuse the state when it extends the previous conversation's client view
                std::vector<int32_t> ids = txt.tok->encode(render_chat(f, ms, true, think, effort));
                ids.insert(ids.begin(), txt.pre.begin(), txt.pre.end());
                if ((int64_t) ids.size() + 8 > ctx) {
                    throw std::runtime_error("prompt of " + std::to_string(ids.size()) + " tokens over the context " + std::to_string(ctx));
                }
                // reuse: the state as the last reply left it (when the client sent it back as rendered), else the
                // snapshot taken at the end of the last request's history (a template that drops the reasoning from
                // history, a regenerated reply), else a reset
                auto starts = [&](const std::vector<int32_t> & p) {
                    return !p.empty() && ids.size() > p.size() && std::equal(p.begin(), p.end(), ids.begin());
                };
                size_t reuse = 0;
                if (starts(kv_view)) {
                    reuse = kv_view.size();
                } else if (starts(snap_ids)) {
                    s.load_recurrent(snap);
                    s.truncate((int32_t) snap_ids.size());
                    pos   = (int32_t) snap_ids.size();
                    reuse = snap_ids.size();
                } else {
                    s.reset();
                    pos = 0;
                }
                const size_t reused = reuse;
                if (std::getenv("TTF_CHAT_IDS")) {
                    std::fprintf(stderr, "[prompt");
                    for (int32_t t : ids) {
                        std::fprintf(stderr, " %d", t);
                    }
                    std::fprintf(stderr, "]\n");
                }
                const auto   t0     = std::chrono::steady_clock::now();
                double       t_snap = 0;
                std::vector<int32_t> base = txt.tok->encode(render_chat(f, ms, false, think, effort));
                base.insert(base.begin(), txt.pre.begin(), txt.pre.end());
                if (base.size() > reuse && base.size() < ids.size() && std::equal(base.begin(), base.end(), ids.begin())) {
                    const std::vector<int32_t> h(ids.begin() + (int64_t) reuse, ids.begin() + (int64_t) base.size());
                    s.step(graph, h, positions(pos, (int64_t) h.size()), true);
                    pos += (int32_t) h.size();
                    reuse    = base.size();
                    const auto ts = std::chrono::steady_clock::now();
                    snap          = s.save_recurrent();
                    t_snap        = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
                    snap_ids = base;
                }
                const std::vector<int32_t> feed(ids.begin() + (int64_t) reuse, ids.end());
                const std::string id      = "chatcmpl-" + std::to_string(std::random_device{}());
                const int64_t     created = (int64_t) std::time(nullptr);
                if (stream) {
                    send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                                 "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
                }
                auto chunk = [&](const json & delta, const json & finish) {
                    json c = { { "id", id }, { "object", "chat.completion.chunk" }, { "created", created }, { "model", name },
                               { "choices", json::array({ json{ { "index", 0 }, { "delta", delta }, { "finish_reason", finish } } }) } };
                    return send_all(fd, "data: " + c.dump() + "\n\n");
                };
                if (stream) {
                    chunk(json{ { "role", "assistant" }, { "content", "" } }, nullptr);
                }
                ttf::StepOut r = s.step(graph, feed, positions(pos, (int64_t) feed.size()), true);
                pos += (int32_t) feed.size();
                const auto t1 = std::chrono::steady_clock::now();
                kv_view.clear();  // set again below only when the state matches the client's next rendering
                int32_t              next = pick(r.logits.data() + (r.T_out - 1) * r.n_vocab, r.n_vocab, temp, top_k, rng);
                std::vector<int32_t> gen;
                std::string          shown, reasoning, content;
                size_t               rsent    = 0;  // reasoning bytes streamed
                bool                 in_think = f.kind == ChatFmt::CHATML && think, alive = true;
                std::string          finish   = "length";
                auto add_content = [&](std::string piece) {
                    if (content.empty()) {  // the answer's leading newlines are dropped (the template trims them)
                        const size_t nz = piece.find_first_not_of('\n');
                        piece           = nz == std::string::npos ? "" : piece.substr(nz);
                    }
                    content += piece;
                    if (stream && !piece.empty()) {
                        alive = chunk(json{ { "content", piece } }, nullptr);
                    }
                };
                auto send_reasoning = [&](size_t upto) {
                    if (stream && upto > rsent) {
                        alive = chunk(json{ { "reasoning_content", reasoning.substr(rsent, upto - rsent) } }, nullptr);
                    }
                    rsent = std::max(rsent, upto);
                };
                for (int i = 0; i < n_max && alive; ++i) {
                    if (std::find(f.stop.begin(), f.stop.end(), next) != f.stop.end()) {
                        finish = "stop";
                        break;
                    }
                    gen.push_back(next);
                    const std::string all = txt.tok->decode(gen, false);
                    if (all.size() > shown.size() && all.compare(0, shown.size(), shown) == 0) {  // whole UTF-8 only
                        const std::string piece = all.substr(shown.size());
                        shown                   = all;
                        if (in_think) {  // the reasoning until </think>, then the answer
                            reasoning += piece;
                            const size_t e = reasoning.find("</think>");
                            if (e != std::string::npos) {
                                const std::string rest = reasoning.substr(e + 8);
                                reasoning.resize(e);
                                send_reasoning(e);
                                in_think = false;
                                add_content(rest);
                            } else {  // hold back a tail that may start "</think>"
                                size_t hold = 0;
                                for (size_t k = std::min<size_t>(7, reasoning.size()); k > 0 && !hold; --k) {
                                    if (reasoning.compare(reasoning.size() - k, k, "</think>", k) == 0) {
                                        hold = k;
                                    }
                                }
                                send_reasoning(reasoning.size() - hold);
                            }
                        } else {
                            add_content(piece);
                        }
                    }
                    if (pos + 2 > ctx) {
                        break;
                    }
                    ttf::StepOut o = s.step(graph, { next }, { pos++ }, true);
                    next           = pick(o.logits.data(), o.n_vocab, temp, top_k, rng);
                }
                if (in_think) {
                    send_reasoning(reasoning.size());
                }
                // KV reuse, exact only: the client sends back the conversation plus this reply; when that rendering
                // extends the tokens the state holds (prompt + generated), feed its tail now (the stop token and the
                // separator) and keep it as the reusable prefix. Otherwise (e.g. a template that drops the reasoning
                // from history) the next request re-prefills from a reset state.
                if (alive && finish == "stop") {
                    std::vector<Msg> ms2 = ms;
                    ms2.push_back(Msg{ "assistant", content, reasoning });
                    std::vector<int32_t> view = txt.tok->encode(render_chat(f, ms2, false, think, effort));
                    view.insert(view.begin(), txt.pre.begin(), txt.pre.end());
                    std::vector<int32_t> real(ids);
                    real.insert(real.end(), gen.begin(), gen.end());
                    if (view.size() > real.size() && view.size() + 8 < (size_t) ctx &&
                        std::equal(real.begin(), real.end(), view.begin())) {
                        const std::vector<int32_t> tail(view.begin() + (int64_t) real.size(), view.end());
                        s.step(graph, tail, positions(pos, (int64_t) tail.size()), true);
                        pos += (int32_t) tail.size();
                        kv_view = view;
                    }
                }
                json usage = { { "prompt_tokens", ids.size() }, { "completion_tokens", gen.size() }, { "total_tokens", ids.size() + gen.size() } };
                if (stream) {
                    if (alive) {
                        chunk(json::object(), finish);
                        send_all(fd, "data: [DONE]\n\n");
                    }
                } else {
                    json msg = { { "role", "assistant" }, { "content", content } };
                    if (f.kind == ChatFmt::CHATML && think) {
                        msg["reasoning_content"] = trim(reasoning);
                    }
                    http_reply(fd, 200, "application/json",
                               json{ { "id", id }, { "object", "chat.completion" }, { "created", created }, { "model", name },
                                     { "choices", json::array({ json{ { "index", 0 }, { "message", msg }, { "finish_reason", finish } } }) },
                                     { "usage", usage } }.dump());
                }
                if (a.verbose) {
                    const double tp = std::chrono::duration<double>(t1 - t0).count();
                    const double tg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
                    std::fprintf(stderr, "[%zu prompt tokens (%zu reused) %.2f s (snapshot %.0f MB %.2f s), %zu generated %.1f tok/s, %s]\n",
                                 ids.size(), reused, tp, snap.size() / 1e6, t_snap, gen.size(), gen.size() / std::max(tg, 1e-9), finish.c_str());
                }
            } else {
                http_reply(fd, 404, "application/json", json{ { "error", { { "message", "not found: " + method + " " + path } } } }.dump());
            }
        } catch (const std::exception & e) {
            http_reply(fd, 400, "application/json", json{ { "error", { { "message", e.what() } } } }.dump());
        }
        ::close(fd);
    }
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
        std::fprintf(stderr, "%s: %.2f s audio in %.2f s (RTF %.3f)%s\n", jb.out.c_str(), sec, ms / 1000, ms / 1000 / sec,
                     o.attempts > 1 ? (" after " + std::to_string(o.attempts) + " attempts, seed " + std::to_string(o.seed)).c_str() : "");
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
int run_tokenize(const std::string & path, bool specials, bool escapes) {
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
    const Text        t   = load_text(kv("ttf.tokenizer.json"), kv("ttf.tokenizer.config"), ir.value("driver", json::object()));
    gguf_free(g);
    for (std::string line; std::getline(std::cin, line);) {
        if (escapes) {  // \n \r \t \\ in the line
            std::string u;
            for (size_t i = 0; i < line.size(); ++i) {
                if (line[i] == '\\' && i + 1 < line.size() && std::strchr("nrt\\", line[i + 1])) {
                    const char e = line[++i];
                    u += e == 'n' ? '\n' : e == 'r' ? '\r' : e == 't' ? '\t' : '\\';
                } else {
                    u += line[i];
                }
            }
            line.swap(u);
        }
        const auto ids = encode(t, line, specials);
        for (size_t k = 0; k < ids.size(); ++k) {
            std::printf(k ? ",%d" : "%d", ids[k]);
        }
        std::printf("\n");
    }
    return 0;
}

const char * USAGE =
    "usage: ttf-home MODEL \"text\" [options]     run the model (text model: continuation; speech model: -o out.wav)\n"
    "       ttf-home MODEL --chat [--system S] [--think | --no-think] [--effort xhigh|medium|low]   text model, stdin turns\n"
    "       ttf-home MODEL --serve [--host 127.0.0.1] [--port 8080]   OpenAI-style /v1/chat/completions (stream too)\n"
    "       ttf-home MODEL --bench [--pp 512] [--tg 128] [--reps 3] [--depth 16]\n"
    "       ttf-home MODEL --list FILE           speech model, one job per line: out.wav<TAB>text[<TAB>seed]\n"
    "       ttf-home MODEL --tokenize [--specials] [--escapes] < lines\n"
    "options: -n N  --temp T  --top-k K  --seed S  -o FILE  --timesteps N  --cfg F  --max-len N  --retry-badcase N\n"
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
            a.temp     = (float) std::atof(next().c_str());
            a.temp_set = true;
        } else if (o == "--top-k") {
            a.top_k     = std::atoi(next().c_str());
            a.top_k_set = true;
        } else if (o == "--chat") {
            a.chat = true;
        } else if (o == "--serve") {
            a.serve = true;
        } else if (o == "--host") {
            a.host = next();
        } else if (o == "--port") {
            a.port = std::atoi(next().c_str());
        } else if (o == "--system") {
            a.system = next();
        } else if (o == "--think") {
            a.think = 1;
        } else if (o == "--no-think") {
            a.think = 0;
        } else if (o == "--effort") {
            a.effort = next();
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
        } else if (o == "--retry-badcase") {  // speech: attempts on a runaway (VoxCPM default 3; 1 = off)
            a.sp.retry_badcase = std::atoi(next().c_str());
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
        } else if (o == "--depth") {  // --bench: decode after a prefill of this many tokens (default 16)
            a.depth = std::max(1, std::atoi(next().c_str()));
        } else if (o == "--pp") {
            a.pp = std::max(32, std::atoi(next().c_str()));
        } else if (o == "--tg") {
            a.tg = std::max(1, std::atoi(next().c_str()));
        } else if (o == "--reps") {
            a.reps = std::max(1, std::atoi(next().c_str()));
        } else if (o == "--tokenize") {
            a.tokenize = true;
        } else if (o == "--escapes") {  // --tokenize: \n \r \t \\ in a line are those characters
            a.escapes = true;
        } else if (o == "--specials") {  // --tokenize: add the special tokens (BOS ...) as HF encode does
            a.specials = true;
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
            return run_tokenize(a.model, a.specials, a.escapes);
        }
        if (a.graph.empty() && !a.bench && !a.chat && !a.serve && a.text.empty() && a.ids.empty() && a.list.empty()) {
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
        if (drv == "lm" && a.chat) {
            return run_chat(*m, a);
        }
        if (drv == "lm" && a.serve) {
            return run_serve(*m, a);
        }
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
