// ttf-run: run a .ttf model's lm driver on token ids (greedy), dump logits for parity.
//   ttf-run MODEL --ids 1,2,3 [--gen N] [--threads N] [--ctx N] [--dump FILE] [--device cpu|cuda]
//   --dump FILE: the prefill logits, f32 [T, n_vocab] raw
//   ttf-run MODEL --graph NAME --in NAME=FILE ... [--out DIR]: one call of one graph on raw f32 / i32 inputs
//   (shape from the IR, the dynamic dim from the file size); outputs to DIR/NAME.bin

#include "ttf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int32_t> parse_ids(const std::string & s) {
    std::vector<int32_t> out;
    std::stringstream    ss(s);
    std::string          tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            out.push_back((int32_t) std::strtol(tok.c_str(), nullptr, 10));
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL --ids 1,2,3 [--gen N] [--threads N] [--ctx N] [--dump FILE]\n", argv[0]);
        return 1;
    }
    std::string          path = argv[1];
    std::vector<int32_t> ids;
    int                  n_gen = 0, n_threads = 4, check_dec = 0;
    int64_t              n_ctx = 2048;
    std::string          dump, device = "cpu", graph_name, out_dir = ".";
    std::map<std::string, std::string> gin;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto        next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: needs a value\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--ids") {
            ids = parse_ids(next());
        } else if (a == "--gen") {
            n_gen = std::atoi(next().c_str());
        } else if (a == "--threads") {
            n_threads = std::atoi(next().c_str());
        } else if (a == "--ctx") {
            n_ctx = std::atoll(next().c_str());
        } else if (a == "--check-decode") {
            check_dec = std::atoi(next().c_str());
        } else if (a == "--device") {
            device = next();
        } else if (a == "--dump") {
            dump = next();
        } else if (a == "--graph") {
            graph_name = next();
        } else if (a == "--in") {
            const std::string v = next();
            const size_t      e = v.find('=');
            gin[v.substr(0, e)] = v.substr(e + 1);
        } else if (a == "--out") {
            out_dir = next();
        } else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 1;
        }
    }
    if (ids.empty() && graph_name.empty()) {
        std::fprintf(stderr, "--ids or --graph required\n");
        return 1;
    }
    std::string err;
    auto        m = ttf::Model::load(path, &err, device);
    if (!m) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    if (!graph_name.empty()) {
        try {
            ttf::Values in;
            for (const auto & spec : m->ir.at("graphs").at(graph_name).at("inputs")) {
                const std::string nm = spec.at("name");
                if (!gin.count(nm)) {
                    std::fprintf(stderr, "--in %s=FILE missing\n", nm.c_str());
                    return 1;
                }
                std::ifstream     f(gin[nm], std::ios::binary | std::ios::ate);
                const int64_t     nb  = f.tellg();
                const bool        i32 = spec.value("dtype", std::string("f32")) == "i32";
                int64_t           known = 1, dyn = -1;
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
            ttf::Session s(*m, n_threads, n_ctx);
            ttf::Values  o = s.run(graph_name, in, false);
            std::fprintf(stderr, "%s: %.1f ms\n", graph_name.c_str(), s.last_ms);
            for (auto & [nm, t] : o) {
                const std::string fn = out_dir + "/" + nm + ".bin";
                FILE *            f  = std::fopen(fn.c_str(), "wb");
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
        } catch (const std::exception & e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
        return 0;
    }
    const auto & drv = m->ir.at("driver");
    if (drv.at("name") != "lm") {
        std::fprintf(stderr, "driver %s: not yet\n", drv.at("name").get<std::string>().c_str());
        return 1;
    }
    const std::string graph = drv.at("graphs").at("step");
    std::fprintf(stderr, "model %s: family %s, D %s, %zu parts, %s, trits %s\n", path.c_str(), m->family.c_str(),
                 m->d_bf16 ? "bf16" : "f32", m->parts.size(), m->device.c_str(), m->q2p ? "q2p" : "tq1p");
    try {
        ttf::Session         s(*m, n_threads, n_ctx);
        if (check_dec > 0) {  // prefill all vs prefill all but N then N decode steps: the state path
            const int64_t        L = (int64_t) ids.size(), P0 = L - check_dec;
            std::vector<int32_t> pa(L);
            for (int64_t i = 0; i < L; ++i) {
                pa[i] = (int32_t) i;
            }
            ttf::StepOut full = s.step(graph, ids, pa, false);
            s.reset();
            s.step(graph, std::vector<int32_t>(ids.begin(), ids.begin() + P0), std::vector<int32_t>(pa.begin(), pa.begin() + P0), true);
            for (int64_t t = P0; t < L; ++t) {
                ttf::StepOut o  = s.step(graph, { ids[t] }, { (int32_t) t }, true);
                const float * a = full.logits.data() + t * full.n_vocab;
                const float * b = o.logits.data();
                double        dn = 0, rn = 0;
                int           ia = 0, ib = 0;
                for (int64_t j = 0; j < full.n_vocab; ++j) {
                    dn += (double) (a[j] - b[j]) * (a[j] - b[j]);
                    rn += (double) a[j] * a[j];
                    ia = a[j] > a[ia] ? (int) j : ia;
                    ib = b[j] > b[ib] ? (int) j : ib;
                }
                std::printf("pos %lld: decode vs prefill rel %.3e, argmax %d / %d\n", (long long) t, std::sqrt(dn / rn), ia, ib);
            }
            return 0;
        }
        std::vector<int32_t> pos(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            pos[i] = (int32_t) i;
        }
        ttf::StepOut r = s.step(graph, ids, pos, dump.empty());
        std::fprintf(stderr, "prefill %zu tokens: %.1f ms\n", ids.size(), s.last_ms);
        if (!dump.empty()) {
            FILE * f = std::fopen(dump.c_str(), "wb");
            if (!f) {
                std::fprintf(stderr, "cannot write %s\n", dump.c_str());
                return 1;
            }
            std::fwrite(r.logits.data(), sizeof(float), r.logits.size(), f);
            std::fclose(f);
            std::fprintf(stderr, "dumped logits [%lld, %lld] to %s\n", (long long) r.T_out, (long long) r.n_vocab,
                         dump.c_str());
        }
        auto argmax_last = [](const ttf::StepOut & o) {
            const float * l  = o.logits.data() + (o.T_out - 1) * o.n_vocab;
            int32_t       bi = 0;
            for (int64_t j = 1; j < o.n_vocab; ++j) {
                if (l[j] > l[bi]) {
                    bi = (int32_t) j;
                }
            }
            return bi;
        };
        int32_t           next = argmax_last(r);
        std::vector<int32_t> gen;
        double            ms = 0;
        int32_t           p  = (int32_t) ids.size();
        s.prof_build = s.prof_alloc = s.prof_compute = 0;
        for (int i = 0; i < n_gen; ++i) {
            gen.push_back(next);
            ttf::StepOut o = s.step(graph, { next }, { p++ }, true);
            ms += s.last_ms;
            next = argmax_last(o);
        }
        if (n_gen > 0) {
            std::printf("gen:");
            for (int32_t t : gen) {
                std::printf(" %d", t);
            }
            std::printf("\n");
            std::fprintf(stderr, "decode %d tokens: %.2f ms/token, %.1f tok/s\n", n_gen, ms / n_gen, 1000.0 * n_gen / ms);
            std::fprintf(stderr, "  per token: build %.2f ms, alloc %.2f ms, compute %.2f ms; %d graph nodes\n",
                         s.prof_build / n_gen, s.prof_alloc / n_gen, s.prof_compute / n_gen, s.prof_nodes);
            std::vector<std::pair<double, std::string>> ops;
            double                                      tot = 0;
            for (auto & [k, v] : s.prof_ops) {
                ops.emplace_back(v.first, k);
                tot += v.first;
            }
            std::sort(ops.rbegin(), ops.rend());
            for (size_t i = 0; i < ops.size() && i < 25; ++i) {
                std::fprintf(stderr, "  %7.3f ms/token %5.1f%%  %s\n", ops[i].first / (n_gen + 1), 100 * ops[i].first / tot,
                             ops[i].second.c_str());
            }
        } else {
            std::printf("next: %d\n", next);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
