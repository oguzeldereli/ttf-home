// ttf-speak: a .ttf lm_diffusion model (VoxCPM2) on text token ids -> 16-bit PCM WAV.
//   ttf-speak MODEL --ids 1,2,3 --out x.wav [--seed N] [--timesteps N] [--cfg F] [--max-len N] [--threads N]
//             [--device cpu|cuda] [--noise FILE (f32 [patches, P, D])] [--feat FILE (writes the latents)] [-v]
#include "ttf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool write_wav(const std::string & path, const std::vector<float> & x, int sr) {
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

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL --ids 1,2,3 --out x.wav [options]\n", argv[0]);
        return 1;
    }
    std::string          path = argv[1], out = "out.wav", device = "cpu", noise_path, feat_path;
    std::vector<int32_t> ids;
    int                  n_threads = 4;
    ttf::SpeakParams     p;
    for (int i = 2; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: needs a value\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--ids") {
            std::stringstream ss(next());
            std::string       tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) {
                    ids.push_back((int32_t) std::strtol(tok.c_str(), nullptr, 10));
                }
            }
        } else if (a == "--out") {
            out = next();
        } else if (a == "--seed") {
            p.seed = (uint32_t) std::strtoul(next().c_str(), nullptr, 10);
        } else if (a == "--timesteps") {
            p.timesteps = std::atoi(next().c_str());
        } else if (a == "--cfg") {
            p.cfg = std::atof(next().c_str());
        } else if (a == "--max-len") {
            p.max_len = std::atoi(next().c_str());
        } else if (a == "--threads") {
            n_threads = std::atoi(next().c_str());
        } else if (a == "--device") {
            device = next();
        } else if (a == "--noise") {
            noise_path = next();
        } else if (a == "--feat") {
            feat_path = next();
        } else if (a == "-v") {
            p.verbose = true;
        } else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 1;
        }
    }
    std::vector<float> noise;
    if (!noise_path.empty()) {
        std::ifstream f(noise_path, std::ios::binary | std::ios::ate);
        noise.resize((size_t) f.tellg() / 4);
        f.seekg(0);
        f.read((char *) noise.data(), noise.size() * 4);
        p.noise = &noise;
    }
    std::string err;
    auto        m = ttf::Model::load(path, &err, device);
    if (!m) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    try {
        ttf::Session  s(*m, n_threads, 4096);
        ttf::SpeakOut o   = ttf::speak(s, ids, p);
        const double  sec = (double) o.wav.size() / o.sample_rate;
        const double  ms  = o.ms_prefill + o.ms_dit + o.ms_lm + o.ms_vae;
        std::fprintf(stderr,
                     "%d patches, %.2f s audio; prefill %.0f ms, dit %.0f ms, lm %.0f ms (locenc %.0f), vae %.0f ms; RTF %.3f\n",
                     o.n_patches, sec, o.ms_prefill, o.ms_dit, o.ms_lm, o.ms_enc, o.ms_vae, ms / 1000.0 / sec);
        std::fprintf(stderr, "session: build %.0f ms, alloc %.0f ms (allocator %.0f), compute %.0f ms\n", s.prof_build,
                     s.prof_alloc, s.prof_galloc, s.prof_compute);
        if (!write_wav(out, o.wav, o.sample_rate)) {
            std::fprintf(stderr, "cannot write %s\n", out.c_str());
            return 1;
        }
        if (!feat_path.empty()) {
            FILE * f = std::fopen(feat_path.c_str(), "wb");
            std::fwrite(o.feat.data(), 4, o.feat.size(), f);
            std::fclose(f);
        }
        std::fprintf(stderr, "wrote %s\n", out.c_str());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
