// lm_diffusion driver (VoxCPM2), zero-shot and reference (voice cloning) modes: the IR graphs in the order of
// VoxCPM2Model._inference.
//   reference: the clip -> vae_encode -> patches; timeline [ref_start, patches, ref_end, text, audio_start], the
//              patch rows through locenc (embd) with fsq on their base output, as VoxCPM2._make_ref_prefix
//   prefill: text embed -> base LM (causal, kv.base) -> residual LM (kv.res) on cat(h, 0)
//   per patch: mu = dit_cond(lm, res); Euler CFM over `dit` (B = 2: conditional + mu = 0, CFG-zero-star);
//              stop head on lm; locenc(patch) -> base step -> fsq -> residual step
//   then vae_decode on all patches.
#include "ttf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

extern "C" void ggml_ttf_w8a8_enable(int on);  // ggml-cpu/ttf-w8a8.cpp

namespace ttf {

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

const Tensor & out1(const Values & o) {
    if (o.size() != 1) {
        throw std::runtime_error("speak: graph with " + std::to_string(o.size()) + " outputs");
    }
    return o.begin()->second;
}

std::vector<float> last_row(const Tensor & t) {  // [T, H] -> row T-1
    const int64_t H = t.shape.back();
    return std::vector<float>(t.f.end() - H, t.f.end());
}

}  // namespace

namespace {

// the reference clip as latent patches [n, P, D] (VoxCPM2._encode_wav, padding right to whole patches)
std::vector<float> encode_ref(Session & s, const std::vector<float> & wav16) {
    const json & a   = s.m.ir.at("driver").at("attr");
    const json & gr  = s.m.ir.at("driver").at("graphs");
    const int    P   = a.at("patch_size"), D = a.at("feat_dim"), hop = a.value("chunk_size", 640);
    if (!gr.contains("vae_encode") || !s.m.ir.at("graphs").contains(gr.at("vae_encode").get<std::string>())) {
        throw std::runtime_error("voice cloning needs the audio VAE's encoder: a VAE file with it (-vae-*.ttf from 17 Nov "
                                 "2026 on) or the single-file model");
    }
    const int64_t pl = (int64_t) P * hop;
    std::vector<float> w = wav16;
    if (w.empty()) {
        throw std::runtime_error("reference audio is empty");
    }
    if (w.size() % pl) {
        w.resize(w.size() + (pl - w.size() % pl), 0.0f);
    }
    Values in;
    in["wav"]      = Tensor::f32({ 1, (int64_t) w.size() }, w);
    const Tensor z = out1(s.run(gr.at("vae_encode").get<std::string>(), in));  // [D, frames]
    const int64_t F = z.shape.back(), n = F / P;
    if (z.shape.front() != D || n < 1) {
        throw std::runtime_error("vae_encode: unexpected output shape");
    }
    std::vector<float> out((size_t) (n * P * D));  // z.view(D, n, P).permute(1, 2, 0)
    for (int64_t t = 0; t < n; ++t) {
        for (int pp = 0; pp < P; ++pp) {
            for (int d = 0; d < D; ++d) {
                out[(t * P + pp) * D + d] = z.f[d * F + t * P + pp];
            }
        }
    }
    return out;
}

SpeakOut speak_once(Session & s, const std::vector<int32_t> & text_ids, const SpeakParams & p,
                    const std::vector<float> * ref) {
    const json & drv = s.m.ir.at("driver");
    if (drv.at("name") != "lm_diffusion") {
        throw std::runtime_error("speak: driver is " + drv.at("name").get<std::string>());
    }
    const json & a  = drv.at("attr");
    const json & gr = drv.at("graphs");
    auto         G  = [&](const char * k) { return gr.at(k).get<std::string>(); };
    const int    P = a.at("patch_size"), Dl = a.at("feat_dim");
    const int    n_ts    = p.timesteps > 0 ? p.timesteps : a.value("timesteps", 10);
    const float  cfg     = p.cfg >= 0 ? p.cfg : a.value("cfg", 2.0f);
    const float  sway    = a.value("sway", 1.0f);
    const float  temp    = p.temperature >= 0 ? p.temperature : a.value("temperature", 1.0f);
    const bool   zstar   = a.value("cfg_zero_star", true);
    const float  zfrac   = a.value("zero_init_frac", 0.04f);
    const int    min_len = p.min_len >= 0 ? p.min_len : a.value("min_len", 2);
    // VoxCPM2.generate: _inference's max_len = min(int(text tokens * ratio + 10), max_len), retry or not
    const int    max_len = std::min(p.max_len > 0 ? p.max_len : a.value("max_len", 2000),
                                    (int) ((double) text_ids.size() * p.badcase_ratio + 10));
    if (a.value("solver", std::string("euler")) != "euler" || a.value("mean_mode", false)) {
        throw std::runtime_error("speak: solver / mean_mode not supported");
    }
    const int64_t PD = (int64_t) P * Dl;

    SpeakOut out;
    s.reset();
    auto t0 = clk::now();

    // prefill. Zero-shot: text ids + audio_start, all text (feat_mask 0, so feat_embed drops out). Reference mode:
    // [ref_start, n x 0, ref_end] + that, the n rows audio (feat_mask 1): embd from locenc, lm_out fsq(h), res feat embd
    const int64_t        nr  = ref ? (int64_t) ref->size() / PD : 0;
    std::vector<int32_t> ids;
    if (nr) {
        ids.push_back(a.value("ref_audio_start", 103));
        ids.insert(ids.end(), nr, 0);
        ids.push_back(a.value("ref_audio_end", 104));
    }
    ids.insert(ids.end(), text_ids.begin(), text_ids.end());
    ids.push_back(a.at("audio_start").get<int32_t>());
    const int64_t        T = (int64_t) ids.size();
    std::vector<int32_t> pos(T);
    for (int64_t i = 0; i < T; ++i) {
        pos[i] = (int32_t) i;
    }
    Values in;
    in["ids"]       = Tensor::i32({ T }, ids);
    Tensor e        = out1(s.run(G("text_embed"), in));
    const float sem = a.value("scale_emb", 1.0f);
    if (sem != 1.0f) {
        for (float & v : e.f) {
            v *= sem;
        }
    }
    const int64_t H = e.shape.back();
    Tensor        fe = Tensor::f32({ T, H }, std::vector<float>(T * H, 0.0f));  // feat_mask * feat_embed
    if (nr) {
        auto te = clk::now();
        in.clear();
        in["feat"]    = Tensor::f32({ nr, P, Dl }, *ref);
        const Tensor ce = out1(s.run(G("locenc"), in));  // [nr, H]
        out.ms_enc += ms_since(te);
        for (int64_t i = 0; i < nr; ++i) {  // rows 1 .. nr: the patches replace the (zero-id) text embedding
            std::memcpy(e.f.data() + (1 + i) * H, ce.f.data() + i * H, H * sizeof(float));
            std::memcpy(fe.f.data() + (1 + i) * H, ce.f.data() + i * H, H * sizeof(float));
        }
    }
    in.clear();
    in["embd"]      = e;
    in["pos"]       = Tensor::i32({ T }, pos);
    Tensor h        = out1(s.run(G("base_step"), in));  // lm_out: text rows keep h, audio rows fsq(h)
    if (nr) {
        in.clear();
        in["h"]        = Tensor::f32({ nr, H }, std::vector<float>(h.f.begin() + H, h.f.begin() + (1 + nr) * H));
        const Tensor q = out1(s.run(G("fsq"), in));
        std::memcpy(h.f.data() + H, q.f.data(), nr * H * sizeof(float));
    }
    in.clear();
    in["lm"]   = h;
    in["feat"] = fe;
    in["pos"]  = Tensor::i32({ T }, pos);
    Tensor r   = out1(s.run(G("res_step"), in));
    std::vector<float> lm = last_row(h), res = last_row(r);
    out.ms_prefill = ms_since(t0);

    // Euler time grid: linspace(1, 0, n + 1) + sway (cos(pi / 2 t) - 1 + t), fp32 as torch
    std::vector<float> ts(n_ts + 1);
    for (int i = 0; i <= n_ts; ++i) {
        // torch linspace (CPU): step = (end - start) / (n - 1); first half start + i step, second half end - (n - 1 - i) step
        const float stp = -1.0f / (float) n_ts;
        const float t   = i < (n_ts + 1) / 2 ? 1.0f + stp * (float) i : 0.0f - stp * (float) (n_ts - i);
        ts[i]         = t + sway * (std::cos((float) M_PI / 2.0f * t) - 1.0f + t);
    }
    const int zero_steps = std::max(1, (int) ((float) (n_ts + 1) * zfrac));

    std::mt19937                    rng(p.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              cond(PD, 0.0f);  // prefix_feat_cond: zero-shot starts from zeros
    int64_t                         pi = (int64_t) T;

    for (int step = 0; step < max_len; ++step) {
        auto tc = clk::now();
        in.clear();
        in["lm"]  = Tensor::f32({ 1, H }, lm);
        in["res"] = Tensor::f32({ 1, H }, res);
        Tensor mu = out1(s.run(G("dit_cond"), in));  // [1, Hd]
        const int64_t Hd = mu.shape.back();

        std::vector<float> x(PD);
        if (p.noise) {
            if ((int64_t) p.noise->size() < (step + 1) * PD) {
                throw std::runtime_error("speak: fed noise too short");
            }
            for (int64_t j = 0; j < PD; ++j) {
                x[j] = (*p.noise)[step * PD + j] * temp;
            }
        } else {
            for (float & v : x) {
                v = nd(rng) * temp;
            }
        }
        float t = ts[0], dt = ts[0] - ts[1];
        Tensor xin = Tensor::f32({ 2, P, Dl }, std::vector<float>(2 * PD));
        Tensor cin = Tensor::f32({ 2, P, Dl }, std::vector<float>(2 * PD));
        Tensor min = Tensor::f32({ 2, Hd }, std::vector<float>(2 * Hd, 0.0f));
        std::copy(mu.f.begin(), mu.f.end(), min.f.begin());
        std::copy(cond.begin(), cond.end(), cin.f.begin());
        std::copy(cond.begin(), cond.end(), cin.f.begin() + PD);
        for (int k = 1; k <= n_ts; ++k) {
            if (!(zstar && k <= zero_steps)) {
                std::copy(x.begin(), x.end(), xin.f.begin());
                std::copy(x.begin(), x.end(), xin.f.begin() + PD);
                in.clear();
                in["x"]    = xin;
                in["mu"]   = min;
                in["t"]    = Tensor::f32({ 2 }, { t, t });
                in["cond"] = cin;
                in["dt"]   = Tensor::f32({ 2 }, { 0.0f, 0.0f });  // mean_mode off: dt input is zero
                Tensor v   = out1(s.run(G("dit"), in));
                const float * pv = v.f.data();
                const float * nv = v.f.data() + PD;
                float         st = 1.0f;
                if (zstar) {
                    float dot = 0.0f, nn = 0.0f;
                    for (int64_t j = 0; j < PD; ++j) {
                        dot += pv[j] * nv[j];
                        nn += nv[j] * nv[j];
                    }
                    st = dot / (nn + 1e-8f);
                }
                for (int64_t j = 0; j < PD; ++j) {
                    const float d = nv[j] * st + cfg * (pv[j] - nv[j] * st);
                    x[j]          = x[j] - dt * d;
                }
            }
            t = t - dt;
            if (k < n_ts) {
                dt = t - ts[k + 1];
            }
        }
        out.ms_dit += ms_since(tc);
        out.feat.insert(out.feat.end(), x.begin(), x.end());
        out.n_patches++;
        cond = x;

        // stop head on the current lm hidden (before this patch is fed back)
        auto tl = clk::now();
        in.clear();
        in["h"]          = Tensor::f32({ 1, H }, lm);
        Tensor lg        = out1(s.run(G("stop"), in));
        const bool stopf = lg.f.size() >= 2 && lg.f[1] > lg.f[0];
        if (p.verbose) {
            std::fprintf(stderr, "patch %d: stop logits %.3f %.3f\n", step, lg.f[0], lg.f[1]);
        }
        if (step > min_len && stopf) {
            out.ms_lm += ms_since(tl);
            break;
        }
        // feed the patch back: locenc -> base step -> fsq -> residual step
        auto te = clk::now();
        in.clear();
        in["feat"]       = Tensor::f32({ 1, P, Dl }, x);
        Tensor ce        = out1(s.run(G("locenc"), in));  // [1, H]
        out.ms_enc += ms_since(te);
        in.clear();
        in["embd"]       = ce;
        in["pos"]        = Tensor::i32({ 1 }, { (int32_t) pi });
        Tensor hb        = out1(s.run(G("base_step"), in));
        in.clear();
        in["h"]          = hb;
        Tensor hf        = out1(s.run(G("fsq"), in));
        in.clear();
        in["lm"]   = hf;
        in["feat"] = ce;
        in["pos"]  = Tensor::i32({ 1 }, { (int32_t) pi });
        Tensor hr  = out1(s.run(G("res_step"), in));
        lm         = hf.f;
        res        = hr.f;
        ++pi;
        out.ms_lm += ms_since(tl);
    }

    if (p.skip_vae) {
        return out;
    }
    // vae: z [D, patches * P] (b d (t p))
    auto          tv = clk::now();
    const int64_t L  = (int64_t) out.n_patches * P;
    Tensor        z  = Tensor::f32({ Dl, L }, std::vector<float>(Dl * L));
    for (int64_t tp = 0; tp < L; ++tp) {
        for (int d = 0; d < Dl; ++d) {
            z.f[d * L + tp] = out.feat[tp * Dl + d];
        }
    }
    in.clear();
    in["z"]         = z;
    out.wav         = out1(s.run(G("vae_decode"), in)).f;
    out.sample_rate = a.value("out_sample_rate", 48000);
    out.ms_vae      = ms_since(tv);
    return out;
}

}  // namespace

// VoxCPM2.generate's retry_badcase: an attempt whose patches reach text tokens * ratio is a runaway; the next attempt
// uses seed + 1 (up to retry_badcase attempts; the last one is kept)
SpeakOut speak(Session & s, const std::vector<int32_t> & text_ids, const SpeakParams & p) {
    ggml_ttf_w8a8_enable(1);  // CPU: W8A8 for the DiT's Q8_0 matmuls (TTF_W8A8=0 keeps ggml's Q8_0)
    SpeakParams        q = p;
    std::vector<float> ref;
    double             ms_ref = 0;
    if (p.ref_audio) {
        auto t0 = clk::now();
        ref     = encode_ref(s, *p.ref_audio);
        ms_ref  = ms_since(t0);
    }
    for (int k = 0;; ++k) {
        SpeakOut o = speak_once(s, text_ids, q, p.ref_audio ? &ref : nullptr);
        o.ms_enc += ms_ref;
        o.attempts = k + 1;
        o.seed     = q.seed;
        if (q.noise || k + 1 >= std::max(1, p.retry_badcase) ||
            (double) o.n_patches < (double) text_ids.size() * p.badcase_ratio) {
            return o;
        }
        if (p.verbose) {
            std::fprintf(stderr, "speak: badcase (%d patches for %zu text tokens), retrying with seed %u\n", o.n_patches,
                         text_ids.size(), q.seed + 1);
        }
        q.seed += 1;
    }
}


namespace {

template <class T> T rd(const unsigned char * p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

}  // namespace

std::vector<float> load_audio(const std::string & path, int sr) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<unsigned char> b;
    unsigned char              buf[1 << 16];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) {
        b.insert(b.end(), buf, buf + n);
    }
    std::fclose(f);
    if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) || std::memcmp(b.data() + 8, "WAVE", 4)) {
        throw std::runtime_error(path + ": not a WAV file (convert it, e.g. ffmpeg -i IN -ar 16000 -ac 1 OUT.wav)");
    }
    int      fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    const unsigned char * data = nullptr;
    size_t                dlen = 0;
    for (size_t o = 12; o + 8 <= b.size();) {
        const uint32_t len = rd<uint32_t>(b.data() + o + 4);
        const unsigned char * c = b.data() + o + 8;
        if (!std::memcmp(b.data() + o, "fmt ", 4) && len >= 16) {
            fmt  = rd<uint16_t>(c);
            ch   = rd<uint16_t>(c + 2);
            rate = rd<uint32_t>(c + 4);
            bits = rd<uint16_t>(c + 14);
            if (fmt == 0xFFFE && len >= 26) {  // WAVE_FORMAT_EXTENSIBLE: the sub-format's first two bytes
                fmt = rd<uint16_t>(c + 24);
            }
        } else if (!std::memcmp(b.data() + o, "data", 4)) {
            data = c;
            dlen = std::min<size_t>(len, b.size() - (o + 8));
            break;
        }
        o += 8 + len + (len & 1);
    }
    if (!data || ch < 1 || !rate || !((fmt == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) ||
                                      (fmt == 3 && (bits == 32 || bits == 64)))) {
        throw std::runtime_error(path + ": unsupported WAV (PCM 8/16/24/32-bit or float; convert with ffmpeg -i IN -ar 16000 -ac 1 OUT.wav)");
    }
    const size_t       bps = bits / 8, n = dlen / (bps * ch);
    std::vector<float> x(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        double acc = 0;
        for (int k = 0; k < ch; ++k) {
            const unsigned char * q = data + (i * ch + k) * bps;
            double v;
            if (fmt == 3) {
                v = bits == 32 ? rd<float>(q) : rd<double>(q);
            } else if (bits == 8) {
                v = ((int) q[0] - 128) / 128.0;
            } else if (bits == 16) {
                v = rd<int16_t>(q) / 32768.0;
            } else if (bits == 24) {
                v = (double) ((int32_t) ((uint32_t) q[0] << 8 | (uint32_t) q[1] << 16 | (uint32_t) q[2] << 24) >> 8) / 8388608.0;
            } else {
                v = rd<int32_t>(q) / 2147483648.0;
            }
            acc += v;
        }
        x[i] = (float) (acc / ch);
    }
    if ((int) rate == sr) {
        return x;
    }
    // windowed-sinc resampling (Kaiser beta 8, 32 zero crossings of the lower rate's band, cutoff 0.97 of its Nyquist)
    const double ratio = (double) sr / rate, fc = 0.97 * std::min(1.0, ratio), beta = 8.0;
    const int    W     = (int) std::ceil(32.0 / fc);
    auto         i0    = [](double z) {  // modified Bessel I0
        double s = 1, t = 1;
        for (int k = 1; k < 30; ++k) {
            t *= (z / (2 * k)) * (z / (2 * k));
            s += t;
        }
        return s;
    };
    const double       ib = 1.0 / i0(beta);
    std::vector<double> kw(4097);  // the Kaiser window on |u| in [0, 1], interpolated
    for (int q = 0; q <= 4096; ++q) {
        const double u = q / 4096.0;
        kw[q]          = i0(beta * std::sqrt(std::max(0.0, 1 - u * u))) * ib;
    }
    const size_t       m  = (size_t) std::floor((double) n * ratio);
    std::vector<float> y(m);
    for (size_t j = 0; j < m; ++j) {
        const double  t  = j / ratio;
        const int64_t c  = (int64_t) std::floor(t);
        double        acc = 0, ws = 0;
        for (int64_t k = c - W + 1; k <= c + W; ++k) {
            const double d = t - (double) k;
            const double u = d / W;
            if (u <= -1 || u >= 1) {
                continue;
            }
            const double xs = fc * d;
            const double h  = (xs == 0 ? 1.0 : std::sin(M_PI * xs) / (M_PI * xs)) * [&] {
                const double au = std::fabs(u) * 4096.0;
                const int    q  = std::min(4095, (int) au);
                return kw[q] + (kw[q + 1] - kw[q]) * (au - q);
            }();
            ws += h;
            if (k >= 0 && k < (int64_t) n) {
                acc += h * x[k];
            }
        }
        y[j] = (float) (ws != 0 ? acc / ws : 0.0);
    }
    return y;
}

}  // namespace ttf
