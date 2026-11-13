// lm_diffusion driver (VoxCPM2), zero-shot mode: the IR graphs in the order of VoxCPM2Model._inference.
//   prefill: text embed -> base LM (causal, kv.base) -> residual LM (kv.res) on cat(h, 0)
//   per patch: mu = dit_cond(lm, res); Euler CFM over `dit` (B = 2: conditional + mu = 0, CFG-zero-star);
//              stop head on lm; locenc(patch) -> base step -> fsq -> residual step
//   then vae_decode on all patches.
#include "ttf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

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

SpeakOut speak(Session & s, const std::vector<int32_t> & text_ids, const SpeakParams & p) {
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
    const int    max_len = p.max_len > 0 ? p.max_len : a.value("max_len", 2000);
    if (a.value("solver", std::string("euler")) != "euler" || a.value("mean_mode", false)) {
        throw std::runtime_error("speak: solver / mean_mode not supported");
    }
    const int64_t PD = (int64_t) P * Dl;

    SpeakOut out;
    s.reset();
    auto t0 = clk::now();

    // prefill: zero-shot timeline = text ids + audio_start, all text (feat_mask 0, so feat_embed drops out)
    std::vector<int32_t> ids = text_ids;
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
    in.clear();
    in["embd"]      = e;
    in["pos"]       = Tensor::i32({ T }, pos);
    Tensor h        = out1(s.run(G("base_step"), in));  // lm_out: text rows keep h (no fsq)
    in.clear();
    in["lm"]   = h;
    in["feat"] = Tensor::f32({ T, H }, std::vector<float>(T * H, 0.0f));
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

}  // namespace ttf
