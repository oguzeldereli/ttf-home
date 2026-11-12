"""MoE TTF GGUFs (TTF format 2: Qwen3.8-Flash-Next from q38_export.py; Tiel-Coder / Ornith-35B-A3B, qwen35moe, from
the Tiel exporter, compact or not) -> what the q38 trainer (python -m ttf.q38.train) starts from: the exporters' inverse.

  OUT/model/   config.json + tokenizer files (from BASE_DIR) and model-L{i}.safetensors / model-globals.safetensors with
               every float weight under its HF name (model.safetensors.index.json), for --model
  OUT/conv/    L{i}.pt = {module path: state dict} of the layer's TernaryExperts / TernLinear, for --conv
  OUT/pq/      the PLE table's product quantisation (meta.pt, codes.bin, keep.bin, ttf.q38.ple_pq.PQTable), for --ple-pq

Per TTF module (q38_export's docstring): T (TQ1_P, or TQ1_0 with block scale 1), ttf_r = r beta (beta from the trits
per expert), c, m = -c mu, b; the adapter's down trits TB (.ttf_ad) and its up factor .ttf_au = r sa TA cb (F16): TA =
sign(au), sa cb_k = the median of |au[n, k]| / r_n over the rows where TA is nonzero, split as sa = 1 (where the
conversion puts it; training keeps it within 1%) and cb = that / sa: the same function (up to au's fp16 rounding),
the trained sa / cb split within that 1%. The PQ
rotation was folded into ple.key_proj / value_proj, so meta.pt's Pi is the identity. The floats come back through
probes of llama.cpp's own converter (renames, V-head reorders, merges [a | b] and splits), as gguf_import.

Compact files (Tiel v2): an expert's .ttf_ad is TQ1_0 [32, K] with cb_k as row k's block scale (rows past the rank
zero) and its .ttf_au exact Q4_0 [N, 32] of a_n TA[n, :] (a = r sa): cb = |row scale| (its sign into TB), TA = sign,
then the same sa / cb split; the expert side vectors are F16. Trits that are not TQ1_P (K % 160 != 0) are TQ1_0 with
block scale 1.

Not recoverable: the TTF / AdamW state (a fresh start), the MTP block, the PLE table's pruned rows (zero, as in the
GGUF), the float tensors' storage rounding (the embedding / head as stored, Q8_0 by default).

  python -m ttfhome.q38_import MODEL.gguf BASE_DIR OUT_DIR [--layers N]
BASE_DIR: the base model's config.json and tokenizer files (no weights).
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .cli import engine_dir                    # noqa: E402
LLAMA = engine_dir(q38=True)
sys.path[:0] = [str(LLAMA / "gguf-py"), str(LLAMA)]
import gguf                                    # noqa: E402
from conversion import get_model_class         # noqa: E402
from .gguf_import import GGUF, Maps, dec_tq1p  # noqa: E402
G_TQ1_P = getattr(gguf.GGMLQuantizationType, "TQ1_P", None)

PFX = "model.language_model."
EXPERTS = {"mlp.gu": "ffn_gate_up_exps", "mlp.dn": "ffn_down_exps"}


def _converter(base_dir, tmp):
    for f in Path(base_dir).iterdir():
        if f.is_file() and not f.name.endswith((".safetensors", ".bin", ".pt")) and f.name != "model.safetensors.index.json":
            (tmp / f.name).symlink_to(f.resolve())
    cfg = json.loads((Path(base_dir) / "config.json").read_text())
    cfg.pop("quantization_config", None)
    (tmp / "config.json").unlink()
    (tmp / "config.json").write_text(json.dumps(cfg))
    base = get_model_class(cfg["architectures"][0])
    base.no_mtp = True

    class Probe(base):
        model_arch = base.model_arch
        _SHARED_INPUT_MERGES = tuple(m for m in getattr(base, "_SHARED_INPUT_MERGES", ())
                                     if m[0] != gguf.MODEL_TENSOR.ATTN_QKV_GATE)

        def index_tensors(self, remote_hf_model_id=None):
            hp = {**self.hparams, **self.hparams.get("text_config", {})}
            type(self)._original_block_count = hp.get("num_hidden_layers")
            return {}
    return Probe(tmp, gguf.LlamaFileType.ALL_F32, tmp / "probe.gguf", eager=True), cfg


def probe_group(conv, items):
    """several HF float tensors that the converter may merge / split / reorder together (one layer's):
    items [(hf name, shape)] -> {gguf name: (source index [gguf shape] into items, flat element index, inverse map)}"""
    def run(make):
        out = {}
        for j, (name, shape) in enumerate(items):
            x = make(j, shape)
            it = conv.filter_tensors((name, lambda x=x: x))
            if it is None:
                continue
            hn = it[0]
            bid = next((int(p) for p in hn.split(".") if p.isdecimal()), None)
            for gn, t in conv.modify_tensors(x, hn, bid):
                out[gn] = torch.as_tensor(t).float()
        return out
    n_of = [int(np.prod(s)) for _, s in items]
    Z = run(lambda j, s: torch.zeros(s))
    O = run(lambda j, s: torch.ones(s))
    I = run(lambda j, s: torch.full(s, float(j)))
    lo = run(lambda j, s: (torch.arange(int(np.prod(s))) % 1024).float().reshape(s))
    hi = run(lambda j, s: (torch.arange(int(np.prod(s))) // 1024).float().reshape(s))
    res = {}
    for gn in Z:
        z, o = Z[gn], O[gn]
        if torch.allclose(z, torch.full_like(z, -1.0)) and torch.allclose(o, torch.full_like(o, -math.e)):
            inv = lambda y: torch.log(-y)                         # -exp(A_log)
        else:
            a = o - z
            assert bool((a.abs() > 1e-6).all()), f"{gn}: not an affine map of its inputs"
            inv = lambda y, a=a, b=z: (y - b) / a
        src = inv(I[gn]).round().long()
        idx = inv(hi[gn]).round().long() * 1024 + inv(lo[gn]).round().long()     # (int64: > 2^24 elements)
        assert int(src.min()) >= 0 and int(src.max()) < len(items), gn
        res[gn] = (src, idx, inv)
    return res, n_of


def _ta_sa_cb(au, rs, R):
    """au [E, N, R] = rs[n] sa TA[n, k] cb[k] -> (TA int8, sa [E], cb [E, R])"""
    TA = torch.sign(au).to(torch.int8)
    m = (au.abs() / rs.abs().clamp_min(1e-30)[..., None])
    m = torch.where(TA != 0, m, torch.full_like(m, float("nan")))
    v = m.nanmedian(1).values                                   # [E, R]: sa cb_k
    v = torch.nan_to_num(v, nan=0.0)
    sa = torch.ones(au.shape[0])                                # (the conversion's sa, which training keeps near 1)
    return TA, sa, v / sa[:, None]


def _trits3(G, n, e0, e1):
    """experts e0:e1 of a 3-D ternary tensor -> (int8 [e1 - e0, N, K], row scales [e1 - e0, N] or None for TQ1_P),
    decoded per expert slice (bounded memory). TQ1_0: T = round(w / s), s = the row's absmax (its block scale)"""
    t = G.t[n]
    K, N = int(t.shape[0]), int(t.shape[1])
    raw = np.asarray(t.data)
    a = np.ascontiguousarray(raw.reshape(-1, N, raw.shape[-1])[e0:e1])
    if t.tensor_type == G_TQ1_P:
        return dec_tq1p(torch.from_numpy(a.view(np.uint8)).reshape(-1, K // 160 * 32), K).view(e1 - e0, N, K), None
    assert t.tensor_type == gguf.GGMLQuantizationType.TQ1_0, (n, t.tensor_type)
    w = torch.from_numpy(gguf.quants.dequantize(a.reshape(-1, a.shape[-1]), t.tensor_type).astype(np.float32)).view(e1 - e0, N, K)
    s = w.abs().amax(-1)
    T = torch.where(s[..., None] > 0, w / s.clamp_min(1e-30)[..., None], torch.zeros_like(w)).round()
    assert bool((T * s[..., None] - w).abs().le(1e-3 * s[..., None].clamp_min(1e-30)).all()), f"{n}: not trits x a row scale"
    return T.to(torch.int8), s


def _read(G, stem, E, K, e0=0, e1=None):
    """the GGUF tensors of one TTF module (or one part of a dense one) under stem, in the GGUF's order (experts e0:e1)"""
    e1 = E if e1 is None else e1
    n = e1 - e0
    if E > 1:
        (Tg, _), (TB, sb) = _trits3(G, stem + ".weight", e0, e1), _trits3(G, stem + ".ttf_ad", e0, e1)
    else:
        Tg, (TB, sb) = G.trits(stem + ".weight")[0][None], G.trits(stem + ".ttf_ad")
        TB, sb = TB[None], None if sb is None else sb[None]
    Tg, TB = Tg[..., :K], TB[..., :K]
    N = Tg.shape[1]
    f = lambda nm, *sh: G.float(stem + nm).reshape(E, *sh)[e0:e1]
    if G.qtype(stem + ".ttf_au") == gguf.GGMLQuantizationType.Q4_0:       # compact: a TA (Q4_0), cb in TB's scales
        R2 = TB.shape[1]
        au = f(".ttf_au", N, R2)
        used = (TB != 0).flatten(2).any(-1) | (au != 0).any(1)            # [n, R2]: rows past the rank are zero
        R = max(8, -(-(int(used.nonzero()[:, 1].max()) + 1 if bool(used.any()) else 8) // 8) * 8)
        TB, au = TB[:, :R], au[..., :R] * sb[:, None, :R]
    else:
        au = f(".ttf_au", N, TB.shape[1])
        if sb is not None and bool((sb > 0).any()):
            assert float((sb[sb > 0] - 1).abs().max()) < 1e-3, f"{stem}.ttf_ad: block scales != 1 (not a compact file)"
    return {"T": Tg, "TB": TB, "r": f(".ttf_r", N), "c": f(".ttf_c", -1)[..., :K],
            "m": f(".ttf_m", -1)[..., :K] if G.has(stem + ".ttf_m") else torch.zeros(n, K),
            "b": f(".ttf_b", N) if G.has(stem + ".ttf_b") else torch.zeros(n, N), "au": au}


def _dense(G, parts, N, K):
    """the GGUF parts [(gguf name, HF rows ri, HF columns ci)] of one dense HF matrix -> its tensors in HF order"""
    out = None
    for gname, ri, ci in parts:
        d = _read(G, gname[:-len(".weight")], 1, len(ci))
        if out is None:
            R = d["TB"].shape[1]
            out = {"T": torch.zeros(1, N, K, dtype=torch.int8), "r": torch.full((1, N), float("nan")),
                   "b": torch.zeros(1, N), "au": torch.zeros(1, N, R), "c": torch.full((1, K), float("nan")),
                   "m": torch.zeros(1, K), "TB": torch.zeros(1, R, K, dtype=torch.int8)}
        out["T"][0, ri[:, None], ci[None, :]] = d["T"][0]
        for k in ("r", "b"):
            out[k][0, ri] = d[k][0]
        out["au"][0, ri] = d["au"][0]
        for k in ("c", "m"):
            out[k][0, ci] = d[k][0]
        out["TB"][0][:, ci] = d["TB"][0]
    assert not bool(out["r"].isnan().any() or out["c"].isnan().any()), f"{parts[0][0]}: rows / columns missing"
    return out


def _module(d, log=print):
    """one TTF module's state dict (TernaryExperts / TernLinear keys) from its tensors (_read / _dense)"""
    from ttf.q38.moe import pack3, _beta_nnz
    Tg, r, c, m, b, TB, au = (d[k] for k in ("T", "r", "c", "m", "b", "TB", "au"))
    E, R = Tg.shape[0], TB.shape[1]
    beta = torch.stack([_beta_nnz(Tg[e]) for e in range(E)])
    rs = r / beta[:, None]
    TA, sa, cb = _ta_sa_cb(au, rs, R)
    return {"wpacked": pack3(Tg.contiguous()), "beta": beta, "row_scale": rs, "col_scale": c,
            "in_mean": torch.where(c != 0, -m / c, torch.zeros_like(m)), "out_bias": b, "ta_packed": pack3(TA.contiguous()), "tb_packed": pack3(TB.contiguous()),
            "ad_cb": cb, "ad_sa": sa}


def import_q38(path, base_dir, out_dir, n_layers=None, log=print, pq_rows=None):
    from safetensors.torch import save_file
    from ttf.q38 import model as QM
    from ttf.q38.train import swap_ternary
    G = GGUF(path)
    assert int(G.kv.get("ttf.format", 0)) == 2, f"{path}: not a TTF format 2 (Flash-Next) GGUF"
    out = Path(out_dir)
    for d in ("model", "conv", "pq"):
        (out / d).mkdir(parents=True, exist_ok=True)
    cfg = QM.text_config(str(base_dir))
    NL = n_layers or cfg.num_hidden_layers
    tmp = Path(tempfile.mkdtemp(prefix="q38imp_"))
    try:
        conv, hcfg = _converter(base_dir, tmp)
        M = Maps(conv)
        cfg.num_hidden_layers = NL
        cfg.layer_types = cfg.layer_types[:NL]
        m = QM.build(cfg, device="meta")
        index = {}
        for i in range(NL):
            L = m.layers[i]
            swap_ternary(L, adapter_rank=16)          # (the structure only; each module's rank comes from its file)
            tern = {n for n, mod in L.named_modules() if hasattr(mod, "wpacked")}
            convd = {}
            for s in sorted(tern):
                mod = L.get_submodule(s)
                E, N, K = mod.E, mod.N, mod.K
                if s in EXPERTS:                                # (64 experts at a time: bounded memory)
                    ch = [_module(_read(G, f"blk.{i}.{EXPERTS[s]}", E, K, e0, min(E, e0 + 64)), log=log)
                          for e0 in range(0, E, 64)]
                    convd[s] = {k: torch.cat([c[k] for c in ch]) for k in ch[0]}
                else:
                    hn = conv.filter_tensors((f"{PFX}layers.{i}.{s}.weight", lambda: None))[0]
                    convd[s] = _module(_dense(G, M.rows_cols(hn, N, K), N, K), log=log)
                want = set(mod.state_dict())
                assert set(convd[s]) == want, (s, set(convd[s]) ^ want)
            torch.save(convd, out / "conv" / f"L{i}.pt")
            # floats: every parameter outside the TTF modules and the PLE table
            items = [(f"{PFX}layers.{i}.{n}", tuple(t.shape)) for n, t in L.named_parameters()
                     if not any(n.startswith(s + ".") for s in tern) and not n.startswith("ple.ple_embedding")]
            fl = _floats(G, conv, items)
            f = f"model-L{i:02d}.safetensors"
            save_file({k: v.contiguous() for k, v in fl.items()}, str(out / "model" / f))
            index.update({k: f for k in fl})
            log(f"layer {i}: {len(convd)} TTF modules, {len(fl)} floats")
        g = {PFX + "embed_tokens.weight": G.float("token_embd.weight").to(torch.bfloat16),
             "lm_head.weight": G.float("output.weight").to(torch.bfloat16)}
        hcm = PFX + ("norm." if getattr(cfg, "plain", False) else "hyper_connection_mixer.")
        g.update(_floats(G, conv, [(hcm + n, tuple(t.shape)) for n, t in m.hyper_connection_mixer.named_parameters()]))
        save_file({k: v.contiguous() for k, v in g.items()}, str(out / "model" / "model-globals.safetensors"))
        index.update({k: "model-globals.safetensors" for k in g})
        (out / "model" / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"ttf_imported": str(path)},
                                                                               "weight_map": index}, indent=1))
        for fn in Path(base_dir).iterdir():
            if fn.is_file() and not fn.name.endswith((".safetensors", ".bin", ".pt")) and fn.name != "model.safetensors.index.json":
                shutil.copy(fn, out / "model" / fn.name)
        if G.has("per_layer_token_embd.pq_cb"):
            _pq(G, out / "pq", log, max_rows=pq_rows)
        (out / "IMPORTED.json").write_text(json.dumps({"from": str(path), "layers": NL, "meta": G.meta}, indent=1))
        log(f"{out}: --model {out / 'model'} --conv {out / 'conv'}" + (f" --ple-pq {out / 'pq'}" if G.has("per_layer_token_embd.pq_cb") else ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _floats(G, conv, items):
    maps, n_of = probe_group(conv, items)
    flat = [torch.full((n,), float("nan")) for n in n_of]
    for gn, (src, idx, inv) in maps.items():
        assert G.has(gn), f"{gn}: not in the GGUF"
        y = inv(G.float(gn).reshape(src.shape))
        for j in torch.unique(src).tolist():
            sel = src == j
            flat[j][idx[sel]] = y[sel]
    out = {}
    for (name, shape), v in zip(items, flat):
        assert not bool(v.isnan().any()), f"{name}: not all elements in the GGUF"
        out[name] = v.reshape(shape).to(torch.bfloat16)
    return out


def _pq(G, d, log, max_rows=None, chunk=1 << 24):
    """per_layer_token_embd.pq_cb [H nS K + 1, D] / pq_blob (keep bitmap u64 [n_w], ranks u32 [n_w], codes [kept, nS])
    -> ple_pq's meta.pt + codes.bin [rows, nS] (pruned rows: code 0) + keep.bin [rows] (Pi = I: the rotation is in
    key / value_proj), streamed (bounded memory). max_rows: only the first rows (a test)"""
    kv = G.kv
    arch = kv["general.architecture"]
    offs = [int(x) for x in kv[f"{arch}.ple.head_offsets"]]
    sizes = [int(x) for x in kv[f"{arch}.ple.head_vocab_sizes"]]
    H, rows = len(offs), max(o + s for o, s in zip(offs, sizes))
    dim = int(kv[f"{arch}.embedding_length_per_layer_input"])
    cb = G.float("per_layer_token_embd.pq_cb")
    assert bool((cb[-1] == 0).all()), "pq_cb: the last (pruned-row) entry is not zero"
    cb = cb[:-1]
    D = cb.shape[1]
    nS = dim // D
    K = cb.shape[0] // (H * nS)
    assert H * nS * K == cb.shape[0], (H, nS, K, cb.shape)
    blob = np.asarray(G.t["per_layer_token_embd.pq_blob"].data).view(np.uint8)
    n_w = (rows + 63) // 64
    bits = blob[:8 * n_w].view(np.uint64)
    rank = blob[8 * n_w:12 * n_w].view(np.uint32)
    codes_k = blob[12 * n_w:]
    n_kept = codes_k.size // nS
    assert codes_k.size == n_kept * nS
    R = min(rows, max_rows or rows)
    fc = np.memmap(d / "codes.bin", dtype=np.uint8, mode="w+", shape=(R, nS))
    fk = np.memmap(d / "keep.bin", dtype=np.uint8, mode="w+", shape=(R,))
    for r0 in range(0, R, chunk):                 # chunk % 64 == 0: whole bitmap words
        r1 = min(R, r0 + chunk)
        w0, w1 = r0 // 64, (r1 + 63) // 64
        k = np.unpackbits(bits[w0:w1].view(np.uint8), bitorder="little")[:r1 - r0].astype(bool)
        pc = np.unpackbits(bits[w0:w1].view(np.uint8), bitorder="little").reshape(-1, 64).sum(1)
        assert int(rank[w0]) == (int(rank[w0 - 1]) + int(pc_prev) if r0 else 0), f"pq_blob ranks at word {w0}"
        if w1 - w0 > 1:
            assert np.array_equal(rank[w0 + 1:w1].astype(np.int64), int(rank[w0]) + np.cumsum(pc[:-1])), "pq_blob ranks"
        pc_prev = int(pc[-1])
        k0, nk = int(rank[w0]), int(k.sum())
        c = np.zeros((r1 - r0, nS), dtype=np.uint8)
        c[k] = codes_k[k0 * nS:(k0 + nk) * nS].reshape(nk, nS)
        fc[r0:r1] = c
        fk[r0:r1] = k
    if R == rows:
        assert int(rank[-1]) + pc_prev == n_kept, ("kept rows", int(rank[-1]) + pc_prev, n_kept)
    fc.flush(); fk.flush(); del fc, fk
    torch.save({"D": D, "K": K, "offs": offs, "total": R,
                "Pi": torch.eye(dim).expand(H, dim, dim).clone(), "cb": cb.reshape(H, nS, K, D).contiguous(),
                "imported": "Pi folded into ple.key_proj / value_proj (the GGUF's)"}, d / "meta.pt")
    log(f"PLE PQ: {H} heads, D {D}, K {K}, {n_kept} of {rows} rows kept" + (f" (first {R} rows written)" if R < rows else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf"); ap.add_argument("base_dir"); ap.add_argument("out_dir")
    ap.add_argument("--layers", type=int, default=None, help="only the first N layers (a test)")
    ap.add_argument("--pq-rows", type=int, default=None, help="only the PLE table's first rows (a test)")
    a = ap.parse_args()
    with torch.inference_mode():
        import_q38(a.gguf, a.base_dir, a.out_dir, a.layers, pq_rows=a.pq_rows)


if __name__ == "__main__":
    main()
