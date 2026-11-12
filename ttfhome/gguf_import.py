"""GGUF -> trainable TTF checkpoint: the inverse of gguf_export.py, so a released GGUF (the only file we publish) can
be fine-tuned with `ttf finetune`.

What the GGUF holds per TTF linear (gguf_export.py's docstring) is the whole trained state of that layer: the trits T,
the row scale r * beta (in the block scale, or in <name>.ttf_r for TQ1_P), the column scale c (.ttf_c), the input mean
mu (from .ttf_m = -c mu) and the bias (.ttf_b); the ternary adapter's down / up layers likewise (.ttf_d*, .ttf_u*,
rank zero-padded), the float adapter (.ttf_fd / .ttf_fu, or .ttf_d / .ttf_u in F16 alone). This module reads them
back, undoes the llama.cpp converter's renames, row / column reorders and affine maps (+1 norm gains, -exp(A_log)) by
pushing index probes through the same converter, and writes the trainer's state dict (both of QwenTern's names for
every decoder tensor) with the args `ttf finetune` needs.

Not recoverable from a GGUF, and so different from the checkpoint the GGUF was made from:
  - the optimizer and the TTF accumulators (the import is a fresh start: --init-ckpt, step 0)
  - float tensors stored lossy (Q6_K / Q4_K embedding or head, F16 matrices): their dequantized values
  - r * beta stored in an fp16 block scale (TQ1_0 / TQ2_0 / Q2_K): r to fp16 precision (TQ1_P's .ttf_r is F32)
  - a row scale's sign (TQ1_0 / TQ2_0 store |d|: the trits of that row flip instead, the same function)
  - a query temperature folded into the query rows (scratch models): it stays folded

  python -m ttfhome.gguf_import MODEL.gguf BASE_HF_DIR OUT.pt
BASE_HF_DIR: the base model's config.json (and tokenizer), as for `ttf export`.
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .cli import engine_dir                    # noqa: E402
LLAMA = engine_dir()
sys.path[:0] = [str(LLAMA / "gguf-py"), str(LLAMA)]
import gguf                                    # noqa: E402
from conversion import get_model_class         # noqa: E402

QT = gguf.GGMLQuantizationType
TQ1_P = getattr(QT, "TQ1_P", None)
TERN_TYPES = {QT.TQ1_0, QT.TQ2_0, QT.Q2_K, QT.Q8_0} | ({TQ1_P} if TQ1_P is not None else set())
FLOAT_TYPES = {QT.F32, QT.F16, QT.BF16}
_GGUF_TO_HF_EMB = {"token_embd.weight": "model.embed_tokens.weight", "output.weight": "lm_head.weight"}


# -------------------------------------------------------------------------------------------- reading the GGUF

class GGUF:
    def __init__(self, path):
        self.r = gguf.GGUFReader(str(path))
        self.t = {t.name: t for t in self.r.tensors}
        self.kv = {}
        for k, f in self.r.fields.items():
            if k.startswith("tokenizer."):
                continue
            try:
                self.kv[k] = f.contents()
            except Exception:
                pass
        self.meta = json.loads(self.kv["ttf.ckpt_meta"]) if "ttf.ckpt_meta" in self.kv else {}

    def has(self, n):
        return n in self.t

    def qtype(self, n):
        return self.t[n].tensor_type

    def float(self, n) -> torch.Tensor:
        """any tensor dequantized to float32, in the GGUF's row-major shape (ne reversed)"""
        t = self.t[n]
        a = np.asarray(t.data)
        if t.tensor_type == QT.F32:
            w = a.astype(np.float32)
        elif t.tensor_type == QT.F16:
            w = a.astype(np.float32)
        else:
            w = gguf.quants.dequantize(a, t.tensor_type)
        shape = [int(x) for x in reversed(t.shape)]
        return torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).reshape(shape)

    def trits(self, n):
        """a ternary weight -> (T int8 [N, K'], s fp32 [N]: the row scale held in the blocks, or None for TQ1_P)"""
        t = self.t[n]
        if TQ1_P is not None and t.tensor_type == TQ1_P:
            K = int(t.shape[0])
            return dec_tq1p(torch.from_numpy(np.asarray(t.data, dtype=np.uint8)).reshape(-1, K // 160 * 32), K), None
        assert t.tensor_type in TERN_TYPES | {QT.F16, QT.F32}, (n, t.tensor_type)
        w = self.float(n)
        s = w.abs().amax(1)
        T = torch.where(s[:, None] > 0, w / s.clamp_min(1e-30)[:, None], torch.zeros_like(w)).round()
        assert torch.equal(T.abs() <= 1, torch.ones_like(T, dtype=torch.bool)), n
        assert torch.allclose(T * s[:, None], w, rtol=1e-3, atol=0), f"{n}: not {{-1, 0, 1}} x a row scale"
        return T.to(torch.int8), s


def dec_tq1p(B: torch.Tensor, K: int) -> torch.Tensor:
    """TQ1_P bytes uint8 [R, K / 160 * 32] -> int8 trits [R, K] (q38_export.enc_tq1p's inverse: element 32 n + b of a
    160-block is digit n of byte b, most significant first; ggml's decode: ((q 3^n mod 256) * 3) >> 8)"""
    R = B.shape[0]
    q = B.to(torch.int32).reshape(R, K // 160, 1, 32)
    p3 = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32).reshape(1, 1, 5, 1)
    d = (((q * p3) & 0xFF) * 3) >> 8                       # [R, K / 160, 5, 32]
    return (d - 1).to(torch.int8).reshape(R, K)


# ------------------------------------------------------------------------------------- trainer-side encodings

def pack_rows(T: torch.Tensor) -> torch.Tensor:
    """int8 trits [N, K] -> the trainer's packed rows uint8 [N, ceil(K / 5)] (5 trits along K, digit j weighs 3^j)"""
    N, K = T.shape
    K5 = -(-K // 5)
    x = torch.zeros(N, K5 * 5, dtype=torch.int16)
    x[:, :K] = T.to(torch.int16) + 1
    x[:, K:] = 1                                           # padding trits are 0
    x = x.reshape(N, K5, 5)
    w = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16)
    return (x * w).sum(-1).to(torch.uint8)


def trit_beta(T: torch.Tensor) -> float:
    nnz = torch.tensor(float(T.count_nonzero()), dtype=torch.float32)
    return float((nnz / T.shape[0]).clamp_min(1.0).rsqrt())


def _row_scale(s, T, name, log):
    """r * beta -> r; an all-zero row stores no scale (its d is 0): it takes the median of the others"""
    z = s == 0
    if bool(z.any()):
        fill = s[~z].median() if bool((~z).any()) else torch.tensor(1.0)
        log(f"{name}: {int(z.sum())} all-zero rows (no stored scale): row scale {float(fill):.3g}")
        s = torch.where(z, fill, s)
    return s / trit_beta(T)


# --------------------------------------------------------------------------------- the converter's maps, probed

def _converter(base_dir: Path, untied: bool, tmp: Path):
    for f in base_dir.iterdir():
        if f.is_file() and not f.name.endswith((".safetensors", ".bin", ".pt")) and f.name != "model.safetensors.index.json":
            (tmp / f.name).symlink_to(f.resolve())
    cfg = json.loads((base_dir / "config.json").read_text())
    if untied:
        cfg["tie_word_embeddings"] = False
        if "text_config" in cfg:
            cfg["text_config"]["tie_word_embeddings"] = False
    (tmp / "config.json").unlink()
    (tmp / "config.json").write_text(json.dumps(cfg))
    base = get_model_class(cfg["architectures"][0])
    if hasattr(base, "no_mtp"):
        base.no_mtp = True

    class Probe(base):
        model_arch = base.model_arch

        def index_tensors(self, remote_hf_model_id=None):
            hp = {**self.hparams, **self.hparams.get("text_config", {})}
            type(self)._original_block_count = hp.get("num_hidden_layers")
            return {}
    return Probe(tmp, gguf.LlamaFileType.ALL_F32, tmp / "probe.gguf", eager=True), cfg


class Maps:
    """HF tensor name -> where the converter puts it, read off probes through its modify_tensors"""

    def __init__(self, conv):
        self.c = conv

    def _run(self, name, x):
        item = self.c.filter_tensors((name, lambda: x))
        if item is None:
            return None
        hn = item[0]
        bid = next((int(p) for p in hn.split(".") if p.isdecimal()), None)
        return [(n, torch.as_tensor(t).float()) for n, t in self.c.modify_tensors(x, hn, bid)]

    def rows_cols(self, name, N, K):
        """a matrix the converter only reorders: [(gguf name, row index, column index)] (as gguf_export._perm)"""
        Rm = torch.arange(N, dtype=torch.float32)[:, None].expand(N, K).contiguous()
        Cm = torch.arange(K, dtype=torch.float32)[None, :].expand(N, K).contiguous()
        ro, co = self._run(name, Rm), self._run(name, Cm)
        out = []
        for (nn_, R), (_, C) in zip(ro, co):
            ri, ci = R[:, 0].long(), C[0, :].long()
            assert torch.equal(R, ri[:, None].float().expand_as(R)) and torch.equal(C, ci[None].float().expand_as(C)), \
                f"{name}: the converter does more than reorder rows / columns"
            out.append((nn_, ri, ci))
        return out

    def general(self, name, shape):
        """any float tensor: [(gguf name, flat source index (gguf shape), inverse fn of the converter's value map)]"""
        n = int(np.prod(shape))
        ar = torch.arange(n)
        lo, hi = (ar % 1024).float().reshape(shape), (ar // 1024).float().reshape(shape)
        Z, O = torch.zeros(shape), torch.ones(shape)
        rz, ro_, rl, rh = (self._run(name, x) for x in (Z, O, lo, hi))
        if rz is None:
            return None
        out = []
        for (gn, z), (_, o), (_, l), (_, h) in zip(rz, ro_, rl, rh):
            a, b = o - z, z
            if torch.allclose(z, torch.full_like(z, -1.0)) and torch.allclose(o, torch.full_like(o, -np.e)):
                # -exp(x) (Qwen3.5 / 3.8: ssm_a = -exp(A_log)); the index probes go through it too
                assert n <= 64, f"{name}: -exp map on a large tensor"
                inv = lambda y: torch.log(-y)
                idx = (torch.log(-h).round() * 1024 + torch.log(-l).round()).long()
            else:
                assert bool((a.abs() > 1e-6).all()), f"{name} -> {gn}: not an affine map of the input"
                inv = lambda y, a=a, b=b: (y - b) / a
                idx = (((h - b) / a).round() * 1024 + ((l - b) / a).round()).long()
            assert int(idx.min()) >= 0 and int(idx.max()) < n, (name, gn)
            out.append((gn, idx, inv))
        return out


# ------------------------------------------------------------------------------------------------------ import

def import_gguf(path, base_dir, out=None, log=print, max_layers=None):
    """max_layers: only decoder layers below it (a test of a large model)"""
    from ttf.qwen import _KEEP_FLOAT, text_config
    import torch.nn as nn
    G = GGUF(path)
    fmt = int(G.kv.get("ttf.format", 0))
    assert fmt in (1, 2), f"{path}: no ttf.format (not a TTF GGUF)"
    base_dir = Path(base_dir).expanduser()
    tc = text_config(str(base_dir))
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    tc._attn_implementation = "sdpa"
    with torch.device("meta"):
        hf = Qwen3_5ForCausalLM(tc)
    untied = G.has("output.weight")
    tmp = Path(tempfile.mkdtemp(prefix="ttfimp_"))
    try:
        conv, cfg = _converter(base_dir, untied, tmp)
        M = Maps(conv)
        sd, ranks, mags, lin_names = {}, {}, {}, []
        for mname, mod in hf.model.layers.named_modules():
            for cname, ch in mod.named_children():
                if isinstance(ch, nn.Linear) and cname not in _KEEP_FLOAT:
                    if max_layers is not None and int(mname.split(".")[0]) >= max_layers:
                        continue
                    lin_names.append((f"model.layers.{mname}.{cname}", ch.out_features, ch.in_features))
        tern_params = {f"{n}.weight" for n, _, _ in lin_names}
        for hb, N, K in lin_names:
            parts = _import_linear(G, M, hb, N, K, log)
            sd.update(parts["sd"])
            if parts["rank"]:
                ranks[hb] = parts["rank"]
            if parts["mag"]:
                mags[hb] = parts["mag"]
        for name, p in hf.state_dict().items():
            if name in tern_params:
                continue
            if max_layers is not None and name.startswith("model.layers.") and int(name.split(".")[2]) >= max_layers:
                continue
            if max_layers is not None and name in _GGUF_TO_HF_EMB.values():
                continue                                  # (a test: the large embedding / head left out)
            sd[name] = _import_float(G, M, name, tuple(p.shape), log)
        if not untied:
            sd.pop("lm_head.weight", None)
        _centred_floats(G, M, sd, log)
        # QwenTern registers hf.model.layers also as layers.
        full = {}
        for k, v in sd.items():
            full["hf." + k] = v
            if k.startswith("model.layers."):
                full[k[len("model."):]] = v
        ext = []
        if untied:
            ext_spec = "untie=1"
        else:
            ext_spec = "untie=0"
        if ranks:
            rk = _rank_spec(ranks, {hb: (N, K) for hb, N, K in lin_names})
            ext_spec += f",rank={rk}"
        ext.append(f"ttfhome.train_ext:{ext_spec}")
        args = {"init_hf": str(base_dir), "ext": ext, "mode": "ttf", "arch_hf": str(base_dir),
                "adapter_rank": (max(mags.values()) if mags else 0), "imported_from": str(path),
                "ttf_format": fmt, "ckpt_meta": G.meta}
        ck = {"model": full, "step": 0, "beta": True, "args": args, "imported": True}
        log(f"imported {len(lin_names)} TTF linears ({len(ranks)} with ternary adapters, {len(mags)} with float "
            f"adapters), {len(sd) - sum(1 for k in sd if k.rsplit('.', 1)[0] in tern_params)} tensors")
        if out:
            torch.save(ck, out)
        return ck
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _centred_floats(G, M, sd, log):
    """a GGUF with blk.i.ssm_beta.bias (an export that folded the centred float in_proj_b, y = W (x - mu) + b, into
    W x + (b - W mu)): in_proj_b back on the centred input, mu = its layer's in_proj_qkv mean, out_bias = that bias +
    W mu. (A folded in_proj_a bias sits in ssm_dt.bias, already dt_bias here: the trainer's --center-floats then starts
    in_proj_a at bias W mu, the same function.)"""
    n = 0
    for k in list(sd):
        m = re.fullmatch(r"model\.layers\.(\d+)\.linear_attn\.in_proj_b\.weight", k)
        if not m or not G.has(f"blk.{m.group(1)}.ssm_beta.bias"):
            continue
        pre = k[:-len("in_proj_b.weight")]
        mu = sd.get(pre + "in_proj_qkv.lin.in_mean", sd.get(pre + "in_proj_qkv.in_mean"))
        assert mu is not None, f"{k}: a folded centring bias but no centred in_proj_qkv"
        W = sd[k].float()
        (gn, ri, _), = M.rows_cols(k, *W.shape)
        b = torch.zeros(W.shape[0])
        b[ri] = G.float(f"blk.{m.group(1)}.ssm_beta.bias")
        sd[pre + "in_proj_b.in_mean"] = mu.clone()
        sd[pre + "in_proj_b.out_bias"] = b + W @ mu
        n += 1
    if n:
        log(f"{n} centred in_proj_b restored from ssm_beta.bias")


def _rank_spec(ranks, shapes):
    """the adapters' ranks as ttfconv.adapters' rank fraction when one reproduces all of them, else a JSON list"""
    def rank_of(K, N, r):
        return max(8, int(round(r * min(K, N) / 8)) * 8) if r < 1 else int(r)
    for r in [x / 100 for x in range(1, 100)]:
        if all(rank_of(shapes[h][1], shapes[h][0], r) == v for h, v in ranks.items()):
            return f"{r:g}"
    vals = set(ranks.values())
    if len(vals) == 1:
        return str(vals.pop())
    raise SystemExit(f"adapter ranks {sorted(vals)}: no single rank rule reproduces them")


def _inv_perm(ri, n):
    inv = torch.full((n,), -1, dtype=torch.long)
    inv[ri] = torch.arange(len(ri))
    return inv


def _import_linear(G, M, hb, N, K, log):
    """one HF linear -> the trainer's tensors for it (plain, or .lin / .ad.0 / .ad.1 with a ternary adapter)"""
    maps = M.rows_cols(hb + ".weight", N, K)
    T = torch.zeros(N, K, dtype=torch.int8); s = torch.zeros(N)
    have_c = have_m = have_b = False
    c, m, b = torch.ones(K), torch.zeros(K), torch.zeros(N)
    ad = None
    seen_rows = torch.zeros(N, dtype=torch.bool)
    for gname, ri, ci in maps:
        stem = gname[:-len(".weight")]
        Tg, sg = G.trits(gname)
        Tg = Tg[:, :K]                                    # (TQ1_P: K padded to 160 with zero trits)
        if sg is None:
            sg = G.float(stem + ".ttf_r")
        T[ri[:, None], ci[None, :]] = Tg
        s[ri] = sg
        seen_rows[ri] = True
        if G.has(stem + ".ttf_c"):
            c[ci] = G.float(stem + ".ttf_c")[:K]; have_c = True
        if G.has(stem + ".ttf_m"):
            m[ci] = G.float(stem + ".ttf_m")[:K]; have_m = True
        if G.has(stem + ".ttf_b"):
            b[ri] = G.float(stem + ".ttf_b"); have_b = True
        if G.has(stem + ".ttf_u") or G.has(stem + ".ttf_fu"):
            ad = (ad or []) + [(stem, ri, ci)]
    assert bool(seen_rows.all()), f"{hb}: rows missing from the GGUF"
    mu = -m / c if have_m else None
    sdict, rank, mag = {}, 0, 0

    def put(pfx, T_, s_, c_, mu_, b_, nm):
        sdict[f"{pfx}.wpacked"] = pack_rows(T_)
        sdict[f"{pfx}.row_scale"] = _row_scale(s_, T_, nm, log)
        sdict[f"{pfx}.col_scale"] = c_
        sdict[f"{pfx}.frozen_scale"] = torch.tensor(0.0)
        if mu_ is not None or b_ is not None:
            sdict[f"{pfx}.in_mean"] = mu_ if mu_ is not None else torch.zeros(T_.shape[1])
            sdict[f"{pfx}.out_bias"] = b_ if b_ is not None else torch.zeros(T_.shape[0])

    tern_ad = ad and any(G.qtype(st + ".ttf_d") not in FLOAT_TYPES for st, _, _ in ad if G.has(st + ".ttf_d"))
    pfx = f"{hb}.lin" if tern_ad else hb
    put(pfx, T, s, c, mu, b if (have_b or have_m) else None, hb)
    r_main = sdict[f"{pfx}.row_scale"]
    if ad:
        for st, ri, ci in ad:
            f = ".ttf_fd" if G.has(st + ".ttf_fd") else (".ttf_d" if G.qtype(st + ".ttf_d") in FLOAT_TYPES else None)
            if f is not None:                             # float adapter: U = r A [N, k], fd = B^T [k, K]
                Bt = G.float(st + f)[:, :K]
                U = G.float(st + f[:-1] + "u")
                k = Bt.shape[0]
                A = torch.zeros(N, k); A[ri] = U
                Bm = torch.zeros(K, k); Bm[ci] = Bt.t()
                sdict[f"{pfx}.mag_A"] = A / r_main[:, None]
                sdict[f"{pfx}.mag_B"] = Bm
                mag = k
        if tern_ad:
            rank = _import_adapter(G, hb, N, K, ad, sdict, put)
    return {"sd": sdict, "rank": rank, "mag": mag}


def _import_adapter(G, hb, N, K, ad, sdict, put):
    """the ternary adapter: .ttf_d [rp, K] (rank zero-padded), .ttf_u [N, rp]; the rank is ckpt_meta's, else the
    last used index rounded up to 8"""
    assert len(ad) == 1, f"{hb}: a ternary adapter on a split linear"
    st, ri, ci = ad[0]
    Td, sd_ = G.trits(st + ".ttf_d")
    Tu, su = G.trits(st + ".ttf_u")
    rp = Td.shape[0]
    if sd_ is None:
        sd_ = G.float(st + ".ttf_d_r")
    if su is None:
        su = G.float(st + ".ttf_u_r")
    bd = G.float(st + ".ttf_d_b") if G.has(st + ".ttf_d_b") else None
    rk = (G.meta.get("adapter_rank") or {}).get(hb)
    if not rk:
        used = (Td != 0).any(1) | (Tu[:, :rp] != 0).any(0)
        if bd is not None:
            used |= bd[:rp] != 0
        last = int(used.nonzero().max()) + 1 if bool(used.any()) else 8
        rk = -(-last // 8) * 8
    Dt = torch.zeros(rk, K, dtype=torch.int8); Dt[:, ci] = Td[:rk, :K]
    cd = torch.ones(K); md = torch.zeros(K); hm = False
    if G.has(st + ".ttf_d_c"):
        cd[ci] = G.float(st + ".ttf_d_c")[:K]
    if G.has(st + ".ttf_d_m"):
        md[ci] = G.float(st + ".ttf_d_m")[:K]; hm = True
    put(f"{hb}.ad.0", Dt, sd_[:rk], cd, (-md / cd) if hm else None, bd[:rk] if bd is not None else None, hb + ".ad.0")
    Ut = torch.zeros(N, rk, dtype=torch.int8); Ut[ri] = Tu[:, :rk]
    sU = torch.zeros(N); sU[ri] = su
    cu = G.float(st + ".ttf_u_c")[:rk] if G.has(st + ".ttf_u_c") else torch.ones(rk)
    mu_u = (-G.float(st + ".ttf_u_m")[:rk] / cu) if G.has(st + ".ttf_u_m") else None
    bu = None
    if G.has(st + ".ttf_u_b"):
        bu = torch.zeros(N); bu[ri] = G.float(st + ".ttf_u_b")
    put(f"{hb}.ad.1", Ut, sU, cu, mu_u, bu, hb + ".ad.1")
    return rk


def _import_float(G, M, name, shape, log):
    if name in _GGUF_TO_HF_EMB.values():                  # the converter passes these through as they are
        gn = {v: k for k, v in _GGUF_TO_HF_EMB.items()}[name]
        if not G.has(gn):
            return None
        w = G.float(gn)
        assert tuple(w.shape) == shape, (name, w.shape, shape)
        log(f"{name}: from {gn} ({G.qtype(gn).name})")
        return w.to(torch.bfloat16)
    maps = M.general(name, shape)
    assert maps, f"{name}: the converter drops it"
    n = int(np.prod(shape))
    flat = torch.full((n,), float("nan"))
    for gn, idx, inv in maps:
        assert G.has(gn), f"{name} -> {gn}: not in the GGUF"
        y = G.float(gn).reshape(idx.shape)
        flat[idx.reshape(-1)] = inv(y).reshape(-1)
    assert not bool(flat.isnan().any()), f"{name}: not all elements found"
    return flat.reshape(shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf"); ap.add_argument("base_dir"); ap.add_argument("out")
    a = ap.parse_args()
    with torch.inference_mode():
        import_gguf(a.gguf, a.base_dir, a.out)


if __name__ == "__main__":
    main()
