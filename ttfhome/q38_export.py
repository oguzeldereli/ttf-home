"""Q38 (Qwen3.8-Flash-Next) TTF conversion -> GGUF for llama.cpp's qwen4exp (branch ttf-q38: TQ1_P + TTF format 2).

Inputs: a slim TTF checkpoint (a directory of L{i}.pt = {"layer": layer i's trained tensors}, from the trainer's
slim save), the HF slim directory (config, tokenizer and the non-expert float weights; the experts and the PLE table
are not needed), and the PQ PLE table (ple_pq.py's meta.pt + the compacted blob from pq_blob.py).

Every TTF linear  y = r * (beta T c(x - mu) + sa TA (cb * (TB c(x - mu)))) + bias  (moe_ttf.TernaryExperts) is written
  <w>.weight   T, TQ1_P (160 trits in 32 bytes, no scale) when K % 160 == 0, else TQ1_0 with block scale 1
  <w>.ttf_c    c,  <w>.ttf_m = -c mu,  <w>.ttf_r = r beta,  <w>.ttf_b = bias
  <w>.ttf_ad   TB [16, K] (the adapter's down trits, on the centred input; same type rule as T)
  <w>.ttf_au   F16 [N, 16]: r_n sa TA[n, k] cb_k
Routed experts (mlp.gu as ffn_gate_up_exps: gate rows then up, mlp.dn as ffn_down_exps) take the same tensors with a
trailing expert dimension. Everything else goes through llama.cpp's own converter (renames, V-head reorder, +1
norms); the dense TTF layers take its row / column order through index matrices (as gguf_export.py).
PLE: per_layer_token_embd.pq_cb (F32 [4, heads * 40 * 256 + 1], the last entry zero for pruned rows) and .pq_blob
(I8: keep bitmap, ranks, codes); the PQ rotation Pi_h is folded into ple.key_proj / value_proj (stored F32).

  python -m ttfhome.q38_export SLIM_CKPT_DIR HF_SLIM_DIR PQ_DIR OUT.gguf [--emb q8_0] [--head q8_0] [--layers N]
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

LLAMA = Path(os.environ.get("TTF_LLAMA", str(Path.home() / "Programs/llama.cpp")))
sys.path[:0] = [str(LLAMA / "gguf-py"), str(LLAMA)]
import gguf                                    # noqa: E402
from conversion import get_model_class         # noqa: E402

QT = gguf.GGMLQuantizationType
PFX = "model.language_model."
PART = ("wpacked", "row_scale", "col_scale", "out_bias", "ad_cb", "ad_sa", "beta", "in_mean", "ta_packed", "tb_packed")
EMB_TYPES = {"f32": QT.F32, "f16": QT.F16, "bf16": QT.BF16, "q8_0": QT.Q8_0, "q6_k": QT.Q6_K, "q4_k": QT.Q4_K}

_LUT = None


def unpack(P: torch.Tensor, K: int) -> torch.Tensor:
    """core pack_rows bytes uint8 [..., ceil(K / 5)] (5 trits along K, digit j weighs 3^j) -> int8 {-1, 0, 1} [..., K]"""
    global _LUT
    if _LUT is None:
        v = torch.arange(256)
        _LUT = torch.stack([(v // 3 ** j) % 3 for j in range(5)], 1).to(torch.int8) - 1      # [256, 5]
    T = _LUT[P.long()]                                                                       # [..., K5, 5]
    return T.reshape(*P.shape[:-1], -1)[..., :K]


def enc_tq1p(T: torch.Tensor) -> np.ndarray:
    """int8 trits [..., K] (K % 160 == 0) -> TQ1_P bytes uint8 [..., K / 160 * 32] (element 32 n + b = digit n of
    byte b, n = 0 most significant; byte = ceil(q 256 / 243))"""
    *lead, K = T.shape
    x = (T.reshape(*lead, K // 160, 5, 32).to(torch.int16) + 1)
    q = x[..., 0, :] * 81 + x[..., 1, :] * 27 + x[..., 2, :] * 9 + x[..., 3, :] * 3 + x[..., 4, :]
    q = (q.to(torch.int32) * 256 + 242) // 243
    return q.to(torch.uint8).reshape(*lead, K // 160 * 32).numpy()


def enc_tq1_0(T: torch.Tensor) -> np.ndarray:
    """int8 trits [R, K] (K % 256 == 0) -> TQ1_0 blocks with d = 1 (the row scale sits in ttf_r)"""
    R, K = T.shape
    w = T.float().numpy()
    return gguf.quants.quantize(w, QT.TQ1_0).reshape(R, -1)


def enc_trits(T: torch.Tensor):
    """-> (bytes, type): TQ1_P when K % 160 == 0, else TQ1_0 (d = 1)"""
    K = T.shape[-1]
    if K % 160 == 0:
        return enc_tq1p(T), QT.TQ1_P
    assert K % 256 == 0 and T.dim() == 2, T.shape
    return enc_tq1_0(T), QT.TQ1_0


class Deferred:
    """stands in for an ndarray in GGUFWriter: shape / dtype / nbytes now, the data from fn() at tofile"""

    def __init__(self, shape, dtype, fn):
        self.shape, self.dtype, self.fn = tuple(int(x) for x in shape), np.dtype(dtype), fn
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize

    def tofile(self, f):
        a = np.ascontiguousarray(self.fn())
        assert a.shape == self.shape and a.dtype == self.dtype, (a.shape, a.dtype, self.shape, self.dtype)
        a.tofile(f)


class Slim:
    """the slim checkpoint: per layer its TTF parts (by sub-module) and its trained floats"""

    def __init__(self, d):
        self.d = Path(d)
        self.n = len(list(self.d.glob("L*.pt")))
        self._c = {}

    def layer(self, i):
        if i not in self._c:
            self._c.clear()
            blob = torch.load(self.d / f"L{i}.pt", map_location="cpu", mmap=True, weights_only=False)
            L = blob["layer"] if "layer" in blob else blob
            ttf, fl = {}, {}
            subs = {k[:-len(".wpacked")] for k in L if k.endswith(".wpacked")}
            for k, v in L.items():
                s, _, p = k.rpartition(".")
                if s in subs and p in PART:
                    ttf.setdefault(s, {})[p] = v
                else:
                    fl[k] = v
            self._c[i] = (ttf, fl)
        return self._c[i]


class TTFLin:
    """one TTF linear (E experts) -> its GGUF-side tensors, before any reorder"""

    def __init__(self, p):
        self.E, self.N, K5 = p["wpacked"].shape
        self.K = p["col_scale"].shape[1]
        assert (self.K + 4) // 5 == K5
        self.p = p
        self.R = p["tb_packed"].shape[1] if "tb_packed" in p else 0

    def T(self, e0=0, e1=None):
        return unpack(self.p["wpacked"][e0:e1], self.K)

    def vecs(self):
        p = self.p
        c = p["col_scale"].float()
        r = p["row_scale"].float() * p["beta"].float()[:, None]
        m = -c * p["in_mean"].float() if "in_mean" in p else None
        b = p["out_bias"].float() if "out_bias" in p else None
        return c, m, r, b

    def adapter(self, e0=0, e1=None):
        """-> (TB int8 [E, R, K], U float [E, N, R] = r sa TA cb)"""
        p = self.p
        TB = unpack(p["tb_packed"][e0:e1], self.K)
        TA = unpack(p["ta_packed"][e0:e1], self.R).float()
        U = TA * p["ad_cb"][e0:e1].float()[:, None, :] * (p["row_scale"][e0:e1].float() * p["ad_sa"][e0:e1].float()[:, None])[..., None]
        return TB, U


def _hf_floats(hf_dir):
    """name -> loader of the HF slim's float tensors (FP8 dequantised), without the PLE shards / scales"""
    from safetensors import safe_open
    idx = json.loads((Path(hf_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    have = {n: f for n, f in idx.items() if (Path(hf_dir) / f).exists()}
    handles = {}

    def get(n):
        f = have[n]
        if f not in handles:
            handles[f] = safe_open(str(Path(hf_dir) / f), "pt")
        return handles[f].get_tensor(n)

    def loader(n):
        def ld():
            t = get(n)
            si = n[:-len(".weight")] + ".weight_scale_inv" if n.endswith(".weight") else None
            if si and si in have:
                s = get(si).float()
                bs = 128
                t = t.float()
                R, C = t.shape
                t = t * s.repeat_interleave(bs, 0)[:R].repeat_interleave(bs, 1)[:, :C]
            return t
        return ld

    out = {}
    for n in have:
        if n.endswith((".weight_scale_inv", ".weight_scale")) or ".ngram_embedding." in n:
            continue
        out[n] = loader(n)
    return out


def export(slim_dir, hf_dir, pq_dir, out, emb="q8_0", head="q8_0", n_layers=None, log=print):
    slim = Slim(slim_dir)
    cfg_full = json.loads((Path(hf_dir) / "config.json").read_text())
    tc = cfg_full["text_config"]
    NL = n_layers or tc["num_hidden_layers"]
    assert slim.n >= NL, f"{slim.n} layer files, {NL} layers"
    pq_meta = torch.load(Path(pq_dir) / "meta.pt", weights_only=False)
    ple_layers = [i - 1 for i in tc.get("ple_layer_ids", [])]

    floats = _hf_floats(hf_dir)
    ttf_subs = {}                         # layer -> TTF sub-module names
    for i in range(NL):
        ttf, fl = slim.layer(i)
        ttf_subs[i] = sorted(ttf)
        for s in ttf:                     # the dense TTF layers replace the HF weights
            floats.pop(f"{PFX}layers.{i}.{s}.weight", None)
        for k, v in fl.items():           # trained floats override the HF ones
            floats[f"{PFX}layers.{i}.{k}"] = (lambda v=v: v)
    if n_layers:                          # (a truncated test model: drop the other layers' floats)
        floats = {k: v for k, v in floats.items()
                  if not (m := re.match(re.escape(PFX) + r"layers\.(\d+)\.", k)) or int(m.group(1)) < NL}

    # PQ rotation folded into the PLE layer's key / value projections: W'[:, h] = W[:, h] Pi_h
    Pi = pq_meta["Pi"].double()           # [heads, 160, 160]
    for i in ple_layers:
        if i >= NL:
            continue
        for pn in ("key_proj", "value_proj"):
            n = f"{PFX}layers.{i}.ple.{pn}.weight"
            ld = floats[n]

            def fold(ld=ld):
                W = ld().double()
                H, d = Pi.shape[0], Pi.shape[1]
                Wh = W.view(W.shape[0], H, d)
                return torch.einsum("ohk,hkn->ohn", Wh, Pi).reshape(W.shape[0], H * d).float()
            floats[n] = fold

    tmp = Path(tempfile.mkdtemp(prefix="q38gguf_"))
    try:
        for f in Path(hf_dir).iterdir():
            if f.is_file() and not f.name.endswith((".safetensors", ".bin", ".pt")) and f.name not in ("config.json", "model.safetensors.index.json"):
                (tmp / f.name).symlink_to(f.resolve())
        cfg = dict(cfg_full)
        cfg.pop("quantization_config", None)      # the floats are dequantised here
        if n_layers:
            cfg["text_config"] = dict(tc, num_hidden_layers=NL, layer_types=tc["layer_types"][:NL])
        (tmp / "config.json").write_text(json.dumps(cfg, indent=2))
        base = get_model_class(cfg["architectures"][0])
        base.no_mtp = True
        model = _make_class(base, slim, floats, ttf_subs, NL, pq_dir, pq_meta, EMB_TYPES[emb], EMB_TYPES[head], log)(
            tmp, gguf.LlamaFileType.MOSTLY_BF16, Path(out), eager=True)
        model.write()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _make_class(base, slim, floats, ttf_subs, NL, pq_dir, pq_meta, eq, hq, log):
    class TTFQ38(base):
        model_arch = base.model_arch
        # the TTF linears stay separate (each has its own side tensors); the float merges stay
        _SHARED_INPUT_MERGES = tuple(m for m in base._SHARED_INPUT_MERGES if m[0] != gguf.MODEL_TENSOR.ATTN_QKV_GATE)

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._ple_row_dim = int(pq_meta["Pi"].shape[1])     # (no shards: the row dim the table would have)

        def index_tensors(self, remote_hf_model_id=None):
            hp = {**self.hparams, **self.hparams.get("text_config", {})}
            type(self)._original_block_count = hp.get("num_hidden_layers")
            out = {}
            for name, ld in floats.items():
                if (item := self.filter_tensors((name, ld))) is not None:
                    out[item[0]] = item[1]
            return out

        def set_gguf_parameters(self):
            super().set_gguf_parameters()
            self.gguf_writer.add_uint32("ttf.format", 2)
            self.gguf_writer.add_string("ttf.act_quant", "trained with int8 per token")

        def tensor_force_quant(self, name, new_name, bid, n_dims):
            if new_name in ("token_embd.weight", "output.weight"):
                return QT.F32                       # quantised in add_tensor below
            if ".ple_key." in new_name or ".ple_value." in new_name or new_name.startswith(("blk.1.ple_k", "blk.1.ple_v")):
                return QT.F32
            if "conv1d" in new_name:
                return QT.F32
            if n_dims >= 2 and new_name.endswith(".weight"):
                return QT.BF16
            return False

        def prepare_tensors(self):
            add = self.gguf_writer.add_tensor

            def add_tensor(name, tensor, raw_shape=None, raw_dtype=None, **kw):
                qt = {"token_embd.weight": eq, "output.weight": hq}.get(name)
                if qt is not None and qt != QT.F32 and tensor.dtype == np.float32:
                    tensor, raw_dtype = gguf.quants.quantize(tensor, qt) if qt in (QT.F16, QT.BF16, QT.Q8_0) else _ggml_q(tensor, qt), qt
                    log(f"{name}: {qt.name}")
                return add(name, tensor, raw_shape=raw_shape, raw_dtype=raw_dtype, **kw)
            self.gguf_writer.add_tensor = add_tensor
            self._write_ttf()
            self._write_pq()
            super().prepare_tensors()

        def _perm(self, name, bid, N, K):
            Rm = torch.arange(N, dtype=torch.float32)[:, None].expand(N, K).contiguous()
            Cm = torch.arange(K, dtype=torch.float32)[None, :].expand(N, K).contiguous()
            ro = list(self.modify_tensors(Rm, name, bid))
            co = list(self.modify_tensors(Cm, name, bid))
            assert [n for n, _ in ro] == [n for n, _ in co] and ro, (name, [n for n, _ in ro])
            res = []
            for (nn, R), (_, C) in zip(ro, co):
                R, C = torch.as_tensor(R), torch.as_tensor(C)
                ri, ci = R[:, 0].long(), C[0, :].long()
                assert torch.equal(R, ri[:, None].float().expand_as(R)) and torch.equal(C, ci[None].float().expand_as(C)), \
                    f"{name}: the converter does more than reorder rows / columns"
                res.append((nn, ri, ci))
            return res

        def _vec(self, name, v):
            self.gguf_writer.add_tensor(name, np.ascontiguousarray(v.float().numpy()), raw_dtype=QT.F32)

        def _raw(self, name, data, qt):
            self.gguf_writer.add_tensor(name, np.ascontiguousarray(data), raw_dtype=qt)

        def _def(self, name, shape, dtype, qt, fn):
            """a tensor computed when the writer reaches it (bounded memory: one layer's arrays at a time)"""
            self.gguf_writer.add_tensor(name, Deferred(shape, dtype, fn), raw_dtype=qt)

        def _write_ttf(self):
            ntr = 0
            for i in range(NL):
                ttf, _ = slim.layer(i)
                for s in ttf_subs[i]:
                    lin = TTFLin(ttf[s])
                    E, N, K, R = lin.E, lin.N, lin.K, lin.R
                    has_m, has_b = "in_mean" in lin.p, "out_bias" in lin.p

                    def L(i=i, s=s):
                        return TTFLin(slim.layer(i)[0][s])
                    if s in ("mlp.gu", "mlp.dn"):
                        assert K % 160 == 0, (s, K)
                        stem = f"blk.{i}." + ("ffn_gate_up_exps" if s == "mlp.gu" else "ffn_down_exps")

                        def w_exp(L=L, stem=stem):
                            print(f"{time.strftime('%H:%M:%S')} {stem}", flush=True)
                            l = L()
                            return np.concatenate([enc_tq1p(l.T(e, e + 64)) for e in range(0, l.E, 64)])

                        def ad_exp(L=L):
                            l = L()
                            return np.concatenate([enc_tq1p(l.adapter(e, e + 64)[0]) for e in range(0, l.E, 64)])

                        def au_exp(L=L):
                            l = L()
                            return np.concatenate([l.adapter(e, e + 64)[1].half().numpy() for e in range(0, l.E, 64)])
                        self._def(stem + ".weight", (E, N, K // 160 * 32), np.uint8, QT.TQ1_P, w_exp)
                        self._def(stem + ".ttf_ad", (E, R, K // 160 * 32), np.uint8, QT.TQ1_P, ad_exp)
                        self._def(stem + ".ttf_au", (E, N, R), np.float16, QT.F16, au_exp)
                        for nm, k, n_ in (("c", 0, K), ("m", 1, K), ("r", 2, N), ("b", 3, N)):
                            if (nm == "m" and not has_m) or (nm == "b" and not has_b):
                                continue
                            self._def(f"{stem}.ttf_{nm}", (E, n_), np.float32, QT.F32,
                                      lambda L=L, k=k: L().vecs()[k].numpy().astype(np.float32))
                        ntr += E * N * K
                        continue
                    assert E == 1, s
                    hname = f"{PFX}layers.{i}.{s}.weight"
                    item = self.filter_tensors((hname, lambda: None))
                    assert item is not None, hname
                    for gname, ri, ci in self._perm(item[0], i, N, K):
                        stem = gname[:-len(".weight")]
                        nr, nc = len(ri), len(ci)
                        bs = (lambda k: k // 160 * 32) if nc % 160 == 0 else (lambda k: k // 256 * 54)
                        qt = QT.TQ1_P if nc % 160 == 0 else QT.TQ1_0
                        self._def(gname, (nr, bs(nc)), np.uint8, qt,
                                  lambda L=L, ri=ri, ci=ci: enc_trits(L().T()[0][ri][:, ci])[0])
                        self._def(stem + ".ttf_ad", (R, bs(nc)), np.uint8, qt,
                                  lambda L=L, ci=ci: enc_trits(L().adapter()[0][0][:, ci].contiguous())[0])
                        self._def(stem + ".ttf_au", (nr, R), np.float16, QT.F16,
                                  lambda L=L, ri=ri: L().adapter()[1][0][ri].half().numpy())
                        for nm, k, idx in (("c", 0, ci), ("m", 1, ci), ("r", 2, ri), ("b", 3, ri)):
                            if (nm == "m" and not has_m) or (nm == "b" and not has_b):
                                continue
                            self._def(f"{stem}.ttf_{nm}", (len(idx),), np.float32, QT.F32,
                                      lambda L=L, k=k, idx=idx: L().vecs()[k][0][idx].numpy().astype(np.float32))
                        ntr += nr * nc
            log(f"TTF: {ntr / 1e9:.2f} G trits (written at the tensor pass)")

        def _write_pq(self):
            cb = pq_meta["cb"].float()                         # [heads, nS, K, D]
            H, nS, K, D = cb.shape
            tab = torch.cat([cb.reshape(-1, D), torch.zeros(1, D)]).numpy()
            self._raw("per_layer_token_embd.pq_cb", tab, QT.F32)
            blob = np.memmap(Path(pq_dir) / "blob.bin", dtype=np.int8, mode="r")
            total = int(pq_meta["total"]); n_w = (total + 63) // 64
            assert (blob.size - 12 * n_w) % nS == 0, (blob.size, n_w, nS)
            self._raw("per_layer_token_embd.pq_blob", blob, QT.I8)
            log(f"PLE PQ: D {D} K {K}, {(blob.size - 12 * n_w) // nS} kept rows of {total}, {blob.size / 1e9:.2f} GB")

    return TTFQ38


def _ggml_q(w, qt):
    """K-quants through libggml-base (as gguf_export.ggml_quantize)"""
    import ctypes
    lib = None
    for d in ("build-cuda/bin", "build-cpu/bin", "build/bin"):
        p = LLAMA / d / "libggml-base.so"
        if p.exists():
            lib = ctypes.CDLL(str(p)); break
    assert lib is not None
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    R, C = w.shape
    out = np.empty((R, lib.ggml_row_size(int(qt), C)), dtype=np.uint8)
    for r0 in range(0, R, 8192):
        src = np.ascontiguousarray(w[r0:r0 + 8192], dtype=np.float32)
        lib.ggml_quantize_chunk(int(qt), src.ctypes.data, out[r0:r0 + len(src)].ctypes.data, 0, len(src), C, None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slim"); ap.add_argument("hf_dir"); ap.add_argument("pq_dir"); ap.add_argument("out")
    ap.add_argument("--emb", default="q8_0", choices=sorted(EMB_TYPES))
    ap.add_argument("--head", default="q8_0", choices=sorted(EMB_TYPES))
    ap.add_argument("--layers", type=int, default=None, help="export only the first N layers (a test model)")
    a = ap.parse_args()
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    with torch.inference_mode():
        export(a.slim, a.hf_dir, a.pq_dir, a.out, a.emb, a.head, a.layers)


if __name__ == "__main__":
    main()
