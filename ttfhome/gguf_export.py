"""TTF checkpoint -> GGUF for llama.cpp (the `ttf` branch: llama.cpp/0001-ttf-side-tensors.patch).

Every TTF linear  y = r * beta * (T Q(c * (x - mu))) + b  [+ U(D(x))]  is written as
  <name>.weight      the trits T with the row scale r * beta in the block scale (exact: TQ1_0, TQ2_0 and Q2_K
                     hold {-1, 0, 1} * d; d is fp16, so r * beta is rounded to fp16, finer than the bf16 the
                     trainer's fused path multiplies by)
  <name>.ttf_c       c (F32 [K]),  <name>.ttf_m = -c * mu (F32 [K]),  <name>.ttf_b = b (F32 [N])
  <name>.ttf_d(_c/_m/_b), <name>.ttf_u(_c/_m/_b)   the ternary adapter's down and up layers, the same form; the
                     adapter rank is padded with zero trits to a multiple of 256 (the block size along K)
  <name>.ttf_fd(_c), <name>.ttf_fu   the float low-rank adapter (F16), when the layer has a ternary adapter too
                     (alone it takes the .ttf_d / .ttf_u names)
Everything else (norms, conv, the float in_proj_a / b, the embedding, the head) goes through llama.cpp's own HF
converter, so its renames, head reorders and +1 norm offsets apply unchanged; the ternary layers take the same row /
column order by passing index matrices through the converter's modify_tensors.

The one difference from the trainer: the trainer quantises each layer input per token to int8 (absmax); llama.cpp
quantises it per block of 256 (CPU, Q8_K) or 32 (CUDA, Q8_1), finer, so the output is a little closer to the float
layer than in training.

  python -m ttfhome.gguf_export CKPT HF_DIR OUT.gguf [--wtype tq2_0|tq1_0|q2_k] [--emb q6_k] [--head q6_k]
HF_DIR: the base model's config.json and tokenizer files (no weights needed).
"""
import argparse
import ctypes
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .cli import engine_dir                    # noqa: E402  (the same llama.cpp tree `ttf engine build` makes)
LLAMA = engine_dir()
sys.path[:0] = [str(LLAMA / "gguf-py"), str(LLAMA)]
import gguf                                    # noqa: E402
from conversion import get_model_class         # noqa: E402

QT = gguf.GGMLQuantizationType
WTYPES = {"tq2_0": QT.TQ2_0, "tq1_0": QT.TQ1_0, "q2_k": QT.Q2_K}
FTYPES = {"tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0, "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
          "q2_k": gguf.LlamaFileType.MOSTLY_Q2_K}
FTYPES_EMB = {"f32": QT.F32, "f16": QT.F16, "bf16": QT.BF16, "q8_0": QT.Q8_0, "q6_k": QT.Q6_K, "q5_k": QT.Q5_K,
              "q4_k": QT.Q4_K, "q4_1": QT.Q4_1}
# int-k embedding / head: Q4_1 from the codes (exact), else the float W through the type asked for
INTK_GGUF = {"token_embd.weight": "model.embed_tokens.weight", "output.weight": "lm_head.weight"}
QK = 256


# ---------------------------------------------------------------------------------------------------------- trits

def unpack_trits(packed: torch.Tensor, K: int) -> torch.Tensor:
    """uint8 [N, ceil(K / 5)], 5 trits per byte base 3 (digit j weighs 3^j) -> int8 {-1, 0, 1} [N, K]"""
    x = packed.to(torch.int16)
    N, K5 = packed.shape
    out = torch.empty(N, K5, 5, dtype=torch.int8)
    for j in range(5):
        out[:, :, j] = (x % 3).to(torch.int8)
        x = x // 3
    return out.view(N, K5 * 5)[:, :K].sub_(1)


def trit_beta(T: torch.Tensor) -> float:
    """the trainer's per-tensor scale 1 / sqrt(max(nnz / N, 1)) (ttf.kernel.beta_from_nnz), in fp32"""
    nnz = torch.tensor(float(T.count_nonzero()), dtype=torch.float32)
    return float((nnz / T.shape[0]).clamp_min(1.0).rsqrt())


def _tq2_qs(q: np.ndarray) -> np.ndarray:
    """codes {0, 1, 2} [n_blocks, 256] -> the 64-byte TQ2_0 / Q2_K layout: in each 128 chunk, element 32 j + l is
    byte l's bits 2 j"""
    nb = q.shape[0]
    q = q.reshape(nb, -1, 4, 32) << np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    return (q[..., 0, :] | q[..., 1, :] | q[..., 2, :] | q[..., 3, :]).reshape(nb, -1)


def pack_ternary(T: torch.Tensor, s: torch.Tensor, qtype) -> np.ndarray:
    """trits T int8 [N, K] (K a multiple of 256) with row scale s [N] -> raw blocks uint8 [N, row bytes]"""
    N, K = T.shape
    assert K % QK == 0, K
    if qtype in (QT.TQ1_0, QT.TQ2_0):
        # gguf's quantiser: d = absmax of the block = |s| (or 0 for an all-zero block), codes round(x / d): exact
        w = (T.float() * s.float()[:, None]).numpy()
        return gguf.quants.quantize(w, qtype)
    assert qtype == QT.Q2_K
    sg = torch.where(s < 0, -1, 1).to(torch.int8)
    q = (T * sg[:, None] + 1).to(torch.uint8).numpy().reshape(-1, QK)          # {0, 1, 2}
    nb = q.shape[0]
    d = np.repeat(s.abs().float().numpy().astype(np.float16), K // QK).reshape(nb, 1)
    scales = np.full((nb, QK // 16), 0x11, dtype=np.uint8)                       # scale 1, min 1: y = d q - d
    blk = np.concatenate([scales, _tq2_qs(q), d.view(np.uint8), d.view(np.uint8)], axis=1)
    return blk.reshape(N, -1)


# ------------------------------------------------------------------------------------- float types via libggml

_GGML = None


def ggml_quantize(w: np.ndarray, qtype) -> np.ndarray:
    """float32 [R, C] -> raw blocks (llama.cpp's own quantisers, through libggml-base)"""
    global _GGML
    if qtype in (QT.F32, QT.F16, QT.BF16, QT.Q8_0):
        return gguf.quants.quantize(w.astype(np.float32), qtype)
    if _GGML is None:
        for d in ("build-cpu/bin", "build-cuda/bin", "build/bin"):
            p = LLAMA / d / "libggml-base.so"
            if p.exists():
                _GGML = ctypes.CDLL(str(p))
                break
        assert _GGML is not None, f"libggml-base.so not found under {LLAMA}"
        _GGML.ggml_quantize_chunk.restype = ctypes.c_size_t
        _GGML.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                              ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        _GGML.ggml_row_size.restype = ctypes.c_size_t
        _GGML.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    R, C = w.shape
    rb = _GGML.ggml_row_size(int(qtype), C)
    out = np.empty((R, rb), dtype=np.uint8)
    for r0 in range(0, R, 8192):                     # row blocks: no float copy of the whole matrix
        src = np.ascontiguousarray(w[r0:r0 + 8192], dtype=np.float32)
        n = src.shape[0]
        _GGML.ggml_quantize_chunk(int(qtype), src.ctypes.data, out[r0:r0 + n].ctypes.data, 0, n, C, None)
    return out


def pack_q4_1_intk(codes: torch.Tensor, lo: torch.Tensor, sc: torch.Tensor) -> np.ndarray:
    """int-k codes q [N, K] (< 16) with W = lo + s q per group of g >= 32 ([N, K / g] each) -> Q4_1 blocks (d = s,
    m = lo, the codes as they are): exact up to the fp16 rounding of s and lo"""
    N, K = codes.shape
    g = K // lo.shape[1]
    assert g % 32 == 0 and int(codes.max()) < 16, (g, int(codes.max()))
    q = codes.to(torch.uint8).numpy().reshape(N * K // 32, 32)
    rep = g // 32
    d = sc.float().numpy().astype(np.float16).repeat(rep, axis=1).reshape(-1, 1)
    m = lo.float().numpy().astype(np.float16).repeat(rep, axis=1).reshape(-1, 1)
    qs = (q[:, :16] | (q[:, 16:] << 4)).astype(np.uint8)
    return np.concatenate([d.view(np.uint8), m.view(np.uint8), qs], axis=1).reshape(N, -1)


# ---------------------------------------------------------------------------------------------- the checkpoint

class TTFCheckpoint:
    """the trainer's state dict split into TTF linears (by HF base name) and the float rest"""

    def __init__(self, path):
        path = str(path)
        self.cfg = None
        if path.endswith(".safetensors"):            # the released format (format ttf-1, config in the metadata)
            from safetensors import safe_open
            with safe_open(path, "pt") as f:
                meta = f.metadata() or {}
                sd = {k: f.get_tensor(k) for k in f.keys()}
            self.cfg = json.loads(meta.get("ttf_config", "{}"))
            self.use_beta = bool(self.cfg.get("beta", True))
            self.args = {}
        else:
            ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            sd = ck.get("model", ck)
            self.use_beta = bool(ck.get("beta", True))
            args = ck.get("args", {}) or {}
            self.args = args if isinstance(args, dict) else vars(args)
            if "cfg" in ck and not any(k.startswith("hf.") for k in sd):
                c = ck["cfg"]
                self.cfg = dict(c) if isinstance(c, dict) else dict(vars(c))
        # QwenTern registers hf.model.layers also as layers.: keep one name for each tensor
        sd = {k[3:]: v for k, v in sd.items() if k.startswith("hf.")} or dict(sd)
        # a model trained from scratch (ttf.model.BitTransformer, Meta layout): HF Llama names; its per-head query
        # temperature exp(s_h) (--qk-temp) goes into the query rows' scale
        self.scratch = "tok_emb.weight" in sd or "tok_emb.m.codes" in sd
        self.out_scale = {}
        if self.scratch:
            if any(".experts." in k for k in sd):
                raise NotImplementedError("MoE models from scratch: not exported yet")
            hd = self.cfg["dim"] // self.cfg["n_heads"]
            ren = {}
            for k in sd:
                m = re.fullmatch(r"layers\.(\d+)\.attn\.qk_logscale", k)
                if m:
                    self.out_scale[f"model.layers.{m.group(1)}.self_attn.q_proj"] = \
                        sd[k].float().exp().repeat_interleave(hd)
                    continue
                ren[k] = _scratch_name(k)
            sd = {ren[k]: v for k, v in sd.items() if k in ren}
        bases = sorted({k[:-len(".wpacked")] for k in sd if k.endswith(".wpacked")})
        self.parts = {}            # sub-layer base name -> its tensors
        self.linears = {}          # HF linear base name -> {"main": part, "d": part, "u": part}
        used = set()
        for b in bases:
            p = {s: sd[f"{b}.{s}"] for s in _PART_KEYS if f"{b}.{s}" in sd}
            used.update(k for k in sd if k.startswith(b + ".") and k[len(b) + 1:] in _PART_KEYS + ("frozen_scale",))
            self.parts[b] = p
        for b in bases:
            m = re.fullmatch(r"(.*)\.(lin|ad\.0|ad\.1)", b)
            if m is None:
                self.linears.setdefault(b, {})["main"] = b
            else:
                self.linears.setdefault(m.group(1), {})[{"lin": "main", "ad.0": "d", "ad.1": "u"}[m.group(2)]] = b
        self.floats = {k: v for k, v in sd.items() if k not in used and not k.endswith("frozen_scale")}
        # int-k embedding / head (--intk-emb): W = lo + s q per group; the float W for the converter, the codes for
        # an exact Q4_1 export
        self.intk = {}
        for b in ("model.embed_tokens", "lm_head"):
            if f"{b}.m.codes" in self.floats:
                q, lo, sc = (self.floats.pop(f"{b}.m.{n}") for n in ("codes", "lo", "s"))
                N, K = q.shape
                g = K // lo.shape[1]
                self.floats[f"{b}.weight"] = (lo.float()[..., None] + sc.float()[..., None]
                                              * q.float().reshape(N, K // g, g)).reshape(N, K)
                self.intk[f"{b}.weight"] = (q, lo, sc)
        left = [k for k in self.floats if k.endswith((".codes", ".m.lo", ".m.s"))]
        assert not left, f"unknown int-k tensors {left[:3]}"

    def width(self, b):
        """K of sub-layer b: from its column scale / input mean, else the multiple of 8 among the widths its packed
        rows (5 trits per byte) can hold"""
        p = self.parts[b]
        if "col_scale" in p or "in_mean" in p:
            return (p["col_scale"] if "col_scale" in p else p["in_mean"]).shape[0]
        B = p["wpacked"].shape[1]
        K = [k for k in range(5 * B - 4, 5 * B + 1) if k % 8 == 0]
        assert len(K) == 1, f"{b}: width unknown ({B} packed bytes per row, no col_scale / in_mean)"
        return K[0]

    def part(self, b, K):
        """sub-layer b -> (T int8 [N, K], s = r * beta fp32 [N], c [K] | None, mu [K] | None, bias [N] | None)"""
        p = self.parts[b]
        T = unpack_trits(p["wpacked"], K)
        beta = trit_beta(T) if self.use_beta else 1.0
        r = p["row_scale"].float() if "row_scale" in p else torch.ones(T.shape[0])
        c = p["col_scale"].float() if "col_scale" in p else None
        return T, r * beta, c, (p["in_mean"].float() if "in_mean" in p else None), \
            (p["out_bias"].float() if "out_bias" in p else None)

    def mag(self, b):
        """the float low-rank adapter of a scratch layer, y = r (beta T Q(c x) + (c x) B A^T): -> (B^T [k, K],
        A [N, k]) or None"""
        p = self.parts[b]
        return (p["mag_B"].float().t().contiguous(), p["mag_A"].float()) if "mag_A" in p else None


_PART_KEYS = ("wpacked", "row_scale", "col_scale", "in_mean", "out_bias", "mag_A", "mag_B")
_SCRATCH = {"attn.wq": "self_attn.q_proj", "attn.wk": "self_attn.k_proj", "attn.wv": "self_attn.v_proj",
            "attn.wo": "self_attn.o_proj", "ffn.w_gate": "mlp.gate_proj", "ffn.w_up": "mlp.up_proj",
            "ffn.w_down": "mlp.down_proj", "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}


def _scratch_name(k):
    if k == "tok_emb.weight":
        return "model.embed_tokens.weight"
    if k.startswith("tok_emb.m."):                   # int-k codes
        return "model.embed_tokens" + k[len("tok_emb"):]
    if k.startswith("lm_head.m."):
        return k
    if k == "norm.weight":
        return "model.norm.weight"
    if k == "lm_head.weight":
        return "lm_head.weight"
    m = re.fullmatch(r"layers\.(\d+)\.(attn\.w[qkvo]|ffn\.w_(?:gate|up|down)|attn_norm|ffn_norm)\.(.*)", k)
    assert m, f"unknown scratch tensor {k}"
    return f"model.layers.{m.group(1)}.{_SCRATCH[m.group(2)]}.{m.group(3)}"


def _llama_config(cfg):
    """ttf ModelConfig -> an HF LlamaForCausalLM config.json"""
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": cfg["vocab_size"],
            "hidden_size": cfg["dim"], "num_hidden_layers": cfg["n_layers"], "num_attention_heads": cfg["n_heads"],
            "num_key_value_heads": cfg.get("n_kv_heads") or cfg["n_heads"], "intermediate_size": cfg["hidden_dim"],
            "max_position_embeddings": cfg["max_seq_len"], "rope_theta": cfg["rope_theta"],
            "rms_norm_eps": cfg["norm_eps"], "tie_word_embeddings": bool(cfg.get("tie_embeddings", True)),
            "hidden_act": "silu", "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float32"}


def _hf_dir_for(hf_dir: Path, ck: TTFCheckpoint, tmp: Path) -> Path:
    """a copy of the HF dir's config and tokenizer (no weights); untied when the checkpoint has its own head"""
    for f in hf_dir.iterdir():
        if f.is_file() and not f.name.endswith((".safetensors", ".bin", ".pt")) and f.name != "model.safetensors.index.json":
            (tmp / f.name).symlink_to(f.resolve())
    if ck.scratch:
        cfg = _llama_config(ck.cfg)
        if (tmp / "config.json").exists():
            (tmp / "config.json").unlink()
        (tmp / "config.json").write_text(json.dumps(cfg, indent=2))
        return tmp
    cfg = json.loads((hf_dir / "config.json").read_text())
    if "lm_head.weight" in ck.floats:
        cfg["tie_word_embeddings"] = False
        if "text_config" in cfg:
            cfg["text_config"]["tie_word_embeddings"] = False
    (tmp / "config.json").unlink()
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2))
    return tmp


LLAMA2_TOKENIZER = "~/.cache/huggingface/hub/models--hf-internal-testing--llama-tokenizer/snapshots/" \
                   "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0"


def export(ckpt, hf_dir, out, wtype="tq2_0", emb=None, head=None, log=print):
    """hf_dir: the base model's config and tokenizer (conversions), or for a model trained from scratch the
    tokenizer's directory (default: the Llama-2 32k tokenizer the data scripts use)"""
    ck = TTFCheckpoint(ckpt)
    # default: int-k codes exactly (Q4_1), a float matrix as Q6_K
    emb = emb or ("q4_1" if "model.embed_tokens.weight" in ck.intk else "q6_k")
    head = head or ("q4_1" if "lm_head.weight" in ck.intk else "q6_k")
    hf_dir = Path(hf_dir or LLAMA2_TOKENIZER).expanduser()
    tmp = Path(tempfile.mkdtemp(prefix="ttfgguf_"))
    try:
        _hf_dir_for(hf_dir, ck, tmp)
        hparams = json.loads((tmp / "config.json").read_text())
        base = get_model_class(hparams["architectures"][0])
        if hasattr(base, "no_mtp"):
            base.no_mtp = True                      # the converted model has no MTP block
        model = _make_class(base, ck, WTYPES[wtype], FTYPES_EMB[emb], FTYPES_EMB[head], log)(
            tmp, FTYPES[wtype], Path(out), eager=True)
        model.write()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _make_class(base, ck, wq, eq, hq, log):
    class TTFModel(base):
        model_arch = base.model_arch
        def index_tensors(self, remote_hf_model_id=None):
            hp = {**self.hparams, **self.hparams.get("text_config", {})}
            type(self)._original_block_count = hp.get("num_hidden_layers")
            out = {}
            for name, t in ck.floats.items():
                if (item := self.filter_tensors((name, lambda t=t: t))) is not None:
                    out[item[0]] = item[1]
            return out

        def set_gguf_parameters(self):
            super().set_gguf_parameters()
            self.gguf_writer.add_uint32("ttf.format", 1)
            self.gguf_writer.add_string("ttf.act_quant", "trained with int8 per token")

        def tensor_force_quant(self, name, new_name, bid, n_dims):
            if new_name in ("token_embd.weight", "output.weight"):
                return QT.F32                       # quantised in add_tensor below, by libggml
            if n_dims >= 2 and new_name.endswith(".weight") and "conv1d" not in new_name:
                return QT.F16
            return False

        def prepare_tensors(self):
            add = self.gguf_writer.add_tensor

            def add_tensor(name, tensor, raw_shape=None, raw_dtype=None, **kw):
                qt = {"token_embd.weight": eq, "output.weight": hq}.get(name)
                ik = ck.intk.get(INTK_GGUF.get(name, ""))
                if ik is not None and qt == QT.Q4_1:
                    W = ck.floats[INTK_GGUF[name]]
                    rows = slice(None, None, max(1, W.shape[0] // 64))     # the converter passed W through as is
                    assert tuple(tensor.shape) == tuple(W.shape) and np.allclose(tensor[rows], W[rows].numpy(), atol=1e-6), \
                        f"{name}: the converter changed the int-k matrix; export it with --emb / --head q6_k"
                    log(f"{name}: int-k codes as Q4_1 (exact)")
                    return add(name, pack_q4_1_intk(*ik), raw_shape=raw_shape, raw_dtype=QT.Q4_1, **kw)
                if qt is not None and qt != QT.F32 and tensor.dtype == np.float32:
                    if tensor.shape[-1] % QK and qt not in (QT.F16, QT.BF16, QT.Q8_0):
                        qt = QT.Q8_0 if tensor.shape[-1] % 32 == 0 else QT.F16     # K-quants need rows of 256
                    tensor, raw_dtype = ggml_quantize(tensor, qt), qt
                    log(f"{name}: {qt.name}")
                return add(name, tensor, raw_shape=raw_shape, raw_dtype=raw_dtype, **kw)
            self.gguf_writer.add_tensor = add_tensor
            self._write_ttf()
            super().prepare_tensors()

        def _perm(self, name, bid, N, K):
            """the converter's renames and row / column reorders of the HF weight `name` [N, K], read off index
            matrices: -> [(gguf name, row index, column index)]"""
            if ck.scratch:                          # Meta layout already: the GGUF order, no reorder
                return [(self.map_tensor_name(name), torch.arange(N), torch.arange(K))]
            Rm = torch.arange(N, dtype=torch.float32)[:, None].expand(N, K).contiguous()
            Cm = torch.arange(K, dtype=torch.float32)[None, :].expand(N, K).contiguous()
            ro = list(self.modify_tensors(Rm, name, bid))
            co = list(self.modify_tensors(Cm, name, bid))
            assert [n for n, _ in ro] == [n for n, _ in co], (name, ro, co)
            res = []
            for (nn, R), (_, C) in zip(ro, co):
                R, C = torch.as_tensor(R), torch.as_tensor(C)
                assert R.dim() == 2, (name, nn, R.shape)
                ri, ci = R[:, 0].long(), C[0, :].long()
                assert torch.equal(R, ri[:, None].float().expand_as(R)) and torch.equal(C, ci[None].float().expand_as(C)), \
                    f"{name}: the converter does more than reorder rows / columns"
                res.append((nn, ri, ci))
            return res

        def _vec(self, name, v):
            self.gguf_writer.add_tensor(name, np.ascontiguousarray(v.float().numpy()), raw_dtype=QT.F32)

        def _w(self, name, T, s):
            if T.shape[1] % QK:
                # rows not a multiple of 256 (small scratch models): Q8_0 holds {-1, 0, 1} d exactly too, at 8.5 bits
                qt = QT.Q8_0 if T.shape[1] % 32 == 0 else QT.F16
                log(f"{name}: {T.shape[1]} columns, not a multiple of {QK}: stored as {qt.name}")
                w = (T.float() * s.float()[:, None]).numpy()
                self.gguf_writer.add_tensor(name, gguf.quants.quantize(w, qt), raw_dtype=qt)
                return
            self.gguf_writer.add_tensor(name, pack_ternary(T, s, wq), raw_dtype=wq)

        def _write_ttf(self):
            nb = 0
            for hb, sub in ck.linears.items():
                hname = hb + ".weight"
                item = self.filter_tensors((hname, lambda: None))
                if item is None:
                    continue
                hname = item[0]
                bid = next((int(p) for p in hname.split(".") if p.isdecimal()), None)
                p = ck.parts[sub["main"]]
                N = p["wpacked"].shape[0]
                K = ck.width(sub["main"])
                T, s, c, mu, b = ck.part(sub["main"], K)
                mag = ck.mag(sub["main"])
                if hb in ck.out_scale:              # the query temperature: the whole output scales
                    e = ck.out_scale[hb]
                    s, b = s * e, (b * e if b is not None else None)
                    mag = (mag[0], mag[1] * e[:, None]) if mag is not None else None
                for gname, ri, ci in self._perm(hname, bid, N, K):
                    stem = gname[:-len(".weight")]
                    self._w(gname, T[ri][:, ci], s[ri])
                    if c is not None:
                        self._vec(stem + ".ttf_c", c[ci])
                    if mu is not None:
                        self._vec(stem + ".ttf_m", -(c[ci] if c is not None else 1.0) * mu[ci])
                    if b is not None:
                        self._vec(stem + ".ttf_b", b[ri])
                    nb += T.numel()
                    if mag is not None:             # float adapter on c x, before the row scale: U = r A
                        Bt, A = mag
                        r = ck.parts[sub["main"]]["row_scale"].float()
                        if hb in ck.out_scale:
                            r = r * ck.out_scale[hb]
                        f = ".ttf_fd" if "d" in sub else ".ttf_d"     # beside a ternary adapter: the second side adapter
                        self.gguf_writer.add_tensor(stem + f, Bt[:, ci].numpy().astype(np.float16), raw_dtype=QT.F16)
                        if c is not None:
                            self._vec(stem + f + "_c", c[ci])
                        if mu is not None:          # the adapter reads the centred input too: c (x - mu)
                            self._vec(stem + f + "_m", -(c[ci] if c is not None else 1.0) * mu[ci])
                        self.gguf_writer.add_tensor(stem + f[:-1] + "u", (r[:, None] * A)[ri].numpy().astype(np.float16),
                                                    raw_dtype=QT.F16)
                    if "d" in sub:
                        self._adapter(stem, sub, K, N, ri, ci)
            log(f"TTF: {len(ck.linears)} linears, {nb / 1e6:.1f} M trits as {wq.name}")

        def _adapter(self, stem, sub, K, N, ri, ci):
            Dp, Up = ck.parts[sub["d"]], ck.parts[sub["u"]]
            r = Dp["wpacked"].shape[0]
            rp = -(-r // QK) * QK
            Td, sd_, cd, mud, bd = ck.part(sub["d"], K)
            Tu, su, cu, muu, bu = ck.part(sub["u"], r)
            pad = rp - r
            Td = torch.cat([Td[:, ci], Td.new_zeros(pad, K)])
            sd_ = torch.cat([sd_, sd_.new_zeros(pad)])
            self._w(stem + ".ttf_d", Td, sd_)
            if cd is not None:
                self._vec(stem + ".ttf_d_c", cd[ci])
            if mud is not None:
                self._vec(stem + ".ttf_d_m", -(cd[ci] if cd is not None else 1.0) * mud[ci])
            if bd is not None:
                self._vec(stem + ".ttf_d_b", torch.cat([bd, bd.new_zeros(pad)]))
            Tu = torch.cat([Tu[ri], Tu.new_zeros(N, pad)], 1)
            self._w(stem + ".ttf_u", Tu, su[ri])
            if cu is not None:
                self._vec(stem + ".ttf_u_c", torch.cat([cu, cu.new_zeros(pad)]))
            if muu is not None:
                cu_ = cu if cu is not None else torch.ones(r)
                self._vec(stem + ".ttf_u_m", torch.cat([-cu_ * muu, muu.new_zeros(pad)]))
            if bu is not None:
                self._vec(stem + ".ttf_u_b", bu[ri])

    return TTFModel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt"); ap.add_argument("hf_dir", help="base model dir (config + tokenizer), or '-' for a "
                    "scratch model with the Llama-2 tokenizer"); ap.add_argument("out")
    ap.add_argument("--wtype", default="tq2_0", choices=sorted(WTYPES))
    ap.add_argument("--emb", default=None, choices=sorted(FTYPES_EMB), help="default: q4_1 for int-k codes (exact), else q6_k")
    ap.add_argument("--head", default=None, choices=sorted(FTYPES_EMB))
    a = ap.parse_args()
    with torch.inference_mode():
        export(a.ckpt, None if a.hf_dir == "-" else a.hf_dir, a.out, a.wtype, a.emb, a.head)


if __name__ == "__main__":
    main()
