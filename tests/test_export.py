"""The GGUF export holds every ternary matrix exactly: a small random TTF model (the trainer's own layers, with
centring and an int-k embedding) is exported as TQ2_0 / TQ1_0 / Q2_K; each weight read back from the file and
dequantized by gguf-py must equal T * r * beta up to the fp16 rounding of the block scale, and the side tensors and
the int-k Q4_1 embedding must match the checkpoint.
  PYTHONPATH=../ttf-core:. pytest tests/test_export.py      (CPU; needs the llama.cpp gguf-py, TTF_LLAMA_CPP)
"""
import numpy as np
import pytest
import torch

from ttfhome.gguf_export import LLAMA2_TOKENIZER, TTFCheckpoint, export, gguf


def _model(tmp_path, intk):
    from ttf.config import ModelConfig
    from ttf.layers import build_kernel_transformer
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32000, dim=256, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=768, max_seq_len=256)
    m = build_kernel_transformer(cfg, grad_checkpoint=False, beta=True)
    sd = m.state_dict()
    g = torch.Generator().manual_seed(1)
    for k, v in list(sd.items()):
        if k.endswith("row_scale"):
            sd[k] = 0.01 + torch.rand(v.shape, generator=g) * 0.05
    if intk:          # the int-k embedding of --intk-emb 2:3 (tied): W = lo + s q per group of 64
        E = sd.pop("tok_emb.weight")
        N, K = E.shape
        sd["tok_emb.m.codes"] = torch.randint(0, 4, (N, K), generator=g, dtype=torch.int8)
        sd["tok_emb.m.lo"] = torch.randn(N, K // 64, generator=g) * 0.02
        sd["tok_emb.m.s"] = torch.rand(N, K // 64, generator=g) * 0.01 + 1e-3
    p = tmp_path / "ckpt.pt"
    torch.save({"model": sd, "cfg": cfg, "beta": True, "args": {}}, p)
    return p


@pytest.mark.parametrize("wtype", ["tq2_0", "tq1_0", "q2_k"])
def test_ternary_exact(tmp_path, wtype):
    ck_path = _model(tmp_path, intk=(wtype == "tq2_0"))
    out = tmp_path / "m.gguf"
    export(ck_path, LLAMA2_TOKENIZER, out, wtype, log=lambda *a: None)
    ck = TTFCheckpoint(ck_path)
    r = gguf.GGUFReader(str(out))
    t = {x.name: x for x in r.tensors}
    names = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
             "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
             "mlp.down_proj": "ffn_down"}
    n = 0
    for hb, sub in ck.linears.items():
        L, mod = hb.split(".")[2], ".".join(hb.split(".")[3:])
        stem = f"blk.{L}.{names[mod]}"
        w = t[stem + ".weight"]
        K = int(w.shape[0])
        T, s, c, mu, b = ck.part(sub["main"], K)
        got = gguf.quants.dequantize(w.data, w.tensor_type).reshape(T.shape)
        ref = (T.float() * s[:, None]).numpy()
        # Meta-layout q / k rows are permuted by the converter: compare rows as sets of (row content)
        if mod in ("self_attn.q_proj", "self_attn.k_proj"):
            got, ref = np.sort(got, axis=0), np.sort(ref, axis=0)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=0)     # fp16 block scale
        if c is not None:
            np.testing.assert_allclose(np.asarray(t[stem + ".weight.ttf_c"].data if stem + ".weight.ttf_c" in t
                                                  else t[stem + ".ttf_c"].data), c.numpy(), rtol=1e-6)
        n += 1
    assert n == 2 * 7
    if wtype == "tq2_0":
        e = t["token_embd.weight"]
        assert e.tensor_type == gguf.GGMLQuantizationType.Q4_1
        q, lo, sc = ck.intk["model.embed_tokens.weight"]
        N, K = q.shape
        f16 = lambda v: v.float().half().float()[..., None]                      # Q4_1 keeps d = s, m = lo in fp16
        W = (f16(lo) + f16(sc) * q.float().reshape(N, K // 64, 64)).reshape(N, K).numpy()
        got = gguf.quants.dequantize(e.data, e.tensor_type).reshape(W.shape)
        np.testing.assert_allclose(got, W, rtol=1e-6, atol=1e-7)
