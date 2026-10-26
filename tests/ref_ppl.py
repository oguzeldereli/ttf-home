"""The trainer's own forward (ttf-core + ttfconv, CUDA) on the protocol of llama-perplexity, for checking an export:
chunks of C tokens, each scored on its second half (positions C/2 .. C-2 predicting the next token), no BOS.
Also writes the chunks' text, so llama-perplexity sees the same tokens (it re-tokenizes; the token count it prints
must match).
  python tests/ref_ppl.py CKPT HF_DIR VAL.bin --chunks 40 [--ctx 512] --text out.txt
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


def build(hf_dir, exts):
    from ttf import ext as X
    from ttf.qwen import build_qwen
    from ttf.layers import KernelTernaryLinear, enable_rc_scales
    X.load(exts)
    mk = lambda i, o: KernelTernaryLinear(i, o, 8, beta=True)
    m, sw = build_qwen(hf_dir, "kernel", grad_checkpoint=False, device="cuda", make_linear=mk)
    for f in X.having("build"):
        sw = f(m, sw, mk, print)
    enable_rc_scales(m)
    return m.cuda().eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("hf_dir"); ap.add_argument("val")
    ap.add_argument("--chunks", type=int, default=40); ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--text", default=None); ap.add_argument("--ext", action="append", default=None)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    args = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    exts = a.ext if a.ext is not None else [e for e in args.get("ext", []) if not e.startswith("ttfconv.cd")]
    m = build(a.hf_dir, exts)
    sd = ck["model"]
    from ttf.layers import KernelTernaryLinear
    for n, mod in m.named_modules():
        if isinstance(mod, KernelTernaryLinear) and f"{n}.in_mean" in sd and getattr(mod, "in_mean", None) is None:
            mod.enable_centering(sd[f"{n}.in_mean"].cuda(), sd[f"{n}.out_bias"].cuda())
    missing, _ = m.load_state_dict(sd, strict=False)
    assert not missing, missing[:4]
    for mod in m.modules():
        if hasattr(mod, "_invalidate"):
            mod._invalidate()
    d = np.memmap(a.val, dtype=np.uint32, mode="r")
    C, first = a.ctx, a.ctx // 2
    toks = torch.from_numpy(np.asarray(d[:a.chunks * C], dtype=np.int64))
    nll = cnt = 0.0
    for i in range(a.chunks):
        t = toks[i * C:(i + 1) * C].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(t[None])[0][0].float()
        nll += float(F.cross_entropy(lg[first:C - 1], t[first + 1:C], reduction="sum"))
        cnt += C - 1 - first
    print(json.dumps({"ckpt": a.ckpt, "chunks": a.chunks, "ctx": C, "nll": nll / cnt, "ppl": math.exp(nll / cnt)}))
    if a.text:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.hf_dir)
        open(a.text, "w").write(tok.decode(toks.tolist()))


if __name__ == "__main__":
    main()
