"""--ext ttfhome.train_ext[:untie=1,rank=R]: the structure of a converted model, for training it (ttf train --init) from a
checkpoint that gguf_import.py rebuilt from a GGUF. Structure only: nothing here fits or converts anything.

  untie=1   the head is its own tensor (not the embedding's): a GGUF with output.weight
  rank=R    every ternary decoder linear y = lin(x) + up(down(x)), down [r, K] and up [N, r] ternary layers; R a fraction
            of min(K, N) (r = max(8, round(R min(K, N) / 8) * 8)) or an int; the state-dict keys .lin, .ad.0, .ad.1
"""
import torch.nn as nn


def rank_of(K, N, r):
    r = float(r)
    return max(8, int(round(r * min(K, N) / 8)) * 8) if r < 1 else int(r)


class Adapted(nn.Module):
    """y = lin(x) + up(down(x)); the sum goes into the up layer's output (acc)"""

    def __init__(self, lin, down, up):
        super().__init__()
        self.lin, self.ad = lin, nn.ModuleList([down, up])
        self.in_features, self.out_features = lin.K, lin.N

    def forward(self, x):
        return self.ad[1](self.ad[0](x), acc=self.lin(x))


class Structure:
    def __init__(self, spec):
        o = dict(kv.split("=", 1) for kv in spec.split(",") if kv)
        self.untie = o.get("untie", "0") == "1"
        self.rank = o.get("rank", "")

    def build(self, model, swapped, make_linear, log):
        hf = model.hf
        if self.untie and hf.lm_head.weight is hf.model.embed_tokens.weight:
            hf.lm_head.weight = nn.Parameter(hf.lm_head.weight.detach().clone(), requires_grad=False)
        if not self.rank:
            return swapped
        dev = next(p for p in model.parameters() if not p.is_meta).device
        layers = model.hf.model.layers
        n = 0
        for name in list(swapped):
            lin = swapped[name]
            r = rank_of(lin.K, lin.N, self.rank)
            q = Adapted(lin, make_linear(lin.K, r).to(dev), make_linear(r, lin.N).to(dev))
            par, cn = name[len("model.layers."):].rsplit(".", 1)
            setattr(layers.get_submodule(par), cn, q)
            swapped[name] = q
            n += 1
        log(f"ttfhome.train_ext: {n} ternary adapters (rank {self.rank})" + (", head untied" if self.untie else ""))
        return swapped


def ttf_ext(spec):
    return Structure(spec)
