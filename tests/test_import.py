"""gguf_import's trit codecs against the exporters' (no GGUF file or engine run needed beyond gguf-py)"""
import torch

from ttfhome.gguf_export import unpack_trits
from ttfhome.gguf_import import dec_tq1p, pack_rows, _inv_perm
from ttfhome.q38_export import enc_tq1p


def test_tq1p_roundtrip():
    g = torch.Generator().manual_seed(0)
    T = torch.randint(-1, 2, (7, 480), generator=g).to(torch.int8)
    assert torch.equal(dec_tq1p(torch.from_numpy(enc_tq1p(T)), 480), T)


def test_pack_rows_matches_trainer_layout():
    g = torch.Generator().manual_seed(1)
    for K in (5, 8, 13, 1024, 3584):
        T = torch.randint(-1, 2, (3, K), generator=g).to(torch.int8)
        P = pack_rows(T)
        assert P.shape == (3, -(-K // 5)) and P.dtype == torch.uint8
        assert torch.equal(unpack_trits(P, K), T)


def test_inv_perm():
    ri = torch.randperm(10)
    assert torch.equal(ri[_inv_perm(ri, 10)], torch.arange(10))
