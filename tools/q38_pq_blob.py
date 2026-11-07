"""ple_pq4 (codes.bin all rows, keep.bin) -> the GGUF pq_blob bytes on stdout: keep bitmap uint64 [n_w] (row r = bit
r & 63 of word r >> 6), rank uint32 [n_w] (kept rows before word w), the kept rows' codes uint8 [n_kept, nS] in row order"""
import sys, numpy as np
d = sys.argv[1]; nS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
keep = np.fromfile(f"{d}/keep.bin", dtype=np.uint8).astype(bool)
total = keep.size; n_w = (total + 63) // 64
kp = np.zeros(n_w * 64, dtype=bool); kp[:total] = keep
bits = np.packbits(kp.reshape(n_w, 64)[:, ::-1], axis=1, bitorder="big").view(">u8").astype("<u8").reshape(-1)
cnt = kp.reshape(n_w, 64).sum(1)
rank = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype("<u4")
assert int(cnt.sum()) < 2**32
out = sys.stdout.buffer
out.write(bits.tobytes()); out.write(rank.tobytes())
codes = np.memmap(f"{d}/codes.bin", dtype=np.uint8, mode="r").reshape(total, nS)
CH = 1 << 22
n = 0
for r0 in range(0, total, CH):
    sel = codes[r0:r0 + CH][keep[r0:r0 + CH]]
    out.write(np.ascontiguousarray(sel).tobytes()); n += len(sel)
out.flush()
print(f"total {total} kept {n} n_w {n_w} bytes {12 * n_w + n * nS}", file=sys.stderr)
