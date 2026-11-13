"""dumpcheck.py: teacher-forced per-op check of the runtime against the IR reference (ttf_ir).

Run the runtime with TTF_DUMP=DIR (every IR value of one call as raw f32 / i32 + DIR/index.json with ggml ne), then
  python dumpcheck.py MODEL.ttf DIR
Every IR node is run by ttf_ir.OPS on the runtime's own inputs and compared with the runtime's output
(worst rel L2 and the fraction of differing elements per op kind).
"""
import sys, json, numpy as np, torch
import os
sys.path.insert(0, os.environ.get('GGUF_PY', os.path.expanduser('~/Programs/llama.cpp/gguf-py')))
sys.path.insert(0, os.environ.get('TTF_IR_PY', os.path.expanduser('~/Projects/ttf/store/stack')))
import gguf, ttf_ir
torch.set_num_threads(4)
path, ddir = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(path)
kv = {f.name: f for f in r.fields.values()}
s = lambda k: kv[k].contents()
ir = json.loads(s('ttf.ir'))
W = {}
for t in r.tensors:
    a = np.array(t.data).reshape([int(x) for x in reversed(list(t.shape))])
    W[t.name] = torch.from_numpy(a.view(np.uint8).copy()) if t.name.endswith('.wpacked') else torch.from_numpy(a.astype(np.float32))
D = ttf_ir.DTYPES[s('ttf.compute_dtype')]
run = ttf_ir.Runner(ir, W, D)
idx = json.load(open(ddir + '/index.json'))


def load(name):
    e = idx[name]; ne = e['ne']
    nd = max([i + 1 for i, n in enumerate(ne) if n > 1] + [1])
    a = np.fromfile(f'{ddir}/{name}.bin', np.float32 if e['type'] == 'f32' else np.int32)
    return torch.from_numpy(a.reshape(list(reversed(ne[:nd]))))


worst, st = {}, {}
for n in ir['graphs']['step']['nodes']:
    if any(i is not None and i not in idx for i in n['in']) or n['out'][0] not in idx:
        print('skip', n['op'], n['out']); continue
    ins = [load(i) if i is not None else None for i in n['in']]
    ref = ttf_ir.OPS[n['op']](ins, n, run.W, st, D)[0].float()
    eng = load(n['out'][0]).reshape(ref.shape)
    d = (eng - ref).abs()
    rel = (d.norm() / ref.norm().clamp_min(1e-30)).item()
    neq = (d > 0).float().mean().item()
    k = n['op'] + ('' if n['op'] != 'ttf.linear' else ('.' + n['p'].split('.')[-1]))
    w = worst.setdefault(k, [0, 0, 0, ''])
    w[2] += 1
    if rel >= w[0]:
        w[0], w[1], w[3] = rel, neq, n['out'][0]
for k, (rel, neq, cnt, where) in sorted(worst.items()):
    print(f'{k:22s} n={cnt:3d} worst rel {rel:.2e}  frac!=0 {neq:.4f}  at {where}')
