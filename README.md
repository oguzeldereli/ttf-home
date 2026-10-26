# ttf-home

Ternary language models on a home computer: train one from scratch, convert a Qwen3.5 / 3.8 model to ternary and keep
training it, fine-tune on your own text, and run the result fast on a consumer GPU or CPU.

Training uses TTF (two-timescale threshold flips, `ttf-core`): weights stay in {-1, 0, +1} the whole time, with a
low-rank accumulator in place of master weights and Adam states, so training needs a fraction of the usual memory.
Inference uses llama.cpp with two small patches (`llama.cpp/`), so a TTF model is an ordinary GGUF file.

## Measured (laptop: RTX 4080 Laptop 12 GB, i9-14900HX)

| model | file | GPU decode | GPU prefill | CPU decode (8 threads) |
|---|---|---|---|---|
| Qwen3.8-27B ternary (training-free conversion, rank-1024 ternary adapters) | 9.2 GiB | 26.5 tok/s | 690 tok/s | 3.0 tok/s |
| same, `--wtype tq1_0` (1.69 bits) | 7.9 GiB | 18.2 tok/s | 510 tok/s | - |
| Qwen3.5-0.8B ternary (TQ2_0) | 0.61 GB | 256 tok/s | 12.9k tok/s | 49 tok/s |
| Qwen3.5-0.8B bf16 (reference) | 1.5 GB | 181 tok/s | 13.7k tok/s | 31 tok/s |

The 27B model fits entirely on the 12 GB card, with a 17k-token context.

The export is exact: next-token loss in llama.cpp matches the trainer's own forward. On 10 x 512 tokens of Wikipedia:

| model | trainer | llama.cpp |
|---|---|---|
| Qwen3.5-0.8B conversion | 2.5897 | 2.5883 |
| 340M scratch model | 2.3100 | 2.3107 |

The small difference comes from activation quantisation: llama.cpp quantises activations per block of 32 or 256
values, while training quantises them per token.

## Install

    pip install -e ../ttf-core -e .           # plus -e ../ttf-conversion for `ttf convert`
    ttf doctor                                # what this machine can train and run
    ttf engine build                          # llama.cpp + the TTF patches (CUDA, Metal, Vulkan or CPU)

## Use

    ttf data my_notes/ books.jsonl --out data/mine            # text / Markdown / JSONL -> token files
    ttf train --data data/mine --out runs/mine --hours 8      # from scratch; the size is chosen by GPU memory
    ttf export runs/mine/ckpt.pt mine.gguf
    ttf generate mine.gguf "Once upon a time"

    ttf convert --base Qwen3.5-0.8B-Base --data data/web --out runs/q08        # convert to ternary + recover
    ttf finetune --ckpt runs/q08/ckpt.pt --base Qwen3.5-0.8B-Base --data data/mine --out runs/q08-mine
    ttf export runs/q08-mine/ckpt.pt q08-mine.gguf --base Qwen3.5-0.8B-Base
    ttf chat q08-mine.gguf          # or: ttf serve q08-mine.gguf   (OpenAI-compatible HTTP API)

    ttf bench mine.gguf ; ttf score mine.gguf data/mine_val.bin

TTF knobs on `train`, `convert` and `finetune`:

| knob | meaning |
|---|---|
| `--theta` | firing threshold, in steps (16 is the measured best at every scale tried) |
| `--rank` | accumulator rank |
| `--lr`, `--min-lr`, `--steps`, `--warmup` | the learning-rate schedule |
| `--seq-len`, `--batch`, `--accum` | the batch shape (the default is 32k tokens per step) |
| `-- ...` | any other trainer option (`python -m ttf.train --help`) |

## Model sizes by GPU

`ttf train --size auto` picks a size from GPU memory:

| GPU memory | size | parameters |
|---|---|---|
| under 7 GB | tiny | 15M |
| 7 to 11 GB | small | 110M |
| 11 to 22 GB | medium | 340M |
| 22 GB and up | large | 670M |

Converting 0.8B models needs about 12 GB. A 27B conversion needs a 48 to 96 GB card; running the result needs about
10 GB.

## GGUF format

- **Which `--wtype`**: `tq2_0` on NVIDIA GPUs and CPUs (fastest); `tq1_0` to save 15% of memory; `q2_k` (2.63
  bits) for AMD / Intel GPUs (Vulkan) and Macs (Metal). llama.cpp has no TQ kernels there yet, but Q2_K runs
  everywhere and holds ternary weights exactly too.
- **Ternary matrices** are stored as `TQ2_0` (2.06 bits per weight, the default), `TQ1_0` (1.69 bits) or `Q2_K`.
  All three hold {-1, 0, +1}·d exactly, with the row scale folded into d.
  - Matrices whose width is not a multiple of 256 fall back to `Q8_0`, which is also exact.
- **Embedding and head**: K-quants (Q6_K by default; a Q4_K embedding costs nothing measurable, a Q4_K head +0.011
  NLL at 0.8B). Int-k codes (`--intk-emb`) export exactly as Q4_1 (d = s, m = lo, the codes as they are).
- **TTF side tensors**, as F32 vectors per linear layer:
  - `<weight>.ttf_c`: input scale;
  - `.ttf_m`: centring, stored as -c·mu;
  - `.ttf_b`: output bias.
- **Ternary adapters** are stored as `.ttf_d*` and `.ttf_u*`.
- The patched llama.cpp applies all of these inside `build_lora_mm`, so every architecture llama.cpp supports gets
  them without per-model code.
- **CUDA** (patches 0002-0009): matrix-vector kernels for `TQ2_0` and `TQ1_0`, prefill for both through the Q8_0 MMQ kernels, and
  the side tensors fused into the activation quantiser of both (bit-identical; +10-20% tok/s), and one warp per row and two ints per thread at batch 1 (+16% at 27B). `test-backend-ops`: 22 / 22
  `TQ1_0` / `TQ2_0` matrix products match the CPU. `TQ2_0` is the fastest; `TQ1_0` is 15% smaller and 31% slower.
