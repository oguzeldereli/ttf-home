# ttf-home

Ternary language models on a home computer:
- run the released ternary Qwen3.8 models;
- train a model from scratch;
- convert a Qwen3.5 / 3.8 model to ternary and keep training it;
- fine-tune on your own text;
- run the result fast on a consumer GPU or CPU.

One command line, `ttf`, does all of it.

Training uses TTF ([ttf-ternary](https://github.com/oguzeldereli/ttf-ternary)): the weights stay in {-1, 0, +1} the
whole time, with a low-rank accumulator in place of master weights and Adam states, so training needs a fraction of the
usual memory. Inference uses llama.cpp with the TTF patches in `llama.cpp/`, so a TTF model is an ordinary GGUF file
that `llama-cli` / `llama-server` run.

| repository | what it does |
|---|---|
| [ttf-ternary](https://github.com/oguzeldereli/ttf-ternary) | the trainer (TTF), and the paper |
| [ttf-conversion](https://github.com/oguzeldereli/ttf-conversion) | pretrained Hugging Face model to ternary checkpoint |
| **ttf-home** (this one) | data, train, convert, fine-tune, export to GGUF, chat / serve |

## Quickstart: chat with a ternary 27B model

You need `git`, `cmake`, a C++ compiler and Python 3.11+; for an NVIDIA GPU, also the CUDA toolkit (`nvcc`). Running
models needs no PyTorch.

```bash
git clone https://github.com/oguzeldereli/ttf-home && cd ttf-home
pip install -e . huggingface_hub
ttf doctor                       # what this machine has and can run
ttf engine build --q38           # llama.cpp + the TTF patches, into ~/.cache/ttf/llama.cpp-q38 (CUDA when a GPU is found)

hf download oguzeldereli/ttf-Qwen3.8-27B-GGUF ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf --local-dir models/ttf-27b
ttf chat models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
ttf serve models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf --port 8080     # OpenAI-compatible API at http://127.0.0.1:8080/v1
```

The whole model, 8.2 GB (1.6 bits per ternary weight), fits on one 12 GB NVIDIA GPU: 29 tokens/s decode and 748
tokens/s prefill on an RTX 3090 Ti. Without a GPU it runs on the CPU, slower.

There are two engines. Both released models use TQ1_P weights (160 trits in 32 bytes), which only the second one,
`--q38` (the patches in `llama.cpp/q38/`), has; `ttf` picks the engine from the GGUF file. Models you export yourself
with the default `--wtype tq2_0` run in the first one (`ttf engine build`, the patches in `llama.cpp/`).

Options after `--` go to llama.cpp unchanged, for example `ttf chat MODEL -- -c 8192 -t 8`. Choose the backend with
`ttf engine build --backend cuda|cpu|metal|vulkan` (the default is `cuda` when `nvidia-smi` finds a GPU):
- `cuda` and `cpu` have fast kernels for the ternary types;
- on `metal` (Macs) and `vulkan` (AMD / Intel GPUs), llama.cpp runs the ternary matrices on the CPU.

## Quickstart: Qwen3.8-Flash-Next (MoE)

The Flash-Next model runs in the same `--q38` engine (llama.cpp's `qwen4exp` architecture):

```bash
hf download oguzeldereli/ttf-Qwen3.8-Flash-Next-GGUF --local-dir models/ttf-flash
ttf chat models/ttf-flash/ttf-Qwen3.8-Flash-Next-TQ1_P.gguf
```

The file is 40.4 GB, so most of the experts stay in host memory: plan for 64 GB of RAM. llama.cpp's `--fit` (on by
default) places what fits on the GPU.

Measured on an RTX 3090 Ti (24 GB) with 12 CPU threads, using `ttf chat MODEL -- -ncmoe 18 -t 12`:
- 29.7 tokens/s decode and 147 tokens/s prefill;
- with all experts on the CPU (`-ncmoe 99`): 22 tokens/s decode at 6.2 GB of VRAM.

Use the number of physical cores for `-t`; hyper-threads make decoding slower.

`-ncmoe 18` (18 layers' experts on the CPU) assumes a 24 GB card with nothing else on it. When you set `-ncmoe`,
llama.cpp's `--fit` switches off, so on a card that is partly in use, leave `-ncmoe` out and let `--fit` place the
experts. `ttf bench` puts all experts on the CPU unless you pass `-ncmoe` (llama-bench has no `--fit`).

**Building on older toolchains.** Extra CMake options go after `--` (or in `--cmake="..."`):
- `Error: unsupported instruction` while compiling: the assembler is older than the CPU. Build without
  `-march=native`: `ttf engine build -- -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON`.
- `undefined reference to cudaMemcpyAsync@libcudart.so...` at link time: point the linker at the CUDA libraries, e.g.
  `-- -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64 -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64`.

## Released models

| model | file | quality | made by |
|---|---|---|---|
| [ttf-Qwen3.8-27B-GGUF](https://huggingface.co/oguzeldereli/ttf-Qwen3.8-27B-GGUF) | 8.2 GB, TQ1_P (1.6 bits) + Q4_K embedding / head | GSM8K 0.895 (full precision 0.895), HumanEval 0.841 (0.933), IFEval 0.735 (0.787) | ttf-conversion + 72M tokens of TTF recovery training |
| [ttf-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/oguzeldereli/ttf-Qwen3.8-Flash-Next-GGUF) | 40.4 GB, TQ1_P (1.6 bits) | GSM8K 0.905 (FP8 0.905), HumanEval 0.848 (0.927), TriviaQA 0.538 (0.776) | ttf-conversion + 1000 steps of TTF recovery training |

GSM8K is scored on the first 200 problems, IFEval (strict) on the first 200 prompts. Knowledge recall (TriviaQA) is
where ternary costs the most. The model cards on Hugging Face have the details.

## Install for training

Training, conversion, fine-tuning and export need an NVIDIA GPU, PyTorch and the trainer:

```bash
pip install "ttf-ternary @ git+https://github.com/oguzeldereli/ttf-ternary" -e ".[train]"
# to convert Qwen models as well:
pip install "ttf-ternary[qwen] @ git+https://github.com/oguzeldereli/ttf-ternary" \
            "ttf-conversion @ git+https://github.com/oguzeldereli/ttf-conversion" -e ".[convert]"
```

Install from a clone (`pip install -e`): `ttf engine build` reads the patches from the repository.

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

`--base` is a local Hugging Face model directory (`hf download Qwen/Qwen3.5-0.8B-Base --local-dir Qwen3.5-0.8B-Base`).
For a conversion, run `ttf data` with `--tokenizer Qwen3.5-0.8B-Base` so the tokens match the base model.

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

Converting 0.8B models needs about 12 GB.

`ttf convert` on a 27B base needs a 24 GB card or larger and a cached teacher (`--kd-cache`, from ttf-ternary
`scripts/quant/teacher_topk.py`):
- Under 80 GB, it keeps the TTF state in host memory at accumulator rank 256, with the embedding and head frozen
  (measured at a 21.7 GiB peak).
- On 80 GB and larger cards, it uses the settings of our 27B run: rank 1536, the state on the GPU, and int-k embedding
  and head.

Running the result needs about 10 GB.

## Speed and exactness (laptop: RTX 4080 Laptop 12 GB, i9-14900HX)

| model | file | GPU decode | GPU prefill | CPU decode (8 threads) |
|---|---|---|---|---|
| Qwen3.8-27B ternary (training-free conversion, rank-1024 ternary adapters) | 9.2 GiB | 26.6 tok/s | 708 tok/s | 3.9 tok/s |
| same, `--wtype tq1_0` (1.69 bits) | 7.9 GiB | 18.2 tok/s | 512 tok/s | - |
| same, TQ2_0 with `--head q4_k` (+0.005 NLL) | 8.9 GiB | 27.4 tok/s | 689 tok/s | - |
| Qwen3.5-0.8B ternary (TQ2_0) | 0.61 GB | 256 tok/s | 12.9k tok/s | 49 tok/s |
| Qwen3.5-0.8B ternary (TQ1_0) | 0.58 GB | 208 tok/s | 11.1k tok/s | 45 tok/s |
| Qwen3.5-0.8B bf16 (reference) | 1.5 GB | 181 tok/s | 13.7k tok/s | 31 tok/s |

The released 27B is TQ1_P (8.2 GB), not in this table. Its TQ2_0 export (the third row's layout plus a rank-16 float
adapter on each linear) decodes at 37.2 tok/s on an RTX 4070 Ti SUPER.

The export is exact: next-token loss in llama.cpp matches the trainer's own forward. On 10 x 512 tokens of Wikipedia:

| model | trainer | llama.cpp |
|---|---|---|
| Qwen3.5-0.8B conversion | 2.5897 | 2.5898 (CUDA) / 2.5911 (CPU) |
| 340M scratch model | 2.3100 | 2.3107 |

The small difference comes from activation quantisation: llama.cpp quantises activations per block of 32 or 256
values, while training quantises them per token.

`ttf` looks for the engines in `~/.cache/ttf/llama.cpp` and `~/.cache/ttf/llama.cpp-q38` (where `ttf engine build` puts
them), or in `TTF_LLAMA_CPP` / `TTF_LLAMA_CPP_Q38`.

## GGUF format

- **Which `--wtype`**:
  - `tq2_0` on NVIDIA GPUs and CPUs (fastest);
  - `tq1_0` to save 15% of memory;
  - `q2_k` (2.63 bits) for AMD / Intel GPUs (Vulkan) and Macs (Metal). llama.cpp has no TQ kernels there yet, but Q2_K
    runs everywhere and holds ternary weights exactly too.
- **Ternary matrices** are stored as `TQ2_0` (2.06 bits per weight, the default), `TQ1_0` (1.69 bits) or `Q2_K`.
  - All three hold {-1, 0, +1}·d exactly, with the row scale folded into d.
  - Matrices whose width is not a multiple of 256 fall back to `Q8_0`, which is also exact.
  - The Flash-Next engine adds `TQ1_P`: 160 trits in 32 bytes (1.6 bits), no block scale. The row scale is a side
    tensor.
- **Embedding and head**: K-quants (Q6_K by default).
  - A Q4_K embedding costs nothing measurable; a Q4_K head costs +0.011 NLL at 0.8B.
  - Int-k codes (`--intk-emb`) export exactly as Q4_1 (d = s, m = lo, the codes as they are).
  - Q4_1 is 5 bits per weight, so 2- or 3-bit codes take more room in the file than in training. For the 27B, the int-k
    embedding and head are 1.6 GB as Q4_1, against 0.95 GB at their own bits.
- **TTF side tensors**, as F32 vectors per linear layer:
  - `<weight>.ttf_c`: input scale;
  - `.ttf_m`: centring, stored as -c·mu;
  - `.ttf_b`: output bias.
- **Adapters**: ternary low-rank adapters are stored as `.ttf_d*` / `.ttf_u*`, float ones as `.ttf_fd` / `.ttf_fu`.
- The patched llama.cpp applies all of these inside `build_lora_mm`, so every architecture llama.cpp supports gets
  them without per-model code.
- **CUDA** (patches 0002-0010):
  - matrix-vector kernels for `TQ2_0` and `TQ1_0`;
  - prefill for both through the Q8_0 MMQ kernels;
  - the side tensors fused into the activation quantiser of both (bit-identical; +10-20% tok/s);
  - one warp per row and two ints per thread at batch 1 (+16% at 27B).
- **Checks**: `test-backend-ops` passes 22 / 22: the `TQ1_0` / `TQ2_0` matrix products match the CPU.
- **Speed**: `TQ2_0` is the fastest; `TQ1_0` is 15% smaller and 31% slower.

Flash-Next GGUF files come from `python -m ttfhome.q38_export`. It takes a slim TTF checkpoint from ttf-ternary's
`ttf.q38` trainer and the PQ PLE table from ttf-conversion's `scripts/q38`.

## Tests

    pip install -e ".[train,test]" && pytest -q tests

`tests/test_export.py` checks that every ternary matrix exports exactly to TQ2_0 / TQ1_0 / Q2_K, and int-k to Q4_1. It
needs the engine's `gguf-py`, so run `ttf engine build` first.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE.md): free for personal, research and other noncommercial use; commercial use
needs a separate licence. llama.cpp itself is MIT-licensed; the patches in `llama.cpp/` modify it. The released model
weights keep the licence of their base models (Apache-2.0).
