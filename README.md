# ttf-home

Ternary language and speech models on a home computer:
- run ternary models fast on a consumer GPU or CPU, in our own runtime (`ttf-home`);
- train a model from scratch;
- convert a Qwen3.5 / 3.8 model to ternary and keep training it;
- fine-tune on your own text.

One command line, `ttf`, does all of it.

Training uses TTF ([ttf-ternary](https://github.com/oguzeldereli/ttf-ternary)): the weights stay in {-1, 0, +1} the
whole time, with a low-rank accumulator in place of master weights and Adam states, so training needs a fraction of the
usual memory.

Inference has two paths:
- **`ttf-home`, our own runtime** (`engine/`): one program on ggml, with its own ternary kernels for CUDA and CPU. It
  runs `.ttf` files: text models (continuation, chat, an OpenAI-style server with a chat page) and speech models
  (VoxCPM2 text-to-speech).
- **llama.cpp with the TTF patches** (`llama.cpp/`): the released Qwen3.8 GGUF files, and `.gguf` exports.

| repository | what it does |
|---|---|
| [ttf-ternary](https://github.com/oguzeldereli/ttf-ternary) | the trainer (TTF), the `.ttf` exporter, and the paper |
| [ttf-conversion](https://github.com/oguzeldereli/ttf-conversion) | pretrained Hugging Face model to ternary checkpoint |
| **ttf-home** (this one) | the runtime; data, train, convert, fine-tune, export, chat / serve |

## The runtime: `ttf-home`

You need `git`, `cmake` and a C++17 compiler; for an NVIDIA GPU, also the CUDA toolkit (`nvcc`). No Python or PyTorch is
needed to run models.

```bash
git clone https://github.com/oguzeldereli/ttf-home && cd ttf-home
cmake -S engine -B engine/build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # -DGGML_CUDA=OFF for CPU only
cmake --build engine/build-cuda -j --target ttf-home
# or, with the Python CLI installed (pip install -e .): ttf engine build
```

```bash
E=engine/build-cuda/ttf-home
$E model.ttf "Once upon a time" -n 128            # a text model continues the text
$E model.ttf --chat                               # chat in the terminal (--no-think / --think, --system "...")
$E model.ttf --serve --port 8080                  # chat page at http://127.0.0.1:8080/ , OpenAI-style API at /v1
$E model.ttf --bench                              # prefill / decode speed, bytes per token
$E speech.ttf "Hello from a ternary model." -o hi.wav          # a speech model speaks the text
$E speech.ttf --list jobs.tsv                     # many: one out.wav<TAB>text[<TAB>seed] per line
```

`--device cpu|cuda` picks the device (CUDA when the build has it), `-v` prints timings, `--help` lists the rest
(`--temp`, `--top-k`, `--seed`, `--ctx`, `--threads`, ...). The same commands through the CLI: `ttf run MODEL.ttf ...`,
`ttf chat MODEL.ttf`, `ttf serve MODEL.ttf`, `ttf bench MODEL.ttf`.

The server's page streams the reply, shows the model's reasoning in a fold above the answer, has a thinking switch,
temperature / top-k / max tokens / system prompt, and a Stop button. The API is `POST /v1/chat/completions` (streaming
too; the last chunk carries `usage` and llama-server style `timings`) and `GET /v1/models`. One request runs at a time;
a follow-up turn reuses the state of the conversation so far.

**Models.** A `.ttf` file holds the ternary weights (1.6 bits per weight), their scales, adapters and the model's
graphs; the program reads the architecture from the file. Supported now: Qwen3.5 / 3.8 dense (gated attention +
Gated DeltaNet), VoxCPM2 (text-to-speech), BitNet b1.58 / Falcon3-1.58bit (Llama architecture) and the TTF
models trained from scratch. Get one:
- [ttf-VoxCPM2](https://huggingface.co/oguzeldereli/ttf-VoxCPM2): ternary VoxCPM2, multilingual text-to-speech, 1.04 GB.
- Export your own with ttf-ternary: `python -m ttf.export.qwen35 HF_DIR CKPT OUT.ttf` (a converted or TTF-trained
  Qwen3.5 checkpoint), `python -m ttf.export.bitnet HF_DIR OUT.ttf` (Microsoft BitNet b1.58,
  tiiuae/Falcon3-*-1.58bit).

**Speed.** Decode, `--bench --tg 256`, tokens per second:

| model | RTX 4070 Ti SUPER | RTX 4080 Laptop (12 GB) |
|---|---|---|
| Qwen3.8-27B ternary, with float adapters | 71.4 | 39.5 |
| the same without adapters | 77.6 (llama.cpp TQ1_P: 79.6) | - |
| Falcon3-1B-1.58bit (int-k head) | 698 (llama.cpp TQ2_0: 642) | 427 |
| Falcon3-3B-1.58bit | 408 (llama.cpp TQ2_0: 423) | - |

- The 27B's ternary matrix-vector kernel reads 544-583 GB/s on the 4070 Ti SUPER (llama.cpp's TQ1_P kernel 454-535 on
  the same shapes).
- Decode keeps the GPU busy between tokens: the next token's step is queued on the GPU before the host reads the
  current one (greedy: the argmax stays on the device; sampling: only the top k come back). The laptop GPU's power cap
  limits what that gains there.
- VoxCPM2 on the laptop: 3 s of speech in 1.2 s (real-time factor 0.39).
- CPU (i9-14900HX, 8 threads): BitNet b1.58 2B4T 38.5-40 tok/s decode (tuned llama.cpp 41.4).

**Exactness.** Every op has a reference: the `.ttf` exporter runs the file's graphs in PyTorch and the runtime's per-op
dump (`TTF_DUMP`) is checked against it. VoxCPM2 through the runtime scores the same as through PyTorch on FLEURS
(30 languages x 8 texts, Whisper-large-v3): median character error 0.080-0.086 over three runtime passes vs 0.074,
within the spread between two seeds of the runtime itself.

## llama.cpp: the released Qwen3.8 GGUF models

The released Qwen3.8 models are GGUF files for llama.cpp with the TTF patches. You need `git`, `cmake`, a C++
compiler and Python 3.11+; for an NVIDIA GPU, also the CUDA toolkit (`nvcc`). Running models needs no PyTorch.

```bash
git clone https://github.com/oguzeldereli/ttf-home && cd ttf-home
pip install -U pip "setuptools>=68"   # an editable install needs setuptools 64+ (PEP 660)
pip install -e . huggingface_hub
ttf doctor                       # what this machine has and can run
ttf engine build --q38           # llama.cpp + the TTF patches, into ~/.cache/ttf/llama.cpp-q38 (CUDA when a GPU is found)

hf download oguzeldereli/ttf-Qwen3.8-27B-GGUF ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf --local-dir models/ttf-27b
ttf chat models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
ttf serve models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf --port 8080     # OpenAI-compatible API at http://127.0.0.1:8080/v1
```

For a coding agent (opencode, Cline, ...), set the context length and give the agent the same number:
`ttf serve MODEL --ctx 65536 -fa on -ctk q8_0 -ctv q8_0`. Without `--ctx`, llama.cpp starts from the model's trained
context (262k) and shrinks it to what fits in free VRAM (about 47k on a 12 GB GPU) without saying so; an agent that
reads 262k from the model then sends more than the server holds and compacts in a loop. The real value is
`n_ctx_slot` in the server log, or `curl http://127.0.0.1:8080/props`. `-ctk q8_0 -ctv q8_0` halves the KV cache.

The whole model, 8.2 GB (1.6 bits per ternary weight), fits on one 12 GB NVIDIA GPU. Without a GPU it runs on the CPU,
slower. Decode on an RTX 3090 Ti (`llama-bench -p 0 -n 128`, engine with `llama.cpp/q38/` patches 0010-0012):

| file | decode | before 0010-0012 |
|---|---|---|
| `ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf` (8.2 GB) | 34.8 tok/s | 29.4 tok/s |
| `ttf-Qwen3.8-27B-noadapters-TQ1_P.gguf` (6.74 GB) | 73.0 tok/s | 43.8 tok/s |
| the same weights with K padded to a multiple of 160 (6.67 GB) | 101.7 tok/s | - |

TQ1_P packs 160 trits per 32 bytes, so a matrix whose width is not a multiple of 160 (the 27B's 17408-wide FFN down
and 6144-wide output projections) was stored in a slower type; `ttf export --wtype tq1_p` now pads those with zero
columns and the engine pads the input, which gives the same product with every matrix in TQ1_P. The speed comes from
patches 0010-0012: a TQ1_P matrix-vector kernel, the TTF scale / centring / bias / adapter fused around it, and fused
DeltaNet kernels (gates, q/k norm, output norm, conv state). Each fusion has an off switch for debugging:
`GGML_CUDA_TTF_LINEAR_OFF=1`, `GGML_CUDA_DISABLE_GDN_GATES=1`, `GGML_CUDA_DISABLE_GDN_L2=1`,
`GGML_CUDA_DISABLE_GDN_OUT_NORM=1`, `GGML_CUDA_DISABLE_CONV_STATE=1`.

There are two engines. Both released models use TQ1_P weights (160 trits in 32 bytes), which only the second one,
`--q38` (the patches in `llama.cpp/q38/`), has; `ttf` picks the engine from the GGUF file. Models you export yourself
with the default `--wtype tq2_0` run in the first one (`ttf engine build`, the patches in `llama.cpp/`).

The 27B repo also has a 6.74 GB variant without ternary adapters, `ttf-Qwen3.8-27B-noadapters-TQ1_P.gguf`. It needs
`llama.cpp/q38/` patch 0009 (an optional bias on the DeltaNet beta projection). `ttf engine build` applies the patches
an existing engine does not have yet and rebuilds, so after `git pull` in ttf-home, run `ttf engine build --q38` again
(an engine without patch 0009 stops with "wrong number of tensors" on that file).

Options `ttf` does not know go to llama.cpp unchanged, for example `ttf chat MODEL -t 8 -fa on` (or after `--`). Choose the backend with
`ttf engine build --backend cuda|cpu|metal|vulkan` (the default is `cuda` when an NVIDIA GPU is found; the CUDA build needs the CUDA toolkit, `nvcc`, not only the driver; `ttf engine build` finds it on PATH, in CUDA_HOME / CUDA_PATH or the newest /usr/local/cuda-X.Y (CUDA 12 and 13 both work); elsewhere, add `--cmake="-DCMAKE_CUDA_COMPILER=/path/to/bin/nvcc"`):
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

Patches 0010-0012 change little here: with the experts on the CPU, the CPU sets the speed (`-ncmoe 20`: 30.7 vs 29.9
tokens/s; `-ncmoe 99`: 22.9 tokens/s).

`-ncmoe 18` (18 layers' experts on the CPU) assumes a 24 GB card with nothing else on it. When you set `-ncmoe`,
llama.cpp's `--fit` switches off, so on a card that is partly in use, leave `-ncmoe` out and let `--fit` place the
experts. `ttf bench` puts all experts on the CPU unless you pass `-ncmoe` (llama-bench has no `--fit`).

**Building on older toolchains.** Extra CMake options go after `--` (or in `--cmake="..."`):
- `Error: unsupported instruction` while compiling: the assembler is older than the CPU. Build without
  `-march=native`: `ttf engine build -- -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON`.
- `undefined reference to cudaMemcpyAsync@libcudart.so...` at link time: point the linker at the CUDA libraries, e.g.
  `-- -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64 -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64`.

## Released models

GGUF (llama.cpp, `ttf engine build --q38`):

| model | file | quality | made by |
|---|---|---|---|
| [ttf-Qwen3.8-27B-GGUF](https://huggingface.co/oguzeldereli/ttf-Qwen3.8-27B-GGUF) | 8.2 GB, TQ1_P (1.6 bits) + Q4_K embedding / head | GSM8K 0.895 (full precision 0.895), HumanEval 0.841 (0.933), IFEval 0.735 (0.787) | ttf-conversion + 72M tokens of TTF recovery training |
| [ttf-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/oguzeldereli/ttf-Qwen3.8-Flash-Next-GGUF) | 40.4 GB, TQ1_P (1.6 bits) | GSM8K 0.905 (FP8 0.905), HumanEval 0.848 (0.927), TriviaQA 0.538 (0.776) | ttf-conversion + 1000 steps of TTF recovery training |

`.ttf` (our runtime):

| model | file | quality | made by |
|---|---|---|---|
| [ttf-VoxCPM2](https://huggingface.co/oguzeldereli/ttf-VoxCPM2) | 1.04 GB, ternary (1.6 bits) + bf16 | English WER 3.22% (full precision 2.56%), speaker similarity 0.951 (0.959), Chinese CER 5.76% (3.24%) | ttf-conversion + 3000 steps of multilingual TTF recovery training |

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

Install from a clone (`pip install -e`): `ttf engine build` reads the patches from the repository. If pip says the
build backend "is missing the 'build_editable' hook", the build ran without isolation (`--no-build-isolation`,
`PIP_NO_BUILD_ISOLATION`, or a pip config) on a setuptools older than 64: `pip install -U "setuptools>=68"` and rerun.

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

    python -m ttf.export.qwen35 Qwen3.5-0.8B-Base runs/q08-mine/ckpt.pt q08-mine.ttf    # or a .ttf for our runtime
    ttf chat q08-mine.ttf ; ttf serve q08-mine.ttf ; ttf bench q08-mine.ttf

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

## llama.cpp: speed and exactness of GGUF exports (laptop: RTX 4080 Laptop 12 GB, i9-14900HX)

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
