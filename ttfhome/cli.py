"""ttf: train, convert, fine-tune and run ternary (TTF) language models on a home computer.

  ttf doctor                                   what this machine has (GPU, memory, the engine, the trainer)
  ttf engine build [--backend cuda|cpu] [--llama]
                                               build the engine (ttf-home on our ggml); --llama: the old llama.cpp
                                               build for .gguf exports (chat / serve / score until the engine has them)
  ttf data INPUT... --out DATA [--tokenizer llama2|DIR] [--val 0.005]
                                               text / Markdown / JSONL (text or chat messages) -> token files
  ttf train --data DATA --out RUN [--size auto|tiny|small|medium|large] [--hours H] [TTF knobs]
                                               a ternary model from scratch
  ttf convert --base HF_DIR --data DATA --out RUN [--steps 1000]
                                               a Qwen3.5 / 3.8 model -> ternary, then TTF recovery training
  ttf finetune --ckpt CKPT --data DATA --out RUN [--steps 300]
                                               keep training a ternary model on your own data
  ttf export CKPT OUT.gguf [--base HF_DIR] [--wtype tq2_0|tq1_0|q2_k]
  ttf run MODEL.ttf "TEXT" [-n N | -o out.wav]  any .ttf model: a text model continues the text, a speech model
                                               speaks it (more options: ttf run MODEL.ttf --help)
  ttf bench MODEL.ttf                          prefill / decode speed and bytes per token (speech: real-time factor)
  .gguf exports (llama.cpp, --llama build): ttf generate | chat | serve | bench | score MODEL.gguf ...

TTF knobs (train / convert / finetune; the defaults are the measured best):
  --theta 16 (firing threshold, in steps)   --rank R (accumulator rank)   --lr / --min-lr   --steps / --warmup
  --seq-len / --batch / --accum            any other trainer option after `--` (python -m ttf.train --help)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
ENGINE_BASE = "0ef6f06d5"                    # the llama.cpp commit the patches apply to
ENGINE_URL = "https://github.com/ggml-org/llama.cpp"


# ------------------------------------------------------------------------------------------------ environment

def home_bin():
    """our engine's binary (ttf-home): $TTF_ENGINE, else this checkout's engine build, the GPU build first"""
    if os.environ.get("TTF_ENGINE"):
        return Path(os.environ["TTF_ENGINE"]).expanduser()
    for b in ("build-cuda", "build-cpu", "build"):
        p = HOME / "engine" / b / "ttf-home"
        if p.exists():
            return p
    raise SystemExit(f"ttf-home not found under {HOME / 'engine'}: run `ttf engine build` (or set TTF_ENGINE)")


def engine_dir():
    return Path(os.environ.get("TTF_LLAMA_CPP", "~/.cache/ttf/llama.cpp")).expanduser()


def engine_bin(name):
    """a llama.cpp binary: the GPU build first"""
    d = engine_dir()
    for b in ("build-cuda", "build-metal", "build-vulkan", "build-cpu", "build"):
        p = d / b / "bin" / name
        if p.exists():
            return p
    raise SystemExit(f"{name} not found under {d}: run `ttf engine build` (or set TTF_LLAMA_CPP)")


def gpu_info():
    """[(name, total GiB)] of the CUDA devices (empty without torch / CUDA)"""
    try:
        import torch
        if torch.cuda.is_available():
            return [(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 2**30)
                    for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode:
        raise SystemExit(r.returncode)
    return r


def cmd_doctor(a):
    import platform
    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}, "
          f"{os.cpu_count()} CPU threads")
    try:
        import torch
        print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    except ImportError:
        print("torch: not installed (needed to train; running models needs only the engine)")
    for n, g in gpu_info():
        print(f"GPU: {n}, {g:.1f} GiB -> train size `{_auto_size(g)}`, runs ternary models up to "
              f"~{_max_params_b(g):.0f}B parameters fully on the GPU")
    try:
        import triton
        print(f"triton {triton.__version__}")
    except ImportError:
        print("triton: not installed (the trainer's kernels need it)")
    try:
        import ttf
        print(f"ttf-core: {Path(ttf.__file__).parent}")
    except ImportError:
        print("ttf-core: not importable (pip install -e ttf-core)")
    try:
        import ttfconv
        print(f"ttf-conversion: {Path(ttfconv.__file__).parent}")
    except ImportError:
        print("ttf-conversion: not importable (needed by `ttf convert`)")
    bins = [b for b in ("build-cuda", "build-cpu", "build") if (HOME / "engine" / b / "ttf-home").exists()]
    print(f"engine: {HOME / 'engine'} ({', '.join(bins) or 'not built: ttf engine build'})")
    d = engine_dir()
    bins = [b for b in ("build-cuda", "build-metal", "build-vulkan", "build-cpu", "build") if (d / b / "bin" / "llama-cli").exists()]
    if bins:
        print(f"llama.cpp (for .gguf exports): {d} ({', '.join(bins)})")


def _max_params_b(gib):
    # measured: the 27B conversion (adapters, Q4_K embedding, Q6_K head) is 9.23 GiB; ~1.5 GiB for cache and buffers
    return max(0.0, (gib - 1.5) * 27 / 9.23)


# ---------------------------------------------------------------------------------------------------- engine

def cmd_engine(a):
    if a.action == "build" and not a.llama:
        if a.backend not in ("cuda", "cpu"):
            raise SystemExit(f"backend {a.backend}: the engine has CUDA and CPU kernels only (yet); --llama for llama.cpp")
        src = HOME / "engine"
        b = src / f"build-{a.backend}"
        flags = ["-DGGML_CUDA=ON"] if a.backend == "cuda" else ["-DGGML_CUDA=OFF"]
        run(["cmake", "-S", src, "-B", b, "-DCMAKE_BUILD_TYPE=Release", *flags, *a.cmake])
        run(["cmake", "--build", b, "-j", str(os.cpu_count()), "--target", "ttf-home"])
        print(f"engine built: {b / 'ttf-home'}")
        return
    d = engine_dir()
    if a.action == "build":
        if not (d / ".git").exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", ENGINE_URL, d])
            run(["git", "-C", d, "checkout", "-B", "ttf", ENGINE_BASE])
            patches = sorted((HOME / "llama.cpp").glob("*.patch"))
            run(["git", "-C", d, "-c", "user.name=ttf", "-c", "user.email=ttf@localhost", "am", *patches])
        flags = {"cuda": ["-DGGML_CUDA=ON"], "cpu": ["-DGGML_CUDA=OFF"], "metal": ["-DGGML_METAL=ON"],
                 "vulkan": ["-DGGML_VULKAN=ON"]}[a.backend]
        b = d / f"build-{a.backend}"
        run(["cmake", "-S", d, "-B", b, "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", *flags, *a.cmake])
        run(["cmake", "--build", b, "-j", str(os.cpu_count()), "--target", "llama-cli", "llama-server",
             "llama-bench", "llama-completion", "llama-perplexity"])
        print(f"engine built: {b / 'bin'}")


# ------------------------------------------------------------------------------------------------------ data

def _tokenizer(spec):
    from tokenizers import Tokenizer
    from .gguf_export import LLAMA2_TOKENIZER
    d = Path(LLAMA2_TOKENIZER if spec == "llama2" else spec).expanduser()
    if spec == "llama2" and not (d / "tokenizer.json").exists():    # not in the HF cache yet: fetch it (~2 MB)
        from huggingface_hub import snapshot_download
        d = Path(snapshot_download("hf-internal-testing/llama-tokenizer", revision=d.name))
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    cfg = json.loads((d / "tokenizer_config.json").read_text()) if (d / "tokenizer_config.json").exists() else {}
    eos = cfg.get("eos_token")
    eos = eos.get("content") if isinstance(eos, dict) else eos
    return tok, d, (tok.token_to_id(eos) if eos else None)


def _docs(paths, tok_dir):
    """the documents of the inputs: .txt / .md files (one document each), .jsonl with a "text" field, or chat
    "messages" (rendered with the tokenizer's chat template)"""
    chat = None
    for p in paths:
        p = Path(p)
        files = sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
        for f in files:
            if f.suffix == ".jsonl":
                for line in f.open():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if "text" in r:
                        yield r["text"]
                    elif "messages" in r:
                        if chat is None:
                            from transformers import AutoTokenizer
                            chat = AutoTokenizer.from_pretrained(tok_dir)
                        yield chat.apply_chat_template(r["messages"], tokenize=False)
            elif f.suffix in (".txt", ".md", ".text", ""):
                yield f.read_text(errors="replace")


def cmd_data(a):
    import numpy as np
    tok, d, eos = _tokenizer(a.tokenizer)
    V = tok.get_vocab_size()
    dt = np.uint16 if V <= 65536 else np.uint32
    ids = []
    n_docs = 0
    for doc in _docs(a.inputs, d):
        ids.extend(tok.encode(doc, add_special_tokens=False).ids)
        if eos is not None:
            ids.append(eos)
        n_docs += 1
    arr = np.asarray(ids, dtype=dt)
    nv = int(len(arr) * a.val)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr[:len(arr) - nv].tofile(f"{out}_train.bin")
    arr[len(arr) - nv:].tofile(f"{out}_val.bin")
    json.dump({"tokenizer": str(d), "vocab": V, "dtype": "u16" if dt == np.uint16 else "u32", "docs": n_docs,
               "train_tokens": len(arr) - nv, "val_tokens": nv}, open(f"{out}.json", "w"), indent=1)
    print(f"{n_docs} documents, {len(arr) - nv:,} train / {nv:,} val tokens -> {out}_train.bin, {out}_val.bin")


def _data_meta(prefix):
    p = Path(f"{prefix}.json")
    return json.loads(p.read_text()) if p.exists() else {}


def _data_args(data, dtype=None):
    """trainer options for DATA: a `ttf data` prefix (DATA_train.bin, DATA_val.bin, DATA.json) or a token .bin file
    (with DATA's dtype given by --data-dtype after `--`, or u32 for conversions)"""
    if data.endswith(".bin"):
        return ["--data", data] + (["--data-dtype", dtype] if dtype else [])
    meta = _data_meta(data)
    if not meta:
        raise SystemExit(f"{data}.json not found: make the data with `ttf data`, or pass a .bin file")
    return ["--data", f"{data}_train.bin", "--val", f"{data}_val.bin", "--data-dtype", meta["dtype"]]


# ----------------------------------------------------------------------------------------------------- train

# (preset, micro-batch, seq) per size; accum fills 32k tokens per step (the measured best: 1 x 16 x 2048)
SIZES = {"tiny": ("s15", 16, 2048), "small": ("small", 4, 2048), "medium": ("d1024_l24", 2, 2048),
         "large": ("d1536_l24", 1, 2048)}


def _auto_size(gib):
    return "large" if gib >= 22 else "medium" if gib >= 11 else "small" if gib >= 7 else "tiny"


def _knobs(a):
    k = []
    for opt, flag in (("theta", "--ttf-theta"), ("rank", "--ttf-rank"), ("lr", "--lr"), ("min_lr", "--min-lr"),
                      ("steps", "--steps"), ("warmup", "--warmup"), ("seq_len", "--seq-len"),
                      ("batch", "--batch-size"), ("accum", "--accum")):
        v = getattr(a, opt, None)
        if v is not None:
            k += [flag, v]
    return k


def _trainer(args, extra):
    run([sys.executable, "-m", "ttf.train", *args, *extra])


def _mem_cap():
    g = gpu_info()
    if not g:
        raise SystemExit("training needs a CUDA GPU (ttf doctor)")
    return g[0][1]


def cmd_train(a, extra):
    gib = _mem_cap()
    size = _auto_size(gib) if a.size == "auto" else a.size
    preset, mb, seq = SIZES[size]
    tokens_per_step = 32768
    seq = a.seq_len or seq
    mb = a.batch or mb
    accum = a.accum or max(1, tokens_per_step // (mb * seq))
    args = ["--recipe", "scratch-110m", "--preset", preset, *_data_args(a.data), "--out-dir", a.out, "--seq-len", seq, "--batch-size", mb, "--accum", accum,
            "--mem-cap-gib", f"{gib * 0.94:.1f}", "--resume"]
    if a.hours and not a.steps:
        a.steps = _steps_for_hours(a, args, extra)
    print(f"training a `{size}` model ({preset}) on {gib:.1f} GiB: {mb} x {accum} x {seq} tokens per step")
    _trainer(args + _knobs(a), extra)


def _steps_for_hours(a, args, extra):
    """the step count that fits the budget, from a timed 6-step probe of this exact setting"""
    import tempfile, time
    with tempfile.TemporaryDirectory() as d:
        probe = [x if x != a.out else d for x in args if x != "--resume"]
        t0 = time.time()
        run([sys.executable, "-m", "ttf.train", *probe, "--steps", 6, "--warmup", 1, "--eval-interval", 10**6,
             "--save-secs", 10**6, *extra], stdout=subprocess.DEVNULL)
        per = (time.time() - t0 - 30) / 6                            # ~30 s of start-up (data, compile)
    n = max(50, int(a.hours * 3600 / max(per, 0.05)))
    print(f"~{per:.2f} s per step: {n} steps in {a.hours} h")
    return n


def _conv_recipe(base, gib, convert=True):
    """the conversion recipe for the base model and this GPU: qwen-27b.toml for the 27B (its settings are those of the
    96 GB-card run; under 80 GiB the TTF state goes to host memory at rank 256, and a new conversion keeps the
    embedding / head frozen: the setting measured at a 21.7 GiB peak), else qwen-0.8b.toml"""
    hc = json.loads((Path(base) / "config.json").read_text())
    big = (hc.get("text_config") or hc).get("hidden_size", 0) > 3000
    recipe = Path(_conv_root()) / "recipes" / ("qwen-27b.toml" if big else "qwen-0.8b.toml")
    small = ["--ttf-store", "fp12s", "--ttf-offload", "async", "--ttf-rank", 256] if big and gib < 80 else []
    if small and convert:
        small += ["--intk-emb", ""]          # (a fine-tune keeps the structure its checkpoint has)
    return big, ["--recipe", recipe, *small]


def cmd_convert(a, extra):
    gib = _mem_cap()
    big, rec = _conv_recipe(a.base, gib)
    if big and not (a.teacher or a.kd_cache):
        raise SystemExit("a 27B conversion needs its distillation targets: --kd-cache DIR (ttf-core "
                         "scripts/quant/teacher_topk.py) or --teacher M where the bf16 teacher (~54 GB) fits beside it")
    kd = ["--kd-cache", a.kd_cache] if a.kd_cache else ["--teacher", a.teacher or a.base]
    args = [*rec, "--init-hf", a.base, *kd, *_data_args(a.data, "u32"), "--out-dir", a.out,
            "--ext", "ttfconv.fold", *([] if big else ["--ext", "ttfconv.cd:sweeps=1"]),
            "--ext", f"ttfconv.adapters:rank={a.adapter_rank},alt=1,prop=1" + (",svd=rand" if big else ""),
            "--mem-cap-gib", f"{gib * 0.94:.1f}", "--resume"]
    _trainer(args + _knobs(a), extra)


def _conv_root():
    import ttfconv
    return Path(ttfconv.__file__).resolve().parents[1]


def cmd_finetune(a, extra):
    import torch
    gib = _mem_cap()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    ca = ck.get("args", {}) or {}
    ca = ca if isinstance(ca, dict) else vars(ca)
    args = ["--init-ckpt", a.ckpt, *_data_args(a.data, "u32" if ca.get("init_hf") else None), "--out-dir", a.out,
            "--mem-cap-gib", f"{gib * 0.94:.1f}", "--resume", "--steps", a.steps or 300]
    if ca.get("init_hf"):
        base = a.base or ca["init_hf"]
        if not Path(base).exists():
            raise SystemExit(f"the checkpoint was converted from {ca['init_hf']}: pass --base (its config + tokenizer)")
        args += [*_conv_recipe(base, gib, convert=False)[1], "--init-hf", base]
        for e in ca.get("ext", []):
            args += ["--ext", e]
        if a.teacher:
            args += ["--teacher", a.teacher]
        else:
            args += ["--kd-weight", 0]
    else:
        args += ["--recipe", "scratch-110m", "--preset", ca.get("preset", "small")]
    a.steps = None
    _trainer(args + _knobs(a), extra)


# ------------------------------------------------------------------------------------------------------ run

def cmd_export(a, extra):
    from .gguf_export import export
    import torch
    with torch.inference_mode():
        export(a.ckpt, a.base, a.out, a.wtype, a.emb, a.head)


def _ngl():
    return ["-ngl", "999"] if gpu_info() or sys.platform == "darwin" else []


def cmd_chat(a, extra):
    run([engine_bin("llama-cli"), "-m", a.model, *_ngl(), "-cnv", *extra])


def cmd_run(a, extra):
    r = subprocess.run([str(home_bin()), a.model, *a.args, *extra])
    raise SystemExit(r.returncode)


def cmd_generate(a, extra):
    if a.model.endswith(".ttf"):
        r = subprocess.run([str(home_bin()), a.model, a.prompt, "-n", str(a.n), *extra])
        raise SystemExit(r.returncode)
    run([engine_bin("llama-completion"), "-m", a.model, *_ngl(), "-p", a.prompt, "-n", a.n, "-no-cnv", *extra])


def cmd_serve(a, extra):
    run([engine_bin("llama-server"), "-m", a.model, *_ngl(), "--port", a.port, "--host", a.host, *extra])


def cmd_bench(a, extra):
    if a.model.endswith(".ttf"):
        r = subprocess.run([str(home_bin()), a.model, "--bench", *extra])
        raise SystemExit(r.returncode)
    run([engine_bin("llama-bench"), "-m", a.model, *_ngl(), *extra])


def cmd_score(a, extra):
    tool = HOME / "tools" / "ttf_score"
    if not tool.exists():
        d = engine_dir()
        b = next(p for p in (d / "build-cuda", d / "build-cpu", d / "build") if (p / "bin").exists())
        run(["g++", "-O2", "-std=c++17", HOME / "tools" / "ttf_score.cpp", f"-I{d}/include", f"-I{d}/ggml/include",
             f"-L{b}/bin", "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{b}/bin", "-o", tool])
    meta = _data_meta(str(a.data).removesuffix("_val.bin").removesuffix(".bin"))
    u16 = ["-u16"] if meta.get("dtype") == "u16" or a.u16 else []
    run([tool, a.model, a.data, "-n", a.chunks, *(["-ngl", "99"] if gpu_info() else []), *u16, *extra])


# ------------------------------------------------------------------------------------------------------ main

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="ttf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("doctor")
    p = sp.add_parser("engine"); p.add_argument("action", choices=["build"])
    p.add_argument("--backend", default="cuda" if gpu_info() else ("metal" if sys.platform == "darwin" else "cpu"),
                   choices=["cuda", "cpu", "metal", "vulkan"])
    p.add_argument("--cmake", nargs="*", default=[])
    p.add_argument("--llama", action="store_true", help="build the old llama.cpp engine (for .gguf exports)")
    p = sp.add_parser("data"); p.add_argument("inputs", nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="llama2", help="llama2 (scratch models) or the base model's dir (conversions)")
    p.add_argument("--val", type=float, default=0.005)

    def knobs(p):
        for o in ("--theta", "--lr", "--min-lr"):
            p.add_argument(o, type=float)
        for o in ("--rank", "--steps", "--warmup", "--seq-len", "--batch", "--accum"):
            p.add_argument(o, type=int)
    p = sp.add_parser("train"); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--size", default="auto", choices=["auto", *SIZES]); p.add_argument("--hours", type=float)
    knobs(p)
    p = sp.add_parser("convert"); p.add_argument("--base", required=True); p.add_argument("--data", required=True)
    p.add_argument("--out", required=True); p.add_argument("--teacher"); p.add_argument("--adapter-rank", default="0.2")
    p.add_argument("--kd-cache", help="a cached teacher top-k (ttf-core scripts/quant/teacher_topk.py) instead of --teacher")
    knobs(p)
    p = sp.add_parser("finetune"); p.add_argument("--ckpt", required=True); p.add_argument("--data", required=True)
    p.add_argument("--out", required=True); p.add_argument("--base"); p.add_argument("--teacher")
    knobs(p)
    p = sp.add_parser("export"); p.add_argument("ckpt"); p.add_argument("out")
    p.add_argument("--base", default=None, help="the base model's dir for a conversion (config + tokenizer)")
    p.add_argument("--wtype", default="tq2_0", choices=["tq2_0", "tq1_0", "q2_k"],
                   help="tq2_0: NVIDIA / CPU (fastest); tq1_0: 15%% smaller; q2_k: AMD / Intel (Vulkan), Mac (Metal)")
    p.add_argument("--emb", default=None, help="q6_k (float) / q4_1 (int-k codes, exact: the default for them) / q4_k ...")
    p.add_argument("--head", default=None)
    p = sp.add_parser("run", add_help=False); p.add_argument("model"); p.add_argument("args", nargs=argparse.REMAINDER)
    for c in ("chat", "bench"):
        p = sp.add_parser(c); p.add_argument("model")
    p = sp.add_parser("generate"); p.add_argument("model"); p.add_argument("prompt"); p.add_argument("-n", default="128")
    p = sp.add_parser("serve"); p.add_argument("model"); p.add_argument("--port", default="8080")
    p.add_argument("--host", default="127.0.0.1")
    p = sp.add_parser("score"); p.add_argument("model"); p.add_argument("data"); p.add_argument("--chunks", default="40")
    p.add_argument("--u16", action="store_true")
    a = ap.parse_args(argv)
    f = globals()[f"cmd_{a.cmd}"]
    return f(a) if a.cmd in ("doctor", "engine", "data") else f(a, extra)


if __name__ == "__main__":
    main()
