"""GSM8K / HumanEval generations from a llama-server, in the q38_gen_kv harness's format (same prompts, chat template
with enable_thinking=False, greedy, max 512 new tokens), so score_bench.py scores both alike.

  python q38_bench_server.py --url http://127.0.0.1:8080 --tok HF_DIR --ref GEN_DIR --out OUT_DIR \
      [--tasks gsm8k:200,humaneval:164] [--par 4]
GEN_DIR: a q38_gen_kv output dir (its jsonl files give the items and prompts, in their order)."""
import argparse, json, os, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(url, body):
    req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--tok", required=True)
    ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="gsm8k:200,humaneval:164"); ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=512)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    os.makedirs(a.out, exist_ok=True)
    for spec in a.tasks.split(","):
        task, lim = spec.split(":")
        items = [json.loads(l) for l in open(os.path.join(a.ref, f"{task}.jsonl"))][:int(lim)]
        fn = os.path.join(a.out, f"{task}.jsonl")
        done = {json.loads(l)["id"] for l in open(fn)} if os.path.exists(fn) else set()
        todo = [it for it in items if it["id"] not in done]
        print(f"{task}: {len(todo)} to do ({len(done)} done)", flush=True)

        def one(it):
            ids = tok(tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)).input_ids
            t = time.time()
            r = post(a.url, {"prompt": ids, "n_predict": a.max_new, "temperature": 0.0, "top_k": 1, "samplers": ["top_k"],
                             "cache_prompt": False, "return_tokens": True})
            gi = r.get("tokens") or tok(r["content"], add_special_tokens=False).input_ids
            return it, gi, r, time.time() - t
        t0 = time.time(); n_tok = 0
        with ThreadPoolExecutor(a.par) as ex, open(fn, "a") as fo:
            for k, (it, gi, r, dt) in enumerate(ex.map(one, todo)):
                fo.write(json.dumps({"id": it["id"], "prompt": it["prompt"], "output": tok.decode(gi, skip_special_tokens=True),
                                     "gen_tokens": len(gi), "timings": r.get("timings")}) + "\n"); fo.flush()
                n_tok += len(gi)
                if (k + 1) % 20 == 0:
                    print(f"  {task} {k + 1}/{len(todo)}: {n_tok / (time.time() - t0):.1f} gen tok/s aggregate", flush=True)
        print(f"{task} done in {time.time() - t0:.0f} s, {n_tok} tokens", flush=True)


if __name__ == "__main__":
    main()
