"""Run one or more configurations over a split via llama-server; append per-item results to jsonl (resumable)."""
import json
import math
import subprocess
import sys
import time

import httpx
import numpy as np

from etb.configs import CTX, DEVICE, LLAMA_BIN, MAX_NARRATIVE_TOKENS, PROCESSED, RAW, SEED, THREADS, configs, gguf_path
from etb.data import LABELS, LABEL_NAMES
from etb.measure import PeakRSS, machine_info

PORT = 8089
URL = f"http://127.0.0.1:{PORT}"
COLD_N = 50  # first 50 items of the split (file order) get a second, cache-free pass
# Model-facing label words: chosen so each starts with a distinct first token, so first-token probs are per-label.
WORDS = dict(zip(LABEL_NAMES, ["NewCard", "CardIssue", "Payment", "ATM", "TopUp", "Transfer", "Exchange", "Account"]))
GRAMMAR = "root ::= " + " | ".join(f'"{w}"' for w in WORDS.values())
SYSTEM = ("You are a triage assistant for a digital bank's customer support team. "
          "Classify the customer message into exactly one category. Reply with the category name only.\n\n"
          + "\n".join(f"{w}: {LABELS[name][0]}" for name, w in WORDS.items()))


def shot_rows() -> list[dict]:
    """One dev example per label (first by id), sorted by id. Dev-only by construction."""
    rows = [json.loads(line) for line in open(PROCESSED / "dev.jsonl")]
    return sorted((next(r for r in rows if r["label"] == name) for name in LABEL_NAMES), key=lambda r: r["id"])


def few_shot() -> list[dict]:
    return [m for r in shot_rows() for m in ({"role": "user", "content": f"Customer message: {r['text']}"},
                                             {"role": "assistant", "content": WORDS[r["label"]]})]


SHOTS = few_shot()


def parse_label(content: str) -> str | None:
    s = content.strip().strip(".").lower()
    return next((name for name, w in WORDS.items() if w.lower() == s), None)


def label_probs(top: list[dict]) -> tuple[dict, float]:
    """Attribute each top-k first token to the single label word it prefixes; renormalise. Also return raw label mass."""
    mass = dict.fromkeys(LABEL_NAMES, 0.0)
    for t in top:
        s = t["token"].strip()
        hits = [name for name, w in WORDS.items() if s and w.startswith(s)]
        if len(hits) == 1:
            mass[hits[0]] += math.exp(t["logprob"]) if "logprob" in t else t["prob"]
    total = sum(mass.values())
    return ({k: v / total for k, v in mass.items()} if total else mass), total


class Server:
    def __init__(self, config: str):
        self.cmd = [LLAMA_BIN / "llama-server", "-m", gguf_path(config), "-t", str(THREADS), "-tb", str(THREADS),
                    "-ngl", "0", "-c", str(CTX), "-np", "1", "-s", str(SEED), "-lm", "none", "--no-webui",
                    "--port", str(PORT)]

    def __enter__(self):
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.rss = PeakRSS(self.proc.pid).__enter__()
        self.http = httpx.Client(base_url=URL, timeout=300)
        while True:
            try:
                if self.http.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited: {self.cmd}")
            time.sleep(0.05)
        self.load_s = time.perf_counter() - t0
        return self

    def __exit__(self, *_):
        self.rss.__exit__()
        self.proc.terminate()
        self.proc.wait()

    def post(self, path, **body):
        r = self.http.post(path, json=body)
        r.raise_for_status()
        return r.json()

    def truncate(self, text: str) -> tuple[str, int, bool]:
        toks = self.post("/tokenize", content=text)["tokens"]
        if len(toks) <= MAX_NARRATIVE_TOKENS:
            return text, len(toks), False
        return self.post("/detokenize", tokens=toks[:MAX_NARRATIVE_TOKENS])["content"], len(toks), True

    def classify(self, text: str, cache: bool = True) -> dict:
        msgs = [{"role": "system", "content": SYSTEM}, *SHOTS, {"role": "user", "content": f"Customer message: {text}"}]
        prompt = self.post("/apply-template", messages=msgs,
                           chat_template_kwargs={"enable_thinking": False})["prompt"]
        t0 = time.perf_counter()
        r = self.post("/completion", prompt=prompt, n_predict=6, temperature=0, seed=SEED, grammar=GRAMMAR,
                      n_probs=50, cache_prompt=cache)
        lat = time.perf_counter() - t0
        probs, mass = label_probs(r["completion_probabilities"][0]["top_logprobs"])
        pred = parse_label(r["content"])
        tm = r["timings"]
        return {"pred": pred, "invalid": pred is None, "conf": probs.get(pred, 0.0) if pred else 0.0,
                "probs": probs, "label_mass": mass, "latency_s": lat, "prompt_n": tm["prompt_n"],
                "prompt_tps": tm["prompt_per_second"], "gen_tps": tm["predicted_per_second"]}


def load_split(split: str) -> list[dict]:
    return [json.loads(line) for line in open(PROCESSED / f"{split}.jsonl")]


def run(config: str, split: str) -> None:
    items = load_split(split)
    out = RAW / DEVICE / split / f"{config}.jsonl"
    meta_path = out.with_suffix(".meta.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"wall_s": 0.0, "peak_rss_mb": 0.0}
    if not todo and "cold_p50_ms" in meta:
        return print(f"{config}/{split}: complete")
    with Server(config) as srv:
        for it in load_split("dev")[:3]:  # warm-up, discarded
            srv.classify(it["text"])
        if todo:
            t0 = time.perf_counter()
            with open(out, "a") as f:
                for i, it in enumerate(todo):
                    text, ntok, trunc = srv.truncate(it["text"])
                    rec = {"id": it["id"], "label": it["label"], "n_tokens": ntok, "truncated": trunc, **srv.classify(text)}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if i % 50 == 0:
                        print(f"{config}/{split}: {len(done) + i + 1}/{len(items)}", flush=True)
            meta.update(wall_s=meta["wall_s"] + time.perf_counter() - t0, resumed=bool(done), load_s=srv.load_s)
        meta.update(cold_latency(srv, items[:COLD_N], {json.loads(l)["id"]: json.loads(l) for l in open(out)}))
        meta.update(config=config, split=split, device=DEVICE, peak_rss_mb=max(meta["peak_rss_mb"], srv.rss.peak_mb),
                    file_mb=gguf_path(config).stat().st_size / 1e6, server_args=[str(a) for a in srv.cmd[1:]],
                    machine=machine_info())
    meta_path.write_text(json.dumps(meta, indent=1))


def cold_latency(srv: Server, items: list[dict], warm: dict) -> dict:
    """Full-prompt latency (no KV-cache reuse) on a fixed subset; also checks predictions match the warm run."""
    recs = [srv.classify(srv.truncate(it["text"])[0], cache=False) for it in items]
    lat = np.array([r["latency_s"] for r in recs]) * 1000
    return {"cold_n": len(recs), "cold_p50_ms": float(np.percentile(lat, 50)), "cold_p95_ms": float(np.percentile(lat, 95)),
            "cold_prompt_n": float(np.median([r["prompt_n"] for r in recs])),
            "cold_prompt_tps": float(np.median([r["prompt_tps"] for r in recs])),
            "cold_warm_agree": float(np.mean([r["pred"] == warm[it["id"]]["pred"] for r, it in zip(recs, items)]))}


if __name__ == "__main__":
    split, *cfgs = sys.argv[1:]
    for c in cfgs or configs():
        if gguf_path(c).exists():
            run(c, split)
        else:
            print(f"skip {c}: no gguf")
