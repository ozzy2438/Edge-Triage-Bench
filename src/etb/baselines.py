"""Majority-class and TF-IDF + logistic regression baselines, measured like the LLM configs (child process, RSS, latency)."""
import json
import subprocess
import sys
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from etb.configs import DEVICE, MODELS_DIR, RAW, SEED
from etb.measure import PeakRSS, machine_info
from etb.run import load_split

BASELINES = {"baseline-majority": MODELS_DIR / "majority.joblib", "baseline-tfidf-lr": MODELS_DIR / "tfidf_lr.joblib"}


def train() -> None:
    tr = load_split("train")
    X, y = [r["text"] for r in tr], [r["label"] for r in tr]
    top, n = Counter(y).most_common(1)[0]
    joblib.dump({"label": top, "prior": n / len(y)}, BASELINES["baseline-majority"])
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                        LogisticRegression(C=10, max_iter=2000, random_state=SEED))
    joblib.dump(clf.fit(X, y), BASELINES["baseline-tfidf-lr"])


def infer(name: str, split: str) -> None:
    """Child process: load model (timed), classify item by item, write jsonl, print load time."""
    t0 = time.perf_counter()
    model = joblib.load(BASELINES[name])
    load_s = time.perf_counter() - t0
    out = RAW / DEVICE / split / f"{name}.jsonl"
    with open(out, "w") as f:
        for it in load_split(split):
            t = time.perf_counter()
            if isinstance(model, dict):
                pred, conf, probs = model["label"], model["prior"], {model["label"]: model["prior"]}
            else:
                p = model.predict_proba([it["text"]])[0]
                i = int(np.argmax(p))
                pred, conf, probs = model.classes_[i], float(p[i]), dict(zip(model.classes_, map(float, p)))
            f.write(json.dumps({"id": it["id"], "label": it["label"], "pred": pred, "invalid": False, "conf": conf,
                                "probs": probs, "latency_s": time.perf_counter() - t}) + "\n")
    print(load_s)


def run(split: str) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    train()
    (RAW / DEVICE / split).mkdir(parents=True, exist_ok=True)
    for name, path in BASELINES.items():
        t0 = time.perf_counter()
        proc = subprocess.Popen([sys.executable, "-m", "etb.baselines", "--infer", name, split], stdout=subprocess.PIPE, text=True)
        with PeakRSS(proc.pid, 0.01) as rss:
            load_s = float(proc.communicate()[0])
        meta = {"config": name, "split": split, "device": DEVICE, "load_s": load_s, "wall_s": time.perf_counter() - t0,
                "peak_rss_mb": rss.peak_mb, "file_mb": path.stat().st_size / 1e6, "machine": machine_info()}
        (RAW / DEVICE / split / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--infer":
        infer(*sys.argv[2:])
    else:
        run(sys.argv[1])
