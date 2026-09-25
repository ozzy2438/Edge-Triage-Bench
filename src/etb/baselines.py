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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from etb.configs import DEVICE, MODELS_DIR, RAW, SEED
from etb.measure import PeakRSS, machine_info
from etb.run import load_split, shot_rows

CURVE = (8, 80, 800)  # single models in the main bench; n=8 is the prompt examples, not a sample
SEEDED = (8, 80, 200, 400, 800)  # post-hoc curve, 5 stratified seeds each
SEEDS = tuple(range(SEED, SEED + 5))
BASELINES = {"baseline-majority": MODELS_DIR / "majority.joblib", "baseline-tfidf-lr": MODELS_DIR / "tfidf_lr.joblib",
             **{f"baseline-tfidf-lr-n{n}": MODELS_DIR / f"tfidf_lr_n{n}.joblib" for n in CURVE}}


def curve_subset(tr: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """Stratified sample of the training split. n=8 is one example per label."""
    sub, _ = train_test_split(tr, train_size=n, stratify=[r["label"] for r in tr], random_state=seed)
    return list(sub)


def fit_tfidf(rows: list[dict]):
    # min_df=2 would delete almost the whole vocabulary at 8/80 examples
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2 if len(rows) >= 800 else 1),
                        LogisticRegression(C=10, max_iter=2000, random_state=SEED))
    return clf.fit([r["text"] for r in rows], [r["label"] for r in rows])


def train() -> None:
    tr = load_split("train")
    top, n = Counter(r["label"] for r in tr).most_common(1)[0]
    joblib.dump({"label": top, "prior": n / len(tr)}, BASELINES["baseline-majority"])
    joblib.dump(fit_tfidf(tr), BASELINES["baseline-tfidf-lr"])
    joblib.dump(fit_tfidf(shot_rows()), BASELINES["baseline-tfidf-lr-n8"])
    for n in CURVE:
        if n != 8:
            joblib.dump(fit_tfidf(curve_subset(tr, n)), BASELINES[f"baseline-tfidf-lr-n{n}"])


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


def seeded_curve(split: str = "test"):
    """Post-hoc: macro-F1 at each training size over 5 stratified seeds. Does not touch the LLM results."""
    import pandas as pd

    from etb.configs import RESULTS
    from etb.metrics import macro_f1

    te, tr = load_split(split), load_split("train")
    X, y = [r["text"] for r in te], [r["label"] for r in te]
    rows = [{"n": n, "seed": s, "kind": "stratified", "macro_f1": float(macro_f1(y, fit_tfidf(curve_subset(tr, n, s)).predict(X)))}
            for n in SEEDED for s in SEEDS]
    rows.append({"n": 8, "seed": -1, "kind": "prompt",
                 "macro_f1": float(macro_f1(y, fit_tfidf(shot_rows()).predict(X)))})
    rows.append({"n": len(tr), "seed": SEED, "kind": "full",
                 "macro_f1": float(macro_f1(y, fit_tfidf(tr).predict(X)))})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / ("learning_curve_seeds.csv" if split == "test" else f"learning_curve_seeds_{split}.csv"), index=False)
    return df


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
