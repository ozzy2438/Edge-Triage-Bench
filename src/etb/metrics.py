"""Quality, abstention and calibration metrics over per-item records."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from etb.configs import SEED
from etb.data import LABEL_NAMES

COVERAGES = (0.5, 0.7, 0.9)


def macro_f1(y, p) -> float:
    return f1_score(y, p, labels=LABEL_NAMES, average="macro", zero_division=0)


def bootstrap_ci(y, p, fn, n=1000, seed=SEED) -> tuple[float, float]:
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return tuple(np.percentile(stats, [2.5, 97.5]))


def risk_coverage(conf, correct) -> tuple[np.ndarray, np.ndarray]:
    """Coverage k/N and selective accuracy of the k most confident items, for k = 1..N (stable sort)."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    c = np.asarray(correct, float)[order]
    k = np.arange(1, len(c) + 1)
    return k / len(c), np.cumsum(c) / k


def selective_acc(conf, correct, coverage: float) -> float:
    cov, acc = risk_coverage(conf, correct)
    return float(acc[int(np.ceil(coverage * len(cov))) - 1])


def aurc(conf, correct) -> float:
    return float(np.mean(1 - risk_coverage(conf, correct)[1]))


def reliability(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    idx = np.clip((conf * bins).astype(int), 0, bins - 1)
    return [(conf[idx == b].mean(), correct[idx == b].mean(), int((idx == b).sum())) for b in range(bins) if (idx == b).any()]


def ece(conf, correct, bins=15) -> float:
    return float(sum(n * abs(c - a) for c, a, n in reliability(conf, correct, bins)) / len(conf))


def summarise(recs: list[dict]) -> dict:
    y = [r["label"] for r in recs]
    p = [r["pred"] or "INVALID" for r in recs]
    correct = np.array([a == b for a, b in zip(y, p)])
    conf = np.array([r["conf"] for r in recs])
    acc = lambda a, b: float(np.mean(a == b))
    lat = np.array([r["latency_s"] for r in recs]) * 1000
    out = {"n": len(recs), "accuracy": acc(np.array(y), np.array(p)), "macro_f1": macro_f1(y, p),
           "invalid_rate": float(np.mean([r["invalid"] for r in recs])),
           "aurc": aurc(conf, correct), "ece": ece(conf, correct),
           "lat_p50_ms": float(np.percentile(lat, 50)), "lat_p95_ms": float(np.percentile(lat, 95)),
           "lat_max_ms": float(lat.max())}
    out["acc_lo"], out["acc_hi"] = bootstrap_ci(y, p, acc)
    out["f1_lo"], out["f1_hi"] = bootstrap_ci(y, p, macro_f1)
    for c in COVERAGES:
        out[f"sel_acc@{int(c * 100)}"] = selective_acc(conf, correct, c)
    out.update({f"f1_{k}": v for k, v in zip(LABEL_NAMES, f1_score(y, p, labels=LABEL_NAMES, average=None, zero_division=0))})
    for k in ("prompt_tps", "gen_tps", "label_mass"):
        if k in recs[0]:
            out[k] = float(np.median([r[k] for r in recs]))
    out["truncated"] = int(sum(r.get("truncated", False) for r in recs))
    return out


def confusion(recs) -> np.ndarray:
    return confusion_matrix([r["label"] for r in recs], [r["pred"] or "INVALID" for r in recs], labels=LABEL_NAMES)
