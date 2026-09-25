"""Path check: Qwen3-0.6B Q4_K_M on 20 dev items, plus TF-IDF on the same items. Writes results/smoke.json."""
import json

import numpy as np

from etb.baselines import fit_tfidf
from etb.configs import RESULTS
from etb.metrics import macro_f1
from etb.run import Server, load_split


def main() -> None:
    dev = load_split("dev")
    items = dev[:20]
    y = [it["label"] for it in items]
    with Server("qwen3-0.6b-Q4_K_M") as srv:
        srv.classify(dev[20]["text"])  # warm-up, discarded
        recs = [srv.classify(it["text"]) for it in items]
    pred = fit_tfidf(load_split("train")).predict([it["text"] for it in items])
    out = {"config": "qwen3-0.6b-Q4_K_M", "n": len(items), "split": "dev",
           "llm_macro_f1": float(macro_f1(y, [r["pred"] or "INVALID" for r in recs])),
           "llm_lat_p50_ms": float(np.percentile([r["latency_s"] for r in recs], 50) * 1000),
           "tfidf_macro_f1": float(macro_f1(y, pred)),
           "invalid": int(sum(r["invalid"] for r in recs))}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "smoke.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
