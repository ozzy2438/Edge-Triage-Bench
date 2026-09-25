import hashlib
import json
import re

from etb.configs import PROCESSED, ROOT
from etb.data import LABEL_NAMES


def load(name):
    return [json.loads(line) for line in open(PROCESSED / f"{name}.jsonl")]


def norm(t):
    return " ".join(t.lower().split())


def test_sizes_labels_and_no_overlap():
    s = {n: load(n) for n in ("train", "dev", "test")}
    assert len(s["dev"]) == 100 and len(s["test"]) == 500
    for n, rows in s.items():
        assert {r["label"] for r in rows} == set(LABEL_NAMES), n
    ids = {n: {r["id"] for r in rows} for n, rows in s.items()}
    texts = {n: {norm(r["text"]) for r in rows} for n, rows in s.items()}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        assert not ids[a] & ids[b] and not texts[a] & texts[b], (a, b)


def test_learning_curve_subsets():
    from etb.baselines import SEEDS, SEEDED, curve_subset

    tr, test = load("train"), {norm(r["text"]) for r in load("test")}
    for n in SEEDED:
        for seed in SEEDS:
            sub = curve_subset(tr, n, seed)
            assert len(sub) == n and {r["label"] for r in sub} == set(LABEL_NAMES)
            assert not {norm(r["text"]) for r in sub} & test
    assert {r["text"] for r in curve_subset(tr, 80, SEEDS[0])} != {r["text"] for r in curve_subset(tr, 80, SEEDS[1])}


def test_hashes_match_data_md():
    md = (ROOT / "DATA.md").read_text()
    for n in ("train", "dev", "test"):
        h = hashlib.sha256((PROCESSED / f"{n}.jsonl").read_bytes()).hexdigest()
        assert re.search(rf"\| {n} \| \d+ \| `{h}` \|", md), n
