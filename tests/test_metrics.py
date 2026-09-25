import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from etb.data import LABEL_NAMES
from etb.metrics import aurc, bootstrap_ci, ece, macro_f1, risk_coverage, selective_acc, summarise


def fake(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(LABEL_NAMES, n)
    p = np.where(rng.random(n) < 0.7, y, rng.choice(LABEL_NAMES, n))
    return y, p, rng.random(n)


def test_macro_f1_and_accuracy_match_sklearn():
    y, p, conf = fake()
    recs = [{"label": a, "pred": b, "conf": c, "invalid": False, "latency_s": 0.1} for a, b, c in zip(y, p, conf)]
    s = summarise(recs)
    assert s["accuracy"] == pytest.approx(accuracy_score(y, p))
    assert s["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert s["acc_lo"] <= s["accuracy"] <= s["acc_hi"]


def test_invalid_counts_as_wrong():
    recs = [{"label": "card_setup", "pred": None, "conf": 0.0, "invalid": True, "latency_s": 0.1},
            {"label": "card_setup", "pred": "card_setup", "conf": 0.9, "invalid": False, "latency_s": 0.1}]
    s = summarise(recs)
    assert s["accuracy"] == 0.5 and s["invalid_rate"] == 0.5


def test_risk_coverage_hand_example():
    conf, correct = [0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1]
    cov, acc = risk_coverage(conf, correct)
    assert list(cov) == [0.25, 0.5, 0.75, 1.0]
    assert acc == pytest.approx([1, 1, 2 / 3, 3 / 4])
    assert selective_acc(conf, correct, 0.5) == 1.0
    assert aurc(conf, correct) == pytest.approx(np.mean([0, 0, 1 / 3, 1 / 4]))


def test_perfect_confidence_ranking_beats_reversed():
    correct = np.array([1] * 50 + [0] * 50)
    assert aurc(np.linspace(1, 0, 100), correct) < aurc(np.linspace(0, 1, 100), correct)


def test_ece():
    assert ece([1.0, 1.0], [1, 1]) == pytest.approx(0.0)
    assert ece([0.9] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.4)


def test_bootstrap_deterministic():
    y, p, _ = fake()
    assert bootstrap_ci(y, p, macro_f1) == bootstrap_ci(y, p, macro_f1)
