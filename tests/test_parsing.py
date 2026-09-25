import math

import pytest

from etb.data import LABEL_NAMES
from etb.run import WORDS, label_probs, parse_label


@pytest.mark.parametrize("s,want", [("NewCard", "card_setup"), (" topup\n", "top_up"), ("ATM.", "cash_atm"),
                                    ("Card", None), ("TopUp Transfer", None), ("", None)])
def test_parse_label(s, want):
    assert parse_label(s) == want


def test_words_cover_labels_and_are_prefix_free():
    assert list(WORDS) == LABEL_NAMES
    w = list(WORDS.values())
    assert not any(a != b and b.startswith(a) for a in w for b in w)


def test_label_probs_attributes_unique_prefixes_only():
    top = [{"token": "Top", "logprob": math.log(0.5)}, {"token": " TopUp", "logprob": math.log(0.1)},
           {"token": "Trans", "logprob": math.log(0.2)}, {"token": "T", "logprob": math.log(0.1)},  # ambiguous
           {"token": "The", "logprob": math.log(0.1)}]
    probs, mass = label_probs(top)
    assert mass == pytest.approx(0.8)
    assert probs["top_up"] == pytest.approx(0.75) and probs["transfer"] == pytest.approx(0.25)
    assert sum(probs.values()) == pytest.approx(1.0)
