"""Normal-mixture confidence sequence."""

import numpy as np

from rggate.rng import philox
from rggate.sequence import ordered_labels, walk


def test_all_wins_pass_and_losses_fail():
    passed = walk([1] * 40, alpha=0.05, margin=0.0)
    assert passed["decision"] == "pass"
    assert passed["systems_used"] < 40
    assert passed["low"] > 0.5
    failed = walk([0] * 40, alpha=0.05, margin=0.0)
    assert failed["decision"] == "fail"
    assert failed["high"] < 0.5


def test_type_i_under_null():
    rng = philox(7, 5)
    rejects = 0
    replicates = 400
    for _ in range(replicates):
        labels = (rng.random(200) < 0.5).astype(int).tolist()
        result = walk(labels, alpha=0.05, margin=0.0)
        if result["decision"] != "undecided":
            rejects += 1
    rate = rejects / replicates
    assert rate <= 0.08


def test_power_at_nine_tenths():
    rng = philox(9, 5)
    passes = 0
    for _ in range(40):
        labels = (rng.random(80) < 0.9).astype(int).tolist()
        result = walk(labels, alpha=0.05, margin=0.0)
        if result["decision"] == "pass":
            passes += 1
    assert passes >= 30


def test_order_uses_philox_not_label_order():
    labels = [("b", 1), ("a", 0), ("c", 1)]
    first = ordered_labels(labels, seed=1, pair_index=0)
    second = ordered_labels(labels, seed=1, pair_index=0)
    assert first == second
    assert sorted(first) == [0, 1, 1]
    other = ordered_labels(labels, seed=2, pair_index=0)
    assert len(other) == 3
