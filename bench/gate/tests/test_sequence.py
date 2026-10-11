"""Stratified normal-mixture walk."""

import math

from rggate.rng import philox
from rggate.sequence import (
    decide_min,
    optimal_rho,
    pair_alpha,
    stratified_labels,
    walk,
    walk_agrapa,
)


def test_rho_one_needs_eleven_and_the_horizon_needs_ten():
    early = walk([1] * 10, alpha=0.05, margin=0.0, rho=1.0)
    assert early["decision"] == "undecided"
    planned = optimal_rho(10, 0.05)
    cleared = walk([1] * 10, alpha=0.05, margin=0.0, rho=planned)
    assert cleared["decision"] == "pass"
    assert cleared["low"] > 0.5
    failed = walk([0] * 40, alpha=0.05, margin=0.0, rho=1.0)
    assert failed["decision"] == "fail"


def test_round_robin_mixes_systems():
    records = [("a", 1), ("a", 1), ("b", 0), ("b", 0)]
    ordered = stratified_labels(records, seed=1, pair_index=0)
    again = stratified_labels(records, seed=1, pair_index=0)
    assert ordered == again
    assert ordered[0] != ordered[1]


def test_decide_min_is_eleven_at_the_default_horizon():
    rho = optimal_rho(50, 0.05)
    assert decide_min(0.05, 0.0, rho) == 11
    assert pair_alpha(0.05, 3) == 0.05 / 3


def test_type_i_shared_effect_cancels_in_the_pair():
    # The same system effect is added to both methods' log call counts.
    # It cancels in the paired sign, so the labels are independent fair
    # coins. This does not exercise a system by method interaction.
    rng = philox(7, 5)
    rho = optimal_rho(50, 0.05)
    rejects = 0
    replicates = 200
    for replicate in range(replicates):
        records = []
        for system in range(8):
            effect = float(rng.normal())
            for _start in range(4):
                candidate = effect + float(rng.normal())
                baseline = effect + float(rng.normal())
                records.append((f"s{system}", 1 if candidate < baseline else 0))
        ordered = stratified_labels(records, seed=100 + replicate, pair_index=3)
        result = walk(ordered, alpha=0.05, margin=0.0, rho=rho)
        if result["decision"] != "undecided":
            rejects += 1
    assert rejects / replicates <= 0.08


def test_power_at_nine_tenths():
    rng = philox(9, 5)
    rho = optimal_rho(50, 0.05)
    passes = 0
    for _ in range(40):
        labels = (rng.random(80) < 0.9).astype(int).tolist()
        result = walk(labels, alpha=0.05, margin=0.0, rho=rho)
        if result["decision"] == "pass":
            passes += 1
    assert passes >= 30


def test_type_i_system_method_interaction():
    # Half the systems win with probability 0.85 and half with 0.15.
    # The mean over starts is 0.5. Starts that share a system are
    # correlated, because they share that probability.
    rng = philox(17, 5)
    alpha = 0.05
    rho = optimal_rho(50, alpha)
    replicates = 1000
    rejects = 0
    n_systems = 20
    for replicate in range(replicates):
        records = []
        for system in range(n_systems):
            probability = 0.85 if system < n_systems // 2 else 0.15
            for _start in range(8):
                records.append((f"s{system}", 1 if float(rng.random()) < probability else 0))
        ordered = stratified_labels(records, seed=200 + replicate, pair_index=4)
        result = walk(ordered, alpha=alpha, margin=0.0, rho=rho)
        if result["decision"] != "undecided":
            rejects += 1
    rate = rejects / replicates
    allowance = alpha + 2.0 * math.sqrt(alpha * (1.0 - alpha) / replicates)
    assert rate <= allowance


def test_family_wise_three_null_pairs():
    rng = philox(19, 5)
    alpha = 0.05
    level = pair_alpha(alpha, 3)
    rho = optimal_rho(50, level)
    replicates = 400
    rejects = 0
    for _replicate in range(replicates):
        hit = False
        for _pair in range(3):
            labels = [1 if float(rng.random()) < 0.5 else 0 for _ in range(80)]
            result = walk(labels, alpha=level, margin=0.0, rho=rho)
            if result["decision"] != "undecided":
                hit = True
        if hit:
            rejects += 1
    assert rejects / replicates <= alpha


def test_agrapa_all_wins_passes():
    result = walk_agrapa([1] * 30, alpha=0.05, margin=0.0)
    assert result["decision"] == "pass"
    assert result["low"] > 0.5


def test_type_i_agrapa_fair_sequences():
    rng = philox(23, 5)
    alpha = 0.05
    replicates = 400
    rejects = 0
    for _replicate in range(replicates):
        labels = [1 if float(rng.random()) < 0.5 else 0 for _ in range(80)]
        result = walk_agrapa(labels, alpha=alpha, margin=0.0)
        if result["decision"] != "undecided":
            rejects += 1
    rate = rejects / replicates
    allowance = alpha + 2.0 * math.sqrt(alpha * (1.0 - alpha) / replicates)
    assert rate <= allowance
