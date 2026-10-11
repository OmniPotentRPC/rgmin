"""Stratified normal-mixture walk."""

from rggate.rng import philox
from rggate.sequence import optimal_rho, stratified_labels, walk, walk_betting


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


def test_type_i_stratified_correlated_starts():
    # A shared system effect shifts both methods' log call counts.
    # Under no method effect the paired win stays fair, and the starts
    # inside a system are correlated through that effect.
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


def test_betting_all_wins_passes():
    result = walk_betting([1] * 30, alpha=0.05, margin=0.0)
    assert result["decision"] == "pass"
    assert result["low"] > 0.5
