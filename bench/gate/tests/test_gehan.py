"""A fast unconverged candidate does not win, and a basin split is not a loss."""

from pathlib import Path

from rggate.contract import load_csv
from rggate.report import analyze

FIXTURE = Path(__file__).parent / "fixtures" / "censor_split.csv"


def test_fast_censoring_is_not_a_win():
    rows = load_csv(FIXTURE)
    report = analyze(rows, [("cand", "base")], resamples=49, plan_n=20)
    pair = report.pairs[0]
    assert pair.wins == 1
    assert pair.losses == 0
    assert pair.indeterminate == 1
    assert pair.splits == 1
    assert pair.n == 0
    assert pair.window_fallback == 1
    assert pair.clusters < report.min_clusters
    assert pair.geomean_low != pair.geomean_low
    by_seed = {row.system: row for row in pair.systems}
    mol = by_seed["mol"]
    assert mol.wins == 1
    assert mol.splits == 1
    assert mol.indeterminate == 1
