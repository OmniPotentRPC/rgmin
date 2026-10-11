"""Golden numbers from the rgmin #38 fixture."""

from pathlib import Path

from rggate.contract import load_csv
from rggate.report import analyze

FIXTURE = Path(__file__).parent / "fixtures" / "pr38_rgmin.csv"
DIM_FIXTURE = Path(__file__).parent / "fixtures" / "pr38_rgmin_dim.csv"


def test_lj38_fire_splits_against_its_baseline():
    rows = load_csv(FIXTURE)
    assert len(rows) == 114
    report = analyze(
        rows,
        [("lbfgs-exp", "lbfgs-wolfe"), ("fire-exp", "fire"), ("cg-exp", "cg-wolfe")],
        resamples=99,
        jobs=1,
    )
    fire = next(row for row in report.pairs if row.candidate == "fire-exp")
    lj38 = next(row for row in fire.systems if row.system == "lj38")
    assert lj38.splits == 5
    assert lj38.wins == 0
    assert lj38.losses == 0
    wolfe = next(
        row for row in report.systems if row.system == "lj38" and row.method == "lbfgs-wolfe"
    )
    assert wolfe.call_median == 164
    assert wolfe.success == 5
    assert report.match == "energy-only"
    assert report.pairs[0].window_fallback == 19


def test_dim_fixture_keeps_only_the_lj38_splits():
    rows = load_csv(DIM_FIXTURE)
    assert len(rows) == 114
    assert {row.system: row.dim for row in rows}["lj38"] == 114
    assert {row.system: row.dim for row in rows}["al"] == 1203
    assert {row.system: row.dim for row in rows}["cu-vac"] == 6141
    report = analyze(
        rows,
        [("lbfgs-exp", "lbfgs-wolfe"), ("fire-exp", "fire"), ("cg-exp", "cg-wolfe")],
        resamples=19,
        jobs=1,
    )
    assert report.match == "energy-only"
    assert report.alpha_pair == 0.05 / 3
    expected = {"lbfgs-exp": (10, 2), "fire-exp": (2, 11), "cg-exp": (3, 11)}
    for pair in report.pairs:
        assert pair.splits == 5
        assert pair.window_fallback == 0
        assert (pair.wins, pair.losses) == expected[pair.candidate]
        assert all(row.splits == 0 or row.system == "lj38" for row in pair.systems)
        lj38 = next(row for row in pair.systems if row.system == "lj38")
        assert lj38.splits == 5
        assert pair.decision == "undecided"
        assert pair.forced is (pair.wins + pair.losses < report.decide_min)
    assert report.decide_min == 13
