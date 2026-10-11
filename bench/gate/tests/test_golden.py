"""Golden numbers from the rgmin #38 fixture."""

from pathlib import Path

from rggate.contract import load_csv
from rggate.report import analyze

FIXTURE = Path(__file__).parent / "fixtures" / "pr38_rgmin.csv"


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
