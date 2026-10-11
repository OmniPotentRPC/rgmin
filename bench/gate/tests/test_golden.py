"""Golden numbers from the rgmin #38 fixture."""

from pathlib import Path

from rggate.contract import load_csv
from rggate.report import analyze

FIXTURE = Path(__file__).parent / "fixtures" / "pr38_rgmin.csv"


def test_lj38_fire_basins_and_lbfgs_median():
    rows = load_csv(FIXTURE)
    assert len(rows) == 114
    report = analyze(
        rows,
        [("lbfgs-exp", "lbfgs-wolfe"), ("fire-exp", "fire"), ("cg-exp", "cg-wolfe")],
        resamples=199,
        jobs=1,
    )
    fire = [row for row in report.systems if row.system == "lj38" and row.method == "fire"]
    assert len(fire) == 1
    assert fire[0].different == 5
    assert fire[0].success == 0
    wolfe = [
        row for row in report.systems if row.system == "lj38" and row.method == "lbfgs-wolfe"
    ]
    assert wolfe[0].call_median == 164
    assert report.different_stationary >= 5
