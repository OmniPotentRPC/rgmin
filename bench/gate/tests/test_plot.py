"""SVG output when matplotlib is installed."""

from pathlib import Path

import pytest

from rggate.contract import load_csv
from rggate.report import analyze, render

FIXTURE = Path(__file__).parent / "fixtures" / "pr38_rgmin.csv"


def test_svg_and_rank_flag(tmp_path):
    pytest.importorskip("matplotlib")
    from rggate.plot import write_svg

    rows = load_csv(FIXTURE)
    report = analyze(rows, [("fire-exp", "fire")], resamples=49)
    path = tmp_path / "gate.svg"
    write_svg(report, path)
    text = path.read_text(encoding="utf-8")
    assert "<svg" in text
    rendered = render(report)
    assert "rank_invariance=na" in rendered
    assert "decision=" in rendered
