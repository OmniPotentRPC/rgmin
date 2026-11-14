"""Rows that report a thread count have to agree."""

from rggate.contract import parse_csv
from rggate.report import analyze

HEADER = "system,method,seed,converged,iters,calls,energy,fmax,wall_s,threads"


def test_matching_thread_rows_are_ok():
    text = "\n".join(
        [
            HEADER,
            "s,a,0,1,1,10,-1,1e-4,0.1,1",
            "s,a,0,1,1,10,-1,1e-4,0.1,2",
            "s,b,0,1,1,12,-1,1e-4,0.1,1",
            "s,b,0,1,1,12,-1,1e-4,0.1,2",
        ]
    )
    report = analyze(parse_csv(text), [("b", "a")], resamples=19)
    assert report.rank_invariance == "ok"
    assert report.pairs[0].n == 1


def test_mismatched_thread_rows_fail():
    text = "\n".join(
        [
            HEADER,
            "s,a,0,1,1,10,-1,1e-4,0.1,1",
            "s,a,0,1,1,11,-1,1e-4,0.1,2",
        ]
    )
    report = analyze(parse_csv(text), [], resamples=19)
    assert report.rank_invariance == "fail"
    assert report.exit_code == 1
