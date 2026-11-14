"""Summary text does not depend on --jobs."""

import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "pr38_rgmin.csv"
ROOT = Path(__file__).resolve().parents[1]


def _run(jobs):
    cmd = [
        sys.executable,
        "-m",
        "rggate",
        str(FIXTURE),
        "--pair",
        "lbfgs-exp=lbfgs-wolfe",
        "--pair",
        "fire-exp=fire",
        "--pair",
        "cg-exp=cg-wolfe",
        "--resamples",
        "99",
        "--jobs",
        str(jobs),
    ]
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert completed.returncode in (0, 1, 2)
    assert completed.stderr == ""
    return completed.returncode, completed.stdout


def test_jobs_are_identical():
    code1, out1 = _run(1)
    code2, out2 = _run(2)
    code4, out4 = _run(4)
    assert out1 == out2 == out4
    assert code1 == code2 == code4
