"""Contract and argument errors are not an undecided gate."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args):
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "rggate", *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_bad_pair_exits_3():
    completed = _run(["missing.csv", "--pair", "not-a-pair"])
    assert completed.returncode == 3


def test_unknown_major_exits_3(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "# rggate-csv 2\nsystem,method,seed,converged,iters,calls,energy,fmax,wall_s\n",
        encoding="utf-8",
    )
    completed = _run([str(path), "--pair", "a=b"])
    assert completed.returncode == 3
    assert "major 2" in completed.stderr
