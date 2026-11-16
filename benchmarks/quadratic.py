"""asv rows for the quadratic cells and the aluminium rows that were missing.

``time_solve`` and ``time_relax`` start after the cell is built. The
spectrum, the start point, and the aluminium packing sit in ``setup``.
FIRE 2.0 on aluminium is already in ``relax.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SIZES = [100, 1000, 10000]
METHODS = ["fire", "fire2", "cg", "trust"]
_QUAD = Path(__file__).resolve().parent.parent / "target" / "release" / "rgquad"
_DRIVER = Path(__file__).resolve().parent / "relax_driver"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    return env


def _quad() -> str:
    named = os.environ.get("RGQUAD")
    if named:
        return named
    root = os.environ.get("ASV_ENV_DIR")
    if root:
        return str(Path(root) / "bin" / "rgquad")
    return str(_QUAD)


def _driver() -> str:
    named = os.environ.get("RELAX_DRIVER")
    if named:
        return named
    root = os.environ.get("ASV_ENV_DIR")
    if root:
        return str(Path(root) / "bin" / "relax_driver")
    return str(_DRIVER)


class _Session:
    def __init__(self, argv: list[str]) -> None:
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=_env(),
        )
        line = self.proc.stdout.readline() if self.proc.stdout is not None else ""
        if not line.startswith("ready"):
            raise RuntimeError(line.strip() or "cell did not start")

    def run(self) -> str:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("cell has no pipes")
        self.proc.stdin.write("run\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("cell closed")
        return line

    def close(self) -> None:
        if self.proc.poll() is None and self.proc.stdin is not None:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)


class Quadratic:
    """FIRE 1.0, FIRE 2.0, Polak-Ribiere, and the restricted step."""

    params = (SIZES, METHODS)
    param_names = ["n", "method"]
    timeout = 600

    def setup(self, n: int, method: str) -> None:
        if not Path(_quad()).is_file():
            raise RuntimeError("quadratic driver is not built")
        self.session = _Session([_quad(), "session", method, str(n)])

    def time_solve(self, n: int, method: str) -> None:
        line = self.session.run()
        if not line.startswith("ok"):
            raise RuntimeError(line.strip())

    def teardown(self, n: int, method: str) -> None:
        self.session.close()


class Aluminium:
    """FIRE 1.0 and Polak-Ribiere on the aluminium packing."""

    params = ["fire", "cg"]
    param_names = ["method"]
    timeout = 600

    def setup(self, method: str) -> None:
        if not Path(_driver()).is_file():
            raise RuntimeError("relax driver is not built")
        self.session = _Session([_driver(), "al", method, "-", "session"])

    def time_relax(self, method: str) -> None:
        line = self.session.run().split()
        if line[0] != "1":
            raise RuntimeError("aluminium relaxation did not converge")

    def teardown(self, method: str) -> None:
        self.session.close()
