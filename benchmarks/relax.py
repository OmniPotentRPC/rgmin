"""asv rows for the atomistic relaxations.

``time_relax`` is one relaxation. ``track_calls`` and ``track_energy``
are the call count and the final energy of that same run. The committed
results are the 20-round comparison, not an ``asv run`` sample.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SYSTEMS = ["lj13", "lj38", "lj75", "pt7", "cu", "al"]
METHODS = ["lbfgs", "fire2"]

_DRIVER = Path(__file__).resolve().parent / "relax_driver"
_CON = Path(__file__).resolve().parent.parent / "bench" / "atomistic" / "data" / "pt_heptamer.con"


def _driver() -> str:
    named = os.environ.get("RELAX_DRIVER")
    if named:
        return named
    env = os.environ.get("ASV_ENV_DIR")
    if env:
        return str(Path(env) / "bin" / "relax_driver")
    return str(_DRIVER)


def _once(system: str, method: str) -> tuple[float, int, float]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [_driver(), system, method, str(_CON), "1"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    line = proc.stdout.strip().splitlines()[-1].split()
    calls = int(line[2])
    energy = float(line[3])
    wall = float(line[5])
    return wall, calls, energy


class Atomistic:
    """L-BFGS and FIRE 2.0 on the six atomistic systems."""

    params = (SYSTEMS, METHODS)
    param_names = ["system", "method"]
    timeout = 600

    def setup(self, system: str, method: str) -> None:
        if not Path(_driver()).is_file():
            raise RuntimeError("relax driver is not built")

    def time_relax(self, system: str, method: str) -> None:
        _once(system, method)

    def track_calls(self, system: str, method: str) -> int:
        return _once(system, method)[1]

    def track_energy(self, system: str, method: str) -> float:
        return _once(system, method)[2]
