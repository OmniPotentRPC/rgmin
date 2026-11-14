# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""SHA-256 of trajectories at several thread counts.

The Pt7 golden pins ASE's EMT, which is NumPy-dispatched. It is not a
pin of rgmin. rgmin's own results must be bit-identical across ISAs
with no environment variable. That check is the diagonal quadratic.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

# One hash per fixture. 3.14 and 3.14t both compare against these values.
# Lennard-Jones does not need a NumPy kernel pin.
LJ38 = "e40ce10a3ff8a9622188ac2e08e139c7afddfdbc0fb34221b0cf71f06ee6afba"
# ASE EMT. Compared only in a process that was started with the pin below.
# Linux, macOS, and arm agree. Windows NumPy does not, on this trajectory.
PT7_EMT = "c10b0f1f1d1a12fe4c667b2afc253b318020965a03cb07150e010384c7bd50a9"
PT7_EMT_WINDOWS = "32ecfa5c86916033bd1b82dbd1059f3777ec6e386eb416f0b9c69b4df96074c8"
# rgmin Solver, pure-Python diagonal quadratic, no NumPy pin.
# The solver uses the default per-atom cap of 0.2.
RGMIN_QUADRATIC = "c133a0ce135758cf1c34cadfd8249b1351369c5886655f15b705784d4671964e"

# NumPy 2.2 still dispatches AVX512F and AVX512_ICL by those names.
# NumPy 2.5 dispatches X86_V3, X86_V4, and AVX512_ICL. A name that is
# not in this build's dispatch list makes the import warn, so the pin
# is the intersection, discovered before the golden process starts.
_STEPS = 5
_SCALES = (1.0, 4.0, 9.0, 16.0, 2.0, 3.0)
_QUADRATIC_X0 = (0.4, -0.2, 0.7, 0.1, -0.3, 0.5)

_PT7_THREADS = """
from test_invariance import assert_emt_numpy_dispatch_disabled, pt7_hashes
assert_emt_numpy_dispatch_disabled()
for workers in (1, 2, 8):
    hashes = pt7_hashes(workers)
    if len(set(hashes)) != 1:
        raise SystemExit(repr(hashes))
    print(hashes[0])
"""

_PT7_ONCE = """
from test_invariance import assert_emt_numpy_dispatch_disabled, trajectory_sha256
assert_emt_numpy_dispatch_disabled()
print(trajectory_sha256("pt7"))
"""

_LJ38_ONCE = """
from test_invariance import trajectory_sha256
print(trajectory_sha256("lj38"))
"""

_DISPATCH_PROBE = """
import json
from numpy._core._multiarray_umath import __cpu_dispatch__
print(json.dumps(list(__cpu_dispatch__)))
"""


def _blocked(name: str) -> bool:
    return name in {"X86_V3", "X86_V4"} or name.startswith("AVX512")


def assert_emt_numpy_dispatch_disabled() -> None:
    """Confirm this process disabled NumPy's AVX-512 and X86_V3/V4 paths.

    ``numpy.show_runtime`` prints the enabled dispatch targets and
    returns None. The same split comes from ``__cpu_features__``. CPUID
    bits that are not dispatch targets stay enabled. This pins ASE's
    EMT, which is NumPy-dispatched, and not rgmin.
    """
    import contextlib
    import io
    import re

    import numpy as np
    from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        shown = np.show_runtime()
    assert shown is None
    match = re.search(r"'found': \[(.*?)\]", buffer.getvalue(), re.DOTALL)
    assert match is not None
    found = re.findall(r"'([^']+)'", match.group(1))
    for name in found:
        assert not _blocked(name), name
    for name in __cpu_dispatch__:
        if _blocked(name):
            assert not __cpu_features__[name], name


def _lj38() -> object:
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.lj import LennardJones

    positions = np.random.default_rng(38).normal(scale=0.8, size=(38, 3))
    atoms = Atoms(numbers=[1] * 38, positions=positions)
    atoms.calc = LennardJones()
    return atoms


def _pt7() -> object:
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.emt import EMT

    a = 2.7
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [a, 0.0, 0.0],
            [-a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, -a, 0.0],
            [0.0, 0.0, a],
            [0.0, 0.0, -a],
        ]
    )
    atoms = Atoms("Pt7", positions=base + np.random.default_rng(7).normal(scale=0.02, size=base.shape))
    atoms.calc = EMT()
    return atoms


def trajectory_sha256(kind: str) -> str:
    from rgmin.ase import LBFGS

    atoms = _lj38() if kind == "lj38" else _pt7()
    frames: list[bytes] = []

    def grab() -> None:
        frames.append(np.array(atoms.get_positions(), dtype=np.float64).tobytes())

    opt = LBFGS(atoms, logfile=None, maxstep=0.1)
    opt.attach(grab, interval=1)
    opt.run(fmax=0.0, steps=_STEPS)
    return hashlib.sha256(b"".join(frames)).hexdigest()


def pt7_hashes(workers: int) -> list[str]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda _i: trajectory_sha256("pt7"), range(workers)))


def _diagonal(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Pure Python potential. No NumPy ufunc, so the hash is rgmin's."""
    coords = [float(value) for value in x]
    total = 0.0
    gradient: list[float] = []
    for scale, coord in zip(_SCALES, coords, strict=True):
        total += 0.5 * scale * coord * coord
        gradient.append(scale * coord)
    return total, np.array(gradient, dtype=np.float64)


def _child_env(pin: str | None, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    root = os.path.dirname(__file__)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("NPY_DISABLE_CPU_FEATURES", None)
    env.pop("PYTHONWARNINGS", None)
    if pin:
        env["NPY_DISABLE_CPU_FEATURES"] = pin
    env.update(extra)
    return env


def _run_child(code: str, *, pin: str | None, **extra: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=_child_env(pin, **extra),
        cwd=os.path.dirname(__file__),
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"child exited {completed.returncode}\n{completed.stdout}\n{completed.stderr}"
        )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _pt7_expect() -> str:
    if sys.platform == "win32":
        return PT7_EMT_WINDOWS
    return PT7_EMT


def _emt_pin() -> str | None:
    """Dispatch names for the AVX-512 and X86_V3/V4 paths, if this build has them."""
    names = json.loads(_run_child(_DISPATCH_PROBE, pin=None)[-1])
    blocked = [name for name in names if _blocked(name)]
    if not blocked:
        return None
    return ",".join(blocked)


# Spawns a ThreadPoolExecutor. The marker keeps pytest-run-parallel from
# nesting another pool around this test.
@pytest.mark.thread_unsafe
def test_python_thread_invariance() -> None:
    pytest.importorskip("ase")
    for workers in (1, 2, 8):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            hashes = list(pool.map(lambda _i: trajectory_sha256("lj38"), range(workers)))
        assert hashes == [LJ38] * workers


def test_lj38_rayon_env_invariance() -> None:
    """Lennard-Jones matches the golden with no NumPy pin."""
    pytest.importorskip("ase")
    for threads in ("1", "2", "3", "8"):
        lines = _run_child(
            _LJ38_ONCE,
            pin=None,
            RAYON_NUM_THREADS=threads,
            OMP_NUM_THREADS=threads,
        )
        assert lines == [LJ38]


def test_pt7_emt_golden_pins_numpy_not_rgmin() -> None:
    """Hash Pt7 EMT with the AVX-512 and X86_V3/V4 paths disabled.

    The variable is set on the child process, before NumPy is imported.
    A conftest assignment is too late once a plugin has imported NumPy.
    Only names from this build's ``__cpu_dispatch__`` are listed, and
    the child asserts ``show_runtime`` and ``__cpu_features__``. This
    pins ASE's EMT, which is NumPy-dispatched, and not rgmin.
    Windows NumPy takes a different value. The thread counts still agree.
    """
    pytest.importorskip("ase")
    pin = _emt_pin()
    expect = _pt7_expect()
    lines = _run_child(_PT7_THREADS, pin=pin)
    assert lines == [expect, expect, expect]
    for threads in ("1", "2", "3", "8"):
        hashed = _run_child(
            _PT7_ONCE,
            pin=pin,
            RAYON_NUM_THREADS=threads,
            OMP_NUM_THREADS=threads,
        )
        assert hashed == [expect]


def test_rgmin_quadratic_golden() -> None:
    """rgmin's own potential matches on every runner with no kernel pin.

    Eight L-BFGS steps on a diagonal quadratic. The objective uses
    Python arithmetic. The start frame is part of the hash. The
    solver takes the default per-atom cap.
    """
    import rgmin

    x = np.array(_QUADRATIC_X0, dtype=np.float64)
    solver = rgmin.Solver(x.size, rgmin.SolverConfig("lbfgs", memory=6, gtol=0.0, maxiter=1))
    frames = [x.tobytes()]
    for _ in range(8):
        solver.step(_diagonal, x)
        frames.append(x.tobytes())
    assert hashlib.sha256(b"".join(frames)).hexdigest() == RGMIN_QUADRATIC
