# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""T5: SHA-256 of trajectories at several thread counts.

The Pt7 golden pins ASE's EMT, which is NumPy-dispatched. It is not a
pin of rgmin. rgmin's own results must be bit-identical across ISAs
with no environment variable. That check is the diagonal quadratic.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

# One hash per fixture. 3.14 and 3.14t both compare against these values.
# Lennard-Jones does not need a NumPy kernel pin.
LJ38 = "51e46a7c111203bb0a2f95894c33324039b6a8452bc118dbfa2c9026a4cf8e19"
# ASE EMT. Compared only in a process that was started with the pin below.
PT7_EMT = "72081cdfbe07a773fbdf1c2218cb141ce0c51a35bfbf2741260c42d7406ad5dc"
# rgmin Solver, pure-Python diagonal quadratic, no NumPy pin.
RGMIN_QUADRATIC = "b5e8baa0803a9c0b1ed7de205a0dde882e042a6c299adab0dd042b6d408d7720"

# Dispatch-target names. NPY_DISABLE_CPU_FEATURES ignores CPUID bit names
# such as AVX512F, and it is read only when NumPy is first imported.
_EMT_DISABLE = (
    "X86_V3,X86_V4,AVX512_SKX,AVX512_CLX,AVX512_CNL,"
    "AVX512_ICL,AVX512_SPR,AVX512_KNL,AVX512_KNM"
)
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


def _child_env(pin_emt: bool, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    root = os.path.dirname(__file__)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    if pin_emt:
        env["NPY_DISABLE_CPU_FEATURES"] = _EMT_DISABLE
    else:
        env.pop("NPY_DISABLE_CPU_FEATURES", None)
    env.update(extra)
    return env


def _run_child(code: str, *, pin_emt: bool, **extra: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=_child_env(pin_emt, **extra),
        cwd=os.path.dirname(__file__),
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


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
            pin_emt=False,
            RAYON_NUM_THREADS=threads,
            OMP_NUM_THREADS=threads,
        )
        assert lines == [LJ38]


def test_pt7_emt_golden_pins_numpy_not_rgmin() -> None:
    """Hash Pt7 EMT with the AVX-512 and X86_V3/V4 paths disabled.

    The variable is set on the child process, before NumPy is imported.
    A conftest assignment is too late once a plugin has imported NumPy.
    The child asserts ``show_runtime`` and ``__cpu_features__``. This
    pins ASE's EMT, which is NumPy-dispatched, and not rgmin.
    """
    pytest.importorskip("ase")
    lines = _run_child(_PT7_THREADS, pin_emt=True)
    assert lines == [PT7_EMT, PT7_EMT, PT7_EMT]
    for threads in ("1", "2", "3", "8"):
        hashed = _run_child(
            _PT7_ONCE,
            pin_emt=True,
            RAYON_NUM_THREADS=threads,
            OMP_NUM_THREADS=threads,
        )
        assert hashed == [PT7_EMT]


def test_rgmin_quadratic_golden() -> None:
    """rgmin's own potential matches on every runner with no kernel pin.

    Eight L-BFGS steps on a diagonal quadratic. The objective uses
    Python arithmetic. The start frame is part of the hash.
    """
    import rgmin

    x = np.array(_QUADRATIC_X0, dtype=np.float64)
    solver = rgmin.Solver(x.size, rgmin.SolverConfig("lbfgs", memory=6, gtol=0.0, maxiter=1))
    frames = [x.tobytes()]
    for _ in range(8):
        solver.step(_diagonal, x)
        frames.append(x.tobytes())
    assert hashlib.sha256(b"".join(frames)).hexdigest() == RGMIN_QUADRATIC
