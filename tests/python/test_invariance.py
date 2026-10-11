# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""T5: SHA-256 of rgmin.ase trajectories at several thread counts."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# Before NumPy. See conftest.py. The rayon subprocess starts from this file.
os.environ.setdefault(
    "NPY_DISABLE_CPU_FEATURES",
    "X86_V3,X86_V4,AVX512_ICL,AVX512_SPR",
)

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones

from rgmin.ase import LBFGS

# One hash per fixture. 3.14 and 3.14t both compare against these values.
GOLDEN = {
    "lj38": "51e46a7c111203bb0a2f95894c33324039b6a8452bc118dbfa2c9026a4cf8e19",
    "pt7": "72081cdfbe07a773fbdf1c2218cb141ce0c51a35bfbf2741260c42d7406ad5dc",
}

_STEPS = 5


def _lj38() -> Atoms:
    positions = np.random.default_rng(38).normal(scale=0.8, size=(38, 3))
    atoms = Atoms(numbers=[1] * 38, positions=positions)
    atoms.calc = LennardJones()
    return atoms


def _pt7() -> Atoms:
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
    atoms = _lj38() if kind == "lj38" else _pt7()
    frames: list[bytes] = []

    def grab() -> None:
        frames.append(np.array(atoms.get_positions(), dtype=np.float64).tobytes())

    opt = LBFGS(atoms, logfile=None, maxstep=0.1)
    opt.attach(grab, interval=1)
    opt.run(fmax=0.0, steps=_STEPS)
    return hashlib.sha256(b"".join(frames)).hexdigest()


@pytest.mark.thread_unsafe
def test_python_thread_invariance() -> None:
    for kind in ("lj38", "pt7"):
        reference = GOLDEN[kind]
        for workers in (1, 2, 8):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                hashes = list(pool.map(lambda _i: trajectory_sha256(kind), range(workers)))
            assert hashes == [reference] * workers


@pytest.mark.thread_unsafe
def test_rayon_env_invariance() -> None:
    code = (
        "from test_invariance import trajectory_sha256\n"
        "import sys\n"
        "print(trajectory_sha256(sys.argv[1]))\n"
    )
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = os.path.dirname(__file__) + os.pathsep + env_base.get("PYTHONPATH", "")
    for kind in ("lj38", "pt7"):
        for threads in ("1", "2", "3", "8"):
            env = dict(env_base)
            env["RAYON_NUM_THREADS"] = threads
            env["OMP_NUM_THREADS"] = threads
            completed = subprocess.run(
                [sys.executable, "-c", code, kind],
                check=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=os.path.dirname(__file__),
            )
            assert completed.stdout.strip() == GOLDEN[kind]
