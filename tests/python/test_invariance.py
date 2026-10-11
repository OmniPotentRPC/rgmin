# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""T5: thread-count invariance of a fixed-step trajectory."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import rgmin

_STEPS = 5
_X0 = np.random.default_rng(38).normal(scale=0.8, size=38 * 3)


def _lj(x: np.ndarray) -> tuple[float, np.ndarray]:
    pos = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    energy = 0.0
    grad = np.zeros_like(pos)
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            r = pos[i] - pos[j]
            d2 = float(r @ r)
            if d2 < 1e-24:
                d2 = 1e-24
            inv = 1.0 / d2
            inv3 = inv * inv * inv
            inv6 = inv3 * inv3
            energy += 4.0 * (inv6 - inv3)
            ded = 4.0 * (-6.0 * inv**7 + 3.0 * inv**4)
            g = (ded * 2.0) * r
            grad[i] += g
            grad[j] -= g
    return energy, grad.reshape(-1)


def _run_lj() -> bytes:
    x = np.array(_X0, dtype=np.float64, copy=True)
    report = rgmin.minimize(_lj, x, method="lbfgs", memory=10, maxiter=_STEPS, gtol=0.0)
    blob = x.tobytes() + np.array([report.value, report.grad_norm], dtype=np.float64).tobytes()
    return hashlib.sha256(blob).digest()


def test_python_thread_invariance() -> None:
    reference = _run_lj()
    for workers in (1, 2, 8):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            hashes = list(pool.map(lambda _i: _run_lj(), range(workers)))
        assert hashes == [reference] * workers


def test_rayon_env_invariance() -> None:
    script = (
        "import hashlib, numpy as np, rgmin\n"
        "from tests.python.test_invariance import _run_lj\n"
        "print(hashlib.sha256(_run_lj()).hexdigest())\n"
    )
    # Importing the helper needs the repo root on PYTHONPATH. The function
    # itself only needs numpy and rgmin, so inline the call via -c after
    # setting the path to this file's directory is unnecessary: spawn the
    # same interpreter with the package already importable.
    code = (
        "import hashlib, os, numpy as np\n"
        "import rgmin\n"
        "from test_invariance import _run_lj\n"
        "print(_run_lj().hex())\n"
    )
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = os.path.dirname(__file__) + os.pathsep + env_base.get("PYTHONPATH", "")
    reference: str | None = None
    for threads in ("1", "2", "3", "8"):
        env = dict(env_base)
        env["RAYON_NUM_THREADS"] = threads
        env["OMP_NUM_THREADS"] = threads
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            cwd=os.path.dirname(__file__),
        )
        digest = completed.stdout.strip()
        if reference is None:
            reference = digest
        assert digest == reference
