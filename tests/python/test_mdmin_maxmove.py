# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""An explicit positive ``maxmove`` caps every MDMin step.

The first MDMin kick on a constant force of 100 with ``istep=0.2`` is
``0.2 * 0.5 * 0.2 * 100 = 2.0``. The tests/mdmin_session.rs tests take
the same step through the Rust session.
"""

from __future__ import annotations

import numpy as np

import rgmin

FORCE = np.array([100.0, 0.0, 0.0])


def _push(x: np.ndarray) -> tuple[float, np.ndarray]:
    return 1.0, -FORCE.copy()


def _first_step(**options: float) -> np.ndarray:
    x = np.zeros(3)
    config = rgmin.SolverConfig("mdmin", istep=0.2, gtol=0.0, maxiter=2, **options)
    rgmin.Solver(x.size, config).step(_push, x)
    return x


def test_explicit_maxmove_caps_the_mdmin_step() -> None:
    moved = float(np.linalg.norm(_first_step(maxmove=0.05)))
    assert abs(moved - 0.05) < 1e-12, moved


def test_explicit_maxmove_binds_below_an_explicit_atom_cap() -> None:
    moved = float(np.linalg.norm(_first_step(maxmove=0.05, atom_maxmove=1.0)))
    assert abs(moved - 0.05) < 1e-12, moved


def test_negative_maxmove_leaves_the_mdmin_step_unscaled() -> None:
    moved = float(np.linalg.norm(_first_step(maxmove=-1.0)))
    assert abs(moved - 2.0) < 1e-12, moved
