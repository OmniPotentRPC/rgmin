# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Stop status and the first-order floor through the Python binding."""

from __future__ import annotations

import numpy as np
import pytest

import rgmin


def _rose(x: np.ndarray) -> tuple[float, np.ndarray]:
    a, b = float(x[0]), float(x[1])
    value = (1.0 - a) ** 2 + 100.0 * (b - a * a) ** 2
    grad = np.array(
        [
            -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
            200.0 * (b - a * a),
        ]
    )
    return value, grad


def test_a_short_run_stays_ok_and_is_not_converged() -> None:
    report, _ = rgmin.minimize(_rose, np.array([-1.2, 1.0]), "lbfgs", maxiter=6, gtol=1e-12)
    assert report.stop == rgmin.Stop.BUDGET
    assert report.converged() is False
    assert report.steps == 6


def test_error_on_unmet_raises_and_names_the_gradient() -> None:
    with pytest.raises(RuntimeError, match="stop unmet"):
        rgmin.minimize(
            _rose,
            np.array([-1.2, 1.0]),
            "lbfgs",
            maxiter=6,
            gtol=1e-12,
            error_on_unmet=True,
        )


def test_backward_error_floor_matches_the_first_order_scale() -> None:
    x = np.array([1.5, -2.0, 0.25])
    value = -3.0
    eps = np.finfo(np.float64).eps
    expect = x.size * eps * (1.0 + abs(value) + np.abs(x).sum())
    assert rgmin.backward_error_floor(value, x) == pytest.approx(expect, rel=0.0, abs=0.0)


def test_backward_polish_is_a_known_flag() -> None:
    config = rgmin.SolverConfig("lbfgs", backward_polish=True, error_on_unmet=False)
    assert config.backward_polish() is True
    assert config.error_on_unmet() is False
