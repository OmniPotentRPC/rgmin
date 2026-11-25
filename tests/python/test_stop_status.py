# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Stop status and the first-order floor through the Python binding."""

from __future__ import annotations

import numpy as np
import pytest

import rgmin


def test_rounding_stop_has_stable_names_and_value() -> None:
    assert rgmin.Stop.LINE_SEARCH_ROUNDING == 9
    assert rgmin.Stop.LineSearchRounding == rgmin.Stop.LINE_SEARCH_ROUNDING


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


def _uphill(x: np.ndarray) -> tuple[float, np.ndarray]:
    """The gradient points uphill, so no trial is accepted."""
    return 0.5 * float(x @ x), -np.array(x, dtype=np.float64, copy=True)


def test_a_no_move_lbfgs_run_is_line_search_like_the_c_one_shot() -> None:
    x0 = np.array([1.0])
    report, x = rgmin.minimize(
        _uphill, x0, "lbfgs", linesearch="wolfe", maxiter=3, gtol=1e-8
    )
    assert x[0] == 1.0
    assert report.stop == rgmin.Stop.LINE_SEARCH
    assert report.converged() is False
    assert report.steps == 3


def test_a_no_move_lbfgs_run_raises_when_the_stop_is_asked_for() -> None:
    with pytest.raises(RuntimeError, match="stop unmet"):
        rgmin.minimize(
            _uphill,
            np.array([1.0]),
            "lbfgs",
            linesearch="wolfe",
            maxiter=3,
            gtol=1e-8,
            error_on_unmet=True,
        )


def test_unmet_error_is_a_runtime_error_that_carries_the_report() -> None:
    kwargs = {"maxiter": 6, "gtol": 1e-12}
    spent, _ = rgmin.minimize(_rose, np.array([-1.2, 1.0]), "lbfgs", **kwargs)
    with pytest.raises(rgmin.StopUnmetError) as raised:
        rgmin.minimize(
            _rose, np.array([-1.2, 1.0]), "lbfgs", error_on_unmet=True, **kwargs
        )
    error = raised.value
    assert isinstance(error, RuntimeError)
    assert isinstance(error.report, rgmin.Report)
    assert error.report.stop == rgmin.Stop.BUDGET
    assert error.report.steps == spent.steps == 6
    assert error.report.value == spent.value
    assert error.report.grad_norm == spent.grad_norm
    assert error.report.grad_norm > 1e-12
    assert str(error).startswith("stop unmet after 6 steps")


def test_unmet_error_carries_the_line_search_stop_of_a_no_move_run() -> None:
    with pytest.raises(rgmin.StopUnmetError) as raised:
        rgmin.minimize(
            _uphill,
            np.array([1.0]),
            "lbfgs",
            linesearch="wolfe",
            maxiter=3,
            gtol=1e-8,
            error_on_unmet=True,
        )
    assert raised.value.report.stop == rgmin.Stop.LINE_SEARCH
    assert raised.value.report.value == 0.5
    assert raised.value.report.grad_norm == 1.0


def test_a_converged_run_does_not_raise_the_unmet_error() -> None:
    report, _ = rgmin.minimize(
        _rose, np.array([-1.2, 1.0]), "lbfgs", maxiter=500, gtol=1e-6, error_on_unmet=True
    )
    assert report.converged() is True


def test_a_run_that_moves_and_spends_its_budget_stays_budget() -> None:
    report, x = rgmin.minimize(
        _uphill, np.array([1.0]), "lbfgs", maxiter=3, gtol=1e-8
    )
    assert x[0] != 1.0
    assert report.stop == rgmin.Stop.BUDGET


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
