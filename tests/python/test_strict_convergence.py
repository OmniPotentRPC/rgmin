# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Strict convergence through ``minimize``. The default still returns."""

from __future__ import annotations

import numpy as np
import pytest

import rgmin


def quad(x: np.ndarray) -> tuple[float, np.ndarray]:
    value = 0.5 * float(x[0]) * float(x[0])
    return value, np.array([float(x[0])])


def test_a_met_tolerance_stays_ok() -> None:
    for strict in (False, True):
        report, _ = rgmin.minimize(
            quad,
            np.array([0.0]),
            "sd",
            maxiter=0,
            gtol=1e-8,
            require_convergence=strict,
        )
        assert report.stop == rgmin.Stop.TOLERANCE
        assert report.converged() is True


def test_budget_and_noise_error_only_when_strict() -> None:
    cases = [
        (np.array([4.0]), {"noise_eps": 0.0}, rgmin.Stop.BUDGET),
        (np.array([0.5]), {"noise_eps": 1.0}, rgmin.Stop.NOISE_FLOOR),
    ]
    for start, extra, stop in cases:
        report, _ = rgmin.minimize(
            quad, start, "sd", maxiter=0, gtol=1e-8, require_convergence=False, **extra
        )
        assert report.stop == stop
        assert report.converged() is False
        with pytest.raises(rgmin.NotConverged, match="not converged") as caught:
            rgmin.minimize(
                quad, start.copy(), "sd", maxiter=0, gtol=1e-8, require_convergence=True, **extra
            )
        assert caught.value.report.stop == stop
        assert caught.value.report.grad_norm > 1e-8


def test_both_flags_raise_not_converged() -> None:
    config = rgmin.SolverConfig("sd", require_convergence=True, error_on_unmet=True)
    assert config.require_convergence() is True
    assert config.error_on_unmet() is True
    with pytest.raises(rgmin.NotConverged) as caught:
        rgmin.minimize(
            quad,
            np.array([4.0]),
            "sd",
            maxiter=0,
            gtol=1e-8,
            require_convergence=True,
            error_on_unmet=True,
        )
    assert caught.value.report.stop == rgmin.Stop.BUDGET
    assert type(caught.value) is rgmin.NotConverged


def test_an_unmoved_step_is_step_tolerance_only_when_strict() -> None:
    def uphill(x: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.5 * float(x[0]) * float(x[0]), np.array([-float(x[0])])

    report, point = rgmin.minimize(
        uphill, np.array([1.0]), "sd", maxiter=4, gtol=1e-8, require_convergence=False
    )
    assert report.stop == rgmin.Stop.BUDGET
    assert point[0] == pytest.approx(1.0)
    with pytest.raises(rgmin.NotConverged) as caught:
        rgmin.minimize(
            uphill, np.array([1.0]), "sd", maxiter=4, gtol=1e-8, require_convergence=True
        )
    assert caught.value.report.stop == rgmin.Stop.STEP_TOLERANCE
    assert caught.value.report.grad_norm > 1e-8
