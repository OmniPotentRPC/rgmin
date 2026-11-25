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


@pytest.mark.parametrize("value", [True, False])
def test_require_convergence_takes_a_bool(value: bool) -> None:
    assert rgmin.SolverConfig("sd", require_convergence=value).require_convergence() is value


@pytest.mark.parametrize("value", [0, 1, 2, -1, 0.0, 1.0, float("nan"), "true", "1"])
def test_require_convergence_refuses_an_int_a_float_or_a_string(value: object) -> None:
    with pytest.raises(ValueError, match="require_convergence must be a bool"):
        rgmin.SolverConfig("sd", require_convergence=value)
    with pytest.raises(ValueError, match="require_convergence must be a bool"):
        rgmin.minimize(quad, np.array([4.0]), "sd", maxiter=0, require_convergence=value)


@pytest.mark.parametrize("value", [np.bool_(True), np.int64(1), None, [True]])
def test_require_convergence_refuses_other_objects(value: object) -> None:
    with pytest.raises(TypeError, match="require_convergence"):
        rgmin.SolverConfig("sd", require_convergence=value)


@pytest.mark.parametrize("key", ["error_on_unmet", "backward_polish", "periodic"])
def test_every_flag_refuses_an_int(key: str) -> None:
    with pytest.raises(ValueError, match=f"{key} must be a bool"):
        rgmin.SolverConfig("sd", **{key: 1})


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


def rosenbrock(x: np.ndarray) -> tuple[float, np.ndarray]:
    a, b = float(x[0]), float(x[1])
    value = (1.0 - a) ** 2 + 100.0 * (b - a * a) ** 2
    grad = np.array([-2.0 * (1.0 - a) - 400.0 * a * (b - a * a), 200.0 * (b - a * a)])
    return value, grad


@pytest.mark.parametrize("method", ["sr1", "sr2", "ode12r"])
def test_many_step_runs_converge_identically_in_strict_mode(method: str) -> None:
    runs = []
    for strict in (False, True):
        report, point = rgmin.minimize(
            rosenbrock,
            np.array([0.5, 0.5]),
            method,
            maxiter=3000,
            gtol=1e-4,
            require_convergence=strict,
        )
        runs.append((report, point))
    (lax, lax_point), (tight, tight_point) = runs
    assert lax.stop == rgmin.Stop.TOLERANCE
    assert tight.stop == rgmin.Stop.TOLERANCE
    assert lax.steps > 1
    assert lax.steps == tight.steps
    assert lax.converged() and tight.converged()
    assert lax_point.tobytes() == tight_point.tobytes()


def test_a_gradient_exactly_at_gtol_is_not_converged() -> None:
    def unit(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(x[0]), np.array([1.0])

    report, _ = rgmin.minimize(unit, np.array([0.0]), "sd", maxiter=0, gtol=1.0)
    assert report.grad_norm == 1.0
    assert report.stop == rgmin.Stop.BUDGET
    assert report.converged() is False
    with pytest.raises(rgmin.NotConverged) as caught:
        rgmin.minimize(
            unit, np.array([0.0]), "sd", maxiter=0, gtol=1.0, require_convergence=True
        )
    assert caught.value.report.stop == rgmin.Stop.BUDGET

    below = np.nextafter(1.0, 0.0)

    def under(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(x[0]), np.array([below])

    report, _ = rgmin.minimize(
        under, np.array([0.0]), "sd", maxiter=0, gtol=1.0, require_convergence=True
    )
    assert report.stop == rgmin.Stop.TOLERANCE
    assert report.converged() is True


def test_not_converged_report_is_a_read_only_property() -> None:
    assert isinstance(rgmin.NotConverged.__dict__["report"], property)
    with pytest.raises(rgmin.NotConverged) as caught:
        rgmin.minimize(
            quad, np.array([4.0]), "sd", maxiter=0, gtol=1e-8, require_convergence=True
        )
    error = caught.value
    assert isinstance(error, RuntimeError)
    assert isinstance(error.report, rgmin.Report)
    assert error.report is error.report
    with pytest.raises(AttributeError):
        error.report = error.report  # type: ignore[misc]


def test_a_bare_not_converged_has_no_report() -> None:
    error = rgmin.NotConverged("not converged")
    assert str(error) == "not converged"
    with pytest.raises(AttributeError, match="report"):
        _ = error.report


def _level_set_retract_oracle(drift: bool):
    saw_trial = False
    origin_after_trial = 0

    def oracle(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal saw_trial, origin_after_trial
        x0 = float(x[0])
        if drift and origin_after_trial >= 1:
            residual = x0 - 0.25
            slope = 0.25 if abs(residual) <= float(np.finfo(float).eps) else 1.0
            return residual, np.array([slope])
        if x0 != 0.0:
            saw_trial = True
        elif saw_trial:
            origin_after_trial += 1
        return 0.0, np.array([1.0])

    return oracle


def test_level_set_retraction_converged_satisfies_require_convergence() -> None:
    assert "level_set" in rgmin.SolverConfig.valid_options()

    report, point = rgmin.minimize(
        _level_set_retract_oracle(True),
        np.array([0.0]),
        "lbfgs",
        maxiter=1,
        gtol=0.5,
        linesearch="wolfe",
        level_set=0.0,
        require_convergence=True,
    )
    assert point[0] == pytest.approx(0.25)
    assert report.stop == rgmin.Stop.TOLERANCE
    assert report.grad_norm < 0.5
    assert report.converged() is True

    with pytest.raises(rgmin.NotConverged) as caught:
        rgmin.minimize(
            _level_set_retract_oracle(True),
            np.array([0.0]),
            "lbfgs",
            maxiter=1,
            gtol=0.1,
            linesearch="wolfe",
            level_set=0.0,
            require_convergence=True,
        )
    assert caught.value.report.stop == rgmin.Stop.BUDGET
    assert caught.value.report.converged() is False

    with pytest.raises(rgmin.NotConverged) as caught:
        rgmin.minimize(
            _level_set_retract_oracle(False),
            np.array([0.0]),
            "lbfgs",
            maxiter=1,
            gtol=0.5,
            linesearch="wolfe",
            level_set=0.0,
            require_convergence=True,
        )
    assert caught.value.report.stop == rgmin.Stop.LINE_SEARCH_ROUNDING
    assert caught.value.report.converged() is False
