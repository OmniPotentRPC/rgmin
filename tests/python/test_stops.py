# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""``etol`` and ``max_evals``: the LAMMPS ``minimize etol`` and ``maxeval`` stops."""

from __future__ import annotations

import math

import numpy as np
import pytest

import rgmin

_STIFFNESS = np.array([1.0, 4.0, 16.0, 40.0])


def _quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
    gradient = _STIFFNESS * x
    return 0.5 * float(x @ gradient), gradient


def _rosenbrock(x: np.ndarray) -> tuple[float, np.ndarray]:
    a, b = x
    value = (1.0 - a) ** 2 + 100.0 * (b - a * a) ** 2
    gradient = np.array([-2.0 * (1.0 - a) - 400.0 * a * (b - a * a), 200.0 * (b - a * a)])
    return value, gradient


def _tally(function):
    calls = [0]

    def counted(x: np.ndarray) -> tuple[float, np.ndarray]:
        calls[0] += 1
        return function(x)

    return counted, calls


def test_the_new_options_are_listed() -> None:
    options = rgmin.SolverConfig.valid_options()
    assert "etol" in options
    assert "max_evals" in options


def test_new_stops_have_stop_members() -> None:
    assert rgmin.Stop.ENERGY_TOLERANCE == rgmin.Stop.EnergyTolerance
    assert rgmin.Stop.EVAL_BUDGET == rgmin.Stop.EvalBudget
    assert int(rgmin.Stop.ENERGY_TOLERANCE) == 6
    assert int(rgmin.Stop.EVAL_BUDGET) == 7


@pytest.mark.parametrize("method", ["lbfgs", "cg", "sd", "bfgs", "fire", "bb", "quickmin"])
def test_etol_ends_a_run_before_gtol(method: str) -> None:
    x0 = np.array([1.0, 1.0, 1.0, 1.0])
    kwargs = {"maxiter": 5000, "gtol": 1e-13, "istep": 0.05}
    plain, _ = rgmin.minimize(_quadratic, x0.copy(), method, **kwargs)
    cut, x = rgmin.minimize(_quadratic, x0.copy(), method, etol=1e-6, **kwargs)
    assert cut.stop == rgmin.Stop.ENERGY_TOLERANCE
    assert plain.stop != rgmin.Stop.ENERGY_TOLERANCE
    assert cut.steps < plain.steps
    assert cut.value >= plain.value
    assert cut.value == pytest.approx(_quadratic(x)[0])


def test_etol_zero_never_stops_a_moving_run() -> None:
    report, _ = rgmin.minimize(
        _rosenbrock, np.array([-1.2, 1.0]), "lbfgs", maxiter=40, gtol=1e-13, etol=0.0
    )
    assert report.stop != rgmin.Stop.ENERGY_TOLERANCE


@pytest.mark.parametrize("etol", [-1.0, math.nan, math.inf, -math.inf])
def test_invalid_etol_is_a_value_error(etol: float) -> None:
    with pytest.raises(ValueError, match="etol"):
        rgmin.SolverConfig("lbfgs", etol=etol)
    with pytest.raises(ValueError, match="etol"):
        rgmin.minimize(_quadratic, np.ones(4), "lbfgs", etol=etol)


@pytest.mark.parametrize("budget", [0, -3])
def test_invalid_max_evals_is_a_value_error(budget: int) -> None:
    with pytest.raises(ValueError, match="max_evals"):
        rgmin.SolverConfig("lbfgs", max_evals=budget)


def test_max_evals_must_be_an_int() -> None:
    with pytest.raises(ValueError, match="max_evals"):
        rgmin.SolverConfig("lbfgs", max_evals=2.5)


def test_particle_swarm_refuses_etol_and_accepts_a_budget() -> None:
    with pytest.raises(ValueError, match="particle swarm"):
        rgmin.Solver(2, rgmin.SolverConfig("pso", etol=1e-3))
    rgmin.Solver(2, rgmin.SolverConfig("pso", max_evals=10))


@pytest.mark.parametrize("method", ["lbfgs", "cg", "sd", "fire"])
def test_max_evals_stops_with_the_count_the_callback_saw(method: str) -> None:
    function, calls = _tally(_rosenbrock)
    config = rgmin.SolverConfig(method, maxiter=5000, gtol=1e-13, max_evals=25)
    solver = rgmin.Solver(2, config)
    x = np.array([-1.2, 1.0])
    report = None
    for _ in range(5000):
        report = solver.step(function, x)
        if report.stop != rgmin.Stop.BUDGET:
            break
    assert report is not None
    assert report.stop == rgmin.Stop.EVAL_BUDGET
    assert solver.evals == calls[0]
    assert solver.evals >= 25


def test_max_evals_through_minimize() -> None:
    function, calls = _tally(_rosenbrock)
    report, _ = rgmin.minimize(
        function, np.array([-1.2, 1.0]), "bfgs", maxiter=5000, gtol=1e-13, max_evals=12
    )
    assert report.stop == rgmin.Stop.EVAL_BUDGET
    assert calls[0] >= 12


def test_a_stop_that_never_fires_leaves_a_run_unchanged() -> None:
    kwargs = {"maxiter": 200, "gtol": 1e-10}
    a, xa = rgmin.minimize(_rosenbrock, np.array([-1.2, 1.0]), "lbfgs", **kwargs)
    b, xb = rgmin.minimize(
        _rosenbrock, np.array([-1.2, 1.0]), "lbfgs", etol=0.0, max_evals=10**9, **kwargs
    )
    assert a.stop == b.stop
    assert a.steps == b.steps
    assert a.value == b.value
    np.testing.assert_array_equal(xa, xb)
