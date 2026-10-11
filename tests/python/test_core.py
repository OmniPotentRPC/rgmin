# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Solver, minimize, and option validation."""

from __future__ import annotations

import numpy as np
import pytest

import rgmin


def _quad(x: np.ndarray) -> tuple[float, np.ndarray]:
    return 0.5 * float(x @ x), np.array(x, dtype=np.float64, copy=True)


def test_minimize_quadratic() -> None:
    x = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    report = rgmin.minimize(_quad, x, method="lbfgs", gtol=1e-12, maxiter=30)
    assert report.grad_norm < 1e-8
    assert abs(report.value) < 1e-12
    assert report.stop == rgmin.Stop.Tolerance


def test_solver_config_round_trip() -> None:
    config = rgmin.SolverConfig("lbfgs", memory=20, maxstep=0.2)
    solver = rgmin.Solver(4, config)
    assert solver.dim == 4
    assert solver.method == rgmin.Method.Lbfgs
    assert config.method_name == "lbfgs"
    x = np.ones(4)
    report = solver.step(_quad, x)
    assert np.isfinite(report.value)


def test_unknown_option_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="valid options") as exc:
        rgmin.SolverConfig("lbfgs", not_an_option=1.0)
    message = str(exc.value)
    assert "memory" in message
    assert "maxstep" in message


def test_unknown_method_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="valid methods") as exc:
        rgmin.SolverConfig("no-such-method")
    assert "lbfgs" in str(exc.value)


def test_method_is_int_enum() -> None:
    assert issubclass(rgmin.Method, int)
    assert int(rgmin.Method.Lbfgs) == 3
    assert rgmin.Method.Fire == 17


def test_bad_rank_and_length() -> None:
    config = rgmin.SolverConfig("sd", maxiter=2)
    solver = rgmin.Solver(3, config)
    with pytest.raises(ValueError, match="rank 1"):
        solver.step(_quad, np.ones((3, 1)))
    with pytest.raises(ValueError, match="expected shape"):
        solver.step(_quad, np.ones(2))


def test_dtype_and_order() -> None:
    config = rgmin.SolverConfig("sd", maxiter=1)
    solver = rgmin.Solver(4, config)
    with pytest.raises(TypeError):
        solver.step(_quad, np.ones(4, dtype=np.float32))
    sliced = np.ones(8)[::2]
    assert not sliced.flags.c_contiguous
    with pytest.raises(TypeError):
        solver.step(_quad, sliced)


def test_nonfinite_objective() -> None:
    def bad(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float("nan"), np.zeros_like(x)

    x = np.zeros(2)
    with pytest.raises(rgmin.NumericalError):
        rgmin.minimize(bad, x, method="sd", maxiter=1)


def test_callback_exception_is_reraised() -> None:
    def boom(x: np.ndarray) -> tuple[float, np.ndarray]:
        raise KeyError("from-callback")

    x = np.zeros(2)
    with pytest.raises(KeyError, match="from-callback"):
        rgmin.minimize(boom, x, method="sd", maxiter=1)


def test_negative_zero_and_subnormal_are_finite() -> None:
    def fg(x: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.5 * float(x @ x), np.array(x, dtype=np.float64, copy=True)

    x = np.array([-0.0, np.ldexp(1.0, -1074)], dtype=np.float64)
    report = rgmin.minimize(fg, x, method="sd", maxiter=5, gtol=1e-16)
    assert np.isfinite(report.value)


def test_import_does_not_pull_ase() -> None:
    import subprocess
    import sys

    code = (
        "import sys, rgmin; "
        "assert 'ase' not in sys.modules; "
        "assert 'rgmin.ase' not in sys.modules; "
        "assert rgmin.__version__"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
