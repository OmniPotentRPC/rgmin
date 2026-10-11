# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""T4: array validation, option parsing, and the F7 mapping."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import rgmin

_finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64)


@settings(max_examples=20, deadline=None)
@given(values=_finite, length=st.integers(min_value=1, max_value=8))
def test_shape_mismatch(values: float, length: int) -> None:
    solver = rgmin.Solver(length, rgmin.SolverConfig("sd", maxiter=1))

    def fg(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(values), np.full(length, values)

    with pytest.raises(ValueError, match="expected shape"):
        solver.step(fg, np.full(length + 1, values))


@settings(max_examples=15, deadline=None)
@given(x=st.lists(_finite, min_size=1, max_size=6))
def test_signed_zero_round_trip(x: list[float]) -> None:
    array = np.array(x, dtype=np.float64)
    array[0] = -0.0

    def fg(point: np.ndarray) -> tuple[float, np.ndarray]:
        assert np.signbit(point[0])
        return 0.0, np.zeros_like(point)

    report = rgmin.Solver(array.shape[0], rgmin.SolverConfig("sd", maxiter=1, gtol=0.0)).step(fg, array)
    assert np.isfinite(report.value)


def test_subnormal_gradient_is_accepted() -> None:
    tiny = np.ldexp(1.0, -1074)

    def fg(x: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.0, np.full_like(x, tiny)

    x = np.zeros(3)
    report = rgmin.minimize(fg, x, method="sd", maxiter=1, gtol=0.0)
    assert np.isfinite(report.grad_norm)


def test_nan_gradient_is_numerical() -> None:
    def fg(x: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(x)
        grad[0] = np.nan
        return 0.0, grad

    with pytest.raises(rgmin.NumericalError):
        rgmin.minimize(fg, np.zeros(2), method="sd", maxiter=1)


@settings(max_examples=20, deadline=None)
@given(
    memory=st.integers(min_value=0, max_value=12),
    gtol=st.floats(min_value=1e-14, max_value=1e-2, allow_nan=False, allow_infinity=False),
)
def test_config_accepts_finite_options(memory: int, gtol: float) -> None:
    config = rgmin.SolverConfig("lbfgs", memory=memory, gtol=gtol, neb_guards=False)
    assert config.method == rgmin.Method.Lbfgs


def test_nonfinite_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        rgmin.SolverConfig("lbfgs", gtol=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        rgmin.SolverConfig("lbfgs", gtol=float("nan"))
