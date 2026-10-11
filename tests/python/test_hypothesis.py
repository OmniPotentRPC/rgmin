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
    report, _out = rgmin.minimize(fg, x, method="sd", maxiter=1, gtol=0.0)
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


@settings(max_examples=15, deadline=None)
@given(
    dtype=st.sampled_from([np.float32, np.int64, np.complex128]),
    order=st.sampled_from(["F", "C"]),
)
def test_dtype_and_order_are_type_errors(dtype: np.dtype, order: str) -> None:
    solver = rgmin.Solver(4, rgmin.SolverConfig("sd", maxiter=1))
    values = np.ones(4, dtype=np.float64)
    array = np.array(values, dtype=dtype, order=order)
    if array.dtype == np.float64 and array.flags.c_contiguous:
        array = np.asfortranarray(array)
    with pytest.raises(TypeError):
        solver.step(lambda x: (0.0, np.zeros_like(x)), array)


@settings(max_examples=10, deadline=None)
@given(index=st.integers(min_value=0, max_value=3))
def test_nan_in_x_reaches_the_callback(index: int) -> None:
    """A NaN coordinate is accepted and the callback still runs.

    The wide-box oracle clips with ``f64::max`` before the first
    evaluation, and ``NaN.max(low)`` is ``low`` (``-1e12``). The binding
    copies the buffer unchanged; the clip is the solver.
    """
    seen: list[np.ndarray] = []

    def fg(x: np.ndarray) -> tuple[float, np.ndarray]:
        seen.append(np.array(x, copy=True))
        return 0.0, np.zeros_like(x)

    x = np.ones(4)
    x[index] = np.nan
    solver = rgmin.Solver(4, rgmin.SolverConfig("sd", maxiter=1, gtol=0.0))
    solver.step(fg, x)
    assert seen
    assert seen[0][index] == -1e12
    assert np.all(np.isfinite(seen[0]))


@settings(max_examples=15, deadline=None)
@given(
    memory=st.integers(min_value=0, max_value=8),
    maxiter=st.integers(min_value=0, max_value=6),
    gtol=st.floats(min_value=1e-12, max_value=1e-3, allow_nan=False, allow_infinity=False),
    method=st.sampled_from(["lbfgs", "sd", "fire", "cg"]),
)
def test_solver_config_round_trip(memory: int, maxiter: int, gtol: float, method: str) -> None:
    config = rgmin.SolverConfig(method, memory=memory, maxiter=maxiter, gtol=gtol)
    again = rgmin.SolverConfig(config.method_name, memory=memory, maxiter=maxiter, gtol=gtol)
    assert again.method == config.method
    assert again.method_name == config.method_name
    solver = rgmin.Solver(2, again)
    assert solver.method == config.method


def test_nonfinite_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        rgmin.SolverConfig("lbfgs", gtol=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        rgmin.SolverConfig("lbfgs", gtol=float("nan"))
