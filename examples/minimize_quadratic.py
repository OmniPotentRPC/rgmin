# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Minimize a convex quadratic with ``rgmin.minimize``."""

from __future__ import annotations

import numpy as np

import rgmin


def f_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    """``f(x) = 0.5 x·x``. The gradient is ``x``."""
    value = 0.5 * float(x @ x)
    return value, np.array(x, dtype=np.float64, copy=True)


def main() -> rgmin.Report:
    x0 = np.array([1.0, -2.0, 0.5, 0.25], dtype=np.float64)
    config = rgmin.SolverConfig("lbfgs", memory=20, gtol=1e-12, maxiter=50)
    solver = rgmin.Solver(x0.shape[0], config)
    report = solver.step(f_and_grad, x0)
    for _ in range(49):
        if report.stop != rgmin.Stop.Budget:
            break
        report = solver.step(f_and_grad, x0)
    direct, _x = rgmin.minimize(f_and_grad, np.array(x0, copy=True), method="sd", maxiter=50, gtol=1e-12)
    assert report.grad_norm < 1e-8
    assert direct.grad_norm < 1e-6
    return report


if __name__ == "__main__":
    main()
