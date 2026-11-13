"""The quadratic from (1, -2) ends below 1e-8."""

import numpy as np

from rgmin import Lbfgs, minimize_eon_cg


def fg(x):
    """Value and gradient of x^2 + y^2."""
    x = np.asarray(x, dtype=float)
    return float(x @ x), 2.0 * x


def test_lbfgs_quadratic():
    value, point, evals = Lbfgs().minimize(np.array([1.0, -2.0]), 40, fg)
    assert value < 1e-8
    assert abs(point[0]) < 1e-4 and abs(point[1]) < 1e-4
    assert evals >= 1


def test_short_gradient_does_not_crash():
    value, _point, evals = Lbfgs().minimize(
        np.array([1.0, -2.0]), 5, lambda _x: (1.0, np.array([1.0]))
    )
    assert evals == 1
    assert not np.isfinite(value)


def rosen(x):
    """Rosenbrock value and gradient."""
    x = np.asarray(x, dtype=float)
    a = 1.0 - x[0]
    b = x[1] - x[0] * x[0]
    return float(a * a + 100.0 * b * b), np.array(
        [-2.0 * a - 400.0 * x[0] * b, 200.0 * b]
    )


def test_eon_cg_rosenbrock_evaluation_count():
    energy, _grad, evals, converged, point = minimize_eon_cg(
        np.array([-1.2, 1.0]), rosen, gtol=1e-5, max_iter=200
    )
    assert converged
    assert evals == 87
    assert energy < 1e-8
    assert abs(point[0] - 1.0) < 1e-3
