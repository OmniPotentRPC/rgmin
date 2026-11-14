"""Lowest Hessian pairs from a product. Continuous integration runs this file."""

import numpy as np

from rgmin import lowest_eigenpairs

lam = np.array([1.0, -2.0, 4.0])


def hvp(_x, vector):
    return lam * np.asarray(vector, dtype=np.float64)


seed = np.zeros((3, 2))
seed[0, 0] = 1.0
seed[1, 0] = 0.2
seed[1, 1] = 0.2
seed[2, 1] = 1.0
values, vectors, residuals, converged = lowest_eigenpairs(hvp, seed, tol=1e-8, max_iter=20)
assert values.shape == (2,) and vectors.shape == (3, 2)
assert residuals.shape == (2,) and converged.all()
assert abs(values[0] + 2.0) < 1e-6
assert abs(values[1] - 1.0) < 1e-6
