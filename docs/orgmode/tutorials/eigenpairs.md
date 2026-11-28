

# Lowest eigenpairs

`lowest_eigenpairs` is block LOBPCG for the k lowest pairs of a symmetric product (Knyazev 2001). The caller supplies `H(x) v`. The seed block has shape `(n, k)`. Values come back ascending, and the vectors are the columns of an `(n, k)` array, with the residual norm and a converged flag for each pair. A pair that misses the tolerance raises `RuntimeError` unless `strict=False`.

The listing is `examples/tutorial_lowest_eigenpairs.py`, whole. Continuous integration runs it, so it is the program as it runs.

`lowest_eigenpairs` is in the maturin package under `python/`, which is not on PyPI. Install it with `python -m pip install './python'` from a checkout, as shows. The PyPI wheel does not have it.

```python
"""Lowest Hessian pairs from a product. Continuous integration runs this file."""

import numpy as np

from rgmin import lowest_eigenpairs

lam = np.array([1.0, -2.0, 4.0])


def hvp(_x: np.ndarray, vector: np.ndarray) -> np.ndarray:
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
```
