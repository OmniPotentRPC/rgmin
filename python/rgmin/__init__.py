# MIT License
# Copyright 2026 Rohit Goswami
"""ASE-shaped optimizers over the rgmin session.

Importing this module does not import ASE. The optimizer classes do,
on construction, because they are :class:`ase.optimize.optimize.Optimizer`
subclasses: ``trajectory``, ``logfile``, ``attach`` and ``run(fmax, steps)``
are ASE's.
"""

from __future__ import annotations

import numpy as np

from rgmin._core import LbfgsPairs, LineSearch, Session, lowest_eigenpairs as _lowest_eigenpairs, version

__all__ = [
    "BFGS",
    "FIRE",
    "LBFGS",
    "LbfgsPairs",
    "LineSearch",
    "Session",
    "__version__",
    "lowest_eigenpairs",
]
__version__ = version


def lowest_eigenpairs(
    hvp, x0_block, *, x=None, tol=1e-8, max_iter=None, precond=None, strict=True
):
    """Lowest eigenpairs of a symmetric product.

    ``hvp(x, v)`` returns ``H(x) v`` without forming ``H``.
    ``x0_block`` has shape ``(n, k)``, one column per starting vector.
    Block LOBPCG (Knyazev 2001) returns the values ascending, the
    vectors as the columns of an ``(n, k)`` array, the residual norms
    ``||H v - lambda v||``, and a boolean converged flag per pair.
    ``x`` is the point where the product is taken; omitted, it is the origin.

    ``precond(block)`` applies ``T`` to an ``(n, m)`` block. Omitted,
    ``T`` is the identity. The default tolerance is ``1e-8``.
    ``max_iter`` defaults to ``max(n, 8)`` block expansions.
    ``strict=True`` raises ``RuntimeError`` when a pair misses the
    tolerance. ``strict=False`` returns the per-pair flags instead.
    ``k = 0``, ``k`` above ``n``, and a non-finite value or product
    are ``ValueError``.
    """
    block = np.asarray(x0_block, dtype=np.float64)
    if block.ndim != 2:
        raise ValueError("x0_block must have shape (n, k)")
    point = None if x is None else np.asarray(x, dtype=np.float64).reshape(-1)

    def product(point, vector):
        out = hvp(np.asarray(point, dtype=np.float64), np.asarray(vector, dtype=np.float64))
        return np.asarray(out, dtype=np.float64).reshape(-1).tolist()

    def apply_t(rows):
        incoming = np.asarray(rows, dtype=np.float64)
        out = np.asarray(precond(incoming), dtype=np.float64)
        if out.shape != incoming.shape:
            raise ValueError("preconditioner must return shape (n, m)")
        return out.reshape(incoming.shape).tolist()

    values, columns, residuals, converged = _lowest_eigenpairs(
        product,
        block.tolist(),
        None if point is None else point.tolist(),
        tol=tol,
        max_iter=max_iter,
        precond=None if precond is None else apply_t,
    )
    values = np.asarray(values, dtype=np.float64)
    vectors = np.asarray(columns, dtype=np.float64).T
    residuals = np.asarray(residuals, dtype=np.float64)
    converged = np.asarray(converged, dtype=bool)
    if strict and not bool(np.all(converged)):
        raise RuntimeError("lowest eigenpairs did not meet the residual tolerance")
    return values, vectors, residuals, converged


def __getattr__(name: str):
    if name in {"BFGS", "FIRE", "LBFGS"}:
        from rgmin import optimize

        return getattr(optimize, name)
    raise AttributeError(f"module 'rgmin' has no attribute {name}")
