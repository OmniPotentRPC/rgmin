# MIT License
# Copyright 2026 Rohit Goswami
"""Geometry optimization through rgmin.

Importing this package does not import ASE. The ASE optimizer classes
are loaded on first use. The extension module is either the maturin
session binding or the nanobind solver binding; this module exposes
whichever symbols that build provides, including ``lowest_eigenpairs``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

import rgmin._core as _core

__version__ = getattr(_core, "__version__", None)
if __version__ is None and hasattr(_core, "version"):
    __version__ = _core.version

_EXPORTS = [
    "BFGS",
    "FIRE",
    "LBFGS",
    "LbfgsPairs",
    "LineSearch",
    "Method",
    "NumericalError",
    "Product",
    "Report",
    "SPD",
    "Session",
    "Solver",
    "SolverConfig",
    "Stiefel",
    "Stop",
    "__version__",
    "lowest_eigenpairs",
    "minimize",
]


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
    native = getattr(_core, "lowest_eigenpairs", None)
    if native is None:
        raise AttributeError("this rgmin build has no lowest_eigenpairs")
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

    values, columns, residuals, converged = native(
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


def minimize(
    f_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    method: str | Any = "lbfgs",
    **options: Any,
) -> tuple[Any, np.ndarray]:
    """Minimize ``f_and_grad`` in place and return ``(report, x)``."""
    native = getattr(_core, "minimize", None)
    if native is None:
        raise AttributeError("this rgmin build has no minimize")
    report, x = native(f_and_grad, x0, method, **options)
    return report, x


class SolverConfig:
    """Validated optimizer options.

    Unknown keywords raise ``ValueError`` and name the valid options.
    A Cartesian solver installs a per-atom cap of 0.2 unless the method
    already limits the step. BFGS, dogleg, and FIRE 2.0 start without
    that cap. ``maxstep`` or ``atom_maxmove`` replaces it on every
    method, and a non-positive value clears it.
    """

    def __init__(self, method: str | Any = "lbfgs", **options: Any) -> None:
        base = getattr(_core, "SolverConfig", None)
        if base is None:
            raise AttributeError("this rgmin build has no SolverConfig")
        self._inner = base(method, **options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def __getattr__(name: str):
    if name in {"BFGS", "FIRE", "LBFGS"}:
        from rgmin import optimize

        return getattr(optimize, name)
    if name in {"Product", "SPD", "Stiefel"}:
        from rgmin import manifold

        return getattr(manifold, name)
    if hasattr(_core, name):
        return getattr(_core, name)
    raise AttributeError(f"module 'rgmin' has no attribute {name}")


__all__ = _EXPORTS
