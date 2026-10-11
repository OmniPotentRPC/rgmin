# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Geometry optimization through the rgmin C ABI.

Importing this package does not import ASE. The ASE optimizer lives in
:mod:`rgmin.ase`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

import rgmin._core as _core
from rgmin._core import Method, NumericalError, Report, Solver, Stop

__version__ = _core.__version__  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]


class SolverConfig(_core.SolverConfig):
    """Validated optimizer options.

    Unknown keywords raise ``ValueError`` and name the valid options.
    """

    def __init__(self, method: str | Method = "lbfgs", **options: Any) -> None:
        super().__init__(method, **options)


def minimize(
    f_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    method: str | Method = "lbfgs",
    **options: Any,
) -> tuple[Report, np.ndarray]:
    """Minimize ``f_and_grad`` in place and return ``(report, x)``."""
    report, x = _core.minimize(f_and_grad, x0, method, **options)
    return report, x


__all__ = [
    "Method",
    "NumericalError",
    "Report",
    "Solver",
    "SolverConfig",
    "Stop",
    "__version__",
    "minimize",
]
