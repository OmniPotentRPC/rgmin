from typing import Callable

import numpy as np
from numpy.typing import NDArray

class Lbfgs:
    """Persistent L-BFGS.

    `step` is ``capped``, ``scaled``, or ``wolfe``.
    `norm` is ``infinity`` or ``euclidean``.
    """

    def __init__(
        self,
        memory: int = ...,
        gtol: float = ...,
        ftol: float = ...,
        step: str = ...,
        max_atom: float = ...,
        norm: str = ...,
    ) -> None: ...
    def set_gtol(self, gtol: float) -> None: ...
    def set_ftol(self, ftol: float) -> None: ...
    def set_step(self, step: str, max_atom: float) -> None: ...
    def set_norm(self, norm: str) -> None: ...
    def minimize(
        self,
        x: NDArray[np.float64],
        max_iter: int,
        fg: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]] | None],
    ) -> tuple[float, NDArray[np.float64], int]:
        """Relax `x`. `fg` returns ``(value, gradient)`` or ``None``."""

def minimize_eon_cg(
    x: NDArray[np.float64],
    fill: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    max_move: float = ...,
    gtol: float = ...,
    max_iter: int = ...,
    cluster: bool = ...,
    atom_only: bool = ...,
) -> tuple[float, float, int, bool, NDArray[np.float64]]:
    """eOn conjugate gradients.

    Returns energy, gradient norm, evaluations, convergence, and the
    point. The quadratic ``x^2 + y^2`` from ``(1, -2)`` finishes below
    ``1e-8``.
    """
