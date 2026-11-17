from collections.abc import Callable
from typing import Any

import numpy
from numpy.typing import NDArray

from rgmin import _core
from rgmin._core import Method as Method
from rgmin._core import NumericalError as NumericalError
from rgmin._core import Report as Report
from rgmin._core import Solver as Solver
from rgmin._core import Stop as Stop
from rgmin._maturin import LbfgsPairs as LbfgsPairs
from rgmin._maturin import LineSearch as LineSearch
from rgmin._maturin import Session as Session
from rgmin.manifold import SPD as SPD
from rgmin.manifold import Product as Product
from rgmin.manifold import Stiefel as Stiefel

__version__: str

class SolverConfig(_core.SolverConfig):
    def __init__(self, method: str | Method = "lbfgs", **options: Any) -> None: ...

def minimize(
    f_and_grad: Callable[[NDArray[numpy.float64]], tuple[float, NDArray[numpy.float64]]],
    x0: NDArray[numpy.float64],
    method: str | Method = "lbfgs",
    **options: Any,
) -> tuple[Report, NDArray[numpy.float64]]: ...

def gromacs_mdp_options(mdp: Any = None, /, **overrides: object) -> dict[str, Any]: ...

def lowest_eigenpairs(
    hvp: Callable[[Any, Any], Any],
    x0_block: Any,
    *,
    x: Any = None,
    tol: float = 1e-8,
    max_iter: int | None = None,
    precond: Callable[[Any], Any] | None = None,
    strict: bool = True,
) -> tuple[Any, Any, Any, Any]: ...

class BFGS:
    maxstep: float
    alpha: float
    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        alpha: float | None = None,
        **kwargs: Any,
    ) -> None: ...
    def run(self, fmax: float = 0.05, steps: int | None = None) -> bool: ...
    def step(self, forces: Any = None) -> None: ...
    def attach(
        self,
        function: Callable[..., Any],
        interval: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def converged(self, forces: Any = None) -> bool: ...

class GromacsSteep:
    emtol: float
    emstep: float
    gromacs_nsteps: int
    maxstep: float
    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        *,
        emtol: float = 10.0,
        emstep: float = 0.01,
        nsteps: int = -1,
        **kwargs: Any,
    ) -> None: ...
    def run(self, fmax: float | None = None, steps: int | None = None) -> bool: ...
    def step(self, forces: Any = None) -> None: ...

class LBFGS:
    maxstep: float
    memory: int
    damping: float
    alpha: float
    use_line_search: bool
    eon_cap: bool
    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        maxstep: float | None = None,
        memory: int = 100,
        damping: float = 1.0,
        alpha: float = 70.0,
        use_line_search: bool = False,
        eon_cap: bool = False,
        **kwargs: Any,
    ) -> None: ...
    def run(self, fmax: float = 0.05, steps: int | None = None) -> bool: ...
    def step(self, forces: Any = None) -> None: ...
    def attach(
        self,
        function: Callable[..., Any],
        interval: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def converged(self, forces: Any = None) -> bool: ...

class FIRE:
    maxstep: float
    dt: float
    dtmax: float
    Nmin: int
    finc: float
    fdec: float
    astart: float
    fa: float
    a: float
    downhill_check: bool
    position_reset_callback: Any
    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        dt: float = 0.1,
        maxstep: float | None = None,
        maxmove: float | None = None,
        dtmax: float = 1.0,
        Nmin: int = 5,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.1,
        fa: float = 0.99,
        a: float = 0.1,
        downhill_check: bool = False,
        position_reset_callback: Any = None,
        **kwargs: Any,
    ) -> None: ...
    def run(self, fmax: float = 0.05, steps: int | None = None) -> bool: ...
    def step(self, forces: Any = None) -> None: ...
    def attach(
        self,
        function: Callable[..., Any],
        interval: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def converged(self, forces: Any = None) -> bool: ...
