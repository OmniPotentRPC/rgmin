from collections.abc import Callable
from typing import Any

from rgmin._core import LineSearch as LineSearch
from rgmin._core import Session as Session

__version__: str

def lowest_eigenpairs(
    hvp: Callable[[Any, Any], Any],
    x0_block: Any,
    *,
    x: Any = None,
    tol: float = 1e-8,
    max_iter: int | None = None,
) -> tuple[Any, Any]: ...

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
