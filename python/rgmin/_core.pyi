from typing import Callable

class Session:
    def __init__(
        self,
        kind: str,
        n: int,
        *,
        max_atom: float = 0.2,
        wolfe: bool = False,
        eon_cap: bool = False,
        memory: int = 8,
        dt: float = 0.1,
        dt_max: float = 1.0,
        n_min: int = 5,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        alpha: float = 0.1,
        f_alpha: float = 0.99,
        downhill: bool = False,
    ) -> None: ...
    def step(
        self,
        x: list[float],
        callback: Callable[[list[float]], tuple[float, list[float]]],
    ) -> tuple[list[float], float, float]: ...

class LineSearch:
    def __init__(
        self,
        kind: str,
        x: list[float],
        direction: list[float],
        *,
        istep: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        beta: float = 0.5,
        maxiter: int = 40,
        tol: float = 1e-10,
    ) -> None: ...
    def offer(self, f: float, gradient: list[float]) -> tuple[list[float], float, float, bool]: ...

version: str

def lowest_eigenpairs(
    hvp: Callable[[list[float], list[float]], list[float]],
    x0_block: list[list[float]],
    x: list[float] | None = None,
    *,
    tol: float = 1e-8,
    max_iter: int | None = None,
    precond: Callable[[list[list[float]]], list[list[float]]] | None = None,
) -> tuple[list[float], list[list[float]], list[float], list[bool]]: ...
