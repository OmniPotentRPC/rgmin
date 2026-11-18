from collections.abc import Callable

import numpy
from numpy.typing import NDArray

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
        cap_two_step_overshoot: bool = False,
        downhill: bool = False,
        h0: float | None = None,
        maxmove: float | None = None,
        preset: str | None = None,
        abc: bool = False,
        accept: str | None = None,
        istep: float | None = None,
        level_set: float | None = None,
        error_on_unmet: bool = False,
        backward_polish: bool = False,
        etol: float | None = None,
        max_evals: int | None = None,
    ) -> None: ...
    def push_pair(self, s: list[float], y: list[float]) -> bool: ...
    def pair_count(self) -> int: ...
    @property
    def steps(self) -> int: ...
    @property
    def evals(self) -> int: ...
    @property
    def stop(self) -> str | None: ...
    def set_etol(self, etol: float | None = None) -> None: ...
    def set_max_evals(self, max_evals: int | None = None) -> None: ...
    def forget(self) -> None: ...
    def set_level_set(self, target: float | None = None) -> None: ...
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

class LbfgsPairs:
    def __init__(self, n: int, capacity: int = 10) -> None: ...
    def __len__(self) -> int: ...
    def dim(self) -> int: ...
    def push(self, s: list[float], y: list[float], rho: float | None = None) -> bool: ...
    def clear(self) -> None: ...
    def direction(
        self,
        gradient: list[float],
        h0: float | NDArray[numpy.float64] | list[float] | Callable[[list[float]], list[float]],
    ) -> list[float]: ...

class RiemannianManifold:
    @staticmethod
    def stiefel(n: int, p: int, retraction: str = "qr") -> RiemannianManifold: ...
    @staticmethod
    def spd(p: int, metric: str = "affine_invariant") -> RiemannianManifold: ...
    @staticmethod
    def product(factors: list[RiemannianManifold]) -> RiemannianManifold: ...
    def __len__(self) -> int: ...
    def factors(self) -> list[tuple[str, int, int]]: ...
    def project(self, x: list[float], v: list[float]) -> list[float]: ...
    def egrad_to_rgrad(self, x: list[float], egrad: list[float]) -> list[float]: ...
    def retract(self, x: list[float], v: list[float]) -> list[float]: ...
    def exp(self, x: list[float], v: list[float]) -> list[float]: ...
    def log(self, x: list[float], y: list[float]) -> list[float]: ...
    def transport(self, x_from: list[float], x_to: list[float], v: list[float]) -> list[float]: ...
    def inner(self, x: list[float], u: list[float], v: list[float]) -> float: ...
    def gradient_step(
        self, x: list[float], egrad: list[float], t: float
    ) -> tuple[list[float], list[float]]: ...

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
