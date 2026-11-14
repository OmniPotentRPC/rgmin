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
) -> tuple[list[float], list[list[float]]]: ...
