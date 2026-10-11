"""Free-threaded bindings for the rgmin geometry optimizer."""

from collections.abc import Callable
import enum
from typing import Annotated

import numpy
from numpy.typing import NDArray


class NumericalError(ArithmeticError):
    pass

class Method(enum.IntEnum):
    """
    Geometry-optimization method. An ``enum.IntEnum`` of ``rgmin_method_t``.
    """

    PolakRibiere = 0

    FletcherReeves = 1

    Bfgs = 2

    Lbfgs = 3

    Sr1 = 4

    Adam = 5

    Steepest = 6

    Sr2 = 7

    Pso = 8

    HestenesStiefel = 9

    DaiYuan = 10

    ConjugateDescent = 11

    HagerZhang = 12

    LiuStorey = 13

    FrPr = 14

    Newton = 15

    Rfo = 16

    Fire = 17

    Bb = 18

    Dogleg = 19

    Fire2 = 20

    QuickMin = 21

    Diis = 22

class Stop(enum.IntEnum):
    """Why a step or a run ended."""

    Tolerance = 0

    NoiseFloor = 1

    StepTolerance = 2

    Budget = 3

class Report:
    """Result of one optimizer step or of ``minimize``."""

    @property
    def value(self) -> float:
        """Scalar value at the accepted point."""

    @property
    def steps(self) -> int:
        """Number of oracle evaluations used."""

    @property
    def grad_norm(self) -> float:
        """Norm of the gradient."""

    @property
    def stop(self) -> Stop:
        """The ``Stop`` reason."""

class SolverConfig:
    """
    Validated, immutable optimizer options. Unknown keywords raise ``ValueError`` listing the valid options.
    """

    def __init__(self, arg0: object, /, **kwargs) -> None: ...

    @property
    def method(self) -> Method: ...

    @property
    def method_name(self) -> str: ...

    @staticmethod
    def valid_options() -> str: ...

    @staticmethod
    def valid_methods() -> str: ...

class Solver:
    """Move-only optimizer bound to one coordinate length."""

    def __init__(self, n: int, config: SolverConfig) -> None: ...

    @property
    def dim(self) -> int: ...

    @property
    def method(self) -> Method: ...

    def step(self, f_and_grad: Callable, x: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu')]) -> Report:
        """Take one step. ``f_and_grad(x)`` returns ``(value, gradient)``."""

def minimize(arg0: Callable, arg1: object, arg2: object, /, **kwargs) -> Report:
    """Minimize ``f_and_grad`` in place, starting at ``x0``."""
