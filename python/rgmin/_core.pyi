"""Free-threaded bindings for the rgmin geometry optimizer."""

from collections.abc import Callable
import enum

import numpy
from numpy.typing import NDArray


class NumericalError(ArithmeticError):
    pass

class Method(enum.IntEnum):
    """
    Geometry-optimization method. Members are UPPER_SNAKE. CamelCase names are aliases of the same values.
    """

    POLAK_RIBIERE = 0

    FLETCHER_REEVES = 1

    BFGS = 2

    LBFGS = 3

    SR1 = 4

    ADAM = 5

    STEEPEST = 6

    SR2 = 7

    PSO = 8

    HESTENES_STIEFEL = 9

    DAI_YUAN = 10

    CONJUGATE_DESCENT = 11

    HAGER_ZHANG = 12

    LIU_STOREY = 13

    FR_PR = 14

    NEWTON = 15

    RFO = 16

    FIRE = 17

    BB = 18

    DOGLEG = 19

    FIRE2 = 20

    QUICK_MIN = 21

    DIIS = 22

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
    """
    Why a step or a run ended. Members are UPPER_SNAKE. CamelCase names are aliases of the same values.
    """

    TOLERANCE = 0

    NOISE_FLOOR = 1

    STEP_TOLERANCE = 2

    BUDGET = 3

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

    A Cartesian solver starts with a per-atom cap of 0.2 (ASE ``maxstep``, eOn ``max_move``). ``maxstep`` or ``atom_maxmove`` replaces it. A non-positive value clears it. A positive ``maxmove`` is the Euclidean step limit and leaves the per-atom default off.
    """

    def __init__(self, method: str | Method = 'lbfgs', **kwargs) -> None: ...

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

    def step(self, f_and_grad: Callable[[NDArray[numpy.float64]], tuple[float, NDArray[numpy.float64]]], x: NDArray[numpy.float64]) -> Report:
        """Take one step. ``f_and_grad(x)`` returns ``(value, gradient)``."""

    def step_hess(self, f_and_grad: Callable[[NDArray[numpy.float64]], tuple[float, NDArray[numpy.float64]]], hessian: Callable[[NDArray[numpy.float64]], NDArray[numpy.float64]], x: NDArray[numpy.float64]) -> Report:
        """
        One Newton, RFO, or dogleg step. ``hessian(x)`` returns an ``(n, n)`` matrix.
        """

def minimize(f_and_grad: Callable[[NDArray[numpy.float64]], tuple[float, NDArray[numpy.float64]]], x0: NDArray[numpy.float64], method: object, **kwargs) -> tuple[Report, NDArray[numpy.float64]]:
    """Minimize ``f_and_grad`` in place and return ``(report, x)``."""
