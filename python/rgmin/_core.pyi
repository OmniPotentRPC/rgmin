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

    ODE12R = 23

    GROMACS_STEEP = 28

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

    Ode12r = 23

    GromacsSteep = 28

class Stop(enum.IntEnum):
    """
    Why a step or a run ended. Members are UPPER_SNAKE. CamelCase names are aliases of the same values.
    """

    TOLERANCE = 0

    NOISE_FLOOR = 1

    STEP_TOLERANCE = 2

    BUDGET = 3

    LINE_SEARCH = 4

    POLISH_UNMET = 5

    ENERGY_TOLERANCE = 6

    EVAL_BUDGET = 7

    MACHINE_PRECISION = 8

    Tolerance = 0

    NoiseFloor = 1

    StepTolerance = 2

    Budget = 3

    LineSearch = 4

    PolishUnmet = 5

    EnergyTolerance = 6

    EvalBudget = 7

    MachinePrecision = 8

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

    def converged(self) -> bool:
        """True when ``stop`` is ``TOLERANCE``."""

class SolverConfig:
    """
    Validated, immutable optimizer options. Unknown keywords raise ``ValueError`` listing the valid options. A Cartesian solver installs a per-atom cap of 0.2 (ASE maxstep, eOn max_move) unless the method already limits the step. BFGS, dogleg, and FIRE 2.0 start without that cap. ``maxstep`` or ``atom_maxmove`` replaces it on every method. A non-positive value clears it. A positive ``maxmove`` is the Euclidean step limit and leaves the per-atom default off.
    """

    def __init__(self, method: str | Method = 'lbfgs', **kwargs) -> None: ...

    @property
    def method(self) -> Method: ...

    @property
    def method_name(self) -> str: ...

    def error_on_unmet(self) -> bool:
        """Whether an unmet stop raises."""

    def backward_polish(self) -> bool:
        """Whether a Hessian one-shot polishes to the backward-error floor."""

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
    def evals(self) -> int:
        """Objective evaluations the steps of this solver have made."""

    @property
    def method(self) -> Method: ...

    def step(self, f_and_grad: Callable[[NDArray[numpy.float64]], tuple[float, NDArray[numpy.float64]]], x: NDArray[numpy.float64]) -> Report:
        """Take one step. ``f_and_grad(x)`` returns ``(value, gradient)``."""

    def step_hess(self, f_and_grad: Callable[[NDArray[numpy.float64]], tuple[float, NDArray[numpy.float64]]], hessian: Callable[[NDArray[numpy.float64]], NDArray[numpy.float64]], x: NDArray[numpy.float64]) -> Report:
        """
        One Newton, RFO, or dogleg step. ``hessian(x)`` returns an ``(n, n)`` matrix.
        """

def minimize(f_and_grad: Callable[[NDArray[numpy.float64]], tuple[float, NDArray[numpy.float64]]], x0: NDArray[numpy.float64], method: object, **kwargs) -> tuple[Report, NDArray[numpy.float64]]:
    """
    Minimize ``f_and_grad`` in place and return ``(report, x)``. ``error_on_unmet=True`` raises ``RuntimeError`` when the stop is ``BUDGET``, ``LINE_SEARCH``, ``POLISH_UNMET``, or ``EVAL_BUDGET``. ``backward_polish`` is stored on the session and is not used by this step loop, which does not form a Hessian.
    """

def backward_error_floor(value: float, x: object) -> float:
    """First-order floor ``n * eps * (1 + |value| + sum |x_i|)``."""

def gromacs_steep_max_constraint_shrinks(istep: float, shrink: float) -> int:
    """Closed-form capped constraint-failure budget for ``gromacs-steep``."""
