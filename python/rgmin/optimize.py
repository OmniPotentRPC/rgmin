# MIT License
# Copyright 2026 Rohit Goswami
"""Drop-in replacements for the ``ase.optimize`` classes.

``from rgmin.optimize import BFGS`` takes the place of
``from ase.optimize import BFGS``. Every class is an
:class:`ase.optimize.optimize.Optimizer`: ``run(fmax, steps)``,
``irun``, ``converged``, ``nsteps``, ``attach``, the log line, the
trajectory frames and ``todict`` are ASE's own code. The class reads
and moves the system only through ``self.optimizable``, so an ASE
filter (``FrechetCellFilter``, ``UnitCellFilter``, ``ExpCellFilter``,
``StrainFilter``), the ASE constraints and any
:class:`ase.utils.abc.Optimizable` work as they do with ASE.

The numerical step is one rgmin :class:`~rgmin.Session` step.
``BFGS``, ``LBFGS``, ``FIRE`` and ``FIRE2`` take ASE's constructor,
positional order and defaults. The other classes expose the rest of
the session's gradient-only minimizers with the same conventions:
``maxstep`` caps the longest atomic displacement and ``alpha`` is the
initial Hessian guess, so the first step is ``-g / alpha``.

A ``restart`` file holds the evaluations the session made, not ASE's
arrays. Constructing the same class on that file replays them through
a fresh session without calling the calculator, so the run continues
exactly where it stopped. The differences from ASE that remain are
listed on each class and in the ASE migration page of the docs.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import numpy as np

from rgmin._maturin import Session

# After the maturin import, so a nanobind build fails there before ASE loads.
from ase.optimize.optimize import Optimizer as _ASEOptimizer

if TYPE_CHECKING:
    from ase.filters import UnitCellFilter

RESTART_FORMAT = "rgmin-replay-1"
_Evaluation = list[list[float] | float]


def _restart_error(message: str) -> Exception:
    from ase.optimize.optimize import RestartError

    return RestartError(message)


def _is_cell_filter(atoms: object) -> TypeGuard[UnitCellFilter]:
    from ase.filters import UnitCellFilter

    return isinstance(atoms, UnitCellFilter)


class _RgminOptimizer(_ASEOptimizer):
    """ASE ``Optimizer`` whose step is one rgmin session step.

    Subclasses set ``kind`` and return the session keywords from
    :meth:`_session_options`. ``_scale`` multiplies the value and
    gradient the session sees: ``1 / alpha`` starts a dense or
    first-order method at the inverse Hessian ``I / alpha``.
    """

    maxstep: float
    kind = "lbfgs"

    if TYPE_CHECKING:
        # Signatures for ASE's unannotated restart I/O; the methods themselves are inherited.
        def dump(self, data: object) -> None: ...

        def load(self) -> object: ...

    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        etol: float | None = None,
        max_evals: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._stops: dict[str, Any] = {
            key: value for key, value in (("etol", etol), ("max_evals", max_evals)) if value is not None
        }
        self.stop: str | None = None
        self._native: Session | None = None
        self._history: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] | None = None
        self._x_session: list[float] | None = None
        self._x_placed: np.ndarray | None = None
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            **kwargs,
        )

    @property
    def _scale(self) -> float:
        return 1.0

    def _session_options(self) -> dict[str, Any]:
        return {"max_atom": self.maxstep}

    def initialize(self) -> None:
        self._native = None
        self._history = []
        self._pending = None
        self._x_session = None
        self._x_placed = None

    def _open(self) -> Session:
        if self._native is None:
            self._native = Session(
                self.kind, self.optimizable.ndofs(), **{**self._session_options(), **self._stops}
            )
            if self._pending:
                self._replay(self._pending)
            self._pending = None
        return self._native

    def _evaluate(self, x: list[float], record: list[_Evaluation] | None) -> tuple[float, list[float]]:
        point = np.asarray(x, dtype=np.float64)
        # A cell filter's set_x(get_x()) is not bitwise idempotent; writing
        # the current point back would cost a calculator call ASE never makes.
        if not np.array_equal(point, np.asarray(self.optimizable.get_x(), dtype=np.float64).reshape(-1)):
            self.optimizable.set_x(point)
        value = float(self.optimizable.get_value())
        gradient = np.asarray(self.optimizable.get_gradient(), dtype=np.float64).reshape(-1)
        if record is not None:
            record.append([list(x), value, gradient.tolist()])
        scale = self._scale
        return value * scale, (gradient * scale).tolist()

    def _start(self) -> list[float]:
        x = np.asarray(self.optimizable.get_x(), dtype=np.float64).reshape(-1)
        expected = self._x_placed
        if expected is None and self._x_session is not None:
            expected = np.asarray(self._x_session)
        if (
            self._x_session is not None
            and expected is not None
            and expected.shape == x.shape
            and np.allclose(x, expected, rtol=0.0, atol=1e-10)
        ):
            return self._x_session
        start: list[float] = x.tolist()
        return start

    # ASE's Dynamics.step is an unannotated placeholder that raises, so pyright infers NoReturn for it.
    def step(self, forces: Any = None) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """One session step from the current point.

        ``forces`` is accepted for ASE's signature. The session asks the
        optimizable for the energy and gradient itself, so a value
        passed here is not used.
        """
        native = self._open()
        reset = self._before_step(native)
        x0 = self._start()
        record: list[_Evaluation] = []
        last: list[list[float]] = []

        def objective(x: list[float]) -> tuple[float, list[float]]:
            last[:] = [x]
            return self._evaluate(x, record)

        updated, value, _gnorm = native.step(x0, objective)
        self.stop = native.stop
        if not last or last[0] != updated:
            self.optimizable.set_x(np.asarray(updated, dtype=np.float64))
        self._x_session = list(updated)
        self._x_placed = np.asarray(self.optimizable.get_x(), dtype=np.float64).reshape(-1).copy()
        if self.restart is not None:
            entry: dict[str, Any] = {"x": list(x0), "evals": record}
            if reset:
                entry["forget"] = True
            self._history.append(entry)
            self._dump_history()

    def _stop_requested(self) -> bool:
        return self.stop in ("EnergyTolerance", "EvalBudget")

    def converged(self, *args: Any, **kwargs: Any) -> bool:
        """ASE's force test, or ``True`` once ``etol`` or ``max_evals`` stopped the session."""
        if self._stop_requested():
            return True
        return cast(bool, cast(Any, super()).converged(*args, **kwargs))

    def gradient_converged(self, gradient: Any = None) -> bool:
        """The ``converged`` of ASE 3.2x and later, which its run loop calls."""
        if self._stop_requested():
            return True
        return cast(bool, cast(Any, super()).gradient_converged(gradient))

    def run(self, fmax: float = 0.05, steps: int | None = None) -> bool:  # type: ignore[override]
        """Run to ``fmax``, an ``etol`` stop, or the ``max_evals`` budget.

        ``etol`` and ``max_evals`` are keywords of every optimizer here.
        The result is ``True`` when the forces or the energy change met
        their tolerance and ``False`` when the step count or the
        evaluation budget ran out. :attr:`stop` is the name of the last
        step's status, such as ``"EnergyTolerance"``, ``None`` before
        the first.
        """
        self.stop = None
        if steps is None:
            converged = cast(Any, super()).run(fmax=fmax)
        else:
            converged = cast(Any, super()).run(fmax=fmax, steps=steps)
        if self.stop == "EvalBudget":
            return False
        return cast(bool, converged)

    def _before_step(self, native: Session) -> bool:
        """Return true after clearing the session memory for this step."""
        return False

    def _forget(self, native: Session) -> None:
        native.forget()
        self._x_session = None
        self._x_placed = None

    def _wrapper_state(self) -> dict[str, Any]:
        return {}

    def _load_wrapper_state(self, state: dict[str, Any]) -> None:
        pass

    def _dump_history(self) -> None:
        data: dict[str, Any] = {
            "format": RESTART_FORMAT,
            "optimizer": self.__class__.__name__,
            "kind": self.kind,
            "steps": self._history,
            "wrapper": self._wrapper_state(),
        }
        if _is_cell_filter(self.atoms):
            data["orig_cell"] = np.asarray(self.atoms.orig_cell).tolist()
        self.dump(data)

    def read(self) -> None:
        data = self.load()
        if not isinstance(data, dict) or data.get("format") != RESTART_FORMAT:
            raise _restart_error(
                f"{self.restart} is not an rgmin restart file. "
                "rgmin optimizers replay their own evaluation log, not ASE's arrays."
            )
        if data.get("optimizer") != self.__class__.__name__:
            raise _restart_error(
                f"{self.restart} was written by {data.get('optimizer')}, not {self.__class__.__name__}"
            )
        self.initialize()
        if "orig_cell" in data and _is_cell_filter(self.atoms):
            self.atoms.orig_cell = np.asarray(data["orig_cell"], dtype=np.float64)
        self._history = list(data.get("steps", []))
        self._pending = list(self._history)
        self._load_wrapper_state(data.get("wrapper", {}))

    def _replay(self, steps: list[dict[str, Any]]) -> None:
        native = self._native
        assert native is not None
        updated: list[float] | None = None
        for index, entry in enumerate(steps):
            evals = list(entry["evals"])
            cursor = iter(evals)
            scale = self._scale

            def answer(
                x: list[float], cursor: Iterator[list[Any]] = cursor, index: int = index
            ) -> tuple[float, list[float]]:
                try:
                    point, value, gradient = next(cursor)
                except StopIteration:
                    raise _restart_error(
                        f"{self.restart}: step {index} asks for more evaluations than were recorded"
                    ) from None
                if not np.allclose(x, point, rtol=0.0, atol=1e-12):
                    raise _restart_error(
                        f"{self.restart}: step {index} does not replay under these parameters"
                    )
                return value * scale, [g * scale for g in gradient]

            if entry.get("forget"):
                native.forget()
            updated, _value, _gnorm = native.step(list(entry["x"]), answer)
            if next(cursor, None) is not None:
                raise _restart_error(f"{self.restart}: step {index} recorded more evaluations than it replays")
        if updated is not None:
            self._x_session = list(updated)
            self._x_placed = None


def _check_maxstep(maxstep: float | None, default: float) -> float:
    return default if maxstep is None else float(maxstep)


class BFGS(_RgminOptimizer):
    """Dense BFGS with the ASE ``BFGS`` constructor.

    ``alpha`` is ASE's initial Hessian ``alpha * I``. The session runs
    inverse BFGS on ``E / alpha`` from the identity, which is the same
    iteration as inverse BFGS on ``E`` from ``I / alpha``; the first
    step is ASE's ``-g / alpha``. ``maxstep`` scales the whole step so
    the longest atom moves at most that far, as ASE does.

    Differences from ASE: a pair with ``s . y <= 1e-12`` (in the scaled
    energy) is skipped, where ASE updates the Hessian anyway and steps
    along its absolute eigenvalues. Until such a pair, the iterates
    agree with ASE to rounding.
    """

    kind = "bfgs"
    defaults = {"maxstep": 0.2, "alpha": 70.0}

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
    ) -> None:
        self.maxstep = _check_maxstep(maxstep, self.defaults["maxstep"])
        if self.maxstep > 1.0:
            warnings.warn(
                "You are using a *very* large value for the maximum step size: %.1f Å" % self.maxstep,
                stacklevel=2,
            )
        self.alpha = self.defaults["alpha"] if alpha is None else float(alpha)
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            **kwargs,
        )

    @property
    def _scale(self) -> float:
        return 1.0 / self.alpha


class LBFGS(_RgminOptimizer):
    """Limited-memory BFGS with the ASE ``LBFGS`` constructor.

    The two-loop starts from ASE's fixed ``H0 = I / alpha`` and keeps
    ``memory`` pairs. ``use_line_search=False`` (the ASE default)
    shortens the step so the longest atom moves at most ``maxstep`` and
    keeps the pairs, which is ASE's length control.

    Differences from ASE:

    - ``damping`` multiplies ``maxstep``. ASE multiplies the step after
      the cap, so a step shorter than ``maxstep`` is also damped there.
      ``damping=1`` (the default) is the same in both.
    - A pair with too little curvature is dropped and a direction that
      is not descent clears the memory. ASE keeps both.
    - ``use_line_search=True`` is rgmin's strong Wolfe search, not
      ASE's ``LineSearch`` with ``c1=0.23``, ``c2=0.46``.
    - ``alpha`` is kept as ``H0_scale = 1 / alpha``. ASE's ``LBFGS``
      has no ``alpha`` attribute either, so ``todict`` and the
      trajectory description match.
    - ``eon_cap=True`` selects the crate default instead: the
      line-search-free eOn cap, which clears the memory when the
      proposal is longer than ``maxstep`` and scales ``H0`` itself;
      ``alpha`` is not used then.
    """

    kind = "lbfgs"

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
        *,
        eon_cap: bool = False,
        **kwargs: Any,
    ) -> None:
        self.maxstep = _check_maxstep(maxstep, self.defaults["maxstep"])
        if self.maxstep > 1.0:
            raise ValueError(
                "You are using a much too large value for the maximum step size: %.1f Angstrom" % self.maxstep
            )
        self.memory = int(memory)
        self.damping = float(damping)
        self.H0_scale = 1.0 / float(alpha)
        self.use_line_search = bool(use_line_search)
        self.eon_cap = bool(eon_cap)
        super().__init__(atoms, restart, logfile, trajectory, **kwargs)

    def _session_options(self) -> dict[str, Any]:
        cap = self.maxstep * self.damping if self.damping > 0.0 else self.maxstep
        options: dict[str, Any] = {
            "max_atom": cap,
            "memory": self.memory,
            "wolfe": self.use_line_search and not self.eon_cap,
            "eon_cap": self.eon_cap,
        }
        if not self.eon_cap:
            options["h0"] = self.H0_scale
        return options

    def _replay_trajectory(self, traj: Any) -> None:
        """Push the secant pairs of a trajectory into a fresh memory."""
        self.initialize()
        native = self._open()
        frames = list(traj)
        for before, after in zip(frames[:-1], frames[1:]):
            s = (after.get_positions() - before.get_positions()).ravel()
            y = (before.get_forces() - after.get_forces()).ravel()
            native.push_pair(s.tolist(), y.tolist())


class LBFGSLineSearch(LBFGS):
    """``LBFGS`` with ``use_line_search=True``, as ``ase.optimize.LBFGSLineSearch``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["use_line_search"] = True
        super().__init__(*args, **kwargs)


class _Fire(_RgminOptimizer):
    """Shared FIRE plumbing: a Euclidean cap and ASE's downhill check."""

    position_reset_callback: Any = None
    downhill_check = False

    def initialize(self) -> None:
        super().initialize()
        self.e_last: float | None = None
        self.r_last: np.ndarray | None = None

    def _before_step(self, native: Session) -> bool:
        if not self.downhill_check:
            return False
        optimizable = self.optimizable
        reset = False
        if self.e_last is not None and self.r_last is not None:
            e = float(optimizable.get_value())
            if e > self.e_last:
                if self.position_reset_callback is not None:
                    self.position_reset_callback(optimizable, self.r_last, e, self.e_last)
                optimizable.set_x(self.r_last)
                self._forget(native)
                reset = True
        self.e_last = float(optimizable.get_value())
        self.r_last = np.asarray(optimizable.get_x(), dtype=np.float64).copy()
        return reset

    def _wrapper_state(self) -> dict[str, Any]:
        if self.e_last is None or self.r_last is None:
            return {}
        return {"e_last": self.e_last, "r_last": self.r_last.tolist()}

    def _load_wrapper_state(self, state: dict[str, Any]) -> None:
        if "e_last" in state:
            self.e_last = float(state["e_last"])
            self.r_last = np.asarray(state["r_last"], dtype=np.float64)


class FIRE(_Fire):
    """FIRE with the ASE ``FIRE`` constructor.

    The iterates are ASE's to rounding. ``maxstep`` caps the Euclidean
    length of the whole displacement, as ASE does. ``maxmove`` is ASE's
    deprecated alias and warns the same way. The session counts the
    current downhill step before comparing with ``Nmin`` and ASE after,
    so the session receives ``Nmin + 1``.

    ``downhill_check=True`` compares the energy with the last kept one
    at the start of each step, as ASE does. After a rise it calls
    ``position_reset_callback(optimizable, r_last, e, e_last)``, moves
    back to ``r_last`` and clears the velocity and the mixing.

    Differences from ASE: after such a reset the time step is kept,
    where ASE halves it, and the next kick uses the force at
    ``r_last``, where ASE reuses the force of the refused point.
    ``a`` (ASE's initial mixing) is ``astart`` here; another value
    warns.
    """

    kind = "fire"

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
    ) -> None:
        if maxmove is not None:
            warnings.warn("Use of `maxmove` is deprecated. Use `maxstep` instead.", FutureWarning, stacklevel=2)
            if maxstep is None:
                maxstep = maxmove
        self.maxstep = _check_maxstep(maxstep, self.defaults["maxstep"])
        self.dt = float(dt)
        self.dtmax = float(dtmax)
        self.Nmin = int(Nmin)
        self.finc = float(finc)
        self.fdec = float(fdec)
        self.astart = float(astart)
        self.fa = float(fa)
        self.a = float(a)
        if self.a != self.astart:
            warnings.warn(
                f"rgmin FIRE starts the mixing at astart={self.astart}; a={self.a} is not used",
                stacklevel=2,
            )
        self.downhill_check = bool(downhill_check)
        self.position_reset_callback = position_reset_callback
        super().__init__(atoms, restart, logfile, trajectory, **kwargs)

    def _session_options(self) -> dict[str, Any]:
        return {
            "max_atom": 0.0,
            "maxmove": self.maxstep,
            "dt": self.dt,
            "dt_max": self.dtmax,
            "n_min": self.Nmin + 1,
            "f_inc": self.finc,
            "f_dec": self.fdec,
            "alpha": self.astart,
            "f_alpha": self.fa,
            "accept": "energy" if self.downhill_check else "none",
        }


class FIRE2(_Fire):
    """FIRE 2.0 with the ASE ``FIRE2`` constructor.

    The session runs Guénolé's semi-implicit Euler integrator with the
    half-step back after an uphill step. ``maxstep`` caps the Euclidean
    length of the displacement, or each component of it with
    ``use_abc=True`` (ABC-FIRE), as ASE does. ``position_reset_callback``
    is accepted and, as in ASE, not called.

    Differences from ASE: the time step floor is ``0.02 * dt`` (ASE's
    ``dtmin=2e-3`` at ``dt=0.1``); another ``dtmin`` is a
    ``ValueError``. Guénolé's initial delay holds ``dt`` for the first
    ``Nmin`` steps, and 2000 consecutive uphill steps stop the session.
    ASE has neither.
    """

    kind = "fire2"

    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        dt: float = 0.1,
        maxstep: float = 0.2,
        dtmax: float = 1.0,
        dtmin: float = 2e-3,
        Nmin: int = 20,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.25,
        fa: float = 0.99,
        position_reset_callback: Any = None,
        use_abc: bool | None = False,
        cap_two_step_overshoot: bool = False,
        **kwargs: Any,
    ) -> None:
        self.dt = float(dt)
        self.maxstep = _check_maxstep(maxstep, self.defaults["maxstep"])
        self.dtmax = float(dtmax)
        self.dtmin = float(dtmin)
        if not np.isclose(self.dtmin, 0.02 * self.dt, rtol=1e-12, atol=0.0):
            raise ValueError(f"rgmin FIRE2 floors the time step at 0.02 * dt = {0.02 * self.dt:g}, not dtmin={dtmin:g}")
        self.Nmin = int(Nmin)
        self.finc = float(finc)
        self.fdec = float(fdec)
        self.astart = float(astart)
        self.fa = float(fa)
        self.a = self.astart
        self.position_reset_callback = position_reset_callback
        self.use_abc = bool(use_abc)
        self.cap_two_step_overshoot = bool(cap_two_step_overshoot)
        super().__init__(atoms, restart, logfile, trajectory, **kwargs)

    def _session_options(self) -> dict[str, Any]:
        return {
            "max_atom": 0.0,
            "maxmove": self.maxstep,
            "preset": "guenole2020",
            "dt": self.dt,
            "dt_max": self.dtmax,
            "n_min": self.Nmin,
            "f_inc": self.finc,
            "f_dec": self.fdec,
            "alpha": self.astart,
            "f_alpha": self.fa,
            "abc": self.use_abc,
            "cap_two_step_overshoot": self.cap_two_step_overshoot,
        }


class _Scaled(_RgminOptimizer):
    """A session method with ASE's ``maxstep`` and ``alpha``.

    The session sees ``E / alpha``, so its unit initial step is
    ``-g / alpha``. ``accept`` is the session acceptance rule:
    ``"energy"`` refuses a rise and halves the step, ``"none"`` takes
    every capped step, ``"nonmonotone"`` compares against the last five
    accepted energies.

    The conjugate-gradient classes take the session's conjugate
    direction at that unit scale under the cap, with no line search.
    That is not a line-search CG: on ASE's rattled Au8 cell ``CG``
    needs as many steps as ``SteepestDescent``. ``LBFGS``, ``BFGS`` or
    ``FIRE`` are the classes to relax with; these are here so every
    session method can be driven from ASE.
    """

    default_accept = "energy"

    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        alpha: float | None = None,
        accept: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.maxstep = _check_maxstep(maxstep, self.defaults["maxstep"])
        self.alpha = 70.0 if alpha is None else float(alpha)
        self.accept = self.default_accept if accept is None else str(accept)
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            **kwargs,
        )

    @property
    def _scale(self) -> float:
        return 1.0 / self.alpha

    def _session_options(self) -> dict[str, Any]:
        return {"max_atom": self.maxstep, "accept": self.accept}


class SR1(_Scaled):
    """Inverse symmetric rank-one quasi-Newton (Nocedal-Wright 6.24)."""

    kind = "sr1"


class SR2(_Scaled):
    """Powell-symmetrized rank-one secant on a dense Hessian."""

    kind = "sr2"


class SteepestDescent(_Scaled):
    """Steepest descent: the step is ``-g / alpha`` under the cap.

    Like the conjugate-gradient classes, which take no line search,
    it reaches the energy of a stiff, badly scaled system (a
    ``StrainFilter`` cell) quickly but approaches a tight ``fmax``
    slowly; use a quasi-Newton class there.
    """

    kind = "sd"


class BarzilaiBorwein(_Scaled):
    """Barzilai-Borwein spectral steepest descent."""

    kind = "bb"


class Adam(_Scaled):
    """Kingma-Ba Adam (``beta1=0.9``, ``beta2=0.999``, ``eps=1e-8``).

    The 0.6.0 session scales that direction to unit per-atom length and
    line-searches from ``learning_rate``, so the opening trial moves the
    longest atom by about ``learning_rate`` rather than each coordinate
    by ``learning_rate`` times the raw moment step. ``maxstep`` still
    caps the longest atomic displacement.
    ``alpha`` scales the gradient, which Adam divides out, so it does
    not set the step here; ``maxstep`` still caps the longest atomic
    displacement. A ``learning_rate`` near ``maxstep`` makes every step
    a capped sign step: the run climbs out of the starting basin and
    where it ends depends on the last bits of the forces.

    ``accept`` defaults to ``"none"``: a refused step resets the
    moments, and under ``"energy"`` that stalls Adam short of a tight
    ``fmax``.
    """

    kind = "adam"
    default_accept = "none"

    def __init__(self, *args: Any, learning_rate: float = 0.02, **kwargs: Any) -> None:
        self.learning_rate = float(learning_rate)
        if not (np.isfinite(self.learning_rate) and self.learning_rate > 0.0):
            raise ValueError(f"learning_rate must be positive and finite, not {learning_rate}")
        super().__init__(*args, **kwargs)

    def _session_options(self) -> dict[str, Any]:
        return {**super()._session_options(), "istep": self.learning_rate}


class CG(_Scaled):
    """Nonlinear conjugate gradients, Polak-Ribière-Plus, ``max(0, β_PR)``."""

    kind = "prp"


class PolakRibiere(_Scaled):
    """Nonlinear conjugate gradients, Polak-Ribière, no restart."""

    kind = "pr"


class FletcherReeves(_Scaled):
    """Nonlinear conjugate gradients, Fletcher-Reeves."""

    kind = "fr"


class HestenesStiefel(_Scaled):
    """Nonlinear conjugate gradients, Hestenes-Stiefel."""

    kind = "hs"


class DaiYuan(_Scaled):
    """Nonlinear conjugate gradients, Dai-Yuan."""

    kind = "dy"


class ConjugateDescent(_Scaled):
    """Nonlinear conjugate gradients, Fletcher's conjugate descent."""

    kind = "cd"


class HagerZhang(_Scaled):
    """Nonlinear conjugate gradients, Hager-Zhang."""

    kind = "hz"


class LiuStorey(_Scaled):
    """Nonlinear conjugate gradients, Liu-Storey."""

    kind = "ls"


class FRPR(_Scaled):
    """Nonlinear conjugate gradients, the Fletcher-Reeves / Polak-Ribière hybrid."""

    kind = "fr-pr"


class EonCG(_Scaled):
    """eOn conjugate gradients: Polak-Ribière with the Powell restart at 0.5."""

    kind = "eon-cg"


class NewtonCG(_Scaled):
    """Steihaug-Toint Newton-CG on forward-difference Hessian products.

    Each product costs one extra gradient, so a step charges more force
    calls than a quasi-Newton step. The trust region is the session's;
    ``maxstep`` still caps the longest atomic displacement.
    """

    kind = "newton-cg"
    default_accept = "none"


class NewtonPCG(NewtonCG):
    """``NewtonCG`` with a rank-16 Nyström preconditioner."""

    kind = "newton-pcg"


__all__ = [
    "BFGS",
    "CG",
    "FIRE",
    "FIRE2",
    "FRPR",
    "LBFGS",
    "SR1",
    "SR2",
    "Adam",
    "BarzilaiBorwein",
    "ConjugateDescent",
    "DaiYuan",
    "EonCG",
    "FletcherReeves",
    "HagerZhang",
    "HestenesStiefel",
    "LBFGSLineSearch",
    "LiuStorey",
    "NewtonCG",
    "NewtonPCG",
    "PolakRibiere",
    "SteepestDescent",
]
