# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""ASE :class:`~ase.optimize.optimize.Optimizer` backed by an rgmin solver.

ASE is imported when this module is imported. ``import rgmin`` does not
import ASE.

Goswami, Gunde, and Jónsson, Front. Chem. (2026),
https://doi.org/10.3389/fchem.2026.1807063.
Goswami, AIP Adv. (2025), https://doi.org/10.1063/5.0283639.
Hjorth Larsen et al., J. Phys.: Condens. Matter 29, 273002 (2017),
https://doi.org/10.1088/1361-648X/aa680e.
"""

from __future__ import annotations

import importlib
import inspect
import threading
import time
import warnings
import weakref
from collections import deque
from collections.abc import Callable
from types import TracebackType
from typing import Any, cast

import numpy as np
from ase.optimize.optimize import Optimizer as _ASEOptimizer

import rgmin

_ASE_KEYS = frozenset({"master", "comm", "loginterval", "force_consistent"})
_HESSIAN_METHODS = frozenset({"newton", "rfo", "dogleg"})
# Evaluation log written by dump(). Not ASE's Hessian or velocity arrays.
RESTART_FORMAT = "rgmin-replay-1"
_RESTART_POSITION_ATOL = float.fromhex("0x1.b7cdfd9d7bdbbp-34")
_RESTART_REPLAY_ATOL = float.fromhex("0x1.19799812dea11p-40")
_PARAMETER_RTOL = float.fromhex("0x1.19799812dea11p-40")
# Serialises _calc_locks and _id_locks. Finalizers must not block on
# this lock: a collection can run while the creating thread already
# holds it. They queue a release instead. The calculator lock is
# acquired only after this one is released.
_calc_guard = threading.Lock()
_calc_locks: weakref.WeakKeyDictionary[Any, threading.Lock] = weakref.WeakKeyDictionary()


class _IdEntry:
    """Lock for one calculator that cannot be weakly referenced.

    ``calc`` keeps the object alive so its id cannot be reused while
    the entry is registered. ``holders`` counts optimizers that are
    using it. The entry is removed when that count reaches zero.
    """

    __slots__ = ("calc", "lock", "holders")

    def __init__(self, calc: Any) -> None:
        self.calc = calc
        self.lock = threading.Lock()
        self.holders = 0


# Keyed by id(calc). An entry is present while at least one optimizer
# holds it. holders is that count.
_id_locks: dict[int, _IdEntry] = {}
_pending_release: deque[_IdEntry] = deque()


class _CalcTable:
    """Hold ``_calc_guard`` and apply queued finalizer releases.

    Draining on entry picks up a release that lost the race to the
    lock. Draining on exit applies a release queued by GC during the
    critical section, including one caused by allocating an entry.
    """

    def __enter__(self) -> None:
        _calc_guard.acquire()
        try:
            _drain_releases()
        except BaseException:
            _calc_guard.release()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            _drain_releases()
        finally:
            _calc_guard.release()


_calc_table = _CalcTable()


def _restart_error(message: str) -> Exception:
    """ASE's ``RestartError`` when that class exists."""
    loaded = _import_attr("ase.optimize.optimize", "RestartError")
    if loaded is None:
        return RuntimeError(message)
    return loaded(message)


def _ase_passthrough(options: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(_ASEOptimizer.__init__).parameters
    passed: dict[str, Any] = {}
    for key in list(options):
        if key not in _ASE_KEYS:
            continue
        if key not in params:
            raise TypeError(f"unexpected keyword argument {key!r}")
        passed[key] = options.pop(key)
    return passed


def _import_attr(module: str, name: str) -> Any | None:
    try:
        loaded = importlib.import_module(module)
    except ImportError:
        return None
    return getattr(loaded, name, None)


def _nonconservative_types() -> tuple[type, ...]:
    found: list[type] = []
    for module, name in (
        ("ase.mep", "NEB"),
        ("ase.mep", "DyNEB"),
        ("ase.mep.neb", "NEBOptimizable"),
        ("ase.mep.dimer", "DimerOptimizable"),
        ("ase.mep", "MinModeAtoms"),
    ):
        cls = _import_attr(module, name)
        if isinstance(cls, type):
            found.append(cls)
    return tuple(found)


def _cell_filter_types() -> tuple[type, ...]:
    found: list[type] = []
    for name in ("FrechetCellFilter", "UnitCellFilter", "ExpCellFilter", "StrainFilter"):
        cls = _import_attr("ase.filters", name)
        if isinstance(cls, type):
            found.append(cls)
    return tuple(found)


def _is_nonconservative(obj: Any) -> bool:
    if obj is None:
        return False
    kinds = _nonconservative_types()
    if kinds and isinstance(obj, kinds):
        return True
    probe = getattr(obj, "is_neb", None)
    return bool(callable(probe) and probe())


def _is_cell_filter(obj: Any) -> bool:
    kinds = _cell_filter_types()
    return bool(kinds and isinstance(obj, kinds))


def _minmode(obj: Any) -> Any | None:
    cls = _import_attr("ase.mep", "MinModeAtoms")
    if not isinstance(cls, type):
        return None
    for candidate in (obj, getattr(obj, "atoms", None), getattr(obj, "dimeratoms", None)):
        if isinstance(candidate, cls):
            return candidate
    return None


class _Geometry:
    """Positions, value, and gradient for one ASE optimizable.

    ASE 3.29 exposes ``get_x``, ``get_gradient``, and ``get_value``.
    ASE 3.23 exposes ``get_positions``, ``get_forces``, and
    ``get_potential_energy``. A min-mode object reports its energy with
    ``get_potential_energy()`` and no ``force_consistent`` argument.
    """

    def __init__(self, opt: Any) -> None:
        self.opt = opt
        self.mode = _minmode(opt)
        self.modern = self.mode is None and all(
            hasattr(opt, name) for name in ("get_x", "get_gradient", "get_value")
        )

    def positions(self) -> np.ndarray:
        if self.modern:
            return np.asarray(self.opt.get_x(), dtype=np.float64).reshape(-1)
        source = self.mode if self.mode is not None else self.opt
        return np.asarray(source.get_positions(), dtype=np.float64).reshape(-1)

    def write(self, x: np.ndarray) -> None:
        if self.modern and hasattr(self.opt, "set_x"):
            self.opt.set_x(np.asarray(x, dtype=np.float64))
            return
        source = self.mode if self.mode is not None else self.opt
        current = np.asarray(source.get_positions(), dtype=np.float64)
        source.set_positions(np.asarray(x, dtype=np.float64).reshape(current.shape))

    def value(self) -> float:
        if self.mode is not None:
            return float(self.mode.get_potential_energy())
        if self.modern:
            return float(self.opt.get_value())
        return float(self.opt.get_potential_energy())

    def gradient(self) -> np.ndarray:
        if self.modern or hasattr(self.opt, "get_gradient"):
            return np.asarray(self.opt.get_gradient(), dtype=np.float64).reshape(-1)
        source = self.mode if self.mode is not None else self.opt
        forces = np.asarray(source.get_forces(), dtype=np.float64)
        return -forces.reshape(-1)

    def ndofs(self) -> int:
        if hasattr(self.opt, "ndofs"):
            return int(self.opt.ndofs())
        return int(self.positions().shape[0])


def _calculator_of(opt: Any) -> Any:
    atoms = getattr(opt, "atoms", None)
    if atoms is None:
        return None
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        return calc
    inner = getattr(atoms, "atoms", None)
    if inner is None:
        return None
    return getattr(inner, "calc", None)


def _id_entry_for(calc: Any) -> _IdEntry | None:
    """Return the id-keyed entry for this object. Caller holds ``_calc_guard``."""
    entry = _id_locks.get(id(calc))
    if entry is None or entry.calc is not calc:
        return None
    return entry


def _lock_for(calc: Any) -> threading.Lock | None:
    """Return the adapter lock for ``calc``.

    A weak-referenceable calculator is stored in ``_calc_locks``. A
    calculator that cannot be weakly referenced borrows the entry an
    optimizer already holds, and a lookup does not create one.
    """
    if calc is None or bool(getattr(calc, "reentrant", False)):
        return None
    try:
        with _calc_table:
            lock = _calc_locks.get(calc)
            if lock is None:
                lock = threading.Lock()
                _calc_locks[calc] = lock
            return lock
    except TypeError:
        with _calc_table:
            entry = _id_entry_for(calc)
            if entry is None:
                return None
            return entry.lock


def _release_id_entry(entry: _IdEntry) -> None:
    """Queue one hold release. May run inside GC while ``_calc_guard`` is held."""
    _pending_release.append(entry)
    if _calc_guard.acquire(blocking=False):
        try:
            _drain_releases()
        finally:
            _calc_guard.release()


def _drain_releases() -> None:
    """Apply queued releases. Caller holds ``_calc_guard``."""
    while True:
        try:
            entry = _pending_release.popleft()
        except IndexError:
            return
        if entry.holders <= 0:
            continue
        entry.holders -= 1
        if entry.holders > 0:
            continue
        key = id(entry.calc)
        if _id_locks.get(key) is entry:
            del _id_locks[key]


class Optimizer(_ASEOptimizer):
    """One rgmin iteration per :meth:`step`.

    ``method`` selects the solver. ``maxstep`` is the per-atom cap in
    angstroms, except on :class:`FIRE` and :class:`FIRE2`, where it is
    the Euclidean length of the whole step, and on :class:`LBFGS`, where
    a positive value is ASE's length control. The default is 0.2.
    ``accept`` overrides the acceptance rule. Remaining solver keywords
    are validated by :class:`rgmin.SolverConfig`. ``master``, ``comm``,
    and ``loginterval`` are forwarded to the ASE base class when that
    class accepts them.

    ``restart`` is an evaluation log (``rgmin-replay-1``), not ASE's
    Hessian or velocity arrays. A file ASE wrote raises
    ``RestartError``. The log replays under the same parameters; a
    step that asks for a point the log does not contain raises
    ``RestartError`` as well.

    A non-conservative optimizable (NEB, dimer, min-mode) uses
    ``accept="step"`` and the L-BFGS NEB guards. A plain atoms object
    or a cell filter keeps energy acceptance. ``accept=`` overrides
    both.

    ASE calculators are not thread-safe. Each thread needs its own
    calculator. rgmin guarantees that a :class:`~rgmin.SolverConfig`
    can be shared, that one :class:`~rgmin.Solver` runs on one thread
    (a second call raises :class:`RuntimeError`), and that the
    objective receives an owned copy of the coordinates. The adapter
    lock stops calls on one calculator from overlapping unless that
    calculator sets ``reentrant`` to true. A calculator that cannot be
    weakly referenced is locked by ``id``. Each optimizer holds that
    entry, and ``close`` or collection releases the hold. The entry
    stays while any optimizer still holds it, and it is removed when
    the last hold is released. The lock does not make a shared
    calculator safe.
    """

    defaults = {"maxstep": 0.2}

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        method: str = "lbfgs",
        maxstep: float | None = None,
        accept: str | None = None,
        hessian: Callable[..., Any] | None = None,
        **options: Any,
    ) -> None:
        self._rgmin_method = method
        self._rgmin_accept = accept
        self._hessian = hessian
        self._rgmin_options = dict(options)
        ase_options = _ase_passthrough(self._rgmin_options)
        if maxstep is None:
            maxstep = float(self.defaults["maxstep"])
        self.maxstep = float(maxstep)
        self._solver: rgmin.Solver | None = None
        self.stop: rgmin.Stop | None = None
        self._history: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] | None = None
        self._x_session: np.ndarray | None = None
        self._x_placed: np.ndarray | None = None
        self._e_last: float | None = None
        self._r_last: np.ndarray | None = None
        self._id_entry: _IdEntry | None = None
        self._id_finalizer: weakref.finalize[[_IdEntry], Optimizer] | None = None
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            **ase_options,
        )
        self._calculator_lock()
        # Solver keywords are refused here, not on the first step.
        rgmin.SolverConfig(self._rgmin_method, **self._config_options())

    def close(self) -> None:
        """Release the id-keyed hold, then close ASE's logfile and trajectory."""
        self._release_id_hold()
        parent_close = getattr(super(), "close", None)
        if callable(parent_close):
            parent_close()

    def _release_id_hold(self) -> None:
        finalizer = self._id_finalizer
        self._id_finalizer = None
        self._id_entry = None
        if finalizer is not None:
            finalizer()

    def _hold_id_lock(self, calc: Any) -> threading.Lock:
        if self._id_entry is not None and self._id_entry.calc is calc:
            return self._id_entry.lock
        self._release_id_hold()
        with _calc_table:
            entry = _id_entry_for(calc)
            if entry is None:
                entry = _IdEntry(calc)
                _id_locks[id(calc)] = entry
            entry.holders += 1
            lock = entry.lock
        self._id_entry = entry
        self._id_finalizer = weakref.finalize(self, _release_id_entry, entry)
        return lock

    def _calculator_lock(self) -> threading.Lock | None:
        calc = _calculator_of(self.optimizable)
        if calc is None or bool(getattr(calc, "reentrant", False)):
            self._release_id_hold()
            return None
        try:
            with _calc_table:
                lock = _calc_locks.get(calc)
                if lock is None:
                    lock = threading.Lock()
                    _calc_locks[calc] = lock
            if self._id_entry is not None:
                self._release_id_hold()
            return lock
        except TypeError:
            return self._hold_id_lock(calc)

    def _needs_step_accept(self) -> bool:
        return _is_nonconservative(self.optimizable) or _is_nonconservative(self.atoms)

    def _config_options(self) -> dict[str, Any]:
        opts = dict(self._rgmin_options)
        opts["maxstep"] = self.maxstep
        # ASE decides convergence with fmax. A positive session gtol
        # would refuse a step while the unscaled force is still above
        # that fmax, which is what stalls BFGS at alpha=70.
        opts.setdefault("gtol", 0.0)
        if self._rgmin_method == "lbfgs" and "linesearch" not in opts and _is_cell_filter(self.atoms):
            opts["linesearch"] = "wolfe"
        if self._rgmin_accept is not None:
            opts["accept"] = self._rgmin_accept
        elif self._needs_step_accept():
            if "linesearch" in opts:
                raise ValueError(
                    f"linesearch={opts['linesearch']!r} has no effect with accept='step', which "
                    "rgmin.ase takes on a non-conservative optimizable (NEB, dimer, min-mode). "
                    "Pass accept='energy' or accept='nonmonotone' to line search."
                )
            opts["accept"] = "step"
            opts.setdefault("neb_guards", True)
        return opts

    def initialize(self) -> None:
        self._solver = None
        self._history = []
        self._pending = None
        self._x_session = None
        self._x_placed = None
        self._e_last = None
        self._r_last = None

    @property
    def _scale(self) -> float:
        """Factor applied to the value and gradient the session sees."""
        return 1.0

    def _apply_session(self, solver: rgmin.Solver) -> None:
        """Set step rules that are not :class:`~rgmin.SolverConfig` keys."""

    def _before_step(self, solver: rgmin.Solver) -> bool:
        """Return true after clearing the session memory for this step."""
        return False

    def _wrapper_state(self) -> dict[str, Any]:
        return {}

    def _load_wrapper_state(self, state: dict[str, Any]) -> None:
        return None

    def _open(self) -> rgmin.Solver:
        geom = _Geometry(self.optimizable)
        if self._solver is None or self._solver.dim != geom.ndofs():
            config = rgmin.SolverConfig(self._rgmin_method, **self._config_options())
            self._solver = rgmin.Solver(geom.ndofs(), config)
            self._apply_session(self._solver)
            pending = self._pending
            self._pending = None
            if pending:
                self._replay(pending)
        assert self._solver is not None
        return self._solver

    def _start_vector(self) -> np.ndarray:
        geom = _Geometry(self.optimizable)
        x = np.array(geom.positions(), dtype=np.float64, copy=True)
        expected = self._x_placed
        if expected is None and self._x_session is not None:
            expected = self._x_session
        if (
            self._x_session is not None
            and expected is not None
            and expected.shape == x.shape
            and np.allclose(x, expected, rtol=0.0, atol=_RESTART_POSITION_ATOL)
        ):
            return np.array(self._x_session, dtype=np.float64, copy=True)
        return x

    def _evaluate(self, x: np.ndarray, record: list[list[Any]] | None = None) -> tuple[float, np.ndarray]:
        geom = _Geometry(self.optimizable)
        point = np.asarray(x, dtype=np.float64).reshape(-1)

        def once() -> tuple[float, np.ndarray]:
            current = geom.positions()
            if point.shape != current.shape or not np.array_equal(point, current):
                geom.write(point)
            value = geom.value()
            gradient = np.array(geom.gradient(), dtype=np.float64, copy=True)
            if record is not None:
                record.append([point.tolist(), value, gradient.tolist()])
            scale = self._scale
            return value * scale, gradient * scale

        lock = self._calculator_lock()
        if lock is None:
            return once()
        with lock:
            return once()

    def _hessian_of(self, x: np.ndarray) -> np.ndarray:
        if self._hessian is None:
            raise ValueError(
                f"{type(self).__name__} requires hessian(x); "
                "the calculator and rgmin adapter do not provide a Hessian"
            )
        return np.asarray(self._hessian(np.asarray(x, dtype=np.float64)), dtype=np.float64)

    def _replay(self, steps: list[dict[str, Any]]) -> None:
        solver = self._solver
        assert solver is not None
        scale = self._scale
        updated: np.ndarray | None = None
        for index, entry in enumerate(steps):
            cursor = iter(entry["evals"])

            def answer(
                x: np.ndarray, cursor: Any = cursor, index: int = index
            ) -> tuple[float, np.ndarray]:
                try:
                    point, value, gradient = next(cursor)
                except StopIteration:
                    raise _restart_error(
                        f"{self.restart}: step {index} asks for more evaluations than were recorded"
                    ) from None
                if not np.allclose(x, point, rtol=0.0, atol=_RESTART_REPLAY_ATOL):
                    raise _restart_error(
                        f"{self.restart}: step {index} does not replay under these parameters"
                    )
                return float(value) * scale, np.asarray(gradient, dtype=np.float64) * scale

            if entry.get("forget"):
                solver.forget()
            updated = np.asarray(entry["x"], dtype=np.float64).copy()
            if self._rgmin_method in _HESSIAN_METHODS:
                solver.step_hess(answer, self._hessian_of, updated)
            else:
                solver.step(answer, updated)
            if next(cursor, None) is not None:
                raise _restart_error(
                    f"{self.restart}: step {index} recorded more evaluations than it replays"
                )
        if updated is None:
            return
        geom = _Geometry(self.optimizable)
        if not np.allclose(geom.positions(), updated, rtol=0.0, atol=_RESTART_POSITION_ATOL):
            geom.write(updated)
        self._x_session = np.array(updated, dtype=np.float64, copy=True)
        self._x_placed = np.array(geom.positions(), dtype=np.float64, copy=True)

    def _dump_history(self) -> None:
        if self.restart is None:
            return
        data: dict[str, Any] = {
            "format": RESTART_FORMAT,
            "optimizer": type(self).__name__,
            "method": self._rgmin_method,
            "steps": self._history,
            "wrapper": self._wrapper_state(),
        }
        orig = getattr(self.atoms, "orig_cell", None)
        if orig is not None:
            data["orig_cell"] = np.asarray(orig, dtype=np.float64).tolist()
        self.dump(data)

    def step(self, forces: Any = None) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        solver = self._open()
        reset = self._before_step(solver)
        geom = _Geometry(self.optimizable)
        x = self._start_vector()
        x0 = x.tolist()
        record: list[list[Any]] = []
        if self._rgmin_method in _HESSIAN_METHODS:
            report = solver.step_hess(
                lambda point: self._evaluate(point, record), self._hessian_of, x
            )
        else:
            report = solver.step(lambda point: self._evaluate(point, record), x)
        self.stop = report.stop
        if not np.array_equal(x, geom.positions()):
            geom.write(x)
        self._x_session = np.array(x, dtype=np.float64, copy=True)
        self._x_placed = np.array(geom.positions(), dtype=np.float64, copy=True)
        if self.restart is not None:
            entry: dict[str, Any] = {"x": x0, "evals": record}
            if reset:
                entry["forget"] = True
            self._history.append(entry)
            self._dump_history()

    def _stop_requested(self) -> bool:
        return self.stop in (rgmin.Stop.ENERGY_TOLERANCE, rgmin.Stop.EVAL_BUDGET)

    def converged(self, *args: Any, **kwargs: Any) -> bool:
        """ASE's force test, or ``True`` once ``etol`` or ``max_evals`` stopped the solver."""
        return self._stop_requested() or bool(cast(Any, super()).converged(*args, **kwargs))

    def gradient_converged(self, gradient: Any = None) -> bool:
        """The ``converged`` of ASE 3.2x and later, which its run loop calls."""
        if self._stop_requested():
            return True
        parent = cast(Any, super())
        check = getattr(parent, "gradient_converged", None)
        if callable(check):
            return bool(check(gradient))
        if gradient is None:
            return bool(parent.converged())
        forces = -np.asarray(gradient, dtype=np.float64)
        get_forces = getattr(self.optimizable, "get_forces", None)
        if callable(get_forces):
            shape = np.asarray(get_forces(), dtype=np.float64).shape
            forces = forces.reshape(shape)
        return bool(parent.converged(forces=forces))

    def run(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self, fmax: float = 0.05, steps: int | None = None
    ) -> bool:
        """Run to ``fmax``, an ``etol`` stop, or the ``max_evals`` budget.

        ``etol`` and ``max_evals`` are :class:`rgmin.SolverConfig` options.
        The return value is ``True`` when the forces or the energy change
        met their tolerance and ``False`` when the step count or the
        evaluation budget ran out. :attr:`stop` is the :class:`rgmin.Stop`
        of the last step, ``None`` before the first.
        """
        self.stop = None
        if steps is None:
            converged = cast(Any, super()).run(fmax=fmax)
        else:
            converged = cast(Any, super()).run(fmax=fmax, steps=steps)
        return bool(converged) and self.stop is not rgmin.Stop.EVAL_BUDGET

    def _log_fmax(self, gradient: Any, forces: Any) -> float:
        if gradient is not None and hasattr(self.optimizable, "gradient_norm"):
            return float(self.optimizable.gradient_norm(gradient))
        vector = gradient if gradient is not None else forces
        if vector is None:
            vector = _Geometry(self.optimizable).gradient()
        block = np.asarray(vector, dtype=np.float64).reshape(-1, 3)
        return float(np.sqrt((block * block).sum(axis=1).max()))

    def log(self, *args: Any, **kwargs: Any) -> None:
        """Write one ASE optimizer line.

        The columns are ``Step``, ``Time``, ``Energy``, and ``fmax``.
        """
        if self.logfile is None:
            return
        gradient = args[0] if args else kwargs.get("gradient")
        forces = kwargs.get("forces")
        if gradient is None and forces is None and not args:
            gradient = _Geometry(self.optimizable).gradient()
        fmax = self._log_fmax(gradient, forces)
        energy = _Geometry(self.optimizable).value()
        now = time.localtime()
        name = type(self).__name__
        write = cast(Any, self.logfile).write
        if self.nsteps == 0:
            header = "%s  %4s %8s %15s  %12s\n" % (
                " " * len(name),
                "Step",
                "Time",
                "Energy",
                "fmax",
            )
            write(header)
        line = "%s:  %3d %02d:%02d:%02d %15.6f %15.6f\n" % (
            name,
            self.nsteps,
            now.tm_hour,
            now.tm_min,
            now.tm_sec,
            energy,
            fmax,
        )
        write(line)
        cast(Any, self.logfile).flush()

    def read(self) -> None:
        data = self.load()
        self.initialize()
        if not isinstance(data, dict) or data.get("format") != RESTART_FORMAT:
            raise _restart_error(
                f"{self.restart} is not an rgmin restart file. "
                "rgmin optimizers replay their own evaluation log, not ASE's arrays."
            )
        if data.get("optimizer") != type(self).__name__:
            raise _restart_error(
                f"{self.restart} was written by {data.get('optimizer')}, not {type(self).__name__}"
            )
        if data.get("method") != self._rgmin_method:
            raise _restart_error(
                f"{self.restart} was written for method {data.get('method')}, not {self._rgmin_method}"
            )
        orig = data.get("orig_cell")
        if orig is not None and hasattr(self.atoms, "orig_cell"):
            self.atoms.orig_cell = np.asarray(orig, dtype=np.float64)
        self._history = list(data.get("steps", []))
        self._pending = list(self._history)
        wrapper = data.get("wrapper", {})
        self._load_wrapper_state(wrapper if isinstance(wrapper, dict) else {})

    def replay_trajectory(self, traj: Any) -> Any:
        """Reinitialize supported optimizer state from an ASE trajectory."""
        from contextlib import ExitStack

        from ase.io.trajectory import Trajectory

        replay = getattr(self, "_replay_trajectory", None)
        if replay is None:
            raise TypeError("This optimizer does not provide _replay_trajectory()")
        with ExitStack() as stack:
            if isinstance(traj, str):
                traj = stack.enter_context(Trajectory(traj, "r"))
            return replay(traj)

    def todict(self) -> dict[str, Any]:
        description = dict(cast(Any, super()).todict())
        description["method"] = self._rgmin_method
        description.update(self._rgmin_options)
        if self._rgmin_accept is not None:
            description["accept"] = self._rgmin_accept
        return description


def _start(
    opt: Optimizer,
    atoms: Any,
    restart: str | None,
    logfile: Any,
    trajectory: Any,
    append_trajectory: bool,
    method: str,
    maxstep: float | None,
    accept: str | None,
    options: dict[str, Any],
    hessian: Callable[..., Any] | None = None,
) -> None:
    Optimizer.__init__(
        opt,
        atoms,
        restart=restart,
        logfile=logfile,
        trajectory=trajectory,
        append_trajectory=append_trajectory,
        method=method,
        maxstep=maxstep,
        accept=accept,
        hessian=hessian,
        **options,
    )


class CG(Optimizer):
    """Nonlinear conjugate gradient.

    The default ``beta`` is Polak-Ribiere, the C ABI method
    ``RGMIN_POLAK_RIBIERE``. ``beta`` selects another conjugacy variant
    exposed by that ABI.

    Polak and Ribiere, Rev. Fr. Inform. Rech. Oper. 3, 35 (1969),
    https://doi.org/10.1051/m2an/196903R100351.
    Nocedal and Wright, *Numerical Optimization*,
    https://doi.org/10.1007/978-0-387-40065-5.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        beta: str = "polak_ribiere",
        **options: Any,
    ) -> None:
        options["beta"] = beta
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "cg",
            maxstep,
            accept,
            options,
        )


class LBFGS(Optimizer):
    """Limited-memory BFGS with the ASE ``LBFGS`` constructor.

    The default is the line-search-free eOn step, whose first step
    probes curvature. ``eon_cap=False`` opts into ASE's fixed
    ``H0 = I / alpha`` and length control, which keeps ``memory`` pairs
    when it shortens a step to ``maxstep``.

    Differences from ASE: ``damping`` multiplies ``maxstep`` (ASE
    damps after the cap; ``damping=1`` agrees). A pair with too little
    curvature is dropped. ``use_line_search=True`` is rgmin's strong
    Wolfe search, not ASE's ``LineSearch``. A non-positive ``maxstep``
    restores the eOn cap of 0.2, and a negative ``maxmove`` turns every
    cap off.

    ASE cell filters use strong Wolfe search because their final
    coordinates are scaled cell deformations rather than Cartesian
    atom positions. Other optimizables keep the capped eOn step.

    Liu and Nocedal, Mathematical Programming 45, 503 (1989),
    https://doi.org/10.1007/BF01589116.
    Nocedal and Wright, *Numerical Optimization*,
    https://doi.org/10.1007/978-0-387-40065-5.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        maxstep: float | None = None,
        memory: int = 100,
        damping: float = 1.0,
        alpha: float = 70.0,
        use_line_search: bool = False,
        *,
        append_trajectory: bool = False,
        accept: str | None = None,
        eon_cap: bool = True,
        **options: Any,
    ) -> None:
        if maxstep is None:
            maxstep = float(self.defaults["maxstep"])
        self.maxstep = float(maxstep)
        if self.maxstep > 1.0:
            raise ValueError(
                "You are using a much too large value for the maximum step size: %.1f Angstrom" % self.maxstep
            )
        self.memory = int(memory)
        self.damping = float(damping)
        self.H0_scale = 1.0 / float(alpha)
        self.use_line_search = bool(use_line_search)
        self.eon_cap = bool(eon_cap)
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "lbfgs",
            self.maxstep,
            accept,
            options,
        )

    def _ase_scaled(self) -> bool:
        if self.eon_cap or self.use_line_search:
            return False
        if "linesearch" in self._rgmin_options or "maxmove" in self._rgmin_options:
            return False
        return self.maxstep > 0.0

    def _config_options(self) -> dict[str, Any]:
        opts = super()._config_options()
        opts["memory"] = self.memory
        if self.use_line_search:
            opts.setdefault("linesearch", "wolfe")
        if self._ase_scaled():
            opts.pop("maxstep", None)
        return opts

    def _apply_session(self, solver: rgmin.Solver) -> None:
        if not self._ase_scaled():
            return
        cap = self.maxstep * self.damping if self.damping > 0.0 else self.maxstep
        solver.set_lbfgs_scaled(cap)
        solver.set_lbfgs_fixed_h0(self.H0_scale)

    def _replay_trajectory(self, traj: Any) -> None:
        """Push the secant pairs of a trajectory into a fresh memory."""
        self.initialize()
        solver = self._open()
        frames = list(traj)
        for before, after in zip(frames[:-1], frames[1:]):
            s = (np.asarray(after.get_positions()) - np.asarray(before.get_positions())).reshape(-1)
            y = (np.asarray(before.get_forces()) - np.asarray(after.get_forces())).reshape(-1)
            solver.push_pair(np.ascontiguousarray(s, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64))


class FIRE(Optimizer):
    """FIRE with the ASE ``FIRE`` constructor.

    ``maxstep`` caps the Euclidean length of the whole displacement.
    ``maxmove`` is ASE's deprecated alias and warns the same way. The
    session counts the current downhill step before comparing with
    ``Nmin`` and ASE after, so the session receives ``Nmin + 1``.

    ``downhill_check=True`` compares the energy with the last kept one
    at the start of each step. After a rise it calls
    ``position_reset_callback(optimizable, r_last, e, e_last)``, moves
    back to ``r_last`` and clears the velocity. The time step is kept,
    where ASE halves it. ``a`` other than ``astart`` warns and is not
    used. A non-positive ``maxstep`` clears the per-atom cap instead of
    installing a Euclidean one.

    Bitzek, Koskinen, Gaehler, Moseler, and Gumbsch, Phys. Rev. Lett. 97,
    170201 (2006), https://doi.org/10.1103/PhysRevLett.97.170201.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
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
        *,
        append_trajectory: bool = False,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        if maxmove is not None:
            warnings.warn("Use of `maxmove` is deprecated. Use `maxstep` instead.", FutureWarning, stacklevel=2)
            if maxstep is None:
                maxstep = maxmove
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
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "fire",
            maxstep,
            accept,
            options,
        )

    def _euclidean_cap(self) -> bool:
        return self.maxstep > 0.0 and "maxmove" not in self._rgmin_options

    def _config_options(self) -> dict[str, Any]:
        opts = super()._config_options()
        if self._euclidean_cap():
            opts.pop("maxstep", None)
        if self.downhill_check and "accept" not in opts:
            opts["accept"] = "energy"
        return opts

    def _apply_session(self, solver: rgmin.Solver) -> None:
        if self._euclidean_cap():
            solver.set_atom_maxmove(0.0)
            solver.set_maxmove(self.maxstep)
        solver.set_fire_config(self.dt, self.dtmax, self.Nmin + 1, self.finc, self.fdec, self.astart, self.fa)

    def _before_step(self, solver: rgmin.Solver) -> bool:
        if not self.downhill_check:
            return False
        reset = False
        if self._e_last is not None and self._r_last is not None:
            energy = float(_Geometry(self.optimizable).value())
            if energy > self._e_last:
                if self.position_reset_callback is not None:
                    self.position_reset_callback(self.optimizable, self._r_last, energy, self._e_last)
                _Geometry(self.optimizable).write(self._r_last)
                solver.forget()
                self._x_session = None
                self._x_placed = None
                reset = True
        geom = _Geometry(self.optimizable)
        self._e_last = float(geom.value())
        self._r_last = np.array(geom.positions(), dtype=np.float64, copy=True)
        return reset

    def _wrapper_state(self) -> dict[str, Any]:
        if self._e_last is None or self._r_last is None:
            return {}
        return {"e_last": self._e_last, "r_last": self._r_last.tolist()}

    def _load_wrapper_state(self, state: dict[str, Any]) -> None:
        if "e_last" in state:
            self._e_last = float(state["e_last"])
            self._r_last = np.asarray(state["r_last"], dtype=np.float64)


class FIRE2(Optimizer):
    """FIRE 2.0 with the ASE ``FIRE2`` constructor.

    The session runs Guénolé's semi-implicit Euler integrator.
    ``maxstep`` caps the Euclidean length of the displacement, or each
    component of it with ``use_abc=True``. ``position_reset_callback``
    is accepted and, as in ASE, not called. The time-step floor is
    ``0.02 * dt``; another ``dtmin`` is a ``ValueError``. A non-positive
    ``maxstep`` clears the per-atom cap instead of installing a
    Euclidean one.

    Guenole, Noehring, Vaid, Houlle, Xie, Prakash, and Bitzek, Comput.
    Mater. Sci. 175, 109584 (2020),
    https://doi.org/10.1016/j.commatsci.2020.109584.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        dt: float = 0.1,
        maxstep: float | None = None,
        dtmax: float = 1.0,
        dtmin: float = 2e-3,
        Nmin: int = 20,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.25,
        fa: float = 0.99,
        position_reset_callback: Any = None,
        use_abc: bool = False,
        *,
        append_trajectory: bool = False,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        self.dt = float(dt)
        self.dtmax = float(dtmax)
        self.dtmin = float(dtmin)
        if not np.isclose(self.dtmin, 0.02 * self.dt, rtol=_PARAMETER_RTOL, atol=0.0):
            raise ValueError(
                f"rgmin FIRE2 floors the time step at 0.02 * dt = {0.02 * self.dt:g}, not dtmin={dtmin:g}"
            )
        self.Nmin = int(Nmin)
        self.finc = float(finc)
        self.fdec = float(fdec)
        self.astart = float(astart)
        self.fa = float(fa)
        self.position_reset_callback = position_reset_callback
        self.use_abc = bool(use_abc)
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "fire2",
            maxstep,
            accept,
            options,
        )

    def _euclidean_cap(self) -> bool:
        return self.maxstep > 0.0 and "maxmove" not in self._rgmin_options

    def _config_options(self) -> dict[str, Any]:
        opts = super()._config_options()
        if self._euclidean_cap():
            opts.pop("maxstep", None)
        return opts

    def _apply_session(self, solver: rgmin.Solver) -> None:
        solver.set_fire_preset("guenole2020")
        if self._euclidean_cap():
            solver.set_atom_maxmove(0.0)
            solver.set_maxmove(self.maxstep)
        solver.set_fire_config(self.dt, self.dtmax, self.Nmin, self.finc, self.fdec, self.astart, self.fa)
        solver.set_fire_abc(self.use_abc)


class QuickMin(Optimizer):
    """Quick-min, the session analogue of ASE ``MDMin``.

    ``dt`` is the Euler time step (the session ``istep``). ``maxstep``
    is the per-atom cap, which is how ASE ``MDMin`` clips. ASE has no
    ``QuickMin`` class and no velocity file this class can read; a
    restart is the evaluation log, and an ASE file raises
    ``RestartError``.

    https://doi.org/10.1142/9789812839664_0016.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        dt: float | None = None,
        **options: Any,
    ) -> None:
        if dt is not None:
            options["istep"] = float(dt)
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "quickmin",
            maxstep,
            accept,
            options,
        )


class BFGS(Optimizer):
    """Dense BFGS with the ASE ``BFGS`` constructor.

    ``alpha`` is ASE's initial Hessian ``alpha * I``. The session runs
    inverse BFGS on ``E / alpha`` from the identity, so the first step
    is ``-g / alpha``. ``maxstep`` is the per-atom cap. A value above
    1 Å warns, as ASE does. A pair with negligible curvature in the
    scaled energy is skipped, where ASE updates the Hessian anyway.

    Nocedal and Wright, *Numerical Optimization*,
    https://doi.org/10.1007/978-0-387-40065-5.
    """

    defaults = {"maxstep": 0.2, "alpha": 70.0}

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        alpha: float | None = None,
        *,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        if maxstep is None:
            maxstep = float(self.defaults["maxstep"])
        self.maxstep = float(maxstep)
        if self.maxstep > 1.0:
            warnings.warn(
                "You are using a *very* large value for the maximum step size: %.1f Å" % self.maxstep,
                stacklevel=2,
            )
        self.alpha = self.defaults["alpha"] if alpha is None else float(alpha)
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "bfgs",
            self.maxstep,
            accept,
            options,
        )

    @property
    def _scale(self) -> float:
        return 1.0 / self.alpha

    def _config_options(self) -> dict[str, Any]:
        opts = super()._config_options()
        opts["istep"] = 1.0
        return opts


class SD(Optimizer):
    """Steepest descent.

    Nocedal and Wright, *Numerical Optimization*,
    https://doi.org/10.1007/978-0-387-40065-5.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(self, atoms, restart, logfile, trajectory, append_trajectory, "sd", maxstep, accept, options)


class Adam(Optimizer):
    """Adam.

    Kingma and Ba, https://doi.org/10.48550/arXiv.1412.6980.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self, atoms, restart, logfile, trajectory, append_trajectory, "adam", maxstep, accept, options
        )


class Newton(Optimizer):
    """Newton on ``hessian(x)``, an ``(n, n)`` matrix.

    Nocedal and Wright, *Numerical Optimization*,
    https://doi.org/10.1007/978-0-387-40065-5.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        hessian: Callable[..., Any] | None = None,
        **options: Any,
    ) -> None:
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "newton",
            maxstep,
            accept,
            options,
            hessian,
        )


class RFO(Optimizer):
    """Rational function optimization on ``hessian(x)``.

    Banerjee, Adams, Simons, and Shepard, J. Phys. Chem. 89, 52 (1985),
    https://doi.org/10.1021/j100247a015.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        hessian: Callable[..., Any] | None = None,
        **options: Any,
    ) -> None:
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "rfo",
            maxstep,
            accept,
            options,
            hessian,
        )


class SR1(Optimizer):
    """Symmetric-rank-one quasi-Newton.

    Nocedal and Wright, *Numerical Optimization*,
    https://doi.org/10.1007/978-0-387-40065-5.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self, atoms, restart, logfile, trajectory, append_trajectory, "sr1", maxstep, accept, options
        )


class SR2(Optimizer):
    """SR2 quasi-Newton update.

    Nocedal and Wright, *Numerical Optimization*,
    https://doi.org/10.1007/978-0-387-40065-5.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self, atoms, restart, logfile, trajectory, append_trajectory, "sr2", maxstep, accept, options
        )


class BB(Optimizer):
    """Barzilai-Borwein spectral steepest descent.

    Barzilai and Borwein, IMA J. Numer. Anal. 8, 141 (1988),
    https://doi.org/10.1093/imanum/8.1.141.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(self, atoms, restart, logfile, trajectory, append_trajectory, "bb", maxstep, accept, options)


class Dogleg(Optimizer):
    """Powell dogleg on ``hessian(x)``.

    Nocedal and Wright, algorithm 4.1,
    https://doi.org/10.1007/978-0-387-40065-5.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        hessian: Callable[..., Any] | None = None,
        **options: Any,
    ) -> None:
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "dogleg",
            maxstep,
            accept,
            options,
            hessian,
        )


class DIIS(Optimizer):
    """Pulay residual subspace extrapolation.

    Pulay, Chem. Phys. Lett. 73, 393 (1980),
    https://doi.org/10.1016/0009-2614(80)80396-4.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self, atoms, restart, logfile, trajectory, append_trajectory, "diis", maxstep, accept, options
        )


class PSO(Optimizer):
    """Particle swarm.

    The C ABI exposes this method, so this class takes one swarm step
    per :meth:`Optimizer.step` and writes the same log columns as the
    local optimizers. It is not a local force relaxation, and the
    parity checks do not use it.

    Kennedy and Eberhart, Proc. ICNN (1995),
    https://doi.org/10.1109/ICNN.1995.488968.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self, atoms, restart, logfile, trajectory, append_trajectory, "pso", maxstep, accept, options
        )


class FletcherReeves(Optimizer):
    """Fletcher-Reeves conjugate gradient.

    Fletcher and Reeves, Comput. J. 7, 149 (1964),
    https://doi.org/10.1093/comjnl/7.2.149.
    Fletcher, *Practical Methods of Optimization*,
    https://doi.org/10.1002/9781118723203.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "fletcher_reeves",
            maxstep,
            accept,
            options,
        )


class HestenesStiefel(Optimizer):
    """Hestenes-Stiefel conjugate gradient.

    Hestenes and Stiefel, J. Res. Natl. Bur. Stand. 49, 409 (1952),
    https://doi.org/10.6028/jres.049.044.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "hestenes_stiefel",
            maxstep,
            accept,
            options,
        )


class DaiYuan(Optimizer):
    """Dai-Yuan conjugate gradient.

    Dai and Yuan, SIAM J. Optim. 10, 177 (1999),
    https://doi.org/10.1137/S1052623497318992.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "dai_yuan",
            maxstep,
            accept,
            options,
        )


class ConjugateDescent(Optimizer):
    """Fletcher conjugate descent.

    Fletcher, Practical Methods of Optimization,
    https://doi.org/10.1002/9781118723203.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "conjugate_descent",
            maxstep,
            accept,
            options,
        )


class HagerZhang(Optimizer):
    """Hager-Zhang conjugate gradient.

    Hager and Zhang, SIAM J. Optim. 16, 170 (2005),
    https://doi.org/10.1137/030601880.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "hager_zhang",
            maxstep,
            accept,
            options,
        )


class LiuStorey(Optimizer):
    """Liu-Storey conjugate gradient.

    Liu and Storey, J. Optim. Theory Appl. 69, 129 (1991),
    https://doi.org/10.1007/BF00940464.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self,
            atoms,
            restart,
            logfile,
            trajectory,
            append_trajectory,
            "liu_storey",
            maxstep,
            accept,
            options,
        )


class FrPr(Optimizer):
    """Gilbert-Nocedal FR-PR hybrid.

    Nocedal and Wright, *Numerical Optimization*,
    https://doi.org/10.1007/978-0-387-40065-5.
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self, atoms, restart, logfile, trajectory, append_trajectory, "fr_pr", maxstep, accept, options
        )


__all__ = [
    "Adam",
    "BB",
    "BFGS",
    "CG",
    "ConjugateDescent",
    "DIIS",
    "DaiYuan",
    "Dogleg",
    "FIRE",
    "FIRE2",
    "FletcherReeves",
    "FrPr",
    "HagerZhang",
    "HestenesStiefel",
    "LBFGS",
    "LiuStorey",
    "Newton",
    "Optimizer",
    "PSO",
    "QuickMin",
    "RFO",
    "SD",
    "SR1",
    "SR2",
]
