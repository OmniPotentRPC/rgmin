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


def _is_nonconservative(obj: Any) -> bool:
    if obj is None:
        return False
    kinds = _nonconservative_types()
    if kinds and isinstance(obj, kinds):
        return True
    probe = getattr(obj, "is_neb", None)
    return bool(callable(probe) and probe())


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
    angstroms. The default is 0.2, the same value as ASE
    ``Optimizer.defaults`` and as the core session. ``accept`` overrides
    the acceptance rule. Remaining
    solver keywords are validated by :class:`rgmin.SolverConfig`.
    ``master``, ``comm``, and ``loginterval`` are forwarded to the ASE
    base class when that class accepts them.

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
        if restart is not None:
            raise NotImplementedError(f"{type(self).__name__}.read is not implemented")
        self._rgmin_method = method
        self._rgmin_accept = accept
        self._hessian = hessian
        self._rgmin_options = dict(options)
        ase_options = _ase_passthrough(self._rgmin_options)
        if maxstep is None:
            maxstep = float(self.defaults["maxstep"])
        self.maxstep = float(maxstep)
        self._solver: rgmin.Solver | None = None
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
        if self._rgmin_accept is not None:
            opts["accept"] = self._rgmin_accept
        elif self._needs_step_accept():
            opts["accept"] = "step"
            opts.setdefault("neb_guards", True)
        return opts

    def initialize(self) -> None:
        opts = self._config_options()
        config = rgmin.SolverConfig(self._rgmin_method, **opts)
        self._solver = rgmin.Solver(_Geometry(self.optimizable).ndofs(), config)

    def _evaluate(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        geom = _Geometry(self.optimizable)

        def once() -> tuple[float, np.ndarray]:
            current = geom.positions()
            if x.shape != current.shape or not np.array_equal(x, current):
                geom.write(x)
            value = geom.value()
            gradient = np.array(geom.gradient(), dtype=np.float64, copy=True)
            return value, gradient

        lock = self._calculator_lock()
        if lock is None:
            return once()
        with lock:
            return once()

    def _hessian_of(self, x: np.ndarray) -> np.ndarray:
        if self._hessian is None:
            raise NotImplementedError(f"{type(self).__name__}.step requires a hessian callable")
        return np.asarray(self._hessian(x), dtype=np.float64)

    def step(self) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        geom = _Geometry(self.optimizable)
        if self._solver is None or self._solver.dim != geom.ndofs():
            self.initialize()
        assert self._solver is not None
        if self._rgmin_method in _HESSIAN_METHODS and self._hessian is None:
            raise NotImplementedError(f"{type(self).__name__}.step requires a hessian callable")
        x = np.array(geom.positions(), dtype=np.float64, copy=True)
        if self._rgmin_method in _HESSIAN_METHODS:
            self._solver.step_hess(self._evaluate, self._hessian_of, x)
        else:
            self._solver.step(self._evaluate, x)
        if not np.array_equal(x, geom.positions()):
            geom.write(x)

    def run(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self, fmax: float = 0.05, steps: int | None = None
    ) -> bool:
        if steps is None:
            converged = cast(Any, super()).run(fmax=fmax)
        else:
            converged = cast(Any, super()).run(fmax=fmax, steps=steps)
        return bool(converged)

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
        raise NotImplementedError(f"{type(self).__name__}.read is not implemented")

    def dump(self, data: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__}.dump is not implemented")

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
    """Limited-memory BFGS.

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
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self, atoms, restart, logfile, trajectory, append_trajectory, "lbfgs", maxstep, accept, options
        )


class FIRE(Optimizer):
    """FIRE inertial relaxation.

    Bitzek, Koskinen, Gaehler, Moseler, and Gumbsch, Phys. Rev. Lett. 97,
    170201 (2006), https://doi.org/10.1103/PhysRevLett.97.170201.
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
            self, atoms, restart, logfile, trajectory, append_trajectory, "fire", maxstep, accept, options
        )


class FIRE2(Optimizer):
    """FIRE 2.0.

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
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        _start(
            self, atoms, restart, logfile, trajectory, append_trajectory, "fire2", maxstep, accept, options
        )


class QuickMin(Optimizer):
    """Quick-min: project the velocity onto the force, then an Euler step.

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
        **options: Any,
    ) -> None:
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
    """Dense BFGS.

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
            self, atoms, restart, logfile, trajectory, append_trajectory, "bfgs", maxstep, accept, options
        )


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
