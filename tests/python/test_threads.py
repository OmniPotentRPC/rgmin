# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Independent solvers on several threads."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

import rgmin


def _quad(x: np.ndarray) -> tuple[float, np.ndarray]:
    return 0.5 * float(x @ x), np.array(x, dtype=np.float64, copy=True)


# Spawns a ThreadPoolExecutor. The marker keeps pytest-run-parallel from
# nesting another pool around this test.
@pytest.mark.thread_unsafe
def test_independent_solvers_match() -> None:
    def once() -> bytes:
        x = np.array([0.4, -0.2, 0.7, 0.1], dtype=np.float64)
        rgmin.minimize(_quad, x, method="lbfgs", memory=8, gtol=1e-14, maxiter=20)
        return x.tobytes()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _i: once(), range(8)))
    assert len(set(results)) == 1


# Spawns a ThreadPoolExecutor. The marker keeps pytest-run-parallel from
# nesting another pool around this test.
@pytest.mark.thread_unsafe
def test_shared_config_is_immutable() -> None:
    config = rgmin.SolverConfig("sd", maxiter=5, gtol=1e-10)
    barrier = threading.Barrier(8)

    def once() -> float:
        barrier.wait()
        x = np.array([1.0, -1.0], dtype=np.float64)
        solver = rgmin.Solver(2, config)
        report = solver.step(_quad, x)
        return report.value

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _i: once(), range(8)))
    assert max(values) - min(values) == 0.0


# Spawns threading.Thread. The marker keeps pytest-run-parallel from
# nesting another pool around this test.
@pytest.mark.thread_unsafe
def test_hammered_solver_raises_in_use() -> None:
    config = rgmin.SolverConfig("sd", maxiter=1)
    solver = rgmin.Solver(2, config)
    entered = threading.Event()
    release = threading.Event()

    def hold(x: np.ndarray) -> tuple[float, np.ndarray]:
        entered.set()
        release.wait(timeout=5)
        return 0.0, np.zeros_like(x)

    x = np.zeros(2)
    holder = threading.Thread(target=lambda: solver.step(hold, x.copy()))
    holder.start()
    assert entered.wait(timeout=5)
    errors: list[BaseException] = []

    def clash() -> None:
        try:
            solver.step(_quad, x.copy())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    workers = [threading.Thread(target=clash) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
    release.set()
    holder.join(timeout=5)
    assert len(errors) == 8
    for error in errors:
        assert isinstance(error, RuntimeError)
        assert "Solver is in use by another thread" in str(error)


# Spawns a ThreadPoolExecutor. The marker keeps pytest-run-parallel from
# nesting another pool around this test. The two atoms share one
# calculator on purpose, to observe the adapter lock.
@pytest.mark.thread_unsafe
def test_shared_calculator_with_and_without_reentrant() -> None:
    ase = pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.calculator import Calculator, PropertyNotImplementedError

    class Shared(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, reentrant: bool) -> None:
            super().__init__()
            self.reentrant = reentrant
            self.inside = 0
            self.max_inside = 0
            self._gate = threading.Lock()
            self._local = threading.local()
            self._meet = threading.Barrier(2)

        def get_property(self, name, atoms=None, allow_calculation=True):  # type: ignore[no-untyped-def]
            if name not in self.implemented_properties:
                raise PropertyNotImplementedError(f"{name} property not implemented")
            if atoms is None:
                atoms = self.atoms
            with self._gate:
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
            try:
                # One meeting per thread. A barrier on every call deadlocks
                # when one step evaluates the calculator more than once.
                if self.reentrant and not getattr(self._local, "met", False):
                    self._local.met = True
                    self._meet.wait(timeout=5)
                x = np.asarray(atoms.get_positions(), dtype=np.float64).reshape(-1)
                energy = 0.5 * float(x @ x)
                forces = -x.reshape(-1, 3)
            finally:
                with self._gate:
                    self.inside -= 1
            if name == "energy":
                return energy
            if name == "forces":
                return forces.copy()
            raise PropertyNotImplementedError(f"{name} not present in this calculation")

    def run(reentrant: bool) -> Shared:
        calc = Shared(reentrant)
        starts = [np.array([[0.2, 0.0, 0.0]]), np.array([[-0.3, 0.1, 0.0]])]
        ready = threading.Barrier(2)

        def once(pos: np.ndarray) -> None:
            atoms = Atoms("H", positions=pos)
            atoms.calc = calc
            from rgmin.ase import SD

            opt = SD(atoms, logfile=None, maxstep=0.05)
            ready.wait(timeout=5)
            opt.step()

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(once, starts))
        return calc

    serial = run(False)
    assert serial.max_inside == 1
    overlapped = run(True)
    assert overlapped.max_inside == 2


def _unweak_calculator() -> Any:
    """Calculator with no weak-reference support and a flat energy."""

    class SlotCalc:
        __slots__ = ("reentrant",)

        def __init__(self) -> None:
            self.reentrant = False

        def get_potential_energy(self, atoms: Any, force_consistent: bool = False) -> float:
            del atoms, force_consistent
            return 0.0

        def get_forces(self, atoms: Any) -> np.ndarray:
            return np.zeros((len(atoms), 3))

    return SlotCalc()


@pytest.mark.thread_unsafe
def test_live_id_locks_survive_interleaved_lookups() -> None:
    """Two live non-weakrefable calculators keep their own locks."""
    pytest.importorskip("ase")
    import weakref

    from ase import Atoms

    from rgmin.ase import SD, _id_locks, _lock_for

    with pytest.raises(TypeError):
        weakref.ref(_unweak_calculator())

    first = _unweak_calculator()
    second = _unweak_calculator()
    lock_first = _lock_for(first)
    lock_second = _lock_for(second)
    assert lock_first is not None and lock_second is not None
    assert lock_first is not lock_second
    assert _lock_for(first) is lock_first
    assert _lock_for(second) is lock_second

    atoms_first = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    atoms_first.calc = first
    atoms_second = Atoms("H", positions=[[-0.2, 0.0, 0.0]])
    atoms_second.calc = second
    opt_first = SD(atoms_first, logfile=None, maxstep=0.05)
    opt_second = SD(atoms_second, logfile=None, maxstep=0.05)
    assert _lock_for(first) is lock_first
    assert _lock_for(second) is lock_second
    assert opt_first._id_entry is not None and opt_second._id_entry is not None
    assert opt_first._id_entry.lock is lock_first
    assert opt_second._id_entry.lock is lock_second

    shared_atoms = Atoms("H", positions=[[0.1, 0.0, 0.0]])
    shared_atoms.calc = first
    opt_shared = SD(shared_atoms, logfile=None, maxstep=0.05)
    assert opt_shared._id_entry is not None
    assert opt_shared._id_entry.lock is lock_first
    opt_shared.close()
    assert id(first) in _id_locks
    assert _lock_for(first) is lock_first
    opt_first.close()
    # first is still live. The entry is gone because no optimizer holds it.
    assert id(first) not in _id_locks
    assert id(second) in _id_locks
    opt_second.close()
    assert id(second) not in _id_locks


@pytest.mark.thread_unsafe
def test_id_locked_calculator_steps_do_not_overlap() -> None:
    """Two optimizers on one non-weakrefable calculator take the same lock."""
    pytest.importorskip("ase")
    import weakref

    from ase import Atoms

    from rgmin.ase import SD, _lock_for

    class SlotCalc:
        __slots__ = ("reentrant", "inside", "max_inside", "gate", "held", "done", "paused")

        def __init__(self) -> None:
            self.reentrant = False
            self.inside = 0
            self.max_inside = 0
            self.gate = threading.Lock()
            self.held = threading.Event()
            self.done = threading.Event()
            self.paused = False

        def get_potential_energy(self, atoms: Any, force_consistent: bool = False) -> float:
            del atoms, force_consistent
            with self.gate:
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
                overlapped = self.inside >= 2
            try:
                if overlapped:
                    self.done.set()
                elif not self.paused:
                    self.paused = True
                    self.held.set()
                    assert self.done.wait(timeout=5)
            finally:
                with self.gate:
                    self.inside -= 1
            return 0.0

        def get_forces(self, atoms: Any) -> np.ndarray:
            return np.zeros((len(atoms), 3))

    calc = SlotCalc()
    with pytest.raises(TypeError):
        weakref.ref(calc)
    atoms_a = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    atoms_a.calc = calc
    atoms_b = Atoms("H", positions=[[-0.3, 0.1, 0.0]])
    atoms_b.calc = calc
    opt_a = SD(atoms_a, logfile=None, maxstep=0.05)
    opt_b = SD(atoms_b, logfile=None, maxstep=0.05)
    assert opt_a._id_entry is not None and opt_b._id_entry is not None
    shared = opt_a._id_entry.lock
    assert shared is opt_b._id_entry.lock
    assert _lock_for(calc) is shared

    start = threading.Barrier(2)
    excluded = threading.Event()

    def run_a() -> None:
        start.wait(timeout=5)
        opt_a.step()

    def run_b() -> None:
        start.wait(timeout=5)
        assert calc.held.wait(timeout=5)
        acquired = shared.acquire(blocking=False)
        if acquired:
            shared.release()
            opt_b.step()
        else:
            excluded.set()
            calc.done.set()
            opt_b.step()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_a), pool.submit(run_b)]
            for future in futures:
                future.result(timeout=30)
        assert excluded.is_set()
        assert calc.max_inside == 1
    finally:
        calc.done.set()
        opt_a.close()
        opt_b.close()
