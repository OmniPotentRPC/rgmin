# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""F3 claim and independent solvers on several threads."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import rgmin


def _quad(x: np.ndarray) -> tuple[float, np.ndarray]:
    return 0.5 * float(x @ x), np.array(x, dtype=np.float64, copy=True)


def test_independent_solvers_match() -> None:
    def once() -> bytes:
        x = np.array([0.4, -0.2, 0.7, 0.1], dtype=np.float64)
        rgmin.minimize(_quad, x, method="lbfgs", memory=8, gtol=1e-14, maxiter=20)
        return x.tobytes()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _i: once(), range(8)))
    assert len(set(results)) == 1


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


def test_shared_calculator_with_and_without_reentrant() -> None:
    ase = pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    class Shared(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, reentrant: bool) -> None:
            super().__init__()
            self.reentrant = reentrant
            self.inside = 0
            self.max_inside = 0
            self._gate = threading.Lock()

        def calculate(self, atoms=None, properties=None, system_changes=None):  # type: ignore[no-untyped-def]
            with self._gate:
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
            try:
                time.sleep(0.05)
                x = np.asarray(atoms.get_positions(), dtype=np.float64).reshape(-1)
                self.results = {"energy": 0.5 * float(x @ x), "forces": -x.reshape(-1, 3)}
            finally:
                with self._gate:
                    self.inside -= 1

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
