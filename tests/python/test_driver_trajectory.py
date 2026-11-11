# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Bit-identical positions for the ASE driver and the Python solver."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import rgmin

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator

from rgmin.ase import FIRE, LBFGS


class _Harmonic(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, x0: np.ndarray) -> None:
        super().__init__()
        self.x0 = np.asarray(x0, dtype=np.float64)

    def calculate(self, atoms=None, properties=None, system_changes=None) -> None:
        x = np.asarray(atoms.positions, dtype=np.float64) - self.x0
        flat = x.ravel()
        self.results = {"energy": 0.5 * float(flat @ flat), "forces": -x}


def _cluster(n: int, seed: int = 1) -> Atoms:
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 3))
    pos[:, 0] = np.arange(n) * 1.15
    pos += rng.normal(scale=0.04, size=pos.shape)
    atoms = Atoms("Cu" * n, positions=pos)
    atoms.calc = _Harmonic(pos.copy())
    atoms.positions += rng.normal(scale=0.03, size=pos.shape)
    return atoms


def _ase_trajectory(cls: type, n: int, steps: int) -> np.ndarray:
    atoms = _cluster(n)
    opt = cls(atoms, logfile=None, maxstep=0.2)
    frames = [atoms.get_positions().copy()]
    for _ in range(steps):
        opt.step()
        frames.append(atoms.get_positions().copy())
    return np.ascontiguousarray(np.stack(frames), dtype=np.float64)


def _bind_trajectory(method: str, dim: int, steps: int) -> np.ndarray:
    g = np.empty(dim)

    def fg(x: np.ndarray, g: np.ndarray = g) -> tuple[float, np.ndarray]:
        np.copyto(g, x)
        return 0.5 * float(x @ x), g

    x = np.linspace(-0.4, 0.7, dim)
    solver = rgmin.Solver(dim, rgmin.SolverConfig(method, maxiter=1, gtol=0.0, maxstep=0.2))
    frames = [x.copy()]
    for _ in range(steps):
        solver.step(fg, x)
        frames.append(x.copy())
    return np.ascontiguousarray(np.stack(frames), dtype=np.float64)


def _goldens() -> dict[str, np.ndarray]:
    path = Path(__file__).with_name("driver_trajectory.hex")
    found: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, shape, digest, blob = line.split()
        raw = bytes.fromhex(blob)
        assert hashlib.sha256(raw).hexdigest() == digest
        dims = tuple(int(part) for part in shape.split(","))
        found[name] = np.frombuffer(raw, dtype=np.float64).reshape(dims).copy()
    return found


def test_driver_trajectories_match_reference() -> None:
    gold = _goldens()
    assert np.array_equal(_ase_trajectory(LBFGS, 7, 8), gold["ase-lbfgs-n7"])
    assert np.array_equal(_ase_trajectory(FIRE, 7, 8), gold["ase-fire-n7"])
    assert np.array_equal(_ase_trajectory(LBFGS, 48, 5), gold["ase-lbfgs-n48"])
    assert np.array_equal(_ase_trajectory(FIRE, 48, 5), gold["ase-fire-n48"])
    assert np.array_equal(_bind_trajectory("lbfgs", 9, 8), gold["bind-lbfgs"])
    assert np.array_equal(_bind_trajectory("fire", 9, 8), gold["bind-fire"])


def test_repeated_run_is_bit_identical() -> None:
    def once(cls: type) -> bytes:
        atoms = _cluster(16, seed=3)
        cls(atoms, logfile=None, maxstep=0.2).run(fmax=1e-12, steps=6)
        return np.ascontiguousarray(atoms.positions, dtype=np.float64).tobytes()

    for cls in (LBFGS, FIRE):
        assert once(cls) == once(cls)
