# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Default per-atom cap, and an explicit maxstep that replaces it.

The start is the seed-1 packing from bench/atomistic lj38, scaled by
one half. The Rust test of the same geometry is tests/default_maxstep.rs.
"""

from __future__ import annotations

import numpy as np

import rgmin

N = 38
CAP = 0.2
BOUND_PAIR = 6.0


def _compressed_lj38() -> np.ndarray:
    radius = 2.4
    state = (1 * 7919 + 17) & 0xFFFFFFFFFFFFFFFF

    def next_u64() -> int:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

    def uniform() -> float:
        return (next_u64() >> 11) / float(1 << 53)

    pos: list[float] = []
    while len(pos) < 3 * N:
        point = [(2.0 * uniform() - 1.0) * radius for _ in range(3)]
        if sum(value * value for value in point) > radius * radius:
            continue
        close = False
        for base in range(0, len(pos), 3):
            dist = sum((pos[base + k] - point[k]) ** 2 for k in range(3))
            if dist < 0.81:
                close = True
                break
        if not close:
            pos.extend(point)
    return np.asarray(pos, dtype=np.float64) * 0.5


def _lj(x: np.ndarray) -> tuple[float, np.ndarray]:
    energy = 0.0
    grad = np.zeros(x.shape, dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            dx = float(x[3 * i] - x[3 * j])
            dy = float(x[3 * i + 1] - x[3 * j + 1])
            dz = float(x[3 * i + 2] - x[3 * j + 2])
            r2 = dx * dx + dy * dy + dz * dz
            r = r2**0.5
            inv = 1.0 / r
            sr6 = inv**6
            sr12 = sr6 * sr6
            energy += 4.0 * (sr12 - sr6)
            scale = (24.0 * inv) * (sr6 - 2.0 * sr12) * inv
            grad[3 * i] += scale * dx
            grad[3 * i + 1] += scale * dy
            grad[3 * i + 2] += scale * dz
            grad[3 * j] -= scale * dx
            grad[3 * j + 1] -= scale * dy
            grad[3 * j + 2] -= scale * dz
    return energy, grad


def _max_pair(x: np.ndarray) -> float:
    best = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = float(x[3 * i] - x[3 * j])
            dy = float(x[3 * i + 1] - x[3 * j + 1])
            dz = float(x[3 * i + 2] - x[3 * j + 2])
            best = max(best, (dx * dx + dy * dy + dz * dz) ** 0.5)
    return best


def _max_atom_step(before: np.ndarray, after: np.ndarray) -> float:
    best = 0.0
    for i in range(N):
        dx = float(after[3 * i] - before[3 * i])
        dy = float(after[3 * i + 1] - before[3 * i + 1])
        dz = float(after[3 * i + 2] - before[3 * i + 2])
        best = max(best, (dx * dx + dy * dy + dz * dz) ** 0.5)
    return best


def _first_move(method: str, **options: float) -> float:
    x = _compressed_lj38()
    before = x.copy()
    rgmin.Solver(x.size, rgmin.SolverConfig(method, maxiter=2, gtol=1e-12, **options)).step(_lj, x)
    return _max_atom_step(before, x)


def test_default_cap_keeps_lj38_bound_and_maxstep_overrides() -> None:
    start = _compressed_lj38()
    assert _lj(start)[0] > 1.0e5
    assert _max_pair(start) < BOUND_PAIR
    for method in ("lbfgs", "bfgs"):
        x = start.copy()
        solver = rgmin.Solver(x.size, rgmin.SolverConfig(method, maxiter=800, gtol=1e-2))
        report = None
        peak = 0.0
        for _ in range(800):
            before = x.copy()
            report = solver.step(_lj, x)
            peak = max(peak, _max_atom_step(before, x))
            if report.grad_norm < 1e-2:
                break
        assert report is not None
        assert report.stop == rgmin.Stop.Tolerance, method
        assert report.value < -160.0, (method, report.value)
        assert _max_pair(x) <= BOUND_PAIR, method
        assert peak <= CAP + 1e-9, (method, peak)
        cleared = _first_move(method, maxstep=0.0)
        assert cleared > 1.0, (method, cleared)
        wider = _first_move(method, maxstep=0.5)
        assert CAP < wider <= 0.5 + 1e-9, (method, wider)
        named = _first_move(method, atom_maxmove=0.5)
        assert CAP < named <= 0.5 + 1e-9, (method, named)
