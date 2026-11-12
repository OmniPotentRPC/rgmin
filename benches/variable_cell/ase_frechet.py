#!/usr/bin/env python3
"""ASE FrechetCellFilter + FIRE / BFGS on the bench_cell starts.

Nearest-neighbour Lennard-Jones, epsilon = 1, sigma = 1, cutoff 1.4,
the same minimum image and the same three geometries as
`examples/bench_cell.rs`. ASE  FrechetCellFilter is the cell chart.
FIRE uses dt 0.1. BFGS uses maxstep 0.2. Both stop at fmax 1e-3 or
300 steps. Five serial repeats. The printed seconds are the median.
"""

import time

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE
from ase.stress import full_3x3_to_voigt_6_stress

EPSILON = 1.0
SIGMA = 1.0
RC = 1.4
REPEATS = 5


class Lj(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.calls += 1
        energy, forces, virial = lj(self.atoms.get_positions(), np.array(self.atoms.cell))
        volume = self.atoms.get_volume()
        stress = virial / volume
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
            "stress": full_3x3_to_voigt_6_stress(stress),
        }


def r_min():
    return 2.0 ** (1.0 / 6.0)


def a_fcc():
    return r_min() * np.sqrt(2.0)


def fcc(n_rep, a):
    basis = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    )
    positions = []
    for ix in range(n_rep):
        for iy in range(n_rep):
            for iz in range(n_rep):
                for site in basis:
                    positions.append((np.array([ix, iy, iz]) + site) * a)
    length = n_rep * a
    cell = np.diag([length, length, length])
    return np.array(positions), cell


def bulk(scale, amp):
    positions, cell = fcc(1, a_fcc() * scale)
    for index in range(positions.size):
        wave = (index * 17) % 10 - 4.5
        positions.ravel()[index] += amp * wave / 4.5
    return positions, cell


def shear(eps):
    positions, cell = fcc(1, a_fcc())
    cell = cell.copy()
    cell[0, 1] = eps * cell[0, 0]
    return positions, cell


def vacancy():
    positions, cell = fcc(2, a_fcc() * 0.97)
    return positions[1:], cell


def image_shift(cell, nx, ny, nz):
    return nx * cell[0] + ny * cell[1] + nz * cell[2]


def image_reach(cell):
    inverse = np.linalg.inv(cell)
    reach = max(np.linalg.norm(inverse[row]) * RC for row in range(3))
    return int(np.ceil(reach))


def lj(positions, cell):
    n = len(positions)
    nmax = image_reach(cell)
    energy = 0.0
    forces = np.zeros_like(positions)
    virial = np.zeros((3, 3))
    for i in range(n):
        for j in range(n):
            for nx in range(-nmax, nmax + 1):
                for ny in range(-nmax, nmax + 1):
                    for nz in range(-nmax, nmax + 1):
                        if i == j and nx == ny == nz == 0:
                            continue
                        d = positions[j] + image_shift(cell, nx, ny, nz) - positions[i]
                        r2 = float(np.dot(d, d))
                        if r2 > RC * RC or r2 < 1e-24:
                            continue
                        r = np.sqrt(r2)
                        inv_r = 1.0 / r
                        sr6 = (SIGMA * inv_r) ** 6
                        sr12 = sr6 * sr6
                        energy += 0.5 * 4.0 * EPSILON * (sr12 - sr6)
                        scale = 24.0 * EPSILON * (-2.0 * sr12 + sr6) * inv_r * inv_r
                        forces[i] += scale * d
                        virial += 0.5 * scale * np.outer(d, d)
    return energy, forces, virial


def residuals(atoms):
    forces = atoms.get_forces()
    max_force = float(np.linalg.norm(forces, axis=1).max())
    stress = atoms.get_stress(voigt=False)
    max_stress = float(np.max(np.abs(stress)))
    return max_force, max_stress, float(atoms.get_potential_energy())


def run(name, positions, cell, arm):
    seconds = []
    calls = []
    last = None
    for _ in range(REPEATS):
        atoms = Atoms(
            numbers=[1] * len(positions),
            positions=positions,
            cell=cell,
            pbc=True,
        )
        calc = Lj()
        atoms.calc = calc
        filt = FrechetCellFilter(atoms)
        opt = FIRE(filt, dt=0.1, logfile=None) if arm == "fire" else BFGS(filt, maxstep=0.2, logfile=None)
        start = time.perf_counter()
        opt.run(fmax=1e-3, steps=300)
        seconds.append(time.perf_counter() - start)
        calls.append(calc.calls)
        last = residuals(atoms)
    seconds.sort()
    max_force, max_stress, energy = last
    converged = max_force < 1e-3 and max_stress < 1e-4
    same = all(count == calls[0] for count in calls)
    print(
        f"{name} ase-{arm} {calls[0]} {seconds[REPEATS // 2]:.6e} "
        f"{max_force:.6e} {max_stress:.6e} {energy:.8f} {int(converged)} calls_stable {same}"
    )


def main():
    systems = [
        ("bulk4", bulk(1.04, 0.02)),
        ("shear4", shear(0.04)),
        ("vacancy31", vacancy()),
    ]
    print("system arm calls seconds max_force max_stress energy converged")
    for name, (positions, cell) in systems:
        energy, _, _ = lj(positions, cell)
        print(f"# {name} start_energy {energy:.12f}")
        for arm in ("fire", "bfgs"):
            run(name, positions, cell, arm)


if __name__ == "__main__":
    main()
