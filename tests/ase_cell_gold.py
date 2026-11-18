#!/usr/bin/env python3
"""ASE 3.29.0 FIRE2 on FrechetCellFilter, step by step.

Writes tests/data/ase_cell_gold.txt. src/cell/gold.rs replays it: the
filter gradient of rgmin against ASE's at every calculator call, the
extended coordinates against ASE's, one FIRE2 step of rgmin from the
ASE state against the next ASE state, and a free-running rgmin session
against the whole trajectory.

Systems (the two potentials are ASE's own, so every force and stress
the replay compares was produced by ASE):
  si  bulk Si, 8 atom cubic cell, ase.calculators.tersoff with the
      Tersoff 1988 Si(C) parameters of the LAMMPS Si.tersoff file.
  cu  bulk Cu, 4 atom cubic cell, ase.calculators.emt.EMT.
Each is rattled and given a mixed strain, so the stress is a full
tensor. Filter options covered: the defaults (exp_cell_factor = N), a
scalar pressure with the hydrostatic option, and a mask with the
constant-volume option.

The optimizer is ASE FIRE2 as in tests/ase_fire2_gold.py: dt = 0.1,
dtmax = 10 dt, dtmin = 0.02 dt, no ABC and no step cap. The script
fails unless the first non-positive power comes after the 20-step
delay of the Guenole preset and no shrink reaches dtmin, the two cases
where rgmin and ASE FIRE2 could differ.

ASE stores the cell block as logm(N F) * exp_cell_factor, where N is
the atom count, so its x differs from rgmin's exp_cell_factor * logm(F)
by exp_cell_factor * ln(N) on the diagonal. The script checks that
offset and the replay subtracts it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import ase
import numpy as np
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.calculators.tersoff import Tersoff, TersoffParameters
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE2
from ase.stress import voigt_6_to_full_3x3_stress
from scipy.linalg import logm

DT = 0.02
N_DELAY = 20
STEPS = 45
SI_TERSOFF = [3.0, 1.0, 0.0, 1.0039e5, 16.217, -0.59825, 0.78734, 1.1e-6,
              1.7322, 471.18, 2.85, 0.15, 2.4799, 1830.8]

STRAIN = np.array([[0.020, 0.008, -0.005],
                   [0.008, -0.015, 0.006],
                   [-0.005, 0.006, 0.010]])


class Recording(Calculator):
    """Wrap a calculator and keep every evaluation, in call order."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, inner: Calculator, log: list) -> None:
        super().__init__()
        self.inner = inner
        self.log = log
        self.stage = 0

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        probe = self.atoms.copy()
        probe.calc = self.inner
        energy = probe.get_potential_energy()
        forces = probe.get_forces()
        stress = probe.get_stress()
        self.results = {"energy": energy, "forces": forces, "stress": stress}
        self.log.append((self.stage, self.atoms.positions.copy(),
                         np.array(self.atoms.cell), energy, forces.copy(),
                         voigt_6_to_full_3x3_stress(stress)))


def structure(name: str):
    if name == "si":
        atoms = bulk("Si", "diamond", a=5.43, cubic=True)
        calc = Tersoff({("Si", "Si", "Si"):
                        TersoffParameters.from_list(SI_TERSOFF)})
    else:
        atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
        calc = EMT()
    rng = np.random.default_rng(20260518)
    atoms.positions += rng.uniform(-0.04, 0.04, atoms.positions.shape)
    cell = np.array(atoms.cell)
    atoms.set_cell(cell @ (np.eye(3) + STRAIN.T), scale_atoms=True)
    return atoms, calc


class Logged(FIRE2):
    def __init__(self, *args, rows, calc, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = rows
        self.calc_ref = calc
        self.first_reset = None
        self.floor_hit = False

    def state(self):
        v = np.zeros(len(self.optimizable.get_x())) if self.v is None else self.v
        return v.copy(), self.dt, self.a, self.Nsteps

    def step(self, f=None):
        k = len(self.rows)
        self.calc_ref.stage = k
        x_in = np.array(self.optimizable.get_x())
        g_in = np.array(self.optimizable.get_gradient())
        enthalpy = self.optimizable.get_value()
        before = self.state()
        if self.v is not None:
            power = np.vdot(-g_in, self.v)
            if power <= 0.0:
                if self.first_reset is None:
                    self.first_reset = k + 1
                if self.dt * self.fdec < self.dtmin:
                    self.floor_hit = True
        super().step(f)
        x_out = np.array(self.optimizable.get_x())
        self.rows.append((x_in, g_in, enthalpy, before, x_out, self.state()))


def fmt(values) -> str:
    return " ".join(repr(float(v)) for v in np.ravel(values))


def run(name: str, options: dict, lines: list) -> None:
    atoms, inner = structure(name)
    n = len(atoms)
    log: list = []
    calc = Recording(inner, log)
    atoms.calc = calc
    orig_cell = np.array(atoms.cell)
    ecf = FrechetCellFilter(atoms, **options)
    factor = ecf.exp_cell_factor
    rows: list = []
    opt = Logged(ecf, logfile=None, dt=DT, dtmax=10 * DT, dtmin=0.02 * DT,
                 maxstep=float("inf"), use_abc=False, rows=rows, calc=calc)
    for _ in range(STEPS):
        opt.step()
    if opt.first_reset is not None and opt.first_reset <= N_DELAY:
        sys.exit(f"{name}: P <= 0 at step {opt.first_reset}")
    if opt.floor_hit:
        sys.exit(f"{name}: a shrink reached dtmin")
    rollbacks = sum(1 for k in range(len(rows))
                    if sum(1 for c in log if c[0] == k) > 1)

    mask = np.array(ecf._utility.mask3x3, dtype=float)
    x0 = rows[0][0]
    offset = x0[3 * n:].reshape(3, 3) - factor * logm(np.eye(3))
    want = factor * np.log(n) * np.eye(3)
    if np.max(np.abs(offset - want)) > 64 * np.finfo(float).eps * factor * np.log(n):
        sys.exit(f"{name}: cell block offset {offset} is not ecf ln N")

    lines.append(
        f"case {name} {n} {len(rows)} {fmt(factor)} "
        f"{fmt(options.get('scalar_pressure', 0.0))} "
        f"{int(options.get('hydrostatic_strain', False))} "
        f"{int(options.get('constant_volume', False))} {fmt(mask)}")
    lines.append(f"orig_cell {fmt(orig_cell)}")
    lines.append(f"rollbacks {rollbacks}")
    for k, (x_in, g_in, h, before, x_out, after) in enumerate(rows):
        calls = [c for c in log if c[0] == k]
        lines.append(f"step {k} {len(calls)}")
        lines.append(f"x {fmt(x_in)}")
        lines.append(f"g {fmt(g_in)}")
        lines.append(f"h {fmt(h)}")
        for _, pos, cell, e, f, st in calls:
            lines.append(f"call {fmt(pos)} {fmt(cell)} {fmt(e)} {fmt(f)} {fmt(st)}")
        lines.append(f"out {fmt(x_out)} {fmt(after[0])} {fmt(after[1])} {fmt(after[2])} {after[3]}")
    print(f"{name} {options}: {len(rows)} steps, {len(log)} calls, "
          f"{rollbacks} rollbacks, first P<=0 at step {opt.first_reset}")


def main() -> None:
    out = Path(__file__).with_name("data") / "ase_cell_gold.txt"
    lines = [
        f"# ase {ase.__version__} FIRE2 + FrechetCellFilter",
        f"dt {DT} dtmax {10 * DT} dtmin {0.02 * DT} nmin {N_DELAY}",
        "# case name natoms steps exp_cell_factor pressure hydrostatic "
        "constant_volume mask (9)",
        "# orig_cell (9), rollbacks",
        "# per step: x (3N+9), gradient (3N+9), enthalpy, call = positions "
        "(3N) cell (9) energy forces (3N) stress (9) in call order, "
        "out = x v dt alpha n_pos after the step. The state before step 0 "
        "is v = 0, dt, alpha 0.25, n_pos 0.",
    ]
    run("si", {}, lines)
    run("cu", {}, lines)
    run("si", {"scalar_pressure": 0.004, "hydrostatic_strain": True}, lines)
    run("cu", {"mask": [1, 1, 1, 0, 0, 0], "constant_volume": True}, lines)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
