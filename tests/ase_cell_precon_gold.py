#!/usr/bin/env python3
"""ASE 3.29.0 PreconLBFGS with the Exp preconditioner on a cell filter.

Writes tests/data/ase_cell_precon.txt. src/cell/precon_gold.rs replays
what the cell block of Packwood et al., J. Chem. Phys. 144, 164109
(2016) changes:

  * mu and mu_c from the two gradient evaluations of estimate_mu, with
    the sine probe on the atoms and 1e-2 I on the strain block, and
    mu_c floored at 1 as ASE floors it;
  * the matrix P, whose strain block is mu_c I;
  * P x = q for a given q;
  * the L-BFGS direction -H g of the pairs ASE stored, with H0 = P^-1.

Two systems, each rattled and strained before the filter is built, so
the filter starts at the identity deformation: 2 x 2 x 2 cubic fcc Cu
(32 atoms) under EMT, which is soft enough that ASE floors both mu and
mu_c at 1, and the 8 atom cubic Si cell under the ASE Tersoff
calculator, where both estimates clear their floors. That matters: estimate_mu builds a second filter
around a copy of the atoms (and drops exp_cell_factor, the mask and the
pressure), which agrees with the first only while the deformation is
the identity, as it is at the first step the script records.

r_NN and r_cut are given, so no nearest-neighbour estimate enters. The
sine period of ASE is taken over the N + 3 rows of the extended
positions, the three strain rows included; the script writes those
lengths so rgmin can be given them. ASE's solve is a direct sparse
solve and the script writes no pyamg result.
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
from ase.optimize.precon import Exp, PreconLBFGS
from ase.optimize.precon.neighbors import get_neighbours
from ase.stress import voigt_6_to_full_3x3_stress

A = 3.0
R_CUT = 3.0
C_STAB = 0.1
SI_TERSOFF = [3.0, 1.0, 0.0, 1.0039e5, 16.217, -0.59825, 0.78734, 1.1e-6,
              1.7322, 471.18, 2.85, 0.15, 2.4799, 1830.8]
STEPS = 6
STRAIN = np.array([[0.012, 0.004, -0.003],
                   [0.004, -0.009, 0.003],
                   [-0.003, 0.003, 0.006]])


class Recording(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

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
        self.results = {"energy": energy, "free_energy": energy,
                        "forces": forces, "stress": stress}
        self.log.append((self.stage, self.atoms.positions.copy(),
                         np.array(self.atoms.cell), energy, forces.copy(),
                         voigt_6_to_full_3x3_stress(stress)))


class Spy(Exp):
    """Exp that keeps what estimate_mu and solve did."""

    def __init__(self, calc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calc = calc
        self.estimates = []
        self.solves = []
        self.stage = 0

    def estimate_mu(self, atoms, H=None):
        before = self.calc.stage
        self.calc.stage = -1
        p = np.array(atoms.get_positions())
        lengths = [float(np.ptp(p[:, i])) for i in range(3)]
        i_all, j_all, r_all, _ = get_neighbours(atoms, self.r_cut)
        keep = i_all < j_all
        pairs = sorted(zip(i_all[keep].tolist(), j_all[keep].tolist(),
                           r_all[keep].tolist()))
        out = super().estimate_mu(atoms, H)
        self.calc.stage = before
        self.estimates.append((p, lengths, out, pairs))
        return out

    def solve(self, x):
        y = super().solve(x)
        self.solves.append((self.stage, np.array(x), np.array(y)))
        return y


class Logged(PreconLBFGS):
    def __init__(self, *args, spy, calc, **kwargs):
        super().__init__(*args, **kwargs)
        self.spy = spy
        self.calc_ref = calc
        self.rows = []
        self.dense0 = None

    def step(self, f=None):
        k = len(self.rows)
        self.calc_ref.stage = k
        self.spy.stage = k
        g = -np.array(self._actual_atoms.get_forces()).reshape(-1)
        super().step(f)
        if k == 0:
            self.dense0 = self.spy.asarray().copy()
        pairs = [(s.copy(), y.copy(), float(r))
                 for s, y, r in zip(self.s, self.y, self.rho)]
        solves = [(q, z) for st, q, z in self.spy.solves if st == k]
        self.rows.append((g, pairs, solves, np.array(self.p).reshape(-1)))


def fmt(values) -> str:
    return " ".join(repr(float(v)) for v in np.ravel(values))


def build(name: str):
    """The system, its calculator and the nearest-neighbour distance."""
    rng = np.random.default_rng(20260519)
    if name == "cu":
        atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat(2)
        atoms.positions += rng.uniform(-0.05, 0.05, atoms.positions.shape)
        return atoms, EMT(), 2.55
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.positions += rng.uniform(-0.03, 0.03, atoms.positions.shape)
    params = TersoffParameters.from_list(SI_TERSOFF)
    return atoms, Tersoff({("Si", "Si", "Si"): params}), 2.35


def run(name: str, lines: list) -> None:
    atoms, inner, r_nn = build(name)
    cell = np.array(atoms.cell)
    atoms.set_cell(cell @ (np.eye(3) + STRAIN.T), scale_atoms=True)
    n = len(atoms)
    log: list = []
    calc = Recording(inner, log)
    atoms.calc = calc
    orig_cell = np.array(atoms.cell)
    ecf = FrechetCellFilter(atoms)
    precon = Spy(calc, A=A, r_cut=R_CUT, r_NN=r_nn, c_stab=C_STAB)
    opt = Logged(ecf, precon=precon, logfile=None, use_armijo=True,
                 spy=precon, calc=calc)
    for _ in range(STEPS):
        opt.step()

    if len(precon.estimates) != 1:
        sys.exit(f"estimate_mu ran {len(precon.estimates)} times")
    p0, lengths, (mu, mu_c), pairs = precon.estimates[0]
    # The evaluation at x0 is the first of the run: the optimizer asked
    # for the forces before estimate_mu did, so estimate_mu found it cached.
    estimate_calls = [log[0]] + [c for c in log if c[0] == -1]
    if len(estimate_calls) != 2:
        sys.exit(f"estimate_mu made {len(estimate_calls) - 1} new evaluations")
    if np.max(np.abs(log[0][1] - p0[:n])) > 64 * np.finfo(float).eps * 8:
        sys.exit("the first evaluation is not at x0")
    # Si is stiff enough that the estimates clear their floors; Cu is
    # soft, so both land on the floor of 1.
    if name == "si" and not (mu > 1.0 and mu_c > 1.0):
        sys.exit(f"si: mu {mu}, mu_c {mu_c} did not clear the floor")
    if name == "cu" and not (mu == 1.0 and mu_c == 1.0):
        sys.exit(f"cu: mu {mu}, mu_c {mu_c} did not sit on the floor")
    dense = opt.dense0
    if dense.shape != (3 * n + 9, 3 * n + 9):
        sys.exit(f"precon shape {dense.shape}")
    if not np.array_equal(precon.asarray(), dense):
        sys.exit("the preconditioner was rebuilt during the run")
    if len({(i, j) for i, j, _ in pairs}) != len(pairs):
        sys.exit("a pair has two images inside r_cut")

    lines.append(f"head {name} {n} {R_CUT!r} {fmt(r_nn)} {A!r} {C_STAB!r} "
                 f"{fmt(mu)} {fmt(mu_c)} {fmt(lengths)} {STEPS}")
    lines.append(f"orig_cell {fmt(orig_cell)}")
    lines.append(f"x0 {fmt(p0)}")
    for _, pos, cell, e, f, s in estimate_calls:
        lines.append(f"call {fmt(pos)} {fmt(cell)} {fmt(e)} {fmt(f)} {fmt(s)}")
    for i, j, r in pairs:
        lines.append(f"pair {i} {j} {fmt(r)}")
    nz = np.argwhere(dense != 0.0)
    for i, j in nz:
        lines.append(f"p {i} {j} {fmt(dense[i, j])}")
    q = np.random.default_rng(7).normal(size=3 * n + 9)
    lines.append(f"solve {fmt(q)} {fmt(precon.solve(q))}")
    for k, (g, prs, solves, p) in enumerate(opt.rows):
        lines.append(f"lbfgs {k} {len(prs)} {len(solves)}")
        lines.append(f"g {fmt(g)}")
        for s, y, r in prs:
            lines.append(f"pair_sy {fmt(s)} {fmt(y)} {fmt(r)}")
        for qz, z in solves:
            lines.append(f"zz {fmt(qz)} {fmt(z)}")
        lines.append(f"dir {fmt(p)}")
    print(f"{name}: mu {mu}, mu_c {mu_c}, {len(pairs)} pairs, "
          f"{len(nz)} entries, pairs per step "
          f"{[len(r[1]) for r in opt.rows]}")


def main() -> None:
    lines = [
        f"# ase {ase.__version__} PreconLBFGS + Exp + FrechetCellFilter",
        "# per case, head: name natoms r_cut r_nn A c_stab mu mu_c "
        "lengths (3) steps",
        "# then orig_cell (9), x0 = extended positions (3N+9),",
        "# two estimate_mu evaluations (positions cell energy forces stress),",
        "# pair i j r (once per bond), P entries p i j value (nonzeros),",
        "# a solve q y, and per L-BFGS step: g, pairs (s y rho), the solves",
        "# of the two-loop (q z) and the direction p.",
    ]
    run("cu", lines)
    run("si", lines)
    out = Path(__file__).with_name("data") / "ase_cell_precon.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
