#!/usr/bin/env python3
"""ASE FixBondLengths minima for the holonomic distance constraints.

Writes tests/data/ase_fix_bond_lengths.txt. tests/holonomic.rs runs
rgmin from the same starts and compares pair distances at the minima.

Water and LJ13 use Lennard-Jones with epsilon = sigma = 1 and rc = 10
without smoothing. Every pair here lies inside rc, so the forces are
the plain pair forces and ASE shifts only the energy. Ethane uses
unit springs on its 21 non-bonded pairs toward their lengths in ASE's
staggered C2H6. Lennard-Jones at sigma = 1 on Angstrom ethane closes
each methyl and leaves several constrained minima, so ASE's optimizers
can stop at different ones from one start. Each target length is the
start length.

FixBondLengths weights its position corrections and its projection of
the forces by the masses, as rgmin's SHAKE and RATTLE do in the metric
M^-1; ASE's optimizers then step along the projected forces without
the masses. Every run is made with unit masses and, under its label
with -mass appended, with the masses of ase.data on water and ethane.
LJ13 has one element, and equal masses give the unit-mass projector.
Each run records its wall time in seconds, without the bond check.

fmax is the UNCMIN gradient tolerance eps^(1/3) (Schnabel, Koontz and
Weiss 1982), as in tests/ase_fire2_gold.py, on ASE's largest per-atom
force. rgmin stops on the same norm.

Runs, all at dt 0.1 with a 0.2 step cap (the step norm for FIRE and
FIRE2, the largest atom step for LBFGS):
  fire-matched   FIRE with dtmax 0.25, rgmin's FIRE ceiling of 2.5 dt.
  fire-ase       FIRE defaults.
  fire2-matched  FIRE2 with rgmin's default FIRE 2.0 parameters.
  fire2-ase      FIRE2 defaults, the Guenole 2020 preset at dt 0.1.
  lbfgs-matched  LBFGS with alpha 1 (H0 = I) and memory 10.
  lbfgs-ase      LBFGS defaults, alpha 70 and memory 100.

k_min is the smallest eigenvalue of the Lagrangian Hessian on the
tangent space without rigid motions, at each ASE end point. A positive
value is the second-order sufficient condition for a strict minimum
up to rigid motion. A run that ends with k_min <= 0 stopped at a
constrained saddle: the data keep it, flagged by its k_min, and the
comparison skips it. A run that FixBondLengths stops, because its
RATTLE iteration on the forces does not reach the fixed absolute
tolerance 1e-13 in 500 sweeps, is kept with its counts at that point
and NaN forces and k_min, and the comparison skips it too.
fire-matched is the reference minimum. The
script fails if a run misses the budget, if the reference fails the
second-order condition, or if another run that passes it ends at a
different minimum.
"""
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import ase
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.data import atomic_masses, atomic_numbers
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.constraints import FixBondLengths
from ase.optimize import FIRE, FIRE2, LBFGS

FMAX = float(np.finfo(float).eps ** (1.0 / 3.0))
RC = 10.0
MAXSTEP = 0.2
DT = 0.1
# A budget only: the script fails if a run does not converge within it.
STEPS = 20_000

RUNS = {
    "fire-matched": (FIRE, dict(dt=DT, maxstep=MAXSTEP, dtmax=2.5 * DT)),
    "fire-ase": (FIRE, dict(dt=DT, maxstep=MAXSTEP)),
    "fire2-matched": (FIRE2, dict(dt=DT, maxstep=MAXSTEP, dtmax=2.5 * DT,
                                  dtmin=1e-12, Nmin=5, astart=0.1)),
    "fire2-ase": (FIRE2, dict(dt=DT, maxstep=MAXSTEP)),
    "lbfgs-matched": (LBFGS, dict(maxstep=MAXSTEP, alpha=1.0, memory=10)),
    "lbfgs-ase": (LBFGS, dict(maxstep=MAXSTEP)),
}


class CountedLJ(LennardJones):
    def __init__(self) -> None:
        super().__init__(sigma=1.0, epsilon=1.0, rc=RC, smooth=False)
        self.calls = 0

    def calculate(self, *args, **kwargs):
        self.calls += 1
        super().calculate(*args, **kwargs)


class Springs(Calculator):
    """E = sum (r_ij - r0_ij)^2 / 2 over the listed pairs."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, springs) -> None:
        super().__init__()
        self.springs = springs
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        x = self.atoms.positions
        e, f = 0.0, np.zeros_like(x)
        for i, j, r0 in self.springs:
            r = x[i] - x[j]
            d = np.linalg.norm(r)
            e += 0.5 * (d - r0) ** 2
            f[i] -= (d - r0) * r / d
            f[j] += (d - r0) * r / d
        self.results = {"energy": e, "forces": f}


def lj_pairs(n: int):
    """(i, j, V'(r), V''(r)) for Lennard-Jones on every pair."""
    v1 = lambda d: 4.0 * (-12.0 * d**-13 + 6.0 * d**-7)
    v2 = lambda d: 4.0 * (156.0 * d**-14 - 42.0 * d**-8)
    return [(i, j, v1, v2) for i in range(n) for j in range(i + 1, n)]


def spring_pairs(springs):
    return [(i, j, lambda d, r0=r0: d - r0, lambda d: 1.0) for i, j, r0 in springs]


def water() -> tuple[np.ndarray, list[tuple[int, int]]]:
    half = np.radians(104.52) / 2.0
    s, c = 0.9572 * np.sin(half), 0.9572 * np.cos(half)
    x = np.array([[0.0, 0.0, 0.0], [s, c, 0.0], [-s, c, 0.0]])
    return x, [(0, 1), (0, 2)]


def row_norms(v: np.ndarray) -> np.ndarray:
    """Euclidean norm over the last index."""
    return np.sqrt((v * v).sum(-1))


def jiggle(x: np.ndarray, seed: int) -> np.ndarray:
    """Move each coordinate by up to 0.1. FIRE and LBFGS keep the
    symmetry of a symmetric start and can stop at a symmetric saddle."""
    return x + np.random.default_rng(seed).uniform(-0.1, 0.1, x.shape)


ETHANE_BONDS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]


def ethane() -> tuple[np.ndarray, list[tuple[int, int]]]:
    return jiggle(molecule("C2H6").positions, 8), ETHANE_BONDS


def ethane_springs():
    x = molecule("C2H6").positions
    return [(i, j, float(np.linalg.norm(x[i] - x[j])))
            for i in range(8) for j in range(i + 1, 8) if (i, j) not in ETHANE_BONDS]


def lj13() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Centred icosahedron at centre distance 2^(1/6), jiggled, with
    the four nearest disjoint pairs fixed."""
    g = (1.0 + np.sqrt(5.0)) / 2.0
    v = []
    for a in (-1.0, 1.0):
        for b in (-g, g):
            v += [[0.0, a, b], [a, b, 0.0], [b, 0.0, a]]
    v = np.array(v) / np.sqrt(1.0 + g * g) * 2.0 ** (1.0 / 6.0)
    x = jiggle(np.vstack([np.zeros(3), v]), 13)
    d = row_norms(x[:, None] - x[None])
    order = sorted((d[i, j], i, j) for i in range(13) for j in range(i + 1, 13))
    pairs, used = [], set()
    for _, i, j in order:
        if i not in used and j not in used:
            pairs.append((i, j))
            used |= {i, j}
        if len(pairs) == 4:
            break
    return x, pairs


def hessian(x: np.ndarray, terms) -> np.ndarray:
    n = len(x)
    h = np.zeros((3 * n, 3 * n))
    for i, j, v1, v2 in terms:
        r = x[i] - x[j]
        d = np.linalg.norm(r)
        e = r / d
        b = v2(d) * np.outer(e, e) + v1(d) / d * (np.eye(3) - np.outer(e, e))
        for p, q, s in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            h[3 * p:3 * p + 3, 3 * q:3 * q + 3] += s * b
    return h


def k_min(x: np.ndarray, pairs, terms, gradient: np.ndarray) -> float:
    """Smallest Lagrangian curvature off the rigid motions, with
    c = (|x_i - x_j|^2 - d^2) / 2 and least-squares multipliers."""
    n = len(x)
    jac = np.zeros((len(pairs), 3 * n))
    lag = hessian(x, terms)
    for k, (i, j) in enumerate(pairs):
        r = x[i] - x[j]
        jac[k, 3 * i:3 * i + 3] = r
        jac[k, 3 * j:3 * j + 3] = -r
    lam = np.linalg.lstsq(jac.T, gradient, rcond=None)[0]
    for k, (i, j) in enumerate(pairs):
        for p, q, s in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            lag[3 * p:3 * p + 3, 3 * q:3 * q + 3] -= lam[k] * s * np.eye(3)
    centred = x - x.mean(0)
    rigid = [np.tile(np.eye(3)[c], n) for c in range(3)]
    rigid += [np.cross(np.eye(3)[c], centred).ravel() for c in range(3)]
    u, sv, vt = np.linalg.svd(np.vstack([jac, *rigid]))
    rank = int((sv > sv[0] * len(sv) * np.finfo(float).eps).sum())
    z = vt[rank:].T
    return float(np.linalg.eigvalsh(z.T @ lag @ z).min())


def run(x0: np.ndarray, pairs, springs, cls, kwargs, masses) -> dict:
    atoms = Atoms(f"H{len(x0)}", positions=x0)
    atoms.set_masses(masses)
    lengths = [float(np.linalg.norm(x0[i] - x0[j])) for i, j in pairs]
    atoms.set_constraint(FixBondLengths(pairs, bondlengths=lengths))
    calc = Springs(springs) if springs else CountedLJ()
    terms = spring_pairs(springs) if springs else lj_pairs(len(x0))
    atoms.calc = calc
    opt = cls(atoms, logfile=None, **kwargs)
    worst = 0.0
    checking = 0.0
    t0 = time.perf_counter()
    try:
        for _ in opt.irun(fmax=FMAX, steps=STEPS):
            t1 = time.perf_counter()
            r = atoms.positions
            worst = max(worst, max(float(abs(np.linalg.norm(r[i] - r[j]) - d) / d)
                                   for (i, j), d in zip(pairs, lengths)))
            checking += time.perf_counter() - t1
    except RuntimeError:
        return dict(steps=opt.nsteps, calls=calc.calls, x=atoms.positions.copy(),
                    g2=np.nan, gmax=np.nan, worst=worst, k_min=np.nan,
                    energy=float(atoms.get_potential_energy()),
                    seconds=time.perf_counter() - t0 - checking)
    seconds = time.perf_counter() - t0 - checking
    forces = atoms.get_forces()
    if row_norms(forces).max() >= FMAX:
        sys.exit(f"{cls.__name__} {kwargs}: not converged in {STEPS} steps")
    raw = atoms.get_forces(apply_constraint=False)
    x = atoms.positions.copy()
    return dict(steps=opt.nsteps, calls=calc.calls, x=x,
                g2=float(np.linalg.norm(forces)),
                gmax=float(row_norms(forces).max()),
                energy=float(atoms.get_potential_energy()), worst=worst,
                k_min=k_min(x, pairs, terms, -raw.ravel()), seconds=seconds)


def distances(x: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(len(x), 1)
    return row_norms(x[i] - x[j])


def main() -> None:
    out = Path(__file__).with_name("data") / "ase_fix_bond_lengths.txt"
    lines = [f"# ase {ase.__version__} FixBondLengths, unit masses or (-mass) the"
             f" masses line, Lennard-Jones epsilon 1 sigma 1 rc {RC} or unit springs,"
             f" {platform.processor() or platform.machine()}",
             f"fmax {FMAX!r}",
             "# system name, atoms, pairs, k_min of the reference run",
             "# start (3N), pairs (i j ...), springs (i j r0 ...) or lj,"
             " masses (N) if real masses differ, then per run:",
             "# run label steps force_calls |F|_2 max_atom|F|"
             " energy worst_relative_bond_error k_min seconds, end (3N)"]
    systems = (("water", water, [], ["O", "H", "H"]),
               ("ethane", ethane, ethane_springs(),
                molecule("C2H6").get_chemical_symbols()),
               ("lj13", lj13, [], None))
    for name, build, springs, symbols in systems:
        x0, pairs = build()
        unit = np.ones(len(x0))
        real = None if symbols is None else np.array(
            [atomic_masses[atomic_numbers[s]] for s in symbols])
        runs = {}
        for label, (cls, kw) in RUNS.items():
            runs[label] = run(x0, pairs, springs, cls, kw, unit)
            if real is not None:
                runs[label + "-mass"] = run(x0, pairs, springs, cls, kw, real)
        ref = runs["fire-matched"]
        kmin = ref["k_min"]
        if not kmin > 0.0:
            sys.exit(f"{name}: reference k_min {kmin} is not positive")
        for label, r in runs.items():
            if not r["k_min"] > 0.0:
                continue
            tol = 2.0 * np.sqrt(2.0) * (r["g2"] + ref["g2"]) / kmin
            off = np.abs(distances(r["x"]) - distances(ref["x"])).max()
            if off > tol:
                sys.exit(f"{name} {label}: another minimum, {off} > {tol}")
        lines.append(f"system {name} {len(x0)} {len(pairs)} {kmin!r}")
        lines.append("start " + " ".join(repr(float(v)) for v in x0.ravel()))
        lines.append("pairs " + " ".join(f"{i} {j}" for i, j in pairs))
        lines.append("springs " + " ".join(f"{i} {j} {r0!r}" for i, j, r0 in springs)
                     if springs else "lj")
        if real is not None:
            lines.append("masses " + " ".join(repr(float(m)) for m in real))
        for label, r in runs.items():
            lines.append(f"run {label} {r['steps']} {r['calls']} {r['g2']!r}"
                         f" {r['gmax']!r} {r['energy']!r} {r['worst']!r}"
                         f" {r['k_min']!r} {r['seconds']!r}")
            lines.append("end " + " ".join(repr(float(v)) for v in r["x"].ravel()))
            print(f"{name:7s} {label:14s} steps {r['steps']:5d} calls {r['calls']:5d}"
                  f" k_min {r['k_min']:.4g} worst bond {r['worst']:.1e}")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.name}: fmax {FMAX}")


if __name__ == "__main__":
    main()
