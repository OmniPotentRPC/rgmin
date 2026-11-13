# Copyright (c) rgmin developers
"""Head-to-head of ASE 3.29 and rgmin on the same geometries.

ASE LBFGS, FIRE and BFGSLineSearch against rgmin capped L-BFGS, Wolfe
L-BFGS, the geometry step and FIRE 2.0. Every arm stops at fmax 0.01.
maxstep is 0.2. ASE LBFGS and the rgmin capped and geometry steps apply
that cap to the longest atomic displacement. ASE FIRE and rgmin FIRE 2.0
apply it to the Euclidean length of the whole displacement. rgmin Wolfe
is the line search with no displacement cap, because a per-atom cap on
that session selects the geometry step. ASE BFGSLineSearch receives
maxstep 0.2 as its line-search bound. Memory, the initial Hessian scale
and the FIRE mixing parameters stay at each library's defaults. The
FIRE timestep is 0.1 on every system, ASE FIRE's default. Nothing is
refit per system.

The starts are the jittered cubic grids of examples/lbfgs_capped.rs and
tests/data/morse_pos_0000.con. The script checks that both potentials
agree at those starts, then runs seven serial repeats and writes the
median wall time.

    python3 -m venv .venv
    .venv/bin/pip install ase
    env -i PATH=".venv/bin:$PATH" HOME="$HOME" LANG="${LANG:-C.UTF-8}" \
        python benches/ase_h2h.py

Raw logs go to benches/ase_h2h/raw/, which is not committed. The table
is benches/ase_h2h/results.md.
"""

from __future__ import annotations

import io
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGSLineSearch, FIRE, LBFGS
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(__file__).resolve().parent / "ase_h2h" / "raw"
TABLE = Path(__file__).resolve().parent / "ase_h2h" / "results.md"
SPACING = 1.1
JITTER = 0.03
CON = ROOT / "tests" / "data" / "morse_pos_0000.con"


class Counted(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        pos = self.atoms.get_positions().ravel()
        energy, grad = self.fn(pos)
        self.results = {
            "energy": float(energy),
            "forces": -np.asarray(grad, dtype=float).reshape(-1, 3),
        }


def grid_start(atoms: int) -> np.ndarray:
    side = int(math.ceil(atoms ** (1.0 / 3.0)))
    x = []
    for k in range(atoms):
        cell = [k % side, (k // side) % side, k // (side * side)]
        for a, c in enumerate(cell):
            x.append(SPACING * c + JITTER * math.sin(3 * k + a))
    return np.asarray(x, dtype=float)


def read_con(path: Path):
    lines = path.read_text().splitlines()
    cell = [float(t) for t in lines[2].split()[:3]]
    start = next(i for i, line in enumerate(lines) if line.startswith("Coordinates"))
    x = []
    for line in lines[start + 1 :]:
        nums = []
        for tok in line.split():
            try:
                nums.append(float(tok))
            except ValueError:
                pass
        if len(nums) < 3:
            break
        x.extend(nums[:3])
    return np.asarray(x, dtype=float), np.asarray(cell, dtype=float)


def lennard_jones(x: np.ndarray):
    n = x.size // 3
    p = x.reshape(n, 3)
    d = p[:, None, :] - p[None, :, :]
    r2 = np.einsum("ija,ija->ij", d, d)
    iu = np.triu_indices(n, 1)
    r2p = r2[iu]
    s6 = r2p ** -3
    energy = 4.0 * np.sum(s6 * s6 - s6)
    du_over_r = 24.0 * (s6 - 2.0 * s6 * s6) / r2p
    coeff = du_over_r[:, None] * d[iu]
    g = np.zeros_like(p)
    np.add.at(g, iu[0], coeff)
    np.add.at(g, iu[1], -coeff)
    return float(energy), g.reshape(-1)


def morse_fn(cell: np.ndarray):
    d0, a, r0, cut = 0.7102, 1.6047, 2.897, 7.0
    xc = math.exp(-a * (cut - r0))
    shift = d0 * (xc * xc - 2.0 * xc)

    def morse(x: np.ndarray):
        n = x.size // 3
        p = x.reshape(n, 3)
        d = p[:, None, :] - p[None, :, :]
        d -= cell * np.round(d / cell)
        r2 = np.einsum("ija,ija->ij", d, d)
        iu = np.triu_indices(n, 1)
        r = np.sqrt(r2[iu])
        keep = r < cut
        r = r[keep]
        pair = d[iu][keep]
        ex = np.exp(-a * (r - r0))
        energy = float(np.sum(d0 * (ex * ex - 2.0 * ex) - shift))
        du = 2.0 * d0 * a * ex * (1.0 - ex)
        coeff = (du / r)[:, None] * pair
        g = np.zeros_like(p)
        np.add.at(g, iu[0][keep], coeff)
        np.add.at(g, iu[1][keep], -coeff)
        return energy, g.reshape(-1)

    return morse


def fmax_of(grad: np.ndarray) -> float:
    g = grad.reshape(-1, 3)
    return float(np.linalg.norm(g, axis=1).max())


def chained_rosen(x: np.ndarray):
    f = 0.0
    g = np.zeros_like(x)
    for i in range(x.size - 1):
        a = 1.0 - x[i]
        b = x[i + 1] - x[i] * x[i]
        f += a * a + 100.0 * b * b
        g[i] += -2.0 * a - 400.0 * x[i] * b
        g[i + 1] += 200.0 * b
    return f, g


def rosen_local_min() -> str:
    """Local minimum of the chained Rosenbrock function for n >= 4."""
    lines = []
    for n in (3, 4, 10):
        x0 = np.array([-1.0] + [1.0] * (n - 1))
        res = minimize(
            lambda x: chained_rosen(x)[0],
            x0,
            jac=lambda x: chained_rosen(x)[1],
            method="BFGS",
            tol=1e-14,
        )
        hess = np.zeros((n, n))
        eps = 1e-6
        for i in range(n):
            step = np.zeros(n)
            step[i] = eps
            hess[:, i] = (
                chained_rosen(res.x + step)[1] - chained_rosen(res.x - step)[1]
            ) / (2.0 * eps)
        eig = np.linalg.eigvalsh(0.5 * (hess + hess.T))
        lines.append(
            f"n={n} f={res.fun:.6f} x0={res.x[0]:.6f} "
            f"xmin={eig[0]:.3e} ginf={np.max(np.abs(res.jac)):.3e}"
        )
    return "\n".join(lines)


def run_rgmin() -> tuple[dict, list[dict]]:
    proc = subprocess.run(
        ["cargo", "run", "--locked", "--release", "--example", "ase_h2h"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "rgmin.txt").write_text(proc.stdout)
    (RAW / "rgmin.err").write_text(proc.stderr)
    meta = {}
    inits = {}
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "meta":
            meta[parts[1]] = parts[2]
        elif parts[0] == "init":
            inits[parts[1]] = (float(parts[2]), float(parts[3]))
        elif parts[0] == "result":
            rows.append(
                {
                    "problem": parts[1],
                    "arm": parts[2],
                    "evals": int(parts[3]),
                    "energy": float(parts[4]),
                    "fmax": float(parts[5]),
                    "ms": float(parts[6]),
                    "converged": parts[7] == "1",
                }
            )
    return {**meta, "init": inits}, rows


def check_starts(meta_init: dict, problems: list[tuple[str, np.ndarray, object]]) -> None:
    for name, x0, fn in problems:
        energy, grad = fn(x0)
        got_e, got_f = energy, fmax_of(grad)
        exp_e, exp_f = meta_init[name]
        if abs(got_e - exp_e) > 1e-6 or abs(got_f - exp_f) > 1e-6:
            raise SystemExit(
                f"{name} potentials differ: python {got_e:.16e} {got_f:.16e} "
                f"rgmin {exp_e:.16e} {exp_f:.16e}"
            )


def run_ase(name, x0, fn, cell, fmax, maxstep, budget) -> list[dict]:
    makers = {
        "ase-lbfgs": lambda atoms: LBFGS(atoms, logfile=io.StringIO(), maxstep=maxstep),
        "ase-fire": lambda atoms: FIRE(atoms, logfile=io.StringIO(), maxstep=maxstep),
        "ase-bfgslinesearch": lambda atoms: BFGSLineSearch(
            atoms, logfile=io.StringIO(), maxstep=maxstep
        ),
    }
    rows = []
    positions = x0.reshape(-1, 3)
    for arm, make in makers.items():
        times = []
        first = None
        for _rep in range(7):
            atoms = Atoms(positions=positions.copy())
            if cell is not None:
                atoms.set_cell(cell)
                atoms.set_pbc(True)
            calc = Counted(fn)
            atoms.calc = calc
            dyn = make(atoms)
            calc.calls = 0
            t0 = time.perf_counter()
            try:
                ok = bool(dyn.run(fmax=fmax, steps=budget))
            except RuntimeError:
                ok = False
            times.append((time.perf_counter() - t0) * 1e3)
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmax_now = float(np.linalg.norm(forces, axis=1).max())
            got = (calc.calls, energy, fmax_now, ok)
            if first is None:
                first = got
            elif first[0] != got[0] or abs(first[1] - got[1]) > 1e-6:
                raise SystemExit(f"{name} {arm} is not repeatable: {first} vs {got}")
        times.sort()
        evals, energy, fmax_now, ok = first
        rows.append(
            {
                "problem": name,
                "arm": arm,
                "evals": evals,
                "energy": energy,
                "fmax": fmax_now,
                "ms": times[len(times) // 2],
                "converged": ok,
            }
        )
        print(
            f"{name:12} {arm:22} {evals:6d} {energy:16.6f} {fmax_now:9.3e} "
            f"{times[len(times) // 2]:10.3f} {ok}",
            flush=True,
        )
    return rows


def write_table(meta: dict, rows: list[dict], rosen: str) -> None:
    order = [
        "ase-lbfgs",
        "ase-fire",
        "ase-bfgslinesearch",
        "rgmin-capped",
        "rgmin-wolfe",
        "rgmin-geometry",
        "rgmin-fire2",
    ]
    problems = []
    for row in rows:
        if row["problem"] not in problems:
            problems.append(row["problem"])
    lines = [
        "# ASE against rgmin",
        "",
        "ASE 3.29.0 and rgmin, seven serial repeats, median wall time.",
        f"fmax {meta['fmax']}, maxstep {meta['maxstep']}, step budget {meta['budget']}.",
        "rgmin L-BFGS memory is 8. ASE LBFGS memory is its default, 100,",
        "and its initial inverse-Hessian scale is its default, 1/70.",
        "ASE FIRE and rgmin FIRE 2.0 use timestep 0.1 and a Euclidean",
        "step cap of 0.2. ASE LBFGS, rgmin capped L-BFGS and the geometry",
        "step cap the longest atomic displacement at 0.2. rgmin Wolfe is",
        "the line search without that cap. No parameter is refit per system.",
        "Evaluations are oracle calls. They do not change across repeats.",
        "Wall time is the whole run in that language, the optimizer and",
        "the potential together. ASE is Python and rgmin is Rust, so the",
        "milliseconds are not a comparison of the optimizer loops alone.",
        "Equal energies are the same basin. A lower energy is a deeper one,",
        "and the force counts of those rows are counts to different endpoints.",
        "",
        "| problem | arm | evals | energy | fmax | median ms | converged |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    by = {(r["problem"], r["arm"]): r for r in rows}
    for problem in problems:
        for arm in order:
            row = by[(problem, arm)]
            flag = "yes" if row["converged"] else "no"
            lines.append(
                f"| {problem} | {arm} | {row['evals']} | {row['energy']:.6f} | "
                f"{row['fmax']:.3e} | {row['ms']:.3f} | {flag} |"
            )
    lines.extend(
        [
            "",
            "The chained Rosenbrock function",
            "`sum_i (1 - x_i)^2 + 100 (x_{i+1} - x_i^2)^2`",
            "has a local minimum for every n >= 4 in the basin of",
            "`(-1, 1, ..., 1)`. SciPy BFGS started there converges",
            "with a positive definite Hessian. For n = 4 the value is",
            "3.701 and the first coordinate is -0.776. For n = 10 the",
            "value is 3.987 and the first coordinate is -0.993, near",
            "`(-1, 1, ..., 1)` with value about 3.98. For n = 3 that",
            "start falls through to the valley floor at 0. A minimizer",
            "that stops at the local minimum has converged.",
            "SciPy BFGS from `(-1, 1, ..., 1)`:",
            "",
            "```",
            rosen,
            "```",
            "",
        ]
    )
    TABLE.write_text("\n".join(lines))


def main() -> None:
    meta, rgmin_rows = run_rgmin()
    for row in rgmin_rows:
        print(
            f"{row['problem']:12} {row['arm']:22} {row['evals']:6d} {row['energy']:16.6f} "
            f"{row['fmax']:9.3e} {row['ms']:10.3f} {row['converged']}",
            flush=True,
        )
    problems = []
    for atoms in (7, 13, 38):
        problems.append((f"LJ{atoms}", grid_start(atoms), lennard_jones, None))
    morse_x, cell = read_con(CON)
    problems.append(("MorsePt256", morse_x, morse_fn(cell), cell))
    check_starts(meta["init"], [(n, x, fn) for n, x, fn, _c in problems])
    fmax = float(meta["fmax"])
    maxstep = float(meta["maxstep"])
    budget = int(meta["budget"])
    ase_rows = []
    for name, x0, fn, cell in problems:
        ase_rows.extend(run_ase(name, x0, fn, cell, fmax, maxstep, budget))
    rosen = rosen_local_min()
    print(rosen, flush=True)
    write_table(meta, rgmin_rows + ase_rows, rosen)


if __name__ == "__main__":
    main()
