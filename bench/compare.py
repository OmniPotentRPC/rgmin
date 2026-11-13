# Copyright (c) rgmin developers
"""ASE and SciPy arms for the same starts as examples/bench_suite.rs.

ASE  LBFGS, FIRE and FIRE2 use fmax 0.01 on the clusters and maxstep 0.2.
Rosenbrock stops at an infinity-norm of 1e-5. SciPy CG is Polak-Ribiere-Polyak
with that tolerance. SciPy L-BFGS-B has no per-atom cap. Each arm is repeated
five times and the wall time is the median. The script writes bench/raw/ase.csv
and bench/raw/scipy.csv, then bench/results.md from every CSV in bench/raw.

    python3 -m venv .venv
    .venv/bin/pip install ase scipy numpy
    env -i PATH=".venv/bin:$PATH" HOME="$HOME" LANG="${LANG:-C.UTF-8}" \
        python bench/compare.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE, FIRE2, LBFGS
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(__file__).resolve().parent / "raw"
TABLE = Path(__file__).resolve().parent / "results.md"
REPEATS = 5
BUDGET = 20_000
CAP = 0.2


class Counted(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        energy, grad = self.fn(self.atoms.get_positions().ravel())
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
            x.append(1.1 * c + 0.03 * math.sin(3 * k + a))
    return np.asarray(x, dtype=float)


def read_con(path: Path):
    lines = path.read_text().splitlines()
    cell = np.asarray([float(t) for t in lines[2].split()[:3]])
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
    return np.asarray(x, dtype=float), cell


def rosenbrock(x: np.ndarray):
    f = 0.0
    g = np.zeros_like(x)
    for i in range(x.size - 1):
        a = 1.0 - x[i]
        b = x[i + 1] - x[i] * x[i]
        f += a * a + 100.0 * b * b
        g[i] += -2.0 * a - 400.0 * x[i] * b
        g[i + 1] += 200.0 * b
    return float(f), g


def lennard_jones(x: np.ndarray):
    n = x.size // 3
    p = x.reshape(n, 3)
    d = p[:, None, :] - p[None, :, :]
    r2 = np.einsum("ija,ija->ij", d, d)
    iu = np.triu_indices(n, 1)
    r2p = r2[iu]
    s6 = r2p**-3
    energy = 4.0 * np.sum(s6 * s6 - s6)
    du = 24.0 * (s6 - 2.0 * s6 * s6) / r2p
    coeff = du[:, None] * d[iu]
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


def grad_of(g: np.ndarray, cluster: bool) -> float:
    if not cluster:
        return float(np.max(np.abs(g)))
    p = g.reshape(-1, 3) if g.size % 3 == 0 else g.reshape(1, -1)
    return float(np.sqrt((p * p).sum(axis=1)).max())


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


def ase_once(kind: str, x0: np.ndarray, fn, cluster: bool, tol: float):
    atoms = Atoms(numbers=np.ones(x0.size // 3, dtype=int), positions=x0.reshape(-1, 3))
    calc = Counted(fn)
    atoms.calc = calc
    if kind == "lbfgs":
        opt = LBFGS(atoms, logfile=None, maxstep=CAP)
    elif kind == "fire":
        opt = FIRE(atoms, logfile=None, maxstep=CAP)
    else:
        opt = FIRE2(atoms, logfile=None, dt=0.1, maxstep=CAP)
    opt.run(fmax=tol, steps=BUDGET)
    energy, grad = fn(atoms.get_positions().ravel())
    return calc.calls, energy, grad_of(grad, cluster), grad_of(grad, True) < tol


def scipy_once(kind: str, x0: np.ndarray, fn, cluster: bool, tol: float):
    calls = {"n": 0}

    def fun(x):
        calls["n"] += 1
        f, g = fn(x)
        return f, g

    method = "L-BFGS-B" if kind == "lbfgsb" else "CG"
    options = {"maxiter": BUDGET, "gtol": tol}
    if kind == "lbfgsb":
        options["ftol"] = 0.0
    res = minimize(fun, x0, method=method, jac=True, options=options)
    energy, grad = fn(res.x)
    return calls["n"], energy, grad_of(grad, cluster), grad_of(grad, cluster) < tol


def time_arm(once, *args):
    times = []
    last = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        last = once(*args)
        times.append((time.perf_counter() - t0) * 1e3)
    evals, energy, grad, ok = last
    return evals, energy, grad, median(times), ok


def write_rows(path: Path, rows: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("problem,arm,library,evals,energy,grad,median_ms,converged\n" + "\n".join(rows) + "\n")


def run_ase():
    rows = []
    cell = None
    morse_x = None
    con = ROOT / "tests" / "data" / "morse_pos_0000.con"
    if con.exists():
        morse_x, cell = read_con(con)
    problems = [
        ("LJ7", grid_start(7), lennard_jones, True, 0.01),
        ("LJ13", grid_start(13), lennard_jones, True, 0.01),
        ("LJ38", grid_start(38), lennard_jones, True, 0.01),
    ]
    if morse_x is not None:
        problems.append(("MorsePt256", morse_x, morse_fn(cell), True, 0.01))
    for name, x0, fn, cluster, tol in problems:
        for kind in ("lbfgs", "fire", "fire2"):
            evals, energy, grad, ms, ok = time_arm(ase_once, kind, x0, fn, cluster, tol)
            line = f"{name},ase-{kind},ase,{evals},{energy:.8e},{grad:.8e},{ms:.6f},{'yes' if ok else 'no'}"
            rows.append(line)
            print(line, flush=True)
    write_rows(RAW / "ase.csv", rows)


def run_scipy():
    rows = []
    problems = [
        ("Rosenbrock2", np.array([-1.2, 1.0]), rosenbrock, False, 1e-5),
        ("Rosenbrock10", np.array([-1.2 if i % 2 == 0 else 1.0 for i in range(10)]), rosenbrock, False, 1e-5),
        ("LJ7", grid_start(7), lennard_jones, True, 0.01),
        ("LJ13", grid_start(13), lennard_jones, True, 0.01),
    ]
    for name, x0, fn, cluster, tol in problems:
        kinds = ("lbfgsb", "cg") if not cluster else ("lbfgsb",)
        for kind in kinds:
            evals, energy, grad, ms, ok = time_arm(scipy_once, kind, x0, fn, cluster, tol)
            line = f"{name},scipy-{kind},scipy,{evals},{energy:.8e},{grad:.8e},{ms:.6f},{'yes' if ok else 'no'}"
            rows.append(line)
            print(line, flush=True)
    write_rows(RAW / "scipy.csv", rows)


def merge():
    lines = ["problem,arm,library,evals,energy,grad,median_ms,converged"]
    for name in ("ase.csv", "eon.csv", "rgmin.csv", "scipy.csv"):
        path = RAW / name
        if not path.exists():
            continue
        body = path.read_text().splitlines()[1:]
        lines.extend(row for row in body if row.strip())
    text = [
        "# Optimizer comparison",
        "",
        "Force calls and median wall time. Five serial repeats.",
        "Clusters stop at a max-atom force of 0.01 and a step cap of 0.2.",
        "Rosenbrock stops at an infinity-norm of 1e-5.",
        "rgmin-eon is the default step with memory 8. rgmin-eon-m20 uses memory 20, the eOn default.",
        "rgmin-scaled shortens the quasi-Newton step and keeps the pairs.",
        "rgmin-fire uses the eOn FIRE time step of 1. rgmin-fire2 and ASE FIRE2 use 0.1.",
        "rgmin-cg is the eOn line-search-free conjugate-gradient step.",
        "rgmin-eon, rgmin-fire and rgmin-cg fill the gradient buffer in place.",
        "The eOn columns are the client algorithms transcribed in bench/eon_ref.cpp",
        "from TheochemUI/eOn commit 579738a9387e8d0ea55496f45dd057812cfab892.",
        "SciPy L-BFGS-B has no per-atom cap. Wall times include the potential.",
        "Raw rows are bench/raw/*.csv.",
        "",
        "| problem | arm | evals | energy | grad | median ms | converged |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    rows = []
    for line in lines[1:]:
        problem, arm, _lib, evals, energy, grad, ms, ok = line.split(",")
        rows.append((problem, arm, int(evals), float(energy), float(grad), float(ms), ok))
    for problem, arm, evals, energy, grad, ms, ok in rows:
        text.append(
            f"| {problem} | {arm} | {evals} | {energy:.6e} | {grad:.3e} | {ms:.3f} | {ok} |"
        )
    text.append("")
    TABLE.write_text("\n".join(text))


def main():
    run_ase()
    run_scipy()
    merge()


if __name__ == "__main__":
    main()
