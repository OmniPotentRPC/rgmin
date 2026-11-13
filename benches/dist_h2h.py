# Copyright (c) rgmin developers
"""Fifty-start distribution of rgmin and ASE.

LJ13, LJ38 and LJ75 use 50 seeded starts. Morse Pt256 is the one
fixture. fmax is 0.01. The rgmin default is the eOn cap. The matched
arm uses ASE's memory 100 and fixed H0 = 1/70. Wolfe is opt-in. FIRE
2.0 uses timestep 0.1 and a Euclidean cap of 0.2. ASE LBFGS, FIRE and
BFGSLineSearch keep their own defaults apart from fmax and maxstep 0.2.

    env -i PATH=".venv/bin:$PATH" HOME="$HOME" LANG="${LANG:-C.UTF-8}" \
        python benches/dist_h2h.py

Raw logs stay in benches/ase_h2h/raw/. The table is
benches/ase_h2h/distribution.md.
"""

from __future__ import annotations

import io
import subprocess
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import BFGSLineSearch, FIRE, LBFGS

from ase_h2h import Counted, lennard_jones, morse_fn, read_con

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(__file__).resolve().parent / "ase_h2h" / "raw"
TABLE = Path(__file__).resolve().parent / "ase_h2h" / "distribution.md"
CON = ROOT / "tests" / "data" / "morse_pos_0000.con"

# Previous default: session Wolfe, memory 8, fmax 0.01, same starts.
BEFORE = {
    "LJ13": (50, 50, 320.5, 329, -30.644585),
    "LJ38": (50, 50, 826.0, 320, -164.186816),
    "LJ75": (50, 50, 1317.5, 473, -376.430563),
}


def stats(calls, energy, conv):
    n = len(calls)
    c = sorted(calls)
    q1, q3 = c[n // 4], c[3 * n // 4]
    med = 0.5 * (c[n // 2 - 1] + c[n // 2]) if n % 2 == 0 else c[n // 2]
    e = sorted(energy)
    em = 0.5 * (e[n // 2 - 1] + e[n // 2]) if n % 2 == 0 else e[n // 2]
    return med, q3 - q1, em, sum(conv), n


def run_rgmin(starts_path: Path) -> tuple[dict, list[dict]]:
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--locked",
            "--release",
            "--example",
            "dist_h2h",
            "--",
            str(starts_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "dist-rgmin.txt").write_text(proc.stdout)
    (RAW / "dist-rgmin.err").write_text(proc.stderr)
    meta = {}
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "meta":
            meta[parts[1]] = parts[2]
        elif parts[0] == "summary":
            rows.append(
                {
                    "problem": parts[1],
                    "arm": parts[2],
                    "n": int(parts[3]),
                    "conv": int(parts[4]),
                    "med": float(parts[5]),
                    "iqr": int(parts[6]),
                    "emed": float(parts[7]),
                }
            )
    return meta, rows


def load_starts(path: Path) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        name = parts[0]
        coords = np.asarray([float(v) for v in parts[2:]], dtype=float)
        out.setdefault(name, []).append(coords)
    return out


def run_ase_one(name, x0, fn, cell, fmax, budget, make) -> tuple[int, float, bool]:
    atoms = Atoms(positions=x0.reshape(-1, 3).copy())
    if cell is not None:
        atoms.set_cell(cell)
        atoms.set_pbc(True)
    calc = Counted(fn)
    atoms.calc = calc
    dyn = make(atoms)
    calc.calls = 0
    try:
        ok = bool(dyn.run(fmax=fmax, steps=budget))
    except RuntimeError:
        ok = False
    energy = float(atoms.get_potential_energy())
    return calc.calls, energy, ok


def run_ase(starts, fmax, budget) -> list[dict]:
    _, cell = read_con(CON)
    morse = morse_fn(cell)
    makers = {
        "ase-lbfgs": lambda atoms: LBFGS(atoms, logfile=io.StringIO(), maxstep=0.2),
        "ase-fire": lambda atoms: FIRE(atoms, logfile=io.StringIO(), maxstep=0.2),
        "ase-bfgslinesearch": lambda atoms: BFGSLineSearch(
            atoms, logfile=io.StringIO(), maxstep=0.2
        ),
    }
    rows = []
    for problem, xs in starts.items():
        fn = lennard_jones if problem.startswith("LJ") else morse
        this_cell = None if problem.startswith("LJ") else cell
        for arm, make in makers.items():
            calls, energy, conv = [], [], []
            t0 = time.perf_counter()
            for x0 in xs:
                c, e, ok = run_ase_one(problem, x0, fn, this_cell, fmax, budget, make)
                calls.append(c)
                energy.append(e)
                conv.append(ok)
            med, iqr, em, k, n = stats(calls, energy, conv)
            rows.append(
                {
                    "problem": problem,
                    "arm": arm,
                    "n": n,
                    "conv": k,
                    "med": med,
                    "iqr": iqr,
                    "emed": em,
                }
            )
            dt = time.perf_counter() - t0
            print(
                f"{problem:12} {arm:22} {k}/{n} med {med:.1f} IQR {iqr} E {em:.3f} ({dt:.1f}s)",
                flush=True,
            )
    return rows


def write_table(rows: list[dict]) -> None:
    order = [
        "ase-lbfgs",
        "ase-fire",
        "ase-bfgslinesearch",
        "rgmin-default",
        "rgmin-matched",
        "rgmin-wolfe",
        "rgmin-fire2",
    ]
    problems = []
    for row in rows:
        if row["problem"] not in problems:
            problems.append(row["problem"])
    by = {(r["problem"], r["arm"]): r for r in rows}
    lines = [
        "# Distribution head-to-head",
        "",
        "Fifty seeded starts for LJ13, LJ38 and LJ75. Morse Pt256 is one",
        "structure. fmax 0.01. A start counts as converged when that force",
        "is reached within 5000 steps. The median of an even sample is the",
        "mean of the two central values. IQR is Q3 minus Q1 at indexes",
        "`n/4` and `3n/4` of the sorted force calls.",
        "",
        "rgmin-default is the line-search-free cap, memory 8, Nocedal-Wright",
        "H0. rgmin-matched is that cap with memory 100 and H0 fixed at 1/70,",
        "ASE LBFGS's defaults. rgmin-wolfe is the opt-in strong Wolfe search.",
        "rgmin-fire2 uses timestep 0.1 and a Euclidean cap of 0.2.",
        "ASE keeps its own memory, Hessian scale and FIRE parameters, with",
        "maxstep 0.2.",
        "",
        "## Before: Wolfe was the session default",
        "",
        "Same starts and fmax, measured before the default changed.",
        "",
        "| problem | conv | median calls | IQR | median energy |",
        "|---|---:|---:|---:|---:|",
    ]
    for problem, (n, k, med, iqr, em) in BEFORE.items():
        lines.append(f"| {problem} | {k}/{n} | {med:.1f} | {iqr} | {em:.3f} |")
    lines.extend(
        [
            "",
            "## After",
            "",
            "| problem | arm | conv | median calls | IQR | median energy |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for problem in problems:
        for arm in order:
            row = by[(problem, arm)]
            lines.append(
                f"| {problem} | {arm} | {row['conv']}/{row['n']} | {row['med']:.1f} | "
                f"{row['iqr']} | {row['emed']:.3f} |"
            )
    lines.append("")
    TABLE.write_text("\n".join(lines))


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    starts_path = RAW / "starts.tsv"
    meta, rgmin_rows = run_rgmin(starts_path)
    for row in rgmin_rows:
        print(
            f"{row['problem']:12} {row['arm']:22} {row['conv']}/{row['n']} "
            f"med {row['med']:.1f} IQR {row['iqr']} E {row['emed']:.3f}",
            flush=True,
        )
    starts = load_starts(starts_path)
    ase_rows = run_ase(starts, float(meta["fmax"]), int(meta["budget"]))
    write_table(rgmin_rows + ase_rows)


if __name__ == "__main__":
    main()
