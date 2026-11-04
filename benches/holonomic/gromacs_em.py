#!/usr/bin/env python3
"""GROMACS energy minimization with constraints, for benches/holonomic.rs.

Runs gmx_d (double precision) on the systems of
tests/data/ase_fix_bond_lengths.txt and on 216 SPC/E waters kept rigid
by SETTLE, and writes benches/holonomic/data/gromacs_em.txt. The bench
evaluates the same energies in Rust and runs rgmin from the same starts.

    GMX=/path/to/gmx_d python3 benches/holonomic/gromacs_em.py

The three small systems keep the units of the ASE data read as nm and
kJ/mol: Lennard-Jones with sigma = 1 nm and epsilon = 1 kJ/mol on every
pair (nrexcl 0 and type 2 constraints, which make no exclusions), or
unit harmonic bonds of type 6 on the non-bonded ethane pairs. A cubic
box of 30 nm with a 10 nm cutoff and the Lennard-Jones force switch
from 9 nm leaves every pair inside the switch and no periodic image
within the cutoff, so the forces are the plain pair forces of the ASE
data and the energies differ from them by a constant.

The water box takes the oxygen positions and orientations of
share/top/spc216.gro, rebuilt on the SETTLE geometry (O-H 0.1 nm, H-H
0.1633 nm), with the SPC/E charges and oxygen Lennard-Jones of
oplsaa.ff/spce.itp. Electrostatics are reaction field with
epsilon_rf = 0 (a conductor): the pair energy is
f q_i q_j (1 / r + k_rf r^2 - c_rf) with k_rf = 1 / (2 r_c^3) and
c_rf = 3 / (2 r_c), whose force vanishes at r_c. Excluded pairs add
f q_i q_j k_rf r^2, which is constant on rigid molecules. Lennard-Jones
uses the GROMACS force switch from 0.8 nm. Energy and force are then
continuous at r_c = 0.9 nm, so a minimizer can drive the force to its
rounding level. The cutoff is on atom pairs with the minimum image in
the 1.86206 nm box, and the pair list is rebuilt at every step.

GROMACS sets every mass to one in minimization (mdlib/mdatoms.cpp), so
its constraint corrections are unweighted. The data carry the masses of
the ASE data and of spce.itp for the bench's mass-weighted runs.

Every run stops at emtol eps^(1/3), the gradient tolerance of the ASE
data. On SPC/E the energy differences a step compares fall below the
rounding of a total near -1.2e4 kJ/mol before the force reaches it, so
GROMACS stops there at machine precision; SPC/E also runs at the
GROMACS default emtol 10 kJ mol^-1 nm^-1.

Runs:
  steep-lincs4   steepest descent, LINCS order 4 (the default), 1 iteration
  steep-lincs8   steepest descent, LINCS order 8
  cg-lincs8      Polak-Ribiere CG, LINCS order 8 (the order grompp asks for)
  lbfgs-lincs8   L-BFGS: mdrun stops, constraints are not implemented
  steep-shake    steepest descent with SHAKE: mdrun stops, SHAKE cannot
                 project the forces; on SETTLE water SHAKE has nothing to do
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GMX = os.environ.get("GMX", "gmx_d")
# The UNCMIN gradient tolerance eps^(1/3), as in the ASE data, and the
# GROMACS default emtol (gmxpreprocess/readir.cpp).
EMTOL = float(np.finfo(float).eps ** (1.0 / 3.0))
EMTOL_DEFAULT = 10.0
# Budgets only: a run that reaches one is reported as such.
NSTEPS = 50_000
TOY_BOX, TOY_RC, TOY_RSW = 30.0, 10.0, 9.0
WATER_RC, WATER_RSW = 0.9, 0.8
DOH, DHH = 0.1, 0.16330
SPCE_Q = (-0.8476, 0.4238, 0.4238)
SPCE_SIGMA, SPCE_EPS = 3.16557e-01, 6.50194e-01
SPCE_MASSES = (15.9994, 1.008, 1.008)
EMSTEP = 0.01

RUNS = {
    "steep-lincs4": ("steep", "lincs", 4),
    "steep-lincs8": ("steep", "lincs", 8),
    "cg-lincs8": ("cg", "lincs", 8),
    "lbfgs-lincs8": ("l-bfgs", "lincs", 8),
    "steep-shake": ("steep", "shake", 4),
}


def ase_systems():
    """Starts, pairs and springs of tests/data/ase_fix_bond_lengths.txt."""
    systems, cur = [], None
    for line in (ROOT / "tests/data/ase_fix_bond_lengths.txt").read_text().splitlines():
        w = line.split()
        if not w or w[0].startswith("#"):
            continue
        if w[0] == "system":
            cur = dict(name=w[1], n=int(w[2]), springs=None, masses=None)
            systems.append(cur)
        elif w[0] == "start":
            cur["x"] = np.array([float(v) for v in w[1:]]).reshape(-1, 3)
        elif w[0] == "pairs":
            ids = [int(v) for v in w[1:]]
            cur["pairs"] = list(zip(ids[0::2], ids[1::2]))
        elif w[0] == "springs":
            t = w[1:]
            cur["springs"] = [(int(t[k]), int(t[k + 1]), float(t[k + 2]))
                              for k in range(0, len(t), 3)]
        elif w[0] == "masses":
            cur["masses"] = [float(v) for v in w[1:]]
    return systems


def spce216():
    """216 rigid waters with the oxygens and H-O-H planes of spc216.gro."""
    gro = Path(os.environ.get("GMXDATA", "")) / "top/spc216.gro"
    if not gro.exists():
        share = Path(subprocess.run([GMX, "-quiet", "--version"], capture_output=True,
                                    text=True).stdout.split("Data prefix:")[1].split()[0])
        gro = share / "share/gromacs/top/spc216.gro"
    lines = gro.read_text().splitlines()
    n = int(lines[1])
    raw = np.array([[float(v) for v in l[20:44].split()] for l in lines[2:2 + n]])
    box = float(lines[2 + n].split()[0])
    half = np.arcsin(DHH / 2.0 / DOH)
    x = []
    for o, h1, h2 in raw.reshape(-1, 3, 3):
        bis = (h1 - o) + (h2 - o)
        bis /= np.linalg.norm(bis)
        side = (h1 - o) - (h2 - o)
        side -= side.dot(bis) * bis
        side /= np.linalg.norm(side)
        for s in (0.0, 1.0, -1.0):
            x.append(o if s == 0.0 else
                     o + DOH * (np.cos(half) * bis + s * np.sin(half) * side))
    return np.array(x), box


def g96(path: Path, x: np.ndarray, box: float, names, per_residue: int) -> None:
    rows = [f"{1 + k // per_residue:5d} MOL   {names[k]:5s}{k + 1:7d}"
            f"{x[k, 0]:15.9f}{x[k, 1]:15.9f}{x[k, 2]:15.9f}" for k in range(len(x))]
    path.write_text("TITLE\nrgmin holonomic bench\nEND\nPOSITION\n" + "\n".join(rows)
                    + f"\nEND\nBOX\n{box:15.9f}{box:15.9f}{box:15.9f}\nEND\n")


def read_g96(path: Path) -> np.ndarray:
    text = path.read_text().split("POSITION")[1].split("END")[0]
    return np.array([[float(v) for v in l.split()[-3:]] for l in text.splitlines() if l.strip()])


def toy_top(sys_: dict) -> tuple[str, list[str]]:
    n = sys_["n"]
    lj = sys_["springs"] is None
    c6c12 = "4.0 4.0" if lj else "0.0 0.0"
    atoms = "\n".join(f"{k + 1} A 1 MOL A{k + 1} {k + 1} 0.0 1.0" for k in range(n))
    cons = "\n".join(f"{i + 1} {j + 1} 2 {float(np.linalg.norm(sys_['x'][i] - sys_['x'][j]))!r}"
                     for i, j in sys_["pairs"])
    bonds = "" if lj else "[ bonds ]\n" + "\n".join(
        f"{i + 1} {j + 1} 6 {r0!r} 1.0" for i, j, r0 in sys_["springs"]) + "\n"
    top = (f"[ defaults ]\n1 1 no 1.0 1.0\n[ atomtypes ]\nA 1.0 0.0 A {c6c12}\n"
           f"[ moleculetype ]\nMOL 0\n[ atoms ]\n{atoms}\n{bonds}"
           f"[ constraints ]\n{cons}\n[ system ]\n{sys_['name']}\n[ molecules ]\nMOL 1\n")
    return top, [f"A{k + 1}" for k in range(n)]


def water_top(nmol: int) -> tuple[str, list[str]]:
    c6 = 4.0 * SPCE_EPS * SPCE_SIGMA**6
    c12 = 4.0 * SPCE_EPS * SPCE_SIGMA**12
    top = (f"[ defaults ]\n1 1 no 1.0 1.0\n[ atomtypes ]\n"
           f"OW {SPCE_MASSES[0]!r} 0.0 A {c6!r} {c12!r}\n"
           f"HW {SPCE_MASSES[1]!r} 0.0 A 0.0 0.0\n"
           "[ moleculetype ]\nSOL 2\n[ atoms ]\n"
           f"1 OW 1 SOL OW 1 {SPCE_Q[0]}\n2 HW 1 SOL HW1 1 {SPCE_Q[1]}\n"
           f"3 HW 1 SOL HW2 1 {SPCE_Q[2]}\n"
           f"[ settles ]\n1 1 {DOH} {DHH}\n[ exclusions ]\n1 2 3\n2 1 3\n3 1 2\n"
           f"[ system ]\nspce216\n[ molecules ]\nSOL {nmol}\n")
    return top, ["OW", "HW1", "HW2"] * nmol


def mdp(integrator: str, algorithm: str, order: int, rc: float, rsw: float,
        emtol: float) -> str:
    return "\n".join([
        f"integrator = {integrator}", f"nsteps = {NSTEPS}", f"emtol = {emtol!r}",
        f"emstep = {EMSTEP!r}", "nstcgsteep = 1000", "cutoff-scheme = Verlet", "nstlist = 1",
        "verlet-buffer-tolerance = -1", f"rlist = {rc}", "coulombtype = Reaction-Field",
        "epsilon-rf = 0", f"rcoulomb = {rc}", "vdwtype = Cut-off",
        "vdw-modifier = Force-switch", f"rvdw-switch = {rsw}", f"rvdw = {rc}", "DispCorr = no", "pbc = xyz", "constraints = none",
        f"constraint-algorithm = {algorithm}", f"lincs-order = {order}", "lincs-iter = 1",
        "shake-tol = 1e-12", "nstxout = 0", "nstvout = 0", "nstfout = 0",
        "nstenergy = 0", "nstlog = 0", "nstcalcenergy = 1", ""])


def run(work: Path, top: str, names, x: np.ndarray, box: float, rc: float, rsw: float,
        cfg, per_residue: int, emtol: float) -> dict:
    integrator, algorithm, order = cfg
    (work / "topol.top").write_text(top)
    (work / "em.mdp").write_text(mdp(integrator, algorithm, order, rc, rsw, emtol))
    g96(work / "start.g96", x, box, names, per_residue)
    pre = subprocess.run([GMX, "-quiet", "grompp", "-f", "em.mdp", "-c", "start.g96",
                          "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "5"],
                         cwd=work, capture_output=True, text=True)
    if pre.returncode != 0:
        return dict(outcome="grompp-error", message=fatal(pre.stderr))
    t0 = time.perf_counter()
    md = subprocess.run([GMX, "-quiet", "mdrun", "-s", "em.tpr", "-nt", "1", "-c", "end.g96",
                         "-g", "md.log", "-e", "ener.edr", "-o", "traj.trr"],
                        cwd=work, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if md.returncode != 0:
        return dict(outcome="mdrun-error", message=fatal(md.stderr))
    log = (work / "md.log").read_text()
    head = re.search(r"\n(\S.*?) (converged to Fmax < \S+|converged to machine precision"
                     r"|did not converge to Fmax < \S+) in (\d+) steps", log)
    outcome = {"converged": "fmax", "did": "budget"}.get(head.group(2).split()[0], "fmax")
    if "machine precision" in head.group(2):
        outcome = "machine-precision"
    evals = re.search(r"Performed (\d+) energy evaluations", log)
    steps = int(head.group(3))
    timed = re.search(r"Time:\s+(\S+)\s+(\S+)", log)
    return dict(outcome=outcome, steps=steps,
                evals=int(evals.group(1)) if evals else steps + 1,
                energy=float(re.search(r"Potential Energy  =\s*(\S+)", log).group(1)),
                fmax=float(re.search(r"Maximum force     =\s*(\S+)", log).group(1)),
                mdrun=float(timed.group(2)) if timed else wall, wall=wall,
                x=read_g96(work / "end.g96"))


def fatal(stderr: str) -> str:
    """The text of the first GROMACS error box, on one line."""
    m = re.search(r"\n(?:Fatal error|Assertion failed|ERROR \d+[^\n]*):\s*\n(.*?)\n\n",
                  stderr, re.S)
    text = m.group(1) if m else stderr.strip().splitlines()[-1]
    return " ".join(text.split())


def main() -> None:
    version = subprocess.run([GMX, "-quiet", "--version"], capture_output=True,
                             text=True).stdout
    gv = re.search(r"GROMACS version:\s*(\S+)", version).group(1)
    simd = re.search(r"SIMD instructions:\s*(\S+)", version).group(1)
    # The compiler name and version, without its path. How it fuses
    # multiply-adds sets the last bits of every run, and through them the
    # step counts of the long SPC/E runs.
    cxx = re.search(r"C\+\+ compiler:\s*\S+\s+(.*)", version).group(1).strip()
    lines = [f"# GROMACS {gv} double precision, SIMD {simd}, {cxx}, mdrun -nt 1,"
             f" {platform.processor() or platform.machine()}",
             f"emstep {EMSTEP!r}",
             "# system name atoms box rc rvdw_switch kind; start (3N);"
             " pairs (i j) or settles d_OH d_HH; springs or lj or spce;"
             " masses (N) if real masses differ",
             "# run label emtol outcome steps energy_evaluations potential max_force"
             " mdrun_seconds; end (3N)",
             "# failed label emtol stage message"]
    systems = []
    for s in ase_systems():
        # The .g96 start keeps nine decimals; both programs start there.
        s["x"] = np.round(s["x"], 9)
        top, names = toy_top(s)
        systems.append((s["name"], s["x"], TOY_BOX, TOY_RC, TOY_RSW, top, names, s))
    xw, box = spce216()
    xw = np.round(xw, 9)
    top, names = water_top(len(xw) // 3)
    systems.append(("spce216", xw, box, WATER_RC, WATER_RSW, top, names, None))
    for name, x, box, rc, rsw, top, names, s in systems:
        kind = "spce" if s is None else ("lj" if s["springs"] is None else "springs")
        lines.append(f"system {name} {len(x)} {box!r} {rc!r} {rsw!r} {kind}")
        lines.append("start " + " ".join(repr(float(v)) for v in x.ravel()))
        if s is None:
            lines.append(f"settles {DOH!r} {DHH!r}")
            lines.append("masses " + " ".join(repr(m) for m in SPCE_MASSES * (len(x) // 3)))
        else:
            lines.append("pairs " + " ".join(f"{i} {j}" for i, j in s["pairs"]))
            if s["springs"]:
                lines.append("springs " + " ".join(f"{i} {j} {r0!r}"
                                                   for i, j, r0 in s["springs"]))
            if s["masses"]:
                lines.append("masses " + " ".join(repr(m) for m in s["masses"]))
        # SPC/E: the energy-tested steps reach machine precision above
        # EMTOL, so the default emtol gives a stop both programs reach.
        for emtol in (EMTOL,) if s is not None else (EMTOL, EMTOL_DEFAULT):
            for label, cfg in RUNS.items():
                with tempfile.TemporaryDirectory() as tmp:
                    r = run(Path(tmp), top, names, x, box, rc, rsw, cfg,
                            3 if s is None else len(x), emtol)
                if "message" in r:
                    lines.append(f"failed {label} {emtol!r} {r['outcome']} {r['message']}")
                    print(f"{name:8s} {label:13s} {emtol:.3g} {r['outcome']}:"
                          f" {r['message'][:60]}")
                    continue
                lines.append(f"run {label} {emtol!r} {r['outcome']} {r['steps']} {r['evals']}"
                             f" {r['energy']!r} {r['fmax']!r} {r['mdrun']!r}")
                lines.append("end " + " ".join(repr(float(v)) for v in r["x"].ravel()))
                print(f"{name:8s} {label:13s} {emtol:.3g} {r['outcome']:17s}"
                      f" steps {r['steps']:6d} E {r['energy']:.10g} Fmax {r['fmax']:.3g}"
                      f" {r['mdrun']:.3f} s")
    out = HERE / "data" / "gromacs_em.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
