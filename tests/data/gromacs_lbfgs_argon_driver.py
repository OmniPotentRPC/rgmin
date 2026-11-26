#!/usr/bin/env python3
"""Reproduce tests/data/gromacs_lbfgs_argon.txt with gmx_d 2023.3.

Each case reruns ``gmx_d mdrun`` for ``nsteps = 0, 1, ..., nsteps``.
GROMACS treats that limit as inclusive (``step <= nsteps``), so the row
``k`` is the minimizer after the accepted iterations ``0..=k``. The
energy and maximum force are the double-precision footer. The evaluation
count is the ``Performed N energy evaluations`` line in ``md.log``, which
includes the initial point and every line-search trial.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

START = """TITLE
argon seven-cluster
END
POSITION
    1 AR    AR      1    5.000000000    5.000000000    5.000000000
    2 AR    AR      2    5.381637096    5.000000000    4.950000000
    3 AR    AR      3    5.190818548    5.330507421    5.050000000
    4 AR    AR      4    4.809181452    5.330507421    4.950000000
    5 AR    AR      5    4.618362904    5.000000000    5.050000000
    6 AR    AR      6    4.809181452    4.669492579    4.950000000
    7 AR    AR      7    5.190818548    4.669492579    5.050000000
END
BOX
   10.000000000   10.000000000   10.000000000
END
"""

# (nbfgscorr, emstep, largest inclusive nsteps)
CASES = (
    (10, 0.02, 12),
    (1, 0.02, 8),
    (3, 0.01, 10),
    (5, 0.05, 8),
    (10, 0.1, 6),
)
EMTOL = "1e-8"


def gmx_version() -> str:
    completed = subprocess.run(
        ["gmx_d", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines() + completed.stderr.splitlines():
        if "GROMACS version:" in line:
            return line.split(":", 1)[1].strip()
    raise RuntimeError("gmx_d --version did not report a version")


def one_run(topology: str, nbfgscorr: int, emstep: float, nsteps: int) -> tuple[str, str, str]:
    mdp = f"""integrator = l-bfgs
nsteps = {nsteps}
emtol = {EMTOL}
emstep = {emstep}
nbfgscorr = {nbfgscorr}
cutoff-scheme = Verlet
nstlist = 1
verlet-buffer-tolerance = -1
vdwtype = Cut-off
vdw-modifier = None
rvdw = 3.0
rlist = 3.0
coulombtype = Cut-off
rcoulomb = 3.0
DispCorr = no
pbc = xyz
constraints = none
nstxout = 0
nstvout = 0
nstfout = 0
nstenergy = 0
nstlog = 1
"""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "start.g96").write_text(START, encoding="ascii")
        (work / "em.mdp").write_text(mdp, encoding="ascii")
        (work / "top.top").write_text(topology, encoding="ascii")
        subprocess.run(
            [
                "gmx_d",
                "-quiet",
                "grompp",
                "-f",
                "em.mdp",
                "-p",
                "top.top",
                "-c",
                "start.g96",
                "-o",
                "em.tpr",
                "-maxwarn",
                "1",
            ],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        )
        completed = subprocess.run(
            [
                "gmx_d",
                "-quiet",
                "mdrun",
                "-v",
                "-s",
                "em.tpr",
                "-nt",
                "1",
                "-ntmpi",
                "1",
                "-c",
                "end.g96",
            ],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        )
        log = (work / "md.log").read_text(encoding="utf-8", errors="replace")
    energy = re.search(r"Potential Energy\s+=\s+(\S+)", completed.stderr)
    fmax = re.search(r"Maximum force\s+=\s+(\S+)", completed.stderr)
    evaluations = re.search(r"Performed (\d+) energy evaluations", log)
    if energy is None or fmax is None or evaluations is None:
        raise RuntimeError(completed.stderr)
    return energy.group(1), fmax.group(1), evaluations.group(1)


def main() -> None:
    topology = (Path(__file__).resolve().parent / "gromacs_steep_argon.top").read_text(encoding="ascii")
    version = gmx_version()
    print(f"# gmx_d {version} double precision")
    print("# package gromacs 2023.3, -nt 1 -ntmpi 1")
    print("# integrator=l-bfgs emtol=1e-8 constraints=none pbc=xyz")
    print("# vdwtype=Cut-off vdw-modifier=None rvdw=3.0 DispCorr=no")
    print("# Row k is the inclusive run nsteps=k (accepted steps 0..=k).")
    print("# Columns: k evaluations energy fmax")
    print("# energy and fmax are the double-precision footer (%21.14e).")
    print("# evaluations counts the initial point and every line-search trial.")
    print("sigma 0.34")
    print("epsilon 1.0")
    print("cutoff 3.0")
    print(f"emtol {EMTOL}")
    for nbfgscorr, emstep, nsteps in CASES:
        print(f"case {nbfgscorr} {emstep} {nsteps}")
        for k in range(nsteps + 1):
            energy, fmax, evaluations = one_run(topology, nbfgscorr, emstep, k)
            print(f"{k} {evaluations} {energy} {fmax}")


if __name__ == "__main__":
    main()
