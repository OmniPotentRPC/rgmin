"""LAMMPS inputs and log parsing shared by the min_style rgmin tests and benchmark.

Every system is generated from LAMMPS commands, so the only external file
is a Cu EAM potential that ships in the LAMMPS ``potentials`` directory
(found through ``LAMMPS_POTENTIALS``).
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

CU_A0 = 3.615
CU_POTENTIAL = "Cu_mishin1.eam.alloy"


def lj_cluster(seed: int = 4928) -> str:
    """Rattled fcc LJ cluster of 177 atoms in vacuum, reduced units."""
    return f"""
units lj
atom_modify map array
boundary p p p
lattice fcc 0.95
region box block -8 8 -8 8 -8 8
create_box 1 box
region ball sphere 0 0 0 2.2
create_atoms 1 region ball
mass 1 1.0
displace_atoms all random 0.04 0.04 0.04 {seed}
pair_style lj/cut 2.5
pair_modify shift yes
pair_coeff 1 1 1.0 1.0
neighbor 0.3 bin
"""


def _cu_header(cells: int) -> str:
    return f"""
units metal
atom_modify map array
boundary p p p
lattice fcc {CU_A0}
region box block 0 {cells} 0 {cells} 0 {cells}
create_box 1 box
create_atoms 1 box
mass 1 63.546
pair_style eam/alloy
pair_coeff * * {CU_POTENTIAL} Cu
neighbor 1.0 bin
"""


def cu_vacancy(cells: int = 4, seed: int = 7193, rattle: float = 0.02) -> str:
    """fcc Cu with one vacancy: 4 x 4 x 4 cells hold 255 atoms."""
    return (
        _cu_header(cells)
        + f"""
region vac sphere 0 0 0 0.1 units box
delete_atoms region vac compress yes
displace_atoms all random {rattle} {rattle} {rattle} {seed} units box
"""
    )


def cu_dumbbell_vacancy(cells: int = 5, seed: int = 2231) -> str:
    """fcc Cu with a <100> split dumbbell at (2,2,2) a and a vacancy at the origin.

    5 x 5 x 5 cells hold 500 atoms: one site removed, one site replaced by
    the pair 2.17 A apart along x, 12.5 A from the vacancy.
    """
    c = 2 * CU_A0
    d = 0.3 * CU_A0
    return (
        _cu_header(cells)
        + f"""
region vac sphere 0 0 0 0.1 units box
region sia sphere {c} {c} {c} 0.1 units box
delete_atoms region vac compress no
delete_atoms region sia compress no
create_atoms 1 single {c - d} {c} {c} units box
create_atoms 1 single {c + d} {c} {c} units box
reset_atoms id
displace_atoms all random 0.02 0.02 0.02 {seed} units box
"""
    )


def cu_large(cells: int = 20, seed: int = 9001) -> str:
    """32,000-site Cu cell with one vacancy, every atom rattled by up to 0.05 A."""
    return cu_vacancy(cells=cells, seed=seed, rattle=0.05)


SYSTEMS = {
    "lj177": lj_cluster,
    "cu-vac": cu_vacancy,
    "cu-sia-vac": cu_dumbbell_vacancy,
    "cu32k-vac": cu_large,
}


def deck(
    system: str,
    style: str,
    modify: str = "",
    etol: float = 0.0,
    ftol: float = 1e-6,
    maxiter: int = 10000,
    maxeval: int = 100000,
    thermo: int = 0,
    extra: str = "",
    plugin: str | None = None,
) -> str:
    """One minimization of ``system`` with ``min_style style``.

    ``extra`` goes after the thermo settings, so it may hold fixes and
    ``thermo_modify`` options.
    """
    body = SYSTEMS[system]()
    load = f"plugin load {plugin}\n" if plugin else ""
    lines = [
        body,
        load,
        "thermo_style custom step pe fnorm fmax",
        "thermo_modify format float %.17g",
        f"thermo {thermo}",
        extra,
        f"min_style {style}",
        modify,
        f"minimize {etol!r} {ftol!r} {maxiter} {maxeval}",
        'print "RGMIN_RESULT pe $(pe:%.17g) fnorm $(fnorm:%.17g) fmax $(fmax:%.17g)"',
    ]
    return "\n".join(line for line in lines if line) + "\n"


@dataclass
class Result:
    """Fields of one LAMMPS minimization log."""

    returncode: int
    log: str
    pe: float = float("nan")
    fnorm: float = float("nan")
    fmax: float = float("nan")
    iterations: int = -1
    evaluations: int = -1
    stop: str = ""
    loop_time: float = float("nan")
    thermo: list[tuple[int, float, float]] = field(default_factory=list)

    @property
    def error(self) -> str:
        m = re.search(r"^ERROR.*$", self.log, re.MULTILINE)
        return m.group(0) if m else ""


def parse(log: str, returncode: int) -> Result:
    res = Result(returncode=returncode, log=log)
    m = re.search(r"RGMIN_RESULT pe (\S+) fnorm (\S+) fmax (\S+)", log)
    if m:
        res.pe, res.fnorm, res.fmax = (float(v) for v in m.groups())
    m = re.search(r"Iterations, force evaluations = (\d+) (\d+)", log)
    if m:
        res.iterations, res.evaluations = int(m.group(1)), int(m.group(2))
    m = re.search(r"Stopping criterion = (.*)", log)
    if m:
        res.stop = m.group(1).strip()
    m = re.search(r"Loop time of (\S+) on", log)
    if m:
        res.loop_time = float(m.group(1))
    block = re.search(r"Step\s+PotEng\s+Fnorm\s+Fmax\s*\n(.*?)\nLoop time", log, re.DOTALL)
    if block:
        for line in block.group(1).splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0].isdigit():
                res.thermo.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return res


def run(
    lmp: str,
    text: str,
    workdir: Path,
    nprocs: int = 1,
    mpirun: str | None = None,
    timeout: float = 1800,
) -> Result:
    """Run ``text`` with ``lmp`` in ``workdir``; MPI when ``nprocs > 1``.

    ``RGMIN_LAMMPS_RUNNER``, such as valgrind, prefixes ``lmp`` inside the
    MPI launcher.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "in.lmp").write_text(text)
    cmd = [lmp, "-in", "in.lmp", "-log", "log.lammps", "-echo", "none", "-screen", "none"]
    cmd = [*shlex.split(os.environ.get("RGMIN_LAMMPS_RUNNER", "")), *cmd]
    if nprocs > 1:
        launcher = shlex.split(mpirun or os.environ.get("RGMIN_LAMMPS_MPIRUN", "mpirun"))
        cmd = [*launcher, "-np", str(nprocs), *cmd]
    proc = subprocess.run(
        cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout, check=False
    )
    logfile = workdir / "log.lammps"
    log = logfile.read_text() if logfile.exists() else ""
    return parse(log + proc.stdout + proc.stderr, proc.returncode)
