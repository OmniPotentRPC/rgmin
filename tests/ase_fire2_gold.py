#!/usr/bin/env python3
"""ASE FIRE2 trajectories on the 3-D extended Rosenbrock function.

Writes tests/data/ase_fire2_rosenbrock.txt. src/fire.rs replays each
step from the ASE state on the Guenole 2020 preset.

The start (-1.2, 1, -1.2) is the extended Rosenbrock start of More,
Garbow and Hillstrom, ACM TOMS 7, 17 (1981). dtmax and dtmin are
Table 1 of Comput. Mater. Sci. 175 (2020) 109584,
DOI 10.1016/j.commatsci.2020.109584 (10 and 0.02 times dt); Nmin,
finc, fdec, astart and fa are the ASE defaults, which equal Table 1.
dt is the largest
power of ten that keeps the preset ceiling 10 dt below
2 / sqrt(lambda_max) of the Hessian at the minimum, the semi-implicit
Euler stability limit. fmax is the UNCMIN gradient tolerance
eps^(1/3) (Schnabel, Koontz and Weiss 1982; R 4.5.2
src/appl/uncmin.c, dfault), whose scaled gradient is the max-norm of g at this minimum;
the 2-norm here is the stricter test. maxstep is ASE's 0.2 with
use_abc. Without ABC ASE caps the step norm but keeps the uncapped
velocity, so that run takes no cap.

ASE FIRE2 omits Guenole's initial delay and clamps dt to dtmin where
Algorithm 2 skips the shrink. The script fails unless the first
non-positive power comes after the 20-step delay and no shrink
reaches dtmin, the two cases where the trajectories could differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import ase
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE2

FMAX = np.finfo(float).eps ** (1.0 / 3.0)
ABC_MAXSTEP = 0.2
N_DELAY = 20
# A budget only: the script fails if a run does not converge within it.
STEPS = 20_000


def rosenbrock(x: np.ndarray) -> tuple[float, np.ndarray]:
    e = 0.0
    g = np.zeros_like(x)
    for i in range(len(x) - 1):
        t = x[i + 1] - x[i] * x[i]
        u = 1.0 - x[i]
        e += 100.0 * t * t + u * u
        g[i] += -400.0 * x[i] * t - 2.0 * u
        g[i + 1] += 200.0 * t
    return e, g


class Rosenbrock(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        e, g = rosenbrock(self.atoms.positions[0].copy())
        self.results = {"energy": e, "forces": -g.reshape(1, 3)}


def hessian_at_minimum() -> np.ndarray:
    h = np.zeros((3, 3))
    for i in range(2):
        h[i, i] += 802.0
        h[i + 1, i + 1] += 200.0
        h[i, i + 1] = h[i + 1, i] = -400.0
    return h


class Logged(FIRE2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []
        self.first_reset = None
        self.floor_hit = False

    def step(self, f=None):
        if self.v is not None:
            power = np.vdot(-self.optimizable.get_gradient(), self.v)
            if power <= 0.0:
                if self.first_reset is None:
                    self.first_reset = self.nsteps + 1
                if self.dt * self.fdec < self.dtmin:
                    self.floor_hit = True
        super().step(f)
        state = [*self.optimizable.get_x(), *self.v, self.dt, self.a, self.Nsteps]
        self.rows.append(state)


def start_step() -> float:
    lam = np.linalg.eigvalsh(hessian_at_minimum()).max()
    return 10.0 ** np.floor(np.log10(2.0 / np.sqrt(lam) / 10.0))


def run(use_abc: bool, dt: float) -> tuple[list[np.ndarray], int]:
    atoms = Atoms("H", positions=[[-1.2, 1.0, -1.2]])
    calc = Rosenbrock()
    atoms.calc = calc
    maxstep = ABC_MAXSTEP if use_abc else float("inf")
    opt = Logged(atoms, logfile=None, dt=dt, dtmax=10 * dt, dtmin=0.02 * dt,
                 maxstep=maxstep, use_abc=use_abc)
    if not opt.run(fmax=FMAX, steps=STEPS):
        sys.exit(f"use_abc={use_abc}: not converged in {STEPS} steps")
    if opt.first_reset is not None and opt.first_reset <= N_DELAY:
        sys.exit(f"use_abc={use_abc}: P <= 0 at step {opt.first_reset}")
    if opt.floor_hit:
        sys.exit(f"use_abc={use_abc}: a shrink reached dtmin")
    return opt.rows, calc.calls


def main() -> None:
    dt = start_step()
    out = Path(__file__).with_name("data") / "ase_fire2_rosenbrock.txt"
    lines = [f"# ase {ase.__version__} FIRE2",
             f"dt {dt} fmax {FMAX} abc_maxstep {ABC_MAXSTEP}",
             "# use_abc steps force_calls, then after each step",
             "# x (3), v (3), dt, alpha, consecutive P > 0 steps"]
    for use_abc in (False, True):
        rows, calls = run(use_abc, dt)
        lines.append(f"{int(use_abc)} {len(rows)} {calls}")
        lines += [" ".join([*(repr(float(v)) for v in r[:-1]), str(r[-1])]) for r in rows]
    out.write_text("\n".join(lines) + "\n")
    lam = np.linalg.eigvalsh(hessian_at_minimum()).max()
    print(f"wrote {out.name}: lambda_max {lam:.6f}, dt {dt}, fmax {FMAX}")


if __name__ == "__main__":
    main()
