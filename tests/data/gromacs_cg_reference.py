#!/usr/bin/env python3
"""Regenerate the GROMACS 2023.3 conjugate-gradient reference transcripts."""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Case:
    name: str
    precision: str
    emtol: str
    nsteps: int


CASES = (
    Case("f64-nsteps-12-emtol-1e-8", "double", "1e-8", 12),
    Case("f32-nsteps-12-emtol-1e-8", "single", "1e-8", 12),
    Case("f64-emtol-1e-2", "double", "1e-2", 100),
    Case("f64-emtol-1e-3", "double", "1e-3", 100),
    Case("f64-emtol-1e-6", "double", "1e-6", 100),
)


def run(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return f"$ {shlex.join(command)}\n{completed.stdout}"


def version(executable: Path) -> str:
    transcript = run([str(executable), "--version"], Path.cwd())
    if not re.search(r"GROMACS version:\s+2023\.3(?:\s|$)", transcript):
        raise RuntimeError(f"{executable} is not GROMACS 2023.3")
    return transcript


def render_mdp(source: str, case: Case) -> str:
    replacements = {
        "integrator": "cg",
        "nsteps": str(case.nsteps),
        "emtol": case.emtol,
        "emstep": "0.02",
    }
    result: list[str] = []
    seen: set[str] = set()
    for line in source.splitlines():
        match = re.match(r"\s*([A-Za-z0-9_-]+)\s*=", line)
        key = match.group(1).lower() if match else ""
        if key in replacements:
            result.append(f"{key} = {replacements[key]}")
            seen.add(key)
        else:
            result.append(line)
    if seen != replacements.keys():
        missing = sorted(replacements.keys() - seen)
        raise RuntimeError(f"base MDP is missing keys: {', '.join(missing)}")
    result.append("nstcgsteep = 1000")
    return "\n".join(result) + "\n"


def required(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not find {label} in GROMACS output")
    return match.group(1)


def summarize(case: Case, mdrun: str, mdlog: str) -> tuple[str, ...]:
    energy = required(r"^Potential Energy\s*=\s*(\S+)", mdrun, "potential energy")
    fmax = required(r"^Maximum force\s*=\s*(\S+)", mdrun, "maximum force")
    atom = required(r"^Maximum force\s*=\s*\S+ on atom (\d+)", mdrun, "force atom")
    fnorm = required(r"^Norm of force\s*=\s*(\S+)", mdrun, "force norm")
    evaluations = required(
        r"^Performed (\d+) energy evaluations in total\.", mdlog, "energy evaluations"
    )
    printed_steps = [int(value) for value in re.findall(r"^Step\s+(\d+),", mdrun, re.MULTILINE)]
    if not printed_steps:
        raise RuntimeError("could not find printed minimization steps")
    steps = max(printed_steps)
    stop = "tolerance"
    if "reached the maximum number of steps" in mdrun:
        stop = "budget"
        steps = case.nsteps
    elif "machine precision" in mdrun:
        stop = "machine-precision"
        steps += 1
    return (
        case.name,
        case.precision,
        case.emtol,
        str(case.nsteps),
        stop,
        str(steps),
        evaluations,
        energy,
        fmax,
        atom,
        fnorm,
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmx-double", type=Path, required=True)
    parser.add_argument("--gmx-single", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=here / "gromacs_cg_reference"
    )
    args = parser.parse_args()

    executables = {
        "double": args.gmx_double.resolve(),
        "single": args.gmx_single.resolve(),
    }
    versions = {precision: version(path) for precision, path in executables.items()}
    if not re.search(r"^Precision:\s+double$", versions["double"], re.MULTILINE):
        raise RuntimeError(f"{executables['double']} is not a double-precision build")
    if not re.search(r"^Precision:\s+mixed$", versions["single"], re.MULTILINE):
        raise RuntimeError(f"{executables['single']} is not a mixed-precision build")

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "gmx_d-version.txt").write_text(versions["double"], encoding="utf-8")
    (output / "gmx-version.txt").write_text(versions["single"], encoding="utf-8")

    base_mdp = (here / "gromacs_steep_argon.mdp").read_text(encoding="ascii")
    topology = (here / "gromacs_steep_argon.top").read_text(encoding="ascii")
    rows: list[tuple[str, ...]] = []
    for case in CASES:
        work = output / case.name
        work.mkdir()
        (work / "input.mdp").write_text(render_mdp(base_mdp, case), encoding="ascii")
        (work / "start.g96").write_text(START, encoding="ascii")
        (work / "topol.top").write_text(topology, encoding="ascii")
        executable = executables[case.precision]
        grompp = run(
            [
                str(executable),
                "-quiet",
                "grompp",
                "-f",
                "input.mdp",
                "-p",
                "topol.top",
                "-c",
                "start.g96",
                "-o",
                "em.tpr",
                "-maxwarn",
                "1",
            ],
            work,
        )
        (work / "grompp.txt").write_text(grompp, encoding="utf-8")
        mdrun = run(
            [
                str(executable),
                "-quiet",
                "mdrun",
                "-v",
                "-s",
                "em.tpr",
                "-deffnm",
                "em",
                "-nt",
                "1",
                "-ntmpi",
                "1",
                "-c",
                "final.g96",
            ],
            work,
        )
        (work / "mdrun.txt").write_text(mdrun, encoding="utf-8")
        mdlog = (work / "em.log").read_text(encoding="utf-8")
        rows.append(summarize(case, mdrun, mdlog))
        for generated in ("em.edr", "em.tpr", "em.trr"):
            (work / generated).unlink(missing_ok=True)

    header = (
        "case",
        "precision",
        "emtol",
        "nsteps_limit",
        "stop",
        "steps",
        "force_evals",
        "final_epot",
        "final_fmax",
        "fmax_atom",
        "final_fnorm",
    )
    summary = "\n".join("\t".join(row) for row in (header, *rows)) + "\n"
    (output / "summary.tsv").write_text(summary, encoding="ascii")
    print(summary, end="")


if __name__ == "__main__":
    main()
