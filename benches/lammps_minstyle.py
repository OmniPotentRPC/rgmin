# Copyright (c) rgmin developers
"""``min_style rgmin`` against LAMMPS's own minimizers.

Every row is one ``minimize etol ftol maxiter maxeval`` of the same
seeded system with the same stopping rule; only ``min_style`` (and, for
rgmin, ``min_modify method``) changes. Every style runs at its defaults:
LAMMPS ``cg`` (Polak-Ribiere, quadratic line search), ``sd``, ``fire``
(FIRE 2.0) and ``quickmin``, and rgmin ``lbfgs``, ``fire`` and
``fire2`` through the plugin.

The systems are the ones the plugin tests relax (``lammps/tests/systems.py``):

- lj177: a rattled fcc LJ cluster of 177 atoms in vacuum, lj/cut 2.5.
- cu-vac: fcc Cu, 4x4x4 cells less one atom (255 atoms), Mishin EAM.
- cu-sia-vac: fcc Cu, 5x5x5 cells with a <100> split dumbbell and a
  vacancy (500 atoms), Mishin EAM.
- cu32k-vac: fcc Cu, 20x20x20 cells less one atom (31,999 atoms),
  every atom rattled by up to 0.05 A, Mishin EAM.

Force calls are the "force evaluations" LAMMPS reports; wall time is
its "Loop time" of the minimize command. Both are read from the log.

    python benches/lammps_minstyle.py --lmp <lmp> --plugin <rgminplugin.so> \\
        [--repeats 5] [--ranks 1] [--case cu-vac]

``LAMMPS_POTENTIALS`` must name the LAMMPS potentials directory. The
table is benches/lammps_minstyle/results.md; every repeat is logged to
benches/lammps_minstyle/raw/, which is not committed.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "lammps_minstyle"
sys.path.insert(0, str(ROOT / "lammps" / "tests"))

import systems  # noqa: E402

ETOL = 0.0
FTOL = 1e-6
MAXITER = 100000
MAXEVAL = 1000000
ARMS = (
    ("lammps", "cg", ""),
    ("lammps", "sd", ""),
    ("lammps", "fire", ""),
    ("lammps", "quickmin", ""),
    ("rgmin", "lbfgs", "min_modify method lbfgs"),
    ("rgmin", "fire", "min_modify method fire"),
    ("rgmin", "fire2", "min_modify method fire2"),
)
CASES = ("lj177", "cu-vac", "cu-sia-vac", "cu32k-vac")


def _git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment(lmp: str, log: str) -> list[str]:
    version = next((ln for ln in log.splitlines() if ln.startswith("LAMMPS (")), "LAMMPS ?")
    rgmin = next((ln.split(":")[0] for ln in log.splitlines() if ln.startswith("rgmin ")), "rgmin ?")
    mpi = subprocess.run(["mpirun", "--version"], capture_output=True, text=True, check=False)
    return [
        f"{rgmin}, git {_git_sha()}",
        f"{version}, MANYBODY and PLUGIN packages, {Path(lmp).name}",
        (mpi.stdout.splitlines() or ["mpirun ?"])[0],
        f"{platform.platform()}, {_cpu()}, {os.cpu_count()} logical CPUs",
    ]


def run_once(args, case: str, kind: str, style: str, modify: str, repeat: int) -> systems.Result:
    text = systems.deck(
        case,
        "rgmin" if kind == "rgmin" else style,
        modify,
        etol=ETOL,
        ftol=args.ftol,
        maxiter=MAXITER,
        maxeval=MAXEVAL,
        plugin=args.plugin if kind == "rgmin" else None,
    )
    work = OUT / "raw" / f"{case}-{kind}-{style}-np{args.ranks}-{repeat}"
    return systems.run(args.lmp, text, work, nprocs=args.ranks, timeout=24 * 3600)


def write_table(env: list[str], rows: list[dict], args) -> str:
    lines = [
        "# min_style rgmin against LAMMPS minimizers",
        "",
        *[f"- {line}" for line in env],
        f"- etol {ETOL}, ftol {args.ftol} (two-norm of the global force), "
        f"maxiter {MAXITER}, maxeval {MAXEVAL},",
        f"  {args.repeats} repeats per row on {args.ranks} MPI rank(s), every style at its defaults.",
        "",
        "Iterations and force calls are identical on every repeat.",
        "Wall time is the LAMMPS loop time of the minimize command, in seconds.",
        "",
        "| case | ranks | min_style | method | iterations | force calls | energy | fnorm "
        "| stop | median s | min s |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {args.ranks} | {row['style']} | {row['method']} | "
            f"{row['iterations']} | {row['evaluations']} | {row['pe']:.10f} | "
            f"{row['fnorm']:.3e} | {row['stop']} | {row['median']:.4g} | {row['min']:.4g} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lmp", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--ftol", type=float, default=FTOL)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--output", default=str(OUT / "results.md"))
    args = parser.parse_args()
    args.lmp = str(Path(args.lmp).resolve())
    args.plugin = str(Path(args.plugin).resolve())

    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tsv = (raw / f"runs-np{args.ranks}.tsv").open("w")
    columns = ("case", "style", "method", "repeat", "iterations", "evaluations", "pe", "fnorm", "stop", "loop_time")
    tsv.write("\t".join(columns) + "\n")

    rows, env = [], None
    for case in args.case or list(CASES):
        for kind, style, modify in ARMS:
            results = [run_once(args, case, kind, style, modify, r) for r in range(args.repeats)]
            for r, res in enumerate(results):
                if res.returncode != 0:
                    raise SystemExit(f"{case} {kind} {style}: {res.error or res.log[-2000:]}")
                tsv.write(
                    "\t".join(
                        str(v)
                        for v in (
                            case,
                            "rgmin" if kind == "rgmin" else style,
                            style if kind == "rgmin" else "",
                            r,
                            res.iterations,
                            res.evaluations,
                            repr(res.pe),
                            repr(res.fnorm),
                            res.stop,
                            repr(res.loop_time),
                        )
                    )
                    + "\n"
                )
                tsv.flush()
            first = results[0]
            if any(
                (r.iterations, r.evaluations) != (first.iterations, first.evaluations) for r in results
            ):
                raise SystemExit(f"{case} {kind} {style}: counts differ between repeats")
            if env is None and kind == "rgmin":
                env = environment(args.lmp, first.log)
            times = [r.loop_time for r in results]
            row = {
                "case": case,
                "style": "rgmin" if kind == "rgmin" else style,
                "method": style if kind == "rgmin" else "",
                "iterations": first.iterations,
                "evaluations": first.evaluations,
                "pe": first.pe,
                "fnorm": first.fnorm,
                "stop": first.stop,
                "median": statistics.median(times),
                "min": min(times),
            }
            rows.append(row)
            print(
                f"{case:11} {row['style']:9} {row['method']:6} {row['iterations']:7d} "
                f"{row['evaluations']:7d} {row['pe']:20.10f} {row['fnorm']:10.3e} "
                f"{row['median']:9.4g} {row['stop']}",
                flush=True,
            )
    tsv.close()
    Path(args.output).write_text(write_table(env or [], rows, args))


if __name__ == "__main__":
    main()
