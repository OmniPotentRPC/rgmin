#!/usr/bin/env python3
"""Serial ABBA cost runs of the two CG arms. One core, five wall repeats."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

GTOL = 1e-5
MAXITER = 8000
REPEATS = 5
SEEDS = range(2000, 2012)

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def abba(n_each):
    seq = []
    while seq.count("A") < n_each or seq.count("B") < n_each:
        for arm in ("A", "B", "B", "A"):
            if seq.count(arm) < n_each:
                seq.append(arm)
            if seq.count("A") >= n_each and seq.count("B") >= n_each:
                return seq
    return seq


def jobs(maxiter):
    variants = ("fr", "pr", "hz")
    out = []
    out.append(
        dict(
            variant="fr",
            case="rosenbrock_fr_20",
            kind="rosenbrock",
            n=2,
            seed=0,
            maxiter=20,
            maxmove="-1",
            gtol=0.0,
        )
    )
    for variant in variants:
        out.append(
            dict(
                variant=variant,
                case="rosenbrock2d",
                kind="rosenbrock",
                n=2,
                seed=0,
                maxiter=maxiter,
                maxmove="-1",
            )
        )
        for n in (2, 10, 100, 1000):
            out.append(
                dict(
                    variant=variant,
                    case="rosenbrock",
                    kind="rosenbrock",
                    n=n,
                    seed=0,
                    maxiter=maxiter,
                    maxmove="-1",
                )
            )
        for n, case in ((38, "lj38"), (75, "lj75")):
            for seed in SEEDS:
                out.append(
                    dict(
                        variant=variant,
                        case=case,
                        kind="lj",
                        n=n,
                        seed=seed,
                        maxiter=maxiter,
                        maxmove="default",
                    )
                )
        for n in (13, 38, 75, 150):
            for seed in (2000, 2001, 2002):
                out.append(
                    dict(
                        variant=variant,
                        case="ladder",
                        kind="lj",
                        n=n,
                        seed=seed,
                        maxiter=maxiter,
                        maxmove="default",
                    )
                )
    return out


def machine_id():
    raw = Path("/etc/machine-id").read_bytes().strip()
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def one(binary, job, gtol):
    cmd = [
        "taskset",
        "-c",
        "0",
        binary,
        "run",
        "--kind",
        job["kind"],
        "--variant",
        job["variant"],
        "--case",
        job["case"],
        "--n",
        str(job["n"]),
        "--seed",
        str(job["seed"]),
        "--maxiter",
        str(job["maxiter"]),
        "--gtol",
        format(job.get("gtol", gtol), ".16g"),
        "--maxmove",
        job["maxmove"],
    ]
    env = os.environ.copy()
    env.update(THREAD_ENV)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    line = proc.stdout.strip().splitlines()
    payload = None
    if line:
        try:
            payload = json.loads(line[-1])
        except json.JSONDecodeError:
            payload = None
    return proc.returncode, payload, proc.stderr[-400:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-a", required=True)
    parser.add_argument("--arm-b", required=True)
    parser.add_argument("--out", default="bench/cg-hardening-cost")
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--gtol", type=float, default=GTOL)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha_a = file_sha(args.arm_a)
    sha_b = file_sha(args.arm_b)
    if sha_a == sha_b:
        print("binaries are identical", file=sys.stderr)
        return 1
    selected = jobs(args.maxiter)
    if args.case:
        selected = [job for job in selected if job["case"] in args.case]
    if args.variant:
        selected = [job for job in selected if job["variant"] in args.variant]
    if args.limit:
        selected = selected[: args.limit]
    sessions = out / "sessions.jsonl"
    mid = machine_id()
    header = {
        "type": "header",
        "machine_id_sha256": mid,
        "binary_sha256": {"A": sha_a, "B": sha_b},
        "gtol": args.gtol,
        "twenty_step_gtol": 0.0,
        "maxiter": args.maxiter,
        "repeats": args.repeats,
        "order": "ABBA",
        "core": 0,
        "wolfe": [1e-4, 0.1, 20],
        "istep": 1.0,
        "rgpot": "3.7.0",
        "linkcell": "0.6.0",
        "lj": "rgpot_lj_curvature_pairs on linkcell::pairs_open",
        "n_jobs": len(selected),
    }
    with sessions.open("w") as handle:
        handle.write(json.dumps(header) + "\n")
        binaries = {"A": args.arm_a, "B": args.arm_b}
        done = 0
        for job in selected:
            for index, arm in enumerate(abba(args.repeats)):
                code, payload, err = one(binaries[arm], job, args.gtol)
                row = {
                    "type": "run",
                    "arm": arm,
                    "repeat_index": index,
                    "exit": code,
                    "stderr": err if code != 0 else "",
                }
                row["gtol"] = job.get("gtol", args.gtol)
                if payload:
                    row.update(payload)
                else:
                    row.update(job)
                    row["ok"] = False
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            done += 1
            if done % 10 == 0 or done == len(selected):
                print(f"jobs {done}/{len(selected)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
