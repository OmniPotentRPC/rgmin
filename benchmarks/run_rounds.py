#!/usr/bin/env python3
"""20 interleaved, order-flipped rounds of develop against the branch.

Each round is two processes. The order flips every round. Each process
runs 31 relaxations and the first is dropped. The round keeps the median
of the other 30. Samples in the asv JSON are those medians, in round
order, so a later bootstrap can pair them.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

SYSTEMS = ["lj13", "lj38", "lj75", "pt7", "cu", "al"]
METHODS = ["lbfgs", "fire2"]
ROUNDS = 20
REPEATS = 31


def parse_lines(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        conv, iters, calls, energy, fmax, wall = line.split()
        rows.append(
            {
                "converged": int(conv),
                "iters": int(iters),
                "calls": int(calls),
                "energy": float(energy),
                "fmax": float(fmax),
                "wall": float(wall),
            }
        )
    return rows


def run(driver: str, system: str, method: str, con: str) -> list[dict]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [driver, system, method, con, str(REPEATS)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode not in (0, 3):
        raise SystemExit(proc.stderr or proc.stdout or f"driver exit {proc.returncode}")
    rows = parse_lines(proc.stdout)
    if len(rows) != REPEATS or any(row["converged"] != 1 for row in rows):
        raise SystemExit(f"{system} {method} did not converge")
    return rows


def asv_doc(commit: str, wall: dict, calls: dict, energy: dict) -> dict:
    systems = SYSTEMS
    methods = METHODS
    names = [
        "benchmarks.Atomistic.time_relax",
        "benchmarks.Atomistic.track_calls",
        "benchmarks.Atomistic.track_energy",
    ]
    columns = [
        "result",
        "params",
        "version",
        "started_at",
        "duration",
        "stats_ci_99_a",
        "stats_ci_99_b",
        "stats_q_25",
        "stats_q_75",
        "stats_number",
        "stats_repeat",
        "samples",
        "profile",
    ]
    params = [systems, methods]
    results = {}
    for name in names:
        values = []
        samples = []
        for system in systems:
            for method in methods:
                if name.endswith("time_relax"):
                    sample = wall[(system, method)]
                    values.append(float(sum(sample) / len(sample)))
                    samples.append(sample)
                elif name.endswith("track_calls"):
                    values.append(calls[(system, method)])
                    samples.append([calls[(system, method)]])
                else:
                    values.append(energy[(system, method)])
                    samples.append([energy[(system, method)]])
        results[name] = [values, params, "1", None, None, None, None, None, None, None, None, samples]
    return {
        "commit_hash": commit,
        "env_name": "capi",
        "date": 0,
        "params": {
            "machine": "unspecified",
            "os": "Linux",
            "arch": "x86_64",
            "cpu": "unspecified",
            "num_cpu": "1",
            "ram": "unspecified",
        },
        "python": "3.12",
        "requirements": {},
        "env_vars": {},
        "result_columns": columns,
        "results": results,
        "durations": {},
        "version": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--develop-driver", required=True)
    parser.add_argument("--branch-driver", required=True)
    parser.add_argument("--develop-hash", required=True)
    parser.add_argument("--branch-hash", required=True)
    parser.add_argument("--con", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wall = {"develop": {}, "branch": {}}
    calls = {"develop": {}, "branch": {}}
    energy = {"develop": {}, "branch": {}}
    drivers = {"develop": args.develop_driver, "branch": args.branch_driver}
    for system in SYSTEMS:
        for method in METHODS:
            series = {name: [] for name in drivers}
            seen = {}
            for round_id in range(ROUNDS):
                order = ["develop", "branch"] if round_id % 2 == 0 else ["branch", "develop"]
                med = {}
                for name in order:
                    rows = run(drivers[name], system, method, args.con)
                    med[name] = statistics.median(row["wall"] for row in rows[1:])
                    seen[name] = rows[1]
                    print(
                        f"{system} {method} round {round_id} {name} "
                        f"wall {med[name]:.6f} calls {rows[1]['calls']}",
                        flush=True,
                    )
                for name in drivers:
                    series[name].append(med[name])
            for name in drivers:
                wall[name][(system, method)] = series[name]
                calls[name][(system, method)] = seen[name]["calls"]
                energy[name][(system, method)] = seen[name]["energy"]
    hashes = {"develop": args.develop_hash, "branch": args.branch_hash}
    for name in drivers:
        doc = asv_doc(hashes[name], wall[name], calls[name], energy[name])
        (out / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    main()
