#!/usr/bin/env python3
"""Booktabs table from two asv result files.

The ratio is the first file over the second (develop over the branch).
Wall ratios are the 20 paired round medians. ``--aa-first`` and
``--aa-second`` are the same commit timed against itself, in the same
round order. The bootstrap of each mean uses numpy Generator seed 5,
10000 draws, percentiles 2.5 and 97.5. Call counts and energies are
the recorded track values. The table has no machine name and no path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 5
DRAWS = 10000


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    rows = {}
    results = data["results"]
    for name, row in results.items():
        payload = {col: row[i] if i < len(row) else None for i, col in enumerate(data["result_columns"])}
        rows[name] = payload
    return rows


def pairs(rows: dict, key: str):
    params = rows[key]["params"]
    systems, methods = params
    result = rows[key]["result"]
    samples = rows[key].get("samples")
    out = []
    index = 0
    for system in systems:
        for method in methods:
            sample = None if samples is None else samples[index]
            out.append((system, method, result[index], sample))
            index += 1
    return out


def bootstrap(values: list[float]) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    sample = np.asarray(values, dtype=float)
    idx = rng.integers(0, sample.size, size=(DRAWS, sample.size))
    means = sample[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(sample.mean()), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("develop")
    parser.add_argument("branch")
    parser.add_argument("--aa-first", required=True)
    parser.add_argument("--aa-second", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    left = load(Path(args.develop))
    right = load(Path(args.branch))
    aa_left = load(Path(args.aa_first))
    aa_right = load(Path(args.aa_second))
    wall_l = pairs(left, "benchmarks.Atomistic.time_relax")
    wall_r = {(s, m): sample for s, m, _, sample in pairs(right, "benchmarks.Atomistic.time_relax")}
    aa_l = {(s, m): sample for s, m, _, sample in pairs(aa_left, "benchmarks.Atomistic.time_relax")}
    aa_r = {(s, m): sample for s, m, _, sample in pairs(aa_right, "benchmarks.Atomistic.time_relax")}
    calls_l = {(s, m): v for s, m, v, _ in pairs(left, "benchmarks.Atomistic.track_calls")}
    calls_r = {(s, m): v for s, m, v, _ in pairs(right, "benchmarks.Atomistic.track_calls")}
    energy_l = {(s, m): v for s, m, v, _ in pairs(left, "benchmarks.Atomistic.track_energy")}
    energy_r = {(s, m): v for s, m, v, _ in pairs(right, "benchmarks.Atomistic.track_energy")}
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"System & Method & Wall ratio & 95\% CI & A/A 95\% CI & Calls develop & Calls branch & Energy develop & Energy branch \\",
        r"\midrule",
    ]
    for system, method, _, sample_l in wall_l:
        ratios = [a / b for a, b in zip(sample_l, wall_r[(system, method)])]
        mean, low, high = bootstrap(ratios)
        aa = [a / b for a, b in zip(aa_l[(system, method)], aa_r[(system, method)])]
        _, aa_low, aa_high = bootstrap(aa)
        lines.append(
            f"{system} & {method} & {mean:.6f} & [{low:.6f}, {high:.6f}] "
            f"& [{aa_low:.6f}, {aa_high:.6f}] & {int(calls_l[(system, method)])} "
            f"& {int(calls_r[(system, method)])} & {energy_l[(system, method)]:.10f} "
            f"& {energy_r[(system, method)]:.10f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
