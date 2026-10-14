#!/usr/bin/env python3
"""Paired wall-time gate for the vecops multiversion clones.

Decision rule, fixed before any timing in this process is read:

The candidate is the simd auto path (RGMIN_SIMD unset). The baseline
is origin/main. Each size is 300, 30000, and 3000000. Dot and axpy are
separate families. A family has three paired geometric-mean ratios,
candidate over baseline, from five interleaved process runs (A B A B,
baseline then candidate). The interval is a BCa interval of that
geometric mean.

Holm is applied inside a family across its three sizes, separately for
the speedup p-values (ratio < 1) and the slowdown p-values (ratio > 1),
at alpha = 0.05. A size is faster when its Holm-adjusted speedup
p-value is below 0.05 and the one-sided BCa upper bound at that test's
Holm alpha lies below 1. A size is slower beyond its interval when its
Holm-adjusted slowdown p-value is below 0.05.

The atomistic comparison is one wall-time ratio, not one of the three
sizes. Its BCa interval is unadjusted. It is faster when that interval
lies entirely below 1.

The clones become default-on only if the atomistic wall interval lies
below 1, or n = 30000 is faster in the dot family or the axpy family,
and no size in either family is slower beyond its interval. Otherwise
simd stays a non-default feature.

Páll and Hess, Comput. Phys. Commun. 184, 2641 (2013),
doi:10.1016/j.cpc.2013.06.003.
Dröge et al., Softw. Pract. Exp. 53, 176 (2023), doi:10.1002/spe.3075.
Efron, JASA 82, 171 (1987), doi:10.1080/01621459.1987.10478410.
Holm, Scand. J. Stat. 6, 65 (1979).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import bootstrap

SIZES = (300, 30_000, 3_000_000)
KERNELS = ("dot", "axpy")
RUNS = 5
ALPHA = 0.05
RESAMPLES = 9999
SEED = 1


def geomean(sample: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(sample))))


def bca_bounds(sample: np.ndarray, alpha: float, rng: np.random.Generator):
    """Two-sided BCa interval at `alpha`, plus the one-sided upper bound."""
    sample = np.asarray(sample, dtype=float)
    result = bootstrap(
        (sample,),
        geomean,
        n_resamples=RESAMPLES,
        confidence_level=1.0 - alpha,
        method="BCa",
        random_state=np.random.RandomState(rng.integers(0, 2**31 - 1)),
    )
    theta = geomean(sample)
    boots = np.empty(RESAMPLES)
    n = sample.size
    for i in range(RESAMPLES):
        boots[i] = geomean(rng.choice(sample, size=n, replace=True))
    jack = np.empty(n)
    for i in range(n):
        jack[i] = geomean(np.delete(sample, i))
    jack_mean = float(jack.mean())
    centre = jack_mean - jack
    den = float(np.sum(centre**2))
    acc = float(np.sum(centre**3) / (6.0 * den**1.5)) if den > 0.0 else 0.0
    props = float(np.mean(boots < theta))
    props = min(max(props, 1.0 / (RESAMPLES + 1)), 1.0 - 1.0 / (RESAMPLES + 1))
    z0 = float(ndtri(props))

    def percentile(prob: float) -> float:
        z = float(ndtri(prob))
        denom = 1.0 - acc * (z0 + z)
        if denom == 0.0 or not math.isfinite(denom):
            adj = 0.5
        else:
            adj = float(ndtr(z0 + (z0 + z) / denom))
        adj = min(max(adj, 0.5 / RESAMPLES), 1.0 - 0.5 / RESAMPLES)
        return float(np.quantile(boots, adj))

    low = float(result.confidence_interval.low)
    high = float(result.confidence_interval.high)
    upper = percentile(1.0 - alpha)
    faster = (1.0 + np.count_nonzero(boots >= 1.0)) / (RESAMPLES + 1.0)
    slower = (1.0 + np.count_nonzero(boots <= 1.0)) / (RESAMPLES + 1.0)
    return {
        "geomean": theta,
        "low": low,
        "high": high,
        "upper": upper,
        "p_faster": float(faster),
        "p_slower": float(slower),
    }


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, pvalues[index] * (m - rank))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def holm_alpha(pvalues: list[float], index: int) -> float:
    """Step-down threshold that applies to `index` once the sorted tests are known."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    rank = order.index(index)
    m = len(pvalues)
    return ALPHA / (m - rank)


def run_once(binary: Path, kernel: str, n: int, iters: int, env_extra: dict) -> float:
    env = os.environ.copy()
    env.pop("RGMIN_SIMD", None)
    env.update(env_extra)
    proc = subprocess.run(
        [str(binary), kernel, str(n), str(iters)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    line = proc.stdout.strip().splitlines()[-1]
    return float(line)


def choose_iters(binary: Path, kernel: str, n: int) -> int:
    probe = run_once(binary, kernel, n, 4, {})
    per = max(probe / 4.0, 1e-9)
    target = 0.25
    return max(4, int(target / per))


def measure_family(main_bin: Path, cand_bin: Path, kernel: str, clone: str):
    rows = []
    env_extra = {} if clone == "auto" else {"RGMIN_SIMD": clone}
    for n in SIZES:
        iters = choose_iters(main_bin, kernel, n)
        base_times = []
        cand_times = []
        for _ in range(RUNS):
            base_times.append(run_once(main_bin, kernel, n, iters, {}))
            cand_times.append(run_once(cand_bin, kernel, n, iters, env_extra))
        ratios = np.asarray(
            [c / b for b, c in zip(base_times, cand_times)], dtype=float
        )
        rows.append(
            {
                "kernel": kernel,
                "clone": clone,
                "n": n,
                "iters": iters,
                "base": base_times,
                "cand": cand_times,
                "ratios": ratios,
            }
        )
    return rows


def decide_family(rows, rng: np.random.Generator):
    stats = [bca_bounds(row["ratios"], ALPHA, rng) for row in rows]
    p_fast = [item["p_faster"] for item in stats]
    p_slow = [item["p_slower"] for item in stats]
    adj_fast = holm(p_fast)
    adj_slow = holm(p_slow)
    out = []
    for index, (row, stat) in enumerate(zip(rows, stats)):
        level = holm_alpha(p_fast, index)
        holm_stat = bca_bounds(row["ratios"], level, rng)
        faster = adj_fast[index] < ALPHA and holm_stat["upper"] < 1.0
        slower = adj_slow[index] < ALPHA
        out.append(
            {
                **row,
                **stat,
                "holm_p_faster": adj_fast[index],
                "holm_p_slower": adj_slow[index],
                "holm_alpha": level,
                "holm_upper": holm_stat["upper"],
                "faster": faster,
                "slower": slower,
            }
        )
    return out


def write_csv(path: Path, rows, method: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write("# rggate-csv 1\n")
        writer = csv.writer(handle)
        writer.writerow(
            [
                "system",
                "method",
                "seed",
                "converged",
                "iters",
                "calls",
                "energy",
                "fmax",
                "wall_s",
                "dim",
            ]
        )
        for row in rows:
            for seed, (base, cand) in enumerate(zip(row["base"], row["cand"])):
                system = f"{row['kernel']}-{row['n']}"
                for name, seconds in (("main", base), (method, cand)):
                    writer.writerow(
                        [
                            system,
                            name,
                            seed,
                            1,
                            1,
                            f"{seconds:.9f}",
                            0.0,
                            0.0,
                            f"{seconds:.9f}",
                            row["n"],
                        ]
                    )


def fmt_times(values):
    return (
        f"median={statistics.median(values):.6g} "
        f"min={min(values):.6g} max={max(values):.6g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-bin", type=Path, required=True)
    parser.add_argument("--cand-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--atomistic", type=Path, default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    families = {}
    for kernel in KERNELS:
        measured = measure_family(args.main_bin, args.cand_bin, kernel, "auto")
        families[kernel] = decide_family(measured, rng)
        write_csv(args.out / f"{kernel}-wall.csv", families[kernel], "auto")
        for clone in ("scalar", "avx2", "avx512"):
            extra = measure_family(args.main_bin, args.cand_bin, kernel, clone)
            extra_decided = decide_family(extra, rng)
            write_csv(args.out / f"{kernel}-{clone}-wall.csv", extra_decided, clone)
            families[f"{kernel}-{clone}"] = extra_decided
    atomistic_below = False
    atomistic_note = "not run"
    if args.atomistic is not None and args.atomistic.is_file():
        atomistic_below, atomistic_note = atomistic_gate(args.atomistic, args.out, rng)
    slower = [
        item
        for kernel in KERNELS
        for item in families[kernel]
        if item["slower"]
    ]
    trigger = atomistic_below or any(
        item["faster"] and item["n"] == 30_000
        for kernel in KERNELS
        for item in families[kernel]
    )
    default_on = trigger and not slower
    lines = [
        f"verdict={'default-on' if default_on else 'opt-in'}",
        f"atomistic={atomistic_note}",
        f"trigger={trigger}",
        f"slower_sizes={len(slower)}",
    ]
    for key, rows in families.items():
        for item in rows:
            lines.append(
                " ".join(
                    [
                        f"{key}",
                        f"n={item['n']}",
                        f"geomean={item['geomean']:.6g}",
                        f"bca=[{item['low']:.6g},{item['high']:.6g}]",
                        f"holm_upper={item['holm_upper']:.6g}",
                        f"holm_p_faster={item['holm_p_faster']:.4g}",
                        f"holm_p_slower={item['holm_p_slower']:.4g}",
                        f"faster={item['faster']}",
                        f"slower={item['slower']}",
                        f"base[{fmt_times(item['base'])}]",
                        f"cand[{fmt_times(item['cand'])}]",
                    ]
                )
            )
    text = "\n".join(lines) + "\n"
    (args.out / "verdict.txt").write_text(text)
    sys.stdout.write(text)
    return 0


def atomistic_gate(csv_path: Path, out: Path, rng: np.random.Generator):
    """Wall ratios from an interleaved atomistic CSV with methods main and auto."""
    by_key = {}
    with csv_path.open() as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    for row in rows:
        key = (row["system"], row["seed"], row.get("rep", "0"))
        by_key.setdefault(key, {})[row["method"]] = float(row["wall_s"])
    ratios = []
    for pair in by_key.values():
        if "main" in pair and "auto" in pair and pair["main"] > 0.0:
            ratios.append(pair["auto"] / pair["main"])
    if len(ratios) < 2:
        return False, f"too few pairs ({len(ratios)})"
    stat = bca_bounds(np.asarray(ratios, dtype=float), ALPHA, rng)
    below = stat["high"] < 1.0
    note = (
        f"n={len(ratios)} geomean={stat['geomean']:.6g} "
        f"bca=[{stat['low']:.6g},{stat['high']:.6g}] below={below}"
    )
    (out / "atomistic-note.txt").write_text(note + "\n")
    return below, note


if __name__ == "__main__":
    raise SystemExit(main())
