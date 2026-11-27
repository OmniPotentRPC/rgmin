"""Summarise the r2 mask-precond cost rows.

A miss is scored as budget + 1. Convergence is a two-sided Fisher exact
test on the 2x2 table. Evaluations are a two-sided Mann-Whitney U test.
The median eval ratio is the median of the paired first-pass B/A
evaluation ratios. Its 95% interval is the 2.5 and 97.5 percentiles of
10000 resamples from numpy's default generator seeded with 20261123.

IDENTICAL means the paired repeats match on evaluation count, iteration
count, and energy bits. PASS means the candidate lost no convergence and
the upper end of that interval is at most 1.10. Anything else is FLAG.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

SYSTEMS = ("lj38", "lj75")
METHODS = ("fire2-exp", "fire2-exp-kick", "eon", "fire2")
BOOT_SAMPLES = 10000
BOOT_SEED = 20261123
RATIO_LIMIT = 1.10
WALL_REPEATS = 10
FIELDS = ("evals", "iterations", "energy_bits")


def scored(row: dict, budget: int) -> int:
    if row.get("converged") and row.get("evals") is not None:
        return int(row["evals"])
    return budget + 1


def median_ratio_ci(ratios: np.ndarray) -> tuple[float, float, float]:
    point = float(np.median(ratios))
    rng = np.random.default_rng(BOOT_SEED)
    n = ratios.size
    draw = rng.integers(0, n, size=(BOOT_SAMPLES, n))
    samples = np.median(ratios[draw], axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return point, float(low), float(high)


def finite_or_none(value: float):
    number = float(value)
    if math.isfinite(number):
        return number
    return None


def fisher_mwu(a_conv: int, a_n: int, b_conv: int, b_n: int, a_evals, b_evals):
    table = [[a_conv, a_n - a_conv], [b_conv, b_n - b_conv]]
    odds, fisher_p = fisher_exact(table, alternative="two-sided")
    mwu = mannwhitneyu(b_evals, a_evals, alternative="two-sided", method="auto")
    return {
        "table_conv_miss": table,
        "fisher_odds_ratio": finite_or_none(odds),
        "fisher_p": finite_or_none(fisher_p),
        "mwu_u": finite_or_none(mwu.statistic),
        "mwu_p": finite_or_none(mwu.pvalue),
    }


def verdict(identical: bool, n_conv_b: int, n_conv_a: int, ci_high: float) -> str:
    if identical:
        return "IDENTICAL"
    if n_conv_b >= n_conv_a and math.isfinite(ci_high) and ci_high <= RATIO_LIMIT:
        return "PASS"
    return "FLAG"


def cell_rows(rows: list[dict], energy: str, method: str, system: str) -> list[dict]:
    out = []
    for row in rows:
        if row.get("energy_arm") != energy:
            continue
        if row.get("method") != method or row.get("system") != system:
            continue
        out.append(row)
    return out


def by_arm(rows: list[dict]) -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {"A": defaultdict(list), "B": defaultdict(list)}
    for row in rows:
        if row.get("error") or row.get("evals") is None:
            continue
        arm = row.get("arm")
        if arm not in out:
            continue
        out[arm][int(row["seed"])].append(row)
    for arm in out.values():
        for items in arm.values():
            items.sort(key=lambda row: int(row["pass"]))
    return out


def bit_same(a: dict, b: dict) -> bool:
    return all(a.get(field) == b.get(field) for field in FIELDS)


def aa_wall(first: list[dict], second: list[dict], budget: int) -> dict:
    a = np.array([float(r["wall_s"]) for r in first], dtype=float)
    b = np.array([float(r["wall_s"]) for r in second], dtype=float)
    med_a = float(np.median(a))
    med_b = float(np.median(b))
    lo, hi = (med_a, med_b) if med_a <= med_b else (med_b, med_a)
    centre = 0.5 * (med_a + med_b)
    width = 0.0 if centre == 0.0 else (hi - lo) / centre
    mwu = mannwhitneyu(b, a, alternative="two-sided", method="auto")
    eval_ratio = []
    for left, right in zip(first, second):
        left_n = scored(left, budget)
        right_n = scored(right, budget)
        if left_n > 0:
            eval_ratio.append(right_n / left_n)
    return {
        "n": len(first),
        "first_median_s": med_a,
        "second_median_s": med_b,
        "band_s": [lo, hi],
        "relative_width": width,
        "mwu_p": finite_or_none(mwu.pvalue),
        "eval_median_ratio": None if not eval_ratio else float(np.median(eval_ratio)),
    }


def cost_block(rows: list[dict], energy: str, method: str, system: str, budget: int) -> dict:
    selected = cell_rows(rows, energy, method, system)
    errors = [row for row in selected if row.get("error") or row.get("evals") is None]
    grouped = by_arm(selected)
    seeds = sorted(set(grouped["A"]) & set(grouped["B"]))
    block = {
        "energy": energy,
        "method": method,
        "system": system,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_errors": len(errors),
    }
    if errors or len(seeds) != 12:
        block["verdict"] = "FLAG"
        block["reason"] = "incomplete"
        return block
    short = []
    for seed in seeds:
        for arm in ("A", "B"):
            if len(grouped[arm][seed]) != WALL_REPEATS:
                short.append({"arm": arm, "seed": seed, "n": len(grouped[arm][seed])})
    if short:
        block["verdict"] = "FLAG"
        block["reason"] = "repeats"
        block["short_repeats"] = short
        return block
    a_rows = [grouped["A"][seed][0] for seed in seeds]
    b_rows = [grouped["B"][seed][0] for seed in seeds]
    a_evals = np.array([scored(r, budget) for r in a_rows], dtype=float)
    b_evals = np.array([scored(r, budget) for r in b_rows], dtype=float)
    n_conv_a = sum(bool(r.get("converged")) for r in a_rows)
    n_conv_b = sum(bool(r.get("converged")) for r in b_rows)
    identical = True
    first_diff = None
    for seed in seeds:
        for a in grouped["A"][seed]:
            for b in grouped["B"][seed]:
                if not bit_same(a, b):
                    identical = False
                    if first_diff is None:
                        field = next(name for name in FIELDS if a.get(name) != b.get(name))
                        first_diff = {
                            "seed": seed,
                            "field": field,
                            "a_pass": a.get("pass"),
                            "b_pass": b.get("pass"),
                        }
    block.update(
        {
            "n_conv_a": n_conv_a,
            "n_conv_b": n_conv_b,
            "median_evals_a": float(np.median(a_evals)),
            "median_evals_b": float(np.median(b_evals)),
            "median_iterations_a": float(np.median([int(r["iterations"]) for r in a_rows])),
            "median_iterations_b": float(np.median([int(r["iterations"]) for r in b_rows])),
            "wall_s_median_a": float(
                np.median([float(r["wall_s"]) for seed in seeds for r in grouped["A"][seed]])
            ),
            "wall_s_median_b": float(
                np.median([float(r["wall_s"]) for seed in seeds for r in grouped["B"][seed]])
            ),
            "wall_repeats": WALL_REPEATS,
            "bit_identical": identical,
            "first_differing_seed": first_diff,
            "same_evals": int(sum(a["evals"] == b["evals"] for a, b in zip(a_rows, b_rows))),
            "same_iterations": int(
                sum(a["iterations"] == b["iterations"] for a, b in zip(a_rows, b_rows))
            ),
            "same_energy_bits": int(
                sum(a["energy_bits"] == b["energy_bits"] for a, b in zip(a_rows, b_rows))
            ),
        }
    )
    if np.any(a_evals <= 0):
        block["verdict"] = "FLAG"
        block["reason"] = "incomplete"
        return block
    ratios = b_evals / a_evals
    point, low, high = median_ratio_ci(ratios)
    block.update(fisher_mwu(n_conv_a, len(seeds), n_conv_b, len(seeds), a_evals, b_evals))
    block["median_eval_ratio_b_over_a"] = point
    block["ci95"] = [low, high]
    block["aa"] = aa_wall(
        [grouped["A"][seed][0] for seed in seeds],
        [grouped["A"][seed][1] for seed in seeds],
        budget,
    )
    block["verdict"] = verdict(identical, n_conv_b, n_conv_a, high)
    return block


def summarize(rows: list[dict], budget: int, protocol: dict) -> dict:
    energies = protocol.get("energies") or sorted({row.get("energy_arm") for row in rows})
    blocks = [
        cost_block(rows, energy, method, system, budget)
        for energy in energies
        for method in METHODS
        for system in SYSTEMS
    ]
    return {
        "protocol": protocol,
        "budget": budget,
        "miss_score": budget + 1,
        "bootstrap_samples": BOOT_SAMPLES,
        "bootstrap_seed": BOOT_SEED,
        "ratio_limit": RATIO_LIMIT,
        "verdicts": blocks,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    rows = json.loads(open(args.rows, encoding="utf-8").read())
    protocol = json.loads(open(args.protocol, encoding="utf-8").read())
    summary = summarize(rows, args.budget, protocol)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("cell arm base_evals cand_evals ratio ci_low ci_high verdict")
    for block in summary["verdicts"]:
        cell = f"{block['energy']}/{block['system']}"
        ratio = block.get("median_eval_ratio_b_over_a")
        ci = block.get("ci95") or [None, None]
        print(
            f"{cell} {block['method']} {block.get('median_evals_a')} "
            f"{block.get('median_evals_b')} {ratio} {ci[0]} {ci[1]} {block['verdict']}"
        )


if __name__ == "__main__":
    main()
