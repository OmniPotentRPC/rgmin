"""Summarise mask-precond cost rows.

A miss is scored as budget + 1. Convergence is a two-sided Fisher exact
test on the 2x2 table. Evaluations are a two-sided Mann-Whitney U test.
The median eval ratio is the median of the paired B/A evaluation ratios.
Its 95% interval is the 2.5 and 97.5 percentiles of 10000 resamples from
numpy's default generator seeded with 20261123.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

SYSTEMS = ("lj38", "lj75")
COST_METHODS = ("fire2-exp", "eon")
IDENTITY_METHOD = "fire2"
BOOT_SAMPLES = 10000
BOOT_SEED = 20261123
RATIO_LIMIT = 1.10


def scored(row: dict, budget: int) -> int:
    if row.get("converged") and row.get("evals") is not None:
        return int(row["evals"])
    return budget + 1


def first_of(rows: list[dict], arm: str) -> dict[tuple, dict]:
    out = {}
    for row in rows:
        if row.get("arm") != arm or row.get("error"):
            continue
        key = (row["method"], row["system"], int(row["seed"]))
        slot = int(row["pass"])
        prev = out.get(key)
        if prev is None or slot < int(prev["pass"]):
            out[key] = row
    return out


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


def verdict(n_conv_b: int, n_conv_a: int, ci_high: float) -> str:
    if n_conv_b >= n_conv_a and math.isfinite(ci_high) and ci_high <= RATIO_LIMIT:
        return "PASS"
    return "FLAG"


def paired(rows: list[dict], method: str, system: str, budget: int):
    by_arm = {"A": {}, "B": {}}
    for row in rows:
        if row.get("method") != method or row.get("system") != system:
            continue
        if row.get("error") or row.get("evals") is None:
            continue
        arm = row["arm"]
        seed = int(row["seed"])
        slot = int(row["pass"])
        prev = by_arm[arm].get(seed)
        if prev is None or slot < int(prev["pass"]):
            by_arm[arm][seed] = row
    seeds = sorted(set(by_arm["A"]) & set(by_arm["B"]))
    a_rows = [by_arm["A"][s] for s in seeds]
    b_rows = [by_arm["B"][s] for s in seeds]
    a_evals = np.array([scored(r, budget) for r in a_rows], dtype=float)
    b_evals = np.array([scored(r, budget) for r in b_rows], dtype=float)
    return seeds, a_rows, b_rows, a_evals, b_evals


def cost_block(rows: list[dict], method: str, system: str, budget: int) -> dict:
    seeds, a_rows, b_rows, a_evals, b_evals = paired(rows, method, system, budget)
    n_conv_a = sum(bool(r.get("converged")) for r in a_rows)
    n_conv_b = sum(bool(r.get("converged")) for r in b_rows)
    block = {
        "method": method,
        "system": system,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_conv_a": n_conv_a,
        "n_conv_b": n_conv_b,
        "median_evals_a": None if a_evals.size == 0 else float(np.median(a_evals)),
        "median_evals_b": None if b_evals.size == 0 else float(np.median(b_evals)),
    }
    if len(seeds) != 12 or np.any(a_evals <= 0):
        block["verdict"] = "FLAG"
        block["reason"] = "incomplete"
        return block
    ratios = b_evals / a_evals
    point, low, high = median_ratio_ci(ratios)
    tests = fisher_mwu(n_conv_a, len(seeds), n_conv_b, len(seeds), a_evals, b_evals)
    block.update(tests)
    block["same_evals"] = int(np.sum(a_evals == b_evals))
    block["same_energy_bits"] = int(sum(a["energy_bits"] == b["energy_bits"] for a, b in zip(a_rows, b_rows)))
    block["same_coord"] = int(sum(a["coord_sha256"] == b["coord_sha256"] for a, b in zip(a_rows, b_rows)))
    block["median_eval_ratio_b_over_a"] = point
    block["ci95"] = [low, high]
    block["verdict"] = verdict(n_conv_b, n_conv_a, high)
    return block


def differ_fields(a: dict, b: dict) -> str | None:
    for field in ("evals", "energy_bits", "coord_sha256", "traj_sha256"):
        if a.get(field) != b.get(field):
            return field
    return None


def bit_identity(rows: list[dict]) -> dict:
    grouped = defaultdict(lambda: {"A": [], "B": []})
    for row in rows:
        if row.get("method") != IDENTITY_METHOD:
            continue
        key = (row.get("system"), int(row.get("seed", -1)))
        arm = row.get("arm")
        if arm in ("A", "B"):
            grouped[key][arm].append(row)
    first = None
    checked = 0
    for system in SYSTEMS:
        for seed in range(2000, 2012):
            arms = grouped.get((system, seed))
            if arms is None or not arms["A"] or not arms["B"]:
                return {
                    "result": "DIFFERS",
                    "first_differing_seed": {
                        "system": system,
                        "seed": seed,
                        "field": "missing",
                    },
                    "compared": checked,
                }
            for a in arms["A"]:
                for b in arms["B"]:
                    field = differ_fields(a, b)
                    checked += 1
                    if field is not None and first is None:
                        first = {
                            "system": system,
                            "seed": seed,
                            "field": field,
                            "a_pass": a.get("pass"),
                            "b_pass": b.get("pass"),
                        }
    if first is None:
        return {"result": "IDENTICAL", "first_differing_seed": None, "compared": checked}
    return {"result": "DIFFERS", "first_differing_seed": first, "compared": checked}


def replicate_notes(rows: list[dict]) -> list[dict]:
    notes = []
    grouped = defaultdict(list)
    for row in rows:
        if row.get("error") or row.get("evals") is None:
            continue
        key = (row["arm"], row["method"], row["system"], int(row["seed"]))
        grouped[key].append(row)
    for key, items in sorted(grouped.items()):
        evals = {int(r["evals"]) for r in items}
        hashes = {r.get("traj_sha256") for r in items}
        if len(evals) > 1 or len(hashes) > 1:
            arm, method, system, seed = key
            notes.append(
                {
                    "arm": arm,
                    "method": method,
                    "system": system,
                    "seed": seed,
                    "evals": sorted(evals),
                }
            )
    return notes


def start_notes(rows: list[dict]) -> list[dict]:
    notes = []
    grouped = defaultdict(dict)
    for row in rows:
        if not row.get("start_sha256"):
            continue
        key = (row["method"], row["system"], int(row["seed"]))
        grouped[key].setdefault(row["arm"], set()).add(row["start_sha256"])
    for key, arms in sorted(grouped.items()):
        hashes = set()
        for values in arms.values():
            hashes |= values
        if len(hashes) > 1:
            method, system, seed = key
            notes.append({"method": method, "system": system, "seed": seed})
    return notes


def summarize(rows: list[dict], budget: int, protocol: dict) -> dict:
    blocks = [cost_block(rows, method, system, budget) for method in COST_METHODS for system in SYSTEMS]
    identity = bit_identity(rows)
    return {
        "protocol": protocol,
        "budget": budget,
        "miss_score": budget + 1,
        "bootstrap_samples": BOOT_SAMPLES,
        "bootstrap_seed": BOOT_SEED,
        "ratio_limit": RATIO_LIMIT,
        "verdicts": blocks,
        "bit_identity": identity["result"],
        "bit_identity_detail": identity,
        "replicate_mismatches": replicate_notes(rows),
        "start_mismatches": start_notes(rows),
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
    for block in summary["verdicts"]:
        ratio = block.get("median_eval_ratio_b_over_a")
        ci = block.get("ci95")
        print(
            f"{block['method']} {block['system']} {block['verdict']} "
            f"conv {block['n_conv_a']}/{block['n_conv_b']} ratio {ratio} ci {ci}"
        )
    print(f"bit_identity {summary['bit_identity']}")


if __name__ == "__main__":
    main()
