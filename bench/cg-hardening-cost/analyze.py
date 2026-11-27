#!/usr/bin/env python3
"""Paired ratios, bootstrap intervals, ladder slopes, and the PASS/FLAG rule."""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20261123
EVAL_CAP = 1.10


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def median(values):
    return statistics.median(values)


def bootstrap_median_ci(ratios):
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(ratios)
    samples = []
    for _ in range(BOOTSTRAP):
        draw = [ratios[rng.randrange(n)] for _ in range(n)]
        samples.append(median(draw))
    samples.sort()
    return median(ratios), percentile(samples, 2.5), percentile(samples, 97.5)


def slope(ns, walls):
    xs = [math.log(n) for n in ns]
    ys = [math.log(w) for w in walls]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    var = sum((x - xbar) ** 2 for x in xs)
    if var == 0.0:
        return None
    cov = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    return cov / var


def load(path):
    header = None
    runs = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "header":
                header = row
            elif row.get("type") == "run":
                runs.append(row)
    return header, runs


def aggregate(runs):
    groups = defaultdict(list)
    for row in runs:
        if not row.get("ok"):
            key = (
                row.get("arm"),
                row.get("variant"),
                row.get("case"),
                row.get("n"),
                row.get("seed"),
            )
            groups[key].append(row)
            continue
        key = (row["arm"], row["variant"], row["case"], row["n"], row["seed"])
        groups[key].append(row)
    rows = []
    for key, items in groups.items():
        arm, variant, case, n, seed = key
        good = [item for item in items if item.get("ok")]
        record = {
            "arm": arm,
            "variant": variant,
            "case": case,
            "n": n,
            "seed": seed,
            "repeats": len(items),
            "ok_repeats": len(good),
        }
        if not good:
            record.update(
                energy_evals="",
                grad_evals="",
                solver_evals="",
                iterations="",
                final_f="",
                grad_norm="",
                converged="",
                wall_s="",
                stable="0",
            )
            rows.append(record)
            continue
        def same(field):
            vals = [item[field] for item in good]
            return vals[0], len(set(map(str, vals))) == 1

        energy, energy_stable = same("energy_evals")
        grad, grad_stable = same("grad_evals")
        solver, solver_stable = same("solver_evals")
        iters, iter_stable = same("iterations")
        final_f, f_stable = same("final_f")
        gnorm, g_stable = same("grad_norm")
        conv, conv_stable = same("converged")
        walls = [item["wall_s"] for item in good]
        record.update(
            energy_evals=energy,
            grad_evals=grad,
            solver_evals=solver,
            iterations=iters,
            final_f=final_f,
            grad_norm=gnorm,
            converged="1" if conv else "0",
            wall_s=median(walls),
            stable="1"
            if all(
                (
                    energy_stable,
                    grad_stable,
                    solver_stable,
                    iter_stable,
                    f_stable,
                    g_stable,
                    conv_stable,
                )
            )
            else "0",
        )
        rows.append(record)
    rows.sort(key=lambda r: (r["variant"], r["case"], r["n"], r["seed"], r["arm"]))
    return rows


def write_rows(path, rows):
    fields = [
        "arm",
        "variant",
        "case",
        "n",
        "seed",
        "energy_evals",
        "grad_evals",
        "solver_evals",
        "iterations",
        "final_f",
        "grad_norm",
        "converged",
        "wall_s",
        "stable",
        "ok_repeats",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cell_table(rows):
    by = defaultdict(dict)
    for row in rows:
        if row["solver_evals"] == "" or row["wall_s"] == "":
            continue
        key = (row["variant"], row["case"], row["n"], row["seed"])
        by[key][row["arm"]] = row
    cells = defaultdict(list)
    for key, arms in by.items():
        if "A" not in arms or "B" not in arms:
            continue
        variant, case, n, seed = key
        a = arms["A"]
        b = arms["B"]
        cells[(variant, case, n)].append((seed, a, b))
    out = []
    for (variant, case, n), pairs in sorted(cells.items()):
        eval_ratios = [b["solver_evals"] / a["solver_evals"] for _, a, b in pairs]
        wall_ratios = [b["wall_s"] / a["wall_s"] for _, a, b in pairs if a["wall_s"] > 0 and b["wall_s"] > 0]
        df = [b["final_f"] - a["final_f"] for _, a, b in pairs]
        conv_a = sum(a["converged"] == "1" for _, a, _ in pairs)
        conv_b = sum(b["converged"] == "1" for _, _, b in pairs)
        eval_med, eval_lo, eval_hi = bootstrap_median_ci(eval_ratios)
        if wall_ratios:
            wall_med, wall_lo, wall_hi = bootstrap_median_ci(wall_ratios)
        else:
            wall_med = wall_lo = wall_hi = None
        verdict = "PASS" if conv_b >= conv_a and eval_hi <= EVAL_CAP else "FLAG"
        out.append(
            {
                "variant": variant,
                "case": case,
                "n": n,
                "seeds": len(pairs),
                "conv_a": conv_a,
                "conv_b": conv_b,
                "eval_ratio": eval_med,
                "eval_lo": eval_lo,
                "eval_hi": eval_hi,
                "wall_ratio": wall_med,
                "wall_lo": wall_lo,
                "wall_hi": wall_hi,
                "median_df": median(df),
                "verdict": verdict,
            }
        )
    return out


def ladder_slopes(rows):
    bucket = defaultdict(list)
    for row in rows:
        if row["case"] != "ladder" or row["wall_s"] == "":
            continue
        bucket[(row["variant"], row["arm"], row["n"])].append(row["wall_s"])
    by_arm = defaultdict(list)
    for (variant, arm, n), walls in bucket.items():
        by_arm[(variant, arm)].append((n, median(walls)))
    slopes = []
    for (variant, arm), points in sorted(by_arm.items()):
        points.sort()
        ns = [n for n, _ in points]
        ws = [w for _, w in points]
        slopes.append(
            {
                "variant": variant,
                "arm": arm,
                "n": ns,
                "wall": ws,
                "slope": slope(ns, ws) if len(points) >= 2 else None,
            }
        )
    return slopes


def fmt(value, digits=4):
    if value is None:
        return ""
    return f"{value:.6g}" if digits == 6 else f"{value:.{digits}f}"


def write_report(path, header, cells, slopes, rows):
    twenty = [c for c in cells if c["case"] == "rosenbrock_fr_20"]
    lines = []
    lines.append("# CG hardening cost")
    lines.append("")
    lines.append("Arm A is develop `0df557c2355011419b4fa60ead390f300940b361`.")
    lines.append("Arm B is `83f6ddc1a32a35da65482231c35f0b25a9bbb37a`.")
    if header:
        lines.append(f"Machine id sha256: `{header.get('machine_id_sha256')}`.")
        binaries = header.get("binary_sha256", {})
        lines.append(f"Binary sha256 A: `{binaries.get('A')}`.")
        lines.append(f"Binary sha256 B: `{binaries.get('B')}`.")
        lines.append(
            f"Protocol: gtol {header.get('gtol')}, maxiter {header.get('maxiter')} "
            f"(the 20-step Rosenbrock FR row uses maxiter 20 and gtol {header.get('twenty_step_gtol')} "
            f"so the iteration cap is reached), Wolfe "
            f"{header.get('wolfe')}, istep {header.get('istep')}, "
            f"{header.get('repeats')} wall repeats, order {header.get('order')}, core {header.get('core')}."
        )
        lines.append(
            f"LJ: {header.get('lj')} (rgpot {header.get('rgpot')}, linkcell {header.get('linkcell')})."
        )
    lines.append("")
    lines.append("## Twenty-step Rosenbrock FR")
    lines.append("")
    if twenty:
        cell = twenty[0]
        a_row = next(r for r in rows if r["case"] == "rosenbrock_fr_20" and r["arm"] == "A")
        b_row = next(r for r in rows if r["case"] == "rosenbrock_fr_20" and r["arm"] == "B")
        observed = (a_row["solver_evals"], b_row["solver_evals"])
        word = "confirm" if observed == (76, 93) else "refute"
        lines.append(
            f"Author: 76 vs 93. Observed solver evals: {observed[0]} vs {observed[1]}. {word}."
        )
        lines.append(
            f"Energy evals {a_row['energy_evals']} vs {b_row['energy_evals']}, "
            f"gradient evals {a_row['grad_evals']} vs {b_row['grad_evals']}, "
            f"iterations {a_row['iterations']} vs {b_row['iterations']}, "
            f"converged {a_row['converged']} vs {b_row['converged']}."
        )
        lines.append(
            f"Cell verdict {cell['verdict']}: conv {cell['conv_a']}/{cell['conv_b']}, "
            f"eval ratio {cell['eval_ratio']:.6g} CI [{cell['eval_lo']:.6g}, {cell['eval_hi']:.6g}]."
        )
    else:
        lines.append("The 20-step row is absent.")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append(
        "Eval and wall ratios are medians of paired per-seed B/A ratios. "
        "The interval is a 10000-draw bootstrap of that median, seed 20261123, "
        "reseeded on each cell, 2.5 and 97.5 percentiles with linear interpolation. "
        "PASS means B converged on at least as many seeds as A and the eval-ratio upper end is at most 1.10. "
        "The eval count is the solver's objective-evaluation counter."
    )
    lines.append("")
    lines.append(
        "| variant | case | N | seeds | conv A | conv B | eval B/A | eval CI | wall B/A | wall CI | median fB-fA | verdict |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in cells:
        lines.append(
            "| {variant} | {case} | {n} | {seeds} | {conv_a} | {conv_b} | {er} | [{elo}, {ehi}] | {wr} | [{wlo}, {whi}] | {df} | {verdict} |".format(
                er=f"{cell['eval_ratio']:.6g}",
                elo=f"{cell['eval_lo']:.6g}",
                ehi=f"{cell['eval_hi']:.6g}",
                wr="" if cell["wall_ratio"] is None else f"{cell['wall_ratio']:.6g}",
                wlo="" if cell["wall_lo"] is None else f"{cell['wall_lo']:.6g}",
                whi="" if cell["wall_hi"] is None else f"{cell['wall_hi']:.6g}",
                df=f"{cell['median_df']:.6g}",
                **cell,
            )
        )
    lines.append("")
    lines.append("## Ladder slope")
    lines.append("")
    lines.append("Ordinary least squares slope of log(median wall) against log(N).")
    lines.append("")
    lines.append("| variant | arm | N | median wall | slope |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in slopes:
        lines.append(
            f"| {item['variant']} | {item['arm']} | {item['n']} | "
            f"{[round(w, 6) for w in item['wall']]} | {item['slope']:.6g} |"
        )
    lines.append("")
    unstable = sum(1 for row in rows if row.get("stable") == "0")
    lines.append(f"Rows whose deterministic fields moved across repeats: {unstable}.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench/cg-hardening-cost")
    args = parser.parse_args()
    out = Path(args.out)
    header, runs = load(out / "sessions.jsonl")
    rows = aggregate(runs)
    write_rows(out / "rows.csv", rows)
    cells = cell_table(rows)
    slopes = ladder_slopes(rows)
    verdict = {
        "rule": "PASS if conv_B >= conv_A and eval_ratio 97.5% <= 1.10, else FLAG",
        "bootstrap": BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "header": header,
        "cells": cells,
        "ladder_slope": slopes,
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    write_report(out / "REPORT.md", header, cells, slopes, rows)
    flagged = sum(1 for cell in cells if cell["verdict"] == "FLAG")
    print(f"cells {len(cells)} flag {flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
