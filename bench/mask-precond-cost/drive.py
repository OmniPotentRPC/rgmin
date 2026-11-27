"""Build both arms and run the mask-precond cost campaign.

Arm A is this checkout. Arm B is a detached worktree at the mask-precond
tip. Each (system, method, seed) runs A, B, B, A on one core.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TIP = "6630fbcc904def63d99a6615b8260bcf07307fab"
BASE = "0df557c2355011419b4fa60ead390f300940b361"
SYSTEMS = ("lj38", "lj75")
METHODS = ("fire2-exp", "eon", "fire2")
ORDER = ("A", "B", "B", "A")
BUDGET = 2000
FMAX = 0.01
SIGMA = 0.02
CORE = 0

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner"
WORKTREE = Path("bench/mask-precond-cost/arm-b")
LOGS = HERE / "logs"


def show(cmd: list[str], cwd: Path, log_name: str) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / log_name
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=child_env())
    print(f"{log_name} EXIT {proc.returncode}")
    return proc.returncode


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "RUSTUP_TOOLCHAIN": "stable",
            "RAYON_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CARGO_TERM_COLOR": "never",
        }
    )
    return env


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return proc.stdout.strip()


def ensure_tip() -> None:
    found = git("rev-parse", TIP)
    if found != TIP:
        raise SystemExit(f"tip resolved to {found}")
    base = git("merge-base", "--is-ancestor", BASE, TIP)
    del base


def ensure_worktree() -> None:
    marker = ROOT / WORKTREE / ".git"
    if marker.exists():
        return
    show(
        ["git", "worktree", "add", "--detach", str(WORKTREE), TIP],
        ROOT,
        "worktree.log",
    )


def copy_runner() -> None:
    dest = ROOT / WORKTREE / "bench" / "mask-precond-cost" / "runner"
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUNNER, dest, ignore=shutil.ignore_patterns("target"))


def build(which: str, cwd: Path) -> int:
    return show(
        ["cargo", "build", "--release", "--manifest-path", "bench/mask-precond-cost/runner/Cargo.toml"],
        cwd,
        f"build-{which}.log",
    )


def binary(which: str) -> Path:
    if which == "A":
        return RUNNER / "target" / "release" / "mask-precond-cost"
    return ROOT / WORKTREE / "bench" / "mask-precond-cost" / "runner" / "target" / "release" / "mask-precond-cost"


def run_one(which: str, method: str, system: str, seed: int, slot: int) -> dict:
    cwd = ROOT if which == "A" else ROOT / WORKTREE
    cmd = [
        str(binary(which)),
        "--arm",
        which,
        "--method",
        method,
        "--system",
        system,
        "--seed",
        str(seed),
        "--budget",
        str(BUDGET),
        "--fmax",
        str(FMAX),
        "--sigma",
        str(SIGMA),
        "--core",
        str(CORE),
        "--fixtures",
        "bench/optimizers/data",
    ]
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env(),
        timeout=300,
    )
    row: dict
    text = proc.stdout.strip().splitlines()
    if text:
        try:
            row = json.loads(text[-1])
        except json.JSONDecodeError:
            row = {"error": "bad json", "converged": False}
    else:
        row = {"error": "no output", "converged": False}
    row["arm"] = which
    row["method"] = method
    row["system"] = system
    row["seed"] = seed
    row["pass"] = slot
    row["exit"] = proc.returncode
    if proc.returncode != 0 and not row.get("error"):
        err = proc.stderr.strip().splitlines()
        row["error"] = err[-1] if err else f"exit {proc.returncode}"
    if proc.returncode != 0:
        err_path = LOGS / "run-errors.log"
        with err_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{which} {method} {system} {seed} {slot} exit {proc.returncode}\n")
            handle.write(proc.stderr[-500:])
            handle.write("\n")
    return row


def write_rows(rows: list[dict]) -> None:
    json_path = HERE / "rows.json"
    csv_path = HERE / "rows.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fields = [
        "arm",
        "method",
        "system",
        "seed",
        "pass",
        "evals",
        "converged",
        "final_E",
        "max_force",
        "wall_s",
        "core",
        "affinity",
        "energy_bits",
        "coord_sha256",
        "traj_sha256",
        "start_sha256",
        "error",
        "exit",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "arm": row.get("arm"),
                    "method": row.get("method"),
                    "system": row.get("system"),
                    "seed": row.get("seed"),
                    "pass": row.get("pass"),
                    "evals": row.get("evals"),
                    "converged": int(bool(row.get("converged"))),
                    "final_E": row.get("energy"),
                    "max_force": row.get("fmax"),
                    "wall_s": row.get("wall_s"),
                    "core": row.get("core"),
                    "affinity": row.get("affinity"),
                    "energy_bits": row.get("energy_bits"),
                    "coord_sha256": row.get("coord_sha256"),
                    "traj_sha256": row.get("traj_sha256"),
                    "start_sha256": row.get("start_sha256"),
                    "error": row.get("error"),
                    "exit": row.get("exit"),
                }
            )


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    ensure_tip()
    code_a = build("A", ROOT)
    if code_a != 0:
        raise SystemExit(code_a)
    ensure_worktree()
    head_b = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT / WORKTREE,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    print(f"arm-b HEAD {head_b}")
    if head_b != TIP:
        raise SystemExit("worktree is not the requested tip")
    copy_runner()
    code_b = build("B", ROOT / WORKTREE)
    if code_b != 0:
        raise SystemExit(code_b)
    rows: list[dict] = []
    total = len(SYSTEMS) * len(METHODS) * 12 * len(ORDER)
    done = 0
    fails = 0
    for system in SYSTEMS:
        for method in METHODS:
            for seed in range(2000, 2012):
                for slot, arm in enumerate(ORDER, start=1):
                    row = run_one(arm, method, system, seed, slot)
                    rows.append(row)
                    done += 1
                    if row.get("exit") != 0 or row.get("error"):
                        fails += 1
                    if done % 12 == 0:
                        print(f"progress {done}/{total} fails {fails}")
                        write_rows(rows)
    write_rows(rows)
    protocol = {
        "base": BASE,
        "optimizer_a": BASE,
        "optimizer_b": TIP,
        "systems": list(SYSTEMS),
        "methods": list(METHODS),
        "seeds": list(range(2000, 2012)),
        "order": list(ORDER),
        "budget": BUDGET,
        "fmax": FMAX,
        "sigma": SIGMA,
        "atom_cap": 0.2,
        "core": CORE,
        "potential": "pairwise Lennard-Jones epsilon=1 sigma=1 from tests/gpmin.rs",
        "neighbours": "linkcell::pairs_open 0.6.0",
        "fire_istep": 0.1,
        "eon_cap": 0.2,
    }
    protocol_path = LOGS / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    summary = HERE / "summary.json"
    code = subprocess.run(
        [
            sys.executable,
            str(HERE / "analyze.py"),
            "--rows",
            str(HERE / "rows.json"),
            "--summary",
            str(summary),
            "--budget",
            str(BUDGET),
            "--protocol",
            str(protocol_path),
        ],
        cwd=ROOT,
    ).returncode
    print(f"analyze EXIT {code}")
    print(f"rows {len(rows)} fails {fails}")
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
