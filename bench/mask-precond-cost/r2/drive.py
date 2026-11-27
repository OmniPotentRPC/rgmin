"""Build both arms and run the r2 mask-precond cost campaign.

Arm A is this checkout (develop's optimizer). Arm B is a detached
worktree at the r2 tip. Each (energy, system, method, seed) runs
A, B, B, A five times on one core, so each arm has ten wall-clock repeats.
The metric kick is compiled only into the candidate.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TIP = "839364e0ac2c33abe113841e1997f83f7babfe80"
BASE = "0df557c2355011419b4fa60ead390f300940b361"
SYSTEMS = ("lj38", "lj75")
METHODS = ("fire2-exp", "fire2-exp-kick", "eon", "fire2")
ORDER = ("A", "B", "B", "A")
CYCLES = 5
BUDGET = 2000
FMAX = 0.01
SIGMA = 0.02
CORE = 0
PREFIX_DEFAULT = "bench/mask-precond-cost/logs/rgpot-prefix"

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner"
WORKTREE = Path("bench/mask-precond-cost/r2/arm-b")
LOGS = HERE / "logs"
BIN = HERE / "bin"


def child_env(prefix: Path | None) -> dict[str, str]:
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
            "CC": "gcc",
            "CXX": "g++",
        }
    )
    if prefix is not None:
        lib = prefix / "lib" / "x86_64-linux-gnu"
        if not (lib / "librgpot.so").exists():
            lib = prefix / "lib"
        env["RGPOT_PREFIX"] = str(prefix)
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(lib) if not prev else str(lib) + os.pathsep + prev
    return env


def show(cmd: list[str], cwd: Path, log_name: str, env: dict[str, str]) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / log_name
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"{log_name} EXIT {proc.returncode}")
    return proc.returncode


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
    git("merge-base", "--is-ancestor", BASE, TIP)


def ensure_worktree() -> None:
    marker = ROOT / WORKTREE / ".git"
    if marker.exists():
        return
    show(
        ["git", "worktree", "add", "--detach", str(WORKTREE), TIP],
        ROOT,
        "worktree.log",
        child_env(None),
    )


def copy_runner() -> None:
    dest = ROOT / WORKTREE / "bench" / "mask-precond-cost" / "r2" / "runner"
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        RUNNER,
        dest,
        ignore=shutil.ignore_patterns("target", "target-kick"),
    )


def cargo(features: list[str], cwd: Path, log_name: str, prefix: Path | None) -> int:
    cmd = [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        "bench/mask-precond-cost/r2/runner/Cargo.toml",
    ]
    if features:
        cmd.extend(["--features", ",".join(features)])
    return show(cmd, cwd, log_name, child_env(prefix))


def built_binary(cwd: Path) -> Path:
    return (
        cwd
        / "bench"
        / "mask-precond-cost"
        / "r2"
        / "runner"
        / "target"
        / "release"
        / "mask-precond-r2"
    )


def resolve_prefix() -> Path | None:
    raw = os.environ.get("RGPOT_PREFIX", PREFIX_DEFAULT)
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if (path / "include" / "rgpot" / "LennardJones" / "LJClusterPot.hpp").exists():
        return path
    return None


def build_arm(which: str, cwd: Path, prefix: Path | None, rgpot: bool) -> tuple[Path, Path | None]:
    features = ["rgpot"] if rgpot else []
    code = cargo(features, cwd, f"build-{which}.log", prefix if rgpot else None)
    if code != 0:
        raise SystemExit(code)
    BIN.mkdir(parents=True, exist_ok=True)
    default = BIN / f"{which}-default"
    shutil.copy2(built_binary(cwd), default)
    kick_features = features + ["metric-kick"]
    code = cargo(kick_features, cwd, f"build-{which}-kick.log", prefix if rgpot else None)
    kick: Path | None = None
    if code == 0:
        kick = BIN / f"{which}-kick"
        shutil.copy2(built_binary(cwd), kick)
    return default, kick


def log_tail(name: str) -> str:
    path = LOGS / name
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-30:])


def run_one(
    binary: Path,
    which: str,
    method: str,
    system: str,
    seed: int,
    slot: int,
    energy: str,
    kick: bool,
    prefix: Path | None,
) -> dict:
    cwd = ROOT if which == "A" else ROOT / WORKTREE
    invoked = method
    if method == "fire2-exp-kick" and not kick:
        invoked = "fire2-exp"
    cmd = [
        str(binary),
        "--arm",
        which,
        "--method",
        invoked,
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
        "--energy",
        energy,
    ]
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env(prefix if energy == "rgpot" else None),
        timeout=300,
    )
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
    row["energy_arm"] = energy
    row["kick"] = kick
    row["exit"] = proc.returncode
    if proc.returncode != 0 and not row.get("error"):
        err = proc.stderr.strip().splitlines()
        row["error"] = err[-1] if err else f"exit {proc.returncode}"
    if proc.returncode != 0:
        err_path = LOGS / "run-errors.log"
        with err_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{which} {energy} {method} {system} {seed} {slot} exit {proc.returncode}\n"
            )
            handle.write(proc.stderr[-500:])
            handle.write("\n")
    return row


def write_rows(rows: list[dict]) -> None:
    json_path = HERE / "rows.json"
    csv_path = HERE / "rows.csv"
    json_path.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    fields = [
        "arm",
        "energy_arm",
        "method",
        "system",
        "seed",
        "pass",
        "evals",
        "iterations",
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
        "kick",
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
                    "energy_arm": row.get("energy_arm"),
                    "method": row.get("method"),
                    "system": row.get("system"),
                    "seed": row.get("seed"),
                    "pass": row.get("pass"),
                    "evals": row.get("evals"),
                    "iterations": row.get("iterations"),
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
                    "kick": int(bool(row.get("kick"))),
                    "error": row.get("error") or "",
                    "exit": row.get("exit"),
                }
            )


def rgpot_version(prefix: Path | None) -> str | None:
    if prefix is None:
        return None
    pc = prefix / "lib" / "x86_64-linux-gnu" / "pkgconfig" / "rgpot.pc"
    if not pc.exists():
        pc = prefix / "lib" / "pkgconfig" / "rgpot.pc"
    if not pc.exists():
        return None
    for line in pc.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return None


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    ensure_tip()
    prefix = resolve_prefix()
    rgpot = prefix is not None
    rgpot_error = ""
    if not rgpot:
        rgpot_error = "RGPOT_PREFIX has no include/rgpot/LennardJones/LJClusterPot.hpp"
        print(f"rgpot skipped: {rgpot_error}")
    code_a = 0
    if rgpot:
        code_a = cargo(["rgpot"], ROOT, "build-A-rgpot.log", prefix)
        if code_a != 0:
            rgpot = False
            rgpot_error = log_tail("build-A-rgpot.log")
            print("rgpot build failed; continuing with the repo LJ arm")
    default_a, kick_a = build_arm("A", ROOT, prefix if rgpot else None, rgpot)
    del kick_a
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
    default_b, kick_b = build_arm("B", ROOT / WORKTREE, prefix if rgpot else None, rgpot)
    if kick_b is None:
        print(log_tail("build-B-kick.log"))
        raise SystemExit("candidate metric-kick build failed")
    energies = ["repo"] + (["rgpot"] if rgpot else [])
    rows: list[dict] = []
    total = len(energies) * len(SYSTEMS) * len(METHODS) * 12 * CYCLES * len(ORDER)
    done = 0
    fails = 0
    binaries = {"A": default_a, "B": default_b}
    for energy in energies:
        for system in SYSTEMS:
            for method in METHODS:
                for seed in range(2000, 2012):
                    slot = 1
                    for _cycle in range(CYCLES):
                        for arm in ORDER:
                            kick = method == "fire2-exp-kick" and arm == "B"
                            binary = kick_b if kick else binaries[arm]
                            row = run_one(
                                binary,
                                arm,
                                method,
                                system,
                                seed,
                                slot,
                                energy,
                                kick,
                                prefix,
                            )
                            rows.append(row)
                            done += 1
                            slot += 1
                            if row.get("exit") != 0 or row.get("error"):
                                fails += 1
                    if done % (CYCLES * len(ORDER)) == 0:
                        print(f"progress {done}/{total} fails {fails} {energy} {system} {method} {seed}")
                        write_rows(rows)
    write_rows(rows)
    protocol = {
        "base": BASE,
        "optimizer_a": BASE,
        "optimizer_b": TIP,
        "systems": list(SYSTEMS),
        "methods": list(METHODS),
        "energies": energies,
        "seeds": list(range(2000, 2012)),
        "order": list(ORDER),
        "cycles": CYCLES,
        "wall_repeats": CYCLES * 2,
        "budget": BUDGET,
        "fmax": FMAX,
        "sigma": SIGMA,
        "atom_cap": 0.2,
        "core": CORE,
        "potential": "pairwise Lennard-Jones epsilon=1 sigma=1 from tests/gpmin.rs",
        "neighbours": "linkcell::pairs_open 0.6.0",
        "rgpot": rgpot_version(prefix) if rgpot else None,
        "rgpot_pes": "rgpot::LJClusterPot default constructor" if rgpot else None,
        "rgpot_error": rgpot_error,
        "fire_istep": 0.1,
        "eon_cap": 0.2,
        "kick": "set_fire_metric_kick(true) on the candidate only, for fire2-exp-kick",
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
