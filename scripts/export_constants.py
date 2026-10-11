#!/usr/bin/env python3
"""Export constants shared by the Rust sources and the contracts.

``--check`` rewrites nothing and exits nonzero when
``formal/golden/constants.json`` disagrees with the sources.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "formal" / "golden" / "constants.json"

# (json key, source, pattern with one group, kind)
SPECS = (
    (
        "vecops.PAR_CHUNK",
        "src/vecops.rs",
        r"const PAR_CHUNK: usize = (\d+)",
        "int",
    ),
    (
        "vecops.PAR_MIN_LEN",
        "src/vecops.rs",
        r"const PAR_MIN_LEN: usize = ([0-9_]+)",
        "int",
    ),
    (
        "trust.RHO_BAD",
        "src/trust.rs",
        r"const RHO_BAD: f64 = ([0-9.]+)",
        "float",
    ),
    (
        "trust.SHRINK",
        "src/trust.rs",
        r"\(0\.25 \* r\)\.max\(1e-16\)",
        "literal-0.25",
    ),
    (
        "linesearch.ZOOM_GUARD",
        "src/linesearch/known.rs",
        r"lo\.a \+ (0\.1) \* w",
        "float",
    ),
)


def parse_value(kind: str, group: str) -> int | float:
    if kind == "int":
        return int(group.replace("_", ""))
    if kind == "float":
        return float(group)
    if kind == "literal-0.25":
        return 0.25
    raise SystemExit(f"unknown kind {kind}")


def collect() -> dict[str, int | float]:
    found: dict[str, int | float] = {}
    for key, rel, pattern, kind in SPECS:
        text = (ROOT / rel).read_text()
        match = re.search(pattern, text)
        if match is None:
            raise SystemExit(f"{key} not found in {rel}")
        group = match.group(1) if match.lastindex else ""
        found[key] = parse_value(kind, group)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    found = collect()
    payload = json.dumps(found, indent=2, sort_keys=True) + "\n"
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != payload:
            print("export_constants: formal/golden/constants.json is stale", file=sys.stderr)
            print(payload, file=sys.stderr)
            return 1
        print("export_constants: ok")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
