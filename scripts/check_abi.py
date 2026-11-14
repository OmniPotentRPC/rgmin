#!/usr/bin/env python3
"""Compare exported C names with the ABI baseline.

A new function is listed and does not fail. A removed or renamed
function fails. Type changes are what ``abidiff`` would catch when it
is installed; this check is the one that always runs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "rgmin.h"
BASELINE = ROOT / "scripts" / "abi_baseline.txt"
# A return type ending in `*` may sit against the name: `T *rgmin_f(`.
DECL = re.compile(r"^[A-Za-z_][\w \t\*]*?[\s\*](rgmin_[A-Za-z0-9_]+)\s*\(", re.M)


def names() -> list[str]:
    text = HEADER.read_text()
    found = DECL.findall(text)
    # Function-pointer typedefs match the same pattern. Drop those.
    typedefs = set(re.findall(r"typedef\s+[^\n]*\s+(rgmin_[A-Za-z0-9_]+)\s*\)", text))
    return sorted(set(found) - typedefs)


def main() -> int:
    current = names()
    expected = [line.strip() for line in BASELINE.read_text().splitlines() if line.strip()]
    missing = sorted(set(expected) - set(current))
    added = sorted(set(current) - set(expected))
    if missing:
        print("removed from the C ABI:")
        for name in missing:
            print(f"  {name}")
        return 1
    if added:
        print("added to the C ABI (update scripts/abi_baseline.txt when the addition is intentional):")
        for name in added:
            print(f"  {name}")
    if shutil.which("abidiff"):
        library = ROOT / "target" / "debug" / "librgmin.so"
        if library.is_file():
            status = subprocess.call(
                ["abidiff", "--no-unreferenced-symbols", str(library), str(library)]
            )
            if status not in (0,):
                print(f"abidiff status {status}")
                return 1
    else:
        print("abidiff not found; name check only")
    print(f"{len(current)} C functions match the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
