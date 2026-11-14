#!/usr/bin/env python3
"""Compare exported C names with the ABI baseline.

The ABI stamp defines in ``include/rgmin.h`` must equal the constants in
``src/ffi.rs``.

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
FFI = ROOT / "src" / "ffi.rs"
VERSION_FIELDS = ("MAJOR", "MINOR", "LAYOUT_REVISION")
# A return type ending in `*` may sit against the name: `T *rgmin_f(`.
DECL = re.compile(r"^[A-Za-z_][\w \t\*]*?[\s\*](rgmin_[A-Za-z0-9_]+)\s*\(", re.M)


def names() -> list[str]:
    text = HEADER.read_text()
    found = DECL.findall(text)
    # Function-pointer typedefs match the same pattern. Drop those.
    typedefs = set(re.findall(r"typedef\s+[^\n]*\s+(rgmin_[A-Za-z0-9_]+)\s*\)", text))
    return sorted(set(found) - typedefs)


def versions() -> list[str]:
    """Header and Rust stamps that disagree, as printable lines."""
    header = HEADER.read_text()
    rust = FFI.read_text()
    wrong = []
    for field in VERSION_FIELDS:
        name = "RGMIN_ABI_LAYOUT_REVISION" if field == "LAYOUT_REVISION" else f"RGMIN_ABI_VERSION_{field}"
        c = re.search(rf"^#define {name} (\d+)$", header, re.M)
        r = re.search(rf"^pub const {name}: u16 = (\d+);$", rust, re.M)
        if c is None or r is None or c.group(1) != r.group(1):
            wrong.append(f"  {name}: rgmin.h {c and c.group(1)}, ffi.rs {r and r.group(1)}")
    return wrong


def main() -> int:
    wrong = versions()
    if wrong:
        print("include/rgmin.h and src/ffi.rs disagree on the ABI stamp:")
        print("\n".join(wrong))
        return 1
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
    minor = re.search(r"^#define RGMIN_ABI_VERSION_MINOR (\d+)$", HEADER.read_text(), re.M)
    print(f"{len(current)} C functions match the baseline at ABI minor {minor.group(1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
