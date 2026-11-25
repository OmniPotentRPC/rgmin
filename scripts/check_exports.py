#!/usr/bin/env python3
"""Fail when librgmin exports a C name outside the rgmin_ namespace.

``nm -D --defined-only`` lists what the shared library exports. Every
``rgmin_`` name there must be declared in ``include/rgmin.h`` and listed in
``scripts/abi_baseline.txt``. The only other names allowed are the
``eindir_`` C ABI of eindir-core, which librgmin carries for callers of
``rgmin_minimize_eindir``, and names the toolchain reserves.

A static archive cannot hide a symbol, so for ``librgmin.a`` the check
reads the members rustc wrote for the rgmin crate. An unmangled global
there is a ``no_mangle`` item of this crate.

With no arguments the check reads ``target/debug/librgmin.so`` and
``target/debug/librgmin.a``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_abi import BASELINE, ROOT, names  # noqa: E402

DEFAULT = (ROOT / "target" / "debug" / "librgmin.so", ROOT / "target" / "debug" / "librgmin.a")
DEPENDENCY_PREFIXES = ("eindir_",)
# Linker and unwinder names. A leading underscore is reserved to the
# implementation, which also covers Rust (`_ZN`, `_R`) mangling.
TOOLCHAIN = ("rust_eh_personality", "DW.ref.rust_eh_personality")


def toolchain(name: str) -> bool:
    return name.startswith("_") or name in TOOLCHAIN


def nm(*args: str) -> list[str]:
    done = subprocess.run(["nm", *args], capture_output=True, text=True, check=True)
    return done.stdout.splitlines()


def shared_exports(path: Path) -> set[str]:
    exported = set()
    for line in nm("-D", "--defined-only", str(path)):
        fields = line.split()
        if fields:
            exported.add(fields[-1].split("@", 1)[0])
    return exported


def archive_exports(path: Path) -> set[str]:
    """Unmangled globals that the rgmin crate's own objects define."""
    exported = set()
    for line in nm("-A", "-g", "--defined-only", str(path)):
        # `archive:member:address type name`
        fields = line.rsplit(None, 2)
        if len(fields) != 3:
            continue
        member = fields[0].rsplit(":", 2)[-2]
        if member.startswith(("rgmin.", "rgmin-")):
            exported.add(fields[2])
    return exported


def problems(path: Path, declared: set[str], baseline: set[str]) -> list[str]:
    shared = path.suffix != ".a"
    exported = shared_exports(path) if shared else archive_exports(path)
    found = []
    for name in sorted(exported):
        if toolchain(name):
            continue
        if name.startswith("rgmin_"):
            if name not in declared:
                found.append(f"  {name} is not declared in include/rgmin.h")
            elif name not in baseline:
                found.append(f"  {name} is not in scripts/abi_baseline.txt")
        elif not (shared and name.startswith(DEPENDENCY_PREFIXES)):
            found.append(f"  {name} is outside the rgmin_ namespace")
    return found


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv] or [path for path in DEFAULT if path.is_file()]
    if not paths:
        print("no library to check; run cargo build first or name one")
        return 1
    declared = set(names())
    baseline = {
        line.strip()
        for line in BASELINE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    failed = False
    for path in paths:
        found = problems(path, declared, baseline)
        if found:
            failed = True
            print(f"{path} exports names outside the C ABI:")
            print("\n".join(found))
        else:
            print(f"{path}: every exported name is in the C ABI")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
