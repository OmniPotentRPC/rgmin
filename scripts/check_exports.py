#!/usr/bin/env python3
"""Fail when librgmin exports a C name outside the rgmin_ namespace.

``nm -D --defined-only`` lists what ``librgmin.so`` exports. Every
``rgmin_`` name there must be declared in ``include/rgmin.h`` and listed in
``scripts/abi_baseline.txt``. Any other name must be reserved to the
toolchain or belong to the C ABI of a dependency crate in
``DEPENDENCY_CRATES``, which rustc exports from every cdylib that links
it. ``librgmin.a`` from the same build tells which crate defines a name.

A static archive cannot hide a symbol, so in ``librgmin.a`` the check
reads the objects rustc wrote for the rgmin crate. An unmangled global
there is a ``no_mangle`` item of this crate and must be an ``rgmin_``
name of the C ABI.

The argument is the directory that holds both libraries, by default
``target/debug``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_abi import BASELINE, ROOT, names

# eindir-core: a deliberate public re-export. rgmin_minimize_eindir and
# its _opts and _sized forms take an eindir_objective_t and an
# eindir_abi_stamp_t, and include/rgmin.h has no rgmin_ name that builds
# either, so a C caller linked against librgmin alone gets them from
# eindir_objective_new and eindir_core_abi_stamp here. tests/cxx_headers.cpp
# links that way. v0.7.0 and v0.8.0 shipped these names, so hiding them
# would break released callers. readcon-core: linked only with `readcon`.
DEPENDENCY_CRATES = ("eindir_core", "readcon_core")
# Linker and unwinder names. A leading underscore is reserved to the
# implementation, which also covers Rust (`_ZN`, `_R`) mangling.
TOOLCHAIN = ("rust_eh_personality", "DW.ref.rust_eh_personality")
# rustc names a crate's objects `<crate>-<hash>.<crate>.<cgu>.rcgu.o`,
# and the library's own crate `<crate>.<cgu>.rcgu.o`.
MEMBER = re.compile(r"^([A-Za-z0-9_]+)[-.].*\.rcgu\.o$")


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


def archive_owners(path: Path) -> dict[str, str]:
    """Each global the archive defines, and the crate whose object defines it.

    Objects that are not rustc's, such as HiGHS or GMP, have no crate.
    """
    owners = {}
    for line in nm("-A", "-g", "--defined-only", str(path)):
        # `archive:member:address type name`
        fields = line.rsplit(None, 2)
        if len(fields) != 3:
            continue
        member = fields[0].rsplit(":", 2)[-2]
        crate = MEMBER.match(member)
        owners[fields[2]] = crate.group(1) if crate else ""
    return owners


def abi_problem(name: str, declared: set[str], baseline: set[str]) -> str | None:
    if name not in declared:
        return f"  {name} is not declared in include/rgmin.h"
    if name not in baseline:
        return f"  {name} is not in scripts/abi_baseline.txt"
    return None


def shared_problems(
    exported: set[str], owners: dict[str, str], declared: set[str], baseline: set[str]
) -> list[str]:
    found = []
    for name in sorted(exported):
        if toolchain(name):
            continue
        if name.startswith("rgmin_"):
            problem = abi_problem(name, declared, baseline)
        elif owners.get(name) in DEPENDENCY_CRATES:
            # Public through the dependency's C ABI; see DEPENDENCY_CRATES.
            problem = None
        else:
            owner = owners.get(name) or "no rust crate"
            problem = f"  {name} ({owner}) is outside the rgmin_ namespace"
        if problem:
            found.append(problem)
    return found


def archive_problems(owners: dict[str, str], declared: set[str], baseline: set[str]) -> list[str]:
    found = []
    for name, crate in sorted(owners.items()):
        if crate != "rgmin" or toolchain(name):
            continue
        if name.startswith("rgmin_"):
            problem = abi_problem(name, declared, baseline)
        else:
            problem = f"  {name} is outside the rgmin_ namespace"
        if problem:
            found.append(problem)
    return found


def report(path: Path, found: list[str]) -> bool:
    if found:
        print(f"{path} exports names outside the C ABI:")
        print("\n".join(found))
        return False
    print(f"{path}: every exported name is in the C ABI")
    return True


def main(argv: list[str]) -> int:
    directory = Path(argv[0]) if argv else ROOT / "target" / "debug"
    shared = directory / "librgmin.so"
    archive = directory / "librgmin.a"
    missing = [str(path) for path in (shared, archive) if not path.is_file()]
    if missing:
        print(f"missing {', '.join(missing)}; run cargo build first")
        return 1
    declared = set(names())
    baseline = {
        line.strip()
        for line in BASELINE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    owners = archive_owners(archive)
    ok = report(shared, shared_problems(shared_exports(shared), owners, declared, baseline))
    ok = report(archive, archive_problems(owners, declared, baseline)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
