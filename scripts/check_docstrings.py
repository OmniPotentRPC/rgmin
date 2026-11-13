#!/usr/bin/env python3
"""Docstrings on the public Python, C, and C++ surfaces.

Each language reference names the quadratic oracle, a value below 1e-8.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = "1e-8"


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def python_functions(path: pathlib.Path) -> None:
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        match = re.match(r"^(def|class) ([A-Za-z_][A-Za-z0-9_]*)", line)
        if not match:
            continue
        window = "\n".join(lines[i + 1 : i + 8])
        if '"""' not in window and "'''" not in window:
            fail(f"{path}: {match.group(2)} has no docstring")


def c_declarations(path: pathlib.Path) -> None:
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not re.search(r"\brgmin_[A-Za-z0-9_]+\s*\(", line):
            continue
        if line.strip().startswith("*") or line.strip().startswith("//"):
            continue
        if line.strip().startswith("typedef"):
            continue
        window = "\n".join(lines[max(0, i - 8) : i])
        if "/*" not in window and "*/" not in window:
            fail(f"{path}:{i + 1}: {line.strip()} has no comment")


def references() -> None:
    for name in ("rust.md", "c.md", "cpp.md", "python.md"):
        path = ROOT / "docs" / "reference" / name
        if not path.is_file():
            fail(f"missing {path}")
        if ORACLE not in path.read_text():
            fail(f"{path} does not name {ORACLE}")


def main() -> None:
    for path in (ROOT / "python" / "rgmin").glob("*.py"):
        python_functions(path)
    c_declarations(ROOT / "include" / "rgmin.h")
    header = (ROOT / "include" / "rgmin" / "optimize.hpp").read_text()
    if "EonCg" not in header or "PolakRibierePlus" not in header:
        fail("C++ Method is missing an enumerator")
    if "/**" not in header:
        fail("optimize.hpp has no doc comment")
    references()


if __name__ == "__main__":
    main()
