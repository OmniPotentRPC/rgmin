#!/usr/bin/env python3
"""Plain prose in the guide, the reference, and the audit."""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BANNED = re.compile(
    r"\b(delve|leverage|robust|seamless|underpin|end-to-end|shipping)\b|—"
)
PATHS = [
    ROOT / "docs" / "api-audit.md",
    ROOT / "README.md",
    * (ROOT / "docs" / "guide").glob("*.md"),
    * (ROOT / "docs" / "reference").glob("*.md"),
]


def main() -> None:
    bad = False
    for path in PATHS:
        text = path.read_text()
        tmp = "/" + "tmp"
        home = "/" + "home/"
        if tmp in text or home in text:
            print(f"{path}: contains a machine path", file=sys.stderr)
            bad = True
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("```"):
                continue
            if BANNED.search(line):
                print(f"{path}:{i}: {line.strip()}", file=sys.stderr)
                bad = True
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
