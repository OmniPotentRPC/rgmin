#!/usr/bin/env python3
"""Record meson build options, or remove the installed libraries.

The build tree lives in the commit cache. ``meson install`` is the
install. This file does not copy libraries.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def record(build: Path, dest: Path) -> None:
    meson = shutil.which("meson")
    if meson is None:
        raise SystemExit("meson was not found on PATH")
    text = subprocess.check_output(
        [meson, "introspect", "--buildoptions", str(build)],
        text=True,
    )
    dest.write_text(text, encoding="utf-8")
    values = {item["name"]: item["value"] for item in json.loads(text)}
    buildtype = values.get("buildtype")
    if buildtype != "release":
        raise SystemExit(f"meson buildtype is {buildtype}")
    print(text, end="")


def uninstall(env_dir: Path) -> None:
    lib = env_dir / "lib"
    for name in ("librgmin.so", "librgmin_asv.so"):
        path = lib / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "uninstall":
        uninstall(Path(argv[2]))
        return 0
    if len(argv) != 3:
        raise SystemExit(
            "usage: record_options.py BUILD_DIR DEST | uninstall ENV_DIR"
        )
    record(Path(argv[1]), Path(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
