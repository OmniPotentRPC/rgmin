#!/usr/bin/env python3
"""Reproduce tests/data/gromacs_steep_argon.txt with gmx_d."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


START = """TITLE
argon seven-cluster
END
POSITION
    1 AR    AR      1    5.000000000    5.000000000    5.000000000
    2 AR    AR      2    5.381637096    5.000000000    4.950000000
    3 AR    AR      3    5.190818548    5.330507421    5.050000000
    4 AR    AR      4    4.809181452    5.330507421    4.950000000
    5 AR    AR      5    4.618362904    5.000000000    5.050000000
    6 AR    AR      6    4.809181452    4.669492579    4.950000000
    7 AR    AR      7    5.190818548    4.669492579    5.050000000
END
BOX
   10.000000000   10.000000000   10.000000000
END
"""


def run() -> None:
    here = Path(__file__).resolve().parent
    mdp = here / "gromacs_steep_argon.mdp"
    top = here / "gromacs_steep_argon.top"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "start.g96").write_text(START, encoding="ascii")
        subprocess.run(
            [
                "gmx_d",
                "-quiet",
                "grompp",
                "-f",
                str(mdp),
                "-p",
                str(top),
                "-c",
                "start.g96",
                "-o",
                "em.tpr",
                "-maxwarn",
                "1",
            ],
            cwd=work,
            check=True,
        )
        mdrun = subprocess.run(
            ["gmx_d", "-quiet", "mdrun", "-v", "-s", "em.tpr", "-nt", "1", "-ntmpi", "1", "-c", "end.g96"],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        )
        print(mdrun.stderr)
        print((work / "end.g96").read_text(encoding="ascii"))


if __name__ == "__main__":
    run()
