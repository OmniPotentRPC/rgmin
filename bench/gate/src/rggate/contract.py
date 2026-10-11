"""CSV contract version 1.

A file is major version 1 when it has no version marker or a marker
``# rggate-csv 1`` (a minor number is allowed: ``1.2``). Any other major
version is refused. Columns ``pot_s``, ``overhead_us_per_iter``,
``endpoint``, ``reference``, ``dim``, and ``threads`` are optional.
``reference`` is a path, or a numeric energy. ``endpoint`` is a
structure path when ``reference`` is absent or is a path. A numeric
``reference`` is an energy window, and ``endpoint`` is left unread
(the ECnoise bench stores the stop status there). ``threads`` is the
worker count of that
row, used only for the rank-invariance check.
"""

from dataclasses import dataclass
import csv
import io
import re

import numpy as np

_VERSION = re.compile(r"^#\s*rggate-csv\s+(\d+)(?:\.(\d+))?\s*$")
_REQUIRED = (
    "system",
    "method",
    "seed",
    "converged",
    "iters",
    "calls",
    "energy",
    "fmax",
    "wall_s",
)


class ContractError(ValueError):
    """The file is not contract version 1."""


@dataclass
class Row:
    system: str
    method: str
    seed: int
    converged: bool
    iters: int
    calls: float
    energy: float
    fmax: float
    wall_s: float
    pot_s: float | None
    overhead_us_per_iter: float | None
    endpoint: str | None
    reference: str | None
    dim: int | None
    threads: int | None
    line: int


def parse_csv(text):
    """Parse contract v1 text into rows. Blank lines are ignored."""
    header = None
    rows = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = _VERSION.match(line)
            if match is None:
                continue
            major = int(match.group(1))
            if major != 1:
                raise ContractError(
                    f"line {lineno}: rggate-csv major {major} is not 1"
                )
            continue
        fields = next(csv.reader(io.StringIO(raw)))
        if header is None:
            header = fields
            missing = [name for name in _REQUIRED if name not in header]
            if missing:
                raise ContractError("missing columns: " + ",".join(missing))
            continue
        if len(fields) != len(header):
            raise ContractError(
                f"line {lineno}: expected {len(header)} columns, got {len(fields)}"
            )
        rows.append(_row(dict(zip(header, fields)), lineno))
    if header is None:
        raise ContractError("csv has no header")
    return rows


def load_csv(path):
    """Read a contract v1 file from disk."""
    with open(path, encoding="utf-8") as handle:
        return parse_csv(handle.read())


def _row(record, line):
    def optional_float(name):
        raw = (record.get(name) or "").strip()
        if raw == "":
            return None
        return float(raw)

    def optional_str(name):
        raw = (record.get(name) or "").strip()
        return raw or None

    converged = record["converged"].strip()
    if converged not in ("0", "1"):
        raise ContractError(f"line {line}: converged must be 0 or 1")
    dim_raw = (record.get("dim") or "").strip()
    dim = int(dim_raw) if dim_raw else None
    threads_raw = (record.get("threads") or "").strip()
    threads = int(threads_raw) if threads_raw else None
    return Row(
        system=record["system"].strip(),
        method=record["method"].strip(),
        seed=int(record["seed"]),
        converged=converged == "1",
        iters=int(record["iters"]),
        calls=float(record["calls"]),
        energy=float(record["energy"]),
        fmax=float(record["fmax"]),
        wall_s=float(record["wall_s"]),
        pot_s=optional_float("pot_s"),
        overhead_us_per_iter=optional_float("overhead_us_per_iter"),
        endpoint=optional_str("endpoint"),
        reference=optional_str("reference"),
        dim=dim,
        threads=threads,
        line=line,
    )


def load_structure(path):
    """Read an endpoint file.

    ``energy <float>`` and ``cell a11 a12 ... a33`` are optional headers.
    Remaining lines are ``x y z``. A zero cell means a free cluster.
    """
    energy = None
    cell = None
    coords = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "energy" and len(parts) == 2:
                energy = float(parts[1])
                continue
            if parts[0] == "cell" and len(parts) == 10:
                cell = np.array([float(v) for v in parts[1:]], dtype=float).reshape(3, 3)
                continue
            if len(parts) < 3:
                raise ContractError(f"{path}: expected x y z, got {line!r}")
            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not coords:
        raise ContractError(f"{path}: no coordinates")
    positions = np.asarray(coords, dtype=float)
    if cell is not None and not np.any(cell):
        cell = None
    return energy, cell, positions
