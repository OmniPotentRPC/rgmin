"""Local minimization timings and memray peaks.

Each call goes through the installed ``librgmin_asv.so``, which calls
``rgmin_minimize`` in the installed ``librgmin.so``. A file left in the
checkout is not used.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path

_LIB = None
_RECORDED = False
_LBFGS = 3
_FIRE = 17


def _record(product: Path, helper: Path) -> None:
    """Write both checksums once per process.

    asv calls setup for every repeat inside a sample. Each round is its
    own process, so one line remains for that round. The first path is
    ``librgmin.so``, the library the minimize call runs.
    """
    global _RECORDED
    if _RECORDED:
        return
    _RECORDED = True
    env = os.environ.get("ASV_ENV_DIR")
    if not env:
        return
    commit = os.environ.get("ASV_COMMIT", "")
    product_sha = hashlib.sha256(product.read_bytes()).hexdigest()
    helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
    line = f"{commit}\t{product}\t{product_sha}\t{helper}\t{helper_sha}\n"
    with (Path(env) / "lib-load.log").open("a", encoding="utf-8") as handle:
        handle.write(line)


def _installed() -> tuple[Path, Path]:
    env = os.environ.get("ASV_ENV_DIR")
    if not env:
        raise RuntimeError("ASV_ENV_DIR is unset; refusing a checkout library")
    lib = Path(env) / "lib"
    product = lib / "librgmin.so"
    helper = lib / "librgmin_asv.so"
    if not product.is_file():
        raise RuntimeError(f"installed librgmin.so is missing: {product}")
    if not helper.is_file():
        raise RuntimeError(f"installed librgmin_asv.so is missing: {helper}")
    return product, helper


def _library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    product, helper = _installed()
    _record(product, helper)
    lib = ctypes.CDLL(str(helper))
    lib.rgmin_asv_quadratic.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.rgmin_asv_quadratic.restype = ctypes.c_int
    _LIB = lib
    return lib


def _code(method: str) -> int:
    if method == "lbfgs":
        return _LBFGS
    if method == "fire":
        return _FIRE
    raise ValueError(method)


class QuadraticMinimize:
    """Wall time of one C minimize on a convex quadratic.

    ``number`` repeats that call inside one sample. asv divides the
    sample back to one call. Each call starts from the same numeric line.
    """

    params = ([16, 64], ["lbfgs", "fire"])
    param_names = ["n", "method"]
    rounds = 10
    repeat = 1
    number = 2000
    sample_time = 0.05
    warmup_time = 1.0
    min_run_count = 1
    timeout = 600

    def setup(self, n: int, method: str) -> None:
        _library()

    def time_minimize(self, n: int, method: str) -> None:
        if _library().rgmin_asv_quadratic(int(n), _code(method)) != 0:
            raise RuntimeError("minimize failed")


class QuadraticMemory:
    """Peak bytes from asv_bench_memray (methods named ray_*)."""

    params = ([16, 64], ["lbfgs", "fire"])
    param_names = ["n", "method"]
    rounds = 2
    number = 1
    repeat = 1
    timeout = 180

    def setup(self, n: int, method: str) -> None:
        _library()

    def ray_minimize(self, n: int, method: str) -> None:
        if _library().rgmin_asv_quadratic(int(n), _code(method)) != 0:
            raise RuntimeError("minimize failed")
