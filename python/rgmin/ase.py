# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""ASE :class:`~ase.optimize.optimize.Optimizer` backed by an rgmin solver.

ASE is imported when this module is imported. ``import rgmin`` does not
import ASE.
"""

from __future__ import annotations

import inspect
import time
from typing import Any, cast

import numpy as np
from ase.optimize.optimize import Optimizer as _ASEOptimizer

import rgmin

_ASE_KEYS = frozenset({"master", "comm", "loginterval", "force_consistent"})

_STEP_ACCEPT_TYPES = frozenset(
    {
        "NEB",
        "DyNEB",
        "NEBOptimizable",
        "DimerOptimizable",
        "MinModeAtoms",
    }
)


def _ase_passthrough(options: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(_ASEOptimizer.__init__).parameters
    passed: dict[str, Any] = {}
    for key in list(options):
        if key not in _ASE_KEYS:
            continue
        if key not in params:
            raise TypeError(f"unexpected keyword argument {key!r}")
        passed[key] = options.pop(key)
    return passed


def _positions(opt: Any) -> np.ndarray:
    if hasattr(opt, "get_x"):
        return np.asarray(opt.get_x(), dtype=np.float64).reshape(-1)
    return np.asarray(opt.get_positions(), dtype=np.float64).reshape(-1)


def _write_positions(opt: Any, x: np.ndarray) -> None:
    if hasattr(opt, "set_x"):
        opt.set_x(np.asarray(x, dtype=np.float64))
        return
    current = np.asarray(opt.get_positions(), dtype=np.float64)
    opt.set_positions(np.asarray(x, dtype=np.float64).reshape(current.shape))


def _value(opt: Any) -> float:
    target = cast(Any, opt)
    for name in ("get_value", "get_potential_energy"):
        getter = getattr(target, name, None)
        if getter is None:
            continue
        try:
            return float(getter())
        except TypeError:
            break
    return float(cast(Any, target.atoms).get_potential_energy())


def _gradient(opt: Any) -> np.ndarray:
    if hasattr(opt, "get_gradient"):
        return np.asarray(opt.get_gradient(), dtype=np.float64).reshape(-1)
    forces = np.asarray(opt.get_forces(), dtype=np.float64)
    return -forces.reshape(-1)


def _ndofs(opt: Any) -> int:
    if hasattr(opt, "ndofs"):
        return int(opt.ndofs())
    return int(_positions(opt).shape[0])


def _nonconservative(opt: Any) -> bool:
    name = type(opt).__name__
    if name in _STEP_ACCEPT_TYPES:
        return True
    lowered = name.lower()
    return "neb" in lowered or "dimer" in lowered or "minmode" in lowered


class Optimizer(_ASEOptimizer):
    """One rgmin iteration per :meth:`step`.

    ``method`` selects the solver. ``maxstep`` is the per-atom cap in
    angstroms (``rgmin_solver_set_atom_maxmove``). ``accept`` overrides
    the acceptance rule. Remaining solver keywords are validated by
    :class:`rgmin.SolverConfig`. ``master``, ``comm``, and ``loginterval``
    are forwarded to the ASE base class.

    A non-conservative optimizable (NEB, dimer, min-mode) uses
    ``accept="step"`` and the L-BFGS NEB guards. A plain atoms object or
    cell filter keeps energy acceptance.
    """

    defaults = {"maxstep": 0.2}

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        method: str = "lbfgs",
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        self._rgmin_method = method
        self._rgmin_accept = accept
        self._rgmin_options = dict(options)
        ase_options = _ase_passthrough(self._rgmin_options)
        if maxstep is None:
            maxstep = float(self.defaults["maxstep"])
        self.maxstep = float(maxstep)
        self._solver: rgmin.Solver | None = None
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            **ase_options,
        )

    def _config_options(self) -> dict[str, Any]:
        opts = dict(self._rgmin_options)
        opts["maxstep"] = self.maxstep
        if self._rgmin_accept is not None:
            opts["accept"] = self._rgmin_accept
        elif _nonconservative(self.optimizable):
            opts["accept"] = "step"
            opts.setdefault("neb_guards", True)
        return opts

    def initialize(self) -> None:
        opts = self._config_options()
        config = rgmin.SolverConfig(self._rgmin_method, **opts)
        self._solver = rgmin.Solver(_ndofs(self.optimizable), config)

    def _evaluate(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        current = _positions(self.optimizable)
        if x.shape != current.shape or not np.array_equal(x, current):
            _write_positions(self.optimizable, x)
        value = _value(self.optimizable)
        gradient = np.array(_gradient(self.optimizable), dtype=np.float64, copy=True)
        return value, gradient

    def step(self) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self._solver is None or self._solver.dim != _ndofs(self.optimizable):
            self.initialize()
        assert self._solver is not None
        x = np.array(_positions(self.optimizable), dtype=np.float64, copy=True)
        self._solver.step(self._evaluate, x)
        if not np.array_equal(x, _positions(self.optimizable)):
            _write_positions(self.optimizable, x)

    def _log_energy(self) -> float:
        opt = cast(Any, self.optimizable)
        for name in ("get_value", "get_potential_energy"):
            getter = getattr(opt, name, None)
            if getter is None:
                continue
            try:
                return float(getter())
            except TypeError:
                break
        return float(cast(Any, self.atoms).get_potential_energy())

    def _log_fmax(self, gradient: Any, forces: Any) -> float:
        if gradient is not None and hasattr(self.optimizable, "gradient_norm"):
            return float(self.optimizable.gradient_norm(gradient))
        vector = gradient if gradient is not None else forces
        if vector is None:
            if hasattr(self.optimizable, "get_gradient"):
                vector = self.optimizable.get_gradient()
            else:
                vector = cast(Any, self.optimizable).get_forces()
        block = np.asarray(vector, dtype=np.float64).reshape(-1, 3)
        return float(np.sqrt((block * block).sum(axis=1).max()))

    def log(self, *args: Any, **kwargs: Any) -> None:
        """Write one ASE optimizer line.

        The columns are ``Step``, ``Time``, ``Energy``, and ``fmax``, in the
        format used by :class:`ase.optimize.optimize.Optimizer`. Energy is
        read without ``force_consistent`` when the optimizable does not
        accept that keyword (``MinModeAtoms``).
        """
        if self.logfile is None:
            return
        gradient = args[0] if args else kwargs.get("gradient")
        forces = kwargs.get("forces")
        if gradient is None and forces is None and not args:
            if hasattr(self.optimizable, "get_gradient"):
                gradient = self.optimizable.get_gradient()
            elif hasattr(self.optimizable, "get_forces"):
                forces = cast(Any, self.optimizable).get_forces()
        fmax = self._log_fmax(gradient, forces)
        energy = self._log_energy()
        now = time.localtime()
        name = type(self).__name__
        write = cast(Any, self.logfile).write
        if self.nsteps == 0:
            header = "%s  %4s %8s %15s  %12s\n" % (
                " " * len(name),
                "Step",
                "Time",
                "Energy",
                "fmax",
            )
            write(header)
        line = "%s:  %3d %02d:%02d:%02d %15.6f %15.6f\n" % (
            name,
            self.nsteps,
            now.tm_hour,
            now.tm_min,
            now.tm_sec,
            energy,
            fmax,
        )
        write(line)

    def read(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} cannot restore {self._rgmin_method} solver state"
        )

    def dump(self, data: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} cannot store {self._rgmin_method} solver state"
        )

    def todict(self) -> dict[str, Any]:
        description = dict(cast(Any, super()).todict())
        description["method"] = self._rgmin_method
        description.update(self._rgmin_options)
        if self._rgmin_accept is not None:
            description["accept"] = self._rgmin_accept
        return description


def _bind(name: str, method: str, doc: str) -> type[Optimizer]:
    def __init__(
        self: Optimizer,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        **options: Any,
    ) -> None:
        Optimizer.__init__(
            self,
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            method=method,
            maxstep=maxstep,
            accept=accept,
            **options,
        )

    __init__.__doc__ = doc
    cls = type(name, (Optimizer,), {"__init__": __init__, "__doc__": doc})
    return cls


class CG(Optimizer):
    """Nonlinear conjugate gradient.

    The default ``beta`` is Polak-Ribiere, the C ABI method
    ``RGMIN_POLAK_RIBIERE``. ``beta`` selects another conjugacy variant
    exposed by that ABI (``fletcher_reeves``, ``hestenes_stiefel``,
    ``dai_yuan``, ``conjugate_descent``, ``hager_zhang``, ``liu_storey``,
    ``fr_pr``).

    Nocedal and Wright, *Numerical Optimization*,
    https://doi.org/10.1007/978-0-387-40065-5.
    Polak and Ribiere, *Note sur la convergence de methodes de directions
    conjuguees* (1969).
    """

    def __init__(
        self,
        atoms: Any,
        restart: str | None = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        accept: str | None = None,
        beta: str = "polak_ribiere",
        **options: Any,
    ) -> None:
        options["beta"] = beta
        Optimizer.__init__(
            self,
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            method="cg",
            maxstep=maxstep,
            accept=accept,
            **options,
        )


LBFGS = _bind(
    "LBFGS",
    "lbfgs",
    "Limited-memory BFGS (Nocedal-Wright 7.4).\n\n"
    "Liu and Nocedal, Mathematical Programming 45, 503 (1989),\n"
    "https://doi.org/10.1007/BF01589116.\n"
    "Nocedal and Wright, *Numerical Optimization*,\n"
    "https://doi.org/10.1007/978-0-387-40065-5.",
)
FIRE = _bind(
    "FIRE",
    "fire",
    "FIRE inertial relaxation.\n\n"
    "Bitzek, Koskinen, Gaehler, Moseler, and Gumbsch, Phys. Rev. Lett. 97,\n"
    "170201 (2006), https://doi.org/10.1103/PhysRevLett.97.170201.",
)
FIRE2 = _bind(
    "FIRE2",
    "fire2",
    "FIRE 2.0.\n\n"
    "Guenole, Noehring, Vaid, Houlle, Xie, Prakash, and Bitzek, Comput.\n"
    "Mater. Sci. 175, 109584 (2020),\n"
    "https://doi.org/10.1016/j.commatsci.2020.109584.",
)
QuickMin = _bind(
    "QuickMin",
    "quickmin",
    "Quick-min: project the velocity onto the force, then an Euler step.",
)
BFGS = _bind(
    "BFGS",
    "bfgs",
    "Dense BFGS.\n\n"
    "Nocedal and Wright, *Numerical Optimization*,\n"
    "https://doi.org/10.1007/978-0-387-40065-5.",
)
SD = _bind(
    "SD",
    "sd",
    "Steepest descent.\n\n"
    "Nocedal and Wright, *Numerical Optimization*,\n"
    "https://doi.org/10.1007/978-0-387-40065-5.",
)
Adam = _bind(
    "Adam",
    "adam",
    "Adam.\n\nKingma and Ba, https://doi.org/10.48550/arXiv.1412.6980.",
)
Newton = _bind(
    "Newton",
    "newton",
    "Newton on a caller Hessian. The ASE path has no Hessian callback,\n"
    "so a step raises ``NotImplementedError`` until a Hessian is supplied\n"
    "through the C API.\n\n"
    "Nocedal and Wright, *Numerical Optimization*,\n"
    "https://doi.org/10.1007/978-0-387-40065-5.",
)
RFO = _bind(
    "RFO",
    "rfo",
    "Rational function optimization on a caller Hessian.\n\n"
    "Banerjee, Adams, Simons, and Shepard, J. Phys. Chem. 89, 52 (1985),\n"
    "https://doi.org/10.1021/j100247a015.",
)
SR1 = _bind(
    "SR1",
    "sr1",
    "Symmetric-rank-one quasi-Newton.\n\n"
    "Nocedal and Wright, *Numerical Optimization*,\n"
    "https://doi.org/10.1007/978-0-387-40065-5.",
)
SR2 = _bind(
    "SR2",
    "sr2",
    "SR2 quasi-Newton update.\n\n"
    "Nocedal and Wright, *Numerical Optimization*,\n"
    "https://doi.org/10.1007/978-0-387-40065-5.",
)
BB = _bind(
    "BB",
    "bb",
    "Barzilai-Borwein spectral steepest descent.\n\n"
    "Barzilai and Borwein, IMA J. Numer. Anal. 8, 141 (1988),\n"
    "https://doi.org/10.1093/imanum/8.1.141.",
)
Dogleg = _bind(
    "Dogleg",
    "dogleg",
    "Powell dogleg on a caller Hessian.\n\n"
    "Nocedal and Wright, algorithm 4.1,\n"
    "https://doi.org/10.1007/978-0-387-40065-5.",
)
DIIS = _bind(
    "DIIS",
    "diis",
    "Pulay residual subspace extrapolation.\n\n"
    "Pulay, Chem. Phys. Lett. 73, 393 (1980),\n"
    "https://doi.org/10.1016/0009-2614(80)80396-4.",
)
PSO = _bind(
    "PSO",
    "pso",
    "Particle swarm.\n\n"
    "Kennedy and Eberhart, Proc. ICNN (1995),\n"
    "https://doi.org/10.1109/ICNN.1995.488968.",
)
FletcherReeves = _bind(
    "FletcherReeves",
    "fletcher_reeves",
    "Fletcher-Reeves conjugate gradient.\n\n"
    "Fletcher and Reeves, Comput. J. 7, 149 (1964),\n"
    "https://doi.org/10.1093/comjnl/7.2.149.",
)
HestenesStiefel = _bind(
    "HestenesStiefel",
    "hestenes_stiefel",
    "Hestenes-Stiefel conjugate gradient.\n\n"
    "Hestenes and Stiefel, J. Res. Natl. Bur. Stand. 49, 409 (1952).",
)
DaiYuan = _bind(
    "DaiYuan",
    "dai_yuan",
    "Dai-Yuan conjugate gradient.\n\n"
    "Dai and Yuan, SIAM J. Optim. 10, 177 (1999),\n"
    "https://doi.org/10.1137/S1052623497318992.",
)
ConjugateDescent = _bind(
    "ConjugateDescent",
    "conjugate_descent",
    "Fletcher conjugate descent.\n\n"
    "Hager and Zhang, SIAM J. Optim. 16, 170 (2005),\n"
    "https://doi.org/10.1137/030601880.",
)
HagerZhang = _bind(
    "HagerZhang",
    "hager_zhang",
    "Hager-Zhang conjugate gradient.\n\n"
    "Hager and Zhang, SIAM J. Optim. 16, 170 (2005),\n"
    "https://doi.org/10.1137/030601880.",
)
LiuStorey = _bind(
    "LiuStorey",
    "liu_storey",
    "Liu-Storey conjugate gradient.\n\n"
    "Liu and Storey, J. Optim. Theory Appl. 69, 129 (1991),\n"
    "https://doi.org/10.1007/BF00940464.",
)
FrPr = _bind(
    "FrPr",
    "fr_pr",
    "Gilbert-Nocedal FR-PR hybrid.\n\n"
    "Nocedal and Wright, *Numerical Optimization*,\n"
    "https://doi.org/10.1007/978-0-387-40065-5.",
)

__all__ = [
    "Adam",
    "BB",
    "BFGS",
    "CG",
    "ConjugateDescent",
    "DIIS",
    "DaiYuan",
    "Dogleg",
    "FIRE",
    "FIRE2",
    "FletcherReeves",
    "FrPr",
    "HagerZhang",
    "HestenesStiefel",
    "LBFGS",
    "LiuStorey",
    "Newton",
    "Optimizer",
    "PSO",
    "QuickMin",
    "RFO",
    "SD",
    "SR1",
    "SR2",
]
