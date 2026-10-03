#!/usr/bin/env python3
"""Mint dest golden-master fixtures from zadorlab/sella and manopt.

Set SELLA_ROOT to a zadorlab/sella checkout (the directory that contains
the `sella/` package). Remint loads optimize/stepper.py,
optimize/restricted_step.py (RestrictedAtomicStep.cons), and
hessian_update.py. Optional MANOPT_ROOT is a NicolasBoumal/manopt
tree; when MATLAB/Octave can see it, factory numbers come from that
source. Otherwise the remint writes the published manopt formulas
(sphere / positive / symmetric / complexcircle / multinomial /
centeredmatrix / sympositivedefinite proj, retr, transp) dest comments
cite.

stdin is unused. stdout is one JSON object:

  source: {sella_root, sella_file, sella_files, manopt, manopt_files?}
  cases:  [{name, kind, s}, ...]

kind is rfo / qn / qn_irc / prfo / ts_bfgs / ras plus factory
proj/retr/transp for Sphere, Positive, Symmetric, ComplexCircle,
Multinomial, CenteredMatrix, and SPD.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# NicolasBoumal/manopt factory sources dest comments cite.
_MANOPT_FACTORY_RELS = (
    "manopt/manifolds/sphere/spherefactory.m",
    "manopt/manifolds/positive/positivefactory.m",
    "manopt/manifolds/euclidean/symmetricfactory.m",
    "manopt/manifolds/complexcircle/complexcirclefactory.m",
    "manopt/manifolds/multinomial/multinomialfactory.m",
    "manopt/manifolds/euclidean/centeredmatrixfactory.m",
    "manopt/manifolds/symfixedrank/sympositivedefinitefactory.m",
)


def _sella_root() -> Path:
    raw = os.environ.get("SELLA_ROOT", "")
    if raw:
        return Path(raw).resolve()
    here = Path(__file__).resolve()
    candidates = [
        here.parents[4] / "Python" / "sella",
        here.parents[3] / "Python" / "sella",
    ]
    for c in candidates:
        if (c / "sella" / "optimize" / "stepper.py").is_file() and (
            c / "sella" / "optimize" / "restricted_step.py"
        ).is_file() and (c / "sella" / "hessian_update.py").is_file():
            return c
    raise SystemExit("SELLA_ROOT not set and no local zadorlab/sella checkout")


def _hess(dim: int, b0: np.ndarray):
    from sella.linalg import ApproximateHessian

    return ApproximateHessian(dim, dim, np.asarray(b0, dtype=float), initialized=True)


def _load_sella_modules(root: Path):
    """Import stepper, restricted_step, hessian_update without sella/__init__.py."""
    import importlib.util
    import types

    if "scipy" not in sys.modules:
        scipy = types.ModuleType("scipy")
        linalg_mod = types.ModuleType("scipy.linalg")

        def _lstsq(a, b, rcond=None):
            x, residuals, rank, singular = np.linalg.lstsq(a, b, rcond=rcond)
            return x, residuals, rank, singular

        linalg_mod.eigh = np.linalg.eigh
        linalg_mod.lstsq = _lstsq
        linalg_mod.solve = np.linalg.solve
        scipy.linalg = linalg_mod
        sys.modules["scipy"] = scipy
        sys.modules["scipy.linalg"] = linalg_mod

    pkg = types.ModuleType("sella")
    pkg.__path__ = [str(root / "sella")]
    sys.modules["sella"] = pkg

    gpu = types.ModuleType("sella._gpu")
    gpu.torch = None

    def _gpu_ok(_n):
        return False

    gpu._gpu_ok = _gpu_ok
    sys.modules["sella._gpu"] = gpu
    pkg._gpu = gpu

    class ApproximateHessian:
        def __init__(self, dim, ncart, B0=None, update_method="TS-BFGS",
                     symm=2, initialized=False):
            self.dim = dim
            self.ncart = ncart
            self.update_method = update_method
            self.symm = symm
            self.initialized = initialized
            self.B = None if B0 is None else np.asarray(B0, dtype=float)
            if self.B is not None:
                self.evals, self.evecs = np.linalg.eigh(self.B)
            else:
                self.evals = None
                self.evecs = None

        def asarray(self):
            return self.B

        def project(self, u):
            bproj = u.T @ self.B @ u
            return ApproximateHessian(bproj.shape[0], 0, bproj, initialized=True)

    linalg = types.ModuleType("sella.linalg")
    linalg.ApproximateHessian = ApproximateHessian
    sys.modules["sella.linalg"] = linalg
    pkg.linalg = linalg

    hess_spec = importlib.util.spec_from_file_location(
        "sella.hessian_update", root / "sella" / "hessian_update.py"
    )
    hess = importlib.util.module_from_spec(hess_spec)
    sys.modules["sella.hessian_update"] = hess
    hess_spec.loader.exec_module(hess)

    opt = types.ModuleType("sella.optimize")
    opt.__path__ = [str(root / "sella" / "optimize")]
    sys.modules["sella.optimize"] = opt

    step_spec = importlib.util.spec_from_file_location(
        "sella.optimize.stepper", root / "sella" / "optimize" / "stepper.py"
    )
    stepper = importlib.util.module_from_spec(step_spec)
    sys.modules["sella.optimize.stepper"] = stepper
    step_spec.loader.exec_module(stepper)

    # restricted_step.py imports PES only for RestrictedAtomicStep.__init__.
    # dest remint calls RestrictedAtomicStep.cons, so stub peswrapper.
    pes = types.ModuleType("sella.peswrapper")
    pes.PES = type("PES", (), {})
    pes.InternalPES = type("InternalPES", (), {})
    sys.modules["sella.peswrapper"] = pes
    pkg.peswrapper = pes

    ras_path = root / "sella" / "optimize" / "restricted_step.py"
    if not ras_path.is_file():
        raise SystemExit(f"SELLA_ROOT missing {ras_path}")
    ras_spec = importlib.util.spec_from_file_location(
        "sella.optimize.restricted_step", ras_path
    )
    ras = importlib.util.module_from_spec(ras_spec)
    sys.modules["sella.optimize.restricted_step"] = ras
    ras_spec.loader.exec_module(ras)
    if Path(getattr(ras, "__file__", "")).name != "restricted_step.py":
        raise SystemExit("failed to load sella/optimize/restricted_step.py")
    if not hasattr(ras, "RestrictedAtomicStep"):
        raise SystemExit("RestrictedAtomicStep missing from restricted_step.py")
    return stepper, hess, ras


def mint_sella(root: Path) -> tuple[dict, list[dict]]:
    stepper, hess, ras = _load_sella_modules(root)
    update_H = hess.update_H
    RestrictedAtomicStep = ras.RestrictedAtomicStep
    PartitionedRationalFunctionOptimization = (
        stepper.PartitionedRationalFunctionOptimization
    )
    QuasiNewton = stepper.QuasiNewton
    QuasiNewtonIRC = stepper.QuasiNewtonIRC
    RationalFunctionOptimization = stepper.RationalFunctionOptimization

    cases: list[dict] = []

    h_eye = _hess(2, np.eye(2))
    g_down = np.array([2.0, 0.0])
    rfo = RationalFunctionOptimization(g_down, h_eye, order=0)
    for alpha in (0.25, 0.5, 1.0):
        s, _ = rfo.get_s(alpha)
        cases.append(
            {
                "name": f"rfo_eye_order0_alpha{alpha}",
                "kind": "rfo",
                "order": 0,
                "alpha": alpha,
                "H": np.eye(2).tolist(),
                "g": g_down.tolist(),
                "s": [float(v) for v in s],
            }
        )

    h_sad = np.array([[-1.0, 0.2], [0.2, 4.0]])
    g_sad = np.array([0.5, 0.4])
    rfo1 = RationalFunctionOptimization(g_sad, _hess(2, h_sad), order=1)
    s, _ = rfo1.get_s(1.0)
    cases.append(
        {
            "name": "rfo_saddle_order1_alpha1",
            "kind": "rfo",
            "order": 1,
            "alpha": 1.0,
            "H": h_sad.tolist(),
            "g": g_sad.tolist(),
            "s": [float(v) for v in s],
        }
    )

    qn = QuasiNewton(g_sad, _hess(2, h_sad), order=1)
    for alpha in (0.0, 0.3, 1.0):
        s, _ = qn.get_s(alpha)
        cases.append(
            {
                "name": f"qn_saddle_order1_alpha{alpha}",
                "kind": "qn",
                "order": 1,
                "alpha": alpha,
                "H": h_sad.tolist(),
                "g": g_sad.tolist(),
                "s": [float(v) for v in s],
            }
        )

    d1 = np.array([-0.05, 0.0])
    qnirc = QuasiNewtonIRC(g_sad, _hess(2, h_sad), order=1, d1=d1)
    s, _ = qnirc.get_s(0.2)
    cases.append(
        {
            "name": "qn_irc_order1_alpha0.2",
            "kind": "qn_irc",
            "order": 1,
            "alpha": 0.2,
            "H": h_sad.tolist(),
            "g": g_sad.tolist(),
            "d1": d1.tolist(),
            "s": [float(v) for v in s],
        }
    )

    prfo = PartitionedRationalFunctionOptimization(g_sad, _hess(2, h_sad), order=1)
    s, _ = prfo.get_s(1.0)
    cases.append(
        {
            "name": "prfo_saddle_order1_alpha1",
            "kind": "prfo",
            "order": 1,
            "alpha": 1.0,
            "H": h_sad.tolist(),
            "g": g_sad.tolist(),
            "s": [float(v) for v in s],
        }
    )

    b0 = np.diag([-1.0, 4.0])
    step = np.array([0.15, 0.05])
    y = np.array([0.35, -0.10])
    bplus = update_H(b0.copy(), step, y, method="TS-BFGS", symm=None)
    cases.append(
        {
            "name": "ts_bfgs_keep_saddle",
            "kind": "ts_bfgs",
            "B": b0.tolist(),
            "step": step.tolist(),
            "y": y.tolist(),
            "s": [float(v) for v in np.asarray(bplus).ravel()],
        }
    )

    # RestrictedAtomicStep.cons from restricted_step.py: max per-atom
    # Euclidean. dest ras_clip scales the Cartesian step by delta/cons.
    s_cart = np.array([0.4, 0.3, 0.0, 0.05, 0.0, 0.0])
    delta = 0.1
    val = RestrictedAtomicStep.cons(None, s_cart)
    scale = delta / val if val > delta else 1.0
    cases.append(
        {
            "name": "ras_clip_max_atom",
            "kind": "ras",
            "delta": delta,
            "step": s_cart.tolist(),
            "s": [float(v) for v in (s_cart * scale)],
        }
    )

    sella_files = [
        "sella/optimize/stepper.py",
        "sella/optimize/restricted_step.py",
        "sella/hessian_update.py",
    ]
    meta = {
        "sella_root": "SELLA_ROOT",
        "sella_file": "sella/optimize/stepper.py",
        "sella_files": sella_files,
        "sella_exists": all((root / rel).is_file() for rel in sella_files),
    }
    return meta, cases


def _manopt_root() -> Path | None:
    raw = os.environ.get("MANOPT_ROOT", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _manopt_clone_root(root: Path) -> Path:
    """Accept the NicolasBoumal/manopt clone or its inner manopt/ directory."""
    if (root / "manopt" / "manifolds" / "sphere" / "spherefactory.m").is_file():
        return root
    if (root / "manifolds" / "sphere" / "spherefactory.m").is_file():
        return root.parent if root.name == "manopt" else root
    raise SystemExit(f"MANOPT_ROOT={root} is not a NicolasBoumal/manopt tree")


def _locate_manopt_factories(root: Path) -> list[Path]:
    """Open the seven dest factory files under MANOPT_ROOT."""
    clone = _manopt_clone_root(root)
    found: list[Path] = []
    for rel in _MANOPT_FACTORY_RELS:
        path = clone / rel
        if not path.is_file() and rel.startswith("manopt/"):
            path = clone / rel[len("manopt/") :]
        if not path.is_file():
            hits = sorted(root.rglob(Path(rel).name))
            if hits:
                path = hits[0]
        if not path.is_file():
            raise SystemExit(f"MANOPT_ROOT={root} missing {Path(rel).name}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if "function M =" not in text:
            raise SystemExit(f"{path} is not a manopt factory")
        found.append(path)
    return found


def _rel_manopt_file(root: Path, path: Path) -> str:
    clone = _manopt_clone_root(root)
    resolved = path.resolve()
    for base in (clone, root, clone / "manopt"):
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return path.name


def _as_f(vals) -> list[float]:
    return [float(a) for a in np.asarray(vals, dtype=float).ravel()]


def _case(name: str, kind: str, x, v, s, y=None) -> dict:
    out = {
        "name": name,
        "kind": kind,
        "x": _as_f(x),
        "v": _as_f(v),
        "s": _as_f(s),
    }
    if y is not None:
        out["y"] = _as_f(y)
    return out


def _sphere_proj(x, v):
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    return v - x * np.dot(x, v)


def _sphere_retr(x, v):
    y = np.asarray(x, dtype=float) + np.asarray(v, dtype=float)
    return y / np.linalg.norm(y)


def _multisym(a):
    a = np.asarray(a, dtype=float)
    return 0.5 * (a + a.T)


def _cc_pairs(z):
    z = np.asarray(z, dtype=float)
    return z.reshape(-1, 2)


def _cc_pack(pairs):
    return np.asarray(pairs, dtype=float).reshape(-1)


def _cc_proj(z, u):
    # manopt: u - real(conj(u).*z).*z on each unit-modulus entry.
    zp = _cc_pairs(z)
    up = _cc_pairs(u)
    dots = np.sum(zp * up, axis=1, keepdims=True)
    return _cc_pack(up - dots * zp)


def _cc_retr(z, v):
    # manopt: sign(z+v) = normalize each complex entry.
    yp = _cc_pairs(z) + _cc_pairs(v)
    nrm = np.linalg.norm(yp, axis=1, keepdims=True)
    return _cc_pack(yp / nrm)


def _mn_proj(x, v):
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    return v - np.sum(v) * x


def _mn_retr(x, v):
    # manopt multinomialfactory first-order retr:
    # Y = X.*exp(eta./X); Y = Y./sum(Y); Y = max(Y, eps)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    y = x * np.exp(v / x)
    y = y / np.sum(y)
    return np.maximum(y, np.finfo(float).eps)


def _center_cols(a, m, n):
    # manopt centeredmatrixfactory default 'cols': X*ones=0, subtract row means.
    mat = np.asarray(a, dtype=float).reshape(m, n)
    return (mat - mat.mean(axis=1, keepdims=True)).ravel()


def _spd_retr(x, v):
    # manopt: Y = symm(X + U + 0.5*U*(X\U))
    n = int(np.sqrt(len(x)))
    xmat = np.asarray(x, dtype=float).reshape(n, n)
    umat = np.asarray(v, dtype=float).reshape(n, n)
    mid = umat @ np.linalg.solve(xmat, umat)
    return _multisym(xmat + umat + 0.5 * mid).ravel()


def _published_factory_cases() -> list[dict]:
    """Published manopt factory algebra dest claims to port.

    These are the MATLAB formulas from the seven dest factories, not dest
    numbers. proj / retr / transp for Sphere, Positive, Symmetric,
    ComplexCircle, Multinomial, CenteredMatrix, and SPD.
    """
    cases: list[dict] = []

    x = np.array([0.0, 1.0, 0.0])
    v = np.array([0.2, 0.3, -0.1])
    proj = _sphere_proj(x, v)
    retr = _sphere_retr(x, proj)
    cases.append(_case("sphere_proj_north", "sphere_proj", x, v, proj))
    cases.append(_case("sphere_retr_north", "sphere_retr", x, proj, retr))
    cases.append(_case("sphere_transp_north", "sphere_transp", x, v, _sphere_proj(retr, v), y=retr))

    xp = np.array([1.5, 0.5, 2.0])
    vp = np.array([0.1, -0.2, 0.0])
    yp = xp * np.exp(vp / xp)
    cases.append(_case("positive_proj_id", "positive_proj", xp, vp, vp))
    cases.append(_case("positive_retr_exp", "positive_retr", xp, vp, yp))
    cases.append(_case("positive_transp_id", "positive_transp", xp, vp, vp, y=yp))

    xs = np.array([1.0, 0.0, 0.0, -1.0])
    vs = np.array([0.0, 0.2, -0.1, 0.0])
    sym = _multisym(vs.reshape(2, 2)).ravel()
    retr_s = xs + sym
    cases.append(_case("symmetric_multisym", "symmetric_proj", xs, vs, sym))
    cases.append(_case("symmetric_retr_plus", "symmetric_retr", xs, sym, retr_s))
    cases.append(_case("symmetric_transp_id", "symmetric_transp", xs, sym, sym, y=retr_s))

    zc = np.array([1.0, 0.0, 0.0, 1.0])
    vc = np.array([0.5, 0.25, -0.1, 0.8])
    pc = _cc_proj(zc, vc)
    rc = _cc_retr(zc, pc)
    cases.append(_case("complexcircle_proj_pairs", "complexcircle_proj", zc, vc, pc))
    cases.append(_case("complexcircle_retr_sign", "complexcircle_retr", zc, pc, rc))
    cases.append(_case("complexcircle_transp_arrive", "complexcircle_transp", zc, vc, _cc_proj(rc, vc), y=rc))

    xm = np.array([0.2, 0.3, 0.5])
    vm = np.array([1.0, 2.0, 3.0])
    pm = _mn_proj(xm, vm)
    rm = _mn_retr(xm, pm)
    cases.append(_case("multinomial_proj_fisher", "multinomial_proj", xm, vm, pm))
    cases.append(_case("multinomial_retr_exp", "multinomial_retr", xm, pm, rm))
    cases.append(_case("multinomial_transp_arrive", "multinomial_transp", xm, vm, _mn_proj(rm, vm), y=rm))

    xcent = np.array([1.0, -0.5, -0.5, 2.0, 0.0, -2.0])
    vcent_amb = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pcent = _center_cols(vcent_amb, 2, 3)
    vcent = np.array([0.3, -0.1, -0.2, 0.0, 0.4, -0.4])
    rcent = xcent + vcent
    cases.append(_case("centered_proj_cols", "centered_proj", xcent, vcent_amb, pcent))
    cases.append(_case("centered_retr_plus", "centered_retr", xcent, vcent, rcent))
    cases.append(_case("centered_transp_id", "centered_transp", xcent, vcent, vcent, y=rcent))

    xspd_p = np.array([1.0, 0.0, 0.0, 2.0])
    vspd_p = np.array([0.3, 1.0, -0.4, 0.5])
    pspd = _multisym(vspd_p.reshape(2, 2)).ravel()
    xspd = np.array([1.0, 0.0, 0.0, 1.0])
    vspd = np.array([0.0, 0.2, 0.2, 0.0])
    rspd = _spd_retr(xspd, vspd)
    cases.append(_case("spd_proj_symm", "spd_proj", xspd_p, vspd_p, pspd))
    cases.append(_case("spd_retr_second", "spd_retr", xspd, vspd, rspd))
    cases.append(_case("spd_transp_id", "spd_transp", xspd, vspd, vspd, y=rspd))
    return cases


def _manopt_tree_runnable(root: Path) -> bool:
    """True when the tree has importmanopt or tools needed to call factories."""
    clone = _manopt_clone_root(root)
    markers = (
        clone / "importmanopt.m",
        clone / "manopt" / "importmanopt.m",
        clone / "manopt" / "tools" / "multisym.m",
        clone / "tools" / "multisym.m",
    )
    return any(path.is_file() for path in markers)


def _matlab_or_octave() -> tuple[str, list[str]] | None:
    octave = shutil.which("octave-cli") or shutil.which("octave")
    if octave:
        return "octave", [octave, "--quiet", "--no-window-system", "--eval"]
    matlab = shutil.which("matlab")
    if matlab:
        return "matlab", [matlab, "-batch"]
    return None


def _parse_floats(line: str) -> list[float]:
    return [float(v) for v in line.split()]


def _run_manopt_engine(root: Path, argv: list[str]) -> list[dict] | None:
    """Call factory proj/retr/transp from a manopt tree via MATLAB/Octave."""
    clone = _manopt_clone_root(root)
    tmp = Path(tempfile.mkdtemp(prefix="rgmin-manopt-gold-"))
    out = tmp / "manopt_gold.txt"
    root_lit = str(clone).replace("'", "''")
    out_lit = str(out).replace("'", "''")
    script = f"""
addpath(genpath('{root_lit}'));
if exist('importmanopt', 'file') == 2
    importmanopt;
end
Ms = spherefactory(3);
xs = [0; 1; 0];
vs = [0.2; 0.3; -0.1];
sproj = Ms.proj(xs, vs);
sretr = Ms.retr(xs, sproj);
strans = Ms.transp(xs, sretr, vs);
Mp = positivefactory(3);
xp = [1.5; 0.5; 2.0];
vp = [0.1; -0.2; 0.0];
pproj = Mp.proj(xp, vp);
pretr = Mp.retr(xp, vp);
ptrans = Mp.transp(xp, pretr, vp);
Msym = symmetricfactory(2);
xsym = [1, 0; 0, -1];
vsym = [0, 0.2; -0.1, 0];
symp = Msym.proj(xsym, vsym);
symr = Msym.retr(xsym, symp);
symt = Msym.transp(xsym, symr, symp);
Mc = complexcirclefactory(2);
zc = [1+0i; 0+1i];
uc = [0.5+0.25i; -0.1+0.8i];
cproj = Mc.proj(zc, uc);
cretr = Mc.retr(zc, cproj);
ctrans = Mc.transp(zc, cretr, uc);
Mm = multinomialfactory(3);
xm = [0.2; 0.3; 0.5];
vm = [1; 2; 3];
mproj = Mm.proj(xm, vm);
mretr = Mm.retr(xm, mproj);
mtrans = Mm.transp(xm, mretr, vm);
Mcent = centeredmatrixfactory(2, 3, 'cols');
xcent = [1, -0.5, -0.5; 2, 0, -2];
vcenta = [1, 2, 3; 4, 5, 6];
cprojm = Mcent.proj(xcent, vcenta);
vcent = [0.3, -0.1, -0.2; 0.0, 0.4, -0.4];
cretrm = Mcent.retr(xcent, vcent);
ctransm = Mcent.transp(xcent, cretrm, vcent);
Mspd = sympositivedefinitefactory(2);
xspdp = [1, 0; 0, 2];
vspdp = [0.3, 1.0; -0.4, 0.5];
spdproj = Mspd.proj(xspdp, vspdp);
xspd = [1, 0; 0, 1];
vspd = [0, 0.2; 0.2, 0];
spdretr = Mspd.retr(xspd, vspd);
spdtrans = Mspd.transp(xspd, spdretr, vspd);
fid = fopen('{out_lit}', 'w');
fprintf(fid, '%.17g %.17g %.17g\\n', sproj(1), sproj(2), sproj(3));
fprintf(fid, '%.17g %.17g %.17g\\n', sretr(1), sretr(2), sretr(3));
fprintf(fid, '%.17g %.17g %.17g\\n', strans(1), strans(2), strans(3));
fprintf(fid, '%.17g %.17g %.17g\\n', pproj(1), pproj(2), pproj(3));
fprintf(fid, '%.17g %.17g %.17g\\n', pretr(1), pretr(2), pretr(3));
fprintf(fid, '%.17g %.17g %.17g\\n', ptrans(1), ptrans(2), ptrans(3));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', symp(1,1), symp(1,2), symp(2,1), symp(2,2));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', symr(1,1), symr(1,2), symr(2,1), symr(2,2));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', symt(1,1), symt(1,2), symt(2,1), symt(2,2));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', real(cproj(1)), imag(cproj(1)), real(cproj(2)), imag(cproj(2)));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', real(cretr(1)), imag(cretr(1)), real(cretr(2)), imag(cretr(2)));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', real(ctrans(1)), imag(ctrans(1)), real(ctrans(2)), imag(ctrans(2)));
fprintf(fid, '%.17g %.17g %.17g\\n', mproj(1), mproj(2), mproj(3));
fprintf(fid, '%.17g %.17g %.17g\\n', mretr(1), mretr(2), mretr(3));
fprintf(fid, '%.17g %.17g %.17g\\n', mtrans(1), mtrans(2), mtrans(3));
fprintf(fid, '%.17g %.17g %.17g %.17g %.17g %.17g\\n', cprojm(1,1), cprojm(1,2), cprojm(1,3), cprojm(2,1), cprojm(2,2), cprojm(2,3));
fprintf(fid, '%.17g %.17g %.17g %.17g %.17g %.17g\\n', cretrm(1,1), cretrm(1,2), cretrm(1,3), cretrm(2,1), cretrm(2,2), cretrm(2,3));
fprintf(fid, '%.17g %.17g %.17g %.17g %.17g %.17g\\n', ctransm(1,1), ctransm(1,2), ctransm(1,3), ctransm(2,1), ctransm(2,2), ctransm(2,3));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', spdproj(1,1), spdproj(1,2), spdproj(2,1), spdproj(2,2));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', spdretr(1,1), spdretr(1,2), spdretr(2,1), spdretr(2,2));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', spdtrans(1,1), spdtrans(1,2), spdtrans(2,1), spdtrans(2,2));
fclose(fid);
"""
    try:
        proc = subprocess.run(
            argv + [script],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not out.is_file():
        return None
    lines = out.read_text().strip().splitlines()
    if len(lines) != 21:
        return None
    try:
        nums = [_parse_floats(line) for line in lines]
    except ValueError:
        return None
    want = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 3, 3, 3, 6, 6, 6, 4, 4, 4]
    if [len(row) for row in nums] != want:
        return None
    (
        sproj,
        sretr,
        strans,
        pproj,
        pretr,
        ptrans,
        symp,
        symr,
        symt,
        cproj,
        cretr,
        ctrans,
        mproj,
        mretr,
        mtrans,
        cprojm,
        cretrm,
        ctransm,
        spdproj,
        spdretr,
        spdtrans,
    ) = nums
    xs = [0.0, 1.0, 0.0]
    vs = [0.2, 0.3, -0.1]
    xp = [1.5, 0.5, 2.0]
    vp = [0.1, -0.2, 0.0]
    xsym = [1.0, 0.0, 0.0, -1.0]
    vsym = [0.0, 0.2, -0.1, 0.0]
    zc = [1.0, 0.0, 0.0, 1.0]
    uc = [0.5, 0.25, -0.1, 0.8]
    xm = [0.2, 0.3, 0.5]
    vm = [1.0, 2.0, 3.0]
    xcent = [1.0, -0.5, -0.5, 2.0, 0.0, -2.0]
    vcenta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    vcent = [0.3, -0.1, -0.2, 0.0, 0.4, -0.4]
    xspdp = [1.0, 0.0, 0.0, 2.0]
    vspdp = [0.3, 1.0, -0.4, 0.5]
    xspd = [1.0, 0.0, 0.0, 1.0]
    vspd = [0.0, 0.2, 0.2, 0.0]
    return [
        _case("sphere_proj_north", "sphere_proj", xs, vs, sproj),
        _case("sphere_retr_north", "sphere_retr", xs, sproj, sretr),
        _case("sphere_transp_north", "sphere_transp", xs, vs, strans, y=sretr),
        _case("positive_proj_id", "positive_proj", xp, vp, pproj),
        _case("positive_retr_exp", "positive_retr", xp, vp, pretr),
        _case("positive_transp_id", "positive_transp", xp, vp, ptrans, y=pretr),
        _case("symmetric_multisym", "symmetric_proj", xsym, vsym, symp),
        _case("symmetric_retr_plus", "symmetric_retr", xsym, symp, symr),
        _case("symmetric_transp_id", "symmetric_transp", xsym, symp, symt, y=symr),
        _case("complexcircle_proj_pairs", "complexcircle_proj", zc, uc, cproj),
        _case("complexcircle_retr_sign", "complexcircle_retr", zc, cproj, cretr),
        _case("complexcircle_transp_arrive", "complexcircle_transp", zc, uc, ctrans, y=cretr),
        _case("multinomial_proj_fisher", "multinomial_proj", xm, vm, mproj),
        _case("multinomial_retr_exp", "multinomial_retr", xm, mproj, mretr),
        _case("multinomial_transp_arrive", "multinomial_transp", xm, vm, mtrans, y=mretr),
        _case("centered_proj_cols", "centered_proj", xcent, vcenta, cprojm),
        _case("centered_retr_plus", "centered_retr", xcent, vcent, cretrm),
        _case("centered_transp_id", "centered_transp", xcent, vcent, ctransm, y=cretrm),
        _case("spd_proj_symm", "spd_proj", xspdp, vspdp, spdproj),
        _case("spd_retr_second", "spd_retr", xspd, vspd, spdretr),
        _case("spd_transp_id", "spd_transp", xspd, vspd, spdtrans, y=spdretr),
    ]


def mint_manopt_formulas() -> tuple[dict, list[dict]]:
    """Mint factory numbers from MANOPT_ROOT when set, else published formulas.

    When MANOPT_ROOT points at a NicolasBoumal/manopt tree, the remint
    records the factory files it used. MATLAB/Octave, if present, runs
    those factories; otherwise the published Python formulas fill in.
    dest tests load only the frozen JSON and do not require this tree.
    """
    root = _manopt_root()
    if root is None:
        return {"manopt": "published-factory-formulas"}, _published_factory_cases()

    factories = _locate_manopt_factories(root)
    rels = [_rel_manopt_file(root, path) for path in factories]
    meta: dict = {
        "manopt": "MANOPT_ROOT",
        "manopt_files": rels,
        "manopt_exists": all(path.is_file() for path in factories),
    }
    engine = _matlab_or_octave() if _manopt_tree_runnable(root) else None
    if engine is not None:
        name, argv = engine
        cases = _run_manopt_engine(root, argv)
        if cases is not None:
            meta["manopt_engine"] = name
            return meta, cases
    meta["manopt_engine"] = "published-factory-formulas"
    return meta, _published_factory_cases()


def main() -> None:
    root = _sella_root()
    sella_meta, sella_cases = mint_sella(root)
    manopt_meta, manopt_cases = mint_manopt_formulas()
    out = {
        "source": {**sella_meta, **manopt_meta},
        "cases": sella_cases + manopt_cases,
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
