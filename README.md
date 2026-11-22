# rgmin

<p align="center">
  <img src="https://raw.githubusercontent.com/OmniPotentRPC/rgmin/main/branding/logo/rgmin-logo-light.svg" width="420" alt="rgmin">
</p>

rgmin finds a local minimum of a smooth function from its value and gradient.
It is built for geometry optimization.
There, a host program such as eOn owns the loop, and a `Solver` takes one step per call.
The step can stay on a manifold, such as the unit sphere or Cartesians with the rigid-body motions removed.
`Lbfgs::minimize` runs a whole minimization from a Rust closure (example below).
Algorithms live only in Rust, over [`eindir`](https://github.com/HaoZeke/eindir) `DifferentiableObjective`s.
Saddle, band, and intrinsic reaction coordinate (IRC) sessions live in [`rgsaddle`](https://github.com/OmniPotentRPC/rgsaddle).

Releases are tagged `vX.Y.Z` on `main` and published to crates.io and PyPI.
This is the Rust rewrite; the C++ xtensor history is `0.0.1` on the previous `main`.

C and C++ reach the Rust solvers through the C application binary interface (ABI) in `include/rgmin.h` (`rgmin_solver_t` / `rgmin_minimize`).
Vectors cross that boundary as DLPack `DLManagedTensorVersioned` tensors from the [`dlpk`](https://crates.io/crates/dlpk) crate.
`include/rgmin/optimize.hpp` is the C++ resource-acquisition-is-initialization (RAII) wrap (`namespace rgmin`).
`include/xts.h` and `include/xts/optimize.hpp` are compatibility aliases for old hosts.
There is no second solver implementation in C++.

`cargo cinstall --prefix=/usr/local --features capi` installs that
header, the C++ wrappers, and `dlpack/dlpack.h`. The pkg-config name
is `rgmin`. Version macros are `RGMIN_H_MAJOR`, `RGMIN_H_MINOR`, and
`RGMIN_H_PATCH`.
The C ABI also accepts an `eindir_objective_t*` directly through
`rgmin_minimize_eindir`, or `rgmin_minimize_eindir_opts` for the
`error_on_unmet` and `backward_polish` flags. The caller supplies the `eindir_abi_stamp_t`, retains
ownership of the objective, and gets the same CPU f64 DLPack optimization path.
This lets rgpot's first-member `eindir_objective_t` embedding reach xtsci
without a callback shim in each consumer.

CPU f64 is wired now. A non-CPU tensor returns `RGMIN_UNSUPPORTED_DEVICE`
so a CUDA path does not change the ABI.

The production unconstrained local method is `Lbfgs` with the eOn cap.
That step is the default, `LbfgsStep::Capped` at 0.2, and it runs no
line search: the longest atom move is at most 0.2, the first step probes
curvature, and a longer proposal clears the memory.
`LbfgsStep::Wolfe` opts into the strong Wolfe line search.
`LbfgsStep::Scaled` shortens the quasi-Newton step and keeps the pairs.
The [`anneal`](https://github.com/HaoZeke/anneal) crate's `WarmLbfgs`
holds an `Lbfgs`, so basin hopping does not need a second two-loop.
Feature `highs` keeps the two-loop direction and projects it with
HiGHS (`Q = I`) onto a box, trust region, or equalities.
Constrained quadratic programming (QP) uses the HiGHS interior point method (IPM), IPX or HiPO, with crossover off.
The C ABI pins the engine (`rgmin_highs_solver_t`).

A session can retract onto an embedded manifold (`set_manifold`), such
as the sphere, SO(3), SE(3), Stiefel `St(n,1)`, SPD, or a cluster with
the rigid-body motions removed. Euclidean is the default.
[`docs/orgmode/howto/manifolds.org`](https://github.com/OmniPotentRPC/rgmin/blob/main/docs/orgmode/howto/manifolds.org)
lists every `ManifoldKind`.

Conjugacy: Fletcher-Reeves, Polak-Ribiere, Polak-Ribiere-plus,
Hestenes-Stiefel, Dai-Yuan, conjugate descent, Hager-Zhang, Liu-Storey,
FR-PR, and `Conjugacy::Hybrid`, the max or min of two of these.
Line search: Brent, Armijo backtracking, Goldstein, strong Wolfe with zoom.
Methods: nonlinear conjugate gradients (NLCG), BFGS, L-BFGS, SR1, SR2, Newton, RFO, Adam, steepest descent, PSO, FIRE and FIRE 2.0, quick-min, Barzilai-Borwein, ODE12r, DIIS, Powell dogleg, Steihaug-Toint Newton-CG (`newton-cg`, `newton-pcg`).
Newton, RFO and the dogleg also take a Hessian from the host.
`Solver::set_exp` installs the exponential preconditioner of
Packwood et al., J. Chem. Phys. 144, 164109 (2016).
`rgmin_linesearch` (ABI minor 37) takes a caller-supplied energy and gradient.
`rgmin_lbfgs_direction` (ABI minor 38) applies a caller-supplied initial inverse Hessian.
`rgmin_lowest_eigenpairs` (ABI minor 39) is block LOBPCG for the k lowest pairs.
A session's line search defaults to strong Wolfe. Steepest descent,
NLCG, BFGS, Adam, DIIS, and L-BFGS after `LbfgsStep::Wolfe` use it; the
default L-BFGS step does not search a line. Each search starts from the
value and gradient the session holds and bounds trials by the
displacement cap (`LineSearch::search_from`). FIRE 2.0 as Guénolé et al. 2020 publish it
is `Solver::set_fire_variant(FireVariant::Guenole2020)`. Quick-min
(`Method::QuickMin`) runs eOn's step with
`Solver::set_quickmin_options(QuickMinOptions::eon())` and one
projection per atom with `QuickMinOptions::per_atom()`.
`bench/atomistic` counts force calls on rgpot potentials.
`bench/gate` (`python -m rggate`, pixi task `gate`) is the censored CSV
gate: same stationary point, Kaplan-Meier, BCa, and a confidence sequence.
`validation/` holds the sympy checks behind the interpolation, the
two-loop map and FIRE 2.0 (`uv run --with sympy python validation/<file>.py`).
`Solver::set_precon` installs Exp, Lindh, or the bonded metric.
L-BFGS uses the matrix as the initial inverse Hessian. FIRE and
nonlinear CG use it as the metric. `select_precon` picks Exp for a
bulk or a surface of at least 100 atoms, Lindh for a molecule or a
cluster whose elements are in the Lindh tables, the bonded metric
when that caller has a force field, and the identity otherwise.
The Python package `rgmin` is a free-threaded nanobind module
(`rgmin._core`) plus `rgmin.ase.Optimizer`, and `python/` is the maturin
ASE drop-in (`rgmin.optimize`: `BFGS`, `LBFGS`, `FIRE`, and the other
gradient methods, plus `lowest_eigenpairs`). ASE is an extra.
[`docs/orgmode/howto/ase-migration.org`](https://github.com/OmniPotentRPC/rgmin/blob/main/docs/orgmode/howto/ase-migration.org)
is the import change.

Narrative docs live in [`docs/orgmode/`](https://github.com/OmniPotentRPC/rgmin/blob/main/docs/orgmode/index.org)
(Diataxis); the explanation quadrant derives the math behind each
solver, and every derivation's exact algebra is pinned by the Lean
(Mathlib) development under [`proofs/lean/`](https://github.com/OmniPotentRPC/rgmin/tree/main/proofs/lean), indexed in
[`docs/orgmode/reference/proofs.org`](https://github.com/OmniPotentRPC/rgmin/blob/main/docs/orgmode/reference/proofs.org).
C ABI HTML is Doxygen plus
[doxyYoda](https://github.com/HaoZeke/doxyYoda):
`scripts/get_doxyyoda.sh` then `doxygen docs/source/Doxyfile_doxyyoda.cfg`.
The Sphinx bibliography is the `rgmin-docs` ookcite collection
exported to `docs/source/references.bib` and cited from orgmode
with `sphinxcontrib-bibtex` (`docs/orgmode/reference/bibliography.org`).
The mark is documented in [`branding/logo/`](https://github.com/OmniPotentRPC/rgmin/blob/main/branding/logo/README.md).

```rust
use ndarray::array;
use rgmin::Lbfgs;

let mut opt = Lbfgs::default();
let (_f, x, _) = opt.minimize(array![-1.2, 1.0].view(), 200, |x| {
    let a = 1.0 - x[0];
    let b = x[1] - x[0] * x[0];
    Some((a * a + 100.0 * b * b, array![-2.0 * a - 400.0 * x[0] * b, 200.0 * b]))
});
```

Install the 0.6 release from crates.io with `cargo add rgmin@0.6 ndarray@0.17` on Rust 1.88 or newer.
[`docs/orgmode/howto/install.org`](https://github.com/OmniPotentRPC/rgmin/blob/main/docs/orgmode/howto/install.org) covers the features, the `develop` branch, the C library from cargo-c or Meson, the Python packages and the LAMMPS plugin.

The optional `periodic-product` feature pulls in `rug` and `gmp-mpfr-sys`, which are LGPL-3.0 or later rather than MIT and build GMP, MPFR and MPC from source with autotools, so they need `m4`, `make`, `diffutils` and a C compiler.
It cannot be combined with `capi`, and `--all-features` stops with a compile error for that reason.

## Python

Install the nanobind package from PyPI, with ASE for `rgmin.ase` or without it:

```bash
python -m pip install rgmin
python -m pip install "rgmin[ase]"
```

There are wheels for Linux (`manylinux_2_28`) on x86_64 and aarch64, macOS 14 or newer on arm64 and x86_64, and Windows on x86_64.
CPython 3.10 and 3.11 each have a wheel, one abi3 wheel serves 3.12 and newer, and the free-threaded 3.14t and 3.15t each have a wheel.
Anywhere else pip builds the sdist, which needs Rust 1.88 or newer, CMake 3.15 or newer and a C++17 compiler.
The maturin ASE drop-in under `python/` (`rgmin.optimize`) is not on PyPI and builds from a checkout, as the install guide above shows.

## LAMMPS

`lammps/` builds `min_style rgmin`, a LAMMPS plugin that runs rgmin's L-BFGS, FIRE, FIRE 2.0, steepest descent and Barzilai-Borwein sessions through the C ABI.
After `plugin load rgminplugin.so`, change `min_style cg` to `min_style rgmin` and keep the rest of the input; it needs LAMMPS 22Jul2025 or later and runs under MPI.
[`docs/orgmode/howto/lammps-migration.org`](https://github.com/OmniPotentRPC/rgmin/blob/main/docs/orgmode/howto/lammps-migration.org) covers the build, the stopping rules and the differences from LAMMPS's minimizers, and the benchmark from `benches/lammps_minstyle.py` ([`benches/lammps_minstyle/results.md`](https://github.com/OmniPotentRPC/rgmin/blob/main/benches/lammps_minstyle/results.md)).

## Citation

If you use rgmin in published work, cite it with [`CITATION.cff`](https://github.com/OmniPotentRPC/rgmin/blob/main/CITATION.cff) and add the version or commit you ran.
[`docs/orgmode/howto/cite.org`](https://github.com/OmniPotentRPC/rgmin/blob/main/docs/orgmode/howto/cite.org) gives a BibTeX entry and shows where to find the version.

## License

MIT.
