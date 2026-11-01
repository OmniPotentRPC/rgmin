# rgmin

<p align="center">
  <img src="branding/logo/rgmin-logo-light.svg" width="420" alt="rgmin">
</p>

rgmin finds a local minimum of a smooth function from its value and gradient.
It is built for geometry optimization.
There, a host program such as eOn owns the loop, and a `Solver` takes one step per call.
The step can stay on a manifold, such as the unit sphere or Cartesians with the rigid-body motions removed.
`Lbfgs::minimize` runs a whole minimization from a Rust closure (example below).
Saddle, band, and intrinsic reaction coordinate (IRC) sessions live in [`rgsaddle`](https://github.com/OmniPotentRPC/rgsaddle).

Releases are tagged `vX.Y.Z` on `main` and published to crates.io.
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

CPU f64 is wired now. A non-CPU tensor returns `RGMIN_UNSUPPORTED_DEVICE`
so a CUDA path does not change the ABI.

The production unconstrained local method is [`Lbfgs`] with strong Wolfe
(Nocedal-Wright 7.4 and algorithms 3.5 / 3.6). anneal `WarmLbfgs` holds
that type so hopping does not ship a second two-loop. Feature `highs` keeps the two-loop direction and projects it with
HiGHS (`Q = I`) onto a box, trust region, or equalities.
Constrained quadratic programming (QP) uses the HiGHS interior point method (IPM), IPX or HiPO, with crossover off.
The C ABI pins the engine (`rgmin_highs_solver_t`).

A session can retract onto an embedded manifold (`set_manifold`):
Euclidean (default), sphere, SO(3), Stiefel `St(n,1)`, SE(3),
SPD (affine-invariant, row-major `n x n`).

Conjugacy: Fletcher-Reeves, Polak-Ribiere, Hestenes-Stiefel, Dai-Yuan,
conjugate descent, Hager-Zhang, Liu-Storey, FR-PR, HybridizedConj.
Line search: Brent, Armijo, Goldstein, strong Wolfe with zoom.
Methods: nonlinear conjugate gradients (NLCG), Broyden-Fletcher-Goldfarb-Shanno (BFGS) and limited-memory BFGS (L-BFGS), the symmetric rank-one (SR1) and rank-two (SR2) secant updates, Newton, rational function optimization (RFO), Adam, steepest descent, particle swarm optimization (PSO), the fast inertial relaxation engine (FIRE) and FIRE 2.0, Barzilai-Borwein, Powell dogleg, Steihaug-Toint Newton-CG (`newton-cg`, `newton-pcg`).
Newton, RFO and the dogleg also take a Hessian from the host.
`Solver::set_exp` installs the exponential preconditioner of
Packwood et al., J. Chem. Phys. 144, 164109 (2016), from a caller
neighbour list. L-BFGS uses it as `H0`. FIRE and nonlinear CG use
it as the metric. A dense pair or Lindh matrix supplied to the
two-loop still wins.

Narrative docs live in [`docs/orgmode/`](docs/orgmode/index.org)
(Diataxis); the explanation quadrant derives the math behind each
solver, and every derivation's exact algebra is pinned by the Lean
(Mathlib) development under [`proofs/lean/`](proofs/lean/), indexed in
[`docs/orgmode/reference/proofs.org`](docs/orgmode/reference/proofs.org).
C ABI HTML is Doxygen plus
[doxyYoda](https://github.com/HaoZeke/doxyYoda):
`scripts/get_doxyyoda.sh` then `doxygen docs/source/Doxyfile_doxyyoda.cfg`.
The Sphinx bibliography is the `rgmin-docs` ookcite collection
exported to `docs/source/references.bib` and cited from orgmode
with `sphinxcontrib-bibtex` (`docs/orgmode/reference/bibliography.org`).
The mark is documented in [`branding/logo/`](branding/logo/README.md).

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

Install the 0.5.2 release from crates.io with `cargo add rgmin ndarray@0.17` on Rust 1.88 or newer.
[`docs/orgmode/howto/install.org`](docs/orgmode/howto/install.org) covers the `develop` branch and how it differs from the release.
It also covers the features of each route and the C library.

MIT.
