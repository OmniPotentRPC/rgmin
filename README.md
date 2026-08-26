# rgmin

<p align="center">
  <img src="branding/logo/rgmin-logo-light.svg" width="420" alt="rgmin">
</p>

Rust rewrite. `main` is this tree (current tag `v0.2.0`). The C++
xtensor history is `0.0.1` on the previous `main`.

Algorithms live only in Rust. A session takes one stepper step on an
embedded manifold and reports. Saddle, band, and IRC sessions live in
[`rgsaddle`](https://github.com/OmniPotentRPC/rgsaddle).

C and C++ reach the Rust solvers through `include/rgmin.h`
(`rgmin_solver_t` / `rgmin_minimize`) over **dlpk**
`DLManagedTensorVersioned` tensors.
`include/rgmin/optimize.hpp` is the C++ RAII wrap (`namespace rgmin`).
`include/xts.h` and `include/xts/optimize.hpp` are compatibility
aliases for old hosts. There is no second solver implementation in C++.

CPU f64 is wired now. A non-CPU tensor returns `RGMIN_UNSUPPORTED_DEVICE`
so a CUDA path does not change the ABI.

The production unconstrained local method is [`Lbfgs`] with strong Wolfe
(Nocedal-Wright 7.4 and algorithms 3.5 / 3.6). anneal `WarmLbfgs` holds
that type so hopping does not ship a second two-loop. Feature `highs` keeps the two-loop direction and projects it with
HiGHS (`Q = I`) onto a box, trust region, or equalities. Constrained
QP uses HiGHS IPM (IPX or HiPO) with crossover off; the C waist pins
the engine (`rgmin_highs_solver_t`). Feature `slepc` links SLEPc EPS
on the lowest-mode waist (MatShell, typed `EPSSet*` / `STSet*` only)
when PETSc/SLEPc are present; otherwise `EigensolverKind::Slepc` stays
`Error::EigenUnavailable`.

A session can retract onto an embedded manifold (`set_manifold`):
Euclidean (default), sphere, SO(3), Stiefel `St(n,1)`, SE(3),
SPD (affine-invariant, row-major `n x n`).

Conjugacy: Fletcher-Reeves, Polak-Ribiere, Hestenes-Stiefel, Dai-Yuan,
conjugate descent, Hager-Zhang, Liu-Storey, FR-PR, HybridizedConj.
Line search: Brent, Armijo, Goldstein, strong Wolfe with zoom.
Methods: NLCG, BFGS, L-BFGS, SR1, SR2, Newton, RFO, Adam, steepest
descent, PSO, FIRE, FIRE 2.0, Barzilai-Borwein, Powell dogleg.

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

MIT. Build and test on the remote builder, not a laptop.
