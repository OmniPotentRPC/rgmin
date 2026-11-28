

# Cautious Riemannian L-BFGS

`minimize_rlbfgs` is Algorithm 2 of Huang, Absil, and Gallivan, *A Riemannian BFGS Method Without Differentiated Retraction for Nonconvex Optimization Problems*, SIAM J. Optim. 28 (2018) 470–495 ([10.1137/17M1127582](https://doi.org/10.1137/17M1127582)). The two-loop inverse Hessian is Huang, Gallivan, and Absil 2015 ([10.1137/140955483](https://doi.org/10.1137/140955483)). On a Euclidean manifold the transport is the identity, so the update is the cautious BFGS method of Li and Fukushima 2001 ([10.1137/S1052623499354242](https://doi.org/10.1137/S1052623499354242)).

`Rlbfgs::default` is the section 7 choice: four pairs, `vartheta(t) = 10^{-4} t`, Armijo constant `10^{-4}`, backtrack one half, and the stop `||grad|| / ||grad_0|| < 10^{-6}`. A pair that fails the cautious test is not stored. The line search does not differentiate the retraction. The vector transport is the parallel transport along the exponential when the manifold provides that pair, and the retraction transport otherwise.

This does not replace `Lbfgs`. The production Euclidean step stays `LbfgsStep::Capped` at `EON_MAX_MOVE`.

```rust
use ndarray::{ArrayView1, array};
use rgmin::{Control, ManifoldKind, Oracle, Rlbfgs, minimize_rlbfgs};

let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
    let value = 0.5 * (x[0] * x[0] + 100.0 * x[1] * x[1]);
    let grad = array![x[0], 100.0 * x[1]];
    (value, grad)
});
let report = minimize_rlbfgs(
    &ManifoldKind::Euclidean,
    &obj,
    array![1.0, 1.0],
    &Control::default().with_maxiter(12).with_gtol(0.0),
    &Rlbfgs::default(),
)?;
assert!(report.converged());
```

The quadratic is the Euclidean reduction. Its minimizer has value 0. Section 7.2 is the relative gradient stop used above. On the sphere the same solver is the joint diagonalization cost of section 7.1, `-(x^T C x)^2`, whose diagonalizer for `C = diag(3, 2, 1)` has cost `-9`.

The C record is `rgmin_rlbfgs_t`. Null parameters select the Rust default. `rgmin_minimize_rlbfgs` takes the session manifold token. ABI minor 51.
