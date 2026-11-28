

# Newton

Shifted Newton needs a Hessian. On a convex quadratic it takes the Newton step `(H)^{-1} g`. The oracle is `HessianOracle::unbounded`.

The listing is `examples/tutorial_newton.rs`, whole. Continuous integration runs it, so it is the program as it builds.

```rust
//! Shifted Newton on a convex quadratic, with the Hessian supplied.
//!
//! `cargo run --example tutorial_newton`

use ndarray::{ArrayView1, array};
use rgmin::{Control, HessianOracle, NewtonKind, minimize_newton};

fn main() {
    let value = |x: ArrayView1<f64>| {
        let f = 0.5 * (4.0 * x[0] * x[0] + x[1] * x[1]) - x[0] + 2.0 * x[1];
        let g = array![4.0 * x[0] - 1.0, x[1] + 2.0];
        (f, g)
    };
    let hess = |_x: ArrayView1<f64>| array![[4.0, 0.0], [0.0, 1.0]];
    let obj = HessianOracle::unbounded(2, value, hess);
    let report = minimize_newton(
        &obj,
        array![0.0, 0.0],
        &Control::default().with_maxiter(40).with_gtol(1e-12),
        NewtonKind::Shifted,
    )
    .unwrap();
    assert!((report.coords[0] - 0.25).abs() < 1e-8, "{}", report.coords);
    assert!((report.coords[1] + 2.0).abs() < 1e-8, "{}", report.coords);
}
```
