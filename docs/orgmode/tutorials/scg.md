

# Scaled conjugate gradient

Møller SCG prices the step with a damped model. There is no line search. A positive finite `Control::maxmove` bounds every trial, including the retry after a non-finite value. Zero, negative, infinite and NaN leave the step unbounded.

The listing is `examples/tutorial_scg.rs`, whole. Continuous integration runs it, so it is the program as it builds.

```rust
//! Møller scaled conjugate gradient. The step length comes from a
//! damped model, not a line search.
//!
//! `cargo run --example tutorial_scg`

use ndarray::{ArrayView1, array};
use rgmin::{Conjugacy, Control, Oracle, Restart, ScgParams, minimize_scg};

fn main() {
    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let f = (x[0] - 1.0).powi(2) + 4.0 * (x[1] + 2.0).powi(2);
        let g = array![2.0 * (x[0] - 1.0), 8.0 * (x[1] + 2.0)];
        (f, g)
    });
    let report = minimize_scg(
        &obj,
        array![0.0, 0.0],
        &Control::default().with_maxiter(40).with_gtol(1e-8),
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
    .unwrap();
    assert!((report.coords[0] - 1.0).abs() < 1e-4, "{}", report.coords);
    assert!((report.coords[1] + 2.0).abs() < 1e-4, "{}", report.coords);
}
```

The example calls `minimize_scg` on a quadratic and checks both coordinates.
