

# BFGS

Dense BFGS keeps an inverse Hessian, starting from the identity. Himmelblau has four minima. From `(0, 0)` the session reaches `(3.584, -1.848)`, not `(3, 2)`.

The listing is `examples/tutorial_bfgs.rs`, whole. Continuous integration runs it, so it is the program as it builds.

```rust
//! Dense BFGS on the Himmelblau function, one of four minima.
//!
//! `cargo run --example tutorial_bfgs`

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Oracle, Solver, StopNorm};

fn himmelblau(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (a, b) = (x[0], x[1]);
    let u = a * a + b - 11.0;
    let v = a + b * b - 7.0;
    let f = u * u + v * v;
    let g = array![4.0 * a * u + 2.0 * v, 2.0 * u + 4.0 * b * v];
    (f, g)
}

fn main() {
    let control = Control::default()
        .with_maxiter(80)
        .with_gtol(1e-8)
        .with_stop_norm(StopNorm::Inf);
    let mut solver = Solver::new(Method::Bfgs, control, 2);
    let obj = Oracle::unbounded(2, himmelblau);
    let mut x = array![0.0, 0.0];
    let report = solver.run(&obj, &mut x).unwrap();
    // Four minima. (0, 0) reaches (3.584, -1.848), not (3, 2).
    assert!(report.value < 1e-10, "value {} at {x}", report.value);
}
```

ASE's `BFGS` multiplies the first step by `1/alpha` with `alpha = 70`. This session does not. The EMT gold in `python/tests/test_ase_optimizer.py` checks the minimum, not that first step.
