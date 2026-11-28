

# L-BFGS

The default step is the eOn cap. The longest atom moves by at most 0.2, the first step probes curvature, and a proposal that long clears the memory. `Lbfgs::builder().wolfe()` is the strong Wolfe search. `Lbfgs::scaled` is the ASE length control: shorten the step and keep the pairs.

The listing is `examples/tutorial_lbfgs.rs`, whole. Continuous integration runs it, so it is the program as it builds.

```rust
//! Default L-BFGS is the eOn cap, with no line search. `Lbfgs::capped` sets the cap.
//!
//! `cargo run --example tutorial_lbfgs`

use ndarray::array;
use rgmin::{EON_MAX_MOVE, Lbfgs};

fn main() {
    let mut opt = Lbfgs::default();
    let x0 = array![-1.2, 1.0];
    let (value, x, evals) = opt.minimize(x0.view(), 200, |x| {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        Some((
            a * a + 100.0 * b * b,
            array![-2.0 * a - 400.0 * x[0] * b, 200.0 * b],
        ))
    });
    assert!(evals > 2, "evals {evals}");
    assert!(value < 1e-8, "value {value}");
    assert!(
        (x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3,
        "{x}"
    );

    let mut one = Lbfgs::capped(EON_MAX_MOVE);
    let (_f, x1, _) = one.minimize(array![1.0, 0.0, 0.0].view(), 1, |v| {
        Some((0.5 * v.dot(&v), v.to_owned()))
    });
    let moved = (1.0 - x1[0]).abs();
    assert!(
        (moved - EON_MAX_MOVE).abs() < 1e-9,
        "first step {moved} is not the eOn cap"
    );
}
```

On the 10-dimensional Rosenbrock start, that cap reaches the minimum at 0. The ASE scale stops in the well near `x0 = -1`, where the infinity norm is already below `1e-5` and the value is about 4. `tests/scaled_rosenbrock10.rs` is that comparison. SciPy's L-BFGS-B also reaches 0; the difference is the step rule.
