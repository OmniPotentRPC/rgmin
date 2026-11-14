//! Default L-BFGS is the eOn cap with no line search, and `Lbfgs::capped`
//! sets that cap explicitly. `Lbfgs::builder().wolfe()` opts into the
//! strong Wolfe search.
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
