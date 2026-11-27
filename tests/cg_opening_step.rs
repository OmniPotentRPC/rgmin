//! The one-shot nonlinear CG opening step is Nocedal-Wright equation 3.60.

use std::sync::{Arc, Mutex};

use ndarray::{ArrayView1, array};
use rgmin::{Conjugacy, Control, LineSearch, Oracle, Restart, minimize};

#[test]
fn the_second_opening_step_follows_the_slope_ratio() {
    // f = x^2 / 2 from x = 1. Fletcher-Reeves, Armijo accepts the first
    // trial. Equation 3.60 then opens the next search at
    // 0.25 * (g0 . d0) / (g1 . d1) = 5/12, not at half of 0.25.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let obj = Oracle::unbounded(1, move |x: ArrayView1<f64>| {
        log.lock().expect("log").push(x[0]);
        (0.5 * x[0] * x[0], array![x[0]])
    });
    let control = Control::default()
        .with_maxiter(2)
        .with_gtol(1e-14)
        .with_istep(0.25)
        .with_maxmove(Some(-1.0));
    let report = minimize(
        &obj,
        array![1.0],
        &control,
        Conjugacy::FletcherReeves,
        Restart::Never,
        LineSearch::backtracking(1e-4, 0.5, 1),
    )
    .expect("minimize");
    assert!(report.steps >= 2, "steps {}", report.steps);
    let pts = seen.lock().expect("log");
    assert!(pts.len() >= 3, "points {pts:?}");
    let expect = 5.0 / 12.0;
    assert!(
        (pts[2] - expect).abs() <= 1e-9,
        "second trial {} want {expect} points {pts:?}",
        pts[2]
    );
}
