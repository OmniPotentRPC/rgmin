//! A conjugate direction with a non-negative slope is replaced by -g.

use std::sync::{Arc, Mutex};

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Conjugacy, Control, LineSearch, Oracle, Restart, minimize};

#[test]
fn an_uphill_fletcher_reeves_direction_is_replaced_by_steepest_descent() {
    // From the origin the gradient is (1, 0). The accepted point is
    // (-1, 0), where the gradient is (-2, 1). Fletcher-Reeves builds
    // d = (-3, -1), and g . d = 5. The line search would walk -d.
    // Nocedal-Wright replaces d with -g = (2, -1) before the search.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let obj = Oracle::unbounded(2, move |x: ArrayView1<f64>| {
        log.lock().expect("log").push(x.to_owned());
        if x[0].abs() <= 1e-12 && x[1].abs() <= 1e-12 {
            return (0.0, array![1.0, 0.0]);
        }
        if (x[0] + 1.0).abs() <= 1e-9 && x[1].abs() <= 1e-9 {
            return (-1.0, array![-2.0, 1.0]);
        }
        (0.0, array![0.0, 0.0])
    });
    let control = Control::default()
        .with_maxiter(2)
        .with_gtol(1e-14)
        .with_istep(1.0)
        .with_maxmove(Some(-1.0));
    minimize(
        &obj,
        array![0.0, 0.0],
        &control,
        Conjugacy::FletcherReeves,
        Restart::Never,
        LineSearch::backtracking(1e-4, 0.5, 1),
    )
    .expect("minimize");
    let pts = seen.lock().expect("log");
    assert!(pts.len() >= 3, "points {pts:?}");
    let x1 = &pts[1];
    assert!(
        (x1[0] + 1.0).abs() <= 1e-12 && x1[1].abs() <= 1e-12,
        "first step {x1:?}"
    );
    let disp: Array1<f64> = &pts[2] - x1;
    let steep = array![2.0, -1.0];
    let cross = (disp[0] * steep[1] - disp[1] * steep[0]).abs();
    let scale = disp.dot(&disp).sqrt() * steep.dot(&steep).sqrt();
    assert!(
        cross <= 1e-9 * scale.max(1.0),
        "trial {disp:?} is not -g {steep:?} points {pts:?}"
    );
}
