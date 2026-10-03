#![cfg(feature = "highs")]

use ndarray::{ArrayView1, array};
use rgmin::{Control, HessianOracle, Method, NewtonKind, Solver};

#[test]
fn constrained_newton_preserves_cross_curvature() {
    let objective = HessianOracle::unbounded(
        2,
        |x: ArrayView1<f64>| {
            (
                2.0 * (x[0] * x[0] + x[1] * x[1]) - 3.0 * x[0] * x[1] + x[0] - 2.0 * x[1],
                array![4.0 * x[0] - 3.0 * x[1] + 1.0, 4.0 * x[1] - 3.0 * x[0] - 2.0],
            )
        },
        |_x: ArrayView1<f64>| array![[4.0, -3.0], [-3.0, 4.0]],
    );
    let mut solver = Solver::new(
        Method::Newton {
            kind: NewtonKind::Shifted,
        },
        Control {
            maxiter: 1,
            gtol: 1e-12,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        },
        2,
    );
    solver.set_highs(true);
    assert!(solver.set_box(Some(vec![0.0, -1.0]), Some(vec![1.0, 1.0])));
    let report = solver.step_hess(&objective, &mut array![0.0, 0.0]).unwrap();
    assert!(
        (report.coords[0] - 2.0 / 7.0).abs() < 1e-6,
        "{:?}",
        report.coords
    );
    assert!(
        (report.coords[1] - 5.0 / 7.0).abs() < 1e-6,
        "{:?}",
        report.coords
    );
}
