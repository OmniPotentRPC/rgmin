//! Shifted Newton finishes a quadratic in one accepted step.

use approx::assert_relative_eq;
use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::{Control, HessianOracle, NewtonKind, minimize_newton};

fn sphere_hess(_x: ArrayView1<f64>) -> Array2<f64> {
    Array2::<f64>::eye(2) * 2.0
}

fn sphere_fg(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let v = x[0] * x[0] + x[1] * x[1];
    (v, array![2.0 * x[0], 2.0 * x[1]])
}

#[test]
fn shifted_newton_kills_a_quadratic() {
    let obj = HessianOracle::unbounded(2, sphere_fg, sphere_hess);
    let report = minimize_newton(
        &obj,
        array![3.0, -4.0],
        &Control {
            maxiter: 10,
            gtol: 1e-10,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        },
        NewtonKind::Shifted,
    )
    .unwrap();
    assert!(report.steps <= 2, "steps {}", report.steps);
    assert!(report.grad_norm < 1e-8, "gnorm {}", report.grad_norm);
    assert_relative_eq!(report.coords[0], 0.0, epsilon = 1e-8);
    assert_relative_eq!(report.coords[1], 0.0, epsilon = 1e-8);
}

#[test]
fn rfo_kills_a_quadratic() {
    let obj = HessianOracle::unbounded(2, sphere_fg, sphere_hess);
    let report = minimize_newton(
        &obj,
        array![3.0, -4.0],
        &Control {
            maxiter: 20,
            gtol: 1e-8,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        },
        NewtonKind::Rfo,
    )
    .unwrap();
    assert!(report.grad_norm < 1e-6, "gnorm {}", report.grad_norm);
}
