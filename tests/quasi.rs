//! Quasi-Newton and Adam on Rosenbrock 2D.

use approx::assert_relative_eq;
use eindir_core::DifferentiableObjective;
use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use rgmin::{Control, LineSearch, Method, minimize_method};

fn control() -> Control {
    Control {
        maxiter: 200,
        gtol: 1e-8,
        istep: 0.1,
        maxmove: None,
        ..Default::default()
    }
}

fn brent() -> LineSearch {
    LineSearch::Brent {
        maxiter: 40,
        tol: 1e-12,
    }
}

fn f0() -> f64 {
    let obj = Rosenbrock::<2>::new();
    obj.value_and_gradient(array![-1.2, 1.0].view()).0
}

#[test]
fn lbfgs_finds_the_banana_minimum() {
    let obj = Rosenbrock::<2>::new();
    let report = minimize_method(
        &obj,
        array![-1.2, 1.0],
        &control(),
        Method::lbfgs(),
        brent(),
    )
    .unwrap();
    assert!(report.value < 1e-8, "L-BFGS value {}", report.value);
    assert_relative_eq!(report.coords[0], 1.0, epsilon = 1e-4);
    assert_relative_eq!(report.coords[1], 1.0, epsilon = 1e-4);
}

#[test]
fn bfgs_and_sr1_reach_the_minimum() {
    let obj = Rosenbrock::<2>::new();
    for method in [Method::Bfgs, Method::Sr1] {
        let report =
            minimize_method(&obj, array![-1.2, 1.0], &control(), method.clone(), brent()).unwrap();
        assert!(report.value < 1e-8, "{method:?} value {}", report.value);
    }
}

#[test]
fn sr2_descends_from_the_classic_start() {
    let obj = Rosenbrock::<2>::new();
    let start_f = f0();
    let report =
        minimize_method(&obj, array![-1.2, 1.0], &control(), Method::Sr2, brent()).unwrap();
    assert!(
        report.value < start_f,
        "SR2 {} -> {}",
        start_f,
        report.value
    );
}

#[test]
fn adam_and_steepest_descend() {
    let obj = Rosenbrock::<2>::new();
    let start_f = f0();
    let adam =
        minimize_method(&obj, array![-1.2, 1.0], &control(), Method::adam(), brent()).unwrap();
    assert!(adam.value < start_f, "Adam {} -> {}", start_f, adam.value);
    let sd = minimize_method(
        &obj,
        array![-1.2, 1.0],
        &control(),
        Method::Steepest,
        brent(),
    )
    .unwrap();
    assert!(sd.value < start_f, "SD {} -> {}", start_f, sd.value);
}
