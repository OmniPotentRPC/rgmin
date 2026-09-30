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
        ftol_rel: None,
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

fn backtracking() -> LineSearch {
    LineSearch::Backtracking {
        c: 1e-4,
        beta: 0.5,
        maxiter: 40,
    }
}

/// A line search that only shrinks must not shrink the opening step of a
/// quasi-Newton method: each iteration starts at `istep`, and L-BFGS
/// reaches the minimum of Rosenbrock instead of stalling.
#[test]
fn lbfgs_under_backtracking_opens_every_search_at_istep() {
    let obj = Rosenbrock::<2>::new();
    let mut c = control();
    c.maxiter = 2000;
    c.gtol = 1e-6;
    let report = minimize_method(
        &obj,
        array![-1.2, 1.0],
        &c,
        Method::Lbfgs { memory: 10 },
        backtracking(),
    )
    .unwrap();
    assert!(
        report.grad_norm < c.gtol,
        "stalled at ||g|| = {} after {} steps",
        report.grad_norm,
        report.steps
    );
    assert!(report.steps < c.maxiter, "ran to maxiter");
    assert_relative_eq!(report.coords[0], 1.0, epsilon = 1e-3);
    assert_relative_eq!(report.coords[1], 1.0, epsilon = 1e-3);
}

/// BFGS under the same search, opening at the unit step a quasi-Newton
/// direction is scaled for, reaches the same point.
#[test]
fn bfgs_under_backtracking_opens_every_search_at_istep() {
    let obj = Rosenbrock::<2>::new();
    let mut c = control();
    c.maxiter = 2000;
    c.gtol = 1e-6;
    c.istep = 1.0;
    let report =
        minimize_method(&obj, array![-1.2, 1.0], &c, Method::Bfgs, backtracking()).unwrap();
    assert!(report.grad_norm < c.gtol, "||g|| = {}", report.grad_norm);
    assert_relative_eq!(report.coords[0], 1.0, epsilon = 1e-3);
    assert_relative_eq!(report.coords[1], 1.0, epsilon = 1e-3);
}
