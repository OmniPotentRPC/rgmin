//! PSO on Rosenbrock 2D inside the builtin box, swarm RNG seed 1.

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

#[test]
fn pso_reports_below_classic_rosenbrock_start() {
    let obj = Rosenbrock::<2>::new();
    let start = array![-1.2, 1.0];
    let f0 = obj.value_and_gradient(start.view()).0;
    let report = minimize_method(
        &obj,
        start,
        &control(),
        Method::pso(),
        LineSearch::default(),
    )
    .unwrap();
    assert!(report.value < f0, "PSO {} -> {}", f0, report.value);
}

#[test]
fn pso_seed_one_is_deterministic() {
    let obj = Rosenbrock::<2>::new();
    let start = array![-1.2, 1.0];
    let a = minimize_method(
        &obj,
        start.clone(),
        &control(),
        Method::pso(),
        LineSearch::default(),
    )
    .unwrap();
    let b = minimize_method(
        &obj,
        start,
        &control(),
        Method::pso(),
        LineSearch::default(),
    )
    .unwrap();
    assert_eq!(a.value, b.value);
    assert_eq!(a.coords, b.coords);
    assert_eq!(a.steps, b.steps);
}
