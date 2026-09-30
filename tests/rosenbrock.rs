//! Rosenbrock 2D must descend under every conjugacy + Brent.

use approx::assert_relative_eq;
use eindir_core::DifferentiableObjective;
use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use rgmin::{Conjugacy, Control, LineSearch, Restart, minimize};

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

#[test]
fn polak_ribiere_finds_the_banana_minimum() {
    let obj = Rosenbrock::<2>::new();
    let report = minimize(
        &obj,
        array![-1.2, 1.0],
        &control(),
        Conjugacy::PolakRibiere,
        Restart::Never,
        brent(),
    )
    .unwrap();
    assert!(report.value < 1e-8, "value {}", report.value);
    assert_relative_eq!(report.coords[0], 1.0, epsilon = 1e-4);
    assert_relative_eq!(report.coords[1], 1.0, epsilon = 1e-4);
}

#[test]
fn every_conjugacy_descends_from_the_classic_start() {
    let obj = Rosenbrock::<2>::new();
    let start = array![-1.2, 1.0];
    let f0 = obj.value_and_gradient(start.view()).0;
    for c in [
        Conjugacy::FletcherReeves,
        Conjugacy::PolakRibiere,
        Conjugacy::HestenesStiefel,
        Conjugacy::DaiYuan,
        Conjugacy::ConjugateDescent,
        Conjugacy::HagerZhang,
        Conjugacy::LiuStorey,
        Conjugacy::FrPr,
    ] {
        let report = minimize(
            &obj,
            start.clone(),
            &control(),
            c.clone(),
            Restart::Never,
            brent(),
        )
        .unwrap();
        assert!(
            report.value < f0,
            "{c:?} did not descend: {} -> {}",
            f0,
            report.value
        );
    }
}

#[test]
fn njws_restart_still_descends() {
    let obj = Rosenbrock::<2>::new();
    let start = array![-1.2, 1.0];
    let f0 = obj.value_and_gradient(start.view()).0;
    let report = minimize(
        &obj,
        start,
        &control(),
        Conjugacy::FletcherReeves,
        Restart::njws(),
        brent(),
    )
    .unwrap();
    assert!(
        report.value < f0 * 0.1,
        "NJWS FR {} -> {}",
        f0,
        report.value
    );
}

#[test]
fn armijo_backtracking_descends() {
    let obj = Rosenbrock::<2>::new();
    let start = array![-1.2, 1.0];
    let f0 = obj.value_and_gradient(start.view()).0;
    let report = minimize(
        &obj,
        start,
        &control(),
        Conjugacy::PolakRibiere,
        Restart::Never,
        LineSearch::Backtracking {
            c: 1e-4,
            beta: 0.5,
            maxiter: 40,
        },
    )
    .unwrap();
    assert!(report.value < f0, "Armijo {} -> {}", f0, report.value);
}

#[test]
fn dim_mismatch_is_an_error() {
    let obj = Rosenbrock::<2>::new();
    let err = minimize(
        &obj,
        array![0.0],
        &control(),
        Conjugacy::PolakRibiere,
        Restart::Never,
        brent(),
    )
    .unwrap_err();
    match err {
        rgmin::Error::Dim { got, dim } => {
            assert_eq!(got, 1);
            assert_eq!(dim, 2);
        }
        other => panic!("expected Dim, got {other}"),
    }
}
