use ndarray::{ArrayView1, array};
use rgmin::{Control, HessianOracle, NewtonKind, minimize_newton};

#[test]
fn rfo_minimization_selects_the_lowest_augmented_mode() {
    let objective = HessianOracle::unbounded(
        1,
        |x: ArrayView1<f64>| {
            let q = x[0];
            (
                0.25 * q.powi(4) - 0.5 * q * q + 0.1 * q,
                array![q.powi(3) - q + 0.1],
            )
        },
        |x: ArrayView1<f64>| array![[3.0 * x[0] * x[0] - 1.0]],
    );
    let report = minimize_newton(
        &objective,
        array![0.0],
        &Control {
            maxiter: 1,
            gtol: 1e-12,
            istep: 1.0,
            maxmove: Some(0.2),
            ftol_rel: None,
        },
        NewtonKind::Rfo,
    )
    .unwrap();

    // The lowest eigenvalue of [[-1, 0.1], [0.1, 0]] is negative.
    // Its RFO direction points downhill and reaches the movement limit.
    assert!(
        (report.coords[0] + 0.2).abs() < 1e-12,
        "RFO step {} does not follow the lowest augmented mode",
        report.coords[0]
    );
    assert!((report.value + 0.0396).abs() < 1e-12);
}

#[cfg(feature = "highs")]
#[test]
fn constrained_rfo_keeps_every_trial_inside_the_trust_and_box() {
    use rgmin::{Accept, Method, Solver};
    use std::sync::Mutex;

    let visited = Mutex::new(Vec::new());
    let objective = HessianOracle::unbounded(
        1,
        |x: ArrayView1<f64>| {
            let q = x[0];
            visited.lock().unwrap().push(q);
            (
                0.25 * q.powi(4) - 0.5 * q * q + 0.1 * q,
                array![q.powi(3) - q + 0.1],
            )
        },
        |x: ArrayView1<f64>| array![[3.0 * x[0] * x[0] - 1.0]],
    );
    let mut solver = Solver::new(
        Method::Newton {
            kind: NewtonKind::Rfo,
        },
        Control {
            maxiter: 1,
            gtol: 1e-12,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        },
        1,
    );
    solver.set_highs(true);
    solver.set_accept(Accept::Energy);
    assert!(solver.set_box(Some(vec![-0.002]), Some(vec![0.002])));
    assert!(solver.set_trust(0.001));
    let report = solver.step_hess(&objective, &mut array![0.0]).unwrap();

    for q in visited.lock().unwrap().iter() {
        assert!(
            q.abs() <= 0.001 + 1e-12,
            "trial {q} leaves the feasible set"
        );
    }
    assert!((report.coords[0] + 0.001).abs() < 1e-9);
}

#[cfg(feature = "highs")]
#[test]
fn inactive_constraints_preserve_the_rfo_model() {
    use rgmin::{Method, Solver};

    let objective = HessianOracle::unbounded(
        1,
        |x: ArrayView1<f64>| ((x[0] - 1.0).powi(2), array![2.0 * (x[0] - 1.0)]),
        |_x: ArrayView1<f64>| array![[2.0]],
    );
    let mut solver = Solver::new(
        Method::Newton {
            kind: NewtonKind::Rfo,
        },
        Control {
            maxiter: 1,
            gtol: 1e-12,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        },
        1,
    );
    solver.set_highs(true);
    assert!(solver.set_box(Some(vec![-2.0]), Some(vec![2.0])));
    let report = solver.step_hess(&objective, &mut array![0.0]).unwrap();
    let expected = (5.0_f64.sqrt() - 1.0) / 2.0;
    assert!(
        (report.coords[0] - expected).abs() < 1e-6,
        "RFO step {} differs from the augmented model {expected}",
        report.coords[0]
    );
}

#[cfg(feature = "highs")]
#[test]
fn constrained_rfo_rejects_without_an_unconstrained_fallback() {
    use rgmin::{Accept, Method, Solver};
    use std::sync::Mutex;

    let visited = Mutex::new(Vec::new());
    let objective = HessianOracle::unbounded(
        1,
        |x: ArrayView1<f64>| {
            let q = x[0];
            visited.lock().unwrap().push(q);
            (q + 1e30 * q.powi(4), array![1.0 + 4e30 * q.powi(3)])
        },
        |x: ArrayView1<f64>| array![[12e30 * x[0] * x[0]]],
    );
    let mut solver = Solver::new(
        Method::Newton {
            kind: NewtonKind::Rfo,
        },
        Control {
            maxiter: 1,
            gtol: 1e-12,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        },
        1,
    );
    solver.set_highs(true);
    solver.set_accept(Accept::Energy);
    assert!(solver.set_box(Some(vec![-0.002]), Some(vec![0.002])));
    assert!(solver.set_trust(0.001));
    let report = solver.step_hess(&objective, &mut array![0.0]).unwrap();

    for q in visited.lock().unwrap().iter() {
        assert!(
            q.abs() <= 0.001 + 1e-12,
            "rejected trial {q} leaves the feasible set"
        );
    }
    assert_eq!(report.coords[0], 0.0);
}
