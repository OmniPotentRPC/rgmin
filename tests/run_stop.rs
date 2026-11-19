//! `Solver::run` reports the stop a step took, and `Budget` when it spent
//! its budget above every stop.

use ndarray::{ArrayView1, array};
use rgmin::{
    Control, FireKind, LineSearch, Method, NoiseFloor, Oracle, Solver, Status, minimize_method,
};

fn rosenbrock() -> impl eindir_core::DifferentiableObjective<f64> {
    Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let (a, b) = (x[0], x[1]);
        (
            (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2),
            array![
                -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
                200.0 * (b - a * a)
            ],
        )
    })
}

/// Three steps from Rosenbrock's start leave every method above `1e-10`.
#[test]
fn a_spent_budget_is_budget_not_iterating() {
    for method in [
        Method::polak_ribiere(),
        Method::Bfgs,
        Method::lbfgs(),
        Method::Sr1,
        Method::Steepest,
        Method::adam(),
        Method::fire(FireKind::V1),
        Method::Bb,
        Method::QuickMin,
        Method::MdMin,
        Method::diis(),
        Method::NewtonCg,
    ] {
        let control = Control::default().with_maxiter(3).with_gtol(1e-10);
        let mut solver = Solver::new(method.clone(), control, 2);
        let mut x = array![-1.2, 1.0];
        let report = solver.run(&rosenbrock(), &mut x).unwrap();
        assert_eq!(solver.steps(), 3, "{method:?}");
        assert!(report.grad_norm >= 1e-10, "{method:?}: {report:?}");
        assert_eq!(report.status, Status::Budget, "{method:?}: {report:?}");
    }
}

/// With `eps_f = 1` no trial on `1/2 x^2` from `0.5` clears the energy
/// noise, so these steps report `NoiseFloor` and keep the start. The run
/// ends there instead of repeating the refused search to its budget.
#[test]
fn a_run_stops_on_the_energy_noise_floor_a_step_reports() {
    let quad = Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]));
    for method in [
        Method::Bfgs,
        Method::Steepest,
        Method::polak_ribiere(),
        Method::adam(),
        Method::diis(),
    ] {
        let control = Control::default()
            .with_maxiter(20)
            .with_gtol(1e-8)
            .with_noise_floor(NoiseFloor::new(0.0).with_eps_f(1.0));
        let mut probe = Solver::new(method.clone(), control.clone(), 1);
        let mut x = array![0.5];
        assert_eq!(
            probe.step(&quad, &mut x).unwrap().status,
            Status::NoiseFloor,
            "{method:?}"
        );
        let mut solver = Solver::new(method.clone(), control, 1);
        let mut x = array![0.5];
        let report = solver.run(&quad, &mut x).unwrap();
        assert_eq!(report.status, Status::NoiseFloor, "{method:?}: {report:?}");
        assert_eq!(solver.steps(), 1, "{method:?}");
        assert_eq!(report.value, 0.125, "{method:?}: {report:?}");
    }
}

/// The session-backed one-shot ends on the same stop. DIIS reported it
/// after spending all twenty steps on the refused search.
#[test]
fn a_session_one_shot_stops_on_the_energy_noise_floor() {
    let quad = Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]));
    let control = Control::default()
        .with_maxiter(20)
        .with_gtol(1e-8)
        .with_noise_floor(NoiseFloor::new(0.0).with_eps_f(1.0));
    let report = minimize_method(
        &quad,
        array![0.5],
        &control,
        Method::diis(),
        LineSearch::default(),
    )
    .unwrap();
    assert_eq!(report.status, Status::NoiseFloor, "{report:?}");
    assert_eq!(report.steps, 1, "{report:?}");
    assert_eq!(report.coords, array![0.5]);
}
