//! Accepted displacements retain the equalities imposed on their QP model.

#![cfg(feature = "highs")]

use std::sync::Mutex;

use approx::assert_relative_eq;
use ndarray::array;
use rgmin::{Accept, Control, Error, ManifoldKind, Method, Oracle, Solver};

fn check_line_search_rollback(curvature: f64, minimum: f64, rhs: f64, expands: bool) {
    let trials = Mutex::new(Vec::new());
    let objective = Oracle::unbounded(1, |x| {
        trials.lock().unwrap().push(x[0]);
        let dx = x[0] - minimum;
        (0.5 * curvature * dx * dx, array![curvature * dx])
    });
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 1);
    solver.set_highs(true);
    assert!(solver.add_equality(vec![(0, 1.0)], rhs));
    let start = array![0.0];
    let mut x = start.clone();
    let result = solver.step(&objective, &mut x);
    assert!(matches!(result, Err(Error::Highs(_))), "{result:?}");
    assert_eq!(x, start);
    assert_eq!(solver.pair_count(), 0);
    assert!(trials.lock().unwrap().iter().any(|v| (*v - rhs).abs() < 1e-7));
    if expands {
        assert!(trials.lock().unwrap().iter().any(|v| *v > rhs + 1e-6));
    } else {
        assert!(trials.lock().unwrap().iter().any(|v| *v > 0.0 && *v < rhs - 1e-6));
    }

    assert!(solver.clear_equalities());
    trials.lock().unwrap().clear();
    let report = solver.step(&objective, &mut x).unwrap();
    assert_eq!(trials.lock().unwrap()[0], 0.0, "rollback must discard the cached trial");
    assert_eq!(report.steps, 1, "rejected displacements are not accepted iterations");
    assert_relative_eq!(x[0], minimum, epsilon = 1e-10);
    assert!(report.grad_norm < 1e-9, "{report:?}");
}

#[test]
fn equality_rejects_a_shrinking_line_search_and_restores_the_session() {
    check_line_search_rollback(100.0, 0.05, 0.5, false);
}

#[test]
fn equality_rejects_an_expanding_line_search_and_restores_the_session() {
    check_line_search_rollback(1.0, 1.0, 0.05, true);
}

#[test]
fn equality_rejects_a_norm_cap_that_changes_the_qp_displacement() {
    let trials = Mutex::new(Vec::new());
    let objective = Oracle::unbounded(3, |x| {
        trials.lock().unwrap().push(x.to_owned());
        let g = &x - &array![1.0, 1.0, 0.0];
        (0.5 * g.dot(&g), g)
    });
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 3);
    solver.set_accept(Accept::Step);
    solver.set_atom_maxmove(0.2);
    solver.set_highs(true);
    assert!(solver.add_equality(vec![(0, 1.0), (1, 1.0)], 0.4));
    let start = array![0.0, 0.0, 0.0];
    let mut x = start.clone();
    let result = solver.step(&objective, &mut x);
    assert!(matches!(result, Err(Error::Highs(_))), "{result:?}");
    assert_eq!(x, start);
    assert_eq!(solver.pair_count(), 0);
    let trials = trials.lock().unwrap();
    assert!(trials.len() >= 2, "the feasible QP must reach the physical norm cap");
    let trial = trials.last().unwrap();
    assert_relative_eq!(trial.dot(trial).sqrt(), 0.2, epsilon = 1e-10);
    assert!((trial[0] + trial[1] - 0.4).abs() > 0.1);
}

#[test]
fn equality_checks_the_retracted_displacement() {
    let objective = Oracle::unbounded(3, |x| (-x[1], array![0.0, -1.0, 0.0]));
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 3);
    solver.set_accept(Accept::Step);
    solver.set_manifold(ManifoldKind::Sphere);
    solver.set_highs(true);
    assert!(solver.add_equality(vec![(0, 1.0)], 0.0));
    let start = array![1.0, 0.0, 0.0];
    let mut x = start.clone();
    let result = solver.step(&objective, &mut x);
    assert!(matches!(result, Err(Error::Highs(_))), "{result:?}");
    assert_eq!(x, start);
    assert_eq!(solver.pair_count(), 0);
}

#[test]
fn equality_accepts_a_feasible_unscaled_displacement() {
    let objective = Oracle::unbounded(1, |x| {
        let dx = x[0] - 1.0;
        (0.5 * dx * dx, array![dx])
    });
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 1);
    solver.set_accept(Accept::Step);
    solver.set_highs(true);
    assert!(solver.add_equality(vec![(0, 1.0)], 0.25));
    let mut x = array![0.0];
    let report = solver.step(&objective, &mut x).unwrap();
    assert_relative_eq!(x[0], 0.25, epsilon = 1e-7);
    assert_relative_eq!(report.value, 0.28125, epsilon = 1e-7);
    assert_eq!(report.steps, 1);
    assert_eq!(solver.pair_count(), 1);
}
