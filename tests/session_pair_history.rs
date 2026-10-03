use ndarray::{ArrayView1, array};
use rgmin::{Control, Lbfgs, LineSearch, Method, Oracle, Solver};
use std::sync::atomic::{AtomicUsize, Ordering};

fn seeded_session(maxmove: Option<f64>) -> Solver {
    let mut solver = Solver::new(
        Method::Lbfgs { memory: 1 },
        Control {
            maxiter: 10,
            gtol: 1e-10,
            istep: 0.1,
            maxmove,
            ftol_rel: None,
        },
        1,
    );
    assert!(solver.push_pair(array![1.0].view(), array![2.0].view()));
    assert_eq!(solver.pair_count(), 1);
    solver
}

#[test]
fn a_zero_curvature_step_keeps_the_retained_pair() {
    let calls = AtomicUsize::new(0);
    let objective = Oracle::unbounded(1, |x: ArrayView1<f64>| {
        calls.fetch_add(1, Ordering::SeqCst);
        (-x[0], array![-1.0])
    });
    let mut solver = seeded_session(Some(0.2));
    let mut x = array![0.0];
    let report = solver.step(&objective, &mut x).unwrap();
    assert_eq!(x, array![0.2]);
    assert_eq!(report.value, -0.2);
    assert_eq!(report.steps, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(solver.pair_count(), 1);
    assert_eq!(
        solver.search_direction(array![-1.0].view()).unwrap(),
        array![0.5]
    );
}

#[test]
fn a_cautiously_refused_pair_does_not_evict_full_history() {
    let calls = AtomicUsize::new(0);
    let objective = Oracle::unbounded(1, |x: ArrayView1<f64>| {
        calls.fetch_add(1, Ordering::SeqCst);
        (0.5 * x[0] * x[0], array![x[0]])
    });
    let mut solver = seeded_session(None);
    solver.set_cautious(2.0, 0.0);
    let mut x = array![1.0];
    let report = solver.step(&objective, &mut x).unwrap();
    assert_eq!(x, array![0.5]);
    assert_eq!(report.value, 0.125);
    assert_eq!(report.steps, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(solver.pair_count(), 1);
    assert_eq!(
        solver.search_direction(array![1.0].view()).unwrap(),
        array![-0.5]
    );
}

#[test]
fn a_standalone_step_records_its_accepted_pair() {
    let calls = AtomicUsize::new(0);
    let objective = Oracle::unbounded(1, |x: ArrayView1<f64>| {
        calls.fetch_add(1, Ordering::SeqCst);
        (0.5 * x[0] * x[0], array![x[0]])
    });
    let mut solver = Lbfgs::with_capacity(1);
    solver.record(array![1.0], array![2.0]);
    let mut x = array![1.0];
    let mut value = 0.5;
    let mut gradient = array![1.0];
    let mut istep = 0.1;
    solver.step_objective(
        &objective,
        &mut x,
        &mut value,
        &mut gradient,
        &mut istep,
        LineSearch::Wolfe {
            c1: 1e-4,
            c2: 0.9,
            maxiter: 20,
        },
        &Control {
            maxmove: None,
            ..Control::default()
        },
        None,
    );
    assert_eq!(x, array![0.5]);
    assert_eq!(value, 0.125);
    assert_eq!(gradient, array![0.5]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(solver.len(), 1);
    assert_eq!(solver.two_loop(array![1.0].view()), array![-1.0]);
}
