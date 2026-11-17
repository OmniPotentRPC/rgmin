mod common;

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Oracle, Solver, Status};

use common::morse_value_grad;

fn control(h: f64, gtol: f64, maxiter: usize) -> Control {
    Control::default()
        .with_istep(h)
        .with_gtol(gtol)
        .with_maxiter(maxiter)
        .with_maxmove(None)
}

#[test]
fn morse_dimer_has_the_exact_accept_reject_h_sequence() {
    let objective = Oracle::unbounded(6, |x: ArrayView1<f64>| {
        morse_value_grad(x, [100.0, 100.0, 100.0])
    });
    let mut x = array![0.0, 0.0, 0.0, 2.8, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsSteep, control(0.1, 0.0, 10), x.len());

    let expected: [f64; 3] = [0.12, 0.024, 0.0288];
    let expected_positions: [f64; 3] = [3.0, 3.0, 2.952];
    for (&h, &distance) in expected.iter().zip(&expected_positions) {
        let report = solver.step(&objective, &mut x).unwrap();
        assert_eq!(report.status, Status::Iterating);
        assert_eq!(solver.gromacs_step_size().unwrap().to_bits(), h.to_bits());
        assert!((x[3] - x[0] - distance).abs() < 1e-14);
    }
}

#[test]
fn tolerance_budget_and_machine_precision_are_distinct_stops() {
    let quadratic = Oracle::unbounded(3, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
    let mut at_tolerance = array![1.0, 0.0, 0.0];
    let mut solver = Solver::new(
        Method::GromacsSteep,
        control(0.1, 1.1, 5),
        at_tolerance.len(),
    );
    let report = solver.run(&quadratic, &mut at_tolerance).unwrap();
    assert_eq!(report.status, Status::Tolerance);
    assert_eq!(report.steps, 0);

    let mut budgeted = array![1.0, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsSteep, control(0.1, 0.0, 2), budgeted.len());
    let report = solver.run(&quadratic, &mut budgeted).unwrap();
    assert_eq!(report.status, Status::Budget);
    assert_eq!(report.steps, 2);

    let linear = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        (x[0], Array1::from_vec(vec![1.0, 0.0, 0.0]))
    });
    let mut rounded = array![1.0, 0.0, 0.0];
    let mut solver = Solver::new(
        Method::GromacsSteep,
        control(f64::MIN_POSITIVE, 0.0, 5),
        rounded.len(),
    );
    let report = solver.run(&linear, &mut rounded).unwrap();
    assert_eq!(report.status, Status::MachinePrecision);
    assert_eq!(rounded[0], 1.0);
}
