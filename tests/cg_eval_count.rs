//! One-shot conjugate gradients keep the half-step opening's evaluation
//! counts on Rosenbrock. The slope-ratio opening is off unless requested.

use ndarray::{Array1, ArrayView1};
use rgmin::{Conjugacy, Control, LineSearch, Oracle, Restart, Status, minimize};

fn rosenbrock(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len();
    let mut value = 0.0;
    let mut grad = Array1::zeros(n);
    for i in 0..n - 1 {
        let gap = 1.0 - x[i];
        let curve = x[i + 1] - x[i] * x[i];
        value += gap * gap + 100.0 * curve * curve;
        grad[i] += -2.0 * gap - 400.0 * x[i] * curve;
        grad[i + 1] += 200.0 * curve;
    }
    (value, grad)
}

fn start(n: usize) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }))
}

fn control(maxiter: usize, gtol: f64) -> Control {
    Control::default()
        .with_maxiter(maxiter)
        .with_gtol(gtol)
        .with_istep(1.0)
        .with_maxmove(-1.0)
}

fn wolfe() -> LineSearch {
    LineSearch::wolfe(1e-4, 0.1, 20)
}

fn run(n: usize, conj: Conjugacy, maxiter: usize, gtol: f64) -> rgmin::Report {
    let obj = Oracle::unbounded(n, move |x: ArrayView1<f64>| rosenbrock(x));
    minimize(
        &obj,
        start(n),
        &control(maxiter, gtol),
        conj,
        Restart::Never,
        wolfe(),
    )
    .expect("minimize")
}

#[test]
fn fletcher_reeves_keeps_seventy_six_evaluations_over_twenty_steps() {
    let report = run(2, Conjugacy::FletcherReeves, 20, 0.0);
    assert_eq!(report.steps, 20, "{report:?}");
    assert_eq!(report.status, Status::Budget);
    assert_eq!(report.evals, 76, "{report:?}");
}

#[test]
fn fletcher_reeves_reaches_gtol_without_stopping_at_iteration_sixteen() {
    let report = run(2, Conjugacy::FletcherReeves, 100, 1e-5);
    assert_eq!(report.status, Status::Tolerance);
    assert_eq!(report.steps, 21, "{report:?}");
    assert_eq!(report.evals, 80, "{report:?}");
}

#[test]
fn polak_ribiere_keeps_the_two_dimensional_counts() {
    let fixed = run(2, Conjugacy::PolakRibiere, 20, 0.0);
    assert_eq!(fixed.steps, 20);
    assert_eq!(fixed.evals, 120, "{fixed:?}");
    let solved = run(2, Conjugacy::PolakRibiere, 500, 1e-5);
    assert_eq!(solved.status, Status::Tolerance);
    assert_eq!(solved.steps, 11, "{solved:?}");
    assert_eq!(solved.evals, 48, "{solved:?}");
}

#[test]
fn polak_ribiere_keeps_the_ten_dimensional_counts() {
    let fixed = run(10, Conjugacy::PolakRibiere, 20, 0.0);
    assert_eq!(fixed.steps, 20);
    assert_eq!(fixed.evals, 58, "{fixed:?}");
    let solved = run(10, Conjugacy::PolakRibiere, 2000, 1e-5);
    assert_eq!(solved.status, Status::Tolerance);
    assert_eq!(solved.steps, 278, "{solved:?}");
    assert_eq!(solved.evals, 638, "{solved:?}");
}
