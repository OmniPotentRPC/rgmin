//! PSO on Rosenbrock 2D inside the builtin box, swarm RNG seed 1.

use eindir_core::DifferentiableObjective;
use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use rgmin::{Control, IterationView, LineSearch, Method, ObserverSignal, minimize_method};

fn control() -> Control {
    Control::default()
        .with_maxiter(200)
        .with_gtol(1e-8)
        .with_istep(0.1)
        .with_maxmove(None)
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

#[test]
fn an_observer_does_not_add_objective_evaluations() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let calls = AtomicUsize::new(0);
    let obj = rgmin::Oracle::unbounded(2, |x| {
        calls.fetch_add(1, Ordering::Relaxed);
        (0.5 * x.dot(&x), x.to_owned())
    });
    let control = Control::default().with_maxiter(4).with_gtol(1e-16);
    let start = array![1.0, -0.5];
    let plain = rgmin::minimize_pso_observed(&obj, start.clone(), &control, 8, 0.5, 1.5, 1.5, None)
        .expect("plain swarm");
    let plain_calls = calls.load(Ordering::Relaxed);
    calls.store(0, Ordering::Relaxed);
    let mut seen = 0usize;
    let observed = rgmin::minimize_pso_observed(
        &obj,
        start,
        &control,
        8,
        0.5,
        1.5,
        1.5,
        Some(&mut |state: IterationView<'_>| {
            seen += 1;
            assert!(state.grad_norm.is_nan(), "gradient {}", state.grad_norm);
            let _ = (state.value, state.step_length);
            ObserverSignal::Continue
        }),
    )
    .expect("observed swarm");
    assert!(seen > 0, "the observer was not called");
    assert_eq!(
        calls.load(Ordering::Relaxed),
        plain_calls,
        "observed oracle calls differ from an unobserved run ({seen} observer calls)"
    );
    assert_eq!(observed.coords, plain.coords);
    assert_eq!(observed.value, plain.value);
    assert_eq!(observed.evals, plain.evals);
}
