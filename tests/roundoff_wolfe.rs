//! Wolfe convergence with cancellation in the computed energy.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, LineSearch, Method, Solver};
use std::sync::atomic::{AtomicUsize, Ordering};

const OFFSET: f64 = 6.000000000000001;

struct Bowl {
    center: f64,
    cancellation: bool,
    bounds: Bounds<f64>,
    calls: AtomicUsize,
}
impl Bowl {
    fn cancellation() -> Self {
        Self {
            center: 0.0,
            cancellation: true,
            bounds: Bounds::new(array![-1.0], array![1.0], 0.0),
            calls: AtomicUsize::new(0),
        }
    }
}
impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let delta = x[0] - self.center;
        let energy = OFFSET + 0.5 * delta * delta;
        if self.cancellation {
            let term = 1e8 * delta;
            (energy + term) - term
        } else {
            energy
        }
    }
}
impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![x[0] - self.center]
    }
}
impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        (self.eval(x), self.grad(x))
    }
}
fn wolfe() -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 20,
    }
}

#[test]
fn known_start_reaches_the_minimum_with_one_trial_despite_one_ulp_energy_rise() {
    let objective = Bowl::cancellation();
    let start = array![2e-8];
    let gradient = objective.grad(start.view());
    let value = objective.eval(start.view());
    assert_eq!(value, 6.0);
    assert!(objective.eval(array![0.0].view()) > value);
    let result = wolfe()
        .search_from(
            |x| objective.value_and_gradient(x),
            start.view(),
            value,
            gradient.view(),
            (-&gradient).view(),
            1.0,
            f64::INFINITY,
        )
        .expect("the exact stationary trial lies within the energy precision window");
    assert_eq!(result.x, array![0.0]);
    assert_eq!(result.g, array![0.0]);
    assert_eq!(result.f, OFFSET);
    assert_eq!(result.evals, 1);
    assert_eq!(objective.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn lbfgs_retains_the_stationary_trial_and_honors_explicit_zero_slack() {
    for slack in [None, Some(0.0)] {
        let objective = Bowl::cancellation();
        let mut point = array![2e-8];
        let mut solver = Solver::new(
            Method::lbfgs(),
            Control {
                gtol: 1e-12,
                ftol_rel: slack,
                ..Control::default()
            },
            1,
        );
        let report = solver.step(&objective, &mut point).unwrap();
        if slack.is_none() {
            assert_eq!(point, array![0.0]);
            assert_eq!(report.grad_norm, 0.0);
            assert_eq!(report.value, OFFSET);
        } else {
            assert_eq!(point, array![2e-8]);
            assert_eq!(report.grad_norm, 2e-8);
            assert_eq!(report.value, 6.0);
        }
        assert_eq!(objective.calls.load(Ordering::Relaxed), 2);
    }
}

#[test]
fn a_box_modified_trial_must_satisfy_its_own_curvature_check() {
    let objective = Bowl {
        center: 2e-8,
        cancellation: false,
        bounds: Bounds::new(array![-1.0], array![1e-10], 0.0),
        calls: AtomicUsize::new(0),
    };
    let mut point = array![0.0];
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            gtol: 1e-12,
            ..Control::default()
        },
        1,
    );
    let report = solver.step(&objective, &mut point).unwrap();
    assert_eq!(point, array![0.0]);
    assert_eq!(report.grad_norm, 2e-8);
    assert_eq!(objective.calls.load(Ordering::Relaxed), 3);
}

#[test]
fn a_resolved_energy_increase_is_rejected_even_at_zero_trial_gradient() {
    let result = wolfe().search_from(
        |_| (6.0 + 1e-10, array![0.0]),
        array![2e-8].view(),
        6.0,
        array![2e-8].view(),
        array![-2e-8].view(),
        1.0,
        1.0,
    );
    assert!(result.is_none());
}

#[test]
fn an_unresolved_energy_rise_without_curvature_improvement_is_rejected() {
    for trial_gradient in [2e-8, f64::NAN, f64::INFINITY] {
        let result = wolfe().search_from(
            |_| (OFFSET, array![trial_gradient]),
            array![2e-8].view(),
            6.0,
            array![2e-8].view(),
            array![-2e-8].view(),
            1.0,
            1.0,
        );
        assert!(result.is_none());
    }
}

#[test]
fn standalone_zoom_uses_the_same_precision_limit() {
    let objective = Bowl::cancellation();
    let start = array![2e-8];
    let direction = array![-2e-8];
    let alpha = rgmin::linesearch::zoom(
        &mut |x| objective.value_and_gradient(x),
        start.view(),
        direction.view(),
        0.0,
        2.0,
        1e-4,
        0.9,
        20,
    );
    assert_eq!(alpha, 1.0);
    assert_eq!(objective.calls.load(Ordering::Relaxed), 2);
}
