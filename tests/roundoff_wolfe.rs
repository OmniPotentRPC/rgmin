//! Wolfe convergence with cancellation in the computed energy.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, Error, Lbfgs, LbfgsStep, LineSearch, LineSearchStopReason, Method, Solver, StopReason,
};
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
    LineSearch::wolfe(1e-4, 0.9, 20)
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
            Control::default().with_gtol(1e-12).with_ftol_rel(slack),
            1,
        );
        solver.set_lbfgs_step(rgmin::LbfgsStep::Wolfe).unwrap();
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
    let mut solver = Solver::new(Method::lbfgs(), Control::default().with_gtol(1e-12), 1);
    solver.set_lbfgs_step(rgmin::LbfgsStep::Wolfe).unwrap();
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

struct RoundedBowl {
    scale: f64,
    offset: f64,
    bounds: Bounds<f64>,
    calls: AtomicUsize,
}

impl RoundedBowl {
    fn new() -> Self {
        Self {
            scale: f64::EPSILON.sqrt().sqrt(),
            offset: OFFSET,
            bounds: Bounds::new(array![f64::NEG_INFINITY], array![f64::INFINITY], 0.0),
            calls: AtomicUsize::new(0),
        }
    }
}

impl Objective<f64> for RoundedBowl {
    fn dim(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let exact = self.offset + 0.5 * self.scale * x[0] * x[0];
        let cancellation = x[0] / f64::EPSILON.sqrt();
        (exact + cancellation) - cancellation
    }
}

impl Gradient<f64> for RoundedBowl {
    fn dim(&self) -> usize {
        1
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![self.scale * x[0]]
    }
}

impl DifferentiableObjective<f64> for RoundedBowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn a_rounding_flat_bracket_stops_without_the_steepest_retry_tail() {
    let objective = RoundedBowl::new();
    let mut point = array![2.0 * f64::EPSILON.sqrt()];
    let base_gradient = objective.grad(point.view());
    let mut solver = Solver::new(Method::lbfgs(), Control::default().with_gtol(0.0), 1);
    solver.set_lbfgs_step(rgmin::LbfgsStep::Wolfe).unwrap();
    assert!(solver.push_pair(array![1.0].view(), array![1.0].view()));
    assert_eq!(solver.pair_count(), 1);

    let report = solver.run(&objective, &mut point).unwrap();

    assert_eq!(report.stop_reason(), StopReason::LineSearchRounding);
    assert_eq!(point, array![2.0 * f64::EPSILON.sqrt()]);
    assert_eq!(report.grad.as_ref(), Some(&base_gradient));
    assert!(objective.calls.load(Ordering::Relaxed) <= 16);
    assert_eq!(solver.pair_count(), 1);
}

#[test]
fn the_detailed_search_reports_a_typed_rounding_stop() {
    let objective = RoundedBowl::new();
    let start = array![2.0 * f64::EPSILON.sqrt()];
    let value = objective.eval(start.view());
    let gradient = objective.grad(start.view());
    let direction = -&gradient;
    let result = wolfe().search_from_result(
        |x| objective.value_and_gradient(x),
        start.view(),
        value,
        gradient.view(),
        direction.view(),
        1.0,
        f64::INFINITY,
    );

    assert_eq!(
        result.stop_reason,
        Some(LineSearchStopReason::RoundingBlocked)
    );
    assert!(result.step.is_none());
    assert!(objective.calls.load(Ordering::Relaxed) <= 10);
}

#[test]
fn direct_lbfgs_stops_the_rounding_tail_without_changing_the_gradient() {
    let objective = RoundedBowl::new();
    let start = array![2.0 * f64::EPSILON.sqrt()];
    let base_gradient = objective.grad(start.view());
    let mut solver = Lbfgs::default();
    solver.step = LbfgsStep::Wolfe;
    solver.gtol = 0.0;
    solver.record(array![1.0], array![2.0]);
    assert_eq!(solver.len(), 1);

    let (_, point, evals) =
        solver.minimize(start.view(), 100, |x| Some(objective.value_and_gradient(x)));

    assert_eq!(point, start);
    assert_eq!(objective.grad(point.view()), base_gradient);
    assert_eq!(solver.len(), 1);
    assert!(evals <= 4);
}

#[test]
fn error_on_unmet_carries_the_rounding_stop() {
    let objective = RoundedBowl::new();
    let mut point = array![2.0 * f64::EPSILON.sqrt()];
    let control = Control::default().with_gtol(0.0).with_error_on_unmet(true);
    let mut solver = Solver::new(Method::lbfgs(), control, 1);
    solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
    assert!(solver.push_pair(array![1.0].view(), array![1.0].view()));

    let error = solver.run(&objective, &mut point).unwrap_err();

    match error {
        Error::StopUnmet { reason, report } => {
            assert_eq!(reason, StopReason::LineSearchRounding);
            assert_eq!(report.stop_reason(), reason);
        }
        other => panic!("unexpected error {other}"),
    }
    assert_eq!(solver.pair_count(), 1);
}
