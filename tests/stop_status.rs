//! Stop status, a resumable L-BFGS session, and the backward-error polish.
//!
//! On `eb8d3e4123183bd3f5ca82852332eeddb5d17f54` the short Rosenbrock run
//! returns `Ok` with `Status::Budget` and no `StopReason`. The same call
//! with the unmet-stop flag does not exist, so it cannot be an error.
//! `minimize_newton` on a point already under `gtol` returns that point
//! even when its gradient is above the first-order floor. The resumed
//! session below does not raise the gradient; a fresh session does.

use eindir_core::{DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, Error, HessianObjective, HessianOracle, HessianVector, Lbfgs, LineSearch, Method,
    NewtonKind, Solver, Status, StopReason, backward_error_floor, minimize_lbfgs, minimize_method,
    minimize_newton, newton_decrement_acceptable, polish_hvp, polish_newton,
};

fn l2(g: &Array1<f64>) -> f64 {
    g.dot(g).sqrt()
}

fn rose() -> impl DifferentiableObjective<f64> {
    rgmin::Oracle::unbounded(2, |x: ArrayView1<f64>| {
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

fn quad_hess() -> impl HessianObjective {
    HessianOracle::unbounded(
        1,
        |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]),
        |_| array![[1.0]],
    )
}

struct IdentityHvp;

impl eindir_core::Objective<f64> for IdentityHvp {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        use std::sync::OnceLock;
        static BOUNDS: OnceLock<eindir_core::Bounds<f64>> = OnceLock::new();
        BOUNDS.get_or_init(|| eindir_core::Bounds::new(array![-1.0e6], array![1.0e6], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * x[0] * x[0]
    }
}

impl eindir_core::Gradient<f64> for IdentityHvp {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![x[0]]
    }
}

impl DifferentiableObjective<f64> for IdentityHvp {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

impl HessianVector for IdentityHvp {
    fn hessian_vector(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        v.to_owned()
    }
}

/// A spent budget above the floor is `MaxIter`, and the default is still `Ok`.
#[test]
fn a_short_rosenbrock_run_is_maxiter_not_converged() {
    let control = Control::default().with_maxiter(6).with_gtol(1e-12);
    let report = minimize_method(
        &rose(),
        array![-1.2, 1.0],
        &control,
        Method::lbfgs(),
        LineSearch::default(),
    )
    .unwrap();
    let floor = backward_error_floor(report.value, report.coords.view());
    assert!(report.grad_norm > floor);
    assert_eq!(report.status, Status::Budget);
    assert_eq!(report.stop_reason(), StopReason::MaxIter);
    assert!(!report.converged());
    assert_eq!(report.steps, 6);
}

/// The same run is an error only when the caller opts in.
#[test]
fn an_unmet_stop_is_an_error_only_when_asked() {
    let control = Control::default()
        .with_maxiter(6)
        .with_gtol(1e-12)
        .with_error_on_unmet(true);
    let err = minimize_lbfgs(
        &rose(),
        array![-1.2, 1.0],
        &control,
        LineSearch::default(),
        10,
    )
    .unwrap_err();
    let Error::StopUnmet { reason, report } = err else {
        panic!("expected StopUnmet");
    };
    assert_eq!(reason, StopReason::MaxIter);
    assert_eq!(report.steps, 6);
    assert!(report.grad_norm > backward_error_floor(report.value, report.coords.view()));
    assert!(!report.coords.is_empty());
}

/// A step that lands under `gtol` is converged, including the last one.
#[test]
fn the_last_step_that_meets_gtol_is_converged() {
    let quad = rgmin::Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]));
    let control = Control::default()
        .with_maxiter(1)
        .with_gtol(0.05)
        .with_maxmove(f64::INFINITY);
    let mut solver = Lbfgs::with_capacity(5);
    solver.norm = rgmin::GradNorm::Euclidean;
    let report = solver
        .minimize_objective(&quad, array![0.15], &control, LineSearch::default())
        .unwrap();
    assert_eq!(report.steps, 1);
    assert!(report.grad_norm < 0.05);
    assert_eq!(report.stop_reason(), StopReason::Converged);
    assert!(report.converged());
}

/// No accepted step is a failed line search, not a generic budget.
#[test]
fn a_search_that_never_moves_is_line_search_failed() {
    let uphill =
        rgmin::Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![-x[0]]));
    let control = Control::default().with_maxiter(3).with_gtol(1e-8);
    let mut solver = Lbfgs::with_capacity(4);
    solver.norm = rgmin::GradNorm::Euclidean;
    let report = solver
        .minimize_objective(&uphill, array![1.0], &control, LineSearch::default())
        .unwrap();
    assert_eq!(report.coords[0], 1.0);
    assert_eq!(report.stop_reason(), StopReason::LineSearchFailed);
    assert!(!report.converged());
    assert_eq!(report.status, Status::Budget);
}

/// The capped L-BFGS session keeps its pairs. A fresh session does not,
/// and that cold step raises the gradient.
#[test]
fn resuming_the_session_does_not_raise_the_gradient() {
    let control = Control::default().with_maxiter(30).with_gtol(1e-14);
    let mut warm = Solver::new(Method::lbfgs(), control, 2);
    let mut x = array![-1.2, 1.0];
    warm.run(&rose(), &mut x).unwrap();
    assert!(warm.pair_count() > 0);
    let (_, g0) = rose().value_and_gradient(x.view());
    let g0 = l2(&g0);
    let mut resumed = x.clone();
    warm.step(&rose(), &mut resumed).unwrap();
    let (_, gw) = rose().value_and_gradient(resumed.view());
    let mut cold = Solver::new(
        Method::lbfgs(),
        Control::default().with_maxiter(1).with_gtol(1e-18),
        2,
    );
    assert_eq!(cold.pair_count(), 0);
    let mut restarted = x.clone();
    cold.step(&rose(), &mut restarted).unwrap();
    let (_, gc) = rose().value_and_gradient(restarted.view());
    assert!(l2(&gw) <= g0);
    assert!(l2(&gc) > g0);
}

#[test]
fn the_floor_is_n_eps_times_the_first_order_scale() {
    let x: Array1<f64> = array![1.5, -2.0, 0.25];
    let value: f64 = -3.0;
    let sum: f64 = x.iter().copied().map(f64::abs).sum();
    let expect = (x.len() as f64) * f64::EPSILON * (1.0 + value.abs() + sum);
    assert_eq!(
        backward_error_floor(value, x.view()).to_bits(),
        expect.to_bits()
    );
}

/// Default `gtol` accepts a point above the floor. The polish does not.
#[test]
fn polish_reaches_the_floor_on_a_quadratic() {
    let obj = quad_hess();
    let start = array![1.0e-6];
    let plain = minimize_newton(
        &obj,
        start.clone(),
        &Control::default(),
        NewtonKind::Shifted,
    )
    .unwrap();
    assert!(plain.converged());
    assert!(plain.grad_norm > backward_error_floor(plain.value, plain.coords.view()));

    let control = Control::default().with_backward_polish(true);
    let polished = minimize_newton(&obj, start, &control, NewtonKind::Shifted).unwrap();
    assert!(polished.converged());
    assert!(polished.grad_norm <= backward_error_floor(polished.value, polished.coords.view()));
    let (_, g) = obj.value_and_gradient(polished.coords.view());
    let hess = obj.hessian(polished.coords.view());
    let p = -hess.dot(&g);
    assert!(newton_decrement_acceptable(
        polished.value,
        polished.coords.view(),
        g.view(),
        p.view()
    ));
}

#[test]
fn an_hvp_polish_reaches_the_same_floor() {
    let start = array![1.0e-6];
    let report = polish_hvp(&IdentityHvp, start, &Control::default().with_maxiter(4)).unwrap();
    assert!(report.converged());
    assert!(report.grad_norm <= backward_error_floor(report.value, report.coords.view()));
}

/// An indefinite Hessian uses the shifted step and does not claim success.
#[test]
fn an_indefinite_polish_stays_unmet() {
    let saddle = HessianOracle::unbounded(
        1,
        |x: ArrayView1<f64>| (-0.5 * x[0] * x[0], array![-x[0]]),
        |_| array![[-1.0]],
    );
    let control = Control::default()
        .with_maxiter(3)
        .with_backward_polish(true)
        .with_error_on_unmet(true);
    let err = polish_newton(&saddle, array![1.0], &control, NewtonKind::Shifted).unwrap_err();
    let Error::StopUnmet { reason, report } = err else {
        panic!("expected StopUnmet");
    };
    assert!(matches!(
        reason,
        StopReason::PolishUnmet | StopReason::LineSearchFailed
    ));
    assert!(!report.converged());
    assert!(report.grad_norm > backward_error_floor(report.value, report.coords.view()));
}

#[test]
fn resume_objective_keeps_the_pairs_a_new_solver_drops() {
    let quad = rgmin::Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let g = array![x[0], 1.0e4 * x[1]];
        (0.5 * (x[0] * g[0] + x[1] * g[1]), g)
    });
    let control = Control::default()
        .with_maxiter(2)
        .with_gtol(1e-14)
        .with_maxmove(f64::INFINITY);
    let mut warm = Lbfgs::with_capacity(5);
    warm.norm = rgmin::GradNorm::Euclidean;
    let first = warm
        .minimize_objective(&quad, array![1.0, 1.0e-2], &control, LineSearch::default())
        .unwrap();
    assert_ne!(warm.len(), 0);
    let kept = warm.len();
    let _ = warm
        .resume_objective(
            &quad,
            first.coords.clone(),
            &Control::default()
                .with_maxiter(1)
                .with_gtol(1e-18)
                .with_maxmove(f64::INFINITY),
            LineSearch::default(),
        )
        .unwrap();
    assert!(warm.len() >= kept);
}
