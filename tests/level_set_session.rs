use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Accept, Control, Error, LbfgsStep, LevelSetConstraint, Method, Solver, Status, StopNorm,
    StopReason,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct CircularForce {
    bounds: Bounds<f64>,
}

impl CircularForce {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(Array1::from_elem(2, -4.0), Array1::from_elem(2, 4.0), 0.0),
        }
    }
}

impl Objective<f64> for CircularForce {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x.dot(&x)
    }
}

impl Gradient<f64> for CircularForce {
    fn dim(&self) -> usize {
        2
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![-x[1], x[0]]
    }
}

impl DifferentiableObjective<f64> for CircularForce {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn level_tolerance(target: f64, value: f64, grad: &Array1<f64>, x: &Array1<f64>) -> f64 {
    let eps = f64::EPSILON;
    let sqrt_eps = eps.sqrt();
    let value_scale = target.abs() + value.abs();
    let grad_scale = rgmin::vecops::nrm2(grad.view());
    let x_scale = rgmin::vecops::nrm2(x.view());
    sqrt_eps * value_scale + grad_scale * eps * (x_scale + 1.0) + eps
}

fn assert_report_bits_equal(a: &rgmin::Report, b: &rgmin::Report) {
    assert_eq!(a.value.to_bits(), b.value.to_bits());
    assert_eq!(a.steps, b.steps);
    assert_eq!(a.grad_norm.to_bits(), b.grad_norm.to_bits());
    assert_eq!(a.stop_norm, b.stop_norm);
    assert_eq!(a.status, b.status);
    match (&a.grad, &b.grad) {
        (Some(ga), Some(gb)) => {
            assert_eq!(ga.len(), gb.len());
            for (ai, bi) in ga.iter().zip(gb.iter()) {
                assert_eq!(ai.to_bits(), bi.to_bits());
            }
        }
        (None, None) => {}
        _ => panic!("gradient presence mismatch"),
    }
}

struct Bowl {
    bounds: Bounds<f64>,
}

impl Bowl {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(Array1::from_elem(2, -4.0), Array1::from_elem(2, 4.0), 0.0),
        }
    }
}

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x[0] * x[0] + 2.0 * x[1] * x[1]
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![2.0 * x[0], 4.0 * x[1]]
    }
}

impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn unset_level_set_keeps_the_unconstrained_path_bit_exact() {
    let objective = CircularForce::new();
    let control = Control::default()
        .with_maxiter(4)
        .with_gtol(0.0)
        .with_istep(0.2)
        .with_maxmove(None)
        .with_stop_norm(StopNorm::Two);
    let mut plain = Solver::new(Method::lbfgs(), control.clone(), 2);
    plain.set_accept(Accept::Step);
    let mut hooked = Solver::new(Method::lbfgs(), control, 2);
    hooked.set_level_set(LevelSetConstraint::new(1.0)).unwrap();
    hooked.clear_level_set();
    hooked.set_accept(Accept::Step);
    let mut x_plain = array![0.8, -0.6];
    let mut x_hooked = x_plain.clone();
    for _ in 0..4 {
        let report_plain = plain.step(&objective, &mut x_plain).unwrap();
        let report_hooked = hooked.step(&objective, &mut x_hooked).unwrap();
        assert_eq!(x_plain.len(), x_hooked.len());
        for (a, b) in x_plain.iter().zip(x_hooked.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_report_bits_equal(&report_plain, &report_hooked);
    }
}

#[test]
fn unset_level_set_keeps_the_capped_eon_step_bit_exact() {
    let objective = Bowl::new();
    let control = Control::default().with_maxiter(4).with_gtol(0.0);
    let mut plain = Solver::new(Method::lbfgs(), control.clone(), 2);
    let mut hooked = Solver::new(Method::lbfgs(), control, 2);
    hooked.set_level_set(LevelSetConstraint::new(1.0)).unwrap();
    hooked.clear_level_set();
    let mut x_plain = array![0.4, -0.3];
    let mut x_hooked = x_plain.clone();
    for _ in 0..4 {
        let report_plain = plain.step(&objective, &mut x_plain).unwrap();
        let report_hooked = hooked.step(&objective, &mut x_hooked).unwrap();
        assert_eq!(x_plain.len(), x_hooked.len());
        for (a, b) in x_plain.iter().zip(x_hooked.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_report_bits_equal(&report_plain, &report_hooked);
    }
}

struct Radius {
    bounds: Bounds<f64>,
}

impl Radius {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(Array1::from_elem(2, -4.0), Array1::from_elem(2, 4.0), 0.0),
        }
    }
}

impl Objective<f64> for Radius {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x.dot(&x)
    }
}

impl Gradient<f64> for Radius {
    fn dim(&self) -> usize {
        2
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![2.0 * x[0], 2.0 * x[1]]
    }
}

impl DifferentiableObjective<f64> for Radius {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn capped_eon_step_stays_on_the_level_set() {
    let objective = Radius::new();
    let mut x = array![2.0, 0.0];
    let target = 1.0;
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control::default().with_maxiter(3).with_gtol(0.0),
        2,
    );
    solver
        .set_level_set(LevelSetConstraint::new(target))
        .unwrap();
    for _ in 0..3 {
        solver.step(&objective, &mut x).unwrap();
        let (value, grad) = objective.value_and_gradient(x.view());
        let tol = level_tolerance(target, value, &grad, &x);
        assert!(
            (value - target).abs() <= tol,
            "capped eOn left the level set: value {value} target {target} tol {tol} x={x:?}"
        );
    }
}

#[test]
fn session_points_stay_on_the_level_set() {
    let objective = CircularForce::new();
    let mut x = array![1.0, 0.0];
    let target = objective.eval(x.view());
    let mut solver = Solver::new(
        Method::Steepest,
        Control::default()
            .with_maxiter(25)
            .with_gtol(0.0)
            .with_istep(0.2)
            .with_maxmove(None)
            .with_stop_norm(StopNorm::Two),
        2,
    );
    solver
        .set_level_set(LevelSetConstraint::new(target))
        .unwrap();
    solver.set_accept(Accept::Step);
    for _ in 0..25 {
        let _ = solver.step(&objective, &mut x).unwrap();
        let (value, grad) = objective.value_and_gradient(x.view());
        let tol = level_tolerance(target, value, &grad, &x);
        assert!(
            (value - target).abs() <= tol,
            "left level set: value {value} target {target} tol {tol} x={x:?}"
        );
    }
}

struct Stuck {
    bounds: Bounds<f64>,
}

impl Stuck {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-f64::INFINITY], array![f64::INFINITY], 0.0),
        }
    }
}

impl Objective<f64> for Stuck {
    fn dim(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, _x: ArrayView1<f64>) -> f64 {
        1.0
    }
}

impl Gradient<f64> for Stuck {
    fn dim(&self) -> usize {
        1
    }

    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        array![f64::EPSILON.sqrt()]
    }
}

impl DifferentiableObjective<f64> for Stuck {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn session_reports_retraction_failure() {
    let objective = Stuck::new();
    let mut x = array![0.0];
    let mut solver = Solver::new(Method::Steepest, Control::default().with_maxiter(1), 1);
    solver.set_level_set(LevelSetConstraint::new(0.0)).unwrap();
    let err = solver.step(&objective, &mut x).unwrap_err();
    assert!(matches!(err, Error::LevelSetStalled { .. }), "{err:?}");
}

struct FinishLevelSetRounding {
    bounds: Bounds<f64>,
    drift: bool,
    saw_trial: AtomicBool,
    origin_after_trial: AtomicUsize,
}

impl FinishLevelSetRounding {
    fn new(drift: bool) -> Self {
        Self {
            bounds: Bounds::new(array![-f64::INFINITY], array![f64::INFINITY], 0.0),
            drift,
            saw_trial: AtomicBool::new(false),
            origin_after_trial: AtomicUsize::new(0),
        }
    }

    fn shifted(x: f64) -> (f64, Array1<f64>) {
        let residual = x - 0.25;
        let slope = if residual.abs() <= f64::EPSILON {
            0.25
        } else {
            1.0
        };
        (residual, array![slope])
    }
}

impl Objective<f64> for FinishLevelSetRounding {
    fn dim(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for FinishLevelSetRounding {
    fn dim(&self) -> usize {
        1
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for FinishLevelSetRounding {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        if self.drift && self.origin_after_trial.load(Ordering::Relaxed) >= 1 {
            return Self::shifted(x[0]);
        }
        if x[0] != 0.0 {
            self.saw_trial.store(true, Ordering::Relaxed);
            return (0.0, array![1.0]);
        }
        if self.saw_trial.load(Ordering::Relaxed) {
            self.origin_after_trial.fetch_add(1, Ordering::Relaxed);
        }
        (0.0, array![1.0])
    }
}

#[test]
fn finish_level_set_clears_stale_line_search_rounding_reason() {
    let run = |drift: bool, gtol: f64, require_convergence: bool| {
        let objective = FinishLevelSetRounding::new(drift);
        let mut x = array![0.0];
        let mut solver = Solver::new(
            Method::lbfgs(),
            Control::default()
                .with_gtol(gtol)
                .with_require_convergence(require_convergence),
            1,
        );
        solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
        solver.set_level_set(LevelSetConstraint::new(0.0)).unwrap();
        let report = solver.step(&objective, &mut x).unwrap();
        (report, x)
    };

    for strict in [false, true] {
        let (kept, x_kept) = run(false, 0.5, strict);
        assert_eq!(x_kept, array![0.0]);
        assert_eq!(kept.coords, x_kept);
        assert_eq!(kept.status, Status::Budget);
        assert_eq!(kept.stop_reason(), StopReason::LineSearchRounding);
        assert!(!kept.converged);

        let (converged, x_conv) = run(true, 0.5, strict);
        assert_eq!(x_conv, array![0.25]);
        assert_eq!(converged.coords, x_conv);
        assert_eq!(converged.status, Status::Tolerance);
        assert_eq!(converged.stop_reason(), StopReason::Converged);
        assert!(converged.converged());
        assert!(converged.converged);

        let (iterating, x_iter) = run(true, 0.1, strict);
        assert_eq!(x_iter, array![0.25]);
        assert_eq!(iterating.coords, x_iter);
        assert_eq!(iterating.status, Status::Iterating);
        assert_eq!(iterating.stop_reason(), StopReason::Iterating);
        assert!(!iterating.converged);
    }

    let objective = FinishLevelSetRounding::new(true);
    let mut x = array![0.0];
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control::default()
            .with_maxiter(1)
            .with_gtol(0.5)
            .with_require_convergence(true),
        1,
    );
    solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
    solver.set_level_set(LevelSetConstraint::new(0.0)).unwrap();
    let run_report = solver.run(&objective, &mut x).unwrap();
    assert_eq!(x, array![0.25]);
    assert_eq!(run_report.status, Status::Tolerance);
    assert!(run_report.converged());
    assert!(run_report.converged);

    let unmet_objective = FinishLevelSetRounding::new(true);
    let mut x_unmet = array![0.0];
    let mut unmet_solver = Solver::new(
        Method::lbfgs(),
        Control::default()
            .with_maxiter(1)
            .with_gtol(0.1)
            .with_require_convergence(true),
        1,
    );
    unmet_solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
    unmet_solver
        .set_level_set(LevelSetConstraint::new(0.0))
        .unwrap();
    let err = unmet_solver
        .run(&unmet_objective, &mut x_unmet)
        .unwrap_err();
    match err {
        Error::NotConverged(report) => {
            assert!(!report.converged);
            assert_eq!(report.status, Status::Budget);
        }
        other => panic!("unexpected error {other}"),
    }
}
