//! SCG must descend without a line search, and survive barrier walls.

use std::sync::OnceLock;

use approx::assert_relative_eq;
use eindir_core::objectives::Rosenbrock;
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::nlcg::{Conjugacy, Restart};
use rgmin::{
    Control, DirectionalCurvature, Report, ScgParams, Status, minimize_scg, minimize_scg_exact,
};

fn control() -> Control {
    Control {
        maxiter: 500,
        gtol: 1e-8,
        istep: 1.0,
        maxmove: None,
        ..Default::default()
    }
}

fn free_bounds(dim: usize) -> &'static Bounds<f64> {
    static B1: OnceLock<Bounds<f64>> = OnceLock::new();
    static B2: OnceLock<Bounds<f64>> = OnceLock::new();
    static B4: OnceLock<Bounds<f64>> = OnceLock::new();
    let (cell, n) = match dim {
        1 => (&B1, 1),
        2 => (&B2, 2),
        _ => (&B4, 4),
    };
    cell.get_or_init(|| Bounds::new(Array1::from_elem(n, -1e12), Array1::from_elem(n, 1e12), 0.0))
}

struct Bowl;
impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        4
    }
    fn bounds(&self) -> &Bounds<f64> {
        free_bounds(4)
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, v)| (i + 1) as f64 * v * v)
            .sum()
    }
}
impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        4
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter(x.iter().enumerate().map(|(i, v)| 2.0 * (i + 1) as f64 * v))
    }
}
impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn quadratic_bowl_converges_fast() {
    let report = minimize_scg(
        &Bowl,
        array![1.0, -2.0, 3.0, -4.0],
        &control(),
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
    .unwrap();
    assert!(report.value < 1e-10, "value {}", report.value);
    assert!(report.steps < 60, "steps {}", report.steps);
}

#[test]
fn zero_iteration_limit_runs_to_convergence() {
    let mut ctrl = control();
    ctrl.maxiter = 0;
    let report = minimize_scg(
        &Bowl,
        array![1.0, -2.0, 3.0, -4.0],
        &ctrl,
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
    .unwrap();
    assert!(report.value < 1e-10, "value {}", report.value);
    assert!(report.steps > 1, "steps {}", report.steps);
}

struct OffsetBowl(f64);

impl Objective<f64> for OffsetBowl {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        free_bounds(2)
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.0 + 0.25 * x.dot(&x)
    }
}

impl Gradient<f64> for OffsetBowl {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        0.5 * &x
    }
}

impl DifferentiableObjective<f64> for OffsetBowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn function_convergence_ignores_objective_offset() {
    let ctrl = Control {
        maxiter: 0,
        gtol: 0.0,
        ..control()
    };
    let params = ScgParams {
        lambda: 100.0,
        tol_sol: 1e-2,
        tol_func: 1e-8,
        ..ScgParams::default()
    };
    for offset in [0.0, 65536.0, -65536.0] {
        let report = minimize_scg(
            &OffsetBowl(offset),
            array![0.5, 0.0],
            &ctrl,
            &params,
            Conjugacy::LiuStorey,
            Restart::Never,
        )
        .unwrap();
        assert!(report.grad_norm < 1e-4, "offset {offset}: {report:?}");
        assert!(report.coords.iter().all(|x| x.abs() < 1e-4));
    }
}

#[test]
fn rosenbrock_finds_the_banana_minimum() {
    let obj = Rosenbrock::<2>::new();
    let report = minimize_scg(
        &obj,
        array![-1.2, 1.0],
        &control(),
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::njws(),
    )
    .unwrap();
    assert!(report.value < 1e-6, "value {}", report.value);
    assert_relative_eq!(report.coords[0], 1.0, epsilon = 1e-3);
    assert_relative_eq!(report.coords[1], 1.0, epsilon = 1e-3);
}

struct Walled;
impl Objective<f64> for Walled {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        free_bounds(2)
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if x[0] >= 1.0 {
            return f64::INFINITY;
        }
        x[0] * x[0] + x[1] * x[1] - (1.0 - x[0]).ln() * 1e-3
    }
}
impl Gradient<f64> for Walled {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        if x[0] >= 1.0 {
            return array![f64::INFINITY, f64::INFINITY];
        }
        array![2.0 * x[0] + 1e-3 / (1.0 - x[0]), 2.0 * x[1]]
    }
}
impl DifferentiableObjective<f64> for Walled {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn barrier_wall_raises_damping_instead_of_dying() {
    // Interior-point style objective: +inf at and past x0 = 1, minimum
    // near the origin. SCG must step around the wall via its damping
    // retries, the way the gpr_optim MAP fit uses it.
    let report = minimize_scg(
        &Walled,
        array![0.9, 0.5],
        &control(),
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
    .unwrap();
    assert!(report.coords[0] < 1.0, "stayed interior");
    assert!(report.value < 1e-3, "value {}", report.value);
}

fn distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b).mapv(|v| v * v).sum().sqrt()
}

/// Accepted points after each of the first `iters` iterations. SCG is
/// deterministic, so a run stopped after `k` iterations is the start of
/// any longer run from the same point.
fn scg_path<O: DifferentiableObjective<f64>>(
    obj: &O,
    init: &Array1<f64>,
    ctrl: &Control,
    iters: usize,
) -> Vec<Array1<f64>> {
    let mut path = vec![init.clone()];
    for k in 1..=iters {
        let limited = Control {
            maxiter: k,
            ..ctrl.clone()
        };
        let report = minimize_scg(
            obj,
            init.clone(),
            &limited,
            &ScgParams::default(),
            Conjugacy::PolakRibiere,
            Restart::Never,
        )
        .unwrap();
        path.push(report.coords);
    }
    path
}

fn assert_steps_within(path: &[Array1<f64>], cap: f64) {
    for (k, pair) in path.windows(2).enumerate() {
        let step = distance(&pair[1], &pair[0]);
        assert!(
            step <= cap * (1.0 + 1e-12),
            "iteration {k}: step {step} > {cap}"
        );
    }
}

#[test]
fn positive_maxmove_bounds_every_step() {
    let start = array![1.0, -2.0, 3.0, -4.0];
    let cap = 0.5;
    let free = scg_path(&Bowl, &start, &control(), 1);
    let first = distance(&free[1], &free[0]);
    assert!(
        first > cap,
        "unbounded first step {first} must exceed the cap"
    );

    let ctrl = Control {
        maxmove: Some(cap),
        ..control()
    };
    assert_steps_within(&scg_path(&Bowl, &start, &ctrl, 40), cap);

    let report = minimize_scg(
        &Bowl,
        start,
        &Control { maxiter: 0, ..ctrl },
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
    .unwrap();
    assert!(report.value < 1e-10, "value {}", report.value);
}

#[test]
fn maxmove_that_is_not_positive_and_finite_leaves_scg_unbounded() {
    let obj = Rosenbrock::<2>::new();
    let run = |maxmove| {
        minimize_scg(
            &obj,
            array![-1.2, 1.0],
            &Control {
                maxmove,
                ..control()
            },
            &ScgParams::default(),
            Conjugacy::PolakRibiere,
            Restart::njws(),
        )
        .unwrap()
    };
    let free = run(None);
    for m in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        let report = run(Some(m));
        assert_eq!(report.coords, free.coords, "maxmove {m}");
        assert_eq!(report.value.to_bits(), free.value.to_bits(), "maxmove {m}");
        assert_eq!(report.steps, free.steps, "maxmove {m}");
    }
}

/// `x^2 / 2`, non-finite on the band `(8.95, 9.05)`.
struct Holed;
impl Objective<f64> for Holed {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        free_bounds(1)
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if x[0] > 8.95 && x[0] < 9.05 {
            return f64::NAN;
        }
        0.5 * x[0] * x[0]
    }
}
impl Gradient<f64> for Holed {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        if x[0] > 8.95 && x[0] < 9.05 {
            return array![f64::NAN];
        }
        array![x[0]]
    }
}
impl DifferentiableObjective<f64> for Holed {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn nonfinite_retry_stays_within_maxmove() {
    // From 10 the bounded step of length 1 reaches 9.0, inside the band.
    // The first retry asks for a step of length 2, which the bound must
    // shorten back to 1; the second retry steps 0.588 to a finite point.
    let start = array![10.0];
    let cap = 1.0;
    let ctrl = Control {
        maxmove: Some(cap),
        ..control()
    };
    let path = scg_path(&Holed, &start, &ctrl, 30);
    assert_steps_within(&path, cap);
    assert!(
        path[1][0] > 9.05 && path[1][0] < 10.0,
        "first point {}",
        path[1][0]
    );
    assert!(path[30][0].abs() < 1e-6, "end {}", path[30][0]);
}

/// Constant value, constant nonzero gradient. The comparison ratio is zero,
/// so the objective test passes and the gradient never falls.
struct Plateau {
    calls: std::sync::atomic::AtomicUsize,
}

impl Objective<f64> for Plateau {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        free_bounds(2)
    }
    fn eval(&self, _x: ArrayView1<f64>) -> f64 {
        let n = self
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n > 40 { f64::NAN } else { 0.0 }
    }
}

impl Gradient<f64> for Plateau {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        array![1.0, 0.0]
    }
}

impl DifferentiableObjective<f64> for Plateau {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn flat_objective_with_no_iteration_limit_returns() {
    let mut ctrl = control();
    ctrl.maxiter = 0;
    ctrl.gtol = 0.0;
    let problem = Plateau {
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let report = minimize_scg(
        &problem,
        array![0.0, 0.0],
        &ctrl,
        &ScgParams {
            tol_sol: 1e-2,
            tol_func: 1e-2,
            ..ScgParams::default()
        },
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
    .expect("a plateau must stop before the objective turns non-finite");
    assert!(
        problem.calls.load(std::sync::atomic::Ordering::Relaxed) < 40,
        "stopped after {} evaluations",
        problem.calls.load(std::sync::atomic::Ordering::Relaxed)
    );
    assert!(report.steps < 40, "steps {}", report.steps);
}

/// `2^60 + x0` with its exact gradient. The damped steps sum to less than
/// half an ulp of `2^60`, so every accepted step is flat to the last bit.
/// Evaluations from `limit` on return NaN, which ends a run that would
/// otherwise not return.
struct RoundingFlat {
    calls: std::sync::atomic::AtomicUsize,
    limit: usize,
}

const ROUNDING_FLAT_VALUE: f64 = 1_152_921_504_606_846_976.0;

impl Objective<f64> for RoundingFlat {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        free_bounds(2)
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let n = self
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n >= self.limit {
            f64::NAN
        } else {
            ROUNDING_FLAT_VALUE + x[0]
        }
    }
}

impl Gradient<f64> for RoundingFlat {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        array![1.0, 0.0]
    }
}

impl DifferentiableObjective<f64> for RoundingFlat {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

/// SCG on [`RoundingFlat`] with the step test off, so only the objective
/// test can recognise the plateau.
fn rounding_flat(maxiter: usize, lambda_limit: f64) -> rgmin::Result<Report> {
    let problem = RoundingFlat {
        calls: std::sync::atomic::AtomicUsize::new(0),
        limit: 10_000,
    };
    minimize_scg(
        &problem,
        array![0.0, 0.0],
        &Control {
            maxiter,
            gtol: 0.0,
            ..control()
        },
        &ScgParams {
            tol_sol: 0.0,
            lambda_limit,
            ..ScgParams::default()
        },
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
}

#[test]
fn unlimited_fit_stops_on_a_flat_plateau_with_unbounded_damping() {
    // The damping saturates below an infinite limit, so no other stop fires.
    let report = rounding_flat(0, f64::INFINITY)
        .expect("a flat unlimited fit must stop before the objective turns non-finite");
    assert_eq!(report.status, Status::StepTolerance, "{report:?}");
    assert_eq!(report.steps, 2, "one period of two parameters");
    assert_eq!(report.value, ROUNDING_FLAT_VALUE);
    assert_eq!(report.grad_norm, 1.0);
}

#[test]
fn flat_plateau_is_not_reported_as_a_damping_failure() {
    let report = rounding_flat(0, ScgParams::default().lambda_limit)
        .expect("a flat objective with an exact gradient is not a gradient error");
    assert_eq!(report.status, Status::StepTolerance, "{report:?}");
    assert_eq!(report.steps, 2);
}

#[test]
fn budget_limited_fit_runs_through_a_flat_plateau() {
    for lambda_limit in [ScgParams::default().lambda_limit, f64::INFINITY] {
        let report = rounding_flat(50, lambda_limit).unwrap();
        assert_eq!(report.status, Status::Budget, "{report:?}");
        assert_eq!(report.steps, 50);
    }
}

/// `sqrt(1 + x^2)` with exact curvature. Undamped Newton maps `x` to
/// `-x^3`, so the first step from 1 lands on -1: the same value and the
/// same gradient norm, a single flat step on the way to the minimum at 0.
struct PseudoHuber;

impl Objective<f64> for PseudoHuber {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        free_bounds(1)
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        (1.0 + x[0] * x[0]).sqrt()
    }
}

impl Gradient<f64> for PseudoHuber {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![x[0] / (1.0 + x[0] * x[0]).sqrt()]
    }
}

impl DifferentiableObjective<f64> for PseudoHuber {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

impl DirectionalCurvature for PseudoHuber {
    fn directional_curvature(&self, x: ArrayView1<f64>, d: ArrayView1<f64>) -> Option<f64> {
        Some(d[0] * d[0] / (1.0 + x[0] * x[0]).powf(1.5))
    }
}

#[test]
fn one_flat_step_does_not_stop_an_unlimited_fit() {
    let params = ScgParams {
        lambda: 1e-9,
        ..ScgParams::default()
    };
    let run = |maxiter| {
        minimize_scg_exact(
            &PseudoHuber,
            array![1.0],
            &Control {
                maxiter,
                gtol: 1e-10,
                ..control()
            },
            &params,
            Conjugacy::PolakRibiere,
            Restart::Never,
        )
        .unwrap()
    };
    let first = run(1);
    assert!((first.coords[0] + 1.0).abs() < 1e-6, "{first:?}");
    assert!(
        (first.value - 2f64.sqrt()).abs() < params.tol_func,
        "{first:?}"
    );
    assert!(
        first.grad_norm > 0.5f64.sqrt() * (1.0 - f64::EPSILON.sqrt()),
        "{first:?}"
    );

    let report = run(0);
    assert!(report.coords[0].abs() < 1e-6, "{report:?}");
    assert!(report.steps > 2, "steps {}", report.steps);
}
