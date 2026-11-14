//! SCG must descend without a line search, and survive barrier walls.

use std::sync::OnceLock;

use approx::assert_relative_eq;
use eindir_core::objectives::Rosenbrock;
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::nlcg::{Conjugacy, Restart};
use rgmin::{Control, ScgParams, minimize_scg};

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
        path.push(report.coords.clone());
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
