use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, ManifoldKind, Method, Solver};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

struct Rayleigh {
    diagonal: Array1<f64>,
    bounds: Bounds<f64>,
    calls: AtomicUsize,
    largest_norm_error: AtomicU64,
}

impl Rayleigh {
    fn new() -> Self {
        Self {
            diagonal: array![-0.02, 0.03, 0.4, 3.0, 30.0, 100.0],
            bounds: Bounds::new(Array1::from_elem(6, -2.0), Array1::from_elem(6, 2.0), 0.0),
            calls: AtomicUsize::new(0),
            largest_norm_error: AtomicU64::new(0.0_f64.to_bits()),
        }
    }
}

impl Objective<f64> for Rayleigh {
    fn dim(&self) -> usize {
        self.diagonal.len()
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Rayleigh {
    fn dim(&self) -> usize {
        self.diagonal.len()
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        &self.diagonal * &x
    }
}

impl DifferentiableObjective<f64> for Rayleigh {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.largest_norm_error
            .fetch_max((x.dot(&x) - 1.0).abs().to_bits(), Ordering::Relaxed);
        let gradient = self.grad(x);
        (0.5 * x.dot(&gradient), gradient)
    }
}

fn solve(retain: bool) -> (f64, f64, usize, f64) {
    let objective = Rayleigh::new();
    let n = objective.diagonal.len();
    let mut solver = Solver::new(
        Method::Lbfgs { memory: n },
        Control {
            maxiter: n * n,
            gtol: 1e-8,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        },
        n,
    );
    solver.set_manifold(ManifoldKind::Sphere);
    solver.set_accept(Accept::Energy);
    let mut direction = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
    let mut residual = f64::INFINITY;
    let mut curvature = f64::NAN;
    for _ in 0..n * n {
        if !retain {
            solver.forget();
        }
        solver.step(&objective, &mut direction).unwrap();
        let action = &objective.diagonal * &direction;
        curvature = direction.dot(&action);
        let r = action - curvature * &direction;
        residual = r.dot(&r).sqrt();
        if residual < 1e-6 {
            break;
        }
    }
    (
        residual,
        curvature,
        objective.calls.load(Ordering::Relaxed),
        f64::from_bits(objective.largest_norm_error.load(Ordering::Relaxed)),
    )
}

#[test]
fn sphere_lbfgs_keeps_every_trial_on_the_sphere() {
    let (_, _, _, norm_error) = solve(true);
    assert!(norm_error < 1e-12, "off-sphere oracle trial: {norm_error}");
}

#[test]
fn sphere_lbfgs_retains_tangent_history_on_a_stiff_spectrum() {
    let warm = solve(true);
    let cold = solve(false);
    assert!(warm.0 < 1e-6, "residual {}", warm.0);
    assert!((warm.1 + 0.02).abs() < 1e-8, "curvature {}", warm.1);
    assert!(warm.2 < cold.2, "warm {} calls, cold {}", warm.2, cold.2);
}

#[test]
fn sphere_lbfgs_initial_path_is_invariant_to_energy_units() {
    let path = |scale: f64| {
        let mut objective = Rayleigh::new();
        objective.diagonal *= scale;
        let n = objective.diagonal.len();
        let mut solver = Solver::new(
            Method::Lbfgs { memory: n },
            Control {
                maxiter: n,
                gtol: 0.0,
                istep: 1.0,
                maxmove: None,
                ftol_rel: None,
            },
            n,
        );
        solver.set_manifold(ManifoldKind::Sphere);
        solver.set_accept(Accept::Energy);
        let mut direction = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
        let mut points = Vec::new();
        for _ in 0..n {
            solver.step(&objective, &mut direction).unwrap();
            points.push(direction.clone());
        }
        (points, objective.calls.load(Ordering::Relaxed))
    };
    let reference = path(1.0);
    for scale in [0.001, 1000.0] {
        let scaled = path(scale);
        assert_eq!(reference.1, scaled.1);
        for (a, b) in reference.0.iter().zip(&scaled.0) {
            assert!((a - b).iter().all(|x| x.abs() < 1e-10));
        }
    }
}

#[test]
fn sphere_lbfgs_resolves_a_stiff_ritz_initializer() {
    let objective = Rayleigh::new();
    let n = objective.diagonal.len();
    let seed = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
    let action = objective.grad(seed.view());
    let a = seed.dot(&action);
    let residual = &action - a * &seed;
    let b = residual.dot(&residual).sqrt();
    let tangent = &residual / b;
    let c = tangent.dot(&objective.grad(tangent.view()));
    let eigenvalue = 0.5 * (a + c - ((a - c).powi(2) + 4.0 * b * b).sqrt());
    let mut direction = &seed - ((a - eigenvalue) / b) * &tangent;
    direction /= direction.dot(&direction).sqrt();
    let old_gradient = residual;
    let mut gradient = objective.grad(direction.view());
    gradient -= &(gradient.dot(&direction) * &direction);
    let mut displacement = &direction - &seed;
    displacement -= &(displacement.dot(&direction) * &direction);
    let change = gradient - &old_gradient + direction.dot(&old_gradient) * &direction;
    let mut solver = Solver::new(
        Method::Lbfgs { memory: n },
        Control {
            maxiter: n * n,
            gtol: 0.0,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        },
        n,
    );
    solver.set_manifold(ManifoldKind::Sphere);
    solver.set_accept(Accept::Energy);
    assert!(solver.push_pair(displacement.view(), change.view()));
    let mut residual = f64::INFINITY;
    for step in 1..n * n {
        solver.step(&objective, &mut direction).unwrap();
        let action = objective.grad(direction.view());
        let curvature = direction.dot(&action);
        let error = action - curvature * &direction;
        residual = error.dot(&error).sqrt();
        println!("Ritz step {step}: residual {residual:.12e}");
        if residual < 1e-6 {
            break;
        }
    }
    assert!(residual < 1e-6, "residual {residual}");
}

#[test]
fn sphere_disabled_caps_preserve_the_unbounded_path() {
    let path = |cap| {
        let objective = Rayleigh::new();
        let n = objective.diagonal.len();
        let mut solver = Solver::new(
            Method::Lbfgs { memory: n },
            Control {
                gtol: 0.0,
                maxmove: cap,
                ..Control::default()
            },
            n,
        );
        solver.set_manifold(ManifoldKind::Sphere);
        solver.set_accept(Accept::Energy);
        let mut point = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
        let mut points = Vec::new();
        for _ in 0..4 {
            solver.step(&objective, &mut point).unwrap();
            points.push(point.clone());
        }
        (points, objective.calls.load(Ordering::Relaxed))
    };
    let reference = path(None);
    for cap in [Some(0.0), Some(-1.0), Some(f64::NAN), Some(f64::INFINITY)] {
        let observed = path(cap);
        assert_eq!(observed.1, reference.1);
        assert_eq!(observed.0, reference.0);
    }
}

#[test]
fn sphere_atom_cap_controls_every_oracle_trial_and_takes_precedence() {
    let origin = Array1::from_elem(6, 1.0 / 6.0_f64.sqrt());
    let calls = AtomicUsize::new(0);
    let objective = rgmin::Oracle::unbounded(6, |point| {
        calls.fetch_add(1, Ordering::Relaxed);
        assert!((point.dot(&point) - 1.0).abs() < 1e-12);
        let displacement = &point - &origin;
        for atom in 0..2 {
            let length = (0..3)
                .map(|k| displacement[3 * atom + k].powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(length <= 0.05 + 1e-12, "atom displacement {length}");
        }
        let gradient = &array![-0.02, 0.03, 0.4, 3.0, 30.0, 100.0] * &point;
        (0.5 * point.dot(&gradient), gradient)
    });
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            gtol: 0.0,
            maxmove: Some(0.001),
            ..Control::default()
        },
        6,
    );
    solver.set_manifold(ManifoldKind::Sphere);
    solver.set_accept(Accept::Energy);
    solver.set_atom_maxmove(0.05);
    let mut point = origin.clone();
    let initial_value = objective.eval(point.view());
    let report = solver.step(&objective, &mut point).unwrap();
    assert!(report.value < initial_value);
    let displacement = &point - &origin;
    assert!(displacement.dot(&displacement).sqrt() > 0.001);
    assert!(calls.load(Ordering::Relaxed) > 2);
}
