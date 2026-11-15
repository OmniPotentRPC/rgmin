//! What every local optimizer promises on a known surface.
//!
//! The checks are the result a caller sees: the point, the value, the
//! gradient norm, the stop, and the step the cap allowed. They do not
//! record how many iterations a method takes or how it stores its
//! history.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{
    Conjugacy, Control, Error, FireKind, HessianObjective, HessianVector, LineSearch, Method,
    NewtonKind, Report, Restart, Status, minimize_method, minimize_method_hess,
    minimize_method_hvp,
};

#[derive(Clone, Copy)]
enum Surface {
    /// `f = 1/2 x^T H x` with `H = diag(1, 4, 1, ...)`. Minimum at the origin.
    Quad,
    /// Standard two-dimensional Rosenbrock. Minimum at `(1, 1)`.
    Rose,
    /// Two Lennard-Jones atoms, `sigma = 1`. The minimum is any placement
    /// whose separation is `2^(1/6)`.
    Dimer,
    /// One coordinate, `f = 1/2 (x - 2)^2`.
    OneD,
    /// The value is not finite. The gradient is finite.
    BadValue,
    /// The gradient is not finite. The value is finite.
    BadGrad,
    /// No coordinates.
    Empty,
}

struct Problem {
    surface: Surface,
    dim: usize,
    bounds: Bounds<f64>,
}

impl Problem {
    fn new(surface: Surface, dim: usize) -> Self {
        let bounds = Bounds::new(
            Array1::from_elem(dim, -1.0e6),
            Array1::from_elem(dim, 1.0e6),
            0.0,
        );
        Self {
            surface,
            dim,
            bounds,
        }
    }

    fn fg(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self.surface {
            Surface::Quad => {
                let mut f = 0.0;
                let mut g = Array1::zeros(x.len());
                for i in 0..x.len() {
                    let lam = if i % 2 == 0 { 1.0 } else { 4.0 };
                    f += 0.5 * lam * x[i] * x[i];
                    g[i] = lam * x[i];
                }
                (f, g)
            }
            Surface::Rose => {
                let (a, b) = (1.0 - x[0], x[1] - x[0] * x[0]);
                let f = a * a + 100.0 * b * b;
                let g = Array1::from_vec(vec![-2.0 * a - 400.0 * x[0] * b, 200.0 * b]);
                (f, g)
            }
            Surface::Dimer => dimer_fg(x),
            Surface::OneD => {
                let d = x[0] - 2.0;
                (0.5 * d * d, Array1::from_vec(vec![d]))
            }
            Surface::BadValue => (f64::NAN, Array1::ones(x.len())),
            Surface::BadGrad => (1.0, Array1::from_elem(x.len(), f64::NAN)),
            Surface::Empty => (0.0, Array1::zeros(0)),
        }
    }

    fn hess(&self, x: ArrayView1<f64>) -> Array2<f64> {
        let n = x.len();
        match self.surface {
            Surface::Quad => {
                let mut h = Array2::zeros((n, n));
                for i in 0..n {
                    h[(i, i)] = if i % 2 == 0 { 1.0 } else { 4.0 };
                }
                h
            }
            Surface::Rose => {
                let (x0, y) = (x[0], x[1]);
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        2.0 - 400.0 * y + 1200.0 * x0 * x0,
                        -400.0 * x0,
                        -400.0 * x0,
                        200.0,
                    ],
                )
                .unwrap()
            }
            Surface::Dimer => dimer_hess(x),
            Surface::OneD => Array2::from_elem((1, 1), 1.0),
            Surface::BadValue | Surface::BadGrad => Array2::eye(n),
            Surface::Empty => Array2::zeros((0, 0)),
        }
    }
}

fn dimer_fg(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mut g = Array1::zeros(x.len());
    let mut diff = [0.0; 3];
    let mut r2 = 0.0;
    for a in 0..3 {
        diff[a] = x[a] - x[3 + a];
        r2 += diff[a] * diff[a];
    }
    if r2 <= 0.0 {
        return (f64::INFINITY, g);
    }
    let r = r2.sqrt();
    let inv_r2 = 1.0 / r2;
    let sr2 = inv_r2;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    let energy = 4.0 * (sr12 - sr6);
    let slope = 24.0 * (sr6 - 2.0 * sr12) * inv_r2;
    for a in 0..3 {
        let piece = slope * diff[a];
        g[a] += piece;
        g[3 + a] -= piece;
    }
    let _ = r;
    (energy, g)
}

fn dimer_hess(x: ArrayView1<f64>) -> Array2<f64> {
    let mut d = [0.0; 3];
    let mut r2 = 0.0;
    for a in 0..3 {
        d[a] = x[a] - x[3 + a];
        r2 += d[a] * d[a];
    }
    let mut h = Array2::zeros((6, 6));
    if r2 <= 0.0 {
        return h;
    }
    let r = r2.sqrt();
    // u' = -24 (2 r^{-13} - r^{-7}), u'' = 24 (26 r^{-14} - 7 r^{-8}).
    let up = -24.0 * (2.0 * r.powi(-13) - r.powi(-7));
    let upp = 24.0 * (26.0 * r.powi(-14) - 7.0 * r.powi(-8));
    let s = up / r;
    let ds = (upp * r - up) / r2;
    for a in 0..3 {
        for b in 0..3 {
            let block = if a == b { s } else { 0.0 } + ds * d[a] * d[b] / r;
            h[(a, b)] = block;
            h[(3 + a, 3 + b)] = block;
            h[(a, 3 + b)] = -block;
            h[(3 + a, b)] = -block;
        }
    }
    h
}

impl Objective<f64> for Problem {
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.fg(x).0
    }
}

impl Gradient<f64> for Problem {
    fn dim(&self) -> usize {
        self.dim
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.fg(x).1
    }
}

impl DifferentiableObjective<f64> for Problem {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.fg(x)
    }
}

impl HessianObjective for Problem {
    fn hessian(&self, x: ArrayView1<f64>) -> Array2<f64> {
        self.hess(x)
    }
}

impl HessianVector for Problem {
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        self.hess(x).dot(&v)
    }
}

#[derive(Clone, Copy)]
enum Drive {
    Grad,
    Hess,
    Hvp,
}

struct Arm {
    name: &'static str,
    method: Method,
    drive: Drive,
    /// Inertial methods need the per-atom cap and a short initial step.
    inertial: bool,
}

fn arms() -> Vec<Arm> {
    let cg = Method::Nlcg {
        conjugacy: Conjugacy::PolakRibiere,
        restart: Restart::Never,
    };
    vec![
        Arm {
            name: "steepest",
            method: Method::Steepest,
            drive: Drive::Grad,
            inertial: false,
        },
        Arm {
            name: "cg",
            method: cg,
            drive: Drive::Grad,
            inertial: false,
        },
        Arm {
            name: "bfgs",
            method: Method::Bfgs,
            drive: Drive::Grad,
            inertial: false,
        },
        Arm {
            name: "lbfgs",
            method: Method::lbfgs(),
            drive: Drive::Grad,
            inertial: false,
        },
        Arm {
            name: "sr1",
            method: Method::Sr1,
            drive: Drive::Grad,
            inertial: false,
        },
        Arm {
            name: "sr2",
            method: Method::Sr2,
            drive: Drive::Grad,
            inertial: false,
        },
        Arm {
            name: "adam",
            method: Method::adam(),
            drive: Drive::Grad,
            inertial: false,
        },
        Arm {
            name: "pso",
            method: Method::pso(),
            drive: Drive::Grad,
            inertial: false,
        },
        Arm {
            name: "fire",
            method: Method::Fire { kind: FireKind::V1 },
            drive: Drive::Grad,
            inertial: true,
        },
        Arm {
            name: "fire2",
            method: Method::Fire { kind: FireKind::V2 },
            drive: Drive::Grad,
            inertial: true,
        },
        Arm {
            name: "bb",
            method: Method::Bb,
            drive: Drive::Grad,
            inertial: true,
        },
        Arm {
            name: "quickmin",
            method: Method::QuickMin,
            drive: Drive::Grad,
            inertial: true,
        },
        Arm {
            name: "ode12r",
            method: Method::Ode12r,
            drive: Drive::Grad,
            inertial: true,
        },
        Arm {
            name: "diis",
            method: Method::diis(),
            drive: Drive::Grad,
            inertial: false,
        },
        Arm {
            name: "newton",
            method: Method::Newton {
                kind: NewtonKind::Shifted,
            },
            drive: Drive::Hess,
            inertial: false,
        },
        Arm {
            name: "rfo",
            method: Method::Newton {
                kind: NewtonKind::Rfo,
            },
            drive: Drive::Hess,
            inertial: false,
        },
        Arm {
            name: "dogleg",
            method: Method::Dogleg,
            drive: Drive::Hess,
            inertial: false,
        },
        Arm {
            name: "newton-cg",
            method: Method::NewtonCg,
            drive: Drive::Hvp,
            inertial: false,
        },
        Arm {
            name: "newton-pcg",
            method: Method::NewtonPcg,
            drive: Drive::Hvp,
            inertial: false,
        },
    ]
}

fn linesearch(arm: &Arm) -> LineSearch {
    let cg = matches!(arm.method, Method::Nlcg { .. });
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: if cg { 0.1 } else { 0.9 },
        maxiter: 30,
    }
}

fn control(arm: &Arm, maxiter: usize) -> Control {
    Control {
        maxiter,
        gtol: 1e-6,
        istep: if arm.inertial { 0.1 } else { 1.0 },
        maxmove: if arm.inertial {
            None
        } else {
            Some(f64::INFINITY)
        },
        ftol_rel: None,
        noise_floor: None,
    }
}

fn run(arm: &Arm, obj: &Problem, x0: Array1<f64>, maxiter: usize) -> Result<Report, Error> {
    let ctl = control(arm, maxiter);
    let ls = linesearch(arm);
    match arm.drive {
        Drive::Grad => minimize_method(obj, x0, &ctl, arm.method.clone(), ls),
        Drive::Hess => minimize_method_hess(obj, x0, &ctl, arm.method.clone(), ls),
        Drive::Hvp => minimize_method_hvp(obj, x0, &ctl, arm.method.clone(), ls),
    }
}

fn l2(g: &Array1<f64>) -> f64 {
    g.dot(g).sqrt()
}

fn agrees(name: &str, obj: &Problem, report: &Report) {
    let (f, g) = obj.fg(report.coords.view());
    let gn = l2(&g);
    assert!(
        (report.value - f).abs() <= 1e-8 * (1.0 + f.abs()),
        "{name}: reported value {} but the surface is {f}",
        report.value
    );
    assert!(
        (report.grad_norm - gn).abs() <= 1e-6 * (1.0 + gn.abs()),
        "{name}: reported gradient norm {} but the surface has {gn}",
        report.grad_norm
    );
}

fn reached_quad(name: &str, report: &Report) {
    assert_eq!(report.status, Status::Tolerance, "{name}");
    assert!(
        report.value.abs() < 1e-8 && report.coords.iter().all(|v| v.abs() < 1e-4),
        "{name}: value {} coords {:?}",
        report.value,
        report.coords
    );
}

fn reached_rose(name: &str, report: &Report) {
    assert_eq!(report.status, Status::Tolerance, "{name}");
    assert!(
        report.value.abs() < 1e-6
            && (report.coords[0] - 1.0).abs() < 1e-3
            && (report.coords[1] - 1.0).abs() < 1e-3,
        "{name}: value {} coords {:?}",
        report.value,
        report.coords
    );
}

fn dimer_distance(x: ArrayView1<f64>) -> f64 {
    let mut r2 = 0.0;
    for a in 0..3 {
        let d = x[a] - x[3 + a];
        r2 += d * d;
    }
    r2.sqrt()
}

fn reached_dimer(name: &str, report: &Report) {
    let r0 = 2.0_f64.powf(1.0 / 6.0);
    let r = dimer_distance(report.coords.view());
    assert_eq!(
        report.status,
        Status::Tolerance,
        "{name}: status {:?} steps {} r {r} value {} g {}",
        report.status,
        report.steps,
        report.value,
        report.grad_norm
    );
    assert!(
        (r - r0).abs() < 1e-3 && (report.value + 1.0).abs() < 1e-4,
        "{name}: r {r} (want {r0}) value {}",
        report.value
    );
}

fn budget(arm: &Arm, surface: Surface) -> usize {
    if !arm.inertial {
        return match surface {
            Surface::Rose if arm.name == "steepest" => 8_000,
            Surface::Rose => 2_000,
            Surface::Dimer => 4_000,
            _ => 400,
        };
    }
    match surface {
        Surface::Rose => 20_000,
        _ => 5_000,
    }
}

#[test]
fn every_optimizer_reaches_the_quadratic_minimum() {
    let obj = Problem::new(Surface::Quad, 2);
    let start = Array1::from_vec(vec![1.5, -0.75]);
    for arm in arms() {
        let report = run(&arm, &obj, start.clone(), budget(&arm, Surface::Quad))
            .unwrap_or_else(|e| panic!("{}: {e}", arm.name));
        reached_quad(arm.name, &report);
        agrees(arm.name, &obj, &report);
    }
}

#[test]
fn every_optimizer_reaches_the_rosenbrock_minimum() {
    let obj = Problem::new(Surface::Rose, 2);
    let start = Array1::from_vec(vec![-1.2, 1.0]);
    for arm in arms() {
        let report = run(&arm, &obj, start.clone(), budget(&arm, Surface::Rose))
            .unwrap_or_else(|e| panic!("{}: {e}", arm.name));
        reached_rose(arm.name, &report);
        agrees(arm.name, &obj, &report);
    }
}

#[test]
fn every_optimizer_relaxes_an_lj_dimer_to_the_known_separation() {
    let obj = Problem::new(Surface::Dimer, 6);
    let mut start = Array1::zeros(6);
    start[3] = 1.4;
    start[4] = 0.15;
    start[5] = -0.05;
    for arm in arms() {
        let report = run(&arm, &obj, start.clone(), budget(&arm, Surface::Dimer))
            .unwrap_or_else(|e| panic!("{}: {e}", arm.name));
        reached_dimer(arm.name, &report);
        agrees(arm.name, &obj, &report);
    }
}

#[test]
fn a_zero_gradient_returns_with_no_step() {
    let obj = Problem::new(Surface::Quad, 2);
    let start = Array1::zeros(2);
    for arm in arms() {
        if arm.name == "pso" {
            // A swarm samples the box. The particle placed on the minimum
            // stays the best point, which is what the report must return.
            let report = run(&arm, &obj, start.clone(), 5).unwrap();
            reached_quad(arm.name, &report);
            agrees(arm.name, &obj, &report);
            continue;
        }
        let report =
            run(&arm, &obj, start.clone(), 5).unwrap_or_else(|e| panic!("{}: {e}", arm.name));
        assert_eq!(report.steps, 0, "{}", arm.name);
        assert_eq!(report.status, Status::Tolerance, "{}", arm.name);
        agrees(arm.name, &obj, &report);
    }
}

#[test]
fn maxiter_zero_takes_no_step_and_maxiter_one_takes_at_most_one() {
    let obj = Problem::new(Surface::Quad, 2);
    let start = Array1::from_vec(vec![1.0, -1.0]);
    for arm in arms() {
        let none =
            run(&arm, &obj, start.clone(), 0).unwrap_or_else(|e| panic!("{}: {e}", arm.name));
        assert_eq!(none.steps, 0, "{}", arm.name);
        let one = run(&arm, &obj, start.clone(), 1).unwrap_or_else(|e| panic!("{}: {e}", arm.name));
        assert!(one.steps <= 1, "{} took {} steps", arm.name, one.steps);
    }
}

#[test]
fn a_nonfinite_oracle_is_an_error_and_not_a_success() {
    let start = Array1::from_vec(vec![0.3, -0.2]);
    for surface in [Surface::BadValue, Surface::BadGrad] {
        let obj = Problem::new(surface, 2);
        for arm in arms() {
            match run(&arm, &obj, start.clone(), 4) {
                Ok(report) => {
                    assert_ne!(
                        report.status,
                        Status::Tolerance,
                        "{} treated a non-finite oracle as converged",
                        arm.name
                    );
                    assert!(
                        !report.value.is_finite() || !report.grad_norm.is_finite(),
                        "{} returned a finite success from a broken oracle: value {} g {}",
                        arm.name,
                        report.value,
                        report.grad_norm
                    );
                }
                Err(_) => {}
            }
        }
    }
}

#[test]
fn one_dimensional_and_empty_inputs_do_not_panic() {
    let one = Problem::new(Surface::OneD, 1);
    let start = Array1::from_vec(vec![0.0]);
    for arm in arms() {
        let report = run(&arm, &one, start.clone(), budget(&arm, Surface::OneD))
            .unwrap_or_else(|e| panic!("{}: {e}", arm.name));
        assert_eq!(report.status, Status::Tolerance, "{}", arm.name);
        assert!(
            (report.coords[0] - 2.0).abs() < 1e-4,
            "{} x {:?}",
            arm.name,
            report.coords
        );
        agrees(arm.name, &one, &report);
    }
    let empty = Problem::new(Surface::Empty, 0);
    for arm in arms() {
        match run(&arm, &empty, Array1::zeros(0), 4) {
            Ok(report) => {
                assert_eq!(report.coords.len(), 0, "{}", arm.name);
                assert_eq!(report.steps, 0, "{}", arm.name);
            }
            Err(Error::Dim { .. } | Error::ManifoldDim { .. }) => {}
            Err(e) => panic!("{}: {e}", arm.name),
        }
    }
}

#[test]
fn a_binding_step_cap_is_the_length_of_the_step() {
    let obj = Problem::new(Surface::Quad, 2);
    let start = Array1::from_vec(vec![4.0, -3.0]);
    let cap = 0.2;
    for arm in arms() {
        if arm.name == "pso" {
            continue;
        }
        let mut ctl = control(&arm, 1);
        ctl.maxmove = Some(cap);
        ctl.istep = 1.0;
        let ls = linesearch(&arm);
        let report = match arm.drive {
            Drive::Grad => minimize_method(&obj, start.clone(), &ctl, arm.method.clone(), ls),
            Drive::Hess => minimize_method_hess(&obj, start.clone(), &ctl, arm.method.clone(), ls),
            Drive::Hvp => minimize_method_hvp(&obj, start.clone(), &ctl, arm.method.clone(), ls),
        }
        .unwrap_or_else(|e| panic!("{}: {e}", arm.name));
        if report.steps == 0 {
            continue;
        }
        let mut n2 = 0.0;
        for i in 0..start.len() {
            let d = report.coords[i] - start[i];
            n2 += d * d;
        }
        let n = n2.sqrt();
        assert!(
            n <= cap * (1.0 + 1.0e-8),
            "{} stepped {n}, past the cap {cap}",
            arm.name
        );
        assert!(
            (n - cap).abs() <= 1.0e-8 * cap,
            "{} stepped {n}, not the cap {cap}",
            arm.name
        );
    }
}
