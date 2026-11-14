//! The same six checks for every local optimizer.
//!
//! FIRE, FIRE 2.0, Barzilai-Borwein, and quick-min take the documented
//! per-coordinate cap of 0.2 and an initial step of 0.1. That is ASE
//! 3.29's `dt` and `maxstep` (`ase/optimize/fire.py` and
//! `ase/optimize/optimize.py` at tag 3.29.0). An uncapped step on
//! Rosenbrock from `(-1.2, 1)` leaves the valley: the initial force is
//! about 200, so `dt = 0.1` with no cap is a jump, not a relaxation.
//! With the cap, FIRE 1.0 reaches `‖g‖ < 1e-8` in 4621 steps and the
//! default FIRE 2.0 schedule in 4622. ASE's
//! mix-then-kick FIRE, same `dt` and `maxstep`, takes 4791. The counts
//! differ because FIRE 1.0 kicks and then mixes on the new force
//! (Bitzek 2006); ASE mixes first. Both finish at the minimum. The
//! published FIRE 2.0 table (Guénolé et al. 2020, `α = 0.25`,
//! `dt_max = 10 dt`) stalls without a cap and reaches the minimum in
//! 7450 steps with it. A direct transcription of that algorithm with
//! the same cap takes 7517; the gap is the velocity rescale after a
//! shortened step.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{
    Control, FireVariant, HessianObjective, HessianVector, LineSearch, Method, Report, Solver,
    Status, minimize_method, minimize_method_hess, minimize_method_hvp,
};

#[derive(Clone, Copy)]
enum Shape {
    Quad,
    Rose,
    Parabola,
    Energy(f64),
}

struct Problem {
    shape: Shape,
    dim: usize,
    bounds: Bounds<f64>,
}

impl Problem {
    fn new(shape: Shape, dim: usize) -> Self {
        Self::boxed(shape, dim, 8.0)
    }

    fn wide(shape: Shape, dim: usize) -> Self {
        Self::boxed(shape, dim, 1.0e6)
    }

    fn boxed(shape: Shape, dim: usize, limit: f64) -> Self {
        Self {
            shape,
            dim,
            bounds: Bounds::new(
                Array1::from_elem(dim, -limit),
                Array1::from_elem(dim, limit),
                0.0,
            ),
        }
    }

    fn fg(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self.shape {
            Shape::Quad => {
                let mut f = 0.0;
                let mut g = Array1::zeros(x.len());
                for i in 0..x.len() {
                    f += 0.5 * x[i] * x[i];
                    g[i] = x[i];
                }
                (f, g)
            }
            Shape::Rose => {
                let (a, b) = (1.0 - x[0], x[1] - x[0] * x[0]);
                let f = a * a + 100.0 * b * b;
                let g = Array1::from_vec(vec![-2.0 * a - 400.0 * x[0] * b, 200.0 * b]);
                (f, g)
            }
            Shape::Parabola => {
                let d = x[0] - 2.0;
                (d * d, Array1::from_vec(vec![2.0 * d]))
            }
            Shape::Energy(bad) => (bad, Array1::ones(x.len())),
        }
    }

    fn hess(&self, x: ArrayView1<f64>) -> Array2<f64> {
        let n = x.len();
        match self.shape {
            Shape::Quad | Shape::Energy(_) => Array2::eye(n),
            Shape::Parabola => Array2::from_elem((1, 1), 2.0),
            Shape::Rose => {
                let (x0, y) = (x[0], x[1]);
                Array2::from_shape_vec((2, 2), vec![
                    2.0 - 400.0 * y + 1200.0 * x0 * x0,
                    -400.0 * x0,
                    -400.0 * x0,
                    200.0,
                ])
                .unwrap()
            }
        }
    }
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
    Plain,
    Hess,
    Hvp,
    LowCall,
    /// Session step with the 0.2 cap and `dt = 0.1`.
    Capped,
    /// Published FIRE 2.0 table, same cap and `dt`.
    Guenole,
}

fn control(maxiter: usize, drive: Drive) -> Control {
    Control {
        maxiter,
        gtol: 1e-8,
        istep: if inertial(drive) { 0.1 } else { 1.0 },
        maxmove: if inertial(drive) {
            None
        } else {
            Some(f64::INFINITY)
        },
        ftol_rel: None,
        noise_floor: None,
    }
}

fn linesearch(cg: bool) -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: if cg { 0.1 } else { 0.9 },
        maxiter: 30,
    }
}

struct Arm {
    name: &'static str,
    method: Method,
    drive: Drive,
}

fn inertial(drive: Drive) -> bool {
    matches!(drive, Drive::Capped | Drive::Guenole)
}

fn catalog() -> Vec<Arm> {
    vec![
        Arm {
            name: "sd",
            method: Method::Steepest,
            drive: Drive::Plain,
        },
        Arm {
            name: "cg",
            method: Method::from_name("cg").unwrap(),
            drive: Drive::Plain,
        },
        Arm {
            name: "fr",
            method: Method::from_name("fr").unwrap(),
            drive: Drive::Plain,
        },
        Arm {
            name: "hs",
            method: Method::from_name("hs").unwrap(),
            drive: Drive::Plain,
        },
        Arm {
            name: "dy",
            method: Method::from_name("dy").unwrap(),
            drive: Drive::Plain,
        },
        Arm {
            name: "dyh",
            method: Method::from_name("dyh").unwrap(),
            drive: Drive::Plain,
        },
        Arm {
            name: "cd",
            method: Method::from_name("cd").unwrap(),
            drive: Drive::Plain,
        },
        Arm {
            name: "hz",
            method: Method::from_name("hz").unwrap(),
            drive: Drive::Plain,
        },
        Arm {
            name: "ls",
            method: Method::from_name("ls").unwrap(),
            drive: Drive::Plain,
        },
        Arm {
            name: "fr-pr",
            method: Method::from_name("fr-pr").unwrap(),
            drive: Drive::Plain,
        },
        Arm {
            name: "pr+",
            method: Method::from_name("pr-plus").unwrap(),
            drive: Drive::Plain,
        },
        Arm {
            name: "low-call",
            method: Method::from_name("low_call_cg").unwrap(),
            drive: Drive::LowCall,
        },
        Arm {
            name: "bfgs",
            method: Method::Bfgs,
            drive: Drive::Plain,
        },
        Arm {
            name: "lbfgs",
            method: Method::lbfgs(),
            drive: Drive::Plain,
        },
        Arm {
            name: "lbfgs-wide",
            method: Method::lbfgs_wide(),
            drive: Drive::Plain,
        },
        Arm {
            name: "sr1",
            method: Method::Sr1,
            drive: Drive::Plain,
        },
        Arm {
            name: "sr2",
            method: Method::Sr2,
            drive: Drive::Plain,
        },
        Arm {
            name: "adam",
            method: Method::adam(),
            drive: Drive::Plain,
        },
        Arm {
            name: "ode12r",
            method: Method::Ode12r,
            drive: Drive::Plain,
        },
        Arm {
            name: "diis",
            method: Method::diis(),
            drive: Drive::Plain,
        },
        Arm {
            name: "pso",
            method: Method::pso(),
            drive: Drive::Plain,
        },
        Arm {
            name: "fire",
            method: Method::from_name("fire").unwrap(),
            drive: Drive::Capped,
        },
        Arm {
            name: "fire2",
            method: Method::from_name("fire2").unwrap(),
            drive: Drive::Capped,
        },
        Arm {
            name: "fire2-guenole",
            method: Method::from_name("fire2").unwrap(),
            drive: Drive::Guenole,
        },
        Arm {
            name: "bb",
            method: Method::Bb,
            drive: Drive::Capped,
        },
        Arm {
            name: "quickmin",
            method: Method::QuickMin,
            drive: Drive::Capped,
        },
        Arm {
            name: "newton",
            method: Method::from_name("newton").unwrap(),
            drive: Drive::Hess,
        },
        Arm {
            name: "rfo",
            method: Method::from_name("rfo").unwrap(),
            drive: Drive::Hess,
        },
        Arm {
            name: "dogleg",
            method: Method::Dogleg,
            drive: Drive::Hess,
        },
        Arm {
            name: "newton-cg",
            method: Method::NewtonCg,
            drive: Drive::Hvp,
        },
        Arm {
            name: "newton-pcg",
            method: Method::NewtonPcg,
            drive: Drive::Hvp,
        },
    ]
}

fn problem(arm: &Arm, shape: Shape, dim: usize) -> Problem {
    if inertial(arm.drive) {
        Problem::wide(shape, dim)
    } else {
        Problem::new(shape, dim)
    }
}

fn session_run(arm: &Arm, obj: &Problem, x0: Array1<f64>, ctl: &Control) -> Report {
    let mut x = x0;
    let mut solver = if matches!(arm.drive, Drive::LowCall) {
        let mut solver = Solver::low_call_cg(ctl.clone(), x.len());
        solver.set_maxmove(f64::INFINITY);
        solver
    } else {
        let mut solver = Solver::new(arm.method.clone(), ctl.clone(), x.len());
        if matches!(arm.drive, Drive::Guenole) {
            solver.set_fire_variant(FireVariant::Guenole2020);
        }
        // After the variant rebuild, so FIRE 2.0 keeps the cap it does
        // not install on its own.
        solver.set_atom_maxmove(0.2);
        solver
    };
    let mut last = None;
    for _ in 0..ctl.maxiter {
        let rep = solver
            .step(obj, &mut x)
            .unwrap_or_else(|e| panic!("{}: {e}", arm.name));
        let done = rep.grad_norm < ctl.gtol;
        last = Some(rep);
        if done {
            last.as_mut().unwrap().status = Status::Tolerance;
            break;
        }
    }
    let mut report = last.unwrap_or_else(|| panic!("{}: no step", arm.name));
    if report.status == Status::Iterating {
        report.status = Status::Budget;
    }
    report
}

fn run(arm: &Arm, obj: &Problem, x0: Array1<f64>, maxiter: usize) -> Report {
    let ctl = control(maxiter, arm.drive);
    let cg = matches!(arm.method, Method::Nlcg { .. }) || matches!(arm.drive, Drive::LowCall);
    let ls = linesearch(cg);
    match arm.drive {
        Drive::Plain => minimize_method(obj, x0, &ctl, arm.method.clone(), ls)
            .unwrap_or_else(|e| panic!("{}: {e}", arm.name)),
        Drive::Hess => minimize_method_hess(obj, x0, &ctl, arm.method.clone(), ls)
            .unwrap_or_else(|e| panic!("{}: {e}", arm.name)),
        Drive::Hvp => minimize_method_hvp(obj, x0, &ctl, arm.method.clone(), ls)
            .unwrap_or_else(|e| panic!("{}: {e}", arm.name)),
        Drive::LowCall | Drive::Capped | Drive::Guenole => session_run(arm, obj, x0, &ctl),
    }
}

fn at_origin(r: &Report) -> bool {
    r.value.abs() < 1e-6 && r.coords.iter().all(|v| v.abs() < 1e-4)
}

fn at_rose(r: &Report) -> bool {
    r.value.abs() < 1e-5 && (r.coords[0] - 1.0).abs() < 1e-3 && (r.coords[1] - 1.0).abs() < 1e-3
}

fn at_two(r: &Report) -> bool {
    r.value.abs() < 1e-6 && (r.coords[0] - 2.0).abs() < 1e-4
}

#[test]
fn every_listed_optimizer_reaches_the_quadratic_minimum() {
    let start = Array1::from_vec(vec![1.5, -2.0]);
    for arm in catalog() {
        let obj = problem(&arm, Shape::Quad, 2);
        let budget = if inertial(arm.drive) { 20_000 } else { 400 };
        let report = run(&arm, &obj, start.clone(), budget);
        assert_eq!(report.status, Status::Tolerance, "{}", arm.name);
        assert!(
            at_origin(&report),
            "{} value {} coords {:?}",
            arm.name,
            report.value,
            report.coords
        );
    }
}

#[test]
fn every_listed_optimizer_reaches_the_rosenbrock_minimum() {
    let start = Array1::from_vec(vec![-1.2, 1.0]);
    // Steps to `‖g‖ < 1e-8` with dt = 0.1 and the 0.2 cap.
    let capped_steps = [
        ("fire", 4621usize),
        ("fire2", 4622),
        ("bb", 208),
        ("quickmin", 15976),
        ("fire2-guenole", 7450),
    ];
    for arm in catalog() {
        let obj = problem(&arm, Shape::Rose, 2);
        let budget = if inertial(arm.drive) {
            20_000
        } else {
            match arm.name {
                "sd" => 8_000,
                "ode12r" => 4_000,
                _ => 800,
            }
        };
        let report = run(&arm, &obj, start.clone(), budget);
        if let Some((_, steps)) = capped_steps.iter().find(|(n, _)| *n == arm.name) {
            assert_eq!(report.steps, *steps, "{}", arm.name);
        }
        assert_eq!(report.status, Status::Tolerance, "{}", arm.name);
        assert!(
            at_rose(&report),
            "{} value {} coords {:?}",
            arm.name,
            report.value,
            report.coords
        );
    }
}

#[test]
fn a_zero_gradient_stops_at_the_known_minimum() {
    for arm in catalog() {
        let obj = problem(&arm, Shape::Quad, 2);
        let report = run(&arm, &obj, Array1::zeros(2), 5);
        assert_eq!(report.status, Status::Tolerance, "{}", arm.name);
        assert!(at_origin(&report), "{}", arm.name);
        // PSO still walks the swarm. The recorded best is the origin.
        if arm.name != "pso" {
            assert_eq!(report.steps, 0, "{}", arm.name);
        }
    }
}

#[test]
fn a_nonfinite_energy_is_not_reported_as_success() {
    let start = Array1::from_vec(vec![1.0, -1.0]);
    for bad in [f64::NAN, f64::INFINITY] {
        for arm in catalog() {
            let obj = problem(&arm, Shape::Energy(bad), 2);
            let report = run(&arm, &obj, start.clone(), 3);
            assert_eq!(report.status, Status::Budget, "{} {bad}", arm.name);
            assert!(
                !report.value.is_finite(),
                "{} value {}",
                arm.name,
                report.value
            );
            assert!(
                report.grad_norm.is_finite(),
                "{} g {}",
                arm.name,
                report.grad_norm
            );
            assert_eq!(report.steps, 3, "{}", arm.name);
        }
    }
}

#[test]
fn one_rosenbrock_step_stops_on_the_iteration_budget() {
    let start = Array1::from_vec(vec![-1.2, 1.0]);
    for arm in catalog() {
        let obj = problem(&arm, Shape::Rose, 2);
        let report = run(&arm, &obj, start.clone(), 1);
        assert_eq!(report.status, Status::Budget, "{}", arm.name);
        assert_eq!(report.steps, 1, "{}", arm.name);
        assert!(!at_rose(&report), "{}", arm.name);
    }
}

#[test]
fn every_listed_optimizer_solves_a_one_dimensional_parabola() {
    for arm in catalog() {
        let obj = problem(&arm, Shape::Parabola, 1);
        let budget = if inertial(arm.drive) { 20_000 } else { 200 };
        let report = run(&arm, &obj, Array1::from_vec(vec![0.0]), budget);
        assert_eq!(report.status, Status::Tolerance, "{}", arm.name);
        assert!(
            at_two(&report),
            "{} value {} x {:?}",
            arm.name,
            report.value,
            report.coords
        );
    }
}

/// Without the 0.2 cap, `dt = 0.1` on this start is a jump. The same
/// solvers reach the minimum once the cap is on, so the miss is the
/// cap, not the velocity update.
#[test]
fn an_uncapped_step_leaves_the_rosenbrock_valley() {
    let start = Array1::from_vec(vec![-1.2, 1.0]);
    for (name, method) in [
        ("fire", Method::from_name("fire").unwrap()),
        ("bb", Method::Bb),
        ("quickmin", Method::QuickMin),
    ] {
        let obj = Problem::wide(Shape::Rose, 2);
        let ctl = Control {
            maxiter: 200,
            gtol: 1e-8,
            istep: 0.1,
            maxmove: Some(f64::INFINITY),
            ftol_rel: None,
            noise_floor: None,
        };
        let mut x = start.clone();
        let mut solver = Solver::new(method, ctl, 2);
        solver.set_atom_maxmove(0.0);
        let mut value = f64::NAN;
        for _ in 0..200 {
            let rep = solver.step(&obj, &mut x).unwrap();
            value = rep.value;
        }
        assert!(
            value > 1.0 || x.iter().any(|v| v.abs() > 10.0),
            "{name} stayed in the valley: f {value} x {x:?}"
        );
    }
}
