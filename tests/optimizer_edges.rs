//! The same six checks for every local optimizer that reaches both
//! known minima.
//!
//! FIRE, FIRE 2.0, Barzilai-Borwein, and quick-min reach the quadratic
//! bowl and the one-dimensional parabola, and they stop on a zero
//! gradient. From the classic Rosenbrock start they do not reach the
//! minimum, so they are not in this table.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{
    Control, HessianObjective, HessianVector, LineSearch, Method, Report, Solver, Status,
    minimize_method, minimize_method_hess, minimize_method_hvp,
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
        Self {
            shape,
            dim,
            bounds: Bounds::new(
                Array1::from_elem(dim, -8.0),
                Array1::from_elem(dim, 8.0),
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
}

fn control(maxiter: usize) -> Control {
    Control {
        maxiter,
        gtol: 1e-8,
        istep: 1.0,
        maxmove: Some(f64::INFINITY),
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

fn run(arm: &Arm, obj: &Problem, x0: Array1<f64>, maxiter: usize) -> Report {
    let ctl = control(maxiter);
    let cg = matches!(arm.method, Method::Nlcg { .. }) || matches!(arm.drive, Drive::LowCall);
    let ls = linesearch(cg);
    match arm.drive {
        Drive::Plain => minimize_method(obj, x0, &ctl, arm.method.clone(), ls),
        Drive::Hess => minimize_method_hess(obj, x0, &ctl, arm.method.clone(), ls),
        Drive::Hvp => minimize_method_hvp(obj, x0, &ctl, arm.method.clone(), ls),
        Drive::LowCall => {
            let mut x = x0;
            let mut solver = Solver::low_call_cg(ctl.clone(), x.len());
            solver.set_maxmove(f64::INFINITY);
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
            Ok(report)
        }
    }
    .unwrap_or_else(|e| panic!("{}: {e}", arm.name))
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
    let obj = Problem::new(Shape::Quad, 2);
    let start = Array1::from_vec(vec![1.5, -2.0]);
    for arm in catalog() {
        let report = run(&arm, &obj, start.clone(), 400);
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
    let obj = Problem::new(Shape::Rose, 2);
    let start = Array1::from_vec(vec![-1.2, 1.0]);
    for arm in catalog() {
        let budget = match arm.name {
            "sd" => 8_000,
            "ode12r" => 4_000,
            _ => 800,
        };
        let report = run(&arm, &obj, start.clone(), budget);
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
    let obj = Problem::new(Shape::Quad, 2);
    for arm in catalog() {
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
        let obj = Problem::new(Shape::Energy(bad), 2);
        for arm in catalog() {
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
    let obj = Problem::new(Shape::Rose, 2);
    let start = Array1::from_vec(vec![-1.2, 1.0]);
    for arm in catalog() {
        let report = run(&arm, &obj, start.clone(), 1);
        assert_eq!(report.status, Status::Budget, "{}", arm.name);
        assert_eq!(report.steps, 1, "{}", arm.name);
        assert!(!at_rose(&report), "{}", arm.name);
    }
}

#[test]
fn every_listed_optimizer_solves_a_one_dimensional_parabola() {
    let obj = Problem::new(Shape::Parabola, 1);
    for arm in catalog() {
        let report = run(&arm, &obj, Array1::from_vec(vec![0.0]), 200);
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
