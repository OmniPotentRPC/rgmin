#![allow(dead_code)]

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{
    Control, FireVariant, HessianObjective, HessianVector, LineSearch, Method, Report, Solver,
    Status, minimize_method, minimize_method_hess, minimize_method_hvp,
};

#[derive(Clone, Copy)]
pub enum Shape {
    Quad,
    Rose,
    Parabola,
    Energy(f64),
}

pub struct Problem {
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
pub enum Drive {
    Plain,
    Hess,
    Hvp,
    LowCall,
    /// Session step with the 0.2 cap and `dt = 0.1`.
    Capped,
    /// Published FIRE 2.0 table, same cap and `dt`.
    Guenole,
}

pub fn control(maxiter: usize, drive: Drive) -> Control {
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

pub fn linesearch(cg: bool) -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: if cg { 0.1 } else { 0.9 },
        maxiter: 30,
    }
}

pub struct Arm {
    name: &'static str,
    method: Method,
    drive: Drive,
}

pub fn inertial(drive: Drive) -> bool {
    matches!(drive, Drive::Capped | Drive::Guenole)
}

pub fn problem(arm: &Arm, shape: Shape, dim: usize) -> Problem {
    if inertial(arm.drive) {
        Problem::wide(shape, dim)
    } else {
        Problem::new(shape, dim)
    }
}

pub fn session_run(arm: &Arm, obj: &Problem, x0: Array1<f64>, ctl: &Control) -> Report {
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

pub fn run(arm: &Arm, obj: &Problem, x0: Array1<f64>, maxiter: usize) -> Report {
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

pub fn reaches_the_quadratic_minimum(arm: &Arm) {
    let start = Array1::from_vec(vec![1.5, -2.0]);
    let obj = problem(arm, Shape::Quad, 2);
    let budget = if inertial(arm.drive) { 20_000 } else { 400 };
    let report = run(arm, &obj, start, budget);
    assert_eq!(report.status, Status::Tolerance, "{}", arm.name);
    assert!(
        at_origin(&report),
        "{} value {} coords {:?}",
        arm.name,
        report.value,
        report.coords
    );
}

pub fn reaches_the_rosenbrock_minimum(arm: &Arm) {
    let start = Array1::from_vec(vec![-1.2, 1.0]);
    let capped_steps = [
        ("fire", 4621usize),
        ("fire2", 4622),
        ("bb", 208),
        ("quickmin", 15976),
        ("fire2-guenole", 7450),
    ];
    let obj = problem(arm, Shape::Rose, 2);
    let budget = if inertial(arm.drive) {
        20_000
    } else {
        match arm.name {
            "sd" => 8_000,
            "ode12r" => 4_000,
            _ => 800,
        }
    };
    let report = run(arm, &obj, start, budget);
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

pub fn a_zero_gradient_stops_at_the_known_minimum(arm: &Arm) {
    let obj = problem(arm, Shape::Quad, 2);
    let report = run(arm, &obj, Array1::zeros(2), 5);
    assert_eq!(report.status, Status::Tolerance, "{}", arm.name);
    assert!(at_origin(&report), "{}", arm.name);
    if arm.name != "pso" {
        assert_eq!(report.steps, 0, "{}", arm.name);
    }
}

pub fn a_nonfinite_energy_is_not_reported_as_success(arm: &Arm) {
    let start = Array1::from_vec(vec![1.0, -1.0]);
    for bad in [f64::NAN, f64::INFINITY] {
        let obj = problem(arm, Shape::Energy(bad), 2);
        let report = run(arm, &obj, start.clone(), 3);
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

pub fn one_rosenbrock_step_stops_on_the_iteration_budget(arm: &Arm) {
    let start = Array1::from_vec(vec![-1.2, 1.0]);
    let obj = problem(arm, Shape::Rose, 2);
    let report = run(arm, &obj, start, 1);
    assert_eq!(report.status, Status::Budget, "{}", arm.name);
    assert_eq!(report.steps, 1, "{}", arm.name);
    assert!(!at_rose(&report), "{}", arm.name);
}

pub fn solves_a_one_dimensional_parabola(arm: &Arm) {
    let obj = problem(arm, Shape::Parabola, 1);
    let budget = if inertial(arm.drive) { 20_000 } else { 200 };
    let report = run(arm, &obj, Array1::from_vec(vec![0.0]), budget);
    assert_eq!(report.status, Status::Tolerance, "{}", arm.name);
    assert!(
        at_two(&report),
        "{} value {} x {:?}",
        arm.name,
        report.value,
        report.coords
    );
}

fn plain(name: &'static str, method: Method) -> Arm {
    Arm {
        name,
        method,
        drive: Drive::Plain,
    }
}

pub fn sd() -> Arm {
    plain("sd", Method::Steepest)
}
pub fn cg() -> Arm {
    plain("cg", Method::from_name("cg").unwrap())
}
pub fn fr() -> Arm {
    plain("fr", Method::from_name("fr").unwrap())
}
pub fn hs() -> Arm {
    plain("hs", Method::from_name("hs").unwrap())
}
pub fn dy() -> Arm {
    plain("dy", Method::from_name("dy").unwrap())
}
pub fn dyh() -> Arm {
    plain("dyh", Method::from_name("dyh").unwrap())
}
pub fn cd() -> Arm {
    plain("cd", Method::from_name("cd").unwrap())
}
pub fn hz() -> Arm {
    plain("hz", Method::from_name("hz").unwrap())
}
pub fn ls() -> Arm {
    plain("ls", Method::from_name("ls").unwrap())
}
pub fn fr_pr() -> Arm {
    plain("fr-pr", Method::from_name("fr-pr").unwrap())
}
pub fn pr_plus() -> Arm {
    plain("pr+", Method::from_name("pr-plus").unwrap())
}
pub fn low_call() -> Arm {
    Arm {
        name: "low-call",
        method: Method::from_name("low_call_cg").unwrap(),
        drive: Drive::LowCall,
    }
}
pub fn bfgs() -> Arm {
    plain("bfgs", Method::Bfgs)
}
pub fn lbfgs() -> Arm {
    plain("lbfgs", Method::lbfgs())
}
pub fn lbfgs_wide() -> Arm {
    plain("lbfgs-wide", Method::lbfgs_wide())
}
pub fn sr1() -> Arm {
    plain("sr1", Method::Sr1)
}
pub fn sr2() -> Arm {
    plain("sr2", Method::Sr2)
}
pub fn adam() -> Arm {
    plain("adam", Method::adam())
}
pub fn ode12r() -> Arm {
    plain("ode12r", Method::Ode12r)
}
pub fn diis() -> Arm {
    plain("diis", Method::diis())
}
pub fn pso() -> Arm {
    plain("pso", Method::pso())
}
pub fn fire() -> Arm {
    Arm {
        name: "fire",
        method: Method::from_name("fire").unwrap(),
        drive: Drive::Capped,
    }
}
pub fn fire2() -> Arm {
    Arm {
        name: "fire2",
        method: Method::from_name("fire2").unwrap(),
        drive: Drive::Capped,
    }
}
pub fn fire2_guenole() -> Arm {
    Arm {
        name: "fire2-guenole",
        method: Method::from_name("fire2").unwrap(),
        drive: Drive::Guenole,
    }
}
pub fn bb() -> Arm {
    Arm {
        name: "bb",
        method: Method::Bb,
        drive: Drive::Capped,
    }
}
pub fn quickmin() -> Arm {
    Arm {
        name: "quickmin",
        method: Method::QuickMin,
        drive: Drive::Capped,
    }
}
pub fn newton() -> Arm {
    Arm {
        name: "newton",
        method: Method::from_name("newton").unwrap(),
        drive: Drive::Hess,
    }
}
pub fn rfo() -> Arm {
    Arm {
        name: "rfo",
        method: Method::from_name("rfo").unwrap(),
        drive: Drive::Hess,
    }
}
pub fn dogleg() -> Arm {
    Arm {
        name: "dogleg",
        method: Method::Dogleg,
        drive: Drive::Hess,
    }
}
pub fn newton_cg() -> Arm {
    Arm {
        name: "newton-cg",
        method: Method::NewtonCg,
        drive: Drive::Hvp,
    }
}
pub fn newton_pcg() -> Arm {
    Arm {
        name: "newton-pcg",
        method: Method::NewtonPcg,
        drive: Drive::Hvp,
    }
}
