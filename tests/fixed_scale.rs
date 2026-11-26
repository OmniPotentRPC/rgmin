//! Opt-in inverse-Hessian scale and ASE line search.
//! Unset, both leave the session step that shipped before they existed.

use std::sync::OnceLock;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, LineSearch, Method, Solver};

struct Quadratic;

impl Objective<f64> for Quadratic {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        static BOUNDS: OnceLock<Bounds<f64>> = OnceLock::new();
        BOUNDS.get_or_init(|| Bounds::new(array![-1e6], array![1e6], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * x.dot(&x)
    }
}

impl Gradient<f64> for Quadratic {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}

impl DifferentiableObjective<f64> for Quadratic {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn open_control() -> Control {
    Control::default().with_istep(1.0).with_maxmove(None)
}

fn bfgs() -> Solver {
    let mut solver = Solver::new(Method::Bfgs, open_control(), 1);
    solver.set_atom_maxmove(0.0);
    solver.set_accept(Accept::Energy);
    solver
}

#[test]
fn an_unset_ase_search_takes_the_full_quadratic_step() {
    let mut solver = bfgs();
    let mut x = Array1::from_elem(1, 1.0);
    solver.step(&Quadratic, &mut x).unwrap();
    assert_eq!(x[0], 0.0);
}

fn ase_bfgs(cap: f64) -> Solver {
    let mut solver = bfgs();
    solver.set_atom_maxmove(cap);
    solver.set_linesearch(LineSearch::wolfe(0.23, 0.46, 20));
    solver.set_ase_linesearch(50.0).unwrap();
    solver
}

#[test]
fn the_ase_search_limits_each_increment_by_the_atom_cap() {
    // ASE evaluates 0.8, 0.6 and 0.4 and stops on its strong-Wolfe test.
    // rgmin's own search under the same cap stops at the cap, 0.8.
    let mut ase = ase_bfgs(0.2);
    let mut x = Array1::from_elem(1, 1.0);
    ase.step(&Quadratic, &mut x).unwrap();
    assert!((x[0] - 0.4).abs() < 1e-12, "{}", x[0]);

    let mut own = bfgs();
    own.set_atom_maxmove(0.2);
    own.set_linesearch(LineSearch::wolfe(0.23, 0.46, 20));
    let mut y = Array1::from_elem(1, 1.0);
    own.step(&Quadratic, &mut y).unwrap();
    assert!((y[0] - 0.8).abs() < 1e-12, "{}", y[0]);
}

#[test]
fn the_ase_search_opens_at_the_unit_step_whatever_istep_is() {
    let mut solver = Solver::new(Method::Bfgs, open_control().with_istep(0.1), 1);
    solver.set_atom_maxmove(0.0);
    solver.set_accept(Accept::Energy);
    solver.set_linesearch(LineSearch::wolfe(0.23, 0.46, 20));
    solver.set_ase_linesearch(50.0).unwrap();
    let mut x = Array1::from_elem(1, 1.0);
    solver.step(&Quadratic, &mut x).unwrap();
    assert_eq!(x[0], 0.0);
}

#[test]
fn the_ase_search_refuses_a_step_it_would_not_run() {
    let mut solver = ase_bfgs(0.2);
    solver.set_linesearch(LineSearch::default());
    let mut x = Array1::from_elem(1, 1.0);
    assert!(solver.step(&Quadratic, &mut x).is_err());
    assert_eq!(x[0], 1.0);

    let mut capped = Solver::new(Method::Bfgs, open_control().with_maxmove(Some(0.5)), 1);
    capped.set_linesearch(LineSearch::wolfe(0.23, 0.46, 20));
    capped.set_ase_linesearch(50.0).unwrap();
    assert!(capped.step(&Quadratic, &mut x).is_err());
}

#[test]
fn the_ase_search_is_refused_off_bfgs_and_lbfgs_and_for_nan() {
    let mut solver = bfgs();
    assert!(solver.set_ase_linesearch(f64::NAN).is_err());
    let mut x = Array1::from_elem(1, 1.0);
    solver.step(&Quadratic, &mut x).unwrap();
    assert_eq!(x[0], 0.0);
    let mut steepest = Solver::new(Method::Steepest, open_control(), 1);
    assert!(steepest.set_ase_linesearch(50.0).is_err());
    let mut lbfgs = Solver::new(Method::lbfgs(), open_control(), 1);
    assert!(lbfgs.set_ase_linesearch(50.0).is_ok());
}

#[test]
fn fixed_h0_scales_a_fresh_bfgs_step_without_a_search() {
    let mut scaled = Solver::new(Method::Bfgs, open_control(), 1);
    scaled.set_atom_maxmove(0.0);
    scaled.set_accept(Accept::Step);
    scaled.set_fixed_h0(0.5).unwrap();
    let mut x = Array1::from_elem(1, 1.0);
    scaled.step(&Quadratic, &mut x).unwrap();
    assert_eq!(x[0], 0.5);

    let mut identity = Solver::new(Method::Bfgs, open_control(), 1);
    identity.set_atom_maxmove(0.0);
    identity.set_accept(Accept::Step);
    let mut y = Array1::from_elem(1, 1.0);
    identity.step(&Quadratic, &mut y).unwrap();
    assert_eq!(y[0], 0.0);
}

#[test]
fn forget_restores_the_bfgs_scale() {
    let mut solver = Solver::new(Method::Bfgs, open_control(), 1);
    solver.set_atom_maxmove(0.0);
    solver.set_accept(Accept::Step);
    solver.set_fixed_h0(0.5).unwrap();
    let mut x = Array1::from_elem(1, 1.0);
    solver.step(&Quadratic, &mut x).unwrap();
    solver.forget();
    x[0] = 1.0;
    solver.step(&Quadratic, &mut x).unwrap();
    assert_eq!(x[0], 0.5);
}

#[test]
fn fixed_h0_scales_an_lbfgs_step_without_a_search() {
    let mut scaled = Solver::new(Method::lbfgs(), open_control(), 1);
    scaled.set_atom_maxmove(0.0);
    scaled.set_accept(Accept::Step);
    scaled.set_fixed_h0(0.5).unwrap();
    let mut x = Array1::from_elem(1, 1.0);
    scaled.step(&Quadratic, &mut x).unwrap();

    let mut plain = Solver::new(Method::lbfgs(), open_control(), 1);
    plain.set_atom_maxmove(0.0);
    plain.set_accept(Accept::Step);
    let mut y = Array1::from_elem(1, 1.0);
    plain.step(&Quadratic, &mut y).unwrap();
    assert_ne!(x[0], y[0]);
    assert_eq!(x[0], 0.5);
}

#[test]
fn fixed_h0_is_refused_on_steepest_descent() {
    let mut solver = Solver::new(Method::Steepest, open_control(), 1);
    assert!(solver.set_fixed_h0(0.5).is_err());
    assert!(solver.set_fixed_h0(0.0).is_err());
    assert!(solver.set_fixed_h0(f64::NAN).is_err());
}
