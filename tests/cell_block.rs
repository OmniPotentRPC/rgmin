//! Cell coordinates appended to the atoms (`Solver::set_cell_block`).
//!
//! Two atoms sit in soft wells and two cell coordinates in a stiff one,
//! the way a strained box is stiffer than an atom. The per-atom cap is
//! 0.1 and the cell cap 0.002, against a cell that starts 0.05 from its
//! minimum, so every method must take at least 25 steps on the cell and
//! no step may move a cell coordinate further than the cell cap.

use std::sync::OnceLock;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, LbfgsStep, Method, Solver, StopNorm};

const ATOMS: usize = 6;
const DIM: usize = ATOMS + 2;
const STIFF: f64 = 1000.0;
const ATOM_CAP: f64 = 0.1;
const CELL_CAP: f64 = 0.002;

struct Wells;

impl Objective<f64> for Wells {
    fn dim(&self) -> usize {
        DIM
    }
    fn bounds(&self) -> &Bounds<f64> {
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| {
            Bounds::new(
                Array1::from_elem(DIM, -1.0e3),
                Array1::from_elem(DIM, 1.0e3),
                0.0,
            )
        })
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        wells(x).0
    }
}

impl Gradient<f64> for Wells {
    fn dim(&self) -> usize {
        DIM
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        wells(x).1
    }
}

impl DifferentiableObjective<f64> for Wells {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        wells(x)
    }
}

fn wells(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mut f = 0.0;
    let mut g = Array1::zeros(DIM);
    for k in 0..DIM {
        let stiff = if k < ATOMS { 1.0 } else { STIFF };
        f += 0.5 * stiff * x[k] * x[k];
        g[k] = stiff * x[k];
    }
    (f, g)
}

fn start() -> Array1<f64> {
    array![0.03, -0.02, 0.01, -0.04, 0.02, 0.05, 0.05, -0.03]
}

fn session(method: Method, block: bool) -> Solver {
    let control = Control::default().with_maxiter(2000).with_gtol(1e-8);
    let mut solver = Solver::new(method, control, DIM);
    solver.set_atom_maxmove(ATOM_CAP);
    if block {
        solver.set_cell_block(ATOMS, CELL_CAP).unwrap();
    }
    solver
}

fn largest_cell_move(before: &Array1<f64>, after: &Array1<f64>) -> f64 {
    (ATOMS..DIM)
        .map(|k| (after[k] - before[k]).abs())
        .fold(0.0, f64::max)
}

/// Relax from [`start`] and return the largest cell move of any step, the
/// number of steps and the final gradient norm.
fn relax(mut solver: Solver) -> (f64, usize, f64) {
    let mut x = start();
    let mut worst = 0.0_f64;
    for step in 1..=2000 {
        let before = x.clone();
        let report = solver.step(&Wells, &mut x).unwrap();
        worst = worst.max(largest_cell_move(&before, &x));
        if report.grad_norm < 1e-8 {
            return (worst, step, report.grad_norm);
        }
    }
    (worst, 2000, f64::INFINITY)
}

fn wolfe_lbfgs() -> Solver {
    let mut solver = session(Method::lbfgs(), true);
    solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
    solver
}

#[test]
fn every_step_keeps_the_cell_under_its_cap_and_converges() {
    let runs = [
        ("lbfgs", session(Method::lbfgs(), true)),
        ("lbfgs wolfe", wolfe_lbfgs()),
        ("sd", session(Method::Steepest, true)),
        ("bb", session(Method::Bb, true)),
    ];
    for (name, solver) in runs {
        let (worst, steps, gnorm) = relax(solver);
        assert!(
            worst <= CELL_CAP * (1.0 + 1e-12),
            "{name}: a step moved the cell by {worst}"
        );
        assert!(
            gnorm < 1e-8,
            "{name}: stopped at |g| = {gnorm} after {steps} steps"
        );
        assert!(
            steps >= 25,
            "{name}: {steps} steps cannot cover 0.05 at {CELL_CAP}"
        );
    }
}

#[test]
fn without_the_block_the_cell_moves_like_an_atom() {
    let mut x = start();
    let before = x.clone();
    session(Method::Steepest, false)
        .step(&Wells, &mut x)
        .unwrap();
    assert!(largest_cell_move(&before, &x) > 10.0 * CELL_CAP);
}

#[test]
fn max_atom_counts_each_cell_coordinate_on_its_own() {
    let x = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.006, 0.008];
    let mut plain = Solver::new(Method::lbfgs(), Control::default(), DIM);
    plain.set_stop_norm(StopNorm::MaxAtom);
    let mut split = Solver::new(Method::lbfgs(), Control::default(), DIM);
    split.set_stop_norm(StopNorm::MaxAtom);
    split.set_cell_block(ATOMS, CELL_CAP).unwrap();
    let mut xa = x.clone();
    let a = plain.step(&Wells, &mut xa).unwrap();
    let mut xb = x.clone();
    let b = split.step(&Wells, &mut xb).unwrap();
    // The report is the norm at the accepted point, so compare each with
    // the norm of the gradient there.
    let ga = wells(xa.view()).1;
    let gb = wells(xb.view()).1;
    let atoms = |g: &Array1<f64>| {
        (0..ATOMS / 3)
            .map(|i| (g[3 * i].powi(2) + g[3 * i + 1].powi(2) + g[3 * i + 2].powi(2)).sqrt())
            .fold(0.0, f64::max)
    };
    let together = atoms(&ga).max((ga[6].powi(2) + ga[7].powi(2)).sqrt());
    let apart = atoms(&gb).max(gb[6].abs()).max(gb[7].abs());
    assert_eq!(a.grad_norm, together);
    assert_eq!(b.grad_norm, apart);
}

#[test]
fn the_split_must_fall_on_whole_atoms_inside_the_vector() {
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), DIM);
    assert!(solver.set_cell_block(DIM + 1, CELL_CAP).is_err());
    assert!(solver.set_cell_block(4, CELL_CAP).is_err());
    assert_eq!(solver.cell_block(), None);
    solver.set_cell_block(ATOMS, CELL_CAP).unwrap();
    assert_eq!(solver.cell_block(), Some((ATOMS, CELL_CAP)));
    solver.set_cell_block(0, f64::INFINITY).unwrap();
    assert_eq!(solver.cell_block(), Some((0, f64::INFINITY)));
    solver.set_cell_block(DIM, CELL_CAP).unwrap();
    assert_eq!(solver.cell_block(), None);
}
