//! An uncapped inertial step on Rosenbrock.
//!
//! FIRE, FIRE 2.0, Barzilai-Borwein, and quick-min take the documented
//! per-coordinate cap of 0.2 and an initial step of 0.1. That is ASE
//! 3.29's `dt` and `maxstep` (`ase/optimize/fire.py` and
//! `ase/optimize/optimize.py` at tag 3.29.0). The per-optimizer checks
//! live next to this file. Without the cap, `dt = 0.1` on this start is
//! a jump: the initial force is about 200. The same solvers reach the
//! minimum once the cap is on, so the miss is the cap, not the velocity
//! update. With the cap, FIRE 1.0 reaches `‖g‖ < 1e-8` in 4621 steps and
//! the default FIRE 2.0 schedule in 4622. ASE's mix-then-kick FIRE, same
//! `dt` and `maxstep`, takes 4791. The counts differ because FIRE 1.0
//! kicks and then mixes on the new force (Bitzek 2006); ASE mixes first.
//! The published FIRE 2.0 table (Guénolé et al. 2020, `α = 0.25`,
//! `dt_max = 10 dt`) stalls without a cap and reaches the minimum in
//! 7450 steps with it. A direct transcription of that algorithm with the
//! same cap takes 7517; the gap is the velocity rescale after a
//! shortened step.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rgmin::{Control, Method, Solver};

struct Problem {
    bounds: Bounds<f64>,
}

impl Problem {
    fn wide() -> Self {
        Self {
            bounds: Bounds::new(
                Array1::from_elem(2, -1.0e6),
                Array1::from_elem(2, 1.0e6),
                0.0,
            ),
        }
    }

    fn fg(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let (a, b) = (1.0 - x[0], x[1] - x[0] * x[0]);
        let f = a * a + 100.0 * b * b;
        let g = Array1::from_vec(vec![-2.0 * a - 400.0 * x[0] * b, 200.0 * b]);
        (f, g)
    }
}

impl Objective<f64> for Problem {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        Self::fg(x).0
    }
}

impl Gradient<f64> for Problem {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Self::fg(x).1
    }
}

impl DifferentiableObjective<f64> for Problem {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        Self::fg(x)
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
        let obj = Problem::wide();
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
