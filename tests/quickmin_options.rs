//! Quick-min session options: the eOn step, the per-atom projection,
//! the curvature bound, and one oracle call per step.

use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, QuickMinOptions, Solver};

/// `E = sum k_i x_i^2 / 2`, counting oracle calls.
struct Quad {
    k: Array1<f64>,
    bounds: Bounds<f64>,
    calls: AtomicUsize,
}

impl Quad {
    fn new(k: Array1<f64>) -> Self {
        let n = k.len();
        Self {
            k,
            bounds: Bounds::new(Array1::from_elem(n, -1e6), Array1::from_elem(n, 1e6), 0.0),
            calls: AtomicUsize::new(0),
        }
    }
}

impl Objective<f64> for Quad {
    fn dim(&self) -> usize {
        self.k.len()
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * self.k.iter().zip(x).map(|(k, v)| k * v * v).sum::<f64>()
    }
}

impl Gradient<f64> for Quad {
    fn dim(&self) -> usize {
        self.k.len()
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        &self.k * &x
    }
}

impl DifferentiableObjective<f64> for Quad {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        (self.eval(x), self.grad(x))
    }
}

fn session(options: QuickMinOptions, istep: f64, dim: usize) -> Solver {
    let control = Control {
        maxiter: 1,
        gtol: 0.0,
        istep,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::QuickMin, control, dim);
    solver.set_quickmin_options(options);
    solver
}

fn run(solver: &mut Solver, obj: &Quad, x: &mut Array1<f64>, steps: usize) -> (f64, usize) {
    let mut gn = f64::INFINITY;
    let mut taken = 0;
    for _ in 0..steps {
        let rep = solver.step(obj, x).unwrap();
        taken += 1;
        gn = rep.grad_norm;
        if gn < 1e-8 {
            break;
        }
    }
    (gn, taken)
}

#[test]
fn the_eon_step_converges_with_one_call_per_step() {
    let obj = Quad::new(array![1.0, 4.0, 9.0]);
    let mut x = array![1.0, -1.0, 0.5];
    let mut solver = session(QuickMinOptions::eon(), 0.2, 3);
    let (gn, steps) = run(&mut solver, &obj, &mut x, 2000);
    assert!(gn < 1e-8, "gnorm {gn}");
    // The first step evaluates the start; each step after that one point.
    assert_eq!(obj.calls.load(Ordering::Relaxed), steps + 1);
}

#[test]
fn both_options_move_at_the_cap_far_from_the_minimum() {
    // E = 25 |x|^2 from a far start: the per-atom cap binds every step,
    // with the eOn velocity left alone and with the default rescale.
    let obj = Quad::new(Array1::from_elem(3, 50.0));
    let cases = [
        (QuickMinOptions::eon(), false),
        (QuickMinOptions::rgmin(), true),
    ];
    for (options, rescaled) in cases {
        let mut solver = session(options, 0.1, 3);
        solver.set_atom_maxmove(0.05);
        let mut x = array![10.0, 0.0, 0.0];
        for _ in 0..20 {
            let before = x.clone();
            solver.step(&obj, &mut x).unwrap();
            let moved = (&x - &before).mapv(|d| d * d).sum().sqrt();
            assert!(moved <= 0.05 * (1.0 + 1e-12), "moved {moved}");
        }
        assert!(x[0] < 10.0 - 0.9, "rescaled {rescaled}: x {}", x[0]);
    }
}

#[test]
fn the_curvature_bound_tames_an_unstable_time_step() {
    // dt = 1 is five times the stability limit 2 / sqrt(100) of the stiff
    // mode. The eOn step diverges; the curvature bound brings dt under
    // the limit after one step and converges.
    let obj = Quad::new(array![1.0, 100.0]);
    let mut eon = session(QuickMinOptions::eon(), 1.0, 2);
    // The default per-atom cap would hide the unstable step.
    eon.set_atom_maxmove(0.0);
    let mut y = array![1.0, 0.1];
    let e0 = obj.eval(y.view());
    for _ in 0..20 {
        eon.step(&obj, &mut y).unwrap();
    }
    assert!(
        obj.eval(y.view()) > 1e3 * e0,
        "eOn energy {}",
        obj.eval(y.view())
    );

    let mut o = QuickMinOptions::eon();
    o.curvature_cap = Some(1.0);
    let mut solver = session(o, 1.0, 2);
    let mut x = array![1.0, 0.1];
    let (gn, _) = run(&mut solver, &obj, &mut x, 5000);
    assert!(gn < 1e-8, "gnorm {gn}");
}

#[test]
fn per_atom_projection_converges() {
    let obj = Quad::new(array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut solver = session(QuickMinOptions::per_atom(), 0.2, 6);
    let mut x = array![1.0, -1.0, 0.5, 0.2, -0.3, 0.7];
    let (gn, _) = run(&mut solver, &obj, &mut x, 3000);
    assert!(gn < 1e-8, "gnorm {gn}");
}

#[test]
fn forget_restarts_from_rest() {
    let obj = Quad::new(array![1.0, 2.0]);
    let mut solver = session(QuickMinOptions::eon(), 0.2, 2);
    let mut x = array![1.0, 1.0];
    for _ in 0..5 {
        solver.step(&obj, &mut x).unwrap();
    }
    solver.forget();
    let start = x.clone();
    let g = obj.grad(start.view());
    solver.step(&obj, &mut x).unwrap();
    // From rest the move is dt^2 F.
    for i in 0..2 {
        assert!((x[i] - (start[i] - 0.04 * g[i])).abs() < 1e-14);
    }
}
