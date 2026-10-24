//! ODE12r through the session: one oracle call per trial, a rejected
//! trial that leaves the iterate alone, the displacement cap, and the
//! Exp metric on a chain whose Hessian is that metric, where the
//! integration takes Newton steps.

use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Exp, ExpParams, Method, Neighbours, Solver};

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

fn session(istep: f64, dim: usize) -> Solver {
    let control = Control {
        maxiter: 1,
        gtol: 0.0,
        istep,
        ..Control::default()
    };
    Solver::new(Method::Ode12r, control, dim)
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
fn converges_with_one_call_per_trial() {
    let obj = Quad::new(array![1.0, 4.0, 9.0]);
    let mut x = array![1.0, -1.0, 0.5];
    let mut solver = session(0.05, 3);
    let (gn, steps) = run(&mut solver, &obj, &mut x, 500);
    assert!(gn < 1e-8, "gnorm {gn} after {steps} steps");
    // The first step evaluates the start; every trial after that is one
    // oracle call, accepted or not.
    assert_eq!(obj.calls.load(Ordering::Relaxed), steps + 1);
    assert!(steps < 120, "took {steps} steps");
}

#[test]
fn a_rejected_trial_keeps_the_iterate_and_costs_one_call() {
    // h k = 40: forward Euler overshoots by a factor 39, the residual
    // test fails, and the session keeps x.
    let obj = Quad::new(array![400.0]);
    let mut x = array![0.01];
    let mut solver = session(0.1, 1);
    let rep = solver.step(&obj, &mut x).unwrap();
    assert_eq!(x, array![0.01]);
    assert_eq!(obj.calls.load(Ordering::Relaxed), 2);
    assert!((rep.grad_norm - 4.0).abs() < 1e-12);
    // The retry reads the cached start and tries a shorter step.
    solver.step(&obj, &mut x).unwrap();
    assert_eq!(obj.calls.load(Ordering::Relaxed), 3);
    let mut x_end = x.clone();
    let (gn, _) = run(&mut solver, &obj, &mut x_end, 500);
    assert!(gn < 1e-8, "gnorm {gn}");
}

#[test]
fn the_displacement_cap_binds_far_from_the_minimum() {
    let obj = Quad::new(Array1::from_elem(3, 50.0));
    let mut solver = session(0.1, 3);
    solver.set_atom_maxmove(0.05);
    let mut x = array![10.0, 0.0, 0.0];
    for _ in 0..20 {
        let before = x.clone();
        solver.step(&obj, &mut x).unwrap();
        let moved = (&x - &before).mapv(|d| d * d).sum().sqrt();
        assert!(moved <= 0.05 * (1.0 + 1e-12), "moved {moved}");
    }
    assert!(x[0] < 10.0 - 0.5, "did not advance: {}", x[0]);
}

#[test]
fn forget_reopens_at_the_initial_step() {
    let obj = Quad::new(array![1.0, 1.0]);
    let mut x = array![1.0, 1.0];
    let mut solver = session(0.2, 2);
    for _ in 0..5 {
        solver.step(&obj, &mut x).unwrap();
    }
    solver.forget();
    let before = x.clone();
    solver.step(&obj, &mut x).unwrap();
    // One accepted Euler step of length 0.2 on E = |x|^2 / 2 scales x by 0.8.
    let expected = before.mapv(|v| 0.8 * v);
    assert!(
        (&x - &expected).mapv(f64::abs).sum() < 1e-12,
        "{x:?} vs {expected:?}"
    );
}

/// Quadratic `E = (x - x0) . P (x - x0) / 2` whose Hessian is the Exp
/// metric of a chain and whose minimum is the chain itself.
struct FixedMetric {
    p: Exp,
    centre: Array1<f64>,
    bounds: Bounds<f64>,
}

impl Objective<f64> for FixedMetric {
    fn dim(&self) -> usize {
        self.p.dim()
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let d = &x - &self.centre;
        0.5 * d.dot(&self.p.apply(d.view()))
    }
}

impl Gradient<f64> for FixedMetric {
    fn dim(&self) -> usize {
        self.p.dim()
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let d = &x - &self.centre;
        self.p.apply(d.view())
    }
}

impl DifferentiableObjective<f64> for FixedMetric {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let d = &x - &self.centre;
        let g = self.p.apply(d.view());
        (0.5 * d.dot(&g), g)
    }
}

fn chain(n: usize) -> (Array1<f64>, Neighbours, ExpParams) {
    let mut pos = Array1::zeros(3 * n);
    let mut neigh = Neighbours::default();
    for i in 0..n {
        pos[3 * i] = i as f64;
        if i + 1 < n {
            neigh.i.push(i);
            neigh.j.push(i + 1);
            neigh.r.push(1.0);
        }
    }
    let params = ExpParams {
        r_nn: Some(1.0),
        mu: Some(1.0),
        ..ExpParams::default()
    };
    (pos, neigh, params)
}

fn chain_steps(precon: bool) -> usize {
    let n = 30;
    let (geom, neigh, params) = chain(n);
    let metric = Exp::assemble(geom.view(), &neigh, &params, 1.0, 1.0).unwrap();
    let obj = FixedMetric {
        p: metric,
        centre: geom.clone(),
        bounds: Bounds::new(
            Array1::from_elem(3 * n, -1e6),
            Array1::from_elem(3 * n, 1e6),
            0.0,
        ),
    };
    let mut x = geom.clone();
    x[1] = 0.2;
    x[4] = -0.15;
    let control = Control {
        maxiter: 1,
        gtol: 1e-8,
        istep: 0.1,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    };
    let mut solver = Solver::new(Method::Ode12r, control, x.len());
    if precon {
        solver.set_exp(geom.view(), &neigh, &params).unwrap();
    }
    let mut taken = 0;
    for _ in 0..400 {
        let g = obj.grad(x.view());
        let gnorm = g.dot(&g).sqrt();
        if gnorm < 1e-8 {
            break;
        }
        solver.step(&obj, &mut x).unwrap();
        taken += 1;
    }
    eprintln!("chain with exp metric {precon}: {taken} trials");
    taken
}

#[test]
fn the_exp_metric_takes_the_newton_steps_on_the_chain() {
    // With P equal to the Hessian, P^{-1} F points at the minimum and the
    // force interpolant of the paper's line search vanishes at the Newton
    // length: h_ls = h (F . y) / (dF . y) = 1 for every tried h. The error
    // control h_ode12 = (1/2) sqrt(2 rtol / ||x - x0||_inf) grows as the
    // chain closes on the minimum, and once it passes 1 the step is the
    // Newton step: 0.1, 0.4, 0.53, 0.68, 0.99, 1, converged. Measured:
    // six trials with the metric, 93 without, where the stiff bond modes
    // bound the Euler step.
    let with = chain_steps(true);
    let without = chain_steps(false);
    assert!(with <= 8, "expected the Newton steps, took {with} trials");
    assert!(
        without >= 10 * with,
        "preconditioned {with} trials, plain {without}"
    );
}
