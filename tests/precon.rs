//! Exponential preconditioner: the paper's μ, and fewer iterations
//! when H0 is that matrix.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rgmin::{Control, Exp, ExpParams, Method, Neighbours, Solver, estimate_mu};

struct FixedMetric {
    p: Exp,
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
        0.5 * x.dot(&self.p.apply(x))
    }
}

impl Gradient<f64> for FixedMetric {
    fn dim(&self) -> usize {
        self.p.dim()
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.p.apply(x)
    }
}

impl DifferentiableObjective<f64> for FixedMetric {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let g = self.p.apply(x);
        (0.5 * x.dot(&g), g)
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
    let mut params = ExpParams::default();
    params.r_nn = Some(1.0);
    params.mu = Some(1.0);
    (pos, neigh, params)
}

fn steps_until(precon: bool, estimate: bool) -> (usize, Option<f64>) {
    let n = 30;
    let (geom, neigh, mut params) = chain(n);
    let metric = Exp::assemble(geom.view(), &neigh, &params, 1.0, 1.0).unwrap();
    let obj = FixedMetric {
        p: metric,
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
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 5 }, control, x.len());
    if precon {
        if estimate {
            params.mu = None;
        }
        solver.set_exp(geom.view(), &neigh, &params).unwrap();
    }
    let mut taken = 0;
    for _ in 0..40 {
        let gnorm = obj.grad(x.view()).dot(&obj.grad(x.view())).sqrt();
        if gnorm < 1e-8 {
            break;
        }
        solver.step(&obj, &mut x).unwrap();
        taken += 1;
    }
    (taken, solver.exp_mu())
}

#[test]
fn preconditioned_lbfgs_takes_fewer_iterations_on_a_chain() {
    let (with, mu) = steps_until(true, false);
    let (without, _) = steps_until(false, false);
    assert!(
        with < without,
        "preconditioned {with} iterations, plain {without}"
    );
    assert!(with <= 3, "expected a short Newton-like run, took {with}");
    assert!((mu.unwrap() - 1.0).abs() < 1e-12);

    let (estimated_steps, estimated_mu) = steps_until(true, true);
    assert!(
        (estimated_mu.unwrap() - 1.0).abs() < 1e-6,
        "on-the-fly mu {}",
        estimated_mu.unwrap()
    );
    assert!(estimated_steps <= 3, "estimated run took {estimated_steps}");
}

#[test]
fn fire_and_nlcg_move_in_the_metric() {
    let n = 8;
    let (geom, neigh, params) = chain(n);
    let metric = Exp::assemble(geom.view(), &neigh, &params, 1.0, 1.0).unwrap();
    let obj = FixedMetric {
        p: metric,
        bounds: Bounds::new(
            Array1::from_elem(3 * n, -1e6),
            Array1::from_elem(3 * n, 1e6),
            0.0,
        ),
    };
    let control = Control {
        maxiter: 1,
        gtol: 1e-10,
        istep: 0.05,
        maxmove: None,
        ftol_rel: None,
    };
    for method in [
        Method::Fire {
            kind: rgmin::FireKind::V1,
        },
        Method::nlcg(rgmin::Conjugacy::PolakRibiere),
    ] {
        let mut plain = geom.clone();
        plain[0] = 0.3;
        let mut pre = plain.clone();
        let mut a = Solver::new(method.clone(), control.clone(), plain.len());
        let mut b = Solver::new(method, control.clone(), pre.len());
        b.set_exp(geom.view(), &neigh, &params).unwrap();
        a.step(&obj, &mut plain).unwrap();
        b.step(&obj, &mut pre).unwrap();
        let delta: f64 = plain
            .iter()
            .zip(pre.iter())
            .map(|(u, v)| (u - v).abs())
            .sum();
        assert!(delta > 1e-8, "metric did not change the step ({delta})");
    }
}

#[test]
fn estimate_mu_recovers_a_scaled_metric() {
    let (geom, neigh, mut params) = chain(6);
    params.lengths = Some([8.0, 2.0, 2.0]);
    let mu_true = 3.5;
    let p1 = Exp::assemble(geom.view(), &neigh, &params, 1.0, 1.0).unwrap();
    let grad = |x: ArrayView1<f64>| mu_true * p1.apply(x);
    let mu = estimate_mu(geom.view(), &neigh, &params, grad).unwrap();
    assert!((mu - mu_true).abs() < 1e-8, "mu {mu}");
}
