//! Two-loop L-BFGS vs HiGHS QP on the same L-BFGS memory.
//! Repo defaults: memory 8, Wolfe c1=1e-4 c2=0.9, gtol 1e-6, max 200.

use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rgmin::{HighsStep, Lbfgs};

fn rosenbrock(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let a = 1.0 - x[0];
    let b = x[1] - x[0] * x[0];
    let f = a * a + 100.0 * b * b;
    let g = Array1::from(vec![-2.0 * a - 400.0 * x[0] * b, 200.0 * b]);
    (f, g)
}

fn quad(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let scales = [1.0, 10.0, 100.0, 1000.0];
    let mut f = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..x.len() {
        let c = scales[i % scales.len()];
        f += 0.5 * c * x[i] * x[i];
        g[i] = c * x[i];
    }
    (f, g)
}

fn run(
    label: &str,
    mut opt: Lbfgs,
    x0: ArrayView1<f64>,
    fg: impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>),
) {
    let t0 = Instant::now();
    let (f, x, evals) = opt.minimize(x0, 200, |v| Some(fg(v)));
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let g = fg(x.view()).1;
    let ginf = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let xmin = x.iter().fold(f64::INFINITY, |a, v| a.min(*v));
    let xmax = x.iter().fold(f64::NEG_INFINITY, |a, v| a.max(*v));
    println!(
        "{:<22} f={:.6e} evals={} ginf={:.3e} ms={:.3} xmin={:.4} xmax={:.4} pairs={}",
        label,
        f,
        evals,
        ginf,
        ms,
        xmin,
        xmax,
        opt.len()
    );
}

fn main() {
    println!("# unconstrained Rosenbrock 2D, start (-1.2, 1.0)");
    let x0 = Array1::from(vec![-1.2, 1.0]);
    run("two-loop", Lbfgs::default(), x0.view(), rosenbrock);
    let mut h = Lbfgs::default();
    h.highs = Some(HighsStep::default());
    run("highs-qp", h, x0.view(), rosenbrock);

    println!("# unconstrained ill-conditioned quadratic n=8");
    let q0 = Array1::from(vec![1.0; 8]);
    run("two-loop", Lbfgs::default(), q0.view(), quad);
    let mut h = Lbfgs::default();
    h.highs = Some(HighsStep::default());
    run("highs-qp", h, q0.view(), quad);

    println!("# boxed quadratic n=8, lo=0 hi=0.5 (true min is 0, outside the box)");
    run("two-loop (no box)", Lbfgs::default(), q0.view(), quad);
    let mut h = Lbfgs::default();
    h.highs = Some(HighsStep {
        trust: None,
        lo: Some(vec![0.0]),
        hi: Some(vec![0.5]),
        equalities: Vec::new(),
        center_axes: None,
    });
    run("highs-qp box[0,0.5]", h, q0.view(), quad);
}
