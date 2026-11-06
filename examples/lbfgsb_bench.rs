//! `Lbfgs` against SciPy L-BFGS-B, traced evaluation by evaluation.
//!
//! `lj75`: the starts of anneal `examples/warm_quench_lj.rs`. A
//! 75-point Lennard-Jones cluster is relaxed once from a seeded random
//! start, then copied 400 times, each copy perturbed uniformly by up to
//! 0.38 per coordinate, with pairs closer than 0.85 pushed apart. No
//! coordinate is bounded. `driver1`: the bound-constrained extended
//! Rosenbrock function of the L-BFGS-B 3.0 distribution on 25
//! variables, from its own start `x = 3` and from 19 seeded starts
//! inside the box.
//!
//! Arms: `lbfgs` is `Lbfgs::minimize` with the defaults (8 pairs,
//! `||g||_inf < 1e-6`). `lbfgs-box` is `Lbfgs::minimize_box`, the
//! generalized Cauchy point and subspace step, with the same settings
//! on `lj75` and with 5 pairs and `1e-5` on `driver1`, the values of the
//! L-BFGS-B driver. `lbfgs-approx` and `lbfgs-box-approx` repeat them
//! with `approx_wolfe = 1e-12`, which turns on the approximate Wolfe
//! test. The iteration cap is 3000 per run.
//!
//! The starts go to the file named by the first argument as
//! `problem x_1 ... x_n`, one per line, so that
//! `docs/figures/lbfgsb/bench_lbfgsb.py` hands SciPy the same points.
//! The trace goes to stdout as CSV
//! (`problem,arm,start,eval,f,ginf,accepted`). `ginf` is the infinity
//! norm of the projected gradient and `accepted` is 1 on an iterate the
//! solver adopted.
//!
//! `cargo run --release --example lbfgsb_bench -- starts.txt > rgmin.csv`

use std::cell::RefCell;
use std::fmt::Write as _;

use ndarray::{Array1, ArrayView1};
use rgmin::Lbfgs;

/// anneal's xorshift stream, so the perturbations match its protocol.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        lo + (hi - lo) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)
    }
}

/// Lennard-Jones value and gradient in reduced units, no cutoff.
fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let inv2 = 1.0 / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            let inv6 = inv2 * inv2 * inv2;
            e += 4.0 * (inv6 * inv6 - inv6);
            let coef = 24.0 * inv2 * (2.0 * inv6 * inv6 - inv6);
            for k in 0..3 {
                g[3 * i + k] -= coef * d[k];
                g[3 * j + k] += coef * d[k];
            }
        }
    }
    (e, g)
}

/// Pushes apart any pair closer than `min_sep`, as anneal does.
fn repair(x: &mut Array1<f64>, min_sep: f64) {
    let n = x.len() / 3;
    for _ in 0..50 {
        let mut worst = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = [
                    x[3 * i] - x[3 * j],
                    x[3 * i + 1] - x[3 * j + 1],
                    x[3 * i + 2] - x[3 * j + 2],
                ];
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if r < min_sep && r > 1e-12 {
                    let push = 0.5 * (min_sep - r) / r;
                    for k in 0..3 {
                        x[3 * i + k] += push * d[k];
                        x[3 * j + k] -= push * d[k];
                    }
                    worst = worst.max(min_sep - r);
                } else if r <= 1e-12 {
                    x[3 * i] += 0.5 * min_sep;
                    worst = min_sep;
                }
            }
        }
        if worst == 0.0 {
            return;
        }
    }
}

fn lj75_starts() -> Vec<Array1<f64>> {
    let mut rng = Rng(20260802);
    let radius = 0.9 * 75f64.cbrt();
    let mut seed = Array1::from_shape_fn(225, |_| rng.uniform(-radius, radius));
    repair(&mut seed, 0.85);
    let (_, base, _) = Lbfgs::default().minimize(seed.view(), 3000, |v| Some(lj(v)));
    (0..400)
        .map(|_| {
            let mut p = base.clone();
            p.mapv_inplace(|v| v + rng.uniform(-0.38, 0.38));
            repair(&mut p, 0.85);
            p
        })
        .collect()
}

/// `driver1`: odd entries (1-based) in `[1, 100]`, even in `[-100, 100]`.
fn driver1(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len();
    let mut f = 0.25 * (x[0] - 1.0).powi(2);
    let mut g = Array1::zeros(n);
    g[0] = 0.5 * (x[0] - 1.0);
    for i in 1..n {
        let t = x[i] - x[i - 1] * x[i - 1];
        f += 4.0 * t * t;
        g[i] += 8.0 * t;
        g[i - 1] -= 16.0 * x[i - 1] * t;
    }
    (f, g)
}

fn driver1_starts() -> Vec<Array1<f64>> {
    let mut rng = Rng(19970101);
    let mut out = vec![Array1::from_elem(25, 3.0)];
    for _ in 0..19 {
        out.push(Array1::from_shape_fn(25, |i| {
            if i % 2 == 0 {
                rng.uniform(1.0, 4.0)
            } else {
                rng.uniform(-4.0, 4.0)
            }
        }));
    }
    out
}

type Fg = fn(ArrayView1<f64>) -> (f64, Array1<f64>);

fn trace(
    problem: &str,
    arm: &str,
    k: usize,
    p: &Array1<f64>,
    fg: Fg,
    bounds: &(Array1<f64>, Array1<f64>),
) {
    let (lo, hi) = bounds;
    let log = RefCell::new(Vec::<(f64, f64, bool)>::new());
    let logged = |v: ArrayView1<f64>| {
        let (f, g) = fg(v);
        let pg = (0..v.len()).fold(0.0_f64, |m, i| {
            m.max(g[i].max(v[i] - hi[i]).min(v[i] - lo[i]).abs())
        });
        log.borrow_mut().push((f, pg, false));
        Some((f, g))
    };
    let watch = |_: usize, f: f64| {
        if let Some(last) = log.borrow_mut().last_mut() {
            last.2 |= last.0 == f;
        }
        true
    };
    let mut opt = Lbfgs::default();
    if problem == "driver1" {
        opt = Lbfgs::with_capacity(5);
        opt.gtol = 1e-5;
    }
    if arm.ends_with("-approx") {
        opt.approx_wolfe = 1e-12;
    }
    let (f, evals) = if arm.starts_with("lbfgs-box") {
        let (f, _, e) =
            opt.minimize_box_watched(p.view(), lo.view(), hi.view(), 3000, logged, watch);
        (f, e)
    } else {
        let (f, _, e) = opt.minimize_watched(p.view(), 3000, logged, watch);
        (f, e)
    };
    let log = log.into_inner();
    assert_eq!(log.len(), evals);
    for (i, (fv, gv, acc)) in log.iter().enumerate() {
        println!(
            "{problem},{arm},{k},{},{fv:e},{gv:e},{}",
            i + 1,
            u8::from(*acc)
        );
    }
    eprintln!("{problem} {arm} start {k}: {evals} evals, f = {f:.10}");
}

fn main() {
    let path = std::env::args().nth(1).expect("path for the starts");
    let lj75 = lj75_starts();
    let drv = driver1_starts();
    let mut text = String::new();
    for (name, starts) in [("lj75", &lj75), ("driver1", &drv)] {
        for p in starts {
            let row: Vec<String> = p.iter().map(|v| format!("{v:e}")).collect();
            writeln!(text, "{name} {}", row.join(" ")).unwrap();
        }
    }
    std::fs::write(&path, text).expect("write the starts");

    let free = (
        Array1::from_elem(225, f64::NEG_INFINITY),
        Array1::from_elem(225, f64::INFINITY),
    );
    let boxed = (
        Array1::from_shape_fn(25, |i| if i % 2 == 0 { 1.0 } else { -100.0 }),
        Array1::from_elem(25, 100.0),
    );
    println!("problem,arm,start,eval,f,ginf,accepted");
    for arm in ["lbfgs", "lbfgs-approx", "lbfgs-box", "lbfgs-box-approx"] {
        for (k, p) in lj75.iter().enumerate() {
            trace("lj75", arm, k, p, lj, &free);
        }
    }
    for arm in ["lbfgs-box", "lbfgs-box-approx"] {
        for (k, p) in drv.iter().enumerate() {
            trace("driver1", arm, k, p, driver1, &boxed);
        }
    }
}
