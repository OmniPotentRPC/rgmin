//! Scaled quadratic and a diagonal restricted step.
//!
//! The spectrum is `lambda_i = 10^(6 i / (n-1))`, the start is the
//! all-ones vector, and the budget is 200 steps at `gtol = 1e-8`.
//! That is the L-BFGS quadratic cell. FIRE uses time step 0.1. The
//! spectrum and the start are built before the clock.
//!
//! `rgquad reps <fire|fire2|cg|trust> <n> <reps>` prints one row per
//! repetition. `rgquad session <method> <n>` builds the cell, writes
//! `ready`, then answers `run` until `quit`.

use std::env;
use std::io::{self, BufRead, Write};
use std::time::Instant;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{Control, FireKind, LineSearch, Method, TrustRegion, minimize_method};

struct Quad {
    lam: Vec<f64>,
    bounds: Bounds<f64>,
}

impl Quad {
    fn new(n: usize) -> Self {
        Self {
            lam: spectrum(n),
            bounds: Bounds::new(Array1::from_elem(n, -1e6), Array1::from_elem(n, 1e6), 0.0),
        }
    }
}

impl Objective<f64> for Quad {
    fn dim(&self) -> usize {
        self.lam.len()
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let mut f = 0.0;
        for i in 0..x.len() {
            f += 0.5 * self.lam[i] * x[i] * x[i];
        }
        f
    }
}

impl Gradient<f64> for Quad {
    fn dim(&self) -> usize {
        self.lam.len()
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let mut g = Array1::zeros(x.len());
        for i in 0..x.len() {
            g[i] = self.lam[i] * x[i];
        }
        g
    }
}

impl DifferentiableObjective<f64> for Quad {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn spectrum(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if n <= 1 {
                1.0
            } else {
                10f64.powf(6.0 * (i as f64) / (n - 1) as f64)
            }
        })
        .collect()
}

enum Cell {
    Opt {
        obj: Quad,
        x0: Array1<f64>,
        control: Control,
        method: Method,
        linesearch: LineSearch,
    },
    Trust {
        hess: Array2<f64>,
        grad: Array1<f64>,
        region: TrustRegion,
    },
}

fn prepare(name: &str, n: usize) -> Cell {
    if name == "trust" {
        let grad = Array1::from_vec(spectrum(n));
        let hess = Array2::from_diag(&grad);
        return Cell::Trust {
            hess,
            grad,
            region: TrustRegion::new(1.0),
        };
    }
    let istep = if name == "fire" || name == "fire2" {
        0.1
    } else {
        1.0
    };
    let method = match name {
        "fire" => Method::Fire { kind: FireKind::V1 },
        "fire2" => Method::Fire { kind: FireKind::V2 },
        "cg" => Method::polak_ribiere(),
        other => panic!("unknown method {other}"),
    };
    let linesearch = if name == "cg" {
        rgmin::linesearch::CG_LINESEARCH
    } else {
        LineSearch::Wolfe {
            c1: 1e-4,
            c2: 0.9,
            maxiter: 20,
        }
    };
    Cell::Opt {
        obj: Quad::new(n),
        x0: Array1::from_elem(n, 1.0),
        control: Control {
            maxiter: 200,
            gtol: 1e-8,
            istep,
            maxmove: Some(f64::INFINITY),
            ftol_rel: None,
            noise_floor: None,
        },
        method,
        linesearch,
    }
}

/// One solve. Returns `(steps, gradient norm)`.
fn once(cell: &Cell) -> (usize, f64) {
    match cell {
        Cell::Opt {
            obj,
            x0,
            control,
            method,
            linesearch,
        } => {
            let report = minimize_method(obj, x0.clone(), control, method.clone(), *linesearch)
                .expect("solve");
            let kept = std::hint::black_box(report);
            (kept.steps, kept.grad_norm)
        }
        Cell::Trust { hess, grad, region } => {
            let step = region.restrict_qn(hess, grad).expect("trust step");
            let kept = std::hint::black_box(step);
            (1, kept.cons)
        }
    }
}

fn timed(cell: &Cell) -> (f64, usize, f64) {
    let t = Instant::now();
    let (steps, g) = once(cell);
    (t.elapsed().as_secs_f64(), steps, g)
}

fn main() {
    let mut args = env::args().skip(1);
    let mode = args.next().expect("mode");
    let method = args.next().expect("method");
    let n: usize = args.next().expect("n").parse().expect("n");
    let cell = prepare(&method, n);
    let mut out = io::stdout();
    match mode.as_str() {
        "reps" => {
            let reps: usize = args.next().expect("reps").parse().expect("reps");
            for rep in 0..reps {
                let (wall, steps, g) = timed(&cell);
                writeln!(
                    out,
                    "n {n} method {method} rep {rep} steps {steps} g {g:.6e} wall {wall:.6}"
                )
                .unwrap();
                out.flush().unwrap();
            }
        }
        "session" => {
            writeln!(out, "ready").unwrap();
            out.flush().unwrap();
            for line in io::stdin().lock().lines() {
                let line = line.unwrap();
                match line.trim() {
                    "quit" => break,
                    "run" => {
                        let (wall, steps, g) = timed(&cell);
                        writeln!(out, "ok {wall:.6} {steps} {g:.6e}").unwrap();
                        out.flush().unwrap();
                    }
                    _ => {}
                }
            }
        }
        other => panic!("unknown mode {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_quadratic_runs_match() {
        for method in ["fire", "fire2", "cg"] {
            let cell = prepare(method, 5);
            let (steps, g) = once(&cell);
            let (again, g2) = once(&cell);
            assert_eq!(steps, again, "{method}");
            assert_eq!(g.to_bits(), g2.to_bits(), "{method}");
        }
    }

    #[test]
    fn repeated_trust_steps_match() {
        let cell = prepare("trust", 5);
        let (_, cons) = once(&cell);
        let (_, again) = once(&cell);
        assert!(cons.is_finite());
        assert_eq!(cons.to_bits(), again.to_bits());
    }
}
