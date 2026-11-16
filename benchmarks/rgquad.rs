//! Scaled quadratic and a diagonal restricted step.
//!
//! The spectrum is `lambda_i = 10^(6 i / (n-1))`, the start is the
//! all-ones vector, and the budget is 200 steps at `gtol = 1e-8`.
//! That is the L-BFGS quadratic cell. FIRE uses time step 0.1. The
//! spectrum and the start are built before the clock.
//!
//! `rgquad reps <fire|fire2|cg|trust|dtrust|lj> <n> <reps>` prints one
//! row per repetition. `trust` is the diagonal spectrum. `dtrust` is
//! that spectrum rotated by the Householder factor of a seeded
//! Gaussian. `lj` is the Lennard-Jones Hessian of a 2x2x2 cube; `n` is
//! ignored. `rgquad session <method> <n>` builds the cell, writes
//! `ready`, then answers `run` until `quit`.

use std::env;
use std::f64::consts::PI;
use std::io::{self, BufRead, Write};
use std::time::Instant;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rgmin::{Control, FireKind, LineSearch, Method, TrustRegion, minimize_method};

/// Seed for the Gaussian whose Householder factor rotates the spectrum.
const DENSE_SEED: u64 = 7;

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

fn standard_normal(rng: &mut StdRng) -> f64 {
    let mut u1 = rng.random::<f64>();
    if u1 == 0.0 {
        u1 = f64::MIN_POSITIVE;
    }
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Orthogonal factor of a seeded Gaussian, by Householder QR.
fn householder_q(n: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(DENSE_SEED);
    let mut a = Array2::zeros((n, n));
    for v in a.iter_mut() {
        *v = standard_normal(&mut rng);
    }
    let mut q = Array2::<f64>::eye(n);
    for k in 0..n {
        let mut norm = 0.0;
        for i in k..n {
            norm += a[(i, k)] * a[(i, k)];
        }
        norm = norm.sqrt();
        if norm == 0.0 {
            continue;
        }
        let sign = if a[(k, k)] >= 0.0 { 1.0 } else { -1.0 };
        let mut v = vec![0.0; n];
        for i in k..n {
            v[i] = a[(i, k)];
        }
        v[k] += sign * norm;
        let mut vv = 0.0;
        for i in k..n {
            vv += v[i] * v[i];
        }
        if vv == 0.0 {
            continue;
        }
        let tau = 2.0 / vv;
        for j in k..n {
            let mut dot = 0.0;
            for i in k..n {
                dot += v[i] * a[(i, j)];
            }
            let scale = tau * dot;
            for i in k..n {
                a[(i, j)] -= scale * v[i];
            }
        }
        for j in 0..n {
            let mut dot = 0.0;
            for i in k..n {
                dot += v[i] * q[(i, j)];
            }
            let scale = tau * dot;
            for i in k..n {
                q[(i, j)] -= scale * v[i];
            }
        }
    }
    q.t().to_owned()
}

/// `Q diag(lambda) Q^T` and the gradient at the all-ones vector.
fn dense_quadratic(n: usize) -> (Array2<f64>, Array1<f64>) {
    let lam = spectrum(n);
    let q = householder_q(n);
    let mut hess = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(i, k)] * lam[k] * q[(j, k)];
            }
            hess[(i, j)] = sum;
            hess[(j, i)] = sum;
        }
    }
    let mut grad = Array1::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += hess[(i, j)];
        }
        grad[i] = sum;
    }
    (hess, grad)
}

/// Lennard-Jones Hessian of a 2x2x2 cube, sigma = 1, every pair included.
fn lennard_jones_cube() -> (Array2<f64>, Array1<f64>) {
    let spacing = 1.2;
    let mut pos = Vec::new();
    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                pos.push([
                    x as f64 * spacing,
                    y as f64 * spacing,
                    z as f64 * spacing,
                ]);
            }
        }
    }
    let atoms = pos.len();
    let dim = 3 * atoms;
    let mut hess = Array2::zeros((dim, dim));
    let mut grad = Array1::zeros(dim);
    for i in 0..atoms {
        for j in (i + 1)..atoms {
            let mut diff = [0.0; 3];
            let mut r2 = 0.0;
            for a in 0..3 {
                diff[a] = pos[i][a] - pos[j][a];
                r2 += diff[a] * diff[a];
            }
            let r = r2.sqrt();
            let up = -24.0 * (2.0 * r.powi(-13) - r.powi(-7));
            let upp = 24.0 * (26.0 * r.powi(-14) - 7.0 * r.powi(-8));
            let s = up / r;
            let ds = (upp * r - up) / r2;
            for a in 0..3 {
                grad[3 * i + a] += s * diff[a];
                grad[3 * j + a] -= s * diff[a];
                for b in 0..3 {
                    let block = if a == b { s } else { 0.0 } + ds * diff[a] * diff[b] / r;
                    let ia = 3 * i + a;
                    let ib = 3 * i + b;
                    let ja = 3 * j + a;
                    let jb = 3 * j + b;
                    hess[(ia, ib)] += block;
                    hess[(ja, jb)] += block;
                    hess[(ia, jb)] -= block;
                    hess[(ja, ib)] -= block;
                }
            }
        }
    }
    (hess, grad)
}

fn prepare(name: &str, n: usize) -> Cell {
    if name == "trust" || name == "dtrust" || name == "lj" {
        let (hess, grad) = if name == "dtrust" {
            dense_quadratic(n)
        } else if name == "lj" {
            lennard_jones_cube()
        } else {
            let grad = Array1::from_vec(spectrum(n));
            (Array2::from_diag(&grad), grad)
        };
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

    #[test]
    fn the_rotated_spectrum_is_dense_symmetric_and_repeatable() {
        let (hess, grad) = dense_quadratic(8);
        let (again, grad2) = dense_quadratic(8);
        let mut off = false;
        for i in 0..8 {
            assert_eq!(grad[i].to_bits(), grad2[i].to_bits());
            for j in 0..8 {
                assert_eq!(hess[(i, j)].to_bits(), hess[(j, i)].to_bits());
                assert_eq!(hess[(i, j)].to_bits(), again[(i, j)].to_bits());
                if i != j && hess[(i, j)] != 0.0 {
                    off = true;
                }
            }
        }
        assert!(off);
    }

    #[test]
    fn the_lennard_jones_cube_has_a_symmetric_hessian() {
        let (hess, grad) = lennard_jones_cube();
        assert_eq!(hess.nrows(), 24);
        assert!(grad.iter().any(|v| v.is_finite() && *v != 0.0));
        for i in 0..24 {
            for j in 0..24 {
                assert_eq!(hess[(i, j)].to_bits(), hess[(j, i)].to_bits());
                assert!(hess[(i, j)].is_finite());
            }
        }
    }
}
