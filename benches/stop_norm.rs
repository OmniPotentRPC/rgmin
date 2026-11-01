//! Session stop points under the two-norm and the max-atom criterion.
//!
//! Geometry L-BFGS (memory 20, `istep = 0.2`) and FIRE (`dt = 0.1`),
//! both with the per-atom cap 0.2, on LJ38 from seeded random starts
//! and on the two Pt256 Morse fixtures. Each run stops when the report
//! clears the tolerance. Every oracle call counts. The final forces are
//! evaluated again outside the session.
//!
//! A second table runs five L-BFGS step rules on those problems and on
//! boxed Rosenbrock 2D. With the cap, the radius-on arm takes the trust
//! step in place of geometry L-BFGS. The two capped rows compare two
//! step rules, not one rule with the radius toggled. Without the cap,
//! the radius-off arms are the raw two-loop under `Accept::None` and
//! the strong Wolfe search under `Accept::Energy`. Budget hits are runs
//! still unconverged after the call or step budget. Errors are runs
//! that stopped on a session error.
//! `cargo bench --bench stop_norm`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use eindir_core::objectives::Rosenbrock;
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rgmin::{Accept, Control, FireKind, Method, Oracle, Solver, StopNorm};

const BUDGET: usize = 10_000;
const REPEATS: usize = 3;

struct Counted<O> {
    inner: O,
    calls: AtomicUsize,
}

impl<O: Objective<f64>> Objective<f64> for Counted<O> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }
}

impl<O: DifferentiableObjective<f64>> Gradient<f64> for Counted<O> {
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.grad(x)
    }
}

impl<O: DifferentiableObjective<f64>> DifferentiableObjective<f64> for Counted<O> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.value_and_gradient(x)
    }
}

/// Pair energy and gradient. `pair(r)` returns `(u, du/dr)`.
fn pairwise(
    x: ArrayView1<f64>,
    cell: Option<[f64; 3]>,
    cut: f64,
    pair: impl Fn(f64) -> (f64, f64),
) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            for a in 0..3 {
                d[a] = x[3 * i + a] - x[3 * j + a];
                if let Some(l) = cell {
                    d[a] -= l[a] * (d[a] / l[a]).round();
                }
            }
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r >= cut {
                continue;
            }
            let (u, du) = pair(r);
            e += u;
            for a in 0..3 {
                g[3 * i + a] += du * d[a] / r;
                g[3 * j + a] -= du * d[a] / r;
            }
        }
    }
    (e, g)
}

fn lj(r: f64) -> (f64, f64) {
    let s6 = r.powi(-6);
    (4.0 * (s6 * s6 - s6), 24.0 * (s6 - 2.0 * s6 * s6) / r)
}

/// tests/newton_cg_morse.rs: Pt Morse, shifted at the 7.0 cutoff.
fn morse(r: f64) -> (f64, f64) {
    let (d0, a, r0): (f64, f64, f64) = (0.7102, 1.6047, 2.897);
    let xc = (-a * (7.0 - r0)).exp();
    let x = (-a * (r - r0)).exp();
    (
        d0 * (x * x - 2.0 * x) - d0 * (xc * xc - 2.0 * xc),
        2.0 * d0 * a * x * (1.0 - x),
    )
}

/// Uniform in a cube at reduced density 0.5, no pair closer than 0.85.
fn lj_start(n: usize, seed: u64) -> Array1<f64> {
    let side = (n as f64 / 0.5).cbrt();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x: Vec<f64> = Vec::with_capacity(3 * n);
    while x.len() < 3 * n {
        let p: [f64; 3] = std::array::from_fn(|_| rng.random::<f64>() * side);
        let clear = x.chunks(3).all(|q| {
            let r2: f64 = (0..3).map(|a| (q[a] - p[a]).powi(2)).sum();
            r2 >= 0.85 * 0.85
        });
        if clear {
            x.extend(p);
        }
    }
    Array1::from(x)
}

fn read_con(name: &str) -> (Array1<f64>, [f64; 3]) {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).expect("fixture");
    let lines: Vec<&str> = text.lines().collect();
    let num = |l: &str| -> Vec<f64> {
        l.split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect()
    };
    let cell = num(lines[2]);
    let start = lines
        .iter()
        .position(|l| l.starts_with("Coordinates"))
        .unwrap();
    let mut x = Vec::new();
    for l in &lines[start + 1..] {
        let v = num(l);
        if v.len() < 3 {
            break;
        }
        x.extend(&v[..3]);
    }
    (Array1::from(x), [cell[0], cell[1], cell[2]])
}

fn max_atom(g: &Array1<f64>) -> f64 {
    g.exact_chunks(3)
        .into_iter()
        .map(|f| f.dot(&f).sqrt())
        .fold(0.0, f64::max)
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    method: fn() -> Method,
    istep: f64,
    stop: Option<StopNorm>,
    cap: Option<f64>,
    accept: Accept,
    radius: f64,
}

struct Run {
    converged: bool,
    error: bool,
    calls: usize,
    energy: f64,
    fmax: f64,
    gnorm: f64,
}

fn solve<O: DifferentiableObjective<f64>>(inner: O, x0: &Array1<f64>, arm: Arm, tol: f64) -> Run {
    let obj = Counted {
        inner,
        calls: AtomicUsize::new(0),
    };
    let control = Control {
        maxiter: BUDGET,
        gtol: tol,
        istep: arm.istep,
        maxmove: None,
    };
    let mut solver = Solver::new((arm.method)(), control, x0.len());
    if let Some(cap) = arm.cap {
        solver.set_atom_maxmove(cap);
    }
    solver.set_accept(arm.accept);
    solver.set_lbfgs_trust(arm.radius);
    if let Some(norm) = arm.stop {
        solver.set_stop_norm(norm);
    }
    let mut x = x0.clone();
    let mut converged = false;
    let mut error = false;
    for _ in 0..BUDGET {
        match solver.step(&obj, &mut x) {
            Ok(rep) if rep.grad_norm < tol => {
                converged = true;
                break;
            }
            Ok(_) => {}
            Err(_) => {
                error = true;
                break;
            }
        }
        if obj.calls.load(Ordering::Relaxed) > BUDGET {
            break;
        }
    }
    let calls = obj.calls.load(Ordering::Relaxed);
    let (energy, g) = obj.inner.value_and_gradient(x.view());
    Run {
        converged,
        error,
        calls,
        energy,
        fmax: max_atom(&g),
        gnorm: g.dot(&g).sqrt(),
    }
}

/// `outcomes` adds the budget-hit and error counts after `converged`.
fn run_case<F>(label: &str, arms: &[Arm], tols: &[f64], starts: usize, outcomes: bool, mut one: F)
where
    F: FnMut(usize, Arm, f64) -> Run,
{
    for &tol in tols {
        for &arm in arms {
            let mut runs = Vec::new();
            let mut times = Vec::new();
            for rep in 0..REPEATS {
                let t0 = Instant::now();
                for s in 0..starts {
                    let run = one(s, arm, tol);
                    if rep == 0 {
                        runs.push(run);
                    }
                }
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(f64::total_cmp);
            let k = starts as f64;
            let mut calls: Vec<usize> = runs.iter().map(|r| r.calls).collect();
            calls.sort_unstable();
            let mean = |v: fn(&Run) -> f64| runs.iter().map(v).sum::<f64>() / k;
            let worst = runs.iter().map(|r| r.fmax).fold(0.0, f64::max);
            let converged = runs.iter().filter(|r| r.converged).count();
            let mut done = format!("{converged}/{starts}");
            if outcomes {
                let errors = runs.iter().filter(|r| r.error).count();
                let budget = starts - converged - errors;
                done = format!("{done} | {budget} | {errors}");
            }
            println!(
                "| {label} | {} | {} | {tol} | {done} | {:.1} | {} | {} | {:.6} | {:.6} | {:.6} | {:.6} | {:.2} |",
                arm.name,
                match arm.stop {
                    Some(norm) => format!("{norm:?}"),
                    None => "default".into(),
                },
                mean(|r| r.calls as f64),
                calls[starts / 2],
                calls[starts - 1],
                mean(|r| r.fmax),
                worst,
                mean(|r| r.gnorm),
                mean(|r| r.energy),
                times[REPEATS / 2],
            );
        }
    }
}

fn main() {
    let lbfgs = || Method::Lbfgs { memory: 20 };
    let fire = || Method::Fire { kind: FireKind::V1 };
    let mut arms = Vec::new();
    for (name, method, istep) in [
        ("lbfgs m=20", lbfgs as fn() -> Method, 0.2),
        ("fire", fire, 0.1),
    ] {
        for stop in [None, Some(StopNorm::MaxAtom)] {
            arms.push(Arm {
                name,
                method,
                istep,
                stop,
                cap: Some(0.2),
                accept: Accept::None,
                radius: 0.0,
            });
        }
    }
    let tols = [0.05, 0.01];
    println!(
        "| case | method | stop | tol | converged | mean calls | median | max | mean final max-atom | worst final max-atom | mean final two-norm | mean E | batch ms (median of 3) |"
    );
    println!("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|");
    let starts = 20;
    let x0: Vec<_> = (0..starts).map(|s| lj_start(38, s as u64)).collect();
    run_case("LJ38", &arms, &tols, starts, false, |s, arm, tol| {
        let f = |x: ArrayView1<f64>| pairwise(x, None, f64::INFINITY, lj);
        solve(Oracle::unbounded(3 * 38, f), &x0[s], arm, tol)
    });
    let fixtures = ["morse_pos_0000.con", "morse_pos_0008.con"];
    let morse_x: Vec<_> = fixtures.iter().map(|f| read_con(f)).collect();
    run_case("Morse Pt256", &arms, &tols, 2, false, |s, arm, tol| {
        let (x0, cell) = &morse_x[s];
        let cell = *cell;
        let f = move |x: ArrayView1<f64>| pairwise(x, Some(cell), 7.0, morse);
        solve(Oracle::unbounded(x0.len(), f), x0, arm, tol)
    });

    let arm = |name, cap, accept, radius| Arm {
        name,
        method: lbfgs as fn() -> Method,
        istep: 0.2,
        stop: None,
        cap,
        accept,
        radius,
    };
    let trust = [
        arm(
            "lbfgs m=20, cap 0.2, geometry step",
            Some(0.2),
            Accept::None,
            0.0,
        ),
        arm(
            "lbfgs m=20, cap 0.2, trust step at radius 0.2 in place of the geometry step",
            Some(0.2),
            Accept::None,
            0.2,
        ),
        arm(
            "lbfgs m=20, no cap, raw two-loop, accept none",
            None,
            Accept::None,
            0.0,
        ),
        arm(
            "lbfgs m=20, no cap, strong Wolfe, accept energy",
            None,
            Accept::Energy,
            0.0,
        ),
        arm(
            "lbfgs m=20, no cap, trust step at radius 0.2",
            None,
            Accept::None,
            0.2,
        ),
    ];
    println!();
    println!(
        "| case | method | stop | tol | converged | budget hits | errors | mean calls | median | max | mean final max-atom | worst final max-atom | mean final two-norm | mean E | batch ms (median of 3) |"
    );
    println!("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    run_case("LJ38", &trust, &tols, starts, true, |s, arm, tol| {
        let f = |x: ArrayView1<f64>| pairwise(x, None, f64::INFINITY, lj);
        solve(Oracle::unbounded(3 * 38, f), &x0[s], arm, tol)
    });
    run_case("Morse Pt256", &trust, &tols, 2, true, |s, arm, tol| {
        let (x0, cell) = &morse_x[s];
        let cell = *cell;
        let f = move |x: ArrayView1<f64>| pairwise(x, Some(cell), 7.0, morse);
        solve(Oracle::unbounded(x0.len(), f), x0, arm, tol)
    });
    let rosen = Array1::from(vec![-1.2, 1.0]);
    run_case("Rosenbrock 2D", &trust, &[1e-6], 1, true, |_, arm, tol| {
        solve(Rosenbrock::<2>::new(), &rosen, arm, tol)
    });
}
