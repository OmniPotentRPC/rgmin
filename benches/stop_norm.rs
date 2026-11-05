//! Session stop points under the two-norm and the max-atom criterion.
//!
//! Geometry L-BFGS (memory 20, `istep = 0.2`), FIRE and FIRE 2.0
//! (`dt = 0.1`; FIRE 2.0 also on the Guénolé 2020 preset, with and
//! without ABC), all with the per-atom cap 0.2, on LJ38 from seeded
//! random starts and on the two Pt256 Morse fixtures. Each run stops
//! when the report clears the tolerance. Steps are `Solver::step`
//! calls. Every oracle call counts. The final forces are evaluated
//! again outside the session.
//! `cargo bench --bench stop_norm`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rgmin::{Control, FireKind, FirePreset, Method, Oracle, Solver, StopNorm};

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
    fire: Option<(FirePreset, bool)>,
}

struct Run {
    converged: bool,
    steps: usize,
    calls: usize,
    energy: f64,
    fmax: f64,
    gnorm: f64,
}

fn solve<F>(f: F, x0: &Array1<f64>, arm: Arm, tol: f64) -> Run
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync + Copy,
{
    let obj = Counted {
        inner: Oracle::unbounded(x0.len(), f),
        calls: AtomicUsize::new(0),
    };
    let control = Control {
        maxiter: BUDGET,
        gtol: tol,
        istep: arm.istep,
        maxmove: None,
        ..Default::default()
    };
    let mut solver = Solver::new((arm.method)(), control, x0.len());
    solver.set_atom_maxmove(0.2);
    if let Some(norm) = arm.stop {
        solver.set_stop_norm(norm);
    }
    if let Some((preset, abc)) = arm.fire {
        solver.set_fire_preset(preset).unwrap();
        solver.set_fire_abc(abc).unwrap();
    }
    let mut x = x0.clone();
    let mut converged = false;
    let mut steps = 0;
    for _ in 0..BUDGET {
        steps += 1;
        match solver.step(&obj, &mut x) {
            Ok(rep) if rep.grad_norm < tol => {
                converged = true;
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }
        if obj.calls.load(Ordering::Relaxed) > BUDGET {
            break;
        }
    }
    let calls = obj.calls.load(Ordering::Relaxed);
    let (energy, g) = f(x.view());
    Run {
        converged,
        steps,
        calls,
        energy,
        fmax: max_atom(&g),
        gnorm: g.dot(&g).sqrt(),
    }
}

fn run_case<F>(label: &str, arms: &[Arm], tols: &[f64], starts: usize, mut one: F)
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
            println!(
                "| {label} | {} | {} | {tol} | {}/{starts} | {:.1} | {:.1} | {} | {} | {:.6} | {:.6} | {:.6} | {:.6} | {:.2} |",
                arm.name,
                match arm.stop {
                    Some(norm) => format!("{norm:?}"),
                    None => "default".into(),
                },
                runs.iter().filter(|r| r.converged).count(),
                mean(|r| r.steps as f64),
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
    let fire2 = || Method::Fire { kind: FireKind::V2 };
    let guenole = Some((FirePreset::Guenole2020, false));
    let abc = Some((FirePreset::Guenole2020, true));
    let mut arms = Vec::new();
    for (name, method, istep, fire) in [
        ("lbfgs m=20", lbfgs as fn() -> Method, 0.2, None),
        ("fire", fire, 0.1, None),
        ("fire2", fire2, 0.1, None),
        ("fire2 guenole2020", fire2, 0.1, guenole),
        ("fire2 guenole2020 abc", fire2, 0.1, abc),
    ] {
        for stop in [None, Some(StopNorm::MaxAtom)] {
            arms.push(Arm {
                name,
                method,
                istep,
                stop,
                fire,
            });
        }
    }
    let tols = [0.05, 0.01];
    println!(
        "| case | method | stop | tol | converged | mean steps | mean calls | median | max | mean final max-atom | worst final max-atom | mean final two-norm | mean E | batch ms (median of 3) |"
    );
    println!("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    let starts = 20;
    let x0: Vec<_> = (0..starts).map(|s| lj_start(38, s as u64)).collect();
    run_case("LJ38", &arms, &tols, starts, |s, arm, tol| {
        let f = |x: ArrayView1<f64>| pairwise(x, None, f64::INFINITY, lj);
        solve(f, &x0[s], arm, tol)
    });
    let fixtures = ["morse_pos_0000.con", "morse_pos_0008.con"];
    let morse_x: Vec<_> = fixtures.iter().map(|f| read_con(f)).collect();
    run_case("Morse Pt256", &arms, &tols, 2, |s, arm, tol| {
        let (x0, cell) = &morse_x[s];
        let cell = *cell;
        let f = move |x: ArrayView1<f64>| pairwise(x, Some(cell), 7.0, morse);
        solve(f, x0, arm, tol)
    });
}
