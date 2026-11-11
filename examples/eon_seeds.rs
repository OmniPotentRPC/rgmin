// Copyright (c) rgmin developers

//! Capped L-BFGS force calls across seeds.
//!
//! Thirty starts on LJ13, LJ38, LJ75, a Morse slab and a Morse vacancy.
//! Seeds 0--19 are the training split. Seeds 20--29 are held out. The
//! product arms are the eOn cap and the opt-in length rules. The other
//! arms are the same client loop with one rule changed: no finite-
//! difference probe, Nocedal-Wright scaling of that probe, a
//! Barzilai-Borwein initial scale, a capped model step that keeps its
//! pairs, or an independent per-atom clip.
//!
//! `env -i PATH HOME LANG RUSTUP_HOME CARGO_HOME cargo run --release --example eon_seeds -- bench/raw/seeds.csv` writes bench/raw/seeds.csv.

use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, LbfgsStep, Method, Solver, StopNorm};

const MAX_MOVE: f64 = 0.2;
const TOL: f64 = 0.01;
const MEMORY: usize = 20;
const FD: f64 = 0.01;
const H0_INIT: f64 = 0.01;
const BUDGET: usize = 4_000;
const N_SEEDS: u64 = 30;
const HELD_FROM: u64 = 20;

fn main() {
    let mut args = std::env::args().skip(1);
    let out_path = args.next().unwrap_or_else(|| "bench/raw/seeds.csv".into());
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut lines = String::from("problem,seed,split,arm,evals,millis,energy,grad,converged\n");
    let mut summary = String::from("problem,split,arm,n,median,q25,q75,iqr,min,max,fail\n");
    for problem in problems() {
        eprintln!("{}", problem.name);
        let mut by_arm: Vec<(&str, Vec<(u64, Row)>)> = Vec::new();
        for arm in arms() {
            let mut rows = Vec::new();
            for seed in 0..N_SEEDS {
                let x0 = start_of(&problem, seed);
                let t0 = Instant::now();
                let row = run_arm(*arm, &problem, &x0);
                let mut row = row;
                row.millis = t0.elapsed().as_secs_f64() * 1e3;
                let split = if seed >= HELD_FROM { "held" } else { "train" };
                let line = format!(
                    "{},{seed},{split},{},{},{:.6},{:.8e},{:.8e},{}",
                    problem.name,
                    arm.name(),
                    row.evals,
                    row.millis,
                    row.energy,
                    row.grad,
                    if row.converged { "yes" } else { "no" }
                );
                lines.push_str(&line);
                lines.push('\n');
                rows.push((seed, row));
            }
            for split in ["train", "held"] {
                let vals: Vec<usize> = rows
                    .iter()
                    .filter(|(seed, row)| {
                        let held = *seed >= HELD_FROM;
                        ((split == "held") == held) && row.converged
                    })
                    .map(|(_, row)| row.evals)
                    .collect();
                let fail = rows
                    .iter()
                    .filter(|(seed, row)| {
                        let held = *seed >= HELD_FROM;
                        ((split == "held") == held) && !row.converged
                    })
                    .count();
                let (median, q25, q75, min, max) = if vals.is_empty() {
                    (0.0, 0.0, 0.0, 0, 0)
                } else {
                    let mut s = vals.clone();
                    s.sort_unstable();
                    (
                        quantile(&s, 0.5),
                        quantile(&s, 0.25),
                        quantile(&s, 0.75),
                        *s.first().unwrap(),
                        *s.last().unwrap(),
                    )
                };
                let iqr = q75 - q25;
                let rec = format!(
                    "{},{split},{},{},{median:.1},{q25:.1},{q75:.1},{iqr:.1},{min},{max},{fail}\n",
                    problem.name,
                    arm.name(),
                    vals.len()
                );
                summary.push_str(&rec);
                eprint!("{rec}");
            }
            by_arm.push((arm.name(), rows));
        }
        let _ = by_arm;
    }
    std::fs::write(&out_path, &lines).expect("csv");
    let sum_path = format!("{out_path}.summary");
    std::fs::write(&sum_path, &summary).expect("summary");
    eprintln!("wrote {out_path}");
}

#[derive(Clone, Copy)]
enum Arm {
    /// Product default: eOn cap, eight-way dots.
    Default,
    /// Shorten the two-loop step and keep the pairs.
    Scaled,
    /// Shorten the model step per atom and clear the pairs.
    Model,
    /// Drop a non-positive pair instead of clearing memory.
    Skip,
    /// Client loop. Finite-difference curvature, then `(s·y)/(y·y)`.
    Client,
    /// No probe. First `H0` is 0.01, then Nocedal-Wright 7.20.
    NoFd,
    /// Probe scale is `(s·y)/(y·y)`, not `1/C`.
    NwProbe,
    /// Later scales are `(s·s)/(s·y)`.
    Bb1,
    /// Later scales are the smaller of the two Barzilai-Borwein values.
    BbMin,
    /// A long downhill step is capped and the pairs are kept.
    Keep,
    /// Each atom of a downhill step is clipped on its own. Pairs are kept.
    AtomClip,
    /// A long step still falls back to `H0` times the force, but the pairs stay.
    Fallback,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Scaled => "scaled",
            Self::Model => "model",
            Self::Skip => "skip",
            Self::Client => "client",
            Self::NoFd => "no-fd",
            Self::NwProbe => "nw-probe",
            Self::Bb1 => "bb1",
            Self::BbMin => "bb-min",
            Self::Keep => "keep",
            Self::AtomClip => "atom-clip",
            Self::Fallback => "fallback",
        }
    }
}

fn arms() -> &'static [Arm] {
    &[
        Arm::Default,
        Arm::Scaled,
        Arm::Model,
        Arm::Skip,
        Arm::Client,
        Arm::NoFd,
        Arm::NwProbe,
        Arm::Bb1,
        Arm::BbMin,
        Arm::Keep,
        Arm::AtomClip,
        Arm::Fallback,
    ]
}

struct Row {
    evals: usize,
    millis: f64,
    energy: f64,
    grad: f64,
    converged: bool,
}

fn quantile(sorted: &[usize], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0] as f64;
    }
    let x = p * (sorted.len() - 1) as f64;
    let lo = x.floor() as usize;
    let hi = x.ceil() as usize;
    let w = x - lo as f64;
    sorted[lo] as f64 * (1.0 - w) + sorted[hi] as f64 * w
}

struct Problem {
    name: &'static str,
    kind: Kind,
    atoms: usize,
    cell: [f64; 3],
}

#[derive(Clone, Copy)]
enum Kind {
    Lj,
    MorseSlab,
    MorseVacancy,
}

fn problems() -> Vec<Problem> {
    let a = 2.897 * std::f64::consts::SQRT_2;
    vec![
        Problem {
            name: "lj13",
            kind: Kind::Lj,
            atoms: 13,
            cell: [0.0; 3],
        },
        Problem {
            name: "lj38",
            kind: Kind::Lj,
            atoms: 38,
            cell: [0.0; 3],
        },
        Problem {
            name: "lj75",
            kind: Kind::Lj,
            atoms: 75,
            cell: [0.0; 3],
        },
        Problem {
            name: "morse-slab",
            kind: Kind::MorseSlab,
            atoms: 0,
            cell: [6.0 * a, 6.0 * a, 0.0],
        },
        Problem {
            name: "morse-vacancy",
            kind: Kind::MorseVacancy,
            atoms: 0,
            cell: [3.0 * a, 3.0 * a, 3.0 * a],
        },
    ]
}

fn start_of(problem: &Problem, seed: u64) -> Vec<f64> {
    let a = 2.897 * std::f64::consts::SQRT_2;
    let ideal = match problem.kind {
        Kind::Lj => grid(problem.atoms, 1.1),
        Kind::MorseSlab => fcc_slab(6, 6, 4, a),
        Kind::MorseVacancy => fcc_vacancy(3, a),
    };
    let amp = match problem.kind {
        Kind::Lj => 0.05,
        Kind::MorseSlab | Kind::MorseVacancy => 0.04,
    };
    if seed == 0 {
        return ideal
            .iter()
            .enumerate()
            .map(|(i, v)| v + amp * ((3 * i) as f64).sin())
            .collect();
    }
    let mut state = 0x9E37_79B9_7F4A_7C15u64
        ^ seed.wrapping_mul(0x1000_0000_01B3)
        ^ problem.name.bytes().fold(0u64, |h, b| {
            h.wrapping_mul(0x1000_001B3).wrapping_add(u64::from(b))
        });
    ideal
        .iter()
        .map(|v| {
            let u = xorshift(&mut state);
            v + amp * (2.0 * u - 1.0)
        })
        .collect()
}

fn xorshift(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 11) as f64 / ((1u64 << 53) as f64)
}

fn grid(atoms: usize, spacing: f64) -> Vec<f64> {
    let side = (atoms as f64).cbrt().ceil() as usize;
    let mut x = Vec::with_capacity(3 * atoms);
    for k in 0..atoms {
        let cell = [k % side, (k / side) % side, k / (side * side)];
        for c in cell {
            x.push(spacing * c as f64);
        }
    }
    x
}

fn fcc_slab(nx: usize, ny: usize, nz: usize, a: f64) -> Vec<f64> {
    let mut x = Vec::new();
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                for b in [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [
                    0.0, 0.5, 0.5,
                ]] {
                    x.push((ix as f64 + b[0]) * a);
                    x.push((iy as f64 + b[1]) * a);
                    x.push((iz as f64 + b[2]) * a);
                }
            }
        }
    }
    x
}

fn fcc_vacancy(n: usize, a: f64) -> Vec<f64> {
    let mut x = fcc_slab(n, n, n, a);
    let center = 0.5 * n as f64 * a;
    let mut best = 0usize;
    let mut best_d = f64::MAX;
    let atoms = x.len() / 3;
    for i in 0..atoms {
        let mut d2 = 0.0;
        for a in 0..3 {
            let q = x[3 * i + a] - center;
            d2 += q * q;
        }
        if d2 < best_d {
            best_d = d2;
            best = i;
        }
    }
    x.drain(3 * best..3 * best + 3);
    x
}

fn pot(kind: Kind, cell: [f64; 3], x: &[f64], g: &mut [f64]) -> f64 {
    match kind {
        Kind::Lj => lennard_jones(x, g),
        Kind::MorseSlab | Kind::MorseVacancy => morse(cell, x, g),
    }
}

fn lennard_jones(x: &[f64], g: &mut [f64]) -> f64 {
    let n = x.len() / 3;
    let mut e = 0.0;
    g.fill(0.0);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            let mut r2 = 0.0;
            for a in 0..3 {
                d[a] = x[3 * i + a] - x[3 * j + a];
                r2 += d[a] * d[a];
            }
            let s6 = r2.powf(-3.0);
            e += 4.0 * (s6 * s6 - s6);
            let du = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
            for a in 0..3 {
                g[3 * i + a] += du * d[a];
                g[3 * j + a] -= du * d[a];
            }
        }
    }
    e
}

fn morse(cell: [f64; 3], x: &[f64], g: &mut [f64]) -> f64 {
    let (d0, alpha, r0): (f64, f64, f64) = (0.7102, 1.6047, 2.897);
    let n = x.len() / 3;
    let mut e = 0.0;
    g.fill(0.0);
    let xc = (-alpha * (7.0 - r0)).exp();
    let shift = d0 * (xc * xc - 2.0 * xc);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            for a in 0..3 {
                d[a] = wrap(x[3 * i + a] - x[3 * j + a], cell[a]);
            }
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if !(1.0e-8..7.0).contains(&r) {
                continue;
            }
            let ex = (-alpha * (r - r0)).exp();
            e += d0 * (ex * ex - 2.0 * ex) - shift;
            let du = 2.0 * d0 * alpha * ex * (1.0 - ex);
            for a in 0..3 {
                let c = du * d[a] / r;
                g[3 * i + a] += c;
                g[3 * j + a] -= c;
            }
        }
    }
    e
}

fn wrap(d: f64, cell: f64) -> f64 {
    if cell > 0.0 {
        d - cell * (d / cell).round()
    } else {
        d
    }
}

fn run_arm(arm: Arm, problem: &Problem, x0: &[f64]) -> Row {
    match arm {
        Arm::Default | Arm::Scaled | Arm::Model | Arm::Skip => run_solver(arm, problem, x0),
        _ => run_engine(arm, problem, x0),
    }
}

fn run_solver(arm: Arm, problem: &Problem, x0: &[f64]) -> Row {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls_in = calls.clone();
    let control = Control {
        maxiter: BUDGET,
        gtol: TOL,
        istep: 1.0,
        stop_norm: StopNorm::MaxAtom,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: MEMORY }, control, x0.len());
    match arm {
        Arm::Default => solver
            .set_lbfgs_step(LbfgsStep::Capped { max_atom: MAX_MOVE })
            .unwrap(),
        Arm::Scaled => solver
            .set_lbfgs_step(LbfgsStep::Scaled { max_atom: MAX_MOVE })
            .unwrap(),
        Arm::Model => {
            solver
                .set_lbfgs_step(LbfgsStep::Capped { max_atom: MAX_MOVE })
                .unwrap();
            solver.set_lbfgs_model_atom(true).unwrap();
        }
        Arm::Skip => {
            solver
                .set_lbfgs_step(LbfgsStep::Capped { max_atom: MAX_MOVE })
                .unwrap();
            solver.set_lbfgs_skip_bad(true).unwrap();
        }
        _ => unreachable!(),
    }
    let kind = problem.kind;
    let cell = problem.cell;
    let mut x = Array1::from_vec(x0.to_vec());
    // `run_fill` is the capped step only. Scaled goes through `run`.
    let converged = if matches!(arm, Arm::Scaled) {
        let obj = rgmin::Oracle::unbounded(x0.len(), move |xv: ArrayView1<f64>| {
            calls_in.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut g = Array1::zeros(xv.len());
            let e = pot(
                kind,
                cell,
                xv.as_slice().unwrap(),
                g.as_slice_mut().unwrap(),
            );
            (e, g)
        });
        match solver.run(&obj, &mut x) {
            Ok(rep) => rep.grad_norm < TOL,
            Err(_) => false,
        }
    } else {
        match solver.run_fill(&mut x, |xv: ArrayView1<f64>, gv: &mut Array1<f64>| {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            pot(
                kind,
                cell,
                xv.as_slice().unwrap(),
                gv.as_slice_mut().unwrap(),
            )
        }) {
            Ok(rep) => rep.grad_norm < TOL,
            Err(_) => false,
        }
    };
    let calls = calls.load(std::sync::atomic::Ordering::Relaxed);
    let mut g = vec![0.0; x.len()];
    let energy = pot(kind, cell, x.as_slice().unwrap(), &mut g);
    Row {
        evals: calls,
        millis: 0.0,
        energy,
        grad: max_atom(&g),
        converged,
    }
}

#[derive(Clone, Copy)]
enum Scale {
    /// `(s·y)/(y·y)`, Nocedal-Wright 7.20.
    Nw,
    /// `(s·s)/(s·y)`.
    Bb1,
    /// The smaller positive Barzilai-Borwein value.
    Min,
}

#[derive(Clone, Copy)]
enum Long {
    /// Clear memory and step along `H0` times the force.
    Reset,
    /// Shorten the model step and keep the pairs.
    Keep,
    /// Clip each atom and keep the pairs.
    Atom,
    /// Step along `H0` times the force, and keep the pairs.
    Fallback,
}

struct Engine {
    scale: Scale,
    probe: bool,
    /// Probe `H0` is the Nocedal-Wright quotient, not `1/C`.
    probe_nw: bool,
    long: Long,
}

fn engine_of(arm: Arm) -> Engine {
    match arm {
        Arm::Client => Engine {
            scale: Scale::Nw,
            probe: true,
            probe_nw: false,
            long: Long::Reset,
        },
        Arm::NoFd => Engine {
            scale: Scale::Nw,
            probe: false,
            probe_nw: false,
            long: Long::Reset,
        },
        Arm::NwProbe => Engine {
            scale: Scale::Nw,
            probe: true,
            probe_nw: true,
            long: Long::Reset,
        },
        Arm::Bb1 => Engine {
            scale: Scale::Bb1,
            probe: true,
            probe_nw: false,
            long: Long::Reset,
        },
        Arm::BbMin => Engine {
            scale: Scale::Min,
            probe: true,
            probe_nw: false,
            long: Long::Reset,
        },
        Arm::Keep => Engine {
            scale: Scale::Nw,
            probe: true,
            probe_nw: false,
            long: Long::Keep,
        },
        Arm::AtomClip => Engine {
            scale: Scale::Nw,
            probe: true,
            probe_nw: false,
            long: Long::Atom,
        },
        Arm::Fallback => Engine {
            scale: Scale::Nw,
            probe: true,
            probe_nw: false,
            long: Long::Fallback,
        },
        _ => unreachable!(),
    }
}

struct Mem {
    s: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    rho: Vec<f64>,
    r_prev: Vec<f64>,
    f_prev: Vec<f64>,
}

fn run_engine(arm: Arm, problem: &Problem, x0: &[f64]) -> Row {
    let eng = engine_of(arm);
    let kind = problem.kind;
    let cell = problem.cell;
    let mut x = x0.to_vec();
    let mut g = vec![0.0; x.len()];
    let mut calls = 1usize;
    let _ = pot(kind, cell, &x, &mut g);
    let mut mem = Mem {
        s: Vec::new(),
        y: Vec::new(),
        rho: Vec::new(),
        r_prev: Vec::new(),
        f_prev: Vec::new(),
    };
    let mut converged = false;
    for iteration in 0..BUDGET {
        if max_atom(&g) < TOL {
            converged = true;
            break;
        }
        let force: Vec<f64> = g.iter().map(|v| -v).collect();
        if iteration > 0 {
            store_pair(&x, &force, &mut mem);
        }
        let dr = engine_step(
            iteration, &eng, kind, cell, &x, &force, &mut mem, &mut calls,
        );
        mem.r_prev = x.clone();
        mem.f_prev = force;
        for i in 0..x.len() {
            x[i] += dr[i];
        }
        let _ = pot(kind, cell, &x, &mut g);
        calls += 1;
    }
    if max_atom(&g) < TOL {
        converged = true;
    }
    let mut gf = vec![0.0; x.len()];
    let energy = pot(kind, cell, &x, &mut gf);
    Row {
        evals: calls,
        millis: 0.0,
        energy,
        grad: max_atom(&gf),
        converged,
    }
}

fn store_pair(x: &[f64], force: &[f64], mem: &mut Mem) {
    let n = x.len();
    let mut s0 = vec![0.0; n];
    let mut y0 = vec![0.0; n];
    for i in 0..n {
        s0[i] = x[i] - mem.r_prev[i];
        y0[i] = mem.f_prev[i] - force[i];
    }
    let sy = dot(&s0, &y0);
    if !sy.is_finite() || sy.abs() < 1e-30 {
        mem.s.clear();
        mem.y.clear();
        mem.rho.clear();
        return;
    }
    mem.rho.push(1.0 / sy);
    mem.s.push(s0);
    mem.y.push(y0);
    while mem.s.len() > MEMORY {
        mem.s.remove(0);
        mem.y.remove(0);
        mem.rho.remove(0);
    }
}

fn engine_step(
    iteration: usize,
    eng: &Engine,
    kind: Kind,
    cell: [f64; 3],
    x: &[f64],
    force: &[f64],
    mem: &mut Mem,
    calls: &mut usize,
) -> Vec<f64> {
    let mut h0 = H0_INIT;
    if iteration > 0 {
        let n = x.len();
        let mut dr = vec![0.0; n];
        let mut yv = vec![0.0; n];
        for i in 0..n {
            dr[i] = x[i] - mem.r_prev[i];
            yv[i] = mem.f_prev[i] - force[i];
        }
        match scale_of(eng.scale, &dr, &yv) {
            ScaleOut::Steep => {
                mem.s.clear();
                mem.y.clear();
                mem.rho.clear();
                return cap_uniform(&scale(force, 1000.0));
            }
            ScaleOut::Value(v) => h0 = v,
        }
    } else if eng.probe {
        let fnorm = dot(force, force).sqrt();
        if fnorm > 0.0 {
            let u = scale(force, 1.0 / fnorm);
            let probe: Vec<f64> = x.iter().zip(&u).map(|(p, ui)| p + FD * ui).collect();
            let mut gp = vec![0.0; x.len()];
            let _ = pot(kind, cell, &probe, &mut gp);
            *calls += 1;
            let mut y = vec![0.0; x.len()];
            let mut s = vec![0.0; x.len()];
            for i in 0..x.len() {
                s[i] = FD * u[i];
                y[i] = gp[i] - (force[i] * -1.0);
                // g_probe - g = g_probe + force, since force = -g.
                y[i] = gp[i] + force[i];
            }
            if eng.probe_nw {
                match scale_of(Scale::Nw, &s, &y) {
                    ScaleOut::Value(v) => h0 = v,
                    ScaleOut::Steep => {
                        mem.s.clear();
                        mem.y.clear();
                        mem.rho.clear();
                        return cap_uniform(&scale(force, 1000.0));
                    }
                }
            } else {
                let mut c = 0.0;
                for i in 0..x.len() {
                    c += y[i] * u[i];
                }
                c /= FD;
                if !(c > 0.0) || !c.is_finite() || c.abs() < 1e-300 {
                    mem.s.clear();
                    mem.y.clear();
                    mem.rho.clear();
                    return cap_uniform(&scale(force, 1000.0));
                }
                h0 = 1.0 / c;
            }
        }
    }
    let n = mem.s.len();
    let mut q: Vec<f64> = force.iter().map(|f| -f).collect();
    let mut alpha = vec![0.0; n];
    for k in (0..n).rev() {
        alpha[k] = mem.rho[k] * dot(&mem.s[k], &q);
        let a = alpha[k];
        for i in 0..q.len() {
            q[i] -= a * mem.y[k][i];
        }
    }
    for v in &mut q {
        *v *= h0;
    }
    for k in 0..n {
        let b = mem.rho[k] * dot(&mem.y[k], &q);
        let coef = alpha[k] - b;
        for i in 0..q.len() {
            q[i] += coef * mem.s[k][i];
        }
    }
    let d: Vec<f64> = q.iter().map(|v| -v).collect();
    let uphill = {
        let dn = dot(&d, &d).sqrt();
        let fnorm = dot(force, force).sqrt();
        dn > 0.0 && fnorm > 0.0 && dot(&d, force) < 0.0
    };
    let motion = max_atom(&d);
    if uphill {
        mem.s.clear();
        mem.y.clear();
        mem.rho.clear();
        return cap_uniform(&scale(force, h0));
    }
    if motion >= MAX_MOVE {
        match eng.long {
            Long::Reset => {
                mem.s.clear();
                mem.y.clear();
                mem.rho.clear();
                cap_uniform(&scale(force, h0))
            }
            Long::Keep => cap_uniform(&d),
            Long::Atom => clip_atoms(d),
            Long::Fallback => cap_uniform(&scale(force, h0)),
        }
    } else {
        d
    }
}

enum ScaleOut {
    Value(f64),
    Steep,
}

fn scale_of(mode: Scale, s: &[f64], y: &[f64]) -> ScaleOut {
    let sy = dot(s, y);
    let yy = dot(y, y);
    let ss = dot(s, s);
    if sy.abs() < 1e-300 || !sy.is_finite() || !yy.is_finite() || sy < 0.0 {
        // Negative curvature: the client resets. A non-positive `s·y`
        // has no usable Barzilai-Borwein step either.
        if sy < 0.0 || sy.abs() < 1e-300 {
            return ScaleOut::Steep;
        }
    }
    let nw = if yy > 0.0 { sy / yy } else { -1.0 };
    let bb = if sy > 0.0 { ss / sy } else { -1.0 };
    let h0 = match mode {
        Scale::Nw => nw,
        Scale::Bb1 => bb,
        Scale::Min => {
            if nw > 0.0 && bb > 0.0 {
                nw.min(bb)
            } else if nw > 0.0 {
                nw
            } else {
                bb
            }
        }
    };
    if h0.is_finite() && h0 > 0.0 {
        ScaleOut::Value(h0)
    } else {
        ScaleOut::Steep
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

fn max_atom(v: &[f64]) -> f64 {
    let mut m = 0.0;
    let mut i = 0;
    while i + 3 <= v.len() {
        let r = (v[i] * v[i] + v[i + 1] * v[i + 1] + v[i + 2] * v[i + 2]).sqrt();
        if r > m {
            m = r;
        }
        i += 3;
    }
    if i < v.len() {
        let mut r2 = 0.0;
        for j in i..v.len() {
            r2 += v[j] * v[j];
        }
        m = m.max(r2.sqrt());
    }
    m
}

fn cap_uniform(v: &[f64]) -> Vec<f64> {
    let m = max_atom(v);
    if m > MAX_MOVE && m > 0.0 {
        scale(v, MAX_MOVE / m)
    } else {
        v.to_vec()
    }
}

fn clip_atoms(mut v: Vec<f64>) -> Vec<f64> {
    let mut i = 0;
    while i + 3 <= v.len() {
        let r = (v[i] * v[i] + v[i + 1] * v[i + 1] + v[i + 2] * v[i + 2]).sqrt();
        if r > MAX_MOVE && r > 0.0 {
            let s = MAX_MOVE / r;
            v[i] *= s;
            v[i + 1] *= s;
            v[i + 2] *= s;
        }
        i += 3;
    }
    v
}
