//! CPU strong scaling: one fixed system, thread counts 1, 2, 4, ... up
//! to the physical cores.
//!
//! The system is a perturbed periodic FCC crystal of `4 L^3` atoms with
//! one Morse bond (Pt parameters of `relax_morse`) to each of the 12
//! nearest neighbours of the ideal lattice. The lattice constant puts
//! every bond at the Morse minimum, so the ideal lattice is the minimum
//! at `E = -6 N D0`. The bond list is fixed, which keeps the potential
//! analytic, and is valid while displacements stay far below the
//! nearest-neighbour distance.
//!
//! Two methods run on each rayon pool. L-BFGS (m = 10) takes the
//! two-loop direction and the strong Wolfe search (c1 = 1e-4, c2 = 0.9,
//! at most 40 trials) under `Accept::Energy`, with no per-atom cap.
//! FIRE 2.0 with the Guenole 2020 preset has no line search; a 0.2
//! Angstrom per-atom cap scales each step. The force evaluation and the
//! `par` vector seam both run on the pool. Every run starts from the same
//! geometry and stops at a max per-atom force below 0.01 eV/Angstrom.
//!
//! CSV goes to stdout. `wall_s` is the median over the repeats,
//! `max_dev` the largest coordinate difference from the first 1-thread
//! run over all repeats, `efficiency` the speedup over 1 thread divided
//! by the thread count. The harness is CPU only; the `gpu` column stays
//! empty. After the timed L-BFGS repeats on each pool, one untimed solve
//! counts the steps that ran the line search, seen as a second oracle
//! call at the step's start, and the accepted steps that meet the strong
//! Wolfe conditions, and reports both on stderr. It exits non-zero when a
//! run does not converge or an L-BFGS step fails either count.
//!
//! `cargo run --release --features par --example strong_scale -- \
//!   --cells 20 --repeats 5 [--max-threads T] [--hardware NAME]`

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use rgmin::{
    Accept, Control, FireKind, FirePreset, LineSearch, Method, Oracle, Solver, StopNorm,
    phase_enable, phase_reset, phase_times, take_gradient,
};

/// Pt Morse parameters of tests/newton_cg_morse.rs: eV, 1/Angstrom, Angstrom.
const D0: f64 = 0.7102;
const ALPHA: f64 = 1.6047;
const R0: f64 = 2.897;

/// Uniform per-coordinate displacement bound, Angstrom: 3.5 % of `R0`.
const AMP: f64 = 0.1;
const SEED: u64 = 0;

const FMAX: f64 = 0.01;
const BUDGET: usize = 2000;

/// Strong Wolfe constants of the L-BFGS search, the `Solver::new` default.
const C1: f64 = 1e-4;
const C2: f64 = 0.9;

/// Half-lattice offsets of the 12 FCC nearest neighbours.
const NN: [[i64; 3]; 12] = [
    [1, 1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, 1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, 1, -1],
    [0, -1, 1],
    [0, -1, -1],
];

struct System {
    x0: Array1<f64>,
    nbr: Vec<[usize; 12]>,
    box_len: f64,
}

/// Perturbed FCC positions and the nearest-neighbour list of the ideal
/// lattice. Atoms sit on the half-lattice points whose coordinates sum
/// to an even number.
fn fcc(cells: usize) -> System {
    let a = R0 * std::f64::consts::SQRT_2;
    let side = 2 * cells as i64;
    let mut index = vec![usize::MAX; (side * side * side) as usize];
    let mut sites = Vec::new();
    for h in 0..side * side * side {
        let p = [h / (side * side), (h / side) % side, h % side];
        if (p[0] + p[1] + p[2]) % 2 == 0 {
            index[h as usize] = sites.len();
            sites.push(p);
        }
    }
    let flat = |p: [i64; 3]| -> usize {
        let w = |v: i64| v.rem_euclid(side);
        index[(w(p[0]) * side * side + w(p[1]) * side + w(p[2])) as usize]
    };
    let nbr = sites
        .iter()
        .map(|p| NN.map(|d| flat([p[0] + d[0], p[1] + d[1], p[2] + d[2]])))
        .collect();
    let mut rng = StdRng::seed_from_u64(SEED);
    let x0 = sites
        .iter()
        .flat_map(|p| p.map(|v| v as f64 * a / 2.0))
        .map(|v| v + rng.random_range(-AMP..AMP))
        .collect();
    System {
        x0,
        nbr,
        box_len: cells as f64 * a,
    }
}

static PHASES: AtomicBool = AtomicBool::new(false);
static OBJ_NS: AtomicU64 = AtomicU64::new(0);
static ALLOC_NS: AtomicU64 = AtomicU64::new(0);
static FORCE_NS: AtomicU64 = AtomicU64::new(0);
static SUM_NS: AtomicU64 = AtomicU64::new(0);

fn phase_add(slot: &AtomicU64, t0: Instant) {
    if PHASES.load(Ordering::Relaxed) {
        slot.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Morse energy and gradient over the fixed bonds, one atom per task.
fn morse(sys: &System, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let x = x.as_slice().expect("contiguous");
    let phases = PHASES.load(Ordering::Relaxed);
    let t_alloc = phases.then(Instant::now);
    let mut g = take_gradient(x.len());
    g.fill(0.0);
    if let Some(t0) = t_alloc {
        phase_add(&ALLOC_NS, t0);
    }
    let t_force = phases.then(Instant::now);
    let e: f64 = g
        .as_slice_mut()
        .expect("contiguous")
        .par_chunks_mut(3)
        .enumerate()
        .map(|(i, gi)| {
            let mut ei = 0.0;
            for &j in &sys.nbr[i] {
                let mut d = [0.0; 3];
                for k in 0..3 {
                    let dx = x[3 * i + k] - x[3 * j + k];
                    d[k] = dx - sys.box_len * (dx / sys.box_len).round();
                }
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let ex = (-ALPHA * (r - R0)).exp();
                ei += 0.5 * D0 * (ex * ex - 2.0 * ex);
                let du = 2.0 * D0 * ALPHA * ex * (1.0 - ex);
                for k in 0..3 {
                    gi[k] += du * d[k] / r;
                }
            }
            ei
        })
        .sum();
    if let Some(t0) = t_force {
        phase_add(&FORCE_NS, t0);
    }
    (e, g)
}

struct Run {
    wall: f64,
    calls: usize,
    steps: usize,
    fmax: f64,
    x: Array1<f64>,
    /// Steps that ran the line search; counted only on an audited run.
    searched: usize,
    /// Accepted steps that meet the strong Wolfe conditions; counted only
    /// on an audited run.
    wolfe: usize,
}

/// Strong Wolfe conditions (Nocedal-Wright 3.7) for the accepted step `s`
/// from `(f0, g0)` to `(f1, g1)`.
fn strong_wolfe(
    s: &Array1<f64>,
    (f0, g0): (f64, &Array1<f64>),
    (f1, g1): (f64, &Array1<f64>),
) -> bool {
    let (d0, d1) = (g0.dot(s), g1.dot(s));
    d0 < 0.0 && f1 <= f0 + C1 * d0 && d1.abs() <= -C2 * d0
}

fn relax(sys: &System, name: &str, audit: bool) -> Run {
    let n = sys.x0.len();
    let (method, istep) = match name {
        "lbfgs" => (Method::Lbfgs { memory: 10 }, 0.2),
        _ => (Method::Fire { kind: FireKind::V2 }, 0.1),
    };
    let control = Control {
        maxiter: BUDGET,
        gtol: FMAX,
        istep,
        maxmove: None,
        stop_norm: StopNorm::MaxAtom,
    };
    if PHASES.load(Ordering::Relaxed) {
        for slot in [&OBJ_NS, &ALLOC_NS, &FORCE_NS, &SUM_NS] {
            slot.store(0, Ordering::Relaxed);
        }
        phase_reset();
    }
    let calls = AtomicUsize::new(0);
    let origin = Mutex::new(Array1::zeros(0));
    let at_origin = AtomicUsize::new(0);
    let obj = Oracle::unbounded(n, |x: ArrayView1<f64>| {
        calls.fetch_add(1, Ordering::Relaxed);
        if audit && *origin.lock().expect("origin") == x {
            at_origin.fetch_add(1, Ordering::Relaxed);
        }
        let t0 = PHASES.load(Ordering::Relaxed).then(Instant::now);
        let ev = morse(sys, x);
        if let Some(t0) = t0 {
            phase_add(&OBJ_NS, t0);
        }
        ev
    });
    let start = Instant::now();
    let mut solver = Solver::new(method, control, n);
    solver.set_stop_norm(StopNorm::MaxAtom);
    match name {
        // Opt into the strong Wolfe search. The crate default has none.
        "lbfgs" => {
            solver
                .set_lbfgs_step(rgmin::LbfgsStep::Wolfe)
                .expect("wolfe");
            solver.set_accept(Accept::Energy);
            solver.set_linesearch(LineSearch::Wolfe {
                c1: C1,
                c2: C2,
                maxiter: 40,
            });
        }
        _ => {
            solver.set_atom_maxmove(0.2);
            solver
                .set_fire_preset(FirePreset::Guenole2020)
                .expect("fire2 preset");
        }
    }
    let mut x = sys.x0.clone();
    let mut last = audit.then(|| morse(sys, x.view()));
    let (mut steps, mut fmax) = (0, f64::INFINITY);
    let (mut searched, mut wolfe) = (0, 0);
    while steps < BUDGET && fmax >= FMAX {
        if audit {
            *origin.lock().expect("origin") = x.clone();
            at_origin.store(0, Ordering::Relaxed);
        }
        let report = solver.step(&obj, &mut x).expect("step");
        fmax = report.grad_norm;
        if let Some((f0, g0)) = last.take() {
            // The session evaluates the start of its first step itself.
            // LineSearch::search takes no start value, so it evaluates the
            // start of every step again; the capped geometry step and the
            // Accept::None trial reuse the value the session holds.
            let own = usize::from(steps == 0);
            searched += usize::from(at_origin.load(Ordering::Relaxed) > own);
            let x0 = std::mem::take(&mut *origin.lock().expect("origin"));
            let g1 = report.grad.clone().expect("session gradient");
            wolfe += usize::from(strong_wolfe(&(&x - &x0), (f0, &g0), (report.value, &g1)));
            last = Some((report.value, g1));
        }
        steps += 1;
    }
    let wall = start.elapsed().as_secs_f64();
    if PHASES.load(Ordering::Relaxed) {
        let p = phase_times();
        eprintln!(
            "phases method={name} wall_ns={} obj_ns={} alloc_ns={} force_ns={} sum_ns={} two_loop_ns={} line_search_ns={} dot_ns={} axpy_ns={} rayon_ns={} dot_calls={} axpy_calls={} rayon_calls={} copy_ns={} elem_ns={}",
            (wall * 1.0e9) as u64,
            OBJ_NS.load(Ordering::Relaxed),
            ALLOC_NS.load(Ordering::Relaxed),
            FORCE_NS.load(Ordering::Relaxed),
            SUM_NS.load(Ordering::Relaxed),
            p.two_loop_ns,
            p.line_search_ns,
            p.dot_ns,
            p.axpy_ns,
            p.rayon_ns,
            p.dot_calls,
            p.axpy_calls,
            p.rayon_calls,
            p.copy_ns,
            p.elem_ns,
        );
    }
    Run {
        wall,
        calls: calls.load(Ordering::Relaxed),
        steps,
        fmax,
        x,
        searched,
        wolfe,
    }
}

/// CPU model name and physical core count from /proc/cpuinfo, falling
/// back to the logical count.
fn host() -> (String, usize) {
    let info = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let field = |l: &str| l.split_once(':').map(|(_, v)| v.trim().to_string());
    let model = info
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(field)
        .unwrap_or_else(|| "unknown".into());
    let mut cores = std::collections::BTreeSet::new();
    let mut socket = String::new();
    for l in info.lines() {
        if l.starts_with("physical id") {
            socket = field(l).unwrap_or_default();
        } else if l.starts_with("core id") {
            cores.insert((socket.clone(), field(l).unwrap_or_default()));
        }
    }
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    let physical = if cores.is_empty() {
        logical
    } else {
        cores.len()
    };
    (model, physical)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |key: &str| {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let num = |key: &str, default: usize| arg(key).map_or(default, |v| v.parse().expect(key));
    if args.iter().any(|a| a == "--phases") {
        PHASES.store(true, Ordering::Relaxed);
        phase_enable();
    }
    let (model, physical) = host();
    let hardware = arg("--hardware").unwrap_or(model);
    let max_threads = num("--max-threads", physical).max(1);
    let repeats = num("--repeats", 5).max(1);
    let sys = fcc(num("--cells", 20).max(2));
    let mut threads: Vec<usize> = (0..)
        .map(|k| 1 << k)
        .take_while(|&t| t < max_threads)
        .collect();
    threads.push(max_threads);

    println!(
        "hardware,cores,gpu,method,threads,n,repeats,wall_s,force_calls,iterations,grad_norm,max_dev,speedup,efficiency"
    );
    let (mut ok, mut wolfe) = (true, true);
    for name in ["lbfgs", "fire2"] {
        let mut base: Option<(f64, Array1<f64>)> = None;
        for &t in &threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .expect("thread pool");
            let runs: Vec<Run> = (0..repeats)
                .map(|_| pool.install(|| relax(&sys, name, false)))
                .collect();
            let mut walls: Vec<f64> = runs.iter().map(|r| r.wall).collect();
            walls.sort_by(f64::total_cmp);
            let m = walls.len();
            let wall = (walls[(m - 1) / 2] + walls[m / 2]) / 2.0;
            let (t1, x1) = base.get_or_insert_with(|| (wall, runs[0].x.clone()));
            let dev = runs
                .iter()
                .flat_map(|r| (&r.x - &*x1).mapv(f64::abs))
                .fold(0.0, f64::max);
            let r = &runs[0];
            ok &= runs.iter().all(|r| r.fmax < FMAX);
            let speedup = *t1 / wall;
            println!(
                "\"{hardware}\",{physical},,{name},{t},{},{repeats},{wall:.6},{},{},{:.6e},{dev:.3e},{speedup:.3},{:.3}",
                sys.x0.len(),
                r.calls,
                r.steps,
                r.fmax,
                speedup / t as f64,
            );
            if name == "lbfgs" {
                let audit = pool.install(|| relax(&sys, name, true));
                ok &= audit.fmax < FMAX;
                wolfe &= audit.searched == audit.steps && audit.wolfe == audit.steps;
                eprintln!(
                    "strong_scale: lbfgs on {t} threads: {} steps, {} searched, {} meet strong Wolfe (c1 = {C1:e}, c2 = {C2}), {} force calls",
                    audit.steps, audit.searched, audit.wolfe, audit.calls,
                );
            }
        }
    }
    if !ok {
        eprintln!("strong_scale: a run did not reach fmax < {FMAX} eV/Angstrom");
    }
    if !wolfe {
        eprintln!(
            "strong_scale: an L-BFGS step ran no line search or missed the strong Wolfe conditions"
        );
    }
    if !(ok && wolfe) {
        std::process::exit(1);
    }
}
