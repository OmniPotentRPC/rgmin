//! One nonlinear CG solve, timed, for the hardening-cost comparison.
//!
//! Lennard-Jones pairs come from linkcell (`lc_pairs_open`). The pair
//! energy and `(dV/dr) / r` come from rgpot (`rgpot_lj_curvature_pairs`).
//! The coordinate accumulation is the same one rgpot's own cluster
//! gradient uses. Rosenbrock never touches those libraries.

use std::env;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rgmin::{minimize_method, Control, LineSearch, Method, Oracle};

const WOLFE_C1: f64 = 1e-4;
const WOLFE_C2: f64 = 0.1;
const WOLFE_ITERS: usize = 20;
const ISTEP: f64 = 1.0;
/// Open-cluster cutoff. Compact LJ starts sit far inside it, so the list
/// is the full pair set of the unshifted potential.
const LJ_CUTOFF: f64 = 1.0e4;
const LJ_EPSILON: f64 = 1.0;
const LJ_SIGMA: f64 = 1.0;

extern "C" {
    fn rgpot_lj_curvature_pairs(
        epsilon: f64,
        sigma: f64,
        npairs: i32,
        r2: *const f64,
        energy: *mut f64,
        dvdr_over_r: *mut f64,
        d2vdr2: *mut f64,
    ) -> i32;

    fn rgpot_lj_gradient(
        epsilon: f64,
        sigma: f64,
        cutoff: f64,
        natoms: i32,
        positions: *const f64,
        energy: *mut f64,
        gradient: *mut f64,
    ) -> i32;
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sphere packing. Radius matches the atomistic LJ38 packing at N = 38
/// and scales with N to the one third. Minimum separation is 0.9.
fn pack(n: usize, seed: u64) -> Vec<f64> {
    let mut radius = 2.4 * (n as f64 / 38.0).cbrt();
    let mut rng = Rng(seed.wrapping_mul(7919).wrapping_add(17));
    loop {
        let mut pos = Vec::with_capacity(3 * n);
        let mut rejects = 0u32;
        while pos.len() < 3 * n {
            let p = [
                (2.0 * rng.uniform() - 1.0) * radius,
                (2.0 * rng.uniform() - 1.0) * radius,
                (2.0 * rng.uniform() - 1.0) * radius,
            ];
            let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
            if r2 > radius * radius {
                continue;
            }
            let close = pos.chunks(3).any(|q| {
                let dx = q[0] - p[0];
                let dy = q[1] - p[1];
                let dz = q[2] - p[2];
                dx * dx + dy * dy + dz * dz < 0.81
            });
            if close {
                rejects += 1;
                if rejects > 100_000 {
                    radius *= 1.05;
                    pos.clear();
                    rejects = 0;
                }
                continue;
            }
            rejects = 0;
            pos.extend_from_slice(&p);
        }
        return pos;
    }
}

fn rosenbrock_start(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect()
}

fn rosenbrock(x: &[f64]) -> (f64, Vec<f64>) {
    let mut f = 0.0;
    let mut g = vec![0.0; x.len()];
    let mut i = 0;
    while i + 1 < x.len() {
        let a = x[i];
        let b = x[i + 1];
        let t1 = b - a * a;
        let t2 = 1.0 - a;
        f += 100.0 * t1 * t1 + t2 * t2;
        g[i] += -400.0 * a * t1 - 2.0 * t2;
        g[i + 1] += 200.0 * t1;
        i += 2;
    }
    (f, g)
}

fn lj_pairs(x: &[f64]) -> Vec<linkcell::Pair> {
    let n = x.len() / 3;
    let pts: Vec<[f64; 3]> = (0..n)
        .map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
        .collect();
    linkcell::pairs_open(&pts, LJ_CUTOFF, None).expect("linkcell pairs")
}

/// Energy and dE/dx. Pair terms are rgpot's. The neighbour list is linkcell's.
fn lennard_jones(x: &[f64]) -> (f64, Vec<f64>) {
    let pairs = lj_pairs(x);
    let m = pairs.len();
    let mut energy = 0.0;
    let mut g = vec![0.0; x.len()];
    if m == 0 {
        return (energy, g);
    }
    let r2: Vec<f64> = pairs.iter().map(|p| p.dist2).collect();
    let mut ep = vec![0.0; m];
    let mut coef = vec![0.0; m];
    let mut curv = vec![0.0; m];
    let rc = unsafe {
        rgpot_lj_curvature_pairs(
            LJ_EPSILON,
            LJ_SIGMA,
            m as i32,
            r2.as_ptr(),
            ep.as_mut_ptr(),
            coef.as_mut_ptr(),
            curv.as_mut_ptr(),
        )
    };
    if rc != 0 {
        panic!("rgpot pair kernel failed ({rc})");
    }
    for (p, (v, s)) in pairs.iter().zip(ep.iter().zip(coef.iter())) {
        energy += *v;
        let i = p.i as usize;
        let j = p.j as usize;
        let dx = x[3 * i] - x[3 * j];
        let dy = x[3 * i + 1] - x[3 * j + 1];
        let dz = x[3 * i + 2] - x[3 * j + 2];
        g[3 * i] += s * dx;
        g[3 * i + 1] += s * dy;
        g[3 * i + 2] += s * dz;
        g[3 * j] -= s * dx;
        g[3 * j + 1] -= s * dy;
        g[3 * j + 2] -= s * dz;
    }
    (energy, g)
}

fn check_lj() -> Result<(), String> {
    let x = vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.0, 0.3, 0.2, 1.3];
    let (e, g) = lennard_jones(&x);
    let mut eref = 0.0;
    let mut gref = vec![0.0; x.len()];
    let rc = unsafe {
        rgpot_lj_gradient(
            LJ_EPSILON,
            LJ_SIGMA,
            LJ_CUTOFF,
            4,
            x.as_ptr(),
            &mut eref,
            gref.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(format!("rgpot_lj_gradient returned {rc}"));
    }
    let de = (e - eref).abs();
    let mut dg = 0.0_f64;
    for (a, b) in g.iter().zip(gref.iter()) {
        dg = dg.max((a - b).abs());
    }
    if de > 1e-9 || dg > 1e-9 {
        return Err(format!("LJ mismatch energy {de} grad {dg}"));
    }
    let pairs = lj_pairs(&x);
    if pairs.is_empty() {
        return Err("linkcell returned no pairs".into());
    }
    println!(
        "{{\"check\":\"ok\",\"pairs\":{},\"energy\":{:.17},\"grad_abs\":{:.17}}}",
        pairs.len(),
        e,
        dg
    );
    Ok(())
}

struct Job {
    kind: String,
    variant: String,
    case_name: String,
    n: usize,
    seed: u64,
    maxiter: usize,
    gtol: f64,
    maxmove: Option<f64>,
}

fn take(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn require(args: &[String], key: &str) -> Result<String, String> {
    take(args, key).ok_or_else(|| format!("missing {key}"))
}

fn parse_job(args: &[String]) -> Result<Job, String> {
    let maxmove = match require(args, "--maxmove")?.as_str() {
        "default" => None,
        text => Some(
            text.parse::<f64>()
                .map_err(|_| format!("bad --maxmove {text}"))?,
        ),
    };
    Ok(Job {
        kind: require(args, "--kind")?,
        variant: require(args, "--variant")?,
        case_name: require(args, "--case")?,
        n: require(args, "--n")?
            .parse()
            .map_err(|_| "bad --n".to_string())?,
        seed: require(args, "--seed")?
            .parse()
            .map_err(|_| "bad --seed".to_string())?,
        maxiter: require(args, "--maxiter")?
            .parse()
            .map_err(|_| "bad --maxiter".to_string())?,
        gtol: require(args, "--gtol")?
            .parse()
            .map_err(|_| "bad --gtol".to_string())?,
        maxmove,
    })
}

fn json_num(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.17}")
    } else if v.is_nan() {
        "null".to_string()
    } else if v.is_sign_positive() {
        "1e9999".to_string()
    } else {
        "-1e9999".to_string()
    }
}

fn run_job(job: &Job) -> Result<(), String> {
    if job.n == 0 {
        return Err("n is 0".into());
    }
    let method = Method::from_name(&job.variant).map_err(|err| format!("variant: {err}"))?;
    let mut control = Control::default()
        .with_maxiter(job.maxiter)
        .with_gtol(job.gtol)
        .with_istep(ISTEP);
    if let Some(cap) = job.maxmove {
        control = control.with_maxmove(cap);
    }
    let linesearch = LineSearch::wolfe(WOLFE_C1, WOLFE_C2, WOLFE_ITERS);
    let start = match job.kind.as_str() {
        "rosenbrock" => {
            if job.n < 2 {
                return Err("rosenbrock needs n >= 2".into());
            }
            rosenbrock_start(job.n)
        }
        "lj" => {
            if job.n < 2 {
                return Err("lj needs n >= 2".into());
            }
            pack(job.n, job.seed)
        }
        other => return Err(format!("unknown kind {other}")),
    };
    let dim = start.len();
    let energy_evals = Arc::new(AtomicUsize::new(0));
    let grad_evals = Arc::new(AtomicUsize::new(0));
    let energy_slot = Arc::clone(&energy_evals);
    let grad_slot = Arc::clone(&grad_evals);
    let kind = job.kind.clone();
    let obj = Oracle::unbounded(dim, move |x: ArrayView1<f64>| {
        let slice = x.as_slice().expect("coordinate view is contiguous");
        let (f, g) = if kind == "lj" {
            lennard_jones(slice)
        } else {
            rosenbrock(slice)
        };
        energy_slot.fetch_add(1, Ordering::Relaxed);
        grad_slot.fetch_add(1, Ordering::Relaxed);
        (f, Array1::from_vec(g))
    });
    let t0 = Instant::now();
    let solved = minimize_method(&obj, Array1::from_vec(start), &control, method, linesearch);
    let wall_s = t0.elapsed().as_secs_f64();
    let e_calls = energy_evals.load(Ordering::Relaxed);
    let g_calls = grad_evals.load(Ordering::Relaxed);
    match solved {
        Ok(report) => {
            println!(
                "{{\"ok\":true,\"kind\":\"{}\",\"variant\":\"{}\",\"case\":\"{}\",\"n\":{},\"seed\":{},\"energy_evals\":{},\"grad_evals\":{},\"solver_evals\":{},\"iterations\":{},\"final_f\":{},\"grad_norm\":{},\"converged\":{},\"status\":\"{:?}\",\"wall_s\":{},\"lj_cutoff\":{}}}",
                job.kind,
                job.variant,
                job.case_name,
                job.n,
                job.seed,
                e_calls,
                g_calls,
                report.evals,
                report.steps,
                json_num(report.value),
                json_num(report.grad_norm),
                if report.converged { "true" } else { "false" },
                report.status,
                json_num(wall_s),
                if job.kind == "lj" {
                    json_num(LJ_CUTOFF)
                } else {
                    "null".to_string()
                }
            );
            Ok(())
        }
        Err(err) => {
            println!(
                "{{\"ok\":false,\"kind\":\"{}\",\"variant\":\"{}\",\"case\":\"{}\",\"n\":{},\"seed\":{},\"energy_evals\":{},\"grad_evals\":{},\"error\":\"{}\",\"wall_s\":{}}}",
                job.kind,
                job.variant,
                job.case_name,
                job.n,
                job.seed,
                e_calls,
                g_calls,
                err.to_string().replace('"', "'"),
                json_num(wall_s)
            );
            Err(format!("solver: {err}"))
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let result = match cmd {
        "check" => check_lj(),
        "run" => parse_job(&args).and_then(|job| run_job(&job)),
        _ => Err("usage: cg-hardening-cost check | run --kind --variant --case --n --seed --maxiter --gtol --maxmove".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}
