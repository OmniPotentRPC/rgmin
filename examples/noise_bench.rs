//! Noise-floor stop against a fixed force tolerance on two Lennard-Jones
//! clusters.
//!
//! rgpot is not linked in this build, so the eOn platinum heptamer is
//! evaluated as Lennard-Jones on the published coordinates. The first
//! seven atoms move. The rest stay fixed. LJ38 is an FCC fragment with
//! σ = 1 Å and ε = 1 eV. Force noise is either f32 rounding or a
//! Gaussian of standard deviation 1e-4 or 1e-3 eV/Å, keyed by the
//! position so a repeated point returns the same force.
//!
//! Each seed runs every system and stop before the next seed. The
//! baseline is the historical `gtol` stop with the floor left off.
//! `floor` stops at `3 * eps_g`. `tight` asks for an fmax below that
//! floor. A budget hit is `converged=0` and is kept.
//!
//! With `--features par` the second table times the sampling phase at
//! 1, 2, 4, and 8 threads. The work is the force evaluator.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use eindir_core::{DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rgmin::{
    Control, LineSearch, Method, NoiseFlag, Status, minimize_method, next_spacing,
    sample_gradient_noise,
};

const SEEDS: u64 = 5;
const BASE_GTOL: f64 = 0.01;
const FLOOR_C: f64 = 3.0;
const MAXITER: usize = 60;

#[derive(Clone, Copy)]
enum ForceNoise {
    F32,
    Gaussian(f64),
}

struct Cluster {
    free: usize,
    fixed: Array1<f64>,
    epsilon: f64,
    sigma: f64,
    noise: ForceNoise,
    seed: u64,
    calls: AtomicU64,
    bounds: eindir_core::Bounds<f64>,
}

impl Cluster {
    fn clean(free_xyz: Array1<f64>, fixed: Array1<f64>, epsilon: f64, sigma: f64) -> Self {
        let n = free_xyz.len();
        Self {
            free: n / 3,
            fixed,
            epsilon,
            sigma,
            noise: ForceNoise::Gaussian(0.0),
            seed: 0,
            calls: AtomicU64::new(0),
            bounds: eindir_core::Bounds::new(
                Array1::from_elem(n, -1.0e3),
                Array1::from_elem(n, 1.0e3),
                0.0,
            ),
        }
    }

    fn with_noise(&self, noise: ForceNoise, seed: u64) -> Self {
        Self {
            free: self.free,
            fixed: self.fixed.clone(),
            epsilon: self.epsilon,
            sigma: self.sigma,
            noise,
            seed,
            calls: AtomicU64::new(0),
            bounds: self.bounds.clone(),
        }
    }

    fn energy_forces(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let n = self.free;
        let f32_arith = matches!(self.noise, ForceNoise::F32);
        let mut energy = 0.0;
        let mut g = vec![0.0; n * 3];
        for i in 0..n {
            for j in (i + 1)..n {
                let (e, fx, fy, fz) = pair(
                    x[3 * i] - x[3 * j],
                    x[3 * i + 1] - x[3 * j + 1],
                    x[3 * i + 2] - x[3 * j + 2],
                    self.epsilon,
                    self.sigma,
                    f32_arith,
                );
                energy += e;
                g[3 * i] -= fx;
                g[3 * i + 1] -= fy;
                g[3 * i + 2] -= fz;
                g[3 * j] += fx;
                g[3 * j + 1] += fy;
                g[3 * j + 2] += fz;
            }
        }
        let nf = self.fixed.len() / 3;
        for i in 0..n {
            for j in 0..nf {
                let (e, fx, fy, fz) = pair(
                    x[3 * i] - self.fixed[3 * j],
                    x[3 * i + 1] - self.fixed[3 * j + 1],
                    x[3 * i + 2] - self.fixed[3 * j + 2],
                    self.epsilon,
                    self.sigma,
                    f32_arith,
                );
                energy += e;
                g[3 * i] -= fx;
                g[3 * i + 1] -= fy;
                g[3 * i + 2] -= fz;
            }
        }
        if let ForceNoise::Gaussian(sigma) = self.noise {
            if sigma > 0.0 {
                let key = position_key(x);
                for (i, component) in g.iter_mut().enumerate() {
                    *component += sigma * normal(self.seed, key, i as u64);
                }
            }
        }
        (energy, Array1::from_vec(g))
    }
}

impl Objective<f64> for Cluster {
    fn dim(&self) -> usize {
        self.free * 3
    }
    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.energy_forces(x).0
    }
}

impl Gradient<f64> for Cluster {
    fn dim(&self) -> usize {
        self.free * 3
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.energy_forces(x).1
    }
}

impl DifferentiableObjective<f64> for Cluster {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.energy_forces(x)
    }
}

fn pair(dx: f64, dy: f64, dz: f64, eps: f64, sigma: f64, f32_arith: bool) -> (f64, f64, f64, f64) {
    if f32_arith {
        let dx = dx as f32;
        let dy = dy as f32;
        let dz = dz as f32;
        let r2 = dx * dx + dy * dy + dz * dz;
        let sr2 = (sigma as f32) * (sigma as f32) / r2;
        let s6 = sr2 * sr2 * sr2;
        let s12 = s6 * s6;
        let e = 4.0 * (eps as f32) * (s12 - s6);
        let df = 24.0 * (eps as f32) / r2 * (2.0 * s12 - s6);
        return (
            e as f64,
            (df * dx) as f64,
            (df * dy) as f64,
            (df * dz) as f64,
        );
    }
    let r2 = dx * dx + dy * dy + dz * dz;
    let sr2 = sigma * sigma / r2;
    let s6 = sr2 * sr2 * sr2;
    let s12 = s6 * s6;
    let e = 4.0 * eps * (s12 - s6);
    let df = 24.0 * eps / r2 * (2.0 * s12 - s6);
    (e, df * dx, df * dy, df * dz)
}

fn position_key(x: ArrayView1<f64>) -> u64 {
    let mut h = 0xA5A5_A5A5_A5A5_A5A5u64;
    for v in x.iter() {
        h = h
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(v.to_bits());
    }
    h
}

fn normal(seed: u64, key: u64, index: u64) -> f64 {
    let u1 = uniform(seed ^ key, 2 * index);
    let u2 = uniform(seed ^ key, 2 * index + 1);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn uniform(seed: u64, index: u64) -> f64 {
    let mut z = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let mant = (z >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
    mant.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
}

fn lj38_start() -> Array1<f64> {
    let nn = 2.0_f64.powf(1.0 / 6.0);
    let a = nn * 2.0_f64.sqrt();
    let mut pts = Vec::new();
    for i in -3..=3 {
        for j in -3..=3 {
            for k in -3..=3 {
                for b in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    let p = [
                        (i as f64 + b[0]) * a,
                        (j as f64 + b[1]) * a,
                        (k as f64 + b[2]) * a,
                    ];
                    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    pts.push((r, p[0], p[1], p[2]));
                }
            }
        }
    }
    pts.sort_by(|u, v| {
        u.0.partial_cmp(&v.0)
            .unwrap()
            .then_with(|| u.1.partial_cmp(&v.1).unwrap())
            .then_with(|| u.2.partial_cmp(&v.2).unwrap())
            .then_with(|| u.3.partial_cmp(&v.3).unwrap())
    });
    let mut x = Array1::zeros(38 * 3);
    for (i, p) in pts.into_iter().take(38).enumerate() {
        x[3 * i] = p.1;
        x[3 * i + 1] = p.2;
        x[3 * i + 2] = p.3;
    }
    x
}

fn pt_heptamer() -> (Array1<f64>, Array1<f64>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/atomistic/data/pt_heptamer.con"
    );
    let text = std::fs::read_to_string(path).expect(path);
    let mut free = Vec::new();
    let mut fixed = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            continue;
        }
        let Ok(x) = parts[0].parse::<f64>() else {
            continue;
        };
        let Ok(y) = parts[1].parse::<f64>() else {
            continue;
        };
        let Ok(z) = parts[2].parse::<f64>() else {
            continue;
        };
        match parts[3] {
            "0" => free.extend_from_slice(&[x, y, z]),
            "1" => fixed.extend_from_slice(&[x, y, z]),
            _ => {}
        }
    }
    assert_eq!(free.len(), 21, "expected 7 free Pt atoms");
    assert_eq!(fixed.len(), 336 * 3, "expected 336 fixed Pt atoms");
    (Array1::from_vec(free), Array1::from_vec(fixed))
}

fn fmax(g: &Array1<f64>) -> f64 {
    g.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

fn rmsd(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let n = a.len() / 3;
    let mut acc = 0.0;
    for i in 0..n {
        let dx = a[3 * i] - b[3 * i];
        let dy = a[3 * i + 1] - b[3 * i + 1];
        let dz = a[3 * i + 2] - b[3 * i + 2];
        acc += dx * dx + dy * dy + dz * dz;
    }
    (acc / n as f64).sqrt()
}

fn linesearch() -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 12,
    }
}

fn relax(cluster: &Cluster, x: Array1<f64>, gtol: f64) -> (Array1<f64>, f64) {
    let control = Control {
        maxiter: 200,
        gtol,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
        noise_eps: None,
        noise_c: FLOOR_C,
    };
    let report = minimize_method(
        cluster,
        x,
        &control,
        Method::Lbfgs { memory: 8 },
        linesearch(),
    )
    .expect("clean relaxation");
    (report.coords, report.value)
}

fn estimate_eps(cluster: &Cluster, x: ArrayView1<f64>) -> f64 {
    let dir = Array1::ones(x.len());
    let mut h = 1e-4;
    for _ in 0..6 {
        let est =
            sample_gradient_noise(cluster, x, dir.view(), h, 0, cluster.seed.wrapping_add(17))
                .expect("gradient noise sample");
        if est.ok == NoiseFlag::Detected && est.eps > 0.0 {
            return est.eps;
        }
        match next_spacing(h, est.ok) {
            Some(next) if next.is_finite() && next > 0.0 => h = next,
            _ => break,
        }
    }
    match cluster.noise {
        ForceNoise::Gaussian(sigma) if sigma > 0.0 => sigma,
        ForceNoise::F32 => (f32::EPSILON as f64) * cluster.epsilon / cluster.sigma,
        ForceNoise::Gaussian(_) => 1e-16,
    }
}

struct Row {
    system: &'static str,
    method: &'static str,
    seed: u64,
    converged: u8,
    iters: usize,
    calls: u64,
    energy: f64,
    fmax: f64,
    wall_s: f64,
    endpoint: &'static str,
}

fn run_case(
    system: &'static str,
    method: &'static str,
    seed: u64,
    template: &Cluster,
    start: &Array1<f64>,
    reference: &Array1<f64>,
    ref_energy: f64,
    noise: ForceNoise,
    stop: Stop,
) -> Row {
    let obj = template.with_noise(noise, seed);
    let eps = if matches!(stop, Stop::Baseline) {
        0.0
    } else {
        estimate_eps(&obj, start.view())
    };
    obj.calls.store(0, Ordering::SeqCst);
    let mut control = Control {
        maxiter: MAXITER,
        gtol: BASE_GTOL,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
        noise_eps: None,
        noise_c: FLOOR_C,
    };
    match stop {
        Stop::Baseline => {}
        Stop::Floor => {
            control.gtol = 1e-14;
            control.noise_eps = Some(eps);
        }
        Stop::Tight => {
            control.gtol = (FLOOR_C * eps / 10.0).max(1e-14);
        }
    }
    let t0 = Instant::now();
    let report = minimize_method(
        &obj,
        start.clone(),
        &control,
        Method::Lbfgs { memory: 8 },
        linesearch(),
    );
    let wall_s = t0.elapsed().as_secs_f64();
    let calls = obj.calls.load(Ordering::SeqCst);
    let (coords, energy, iters, status) = match report {
        Ok(report) => (report.coords, report.value, report.steps, report.status),
        Err(_) => (start.clone(), f64::NAN, MAXITER, Status::Budget),
    };
    let g = obj.grad(coords.view());
    let fm = fmax(&g);
    let same = (energy - ref_energy).abs() <= 1e-4 && rmsd(coords.view(), reference.view()) < 0.05;
    let converged = match status {
        Status::Tolerance | Status::NoiseFloor if same => 1,
        _ => 0,
    };
    let endpoint = match status {
        Status::Tolerance => "Tolerance",
        Status::NoiseFloor => "NoiseFloor",
        Status::Budget => "Budget",
        Status::Iterating => "Iterating",
    };
    Row {
        system,
        method,
        seed,
        converged,
        iters,
        calls,
        energy,
        fmax: fm,
        wall_s,
        endpoint,
    }
}

#[derive(Clone, Copy)]
enum Stop {
    Baseline,
    Floor,
    Tight,
}

fn median_range(values: &mut [f64]) -> (f64, f64, f64) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values[values.len() / 2];
    (mid, values[0], *values.last().unwrap())
}

fn main() {
    let lj = Cluster::clean(lj38_start(), Array1::zeros(0), 1.0, 1.0);
    let (free, fixed) = pt_heptamer();
    let pt = Cluster::clean(free.clone(), fixed, 0.2, 2.45);
    let (lj_ref, lj_e) = relax(&lj, lj38_start(), 1e-8);
    let (pt_ref, pt_e) = relax(&pt, free, 1e-8);
    let lj_start = kick(&lj_ref, 0.12);
    let pt_start = kick(&pt_ref, 0.12);

    let noises = [
        ("f32", ForceNoise::F32),
        ("g1e-4", ForceNoise::Gaussian(1e-4)),
        ("g1e-3", ForceNoise::Gaussian(1e-3)),
    ];
    let stops = [
        ("baseline", Stop::Baseline),
        ("floor", Stop::Floor),
        ("tight", Stop::Tight),
    ];

    println!("system,method,seed,converged,iters,calls,energy,fmax,wall_s,endpoint");
    let mut rows = Vec::new();
    for seed in 0..SEEDS {
        for (system, template, start, reference, ref_e) in [
            ("lj38", &lj, &lj_start, &lj_ref, lj_e),
            ("pt7", &pt, &pt_start, &pt_ref, pt_e),
        ] {
            for (nname, noise) in noises {
                for (sname, stop) in stops {
                    let method = match (nname, sname) {
                        ("f32", "baseline") => "f32-baseline",
                        ("f32", "floor") => "f32-floor",
                        ("f32", "tight") => "f32-tight",
                        ("g1e-4", "baseline") => "g1e-4-baseline",
                        ("g1e-4", "floor") => "g1e-4-floor",
                        ("g1e-4", "tight") => "g1e-4-tight",
                        ("g1e-3", "baseline") => "g1e-3-baseline",
                        ("g1e-3", "floor") => "g1e-3-floor",
                        ("g1e-3", "tight") => "g1e-3-tight",
                        _ => unreachable!(),
                    };
                    let row = run_case(
                        system, method, seed, template, start, reference, ref_e, noise, stop,
                    );
                    println!(
                        "{system},{method},{},{},{},{},{:.8},{:.6e},{:.6},{endpoint}",
                        row.seed,
                        row.converged,
                        row.iters,
                        row.calls,
                        row.energy,
                        row.fmax,
                        row.wall_s,
                        endpoint = row.endpoint
                    );
                    rows.push(row);
                }
            }
        }
    }

    eprintln!("# median wall_s and range over {SEEDS} seeds");
    let mut keys: Vec<(&str, &str)> = rows.iter().map(|r| (r.system, r.method)).collect();
    keys.sort();
    keys.dedup();
    for (system, method) in keys {
        let mut walls: Vec<f64> = rows
            .iter()
            .filter(|r| r.system == system && r.method == method)
            .map(|r| r.wall_s)
            .collect();
        let calls: Vec<u64> = rows
            .iter()
            .filter(|r| r.system == system && r.method == method)
            .map(|r| r.calls)
            .collect();
        let hits = rows
            .iter()
            .filter(|r| r.system == system && r.method == method && r.converged == 1)
            .count();
        let (med, lo, hi) = median_range(&mut walls);
        eprintln!(
            "# {system} {method}: median {med:.4}s range {lo:.4}..{hi:.4}s calls {calls:?} converged {hits}/{SEEDS}"
        );
    }

    scaling(&pt);
}

fn kick(x: &Array1<f64>, delta: f64) -> Array1<f64> {
    let mut y = x.clone();
    y[0] += delta;
    y
}

fn scaling(pt: &Cluster) {
    #[cfg(not(feature = "par"))]
    {
        let _ = pt;
        eprintln!("# sampling scale: build with --features par to time 1, 2, 4, and 8 threads");
    }
    #[cfg(feature = "par")]
    {
        let x = pt_heptamer().0;
        let dir = Array1::ones(x.len());
        let threads = [1usize, 2, 4, 8];
        let mut samples = vec![Vec::new(); threads.len()];
        for _ in 0..SEEDS {
            for (slot, &n) in threads.iter().enumerate() {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap();
                let dt = pool.install(|| {
                    let t0 = Instant::now();
                    let values = rgmin::line_values(pt, x.view(), dir.view(), 1e-4, 31).unwrap();
                    assert_eq!(values.len(), 32);
                    t0.elapsed().as_secs_f64()
                });
                samples[slot].push(dt);
            }
        }
        let (t1, _, _) = median_range(&mut samples[0].clone());
        eprintln!(
            "# sampling phase, Pt heptamer LJ, m=31 (32 force evaluations), {SEEDS} interleaved repeats"
        );
        eprintln!("# threads median_s lo_s hi_s speedup efficiency");
        for (slot, &n) in threads.iter().enumerate() {
            let (med, lo, hi) = median_range(&mut samples[slot]);
            let speedup = t1 / med;
            let efficiency = speedup / n as f64;
            eprintln!("# {n} {med:.6} {lo:.6} {hi:.6} {speedup:.3} {efficiency:.3}");
        }
        eprintln!(
            "# limiter: the force evaluator. Thirty-two samples still fill an 8-thread pool; efficiency drops as the pair loops contend for memory."
        );
    }
}
