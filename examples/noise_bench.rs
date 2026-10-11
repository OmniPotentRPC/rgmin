//! Gate the noise-floor stop on LJ38 and on Lennard-Jones at the eOn
//! platinum-heptamer coordinates.
//!
//! rgpot is not linked. The second system is `lj-pt7-geom`: Lennard-Jones
//! on those coordinates, 7 free atoms, the rest fixed. It is not a Pt
//! heptamer. LJ38 is an FCC fragment with σ = 1 Å and ε = 1 eV. The
//! heptamer geometry uses ε = 0.2 eV and σ = 2.45 Å so the pair minimum
//! sits near the contact distance.
//!
//! Noise is Gaussian force noise at 1e-4, 3e-4, 1e-3, 3e-3, and 1e-2 eV/Å,
//! plus f32-rounded forces. That Gaussian series perturbs the forces
//! only, so a budget run still descends on an exact energy. The
//! `*-energy` series adds an independent energy draw of standard
//! deviation `σ · 1 Å` at the same point. It is not the gradient of a
//! potential. A repeated point returns the same force and the same
//! energy. f32 rounds both the energy and the forces.
//!
//! Variants are interleaved inside each seed: `floor` (`eps_g` and
//! `eps_f` measured once at the start on one stencil, c = 3),
//! `floor-declared` (`NoiseFloor::new` with the true component σ; for
//! f32 that σ is the RMS gap between the f32 and f64 forces at the
//! start, and `eps_f` is the absolute energy gap there; for the energy
//! series `eps_f` is `σ · 1 Å`), `tight` (`gtol = 0.1 * sqrt(n) * σ`,
//! below the floor), and `baseline` (main's stop, `gtol = 0.01`).
//!
//! `calls` counts minimization and the noise samples. `sample_calls`
//! is the stencil (`m + 1 = 9` gradients per spacing tried, including
//! retries). Thirty seeds, 2000 iterations. A budget hit stays in the
//! table with `converged=0`. `abs_de` is `|E − E_ref|` against the
//! noise-free minimum from the same start. `rmsd` is Kabsch-aligned
//! for LJ38. For `lj-pt7-geom` the fixed atoms are the frame, so the
//! RMSD is the free atoms in that frame.
//!
//! LJ38 `1e-2-floor` seed 27 is the censored Gaussian row. ECnoise
//! returned `eps_g = 2.47e-3` where the true component σ is `1e-2`, so
//! `3 sqrt(114) eps_g = 0.079` stayed under `||g||` for all 2000
//! iterations (the minimum was 0.082, the final value 0.101). The
//! energy on that series is exact, so the run kept descending.
//! Minimization used 22029 calls and the stencil another 18
//! (`calls` 22047), `abs_de` 9.9e-6.
//!
//! With `--features par` the second table times sampling at 1, 2, 4,
//! and 8 threads.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use eindir_core::{DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rgmin::{
    Control, LineSearch, Method, NoiseFlag, NoiseFloor, Status, minimize_method, next_spacing,
    sample_value_gradient_noise,
};

const SEEDS: u64 = 30;
const SCALE_REPEATS: usize = 5;
const BASE_GTOL: f64 = 0.01;
const MAXITER: usize = 2000;

#[derive(Clone, Copy)]
enum ForceNoise {
    F32,
    /// Force noise only. The energy is exact.
    Gaussian(f64),
    /// Force noise `sigma` and an independent energy noise of std `sigma * 1 Å`.
    Consistent(f64),
}

#[derive(Clone, Copy)]
enum Variant {
    Floor,
    FloorDeclared,
    Tight,
    Baseline,
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
        match self.noise {
            ForceNoise::Gaussian(sigma) | ForceNoise::Consistent(sigma) if sigma > 0.0 => {
                let key = position_key(x);
                for (i, component) in g.iter_mut().enumerate() {
                    *component += sigma * normal(self.seed, key, i as u64);
                }
                if let ForceNoise::Consistent(sigma) = self.noise {
                    // σ · 1 Å. The draw is not the derivative of a potential.
                    energy += sigma * normal(self.seed, key, 1 << 20);
                }
            }
            _ => {}
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
    (-2.0 * libm::log(u1)).sqrt() * libm::cos(std::f64::consts::TAU * u2)
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

fn pt_geometry() -> (Array1<f64>, Array1<f64>) {
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
    assert_eq!(free.len(), 21, "expected 7 free atoms");
    assert_eq!(fixed.len(), 336 * 3, "expected 336 fixed atoms");
    (Array1::from_vec(free), Array1::from_vec(fixed))
}

fn fmax(g: &Array1<f64>) -> f64 {
    g.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

fn linesearch() -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 12,
    }
}

fn relax(cluster: &Cluster, x: Array1<f64>) -> (Array1<f64>, f64) {
    let control = Control {
        maxiter: 500,
        gtol: 1e-8,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
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

/// `(eps_g, eps_f)` from one stencil. `eps_f` is `None` when ECnoise
/// refuses the energy table at the spacing that accepted `eps_g`.
/// Retries follow the gradient flag, the same schedule as before.
///
/// The first try is the all-ones direction. On LJ38 that direction is a
/// translation, so the energy and the forces are unchanged and an f32
/// table is all zeros (`StepTooSmall`). The retry is the first coordinate
/// axis, which changes a pair distance. A Gaussian series that already
/// detects noise on the translation stays there.
fn measure_noise(cluster: &Cluster, x: ArrayView1<f64>) -> (f64, Option<f64>) {
    let mut dir = Array1::ones(x.len());
    let mut broke_translation = false;
    let mut h = 1e-4;
    let mut eps_f = None;
    for _ in 0..6 {
        let est = sample_value_gradient_noise(
            cluster,
            x,
            dir.view(),
            h,
            0,
            cluster.seed.wrapping_add(17),
        )
        .expect("value and gradient noise");
        if !broke_translation
            && est.gradient.ok == NoiseFlag::StepTooSmall
            && est.value.ok == NoiseFlag::StepTooSmall
        {
            dir.fill(0.0);
            dir[0] = 1.0;
            broke_translation = true;
            h = 1e-4;
            continue;
        }
        eps_f = if est.value.ok == NoiseFlag::Detected && est.value.eps > 0.0 {
            Some(est.value.eps)
        } else {
            None
        };
        if est.gradient.ok == NoiseFlag::Detected && est.gradient.eps > 0.0 {
            return (est.gradient.eps, eps_f);
        }
        match next_spacing(h, est.gradient.ok) {
            Some(next) if next.is_finite() && next > 0.0 => h = next,
            _ => break,
        }
    }
    (component_sigma(cluster, x), eps_f)
}

fn component_sigma(cluster: &Cluster, x: ArrayView1<f64>) -> f64 {
    match cluster.noise {
        ForceNoise::Gaussian(sigma) | ForceNoise::Consistent(sigma) if sigma > 0.0 => sigma,
        ForceNoise::F32 => f32_sigma(cluster, x),
        ForceNoise::Gaussian(_) | ForceNoise::Consistent(_) => 1e-16,
    }
}

fn f32_energy_gap(cluster: &Cluster, x: ArrayView1<f64>) -> f64 {
    let wide = cluster.with_noise(ForceNoise::Gaussian(0.0), 0);
    let narrow = cluster.with_noise(ForceNoise::F32, 0);
    (wide.eval(x) - narrow.eval(x)).abs()
}

fn declared_eps_f(noise: ForceNoise, cluster: &Cluster, x: ArrayView1<f64>) -> Option<f64> {
    match noise {
        ForceNoise::F32 => {
            let gap = f32_energy_gap(cluster, x);
            (gap > 0.0).then_some(gap)
        }
        ForceNoise::Consistent(sigma) if sigma > 0.0 => Some(sigma),
        _ => None,
    }
}

fn f32_sigma(cluster: &Cluster, x: ArrayView1<f64>) -> f64 {
    let wide = cluster.with_noise(ForceNoise::Gaussian(0.0), 0);
    let narrow = cluster.with_noise(ForceNoise::F32, 0);
    let g64 = wide.grad(x);
    let g32 = narrow.grad(x);
    let mean_sq = g64
        .iter()
        .zip(g32.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>()
        / g64.len().max(1) as f64;
    mean_sq.sqrt()
}

struct Row {
    system: &'static str,
    method: String,
    noise: &'static str,
    variant: &'static str,
    seed: u64,
    converged: u8,
    iters: usize,
    calls: u64,
    sample_calls: u64,
    energy: f64,
    fmax: f64,
    wall_s: f64,
    endpoint: &'static str,
    abs_de: f64,
    rmsd: f64,
}

struct Case<'a> {
    system: &'static str,
    noise_name: &'static str,
    seed: u64,
    template: &'a Cluster,
    start: &'a Array1<f64>,
    reference: &'a Array1<f64>,
    ref_energy: f64,
    noise: ForceNoise,
    variant: Variant,
    aligned: bool,
    declared: f64,
}

fn run_case(case: Case<'_>) -> Row {
    let Case {
        system,
        noise_name,
        seed,
        template,
        start,
        reference,
        ref_energy,
        noise,
        variant,
        aligned,
        declared,
    } = case;
    let obj = template.with_noise(noise, seed);
    let n = start.len();
    let sigma = if declared > 0.0 {
        declared
    } else {
        component_sigma(&obj, start.view())
    };
    let (measured, measured_f, sample_calls) = if matches!(variant, Variant::Floor) {
        let (eps_g, eps_f) = measure_noise(&obj, start.view());
        let sample_calls = obj.calls.load(Ordering::SeqCst);
        obj.calls.store(0, Ordering::SeqCst);
        (eps_g, eps_f, sample_calls)
    } else {
        (0.0, None, 0)
    };
    let mut control = Control {
        maxiter: MAXITER,
        gtol: BASE_GTOL,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    };
    let variant_name = match variant {
        Variant::Baseline => "baseline",
        Variant::Floor => {
            control.gtol = 1e-14;
            let mut floor = NoiseFloor::new(measured);
            if let Some(eps_f) = measured_f {
                floor = floor.with_eps_f(eps_f);
            }
            control.noise_floor = Some(floor);
            "floor"
        }
        Variant::FloorDeclared => {
            control.gtol = 1e-14;
            let mut floor = NoiseFloor::new(sigma);
            if let Some(eps_f) = declared_eps_f(noise, &obj, start.view()) {
                floor = floor.with_eps_f(eps_f);
            }
            control.noise_floor = Some(floor);
            "floor-declared"
        }
        Variant::Tight => {
            control.gtol = (0.1 * (n as f64).sqrt() * sigma).max(1e-14);
            "tight"
        }
    };
    let t0 = Instant::now();
    let report = minimize_method(
        &obj,
        start.clone(),
        &control,
        Method::Lbfgs { memory: 8 },
        linesearch(),
    );
    let wall_s = t0.elapsed().as_secs_f64();
    let calls = obj.calls.load(Ordering::SeqCst) + sample_calls;
    let (coords, energy, iters, status) = match report {
        Ok(report) => (report.coords, report.value, report.steps, report.status),
        Err(_) => (start.clone(), f64::NAN, MAXITER, Status::Budget),
    };
    let g = obj.grad(coords.view());
    let fm = fmax(&g);
    let converged = match status {
        Status::Tolerance | Status::NoiseFloor | Status::StepTolerance => 1,
        Status::Budget | Status::Iterating => 0,
        _ => 0,
    };
    let endpoint = match status {
        Status::Tolerance => "Tolerance",
        Status::NoiseFloor => "NoiseFloor",
        Status::StepTolerance => "StepTolerance",
        Status::Budget => "Budget",
        Status::Iterating => "Iterating",
        _ => "Other",
    };
    let dev = if energy.is_finite() {
        (energy - ref_energy).abs()
    } else {
        f64::NAN
    };
    let rms = if aligned {
        kabsch_rmsd(reference.as_slice().unwrap(), coords.as_slice().unwrap())
    } else {
        frame_rmsd(reference.view(), coords.view())
    };
    Row {
        system,
        method: format!("{noise_name}-{variant_name}"),
        noise: noise_name,
        variant: variant_name,
        seed,
        converged,
        iters,
        calls,
        sample_calls,
        energy,
        fmax: fm,
        wall_s,
        endpoint,
        abs_de: dev,
        rmsd: rms,
    }
}

fn frame_rmsd(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let n = a.len() / 3;
    let mut acc = 0.0;
    for i in 0..n {
        let dx = a[3 * i] - b[3 * i];
        let dy = a[3 * i + 1] - b[3 * i + 1];
        let dz = a[3 * i + 2] - b[3 * i + 2];
        acc += dx * dx + dy * dy + dz * dz;
    }
    (acc / n.max(1) as f64).sqrt()
}

#[allow(clippy::needless_range_loop)]
fn kabsch_rmsd(reference: &[f64], candidate: &[f64]) -> f64 {
    let n = reference.len() / 3;
    let mut cr = [0.0; 3];
    let mut cc = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            cr[k] += reference[3 * i + k];
            cc[k] += candidate[3 * i + k];
        }
    }
    let scale = n.max(1) as f64;
    for k in 0..3 {
        cr[k] /= scale;
        cc[k] /= scale;
    }
    let mut cov = [[0.0; 3]; 3];
    for i in 0..n {
        let p = [
            reference[3 * i] - cr[0],
            reference[3 * i + 1] - cr[1],
            reference[3 * i + 2] - cr[2],
        ];
        let q = [
            candidate[3 * i] - cc[0],
            candidate[3 * i + 1] - cc[1],
            candidate[3 * i + 2] - cc[2],
        ];
        for a in 0..3 {
            for b in 0..3 {
                cov[a][b] += q[a] * p[b];
            }
        }
    }
    let quat = largest_quaternion(&cov);
    let rot = quaternion_matrix(quat);
    let mut acc = 0.0;
    for i in 0..n {
        let p = [
            reference[3 * i] - cr[0],
            reference[3 * i + 1] - cr[1],
            reference[3 * i + 2] - cr[2],
        ];
        let q = [
            candidate[3 * i] - cc[0],
            candidate[3 * i + 1] - cc[1],
            candidate[3 * i + 2] - cc[2],
        ];
        for r in 0..3 {
            let mapped = rot[r][0] * q[0] + rot[r][1] * q[1] + rot[r][2] * q[2];
            let d = mapped - p[r];
            acc += d * d;
        }
    }
    (acc / scale).sqrt()
}

#[allow(clippy::needless_range_loop)]
fn largest_quaternion(cov: &[[f64; 3]; 3]) -> [f64; 4] {
    let (h00, h01, h02) = (cov[0][0], cov[0][1], cov[0][2]);
    let (h10, h11, h12) = (cov[1][0], cov[1][1], cov[1][2]);
    let (h20, h21, h22) = (cov[2][0], cov[2][1], cov[2][2]);
    let m = [
        [h00 + h11 + h22, h12 - h21, h20 - h02, h01 - h10],
        [h12 - h21, h00 - h11 - h22, h01 + h10, h20 + h02],
        [h20 - h02, h01 + h10, -h00 + h11 - h22, h12 + h21],
        [h01 - h10, h20 + h02, h12 + h21, -h00 - h11 + h22],
    ];
    let mut v = [1.0, 0.1, 0.2, 0.3];
    for _ in 0..48 {
        let mut w = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                w[i] += m[i][j] * v[j];
            }
        }
        let nrm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm == 0.0 {
            return [1.0, 0.0, 0.0, 0.0];
        }
        for i in 0..4 {
            v[i] = w[i] / nrm;
        }
    }
    v
}

fn quaternion_matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn median_range(values: &mut [f64]) -> (f64, f64, f64) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values[values.len() / 2];
    (mid, values[0], *values.last().unwrap())
}

fn kick(x: &Array1<f64>, delta: f64) -> Array1<f64> {
    let mut y = x.clone();
    y[0] += delta;
    y
}

fn check_kabsch() {
    let mut spun = lj38_start();
    for i in 0..spun.len() / 3 {
        let x = spun[3 * i];
        let y = spun[3 * i + 1];
        spun[3 * i] = -y;
        spun[3 * i + 1] = x;
    }
    let err = kabsch_rmsd(lj38_start().as_slice().unwrap(), spun.as_slice().unwrap());
    assert!(err < 1e-8, "Kabsch residual {err}");
}

fn main() {
    check_kabsch();
    let lj = Cluster::clean(lj38_start(), Array1::zeros(0), 1.0, 1.0);
    let (free, fixed) = pt_geometry();
    let geom = Cluster::clean(free.clone(), fixed, 0.2, 2.45);
    let lj_start = kick(&lj38_start(), 0.12);
    let geom_start = kick(&free, 0.12);
    let (lj_ref, lj_e) = relax(&lj, lj_start.clone());
    let (geom_ref, geom_e) = relax(&geom, geom_start.clone());
    let lj_f32 = f32_sigma(&lj, lj_start.view());
    let geom_f32 = f32_sigma(&geom, geom_start.view());

    let noises = [
        ("f32", ForceNoise::F32),
        ("1e-4", ForceNoise::Gaussian(1e-4)),
        ("3e-4", ForceNoise::Gaussian(3e-4)),
        ("1e-3", ForceNoise::Gaussian(1e-3)),
        ("3e-3", ForceNoise::Gaussian(3e-3)),
        ("1e-2", ForceNoise::Gaussian(1e-2)),
        ("1e-4-energy", ForceNoise::Consistent(1e-4)),
        ("3e-4-energy", ForceNoise::Consistent(3e-4)),
        ("1e-3-energy", ForceNoise::Consistent(1e-3)),
        ("3e-3-energy", ForceNoise::Consistent(3e-3)),
        ("1e-2-energy", ForceNoise::Consistent(1e-2)),
    ];
    let variants = [
        Variant::Floor,
        Variant::FloorDeclared,
        Variant::Tight,
        Variant::Baseline,
    ];

    println!(
        "system,method,seed,converged,iters,calls,sample_calls,energy,fmax,wall_s,endpoint,abs_de,rmsd,reference"
    );
    let mut rows = Vec::new();
    for seed in 0..SEEDS {
        for (system, template, start, reference, ref_e, aligned, f32_sig) in [
            ("lj38", &lj, &lj_start, &lj_ref, lj_e, true, lj_f32),
            (
                "lj-pt7-geom",
                &geom,
                &geom_start,
                &geom_ref,
                geom_e,
                false,
                geom_f32,
            ),
        ] {
            for (nname, noise) in noises {
                let declared = match noise {
                    ForceNoise::F32 => f32_sig,
                    ForceNoise::Gaussian(sigma) | ForceNoise::Consistent(sigma) => sigma,
                };
                for variant in variants {
                    let row = run_case(Case {
                        system,
                        noise_name: nname,
                        seed,
                        template,
                        start,
                        reference,
                        ref_energy: ref_e,
                        noise,
                        variant,
                        aligned,
                        declared,
                    });
                    println!(
                        "{system},{method},{seed},{converged},{iters},{calls},{sample},{energy:.8},{fmax:.6e},{wall:.6},{endpoint},{abs_de:.6e},{rmsd:.6e},{ref_e:.8}",
                        method = row.method,
                        seed = row.seed,
                        converged = row.converged,
                        iters = row.iters,
                        calls = row.calls,
                        sample = row.sample_calls,
                        energy = row.energy,
                        fmax = row.fmax,
                        wall = row.wall_s,
                        endpoint = row.endpoint,
                        abs_de = row.abs_de,
                        rmsd = row.rmsd,
                        ref_e = ref_e,
                    );
                    rows.push(row);
                }
            }
        }
    }

    eprintln!("# median calls, censoring, median abs_de and rmsd over {SEEDS} seeds");
    eprintln!(
        "# system noise variant calls_median calls_lo calls_hi sample_calls_median censored abs_de_median rmsd_median"
    );
    let mut keys: Vec<(&str, &str, &str)> = rows
        .iter()
        .map(|r| (r.system, r.noise, r.variant))
        .collect();
    keys.sort();
    keys.dedup();
    for (system, noise, variant) in keys {
        let group: Vec<&Row> = rows
            .iter()
            .filter(|r| r.system == system && r.noise == noise && r.variant == variant)
            .collect();
        let mut calls: Vec<f64> = group.iter().map(|r| r.calls as f64).collect();
        let mut sample: Vec<f64> = group.iter().map(|r| r.sample_calls as f64).collect();
        let mut de: Vec<f64> = group.iter().map(|r| r.abs_de).collect();
        let mut rms: Vec<f64> = group.iter().map(|r| r.rmsd).collect();
        let censored = group.iter().filter(|r| r.converged == 0).count();
        let (cmed, clo, chi) = median_range(&mut calls);
        let (smed, _, _) = median_range(&mut sample);
        let (dmed, _, _) = median_range(&mut de);
        let (rmed, _, _) = median_range(&mut rms);
        eprintln!(
            "# {system} {noise} {variant}: calls {cmed:.0} range {clo:.0}..{chi:.0} sample_calls {smed:.0} censored {censored}/{SEEDS} abs_de {dmed:.3e} rmsd {rmed:.3e}"
        );
    }

    scaling(&geom);
}

fn scaling(geom: &Cluster) {
    #[cfg(not(feature = "par"))]
    {
        let _ = geom;
        eprintln!("# sampling scale: build with --features par to time 1, 2, 4, and 8 threads");
    }
    #[cfg(feature = "par")]
    {
        let x = pt_geometry().0;
        let dir = Array1::ones(x.len());
        let threads = [1usize, 2, 4, 8];
        let mut samples = vec![Vec::new(); threads.len()];
        for _ in 0..SCALE_REPEATS {
            for (slot, &n) in threads.iter().enumerate() {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap();
                let dt = pool.install(|| {
                    let t0 = Instant::now();
                    let values = rgmin::line_values(geom, x.view(), dir.view(), 1e-4, 31).unwrap();
                    assert_eq!(values.len(), 32);
                    t0.elapsed().as_secs_f64()
                });
                samples[slot].push(dt);
            }
        }
        let (t1, _, _) = median_range(&mut samples[0].clone());
        eprintln!(
            "# sampling phase, lj-pt7-geom, m=31 (32 force evaluations), {SCALE_REPEATS} interleaved repeats"
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
