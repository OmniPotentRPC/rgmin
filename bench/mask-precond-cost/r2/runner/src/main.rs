//! One minimization of LJ38 or LJ75.
//!
//! The repo potential is the pairwise Lennard-Jones in `tests/gpmin.rs`
//! (`epsilon = 1`, `sigma = 1`, no cutoff). The rgpot arm calls
//! `rgpot::LJClusterPot` with that library's default configuration.
//! Neighbour lists for the exponential metric come from
//! `linkcell::pairs_open`. Starts are the committed minima rattled by a
//! mean-zero normal displacement of width `sigma`, drawn from the
//! splitmix64 generator used by `bench/atomistic`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rgmin::{
    Control, ExpParams, FireKind, Method, Neighbours, Solver, StopNorm, max_atom_displacement,
    nearest_neighbour_distance, needs_rebuild,
};
use sha2::{Digest, Sha256};

const SEARCH0: f64 = 16.0;
const ATOM_CAP: f64 = 0.2;

trait Calls {
    fn force(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>);
    fn calls(&self) -> usize;
}

struct Lj {
    calls: AtomicUsize,
    dim: usize,
    bounds: Bounds<f64>,
}

impl Lj {
    fn new(dim: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            dim,
            bounds: Bounds::new(
                Array1::from_elem(dim, -1.0e6),
                Array1::from_elem(dim, 1.0e6),
                0.0,
            ),
        }
    }
}

impl Calls for Lj {
    fn force(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        lj_cluster(x)
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl Objective<f64> for Lj {
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.force(x).0
    }
}

impl Gradient<f64> for Lj {
    fn dim(&self) -> usize {
        self.dim
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.force(x).1
    }
}

impl DifferentiableObjective<f64> for Lj {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.force(x)
    }
}

struct RgpotLj {
    calls: AtomicUsize,
    dim: usize,
    bounds: Bounds<f64>,
    z: Vec<i32>,
    cell: [f64; 9],
}

impl RgpotLj {
    fn new(dim: usize) -> Self {
        let n = dim / 3;
        Self {
            calls: AtomicUsize::new(0),
            dim,
            bounds: Bounds::new(
                Array1::from_elem(dim, -1.0e6),
                Array1::from_elem(dim, 1.0e6),
                0.0,
            ),
            z: vec![18; n],
            cell: [100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0],
        }
    }
}

impl Calls for RgpotLj {
    fn force(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        rgpot_cluster(&self.z, &self.cell, x)
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl Objective<f64> for RgpotLj {
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.force(x).0
    }
}

impl Gradient<f64> for RgpotLj {
    fn dim(&self) -> usize {
        self.dim
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.force(x).1
    }
}

impl DifferentiableObjective<f64> for RgpotLj {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.force(x)
    }
}

#[cfg(feature = "rgpot")]
unsafe extern "C" {
    fn rgpot_lj_force(
        natoms: usize,
        pos: *const f64,
        atmnrs: *const i32,
        cell: *const f64,
        forces: *mut f64,
        energy: *mut f64,
    ) -> i32;
}

#[cfg(feature = "rgpot")]
fn rgpot_cluster(z: &[i32], cell: &[f64; 9], coords: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = coords.len() / 3;
    let mut forces = vec![0.0; coords.len()];
    let mut energy = 0.0;
    let rc = unsafe {
        rgpot_lj_force(
            n,
            coords.as_ptr(),
            z.as_ptr(),
            cell.as_ptr(),
            forces.as_mut_ptr(),
            &mut energy,
        )
    };
    if rc != 0 {
        return (f64::NAN, Array1::from_elem(coords.len(), f64::NAN));
    }
    let mut g = Array1::zeros(coords.len());
    for (i, f) in forces.iter().enumerate() {
        g[i] = -f;
    }
    (energy, g)
}

#[cfg(not(feature = "rgpot"))]
fn rgpot_cluster(_z: &[i32], _cell: &[f64; 9], coords: ArrayView1<f64>) -> (f64, Array1<f64>) {
    (f64::NAN, Array1::from_elem(coords.len(), f64::NAN))
}

/// Pairwise Lennard-Jones (`sigma`, `epsilon`) with analytic gradient.
fn lj_cluster(coords: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    let sigma = 1.0;
    let epsilon = 1.0;
    let n = coords.len() / 3;
    let sigma2 = sigma * sigma;
    let mut e = 0.0;
    let mut g = Array1::zeros(coords.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[3 * i] - coords[3 * j];
            let dy = coords[3 * i + 1] - coords[3 * j + 1];
            let dz = coords[3 * i + 2] - coords[3 * j + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let inv_r2 = 1.0 / r2;
            let sr2 = sigma2 * inv_r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            e += 4.0 * epsilon * (sr12 - sr6);
            let factor = 24.0 * epsilon * (sr6 - 2.0 * sr12) * inv_r2;
            let fx = factor * dx;
            let fy = factor * dy;
            let fz = factor * dz;
            g[3 * i] += fx;
            g[3 * i + 1] += fy;
            g[3 * i + 2] += fz;
            g[3 * j] -= fx;
            g[3 * j + 1] -= fy;
            g[3 * j + 2] -= fz;
        }
    }
    (e, g)
}

/// splitmix64, the same recurrence as `bench/atomistic`.
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
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn load_xyz(path: &str) -> Result<Vec<[f64; 3]>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut pts = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let x: f64 = it
            .next()
            .ok_or("short row")?
            .parse()
            .map_err(|_| format!("row {n}"))?;
        let y: f64 = it
            .next()
            .ok_or("short row")?
            .parse()
            .map_err(|_| format!("row {n}"))?;
        let z: f64 = it
            .next()
            .ok_or("short row")?
            .parse()
            .map_err(|_| format!("row {n}"))?;
        pts.push([x, y, z]);
    }
    Ok(pts)
}

fn start_from(minima: &[[f64; 3]], seed: u64, sigma: f64) -> Array1<f64> {
    let mut rng = Rng(seed.wrapping_mul(7919) + 17);
    let mut disp = Vec::with_capacity(minima.len());
    for _ in minima {
        disp.push([
            sigma * rng.normal(),
            sigma * rng.normal(),
            sigma * rng.normal(),
        ]);
    }
    let n = disp.len() as f64;
    let mut mean = [0.0; 3];
    for d in &disp {
        for k in 0..3 {
            mean[k] += d[k];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut pos = Vec::with_capacity(minima.len() * 3);
    for (p, d) in minima.iter().zip(&disp) {
        for k in 0..3 {
            pos.push(p[k] + d[k] - mean[k]);
        }
    }
    Array1::from_vec(pos)
}

fn fmax(g: &Array1<f64>) -> f64 {
    let mut best = 0.0_f64;
    let mut i = 0;
    while i + 3 <= g.len() {
        let r = (g[i] * g[i] + g[i + 1] * g[i + 1] + g[i + 2] * g[i + 2]).sqrt();
        best = best.max(r);
        i += 3;
    }
    best
}

fn sha_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_sha(hasher)
}

fn hex_sha(hasher: Sha256) -> String {
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn note(hasher: &mut Sha256, energy: f64, x: &Array1<f64>) {
    hasher.update(energy.to_le_bytes());
    for v in x.iter() {
        hasher.update(v.to_le_bytes());
    }
}

fn coord_sha(x: &Array1<f64>) -> String {
    let mut raw = Vec::with_capacity(x.len() * 8);
    for v in x.iter() {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    sha_bytes(&raw)
}

/// Open-cluster pairs from linkcell, sorted by atom index.
fn neighbour_list(x: &Array1<f64>, cutoff: f64) -> Result<Neighbours, String> {
    let n = x.len() / 3;
    let mut xyz = vec![[0.0; 3]; n];
    for i in 0..n {
        xyz[i] = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
    }
    let rows = linkcell::pairs_open(&xyz, cutoff, None).map_err(|e| format!("linkcell: {e:?}"))?;
    let mut pairs = Vec::with_capacity(rows.len());
    for row in rows {
        if row.shift != [0, 0, 0] {
            return Err("linkcell open list returned a nonzero shift".into());
        }
        if row.i == row.j {
            continue;
        }
        let (a, b) = if row.i < row.j {
            (row.i, row.j)
        } else {
            (row.j, row.i)
        };
        pairs.push((a, b));
    }
    pairs.sort_unstable();
    pairs.dedup();
    if pairs.is_empty() {
        return Err("linkcell returned no pairs".into());
    }
    let mut neigh = Neighbours::default();
    for (i, j) in pairs {
        neigh.i.push(i);
        neigh.j.push(j);
    }
    Ok(neigh)
}

fn metric_neighbours(x: &Array1<f64>) -> Result<(Neighbours, f64), String> {
    let mut cutoff = SEARCH0;
    for _ in 0..4 {
        let neigh = neighbour_list(x, cutoff)?;
        let r_nn = nearest_neighbour_distance(x.view(), &neigh)
            .ok_or_else(|| "no nearest neighbour".to_string())?;
        if 2.0 * r_nn < cutoff {
            return Ok((neigh, r_nn));
        }
        cutoff = (2.0 * r_nn * 1.25).max(cutoff * 2.0);
    }
    Err("neighbour cutoff did not cover 2 r_nn".into())
}

fn pin(core: usize) -> Result<String, String> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core, &mut set);
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if rc != 0 {
            return Err(format!("sched_setaffinity rc {rc}"));
        }
        let mut got: libc::cpu_set_t = std::mem::zeroed();
        let rc = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut got);
        if rc != 0 {
            return Err(format!("sched_getaffinity rc {rc}"));
        }
        let mut cpus = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &got) {
                cpus.push(cpu.to_string());
            }
        }
        Ok(cpus.join(","))
    }
}

struct Args {
    arm: String,
    method: String,
    system: String,
    seed: u64,
    budget: usize,
    fmax: f64,
    sigma: f64,
    core: usize,
    fixtures: String,
    energy_arm: String,
}

fn arg(flag: &str) -> Result<String, String> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == flag {
            return it.next().ok_or_else(|| format!("missing {flag}"));
        }
    }
    Err(format!("missing {flag}"))
}

fn parse() -> Result<Args, String> {
    Ok(Args {
        arm: arg("--arm")?,
        method: arg("--method")?,
        system: arg("--system")?,
        seed: arg("--seed")?.parse().map_err(|_| "seed")?,
        budget: arg("--budget")?.parse().map_err(|_| "budget")?,
        fmax: arg("--fmax")?.parse().map_err(|_| "fmax")?,
        sigma: arg("--sigma")?.parse().map_err(|_| "sigma")?,
        core: arg("--core")?.parse().map_err(|_| "core")?,
        fixtures: arg("--fixtures")?,
        energy_arm: arg("--energy")?,
    })
}

fn emit(value: serde_json::Value) {
    println!("{value}");
}

fn main() {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            emit(serde_json::json!({"error": e}));
            std::process::exit(2);
        }
    };
    let affinity = match pin(args.core) {
        Ok(s) => s,
        Err(e) => {
            emit(serde_json::json!({"arm": args.arm, "error": e}));
            std::process::exit(2);
        }
    };
    let file = match args.system.as_str() {
        "lj38" => "lj38.xyz",
        "lj75" => "lj75.xyz",
        other => {
            emit(serde_json::json!({"error": format!("unknown system {other}")}));
            std::process::exit(2);
        }
    };
    let path = format!("{}/{file}", args.fixtures.trim_end_matches('/'));
    let minima = match load_xyz(&path) {
        Ok(p) => p,
        Err(e) => {
            emit(serde_json::json!({"error": e}));
            std::process::exit(2);
        }
    };
    let expect = if args.system == "lj38" { 38 } else { 75 };
    if minima.len() != expect {
        emit(serde_json::json!({"error": format!("expected {expect} atoms, got {}", minima.len())}));
        std::process::exit(2);
    }
    let x0 = start_from(&minima, args.seed, args.sigma);
    let outcome = match minimize(&args, x0) {
        Ok(v) => v,
        Err(e) => {
            emit(serde_json::json!({
                "arm": args.arm,
                "method": args.method,
                "system": args.system,
                "seed": args.seed,
                "energy_arm": args.energy_arm,
                "core": args.core,
                "affinity": affinity,
                "error": e,
                "converged": false,
            }));
            std::process::exit(1);
        }
    };
    emit(serde_json::json!({
        "arm": args.arm,
        "method": args.method,
        "system": args.system,
        "seed": args.seed,
        "energy_arm": args.energy_arm,
        "evals": outcome.evals,
        "iterations": outcome.iterations,
        "converged": outcome.converged,
        "energy": outcome.energy,
        "energy_bits": outcome.energy_bits,
        "fmax": outcome.fmax,
        "wall_s": outcome.wall_s,
        "core": args.core,
        "affinity": affinity,
        "coord_sha256": outcome.coord_sha256,
        "traj_sha256": outcome.traj_sha256,
        "start_sha256": outcome.start_sha256,
        "kick": outcome.kick,
        "error": null,
    }));
}

struct Outcome {
    evals: usize,
    iterations: usize,
    converged: bool,
    energy: f64,
    energy_bits: u64,
    fmax: f64,
    wall_s: f64,
    coord_sha256: String,
    traj_sha256: String,
    start_sha256: String,
    kick: bool,
}

#[cfg(feature = "metric-kick")]
fn enable_kick(solver: &mut Solver) -> Result<(), String> {
    solver
        .set_fire_metric_kick(true)
        .map_err(|e| format!("set_fire_metric_kick: {e}"))
}

#[cfg(not(feature = "metric-kick"))]
fn enable_kick(_solver: &mut Solver) -> Result<(), String> {
    Err("set_fire_metric_kick is absent on this checkout".into())
}

fn minimize(args: &Args, x: Array1<f64>) -> Result<Outcome, String> {
    if args.energy_arm == "rgpot" && !cfg!(feature = "rgpot") {
        return Err(
            "rgpot arm was not built: set RGPOT_PREFIX and pass --features rgpot".into(),
        );
    }
    if args.energy_arm != "repo" && args.energy_arm != "rgpot" {
        return Err(format!("unknown energy {}", args.energy_arm));
    }
    let (metric, fire, kick) = match args.method.as_str() {
        "fire2" => (false, true, false),
        "fire2-exp" => (true, true, false),
        "fire2-exp-kick" => (true, true, true),
        "eon" => (true, false, false),
        other => return Err(format!("unknown method {other}")),
    };
    let istep = if fire { 0.1 } else { 1.0 };
    let control = Control::default()
        .with_maxiter(1)
        .with_gtol(0.0)
        .with_istep(istep)
        .with_maxmove(None)
        .with_stop_norm(StopNorm::MaxAtom);
    let method = if fire {
        Method::fire(FireKind::V2)
    } else {
        Method::lbfgs()
    };
    let mut solver = Solver::new(method, control, x.len());
    if fire {
        solver.set_atom_maxmove(ATOM_CAP);
    } else {
        solver
            .set_lbfgs_capped(ATOM_CAP)
            .map_err(|e| format!("cap: {e}"))?;
    }
    if kick {
        enable_kick(&mut solver)?;
    }
    let start_sha256 = coord_sha(&x);
    if args.energy_arm == "rgpot" {
        let obj = RgpotLj::new(x.len());
        descend(args, &mut solver, &obj, x, start_sha256, metric, kick)
    } else {
        let obj = Lj::new(x.len());
        descend(args, &mut solver, &obj, x, start_sha256, metric, kick)
    }
}

fn descend<O>(
    args: &Args,
    solver: &mut Solver,
    obj: &O,
    mut x: Array1<f64>,
    start_sha256: String,
    metric: bool,
    kick: bool,
) -> Result<Outcome, String>
where
    O: DifferentiableObjective<f64> + Calls,
{
    let mut traj = Sha256::new();
    let t0 = Instant::now();
    let (mut energy, g0) = obj.force(x.view());
    note(&mut traj, energy, &x);
    let mut fm = fmax(&g0);
    let mut converged = fm < args.fmax && energy.is_finite();
    let mut mu: Option<f64> = None;
    let mut built_at: Option<Array1<f64>> = None;
    let mut gate = 1.0_f64;
    let mut steps = 0usize;
    while !converged && obj.calls() < args.budget && energy.is_finite() {
        if metric {
            let disp = built_at
                .as_ref()
                .map(|p| max_atom_displacement(p.view(), x.view()))
                .unwrap_or(f64::INFINITY);
            if needs_rebuild(disp, gate) {
                let (neigh, r_nn) = metric_neighbours(&x)?;
                gate = r_nn;
                let mut params = ExpParams::default();
                params.mu = mu;
                params.r_nn = Some(r_nn);
                params.r_cut = Some(2.0 * r_nn);
                solver
                    .set_exp(x.view(), &neigh, &params)
                    .map_err(|e| format!("set_exp: {e}"))?;
                built_at = Some(x.clone());
            }
        }
        if obj.calls() >= args.budget {
            break;
        }
        let report = match solver.step(obj, &mut x) {
            Ok(r) => r,
            Err(e) => return Err(format!("step {steps}: {e}")),
        };
        steps += 1;
        energy = report.value;
        let grad = report
            .grad
            .ok_or_else(|| format!("step {steps} returned no gradient"))?;
        fm = fmax(&grad);
        note(&mut traj, energy, &x);
        if mu.is_none() {
            mu = solver.precon_scale();
        }
        if fm < args.fmax && energy.is_finite() {
            converged = true;
        }
        if !energy.is_finite() || !fm.is_finite() {
            break;
        }
    }
    let wall_s = t0.elapsed().as_secs_f64();
    Ok(Outcome {
        evals: obj.calls(),
        iterations: steps,
        converged,
        energy,
        energy_bits: energy.to_bits(),
        fmax: fm,
        wall_s,
        coord_sha256: coord_sha(&x),
        traj_sha256: hex_sha(traj),
        start_sha256,
        kick,
    })
}
