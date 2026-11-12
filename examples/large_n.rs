// Copyright (c) rgmin developers

//! Per-step optimizer overhead on large Lennard-Jones and Morse systems.
//!
//! Clusters are finite FCC chunks. Slabs are periodic in x and y. The
//! arms are the default capped L-BFGS (memory 20, the eOn client
//! setting, no line search), FIRE 1.0, and the eOn conjugate-gradient
//! step. Strong Wolfe is not on this path.
//!
//! Overhead is wall time minus the time spent inside the force call.
//! Force calls are the count to fmax, or to the iteration budget when
//! the run stops first.
//!
//! `cargo run --release --example large_n -- bench/raw/large_n.csv`
//! `cargo run --release --example large_n -- --pin tests/data/traj_pin.txt`

use std::time::Instant;

use ndarray::Array1;
use rgmin::{Control, EonCg, FireKind, Method, Solver, StopNorm, minimize_eon_cg};

const FMAX: f64 = 1.0e-2;
const BUDGET: usize = 400;
const CAP: f64 = 0.2;
const MEMORY: usize = 20;

#[derive(Clone, Copy)]
enum Kind {
    Lj,
    Morse,
}

#[derive(Clone, Copy)]
enum Arm {
    Lbfgs,
    Fire,
    Cg,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Lbfgs => "lbfgs",
            Self::Fire => "fire",
            Self::Cg => "cg",
        }
    }

    fn all() -> &'static [Arm] {
        &[Self::Lbfgs, Self::Fire, Self::Cg]
    }
}

struct System {
    name: &'static str,
    kind: Kind,
    atoms: usize,
    x0: Vec<f64>,
    /// Positive entries are periodic lengths. Zero is free.
    cell: [f64; 3],
    cutoff: f64,
}

struct Cells {
    head: Vec<i32>,
    next: Vec<i32>,
}

impl Cells {
    fn new() -> Self {
        Self {
            head: Vec::new(),
            next: Vec::new(),
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let first = args.next();
    if first.as_deref() == Some("--pin") {
        let path = args
            .next()
            .unwrap_or_else(|| "tests/data/traj_pin.txt".to_string());
        let text = pin_text();
        std::fs::write(&path, &text).expect("pin file");
        eprintln!("wrote {path} ({} bytes)", text.len());
        return;
    }
    let out = first.unwrap_or_else(|| "bench/raw/large_n.csv".to_string());
    let only = std::env::var("RGMIN_ONLY").ok();
    let repeats = std::env::var("RGMIN_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1usize)
        .max(1);
    let mut systems = systems();
    if let Some(only) = only.as_deref() {
        let names: Vec<&str> = only.split(',').collect();
        systems.retain(|s| names.contains(&s.name));
    }
    check_cell_list(&systems[0]);
    let mut rows = String::from(
        "system,potential,geometry,atoms,arm,steps,force_calls,fmax,energy,converged,wall_ms,force_ms,overhead_ms,overhead_us_per_step\n",
    );
    for sys in &systems {
        let mut g = vec![0.0; sys.x0.len()];
        let mut cells = Cells::new();
        let e0 = eval(&mut cells, sys, &sys.x0, &mut g);
        eprintln!(
            "{} atoms {} E/atom {:.6} fmax {:.4e}",
            sys.name,
            sys.atoms,
            e0 / sys.atoms as f64,
            max_atom(&g)
        );
        for arm in Arm::all() {
            for rep in 0..repeats {
                let row = run(sys, *arm);
                let _ = rep;
                eprintln!(
                    "{:16} {:5} steps {:4} calls {:4} fmax {:.3e} conv {} over {:.3} us/step",
                    sys.name,
                    arm.name(),
                    row.steps,
                    row.calls,
                    row.fmax,
                    row.converged as u8,
                    row.overhead_us
                );
                rows.push_str(&format!(
                    "{},{},{},{},{},{},{},{:.6e},{:.8e},{},{:.4},{:.4},{:.4},{:.4}\n",
                    sys.name,
                    pot_name(sys.kind),
                    geom_name(sys.name),
                    sys.atoms,
                    arm.name(),
                    row.steps,
                    row.calls,
                    row.fmax,
                    row.energy,
                    row.converged as u8,
                    row.wall_ms,
                    row.force_ms,
                    row.overhead_ms,
                    row.overhead_us
                ));
                std::fs::write(&out, &rows).expect("csv");
            }
        }
    }
    eprintln!("wrote {out}");
}

struct Row {
    steps: usize,
    calls: usize,
    fmax: f64,
    energy: f64,
    converged: bool,
    wall_ms: f64,
    force_ms: f64,
    overhead_ms: f64,
    overhead_us: f64,
}

fn run(sys: &System, arm: Arm) -> Row {
    let mut cells = Cells::new();
    let mut calls = 0usize;
    let mut force_ns = 0u128;
    let mut x = Array1::from(sys.x0.clone());
    let t0 = Instant::now();
    let (steps, energy, fmax, converged) = match arm {
        Arm::Lbfgs | Arm::Fire => {
            let method = match arm {
                Arm::Lbfgs => Method::Lbfgs { memory: MEMORY },
                Arm::Fire => Method::Fire { kind: FireKind::V1 },
                Arm::Cg => unreachable!(),
            };
            let control = Control {
                maxiter: BUDGET,
                gtol: FMAX,
                istep: 1.0,
                stop_norm: StopNorm::MaxAtom,
                ..Control::default()
            };
            let mut solver = Solver::new(method, control, x.len());
            solver.set_atom_maxmove(CAP);
            let result = solver.run_fill(&mut x, |v, g| {
                calls += 1;
                let t = Instant::now();
                let e = eval(
                    &mut cells,
                    sys,
                    v.as_slice().expect("coordinates are contiguous"),
                    g.as_slice_mut().expect("gradient is contiguous"),
                );
                force_ns += t.elapsed().as_nanos();
                e
            });
            let report = result.expect("fill step");
            let fmax = report.grad_norm;
            (report.steps, report.value, fmax, fmax < FMAX)
        }
        Arm::Cg => {
            let xs = x.as_slice_mut().expect("coordinates are contiguous");
            let opts = EonCg {
                max_move: CAP,
                gtol: FMAX,
                max_iter: BUDGET,
                cluster: true,
                atom_only: false,
            };
            let rep = minimize_eon_cg(
                xs,
                |pos, g| {
                    calls += 1;
                    let t = Instant::now();
                    let e = eval(&mut cells, sys, pos, g);
                    force_ns += t.elapsed().as_nanos();
                    e
                },
                &opts,
            );
            let steps = rep.evals.saturating_sub(1) / 2;
            (steps, rep.energy, rep.grad, rep.converged)
        }
    };
    let wall_ms = t0.elapsed().as_secs_f64() * 1.0e3;
    let force_ms = force_ns as f64 / 1.0e6;
    let overhead_ms = (wall_ms - force_ms).max(0.0);
    let overhead_us = if steps == 0 {
        0.0
    } else {
        overhead_ms * 1.0e3 / steps as f64
    };
    Row {
        steps,
        calls,
        fmax,
        energy,
        converged,
        wall_ms,
        force_ms,
        overhead_ms,
        overhead_us,
    }
}

fn pot_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Lj => "lj",
        Kind::Morse => "morse",
    }
}

fn geom_name(name: &str) -> &'static str {
    if name.contains("slab") {
        "slab"
    } else {
        "cluster"
    }
}

fn systems() -> Vec<System> {
    let mut out = Vec::new();
    for &n in &[500usize, 5_000, 50_000] {
        out.push(cluster(Kind::Lj, n));
        out.push(slab(Kind::Lj, n));
        out.push(cluster(Kind::Morse, n));
        out.push(slab(Kind::Morse, n));
    }
    out
}

fn cluster(kind: Kind, atoms: usize) -> System {
    let (spacing, cutoff) = match kind {
        Kind::Lj => (1.122462048309373, 2.5),
        Kind::Morse => (2.897, 7.0),
    };
    let a = spacing * std::f64::consts::SQRT_2;
    System {
        name: match kind {
            Kind::Lj => match atoms {
                500 => "lj-cluster-500",
                5_000 => "lj-cluster-5000",
                _ => "lj-cluster-50000",
            },
            Kind::Morse => match atoms {
                500 => "morse-cluster-500",
                5_000 => "morse-cluster-5000",
                _ => "morse-cluster-50000",
            },
        },
        kind,
        atoms,
        x0: fcc_chunk(atoms, a, 0.02),
        cell: [0.0; 3],
        cutoff,
    }
}

fn slab(kind: Kind, atoms: usize) -> System {
    let (a, cutoff) = match kind {
        Kind::Lj => (1.122462048309373 * std::f64::consts::SQRT_2, 2.5),
        Kind::Morse => (2.897 * std::f64::consts::SQRT_2, 7.0),
    };
    let (nx, ny, nz) = slab_counts(atoms);
    let (x0, cell) = fcc_slab(nx, ny, nz, a, 0.02);
    debug_assert_eq!(x0.len() / 3, atoms);
    System {
        name: match kind {
            Kind::Lj => match atoms {
                500 => "lj-slab-500",
                5_000 => "lj-slab-5000",
                _ => "lj-slab-50000",
            },
            Kind::Morse => match atoms {
                500 => "morse-slab-500",
                5_000 => "morse-slab-5000",
                _ => "morse-slab-50000",
            },
        },
        kind,
        atoms,
        x0,
        cell,
        cutoff,
    }
}

fn slab_counts(atoms: usize) -> (usize, usize, usize) {
    match atoms {
        500 => (5, 5, 5),
        5_000 => (25, 10, 5),
        50_000 => (50, 25, 10),
        _ => panic!("slab size {atoms} is not a 4*nx*ny*nz target"),
    }
}

fn fcc_basis() -> [[f64; 3]; 4] {
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
}

fn fcc_slab(nx: usize, ny: usize, nz: usize, a: f64, rattle: f64) -> (Vec<f64>, [f64; 3]) {
    let mut x = Vec::new();
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                for b in fcc_basis() {
                    let p = [
                        (ix as f64 + b[0]) * a,
                        (iy as f64 + b[1]) * a,
                        (iz as f64 + b[2]) * a,
                    ];
                    let k = x.len() / 3;
                    for (c, v) in p.iter().enumerate() {
                        x.push(v + rattle * ((k * 3 + c) as f64 * 1.7).sin());
                    }
                }
            }
        }
    }
    (x, [nx as f64 * a, ny as f64 * a, 0.0])
}

/// First `atoms` sites of an FCC cube, rattled. Not periodic.
fn fcc_chunk(atoms: usize, a: f64, rattle: f64) -> Vec<f64> {
    let cells = ((atoms as f64 / 4.0).cbrt().ceil() as usize).max(1);
    let mut raw = Vec::new();
    for iz in 0..cells {
        for iy in 0..cells {
            for ix in 0..cells {
                for b in fcc_basis() {
                    raw.push([
                        (ix as f64 + b[0]) * a,
                        (iy as f64 + b[1]) * a,
                        (iz as f64 + b[2]) * a,
                    ]);
                }
            }
        }
    }
    raw.truncate(atoms);
    let mut x = Vec::with_capacity(3 * raw.len());
    for (k, p) in raw.iter().enumerate() {
        for (c, v) in p.iter().enumerate() {
            x.push(v + rattle * ((k * 3 + c) as f64 * 1.7).sin());
        }
    }
    x
}

fn max_atom(g: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    let mut i = 0;
    while i < g.len() {
        let end = (i + 3).min(g.len());
        let mut a2 = 0.0;
        for v in &g[i..end] {
            a2 += v * v;
        }
        worst = worst.max(a2.sqrt());
        i = end;
    }
    worst
}

fn mic(d: f64, boxl: f64) -> f64 {
    if boxl > 0.0 {
        d - boxl * (d / boxl).round()
    } else {
        d
    }
}

fn eval(cells: &mut Cells, sys: &System, x: &[f64], g: &mut [f64]) -> f64 {
    g.fill(0.0);
    match sys.kind {
        Kind::Lj => lj_cell(cells, sys.cell, sys.cutoff, x, g),
        Kind::Morse => morse_cell(cells, sys.cell, sys.cutoff, x, g),
    }
}

fn lj_pair(r2: f64) -> (f64, f64) {
    let s6 = r2.powi(-3);
    let e = 4.0 * (s6 * s6 - s6);
    let du = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
    (e, du)
}

fn lj_shift(rc: f64) -> f64 {
    lj_pair(rc * rc).0
}

fn morse_params() -> (f64, f64, f64, f64) {
    let (d0, alpha, r0): (f64, f64, f64) = (0.7102, 1.6047, 2.897);
    let xc = (-alpha * (7.0 - r0)).exp();
    let shift = d0 * (xc * xc - 2.0 * xc);
    (d0, alpha, r0, shift)
}

fn build_cells(
    cells: &mut Cells,
    x: &[f64],
    cell: [f64; 3],
    rc: f64,
) -> (usize, usize, usize, [f64; 3], [f64; 3]) {
    let n = x.len() / 3;
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for i in 0..n {
        for a in 0..3 {
            lo[a] = lo[a].min(x[3 * i + a]);
            hi[a] = hi[a].max(x[3 * i + a]);
        }
    }
    let mut ncell = [1usize; 3];
    let mut origin = [0.0; 3];
    let mut inv = [1.0 / rc; 3];
    for a in 0..3 {
        if cell[a] > 0.0 {
            origin[a] = 0.0;
            ncell[a] = (cell[a] / rc).floor().max(1.0) as usize;
            inv[a] = ncell[a] as f64 / cell[a];
        } else {
            origin[a] = lo[a];
            let span = (hi[a] - lo[a]).max(0.0);
            ncell[a] = (span / rc).floor() as usize + 1;
            inv[a] = 1.0 / rc;
        }
    }
    let ntot = ncell[0] * ncell[1] * ncell[2];
    cells.head.clear();
    cells.head.resize(ntot, -1);
    if cells.next.len() != n {
        cells.next.resize(n, -1);
    }
    for i in 0..n {
        let mut idx = [0usize; 3];
        for a in 0..3 {
            let t = (x[3 * i + a] - origin[a]) * inv[a];
            let mut k = t.floor() as isize;
            if cell[a] > 0.0 {
                let m = ncell[a] as isize;
                k = k.rem_euclid(m);
            } else if k < 0 {
                k = 0;
            } else if k >= ncell[a] as isize {
                k = ncell[a] as isize - 1;
            }
            idx[a] = k as usize;
        }
        let slot = (idx[2] * ncell[1] + idx[1]) * ncell[0] + idx[0];
        cells.next[i] = cells.head[slot];
        cells.head[slot] = i as i32;
    }
    (ncell[0], ncell[1], ncell[2], origin, inv)
}

fn for_pairs(
    cells: &mut Cells,
    x: &[f64],
    cell: [f64; 3],
    rc: f64,
    mut pair: impl FnMut(usize, usize, [f64; 3], f64),
) {
    let (nx, ny, nz, _, _) = build_cells(cells, x, cell, rc);
    let rc2 = rc * rc;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let home = (iz * ny + iy) * nx + ix;
                let mut a = cells.head[home];
                while a >= 0 {
                    let i = a as usize;
                    a = cells.next[i];
                    for dz in -1isize..=1 {
                        for dy in -1isize..=1 {
                            for dx in -1isize..=1 {
                                let Some([jx, jy, jz]) =
                                    neighbor([ix, iy, iz], [dx, dy, dz], [nx, ny, nz], cell)
                                else {
                                    continue;
                                };
                                let slot = (jz * ny + jy) * nx + jx;
                                let mut b = cells.head[slot];
                                while b >= 0 {
                                    let j = b as usize;
                                    b = cells.next[j];
                                    if j <= i {
                                        continue;
                                    }
                                    let mut d = [0.0; 3];
                                    let mut r2 = 0.0;
                                    for c in 0..3 {
                                        d[c] = mic(x[3 * i + c] - x[3 * j + c], cell[c]);
                                        r2 += d[c] * d[c];
                                    }
                                    if r2 < rc2 && r2 > 1.0e-16 {
                                        pair(i, j, d, r2);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn neighbor(i: [usize; 3], d: [isize; 3], n: [usize; 3], cell: [f64; 3]) -> Option<[usize; 3]> {
    let mut out = [0usize; 3];
    for a in 0..3 {
        let v = i[a] as isize + d[a];
        out[a] = if cell[a] > 0.0 {
            v.rem_euclid(n[a] as isize) as usize
        } else if v < 0 || v >= n[a] as isize {
            return None;
        } else {
            v as usize
        };
    }
    Some(out)
}

fn lj_cell(cells: &mut Cells, cell: [f64; 3], rc: f64, x: &[f64], g: &mut [f64]) -> f64 {
    let shift = lj_shift(rc);
    let mut e = 0.0;
    for_pairs(cells, x, cell, rc, |i, j, d, r2| {
        let (ep, du) = lj_pair(r2);
        e += ep - shift;
        for a in 0..3 {
            let c = du * d[a];
            g[3 * i + a] += c;
            g[3 * j + a] -= c;
        }
    });
    e
}

fn morse_cell(cells: &mut Cells, cell: [f64; 3], rc: f64, x: &[f64], g: &mut [f64]) -> f64 {
    let (d0, alpha, r0, shift) = morse_params();
    let mut e = 0.0;
    for_pairs(cells, x, cell, rc, |i, j, d, r2| {
        let r = r2.sqrt();
        if r < 1.0e-8 {
            return;
        }
        let ex = (-alpha * (r - r0)).exp();
        e += d0 * (ex * ex - 2.0 * ex) - shift;
        let du = 2.0 * d0 * alpha * ex * (1.0 - ex) / r;
        for a in 0..3 {
            let c = du * d[a];
            g[3 * i + a] += c;
            g[3 * j + a] -= c;
        }
    });
    e
}

fn naive(sys: &System, x: &[f64], g: &mut [f64]) -> f64 {
    g.fill(0.0);
    let n = x.len() / 3;
    let rc2 = sys.cutoff * sys.cutoff;
    let mut e = 0.0;
    let shift = match sys.kind {
        Kind::Lj => lj_shift(sys.cutoff),
        Kind::Morse => morse_params().3,
    };
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            let mut r2 = 0.0;
            for a in 0..3 {
                d[a] = mic(x[3 * i + a] - x[3 * j + a], sys.cell[a]);
                r2 += d[a] * d[a];
            }
            if r2 >= rc2 || r2 < 1.0e-16 {
                continue;
            }
            let (ep, du) = match sys.kind {
                Kind::Lj => {
                    let (ep, du_r) = lj_pair(r2);
                    (ep - shift, du_r)
                }
                Kind::Morse => {
                    let (d0, alpha, r0, _) = morse_params();
                    let r = r2.sqrt();
                    let ex = (-alpha * (r - r0)).exp();
                    (
                        d0 * (ex * ex - 2.0 * ex) - shift,
                        2.0 * d0 * alpha * ex * (1.0 - ex) / r,
                    )
                }
            };
            e += ep;
            for a in 0..3 {
                let c = du * d[a];
                g[3 * i + a] += c;
                g[3 * j + a] -= c;
            }
        }
    }
    e
}

fn check_cell_list(sample: &System) {
    let small = cluster(sample.kind, 32);
    let mut slab = slab(Kind::Lj, 500);
    slab.x0.truncate(3 * 40);
    slab.atoms = 40;
    for sys in [&small, &slab] {
        let mut g_cell = vec![0.0; sys.x0.len()];
        let mut g_naive = vec![0.0; sys.x0.len()];
        let mut cells = Cells::new();
        let e_cell = eval(&mut cells, sys, &sys.x0, &mut g_cell);
        let e_naive = naive(sys, &sys.x0, &mut g_naive);
        let de = (e_cell - e_naive).abs();
        let mut dg = 0.0_f64;
        for (a, b) in g_cell.iter().zip(&g_naive) {
            dg = dg.max((a - b).abs());
        }
        if de > 1e-9 || dg > 1e-9 {
            panic!(
                "{} cell list disagrees with the pair loop: dE {de} dG {dg}",
                sys.name
            );
        }
    }
}

/// Ten steps on a 7-atom Lennard-Jones chunk. The file is the reference
/// the trajectory test reads back.
fn pin_text() -> String {
    let mut sys = cluster(Kind::Lj, 16);
    sys.x0.truncate(3 * 7);
    sys.atoms = 7;
    sys.name = "lj-pin-7";
    let mut out = String::from("# arm call energy x...\n");
    for arm in Arm::all() {
        let mut cells = Cells::new();
        let mut x = Array1::from(sys.x0.clone());
        let mut call = 0usize;
        let record = |out: &mut String, call: &mut usize, energy: f64, pos: &[f64]| {
            *call += 1;
            push_pin(out, *arm, *call, energy, pos);
        };
        match arm {
            Arm::Cg => {
                let opts = EonCg {
                    max_move: CAP,
                    gtol: 0.0,
                    max_iter: 10,
                    cluster: true,
                    atom_only: false,
                };
                let xs = x.as_slice_mut().unwrap();
                let _ = minimize_eon_cg(
                    xs,
                    |pos, grad| {
                        let e = eval(&mut cells, &sys, pos, grad);
                        record(&mut out, &mut call, e, pos);
                        e
                    },
                    &opts,
                );
            }
            Arm::Lbfgs | Arm::Fire => {
                let method = match arm {
                    Arm::Lbfgs => Method::Lbfgs { memory: MEMORY },
                    _ => Method::Fire { kind: FireKind::V1 },
                };
                let control = Control {
                    maxiter: 10,
                    gtol: 0.0,
                    istep: 1.0,
                    stop_norm: StopNorm::MaxAtom,
                    ..Control::default()
                };
                let mut solver = Solver::new(method, control, x.len());
                solver.set_atom_maxmove(CAP);
                solver
                    .run_fill(&mut x, |v, grad| {
                        let e = eval(
                            &mut cells,
                            &sys,
                            v.as_slice().unwrap(),
                            grad.as_slice_mut().unwrap(),
                        );
                        record(&mut out, &mut call, e, v.as_slice().unwrap());
                        e
                    })
                    .expect("pin run");
            }
        }
    }
    out
}

fn push_pin(out: &mut String, arm: Arm, call: usize, energy: f64, x: &[f64]) {
    out.push_str(&format!("{arm} {call} {energy:.16e}", arm = arm.name()));
    for v in x {
        out.push_str(&format!(" {v:.16e}"));
    }
    out.push('\n');
}
