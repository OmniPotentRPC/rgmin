// Copyright (c) rgmin developers

//! Evaluation counts on clusters where the force call dominates.
//!
//! Lennard-Jones 75, 98 and 150, a Morse platinum slab, Gupta platinum
//! and copper slabs and nanoparticles, and a flexible TIP3P water
//! cluster. The arms are the eOn cap, FIRE, and the opt-in step rules.
//! A row that stops on the iteration budget is not a win.
//!
//! `env -i PATH HOME LANG RUSTUP_HOME CARGO_HOME cargo run --release --example bench_algo -- bench/raw/algo.csv bench/raw/steps.csv`

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rgmin::{
    Accept, Control, EonCg, EonFeat, ExpParams, FireKind, LbfgsStep, LindhSpec, Method, Neighbours,
    Oracle, Solver, StopNorm, TrainRow, apply_inverse, exp_gate_open, minimize_eon_cg,
    select_train_setting,
};

const CAP: f64 = 0.2;

#[derive(Clone, Copy)]
enum Arm {
    Eon,
    Fire,
    FireGlobal,
    Fire2,
    FireAbc,
    FixedH0,
    Scaled,
    Exp,
    FireExp,
    Euclid,
    Trust,
    Bb,
    BbNonmonotone,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Eon => "lbfgs-eon",
            Self::Fire => "fire",
            Self::FireGlobal => "fire-global",
            Self::Fire2 => "fire2",
            Self::FireAbc => "fire2-abc",
            Self::FixedH0 => "lbfgs-h0",
            Self::Scaled => "lbfgs-scaled",
            Self::Exp => "lbfgs-exp",
            Self::FireExp => "fire-exp",
            Self::Euclid => "lbfgs-euclid",
            Self::Trust => "lbfgs-trust",
            Self::Bb => "bb",
            Self::BbNonmonotone => "bb-nonmonotone",
        }
    }

    fn all() -> &'static [Arm] {
        &[
            Self::Eon,
            Self::Fire,
            Self::FireGlobal,
            Self::Fire2,
            Self::FireAbc,
            Self::FixedH0,
            Self::Scaled,
            Self::Exp,
            Self::FireExp,
            Self::Euclid,
            Self::Trust,
            Self::Bb,
            Self::BbNonmonotone,
        ]
    }
}

struct Problem {
    name: &'static str,
    split: &'static str,
    tol: f64,
    budget: usize,
    x0: Array1<f64>,
    /// Neighbour cutoff for the exponential preconditioner.
    rc: f64,
    /// Periodic lengths. A non-positive entry is not periodic.
    cell: [f64; 3],
    kind: Kind,
}

#[derive(Clone, Copy)]
enum Kind {
    Lj,
    Morse,
    Gupta(Gupta),
    Water,
}

#[derive(Clone, Copy)]
struct Gupta {
    a: f64,
    xi: f64,
    p: f64,
    q: f64,
    r0: f64,
    r_on: f64,
    rc: f64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let first = args.next();
    if first.as_deref() == Some("--overhead") {
        overhead_only();
        return;
    }
    if first.as_deref() == Some("--lindh") {
        let out = args
            .next()
            .unwrap_or_else(|| "bench/raw/lindh-scale.csv".to_string());
        let repeats = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
        lindh_scale(&out, repeats);
        return;
    }
    if first.as_deref() == Some("--beat") {
        let out = args
            .next()
            .unwrap_or_else(|| "bench/raw/beat.csv".to_string());
        let repeats = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
        let only = args.next();
        beat(&out, repeats, only.as_deref());
        return;
    }
    let out = first.unwrap_or_else(|| "bench/raw/algo.csv".to_string());
    let steps_path = args
        .next()
        .unwrap_or_else(|| "bench/raw/steps.csv".to_string());
    let only = args.next();
    let problems = problems();
    for p in &problems {
        check_force(p);
        eprintln!(
            "{} atoms {} energy/atom {:.6} |F|0 {:.4e}",
            p.name,
            p.x0.len() / 3,
            energy_of(p) / (p.x0.len() / 3) as f64,
            initial_force(p)
        );
    }
    let mut rows = String::from("problem,atoms,split,arm,evals,gnorm,energy,converged\n");
    let mut steps = String::from(
        "problem,curvature,fnorm,fnorm_prev,cos_angle,cap_ratio,atom_mean,atom_max,step_len,reset\n",
    );
    for p in &problems {
        if only.as_ref().is_some_and(|s| !p.name.contains(s.as_str())) {
            continue;
        }
        let atoms = p.x0.len() / 3;
        // The dense neighbour matrix of the largest slab is left to the
        // unpreconditioned arms. The exponential solve is timed where
        // the neighbour list stays modest.
        let skip_exp = atoms > 1_600;
        for arm in Arm::all() {
            if skip_exp && matches!(arm, Arm::Exp | Arm::FireExp) {
                continue;
            }
            let (evals, gnorm, energy, converged, log) = run_arm(p, *arm);
            eprintln!(
                "{:16} {:16} evals {:5} |F| {:.3e} conv {converged}",
                p.name,
                arm.name(),
                evals,
                gnorm
            );
            rows.push_str(&format!(
                "{},{},{},{},{},{:.6e},{:.6e},{}\n",
                p.name,
                atoms,
                p.split,
                arm.name(),
                evals,
                gnorm,
                energy,
                converged as u8
            ));
            if matches!(arm, Arm::Eon) && (p.name == "lj75" || p.name == "lj150") {
                let mut prev = 0.0;
                for f in log {
                    steps.push_str(&format!(
                        "{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{}\n",
                        p.name,
                        f.curvature,
                        f.fnorm,
                        prev,
                        f.cos_angle,
                        f.cap_ratio,
                        f.atom_mean,
                        f.atom_max,
                        f.step_len,
                        f.reset
                    ));
                    prev = f.fnorm;
                }
            }
        }
    }
    std::fs::write(&out, rows).expect("algo csv");
    std::fs::write(&steps_path, steps).expect("step log");
    eprintln!("wrote {out}");
}

fn run_arm(p: &Problem, arm: Arm) -> (usize, f64, f64, bool, Vec<EonFeat>) {
    let n = p.x0.len();
    let istep = match arm {
        Arm::Fire2 | Arm::FireAbc => 0.1,
        _ => 1.0,
    };
    let control = Control {
        maxiter: p.budget,
        gtol: p.tol,
        istep,
        stop_norm: StopNorm::MaxAtom,
        ..Control::default()
    };
    let method = match arm {
        Arm::Fire | Arm::FireGlobal | Arm::FireExp => Method::Fire { kind: FireKind::V1 },
        Arm::Fire2 | Arm::FireAbc => Method::Fire { kind: FireKind::V2 },
        Arm::Bb | Arm::BbNonmonotone => Method::Bb,
        _ => Method::Lbfgs { memory: 20 },
    };
    let mut solver = Solver::new(method, control, n);
    match arm {
        Arm::FireGlobal => solver.set_maxmove(CAP),
        Arm::Bb | Arm::BbNonmonotone => solver.set_atom_maxmove(CAP),
        Arm::Scaled => solver
            .set_lbfgs_step(LbfgsStep::Scaled { max_atom: CAP })
            .unwrap(),
        _ => solver.set_atom_maxmove(CAP),
    }
    if matches!(arm, Arm::FixedH0) {
        // ASE's fixed initial scale, H0 = I/70.
        solver.set_lbfgs_fixed_h0(Some(1.0 / 70.0)).unwrap();
    }
    if matches!(arm, Arm::Euclid) {
        solver.set_lbfgs_euclidean_cap(true).unwrap();
    }
    if matches!(arm, Arm::Trust) {
        solver.set_lbfgs_trust(true).unwrap();
    }
    if matches!(arm, Arm::FireAbc) {
        solver.set_fire_abc(true).unwrap();
    }
    if matches!(arm, Arm::Bb) {
        solver.set_accept(Accept::Energy);
    }
    if matches!(arm, Arm::BbNonmonotone) {
        solver.set_accept(Accept::Nonmonotone);
    }
    if matches!(arm, Arm::Exp | Arm::FireExp) {
        let nbr = neighbours(&p.x0, p.rc, p.cell);
        let params = ExpParams {
            mu: None,
            r_cut: Some(p.rc),
            ..ExpParams::default()
        };
        solver.set_exp(p.x0.view(), &nbr, &params).unwrap();
    }
    let log_steps = matches!(arm, Arm::Eon) && (p.name == "lj75" || p.name == "lj150");
    if log_steps {
        solver.start_eon_log();
    }
    let calls = AtomicUsize::new(0);
    let kind = p.kind;
    let cell = p.cell;
    let mut x = p.x0.clone();
    let use_fill = matches!(
        arm,
        Arm::Eon
            | Arm::Fire
            | Arm::FireGlobal
            | Arm::Fire2
            | Arm::Euclid
            | Arm::Trust
            | Arm::FixedH0
    );
    let report = if use_fill {
        solver.run_fill(&mut x, |v, g| {
            calls.fetch_add(1, Ordering::Relaxed);
            potential(kind, cell, v, g.as_slice_mut().unwrap())
        })
    } else {
        let oracle = Oracle::unbounded(n, |v: ArrayView1<f64>| {
            calls.fetch_add(1, Ordering::Relaxed);
            let mut g = vec![0.0; v.len()];
            let e = potential(kind, cell, v, &mut g);
            (e, Array1::from(g))
        });
        solver.run(&oracle, &mut x)
    };
    let evals = calls.load(Ordering::Relaxed);
    let log = if log_steps {
        solver.take_eon_log()
    } else {
        Vec::new()
    };
    match report {
        Ok(rep) => (evals, rep.grad_norm, rep.value, rep.grad_norm < p.tol, log),
        Err(_) => (evals, f64::INFINITY, f64::NAN, false, log),
    }
}

fn potential(kind: Kind, cell: [f64; 3], x: ArrayView1<f64>, g: &mut [f64]) -> f64 {
    match kind {
        Kind::Lj => lennard_jones(x, g),
        Kind::Morse => morse(cell, x, g),
        Kind::Gupta(gp) => gupta(gp, cell, x, g),
        Kind::Water => water(x, g),
    }
}

fn energy_of(p: &Problem) -> f64 {
    let mut g = vec![0.0; p.x0.len()];
    potential(p.kind, p.cell, p.x0.view(), &mut g)
}

fn initial_force(p: &Problem) -> f64 {
    let mut g = vec![0.0; p.x0.len()];
    let _ = potential(p.kind, p.cell, p.x0.view(), &mut g);
    let mut worst = 0.0_f64;
    let mut i = 0;
    while i + 3 <= g.len() {
        let r = (g[i] * g[i] + g[i + 1] * g[i + 1] + g[i + 2] * g[i + 2]).sqrt();
        worst = worst.max(r);
        i += 3;
    }
    worst
}

fn check_force(p: &Problem) {
    let x = &p.x0;
    let mut g = vec![0.0; x.len()];
    let e0 = potential(p.kind, p.cell, x.view(), &mut g);
    let eps = 1.0e-6;
    let mut worst = 0.0_f64;
    let ncheck = x.len().min(12);
    for i in 0..ncheck {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let mut dump = vec![0.0; x.len()];
        let ep = potential(p.kind, p.cell, xp.view(), &mut dump);
        let em = potential(p.kind, p.cell, xm.view(), &mut dump);
        let num = (ep - em) / (2.0 * eps);
        worst = worst.max((num - g[i]).abs());
        let _ = e0;
    }
    if worst > 1.0e-3 {
        panic!(
            "{} force disagrees with a central difference by {worst}",
            p.name
        );
    }
}

fn lennard_jones(x: ArrayView1<f64>, g: &mut [f64]) -> f64 {
    let n = x.len() / 3;
    let mut e = 0.0;
    g.fill(0.0);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 < 1.0e-16 {
                continue;
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

/// Morse platinum. The well is the one in `eon_ref.cpp`: D = 0.7102 eV,
/// a = 1.6047 Å⁻¹, r0 = 2.897 Å, cutoff 7 Å.
fn morse(cell: [f64; 3], x: ArrayView1<f64>, g: &mut [f64]) -> f64 {
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

/// Cleri and Rosato, Phys. Rev. B 48, 22 (1993). The pair and density
/// terms are cut with a cosine from the fourth neighbour to the fifth.
fn gupta(gp: Gupta, cell: [f64; 3], x: ArrayView1<f64>, g: &mut [f64]) -> f64 {
    let n = x.len() / 3;
    let mut rho = vec![0.0; n];
    let mut e = 0.0;
    g.fill(0.0);
    let mut pairs: Vec<(usize, usize, f64, f64, [f64; 3])> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            for a in 0..3 {
                d[a] = wrap(x[3 * i + a] - x[3 * j + a], cell[a]);
            }
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r >= gp.rc || r < 1.0e-8 {
                continue;
            }
            let (fc, dfc) = cosine_cut(r, gp.r_on, gp.rc);
            let er = (gp.a * (-gp.p * (r / gp.r0 - 1.0)).exp()) * fc;
            let der = gp.a * (-gp.p * (r / gp.r0 - 1.0)).exp() * (-gp.p / gp.r0 * fc + dfc);
            let ed = (gp.xi * gp.xi * (-2.0 * gp.q * (r / gp.r0 - 1.0)).exp()) * fc;
            let ded = gp.xi
                * gp.xi
                * (-2.0 * gp.q * (r / gp.r0 - 1.0)).exp()
                * (-2.0 * gp.q / gp.r0 * fc + dfc);
            e += 2.0 * er;
            rho[i] += ed;
            rho[j] += ed;
            pairs.push((i, j, der, ded, d));
        }
    }
    let mut embed = vec![0.0; n];
    for i in 0..n {
        if rho[i] > 0.0 {
            let s = rho[i].sqrt();
            e -= s;
            embed[i] = -0.5 / s;
        }
    }
    for (i, j, der, ded, d) in pairs {
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let dedr = 2.0 * der + (embed[i] + embed[j]) * ded;
        for a in 0..3 {
            let c = dedr * d[a] / r;
            g[3 * i + a] += c;
            g[3 * j + a] -= c;
        }
    }
    e
}

fn cosine_cut(r: f64, r_on: f64, rc: f64) -> (f64, f64) {
    if r <= r_on {
        return (1.0, 0.0);
    }
    if r >= rc {
        return (0.0, 0.0);
    }
    let u = (r - r_on) / (rc - r_on);
    let pi = std::f64::consts::PI;
    let fc = 0.5 * (1.0 + (pi * u).cos());
    let dfc = -0.5 * pi / (rc - r_on) * (pi * u).sin();
    (fc, dfc)
}

/// Flexible TIP3P. Geometry and Lennard-Jones from Jorgensen et al.,
/// J. Chem. Phys. 79, 926 (1983). Bond and angle constants are the
/// LAMMPS `Howto_tip3p` flexible values. Charges use the real-unit
/// Coulomb prefactor. Intramolecular pairs carry the bond and the
/// angle only.
fn water(x: ArrayView1<f64>, g: &mut [f64]) -> f64 {
    const R0: f64 = 0.9572;
    const THETA0: f64 = 104.52 * std::f64::consts::PI / 180.0;
    const KB: f64 = 450.0;
    const KA: f64 = 55.0;
    const EPS: f64 = 0.1521;
    const SIG: f64 = 3.1507;
    const QO: f64 = -0.834;
    const QH: f64 = 0.417;
    const KE: f64 = 332.06371;
    let n = x.len() / 3;
    let mols = n / 3;
    g.fill(0.0);
    let mut e = 0.0;
    for m in 0..mols {
        let o = 3 * m;
        let h1 = o + 1;
        let h2 = o + 2;
        e += bond(x, g, o, h1, R0, KB);
        e += bond(x, g, o, h2, R0, KB);
        e += angle(x, g, o, h1, h2, THETA0, KA);
    }
    let sig2 = SIG * SIG;
    for a in 0..mols {
        for b in (a + 1)..mols {
            for ia in 0..3 {
                for ib in 0..3 {
                    let i = 3 * a + ia;
                    let j = 3 * b + ib;
                    let d = [
                        x[3 * i] - x[3 * j],
                        x[3 * i + 1] - x[3 * j + 1],
                        x[3 * i + 2] - x[3 * j + 2],
                    ];
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if r2 < 1.0e-16 {
                        continue;
                    }
                    let r = r2.sqrt();
                    let qi = if ia == 0 { QO } else { QH };
                    let qj = if ib == 0 { QO } else { QH };
                    let coul = KE * qi * qj / r;
                    e += coul;
                    let mut du = -coul / r;
                    if ia == 0 && ib == 0 {
                        let s6 = (sig2 / r2).powi(3);
                        e += 4.0 * EPS * (s6 * s6 - s6);
                        du += 24.0 * EPS * (s6 - 2.0 * s6 * s6) / r2 * r;
                    }
                    for c in 0..3 {
                        let t = du * d[c] / r;
                        g[3 * i + c] += t;
                        g[3 * j + c] -= t;
                    }
                }
            }
        }
    }
    e
}

fn bond(x: ArrayView1<f64>, g: &mut [f64], i: usize, j: usize, r0: f64, k: f64) -> f64 {
    let d = [
        x[3 * i] - x[3 * j],
        x[3 * i + 1] - x[3 * j + 1],
        x[3 * i + 2] - x[3 * j + 2],
    ];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if r < 1.0e-12 {
        return 0.0;
    }
    let dr = r - r0;
    let du = 2.0 * k * dr;
    for a in 0..3 {
        let t = du * d[a] / r;
        g[3 * i + a] += t;
        g[3 * j + a] -= t;
    }
    k * dr * dr
}

fn angle(
    x: ArrayView1<f64>,
    g: &mut [f64],
    o: usize,
    h1: usize,
    h2: usize,
    theta0: f64,
    k: f64,
) -> f64 {
    let u = [
        x[3 * h1] - x[3 * o],
        x[3 * h1 + 1] - x[3 * o + 1],
        x[3 * h1 + 2] - x[3 * o + 2],
    ];
    let v = [
        x[3 * h2] - x[3 * o],
        x[3 * h2 + 1] - x[3 * o + 1],
        x[3 * h2 + 2] - x[3 * o + 2],
    ];
    let lu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let lv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if lu < 1.0e-12 || lv < 1.0e-12 {
        return 0.0;
    }
    let c = ((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (lu * lv))
        .clamp(-1.0 + 1.0e-12, 1.0 - 1.0e-12);
    let theta = c.acos();
    let sin = (1.0 - c * c).sqrt();
    let factor = 2.0 * k * (theta - theta0) * (-1.0 / sin);
    let inv_lu = 1.0 / lu;
    let inv_lv = 1.0 / lv;
    for a in 0..3 {
        let dc_h1 = v[a] * inv_lu * inv_lv - c * u[a] * inv_lu * inv_lu;
        let dc_h2 = u[a] * inv_lu * inv_lv - c * v[a] * inv_lv * inv_lv;
        g[3 * h1 + a] += factor * dc_h1;
        g[3 * h2 + a] += factor * dc_h2;
        g[3 * o + a] += factor * (-dc_h1 - dc_h2);
    }
    k * (theta - theta0) * (theta - theta0)
}

fn wrap(d: f64, cell: f64) -> f64 {
    if cell > 0.0 {
        d - cell * (d / cell).round()
    } else {
        d
    }
}

fn neighbours(x: &Array1<f64>, rc: f64, cell: [f64; 3]) -> Neighbours {
    let n = x.len() / 3;
    let mut i = Vec::new();
    let mut j = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            let mut r2 = 0.0;
            for c in 0..3 {
                let d = wrap(x[3 * a + c] - x[3 * b + c], cell[c]);
                r2 += d * d;
            }
            if r2 < rc * rc && r2 > 1.0e-16 {
                i.push(a);
                j.push(b);
            }
        }
    }
    Neighbours {
        i,
        j,
        r: Vec::new(),
        anchor: Vec::new(),
    }
}

fn problems() -> Vec<Problem> {
    let mut out = vec![
        lj("lj75", 75, "train", 400),
        lj("lj98", 98, "train", 400),
        lj("lj150", 150, "heldout", 400),
        morse_slab("morse-pt-slab", 6, 6, 4, "train", 160),
    ];
    let pt = gupta_pt();
    let cu = gupta_cu();
    out.push(gupta_slab("gupta-pt-slab", pt, 8, 8, 4, "train", 120));
    out.push(gupta_slab("gupta-cu-slab", cu, 14, 14, 6, "heldout", 60));
    out.push(gupta_nano("gupta-pt-nano", pt, 3.4, "heldout", 160));
    out.push(gupta_nano("gupta-cu-nano", cu, 5.2, "heldout", 80));
    out.push(water_cluster("water16", 16, "heldout", 2_000));
    out
}

fn lj(name: &'static str, atoms: usize, split: &'static str, budget: usize) -> Problem {
    Problem {
        name,
        split,
        tol: 1.0e-2,
        budget,
        x0: grid(atoms, 1.1, 0.03),
        rc: 2.5,
        cell: [0.0; 3],
        kind: Kind::Lj,
    }
}

fn morse_slab(
    name: &'static str,
    nx: usize,
    ny: usize,
    nz: usize,
    split: &'static str,
    budget: usize,
) -> Problem {
    let a = 2.897 * std::f64::consts::SQRT_2;
    let (x0, cell) = fcc_slab(nx, ny, nz, a, 0.04);
    Problem {
        name,
        split,
        tol: 1.0e-2,
        budget,
        x0,
        rc: 6.5,
        cell,
        kind: Kind::Morse,
    }
}

fn gupta_pt() -> Gupta {
    // Cleri and Rosato, Table I, platinum.
    let r0 = 2.775;
    Gupta {
        a: 0.2975,
        xi: 2.695,
        p: 10.612,
        q: 4.004,
        r0,
        r_on: 2.0 * r0,
        rc: r0 * 5.0_f64.sqrt(),
    }
}

fn gupta_cu() -> Gupta {
    // Cleri and Rosato, Table I, copper.
    let r0 = 2.556;
    Gupta {
        a: 0.0855,
        xi: 1.224,
        p: 10.960,
        q: 2.278,
        r0,
        r_on: 2.0 * r0,
        rc: r0 * 5.0_f64.sqrt(),
    }
}

fn gupta_slab(
    name: &'static str,
    gp: Gupta,
    nx: usize,
    ny: usize,
    nz: usize,
    split: &'static str,
    budget: usize,
) -> Problem {
    let a = gp.r0 * std::f64::consts::SQRT_2;
    let (x0, cell) = fcc_slab(nx, ny, nz, a, 0.04);
    let rc = gp.rc;
    Problem {
        name,
        split,
        tol: 1.0e-2,
        budget,
        x0,
        rc,
        cell,
        kind: Kind::Gupta(gp),
    }
}

fn gupta_nano(
    name: &'static str,
    gp: Gupta,
    radius_over_a: f64,
    split: &'static str,
    budget: usize,
) -> Problem {
    let a = gp.r0 * std::f64::consts::SQRT_2;
    let x0 = fcc_ball(a, radius_over_a * a, 0.03);
    let rc = gp.rc;
    Problem {
        name,
        split,
        tol: 1.0e-2,
        budget,
        x0,
        rc,
        cell: [0.0; 3],
        kind: Kind::Gupta(gp),
    }
}

fn water_cluster(name: &'static str, mols: usize, split: &'static str, budget: usize) -> Problem {
    Problem {
        name,
        split,
        tol: 5.0e-2,
        budget,
        x0: water_start(mols),
        rc: 5.0,
        cell: [0.0; 3],
        kind: Kind::Water,
    }
}

fn grid(atoms: usize, spacing: f64, jitter: f64) -> Array1<f64> {
    let side = (atoms as f64).cbrt().ceil() as usize;
    let mut x = Vec::with_capacity(3 * atoms);
    for k in 0..atoms {
        let cell = [k % side, (k / side) % side, k / (side * side)];
        for (a, c) in cell.iter().enumerate() {
            x.push(spacing * *c as f64 + jitter * ((3 * k + a) as f64).sin());
        }
    }
    Array1::from(x)
}

fn fcc_basis() -> [[f64; 3]; 4] {
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
}

fn fcc_slab(nx: usize, ny: usize, nz: usize, a: f64, rattle: f64) -> (Array1<f64>, [f64; 3]) {
    let mut x = Vec::new();
    let basis = fcc_basis();
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                for b in basis {
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
    (Array1::from(x), [nx as f64 * a, ny as f64 * a, 0.0])
}

fn fcc_ball(a: f64, radius: f64, rattle: f64) -> Array1<f64> {
    let n = (radius / a).ceil() as i32 + 1;
    let basis = fcc_basis();
    let mut raw = Vec::new();
    for iz in -n..=n {
        for iy in -n..=n {
            for ix in -n..=n {
                for b in basis {
                    let p = [
                        (ix as f64 + b[0]) * a,
                        (iy as f64 + b[1]) * a,
                        (iz as f64 + b[2]) * a,
                    ];
                    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    if r <= radius {
                        raw.push(p);
                    }
                }
            }
        }
    }
    let mut x = Vec::with_capacity(3 * raw.len());
    for (k, p) in raw.iter().enumerate() {
        for (c, v) in p.iter().enumerate() {
            x.push(v + rattle * ((k * 3 + c) as f64 * 1.3).sin());
        }
    }
    Array1::from(x)
}

fn water_start(mols: usize) -> Array1<f64> {
    let side = (mols as f64).cbrt().ceil() as usize;
    let r = 0.9572;
    let theta = 104.52 * std::f64::consts::PI / 180.0;
    let h1 = [r * 1.02, 0.01, 0.0];
    let h2 = [r * theta.cos(), r * theta.sin() * 0.98, 0.02];
    let mut x = Vec::with_capacity(9 * mols);
    for m in 0..mols {
        let ix = m % side;
        let iy = (m / side) % side;
        let iz = m / (side * side);
        let o = [
            3.4 * ix as f64 + 0.05 * (m as f64).sin(),
            3.4 * iy as f64,
            3.4 * iz as f64,
        ];
        for p in [
            o,
            [o[0] + h1[0], o[1] + h1[1], o[2] + h1[2]],
            [o[0] + h2[0], o[1] + h2[1], o[2] + h2[2]],
        ] {
            x.extend_from_slice(&p);
        }
    }
    Array1::from(x)
}

#[derive(Clone, Copy)]
struct BeatOpt {
    memory: usize,
    cap: f64,
    model_atom: bool,
    compact: bool,
    skip_bad: bool,
    lindh: bool,
    gate: bool,
    cg_atom: bool,
}

fn beat(out: &str, repeats: usize, only: Option<&str>) {
    let problems: Vec<_> = problems()
        .into_iter()
        .filter(|p| only.is_none_or(|s| p.name.contains(s)))
        .collect();
    for p in &problems {
        check_force(p);
    }
    let mut rows =
        String::from("problem,atoms,split,arm,repeat,evals,energy,gnorm,millis,converged\n");
    let mut spreads = String::from("problem,c_force,c_bond,spread,gate_open,probe_evals\n");
    let arms = [
        ("lbfgs-eon", eon_opt()),
        ("cg-atom", cg_opt()),
        ("lbfgs-model", model_opt()),
        ("lbfgs-gate", gate_opt()),
        ("lbfgs-lindh", lindh_opt(false)),
        ("lbfgs-lindh-atom", lindh_opt(true)),
        ("lbfgs-compact", compact_opt()),
        ("lbfgs-skip", skip_opt()),
    ];
    for p in &problems {
        let (cf, cb, probes) = rayleigh(p);
        let spread = if cf > 0.0 && cb > 0.0 {
            cf.max(cb) / cf.min(cb)
        } else {
            f64::INFINITY
        };
        let open = exp_gate_open(cf, cb, rgmin::precon::EXP_GATE_MAX_SPREAD);
        spreads.push_str(&format!(
            "{},{:.6e},{:.6e},{:.6e},{},{}\n",
            p.name, cf, cb, spread, open as u8, probes
        ));
        eprintln!(
            "{:16} rayleigh force {:.3e} bond {:.3e} spread {:.3} gate {open}",
            p.name, cf, cb, spread
        );
        for (name, opt) in arms {
            for rep in 0..repeats {
                let t0 = Instant::now();
                let (evals, gnorm, energy, converged, _) = run_beat(p, opt, open);
                let millis = t0.elapsed().as_secs_f64() * 1.0e3;
                eprintln!(
                    "{:16} {:16} r{rep} evals {evals:5} |F| {gnorm:.3e} {millis:.2} ms conv {converged}",
                    p.name, name
                );
                push_row(
                    &mut rows,
                    BeatRow {
                        problem: p,
                        arm: name,
                        rep,
                        evals,
                        energy,
                        gnorm,
                        millis,
                        converged,
                    },
                );
            }
        }
    }
    let mut train = Vec::new();
    for p in problems.iter().filter(|p| p.split == "train") {
        for &memory in &[8usize, 12, 20, 30] {
            for &cap in &[0.10_f64, 0.15, 0.20, 0.30, 0.40] {
                let mut opt = eon_opt();
                opt.memory = memory;
                opt.cap = cap;
                let name = format!("grid-m{memory}-c{cap:.2}");
                let t0 = Instant::now();
                let (evals, gnorm, energy, converged, _) = run_beat(p, opt, false);
                let millis = t0.elapsed().as_secs_f64() * 1.0e3;
                eprintln!(
                    "{:<16} {:<16} evals {evals:5} conv {converged}",
                    p.name, name
                );
                push_row(
                    &mut rows,
                    BeatRow {
                        problem: p,
                        arm: &name,
                        rep: 0,
                        evals,
                        energy,
                        gnorm,
                        millis,
                        converged,
                    },
                );
                train.push(TrainRow {
                    memory,
                    cap,
                    evals,
                    converged,
                    problem: p.name.to_string(),
                });
            }
        }
    }
    if let Some((memory, cap)) = select_train_setting(&train) {
        eprintln!("train setting memory {memory} cap {cap}");
        let same = memory == 20 && (cap - 0.2).abs() < 1e-12;
        if !same {
            for p in &problems {
                let mut opt = eon_opt();
                opt.memory = memory;
                opt.cap = cap;
                for rep in 0..repeats {
                    let t0 = Instant::now();
                    let (evals, gnorm, energy, converged, _) = run_beat(p, opt, false);
                    let millis = t0.elapsed().as_secs_f64() * 1.0e3;
                    push_row(
                        &mut rows,
                        BeatRow {
                            problem: p,
                            arm: "lbfgs-tuned",
                            rep,
                            evals,
                            energy,
                            gnorm,
                            millis,
                            converged,
                        },
                    );
                    eprintln!(
                        "{:16} lbfgs-tuned r{rep} evals {evals} {millis:.2} ms conv {converged}",
                        p.name
                    );
                }
            }
        }
    } else {
        eprintln!("no training setting converged on every training problem");
    }
    std::fs::write(out, &rows).expect("beat csv");
    let spread_path = if out.ends_with("beat.csv") {
        out.replace("beat.csv", "spread.csv")
    } else {
        format!("{out}.spread.csv")
    };
    std::fs::write(&spread_path, spreads).expect("spread csv");
    eprintln!("wrote {out} and {spread_path}");
}

fn eon_opt() -> BeatOpt {
    BeatOpt {
        memory: 20,
        cap: CAP,
        model_atom: false,
        compact: false,
        skip_bad: false,
        lindh: false,
        gate: false,
        cg_atom: false,
    }
}

fn cg_opt() -> BeatOpt {
    BeatOpt {
        cg_atom: true,
        ..eon_opt()
    }
}

fn model_opt() -> BeatOpt {
    BeatOpt {
        model_atom: true,
        ..eon_opt()
    }
}

fn gate_opt() -> BeatOpt {
    BeatOpt {
        gate: true,
        ..eon_opt()
    }
}

fn lindh_opt(model_atom: bool) -> BeatOpt {
    BeatOpt {
        lindh: true,
        model_atom,
        ..eon_opt()
    }
}

fn compact_opt() -> BeatOpt {
    BeatOpt {
        compact: true,
        ..eon_opt()
    }
}

fn skip_opt() -> BeatOpt {
    BeatOpt {
        skip_bad: true,
        ..eon_opt()
    }
}

struct BeatRow<'a> {
    problem: &'a Problem,
    arm: &'a str,
    rep: usize,
    evals: usize,
    energy: f64,
    gnorm: f64,
    millis: f64,
    converged: bool,
}

fn push_row(rows: &mut String, row: BeatRow<'_>) {
    let p = row.problem;
    rows.push_str(&format!(
        "{},{},{},{},{},{},{:.6e},{:.6e},{:.4},{}\n",
        p.name,
        p.x0.len() / 3,
        p.split,
        row.arm,
        row.rep,
        row.evals,
        row.energy,
        row.gnorm,
        row.millis,
        row.converged as u8
    ));
}

fn run_beat(p: &Problem, opt: BeatOpt, gate_open: bool) -> (usize, f64, f64, bool, f64) {
    if opt.cg_atom {
        let (evals, gnorm, energy, converged, force_ms) = run_cg_atom(p);
        return (evals, gnorm, energy, converged, force_ms);
    }
    let mut probes = 0usize;
    let install_exp = opt.gate && gate_open && p.x0.len() / 3 <= 1_600;
    if opt.gate {
        let (_, _, n) = rayleigh(p);
        probes = n;
    }
    let n = p.x0.len();
    let control = Control {
        maxiter: p.budget,
        gtol: p.tol,
        istep: 1.0,
        stop_norm: StopNorm::MaxAtom,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: opt.memory }, control, n);
    solver.set_atom_maxmove(opt.cap);
    if opt.model_atom {
        solver.set_lbfgs_model_atom(true).unwrap();
    }
    if opt.compact {
        solver.set_lbfgs_compact_tr(true).unwrap();
    }
    if opt.skip_bad {
        solver.set_lbfgs_skip_bad(true).unwrap();
    }
    if opt.lindh {
        solver
            .set_lindh_model(Some(LindhSpec {
                z: atomic_numbers(p),
                c_stab: 0.1,
                cell: p.cell,
            }))
            .unwrap();
    }
    if install_exp {
        let nbr = neighbours(&p.x0, p.rc, p.cell);
        let params = ExpParams {
            mu: None,
            r_cut: Some(p.rc),
            ..ExpParams::default()
        };
        solver.set_exp(p.x0.view(), &nbr, &params).unwrap();
    }
    let calls = AtomicUsize::new(0);
    let force_ns = std::sync::atomic::AtomicU64::new(0);
    let kind = p.kind;
    let cell = p.cell;
    let mut x = p.x0.clone();
    let report = if install_exp {
        let oracle = Oracle::unbounded(n, |v: ArrayView1<f64>| {
            calls.fetch_add(1, Ordering::Relaxed);
            let mut g = vec![0.0; v.len()];
            let t0 = Instant::now();
            let e = potential(kind, cell, v, &mut g);
            force_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            (e, Array1::from(g))
        });
        solver.run(&oracle, &mut x)
    } else {
        solver.run_fill(&mut x, |v, g| {
            calls.fetch_add(1, Ordering::Relaxed);
            let t0 = Instant::now();
            let e = potential(kind, cell, v, g.as_slice_mut().unwrap());
            force_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            e
        })
    };
    let evals = calls.load(Ordering::Relaxed) + probes;
    let force_ms = force_ns.load(Ordering::Relaxed) as f64 / 1.0e6;
    match report {
        Ok(rep) => (
            evals,
            rep.grad_norm,
            rep.value,
            rep.grad_norm < p.tol,
            force_ms,
        ),
        Err(_) => (evals, f64::INFINITY, f64::NAN, false, force_ms),
    }
}

fn run_cg_atom(p: &Problem) -> (usize, f64, f64, bool, f64) {
    let calls = AtomicUsize::new(0);
    let kind = p.kind;
    let cell = p.cell;
    let mut raw = p.x0.to_vec();
    let opts = EonCg {
        max_move: CAP,
        gtol: p.tol,
        max_iter: p.budget,
        cluster: true,
        atom_only: true,
    };
    let force_ns = std::sync::atomic::AtomicU64::new(0);
    let rep = minimize_eon_cg(
        &mut raw,
        |xs, gs| {
            calls.fetch_add(1, Ordering::Relaxed);
            let t0 = Instant::now();
            let e = potential(kind, cell, ArrayView1::from(xs), gs);
            force_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            e
        },
        &opts,
    );
    (
        calls.load(Ordering::Relaxed),
        rep.grad,
        rep.energy,
        rep.converged,
        force_ns.load(Ordering::Relaxed) as f64 / 1.0e6,
    )
}

/// Force-direction and shortest-bond Rayleigh quotients.
/// Three evaluations: the point, the force probe, the bond probe.
fn rayleigh(p: &Problem) -> (f64, f64, usize) {
    let n = p.x0.len();
    let mut g = vec![0.0; n];
    let _ = potential(p.kind, p.cell, p.x0.view(), &mut g);
    let mut n2 = 0.0;
    for v in &g {
        n2 += v * v;
    }
    let norm = n2.sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return (f64::NAN, f64::NAN, 1);
    }
    let fd = rgmin::EON_FINITE_DIFFERENCE;
    let c_force = directional(p, &g, &force_unit(&g, norm), fd);
    let c_bond = directional(p, &g, &bond_unit(&p.x0, p.cell), fd);
    (c_force, c_bond, 3)
}

fn force_unit(g: &[f64], norm: f64) -> Vec<f64> {
    g.iter().map(|v| -v / norm).collect()
}

fn directional(p: &Problem, g: &[f64], u: &[f64], fd: f64) -> f64 {
    let mut probe = p.x0.to_vec();
    for i in 0..probe.len() {
        probe[i] += fd * u[i];
    }
    let mut gp = vec![0.0; g.len()];
    let _ = potential(p.kind, p.cell, ArrayView1::from(&probe), &mut gp);
    let mut curv = 0.0;
    for i in 0..g.len() {
        curv += (gp[i] - g[i]) * u[i];
    }
    curv / fd
}

fn bond_unit(x: &Array1<f64>, cell: [f64; 3]) -> Vec<f64> {
    let n = x.len() / 3;
    let mut best = f64::MAX;
    let mut which = (0usize, 1usize);
    let mut vec = [1.0, 0.0, 0.0];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            for a in 0..3 {
                d[a] = wrap(x[3 * i + a] - x[3 * j + a], cell[a]);
            }
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r < best && r > 1.0e-8 {
                best = r;
                which = (i, j);
                vec = d;
            }
        }
    }
    let mut u = vec![0.0; x.len()];
    if best.is_finite() && best > 0.0 && best < f64::MAX / 2.0 {
        let scale = 1.0 / (best * std::f64::consts::SQRT_2);
        for a in 0..3 {
            u[3 * which.0 + a] = vec[a] * scale;
            u[3 * which.1 + a] = -vec[a] * scale;
        }
    }
    u
}

fn atomic_numbers(p: &Problem) -> Vec<u8> {
    let n = p.x0.len() / 3;
    match p.kind {
        Kind::Lj => Vec::new(),
        Kind::Morse => vec![78; n],
        Kind::Gupta(gp) => {
            let z = if (gp.r0 - 2.775).abs() < 1.0e-6 {
                78
            } else {
                29
            };
            vec![z; n]
        }
        Kind::Water => {
            let mut z = Vec::with_capacity(n);
            for _ in 0..(n / 3) {
                z.extend_from_slice(&[8, 1, 1]);
            }
            z
        }
    }
}

fn lindh_scale(out: &str, repeats: usize) {
    let problems = problems();
    for p in &problems {
        check_force(p);
    }
    let mut rows = String::from(
        "problem,atoms,split,arm,repeat,evals,energy,gnorm,millis,force_millis,overhead_millis,effective_s,converged\n",
    );
    let arms = [
        ("lbfgs-eon", eon_opt()),
        ("lbfgs-lindh", lindh_opt(false)),
        ("lbfgs-lindh-atom", lindh_opt(true)),
    ];
    for p in &problems {
        for (name, opt) in arms {
            for rep in 0..repeats {
                let t0 = Instant::now();
                let (evals, gnorm, energy, converged, force_ms) = run_beat(p, opt, false);
                let millis = t0.elapsed().as_secs_f64() * 1.0e3;
                let overhead = (millis - force_ms).max(0.0);
                let effective = evals as f64 + overhead / 1.0e3;
                eprintln!(
                    "{:16} {:16} r{rep} evals {evals:5} E {energy:.6e} |F| {gnorm:.3e} wall {millis:.2} force {force_ms:.2} over {overhead:.2} eff {effective:.3} conv {converged}",
                    p.name, name
                );
                rows.push_str(&format!(
                    "{},{},{},{},{},{},{:.6e},{:.6e},{:.4},{:.4},{:.4},{:.6},{}\n",
                    p.name,
                    p.x0.len() / 3,
                    p.split,
                    name,
                    rep,
                    evals,
                    energy,
                    gnorm,
                    millis,
                    force_ms,
                    overhead,
                    effective,
                    converged as u8
                ));
            }
        }
    }
    let basin_path = out.replace("lindh-scale.csv", "lindh-basin.csv");
    let basin = basin_log(problems.iter().find(|p| p.name == "lj150"));
    std::fs::write(&basin_path, basin).expect("basin csv");
    let over_path = out.replace("lindh-scale.csv", "lindh-overhead.csv");
    let over = overhead_table();
    std::fs::write(&over_path, over).expect("overhead csv");
    std::fs::write(out, rows).expect("lindh csv");
    eprintln!("wrote {out}");
}

fn overhead_only() {
    let text = overhead_table();
    eprintln!("{text}");
    std::fs::write("bench/raw/lindh-overhead.csv", text).expect("overhead csv");
}

fn overhead_table() -> String {
    let mut rows = String::from("system,atoms,what,repeat,millis\n");
    let problems = problems();
    if let Some(p) = problems.iter().find(|p| p.name == "gupta-cu-slab") {
        time_force(&mut rows, "gupta-cu-slab", p, 5);
        time_apply(
            &mut rows,
            "gupta-cu-slab",
            &p.x0,
            &atomic_numbers(p),
            p.cell,
            5,
        );
    }
    let side = 47usize;
    let n = side * side * side;
    let mut x = Vec::with_capacity(3 * n);
    for iz in 0..side {
        for iy in 0..side {
            for ix in 0..side {
                x.push(1.2 * ix as f64);
                x.push(1.2 * iy as f64);
                x.push(1.2 * iz as f64);
            }
        }
    }
    let cube = Array1::from(x);
    time_lj(&mut rows, n, &cube, 1);
    time_apply(&mut rows, "lj-cube", &cube, &[], [0.0; 3], 5);
    rows
}

fn time_force(rows: &mut String, name: &str, p: &Problem, repeats: usize) {
    for rep in 0..repeats {
        let mut g = vec![0.0; p.x0.len()];
        let t0 = Instant::now();
        let _ = potential(p.kind, p.cell, p.x0.view(), &mut g);
        let millis = t0.elapsed().as_secs_f64() * 1.0e3;
        eprintln!("{name} force r{rep} {millis:.3} ms");
        rows.push_str(&format!(
            "{name},{},force,{rep},{millis:.4}\n",
            p.x0.len() / 3
        ));
    }
}

fn time_lj(rows: &mut String, n: usize, x: &Array1<f64>, repeats: usize) {
    for rep in 0..repeats {
        let mut g = vec![0.0; x.len()];
        let t0 = Instant::now();
        let _ = lennard_jones(x.view(), &mut g);
        let millis = t0.elapsed().as_secs_f64() * 1.0e3;
        eprintln!("lj-cube force r{rep} {millis:.3} ms atoms {n}");
        rows.push_str(&format!("lj-cube,{n},force,{rep},{millis:.4}\n"));
    }
}

fn time_apply(
    rows: &mut String,
    name: &str,
    x: &Array1<f64>,
    z: &[u8],
    cell: [f64; 3],
    repeats: usize,
) {
    let mut rhs = vec![0.0; x.len()];
    for (i, v) in rhs.iter_mut().enumerate() {
        *v = ((i % 7) as f64 * 0.17).sin();
    }
    for rep in 0..repeats {
        let mut trial = rhs.clone();
        let t0 = Instant::now();
        let ok = apply_inverse(x.as_slice().unwrap(), z, 0.1, cell, 1.0, &mut trial);
        let millis = t0.elapsed().as_secs_f64() * 1.0e3;
        eprintln!("{name} apply r{rep} {millis:.3} ms ok {ok}");
        rows.push_str(&format!(
            "{name},{},lindh-apply,{rep},{millis:.4}\n",
            x.len() / 3
        ));
    }
}

/// One pass of LJ150. Each accepted step is checked for a downhill
/// slope and the per-atom cap. The energies are the basin record.
fn basin_log(problem: Option<&Problem>) -> String {
    let mut rows = String::from("problem,arm,step,energy,slope,max_atom,descent,within_cap\n");
    let Some(p) = problem else {
        return rows;
    };
    for (name, opt) in [
        ("lbfgs-eon", eon_opt()),
        ("lbfgs-lindh-atom", lindh_opt(true)),
    ] {
        let n = p.x0.len();
        let control = Control {
            maxiter: p.budget,
            gtol: p.tol,
            istep: 1.0,
            stop_norm: StopNorm::MaxAtom,
            ..Control::default()
        };
        let mut solver = Solver::new(Method::Lbfgs { memory: opt.memory }, control, n);
        solver.set_atom_maxmove(opt.cap);
        if opt.model_atom {
            solver.set_lbfgs_model_atom(true).unwrap();
        }
        if opt.lindh {
            solver
                .set_lindh_model(Some(LindhSpec {
                    z: atomic_numbers(p),
                    c_stab: 0.1,
                    cell: p.cell,
                }))
                .unwrap();
        }
        let oracle = Oracle::unbounded(n, |v: ArrayView1<f64>| {
            let mut g = vec![0.0; v.len()];
            let e = potential(p.kind, p.cell, v, &mut g);
            (e, Array1::from(g))
        });
        let mut x = p.x0.clone();
        let mut g = vec![0.0; n];
        let mut energy = potential(p.kind, p.cell, x.view(), &mut g);
        for step in 0..p.budget {
            let gnorm = max_atom_of(&g);
            if gnorm < p.tol {
                break;
            }
            let x0 = x.clone();
            let g0 = g.clone();
            let report = solver.step(&oracle, &mut x);
            let Ok(rep) = report else {
                rows.push_str(&format!(
                    "{},{name},{step},{energy:.6e},nan,nan,0,0\n",
                    p.name
                ));
                break;
            };
            let mut slope = 0.0_f64;
            let mut max_a = 0.0_f64;
            let mut i = 0;
            while i < n {
                let end = (i + 3).min(n);
                let mut a2 = 0.0;
                for k in i..end {
                    let d = x[k] - x0[k];
                    slope += g0[k] * d;
                    a2 += d * d;
                }
                max_a = max_a.max(a2.sqrt());
                i = end;
            }
            energy = rep.value;
            if let Some(grad) = &rep.grad {
                g = grad.to_vec();
            }
            let descent = slope < 0.0;
            let within = max_a <= opt.cap + 1.0e-8;
            rows.push_str(&format!(
                "{},{name},{step},{energy:.6e},{slope:.6e},{max_a:.6e},{},{}\n",
                p.name, descent as u8, within as u8
            ));
            if !descent || !within {
                eprintln!("{name} step {step} slope {slope:.3e} max_atom {max_a:.3e} UNSAFE");
            }
        }
        eprintln!("{name} basin energy {energy:.6e} |F| {}", max_atom_of(&g));
    }
    rows
}

fn max_atom_of(g: &[f64]) -> f64 {
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
