//! Morse bulk dogleg, and the per-atom cap that froze its trust radius.
//!
//! Optbench `minimization/morse-bulk` (256 atoms, 768 coordinates, force
//! stop 1e-3 eV/Å, 10000 force calls) is the regression. rgmin 0.4.1
//! commit e34b86bd installed a per-atom displacement cap of 0.2 and
//! `step_dogleg` used that length as the Euclidean trust ceiling. Nocedal
//! and Wright, algorithm 4.1, grow the radius only when the trust step
//! meets the boundary, so the clipped length left the radius near the
//! initial 0.1. The potential matches optbench's shifted Morse
//! (`D0` 0.7102, `alpha` 1.6047, `r0` 2.897, cutoff 7).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{Control, HessianOracle, Method, NewtonKind, Solver};

const MORSE_D0: f64 = 0.7102;
const MORSE_ALPHA: f64 = 1.6047;
const MORSE_R0: f64 = 2.897;
const MORSE_CUTOFF: f64 = 7.0;

const LJ_EPSILON: f64 = 1.0;
const LJ_SIGMA: f64 = 1.0;
const LJ_CUTOFF: f64 = 40.0;

fn min_image(d: f64, length: f64) -> f64 {
    d - length * (d / length).round()
}

fn morse_pairs(pos: ArrayView1<f64>, box_l: [f64; 3]) -> Vec<(usize, usize, [f64; 3], f64)> {
    let n = pos.len() / 3;
    let cut2 = MORSE_CUTOFF * MORSE_CUTOFF;
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut diff = [0.0; 3];
            let mut r2 = 0.0;
            for a in 0..3 {
                let d = min_image(pos[3 * i + a] - pos[3 * j + a], box_l[a]);
                diff[a] = d;
                r2 += d * d;
            }
            if r2 > 0.0 && r2 < cut2 {
                pairs.push((i, j, diff, r2));
            }
        }
    }
    pairs
}

fn morse_value_grad(pos: ArrayView1<f64>, box_l: [f64; 3]) -> (f64, Array1<f64>) {
    let pairs = morse_pairs(pos, box_l);
    let rc = MORSE_CUTOFF;
    let xc = (-MORSE_ALPHA * (rc - MORSE_R0)).exp();
    let shift = MORSE_D0 * (xc * xc - 2.0 * xc);
    let mut energy = 0.0;
    let mut grad = Array1::<f64>::zeros(pos.len());
    for (i, j, diff, r2) in pairs {
        let r = r2.sqrt();
        let x = (-MORSE_ALPHA * (r - MORSE_R0)).exp();
        energy += MORSE_D0 * (x * x - 2.0 * x) - shift;
        let slope = (2.0 * MORSE_D0 * MORSE_ALPHA * x * (1.0 - x)) / r;
        for a in 0..3 {
            let piece = slope * diff[a];
            grad[3 * i + a] += piece;
            grad[3 * j + a] -= piece;
        }
    }
    (energy, grad)
}

fn morse_hessian(pos: ArrayView1<f64>, box_l: [f64; 3]) -> Array2<f64> {
    let dim = pos.len();
    let mut hessian = Array2::<f64>::zeros((dim, dim));
    let pairs = morse_pairs(pos, box_l);
    for (i, j, diff, r2) in pairs {
        let r = r2.sqrt();
        let x = (-MORSE_ALPHA * (r - MORSE_R0)).exp();
        let first = 2.0 * MORSE_D0 * MORSE_ALPHA * x * (1.0 - x);
        let second = 2.0 * MORSE_D0 * MORSE_ALPHA * MORSE_ALPHA * x * (2.0 * x - 1.0);
        let s = first / r;
        let outer_scale = (second - s) / r2;
        for a in 0..3 {
            for b in 0..3 {
                let mut block = -outer_scale * diff[a] * diff[b];
                if a == b {
                    block -= s;
                }
                let ia = 3 * i + a;
                let ib = 3 * i + b;
                let ja = 3 * j + a;
                let jb = 3 * j + b;
                hessian[(ia, jb)] += block;
                hessian[(jb, ia)] += block;
                hessian[(ia, ib)] -= block;
                hessian[(ja, jb)] -= block;
            }
        }
    }
    // The optbench scatter writes both triangles and then averages them.
    for i in 0..dim {
        for j in (i + 1)..dim {
            let m = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            hessian[(i, j)] = m;
            hessian[(j, i)] = m;
        }
    }
    hessian
}

fn lj_value_grad(pos: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = pos.len() / 3;
    let cut2 = LJ_CUTOFF * LJ_CUTOFF;
    let mut energy = 0.0;
    let mut grad = Array1::<f64>::zeros(pos.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let mut diff = [0.0; 3];
            let mut r2 = 0.0;
            for a in 0..3 {
                let d = pos[3 * i + a] - pos[3 * j + a];
                diff[a] = d;
                r2 += d * d;
            }
            if r2 <= 0.0 || r2 >= cut2 {
                continue;
            }
            let inv_r2 = 1.0 / r2;
            let sr2 = (LJ_SIGMA * LJ_SIGMA) * inv_r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            energy += 4.0 * LJ_EPSILON * (sr12 - sr6);
            let slope = 24.0 * LJ_EPSILON * (sr6 - 2.0 * sr12) * inv_r2;
            for a in 0..3 {
                let piece = slope * diff[a];
                grad[3 * i + a] += piece;
                grad[3 * j + a] -= piece;
            }
        }
    }
    (energy, grad)
}

fn lj_hessian(pos: ArrayView1<f64>) -> Array2<f64> {
    let n = pos.len() / 3;
    let dim = pos.len();
    let cut2 = LJ_CUTOFF * LJ_CUTOFF;
    let mut hessian = Array2::<f64>::zeros((dim, dim));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut diff = [0.0; 3];
            let mut r2 = 0.0;
            for a in 0..3 {
                let d = pos[3 * i + a] - pos[3 * j + a];
                diff[a] = d;
                r2 += d * d;
            }
            if r2 <= 0.0 || r2 >= cut2 {
                continue;
            }
            let inv_r2 = 1.0 / r2;
            let sr2 = (LJ_SIGMA * LJ_SIGMA) * inv_r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            let r = r2.sqrt();
            let first = 24.0 * LJ_EPSILON * (-2.0 * sr12 + sr6) / r;
            let second = 24.0 * LJ_EPSILON * (26.0 * sr12 - 7.0 * sr6) * inv_r2;
            let s = first / r;
            let outer_scale = (second - s) / r2;
            for a in 0..3 {
                for b in 0..3 {
                    let mut block = -outer_scale * diff[a] * diff[b];
                    if a == b {
                        block -= s;
                    }
                    let ia = 3 * i + a;
                    let ib = 3 * i + b;
                    let ja = 3 * j + a;
                    let jb = 3 * j + b;
                    hessian[(ia, jb)] += block;
                    hessian[(jb, ia)] += block;
                    hessian[(ia, ib)] -= block;
                    hessian[(ja, jb)] -= block;
                }
            }
        }
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            let m = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            hessian[(i, j)] = m;
            hessian[(j, i)] = m;
        }
    }
    hessian
}

fn read_con(text: &str) -> (Array1<f64>, [f64; 3]) {
    let lines: Vec<&str> = text.lines().collect();
    let mut box_l = None;
    for i in 0..lines.len().saturating_sub(1) {
        let parts: Vec<f64> = lines[i]
            .split_whitespace()
            .filter_map(|p| p.parse().ok())
            .collect();
        if parts.len() != 3 {
            continue;
        }
        let angles: Vec<f64> = lines[i + 1]
            .split_whitespace()
            .filter_map(|p| p.parse().ok())
            .collect();
        if angles.len() == 3 && angles.iter().all(|a| (a - 90.0).abs() < 1e-4) {
            box_l = Some([parts[0], parts[1], parts[2]]);
            break;
        }
    }
    let start = lines
        .iter()
        .position(|line| line.starts_with("Coordinates of Component"))
        .expect("con coordinates");
    let mut coords = Vec::new();
    for line in &lines[start + 1..] {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            break;
        }
        let (Ok(x), Ok(y), Ok(z)) = (
            parts[0].parse::<f64>(),
            parts[1].parse::<f64>(),
            parts[2].parse::<f64>(),
        ) else {
            break;
        };
        coords.extend([x, y, z]);
    }
    (Array1::from(coords), box_l.expect("orthorhombic box"))
}

fn load_fixture(name: &str) -> (Array1<f64>, [f64; 3]) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {path:?}: {err}"));
    read_con(&text)
}

fn fixture_start() -> (Array1<f64>, [f64; 3]) {
    load_fixture("morse_pos_0000.con")
}

fn max_atom_step(before: &Array1<f64>, after: &Array1<f64>) -> f64 {
    let n = before.len();
    let mut m = 0.0_f64;
    let mut i = 0;
    while i + 3 <= n {
        let dx = after[i] - before[i];
        let dy = after[i + 1] - before[i + 1];
        let dz = after[i + 2] - before[i + 2];
        m = m.max((dx * dx + dy * dy + dz * dz).sqrt());
        i += 3;
    }
    m
}

fn euclid(v: &Array1<f64>) -> f64 {
    v.iter().map(|t| t * t).sum::<f64>().sqrt()
}

struct Run {
    grad_calls: usize,
    hess_calls: usize,
    steps: usize,
    energy: f64,
    grad_norm: f64,
    max_euclid: f64,
    max_atom: f64,
}

fn control(gtol: f64, maxiter: usize) -> Control {
    Control {
        maxiter,
        gtol,
        istep: 0.1,
        maxmove: None,
    }
}

fn run_method(
    method: Method,
    start: &Array1<f64>,
    cap: Option<f64>,
    gtol: f64,
    step_budget: usize,
    fg: impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    hess: impl Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
) -> Run {
    let dim = start.len();
    let grad_calls = Arc::new(AtomicUsize::new(0));
    let hess_calls = Arc::new(AtomicUsize::new(0));
    let gcount = Arc::clone(&grad_calls);
    let hcount = Arc::clone(&hess_calls);
    let obj = HessianOracle::unbounded(
        dim,
        move |x| {
            gcount.fetch_add(1, Ordering::Relaxed);
            fg(x)
        },
        move |x| {
            hcount.fetch_add(1, Ordering::Relaxed);
            hess(x)
        },
    );
    let mut solver = Solver::new(method, control(gtol, step_budget.max(1)), dim);
    if let Some(cap) = cap {
        solver.set_atom_maxmove(cap);
    }
    let mut x = start.clone();
    let mut max_euclid = 0.0_f64;
    let mut max_atom = 0.0_f64;
    let mut last_norm = f64::MAX;
    let mut last_energy = 0.0;
    let mut steps = 0;
    for _ in 0..step_budget {
        let before = x.clone();
        let rep = solver.step_hess(&obj, &mut x).expect("dogleg step");
        let delta = &x - &before;
        max_euclid = max_euclid.max(euclid(&delta));
        max_atom = max_atom.max(max_atom_step(&before, &x));
        last_norm = rep.grad_norm;
        last_energy = rep.value;
        steps = rep.steps;
        if rep.grad_norm <= gtol {
            break;
        }
    }
    Run {
        grad_calls: grad_calls.load(Ordering::Relaxed),
        hess_calls: hess_calls.load(Ordering::Relaxed),
        steps,
        energy: last_energy,
        grad_norm: last_norm,
        max_euclid,
        max_atom,
    }
}

/// Thirty-two atoms, each 1 Å off the origin. A Euclidean ceiling of 0.2
/// cannot reach the bowl inside a short budget. The per-atom cap of 0.2
/// still limits every accepted move.
#[test]
fn per_atom_cap_does_not_freeze_the_dogleg_radius() {
    let n_atoms = 32;
    let dim = 3 * n_atoms;
    let mut start = Array1::zeros(dim);
    for i in 0..n_atoms {
        start[3 * i] = 1.0;
    }
    let quad_fg = |x: ArrayView1<f64>| {
        let f = 0.5 * x.iter().map(|v| v * v).sum::<f64>();
        let g = x.to_owned();
        (f, g)
    };
    let quad_h = |x: ArrayView1<f64>| Array2::<f64>::eye(x.len());
    let capped = run_method(Method::Dogleg, &start, Some(0.2), 1e-8, 16, quad_fg, quad_h);
    assert!(
        capped.grad_norm < 1e-8,
        "capped dogleg stalled at |g|={} after {} hessians (euclid step {})",
        capped.grad_norm,
        capped.hess_calls,
        capped.max_euclid
    );
    assert!(
        capped.max_euclid > 0.5,
        "trust steps never left the 0.2 Euclidean ball: {}",
        capped.max_euclid
    );
    assert!(
        capped.max_atom <= 0.2 + 1e-9,
        "per-atom cap was not applied: {}",
        capped.max_atom
    );
    let uncapped = run_method(Method::Dogleg, &start, None, 1e-8, 16, quad_fg, quad_h);
    assert!(
        uncapped.grad_norm < 1e-8,
        "uncapped dogleg |g|={}",
        uncapped.grad_norm
    );
}

#[test]
fn morse_pos_0000_matches_the_optbench_potential() {
    let (x, box_l) = fixture_start();
    assert_eq!(x.len(), 768);
    let (energy, grad) = morse_value_grad(x.view(), box_l);
    assert!(
        (energy - (-1392.9897402515905)).abs() < 1e-8,
        "energy {energy}"
    );
    let gnorm = euclid(&grad);
    assert!((gnorm - 10.657626334987627).abs() < 1e-6, "gnorm {gnorm}");
    // One central-difference column against the analytic Hessian.
    let h = morse_hessian(x.view(), box_l);
    let eps = 1e-6;
    let mut xp = x.clone();
    let mut xm = x.clone();
    xp[0] += eps;
    xm[0] -= eps;
    let gp = morse_value_grad(xp.view(), box_l).1;
    let gm = morse_value_grad(xm.view(), box_l).1;
    let mut worst = 0.0_f64;
    for k in 0..x.len() {
        let fd = (gp[k] - gm[k]) / (2.0 * eps);
        worst = worst.max((fd - h[(k, 0)]).abs());
    }
    assert!(worst < 1e-4, "hessian column mismatch {worst}");
}

/// Optbench start `pos_0000`, the Morse bulk row. Both the default
/// (no cap) and the tuned cap of 0.2 must clear 1e-3 inside the
/// 10000 force-call budget. 0.4.0 default did this in about 10
/// gradients and 9 Hessians. 0.4.1 and every tuned copy did not.
#[test]
fn morse_pos_0000_dogleg_clears_the_force_stop_with_the_cap() {
    let (start, cell) = fixture_start();
    let fg = move |x: ArrayView1<f64>| morse_value_grad(x, cell);
    let hs = move |x: ArrayView1<f64>| morse_hessian(x, cell);
    let gtol = 1e-3;
    let uncapped = run_method(Method::Dogleg, &start, None, gtol, 40, fg, hs);
    let capped = run_method(Method::Dogleg, &start, Some(0.2), gtol, 40, fg, hs);
    let newton = run_method(
        Method::Newton {
            kind: NewtonKind::Shifted,
        },
        &start,
        Some(0.2),
        gtol,
        40,
        fg,
        hs,
    );
    let dim = start.len();
    for (label, run) in [
        ("uncapped", &uncapped),
        ("capped", &capped),
        ("newton", &newton),
    ] {
        let force = run.grad_calls + dim * run.hess_calls;
        eprintln!(
            "morse pos_0000 {label}: grads={} hess={} steps={} force={} |g|={:.3e} energy={:.6} max_euclid={:.3} max_atom={:.3}",
            run.grad_calls,
            run.hess_calls,
            run.steps,
            force,
            run.grad_norm,
            run.energy,
            run.max_euclid,
            run.max_atom
        );
    }
    for (label, run) in [("uncapped", &uncapped), ("capped", &capped)] {
        let force = run.grad_calls + dim * run.hess_calls;
        assert!(
            run.grad_norm <= gtol && force < 10_000,
            "{label} missed the Morse stop: |g|={} force={force} hess={}",
            run.grad_norm,
            run.hess_calls
        );
        assert!(
            run.hess_calls <= 16,
            "{label} used {} Hessians; 0.4.0 default was about 9",
            run.hess_calls
        );
    }
    assert!(
        capped.max_atom <= 0.2 + 1e-8,
        "cap let an atom move {}",
        capped.max_atom
    );
    assert!(
        capped.max_euclid > 0.2 + 1e-8,
        "capped dogleg stayed inside the old 0.2 Euclidean ceiling ({})",
        capped.max_euclid
    );
    let newton_force = newton.grad_calls + dim * newton.hess_calls;
    assert!(
        newton.grad_norm <= gtol && newton_force < 10_000 && newton.hess_calls <= 16,
        "capped Newton regressed: |g|={} hess={} force={newton_force}",
        newton.grad_norm,
        newton.hess_calls
    );
}

/// `pos_0008` is the first Morse start the 0.4.1 cap pushed over the
/// 10000 force-call budget: 14 Hessians, force norm already below
/// 1e-3, harness failure `cap`. 0.4.0 cleared it in 7 Hessians.
#[test]
fn morse_pos_0008_capped_dogleg_stays_inside_the_force_budget() {
    let (start, cell) = load_fixture("morse_pos_0008.con");
    let fg = move |x: ArrayView1<f64>| morse_value_grad(x, cell);
    let hs = move |x: ArrayView1<f64>| morse_hessian(x, cell);
    let gtol = 1e-3;
    let uncapped = run_method(Method::Dogleg, &start, None, gtol, 40, fg, hs);
    let capped = run_method(Method::Dogleg, &start, Some(0.2), gtol, 40, fg, hs);
    let dim = start.len();
    for (label, run) in [("uncapped", &uncapped), ("capped", &capped)] {
        let force = run.grad_calls + dim * run.hess_calls;
        eprintln!(
            "morse pos_0008 {label}: grads={} hess={} steps={} force={} |g|={:.3e} energy={:.6} max_euclid={:.3} max_atom={:.3}",
            run.grad_calls,
            run.hess_calls,
            run.steps,
            force,
            run.grad_norm,
            run.energy,
            run.max_euclid,
            run.max_atom
        );
        assert!(
            run.grad_norm <= gtol && force < 10_000,
            "{label} still blows the Morse budget: |g|={} force={force} hess={}",
            run.grad_norm,
            run.hess_calls
        );
        assert!(
            run.hess_calls <= 12,
            "{label} used {} Hessians; 0.4.0 used 7 and 0.4.1 used 14",
            run.hess_calls
        );
    }
    assert!(
        capped.max_atom <= 0.2 + 1e-8,
        "atom move {}",
        capped.max_atom
    );
}

fn con_dir_runs(dir: &str, kind: &str) -> Vec<(String, Array1<f64>, [f64; 3])> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("read {dir}: {err}"))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("con"))
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "{kind} directory {dir} has no .con files"
    );
    paths
        .into_iter()
        .map(|path| {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("start")
                .to_string();
            let text = std::fs::read_to_string(&path).expect("con");
            let (coords, box_l) = read_con(&text);
            (name, coords, box_l)
        })
        .collect()
}

struct Summary {
    n: usize,
    success: usize,
    cap_hits: usize,
    mean_grad: f64,
    mean_hess: f64,
    median_grad: f64,
}

fn summarize(rows: &[(bool, usize, usize, bool)]) -> Summary {
    let n = rows.len();
    let success = rows.iter().filter(|row| row.0).count();
    let cap_hits = rows.iter().filter(|row| row.3).count();
    let mean_grad = rows.iter().map(|row| row.1 as f64).sum::<f64>() / n.max(1) as f64;
    let mean_hess = rows.iter().map(|row| row.2 as f64).sum::<f64>() / n.max(1) as f64;
    let mut grads: Vec<usize> = rows.iter().map(|row| row.1).collect();
    grads.sort_unstable();
    let median_grad = if grads.is_empty() {
        0.0
    } else if grads.len() % 2 == 1 {
        grads[grads.len() / 2] as f64
    } else {
        0.5 * (grads[grads.len() / 2 - 1] + grads[grads.len() / 2]) as f64
    };
    Summary {
        n,
        success,
        cap_hits,
        mean_grad,
        mean_hess,
        median_grad,
    }
}

fn worker_count(n: usize) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4);
    cores.min(n.max(1))
}

/// Full optbench Morse campaign. Set `MORSE_CON_DIR` to the directory of
/// `pos_*.con` files. Not part of the default suite: 100 dense 768
/// factorizations.
#[test]
#[ignore]
fn benchmark_morse_bulk_dogleg_campaign() {
    let dir =
        std::env::var("MORSE_CON_DIR").unwrap_or_else(|_| "/tmp/morse-con/morse-bulk".to_string());
    if !std::path::Path::new(&dir).is_dir() {
        eprintln!("skipping Morse campaign, {dir} is missing");
        return;
    }
    let starts = con_dir_runs(&dir, "morse");
    let gtol = 1e-3;
    let workers = worker_count(starts.len());
    let chunk = starts.len().div_ceil(workers);
    let parts: Vec<Vec<(bool, usize, usize, bool)>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for group in starts.chunks(chunk) {
            handles.push(scope.spawn(move || {
                let mut rows = Vec::new();
                for (name, coords, box_l) in group {
                    let cell = *box_l;
                    let fg = |x: ArrayView1<f64>| morse_value_grad(x, cell);
                    let hs = |x: ArrayView1<f64>| morse_hessian(x, cell);
                    let dim = coords.len();
                    let mut triple = Vec::new();
                    for (label, method, cap) in [
                        ("dogleg-uncapped", Method::Dogleg, None),
                        ("dogleg-capped", Method::Dogleg, Some(0.2)),
                        (
                            "newton-capped",
                            Method::Newton {
                                kind: NewtonKind::Shifted,
                            },
                            Some(0.2),
                        ),
                    ] {
                        let run = run_method(method, coords, cap, gtol, 60, fg, hs);
                        let force = run.grad_calls + dim * run.hess_calls;
                        let at_force = run.grad_norm <= gtol && force < 10_000;
                        triple.push((at_force, run.grad_calls, run.hess_calls, !at_force));
                        if !at_force {
                            eprintln!(
                                "{name} {label} grads={} hess={} force={} |g|={:.3e}",
                                run.grad_calls, run.hess_calls, force, run.grad_norm
                            );
                        }
                    }
                    rows.push(triple);
                }
                rows
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("morse worker"))
            .collect()
    });
    let mut uncapped_rows = Vec::new();
    let mut capped_rows = Vec::new();
    let mut newton_rows = Vec::new();
    for triple in parts {
        uncapped_rows.push(triple[0]);
        capped_rows.push(triple[1]);
        newton_rows.push(triple[2]);
    }
    for (label, rows) in [
        ("dogleg-uncapped", &uncapped_rows),
        ("dogleg-capped", &capped_rows),
        ("newton-capped", &newton_rows),
    ] {
        let summary = summarize(rows);
        eprintln!(
            "MORSE {label}: n={} success={} cap={} mean_grad={:.2} median_grad={:.1} mean_hess={:.2}",
            summary.n,
            summary.success,
            summary.cap_hits,
            summary.mean_grad,
            summary.median_grad,
            summary.mean_hess
        );
    }
    let capped = summarize(&capped_rows);
    let uncapped = summarize(&uncapped_rows);
    assert!(
        capped.success >= 98 && capped.mean_grad <= 12.0,
        "capped Morse dogleg success {} mean grad {}",
        capped.success,
        capped.mean_grad
    );
    assert!(
        uncapped.success >= 98 && uncapped.mean_grad <= 12.0,
        "uncapped Morse dogleg success {} mean grad {}",
        uncapped.success,
        uncapped.mean_grad
    );
    let newton = summarize(&newton_rows);
    assert!(
        newton.success == newton.n && newton.mean_grad < 20.0,
        "capped Newton success {} / {} mean grad {}",
        newton.success,
        newton.n,
        newton.mean_grad
    );
}

/// LJ38 dogleg with the campaign's per-atom cap of 0.2. Set `LJ_CON_DIR`.
#[test]
#[ignore]
fn benchmark_lj38_dogleg_campaign() {
    let dir = std::env::var("LJ_CON_DIR").unwrap_or_else(|_| "/tmp/lj38/lj38-clusters".to_string());
    if !std::path::Path::new(&dir).is_dir() {
        eprintln!("skipping LJ38 campaign, {dir} is missing");
        return;
    }
    let starts = con_dir_runs(&dir, "lj38");
    let gtol = 1e-2;
    let workers = worker_count(starts.len());
    let chunk = starts.len().div_ceil(workers);
    let rows: Vec<(bool, usize, usize, bool)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for group in starts.chunks(chunk) {
            handles.push(scope.spawn(move || {
                let mut rows = Vec::new();
                for (_name, coords, _) in group {
                    let dim = coords.len();
                    let run = run_method(
                        Method::Dogleg,
                        coords,
                        Some(0.2),
                        gtol,
                        400,
                        lj_value_grad,
                        lj_hessian,
                    );
                    let force = run.grad_calls + dim * run.hess_calls;
                    let at_force = run.grad_norm <= gtol && force < 10_000;
                    rows.push((at_force, run.grad_calls, run.hess_calls, !at_force));
                }
                rows
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("lj worker"))
            .collect()
    });
    let summary = summarize(&rows);
    eprintln!(
        "LJ38 dogleg cap 0.2: n={} success={} cap={} mean_grad={:.2} median_grad={:.1} mean_hess={:.2}",
        summary.n,
        summary.success,
        summary.cap_hits,
        summary.mean_grad,
        summary.median_grad,
        summary.mean_hess
    );
    assert!(
        summary.n < 1000 || (summary.success >= 673 && summary.mean_grad <= 82.0),
        "LJ38 dogleg lost the reference 673/1000 at 81.9 gradients: success {} mean {}",
        summary.success,
        summary.mean_grad
    );
}
