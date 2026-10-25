//! Steihaug-Toint Newton-CG on the same Morse and LJ38 rows as dogleg.
//!
//! Hessian access on this path is a product. A dense factorization is
//! the comparison, not the implementation. Each product is one force
//! call; each dense Hessian is `n` force calls.

#![allow(clippy::needless_range_loop)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{
    Control, HessianOracle, HvpOracle, LineSearch, Method, NewtonKind, Oracle, Solver,
    minimize_method,
};

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

fn pair_block(diff: [f64; 3], r2: f64, first: f64, second: f64) -> [[f64; 3]; 3] {
    let r = r2.sqrt();
    let s = first / r;
    let outer_scale = (second - s) / r2;
    let mut block = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            block[a][b] = -outer_scale * diff[a] * diff[b];
            if a == b {
                block[a][b] -= s;
            }
        }
    }
    block
}

fn scatter_block(hessian: &mut Array2<f64>, i: usize, j: usize, block: [[f64; 3]; 3]) {
    for a in 0..3 {
        for b in 0..3 {
            let ia = 3 * i + a;
            let ib = 3 * i + b;
            let ja = 3 * j + a;
            let jb = 3 * j + b;
            let value = block[a][b];
            hessian[(ia, jb)] += value;
            hessian[(jb, ia)] += value;
            hessian[(ia, ib)] -= value;
            hessian[(ja, jb)] -= value;
        }
    }
}

fn apply_block(hv: &mut Array1<f64>, v: ArrayView1<f64>, i: usize, j: usize, block: [[f64; 3]; 3]) {
    let mut dv = [0.0; 3];
    for a in 0..3 {
        dv[a] = v[3 * j + a] - v[3 * i + a];
    }
    let mut bdv = [0.0; 3];
    for a in 0..3 {
        for b in 0..3 {
            bdv[a] += block[a][b] * dv[b];
        }
    }
    for a in 0..3 {
        hv[3 * i + a] += bdv[a];
        hv[3 * j + a] -= bdv[a];
    }
}

fn morse_hessian(pos: ArrayView1<f64>, box_l: [f64; 3]) -> Array2<f64> {
    let mut hessian = Array2::<f64>::zeros((pos.len(), pos.len()));
    for (i, j, diff, r2) in morse_pairs(pos, box_l) {
        let x = (-MORSE_ALPHA * (r2.sqrt() - MORSE_R0)).exp();
        let first = 2.0 * MORSE_D0 * MORSE_ALPHA * x * (1.0 - x);
        let second = 2.0 * MORSE_D0 * MORSE_ALPHA * MORSE_ALPHA * x * (2.0 * x - 1.0);
        scatter_block(&mut hessian, i, j, pair_block(diff, r2, first, second));
    }
    symmetrize(&mut hessian);
    hessian
}

fn morse_hvp(pos: ArrayView1<f64>, v: ArrayView1<f64>, box_l: [f64; 3]) -> Array1<f64> {
    let mut hv = Array1::<f64>::zeros(pos.len());
    for (i, j, diff, r2) in morse_pairs(pos, box_l) {
        let x = (-MORSE_ALPHA * (r2.sqrt() - MORSE_R0)).exp();
        let first = 2.0 * MORSE_D0 * MORSE_ALPHA * x * (1.0 - x);
        let second = 2.0 * MORSE_D0 * MORSE_ALPHA * MORSE_ALPHA * x * (2.0 * x - 1.0);
        apply_block(&mut hv, v, i, j, pair_block(diff, r2, first, second));
    }
    hv
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

fn lj_pairs(pos: ArrayView1<f64>) -> Vec<(usize, usize, [f64; 3], f64, f64, f64)> {
    let n = pos.len() / 3;
    let cut2 = LJ_CUTOFF * LJ_CUTOFF;
    let mut pairs = Vec::new();
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
            pairs.push((i, j, diff, r2, first, second));
        }
    }
    pairs
}

fn lj_hessian(pos: ArrayView1<f64>) -> Array2<f64> {
    let mut hessian = Array2::<f64>::zeros((pos.len(), pos.len()));
    for (i, j, diff, r2, first, second) in lj_pairs(pos) {
        scatter_block(&mut hessian, i, j, pair_block(diff, r2, first, second));
    }
    symmetrize(&mut hessian);
    hessian
}

fn lj_hvp(pos: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
    let mut hv = Array1::<f64>::zeros(pos.len());
    for (i, j, diff, r2, first, second) in lj_pairs(pos) {
        apply_block(&mut hv, v, i, j, pair_block(diff, r2, first, second));
    }
    hv
}

fn symmetrize(hessian: &mut Array2<f64>) {
    let dim = hessian.nrows();
    for i in 0..dim {
        for j in (i + 1)..dim {
            let m = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            hessian[(i, j)] = m;
            hessian[(j, i)] = m;
        }
    }
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

fn max_atom_step(before: &Array1<f64>, after: &Array1<f64>) -> f64 {
    let mut m = 0.0_f64;
    let mut i = 0;
    while i + 3 <= before.len() {
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
    products: usize,
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
        ftol_rel: None,
        noise_floor: None,
    }
}

fn run_dense(
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
    for _ in 0..step_budget {
        let before = x.clone();
        let rep = solver.step_hess(&obj, &mut x).expect("dense step");
        max_euclid = max_euclid.max(euclid(&(&x - &before)));
        max_atom = max_atom.max(max_atom_step(&before, &x));
        last_norm = rep.grad_norm;
        last_energy = rep.value;
        if rep.grad_norm <= gtol {
            break;
        }
    }
    Run {
        grad_calls: grad_calls.load(Ordering::Relaxed),
        products: hess_calls.load(Ordering::Relaxed),
        energy: last_energy,
        grad_norm: last_norm,
        max_euclid,
        max_atom,
    }
}

fn run_newton_cg(
    method: Method,
    start: &Array1<f64>,
    cap: Option<f64>,
    maxmove: Option<f64>,
    gtol: f64,
    step_budget: usize,
    fg: impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    hvp: impl Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
) -> Run {
    let dim = start.len();
    let obj = HvpOracle::unbounded(dim, fg, hvp);
    let mut ctrl = control(gtol, step_budget.max(1));
    ctrl.maxmove = maxmove;
    let mut solver = Solver::new(method, ctrl, dim);
    if let Some(cap) = cap {
        solver.set_atom_maxmove(cap);
    }
    let mut x = start.clone();
    let mut max_euclid = 0.0_f64;
    let mut max_atom = 0.0_f64;
    let mut last_norm = f64::MAX;
    let mut last_energy = 0.0;
    for _ in 0..step_budget {
        let before = x.clone();
        let rep = solver.step_hvp(&obj, &mut x).expect("newton-cg step");
        max_euclid = max_euclid.max(euclid(&(&x - &before)));
        max_atom = max_atom.max(max_atom_step(&before, &x));
        last_norm = rep.grad_norm;
        last_energy = rep.value;
        if rep.grad_norm <= gtol {
            break;
        }
    }
    Run {
        grad_calls: solver.gradient_calls(),
        products: solver.hvp_calls(),
        energy: last_energy,
        grad_norm: last_norm,
        max_euclid,
        max_atom,
    }
}

fn cluster_start(n_atoms: usize) -> Array1<f64> {
    let mut start = Array1::zeros(3 * n_atoms);
    for i in 0..n_atoms {
        start[3 * i] = 1.0;
    }
    start
}

#[test]
fn counters_split_gradients_from_hessian_vector_products() {
    let grads = Arc::new(AtomicUsize::new(0));
    let gcount = Arc::clone(&grads);
    let obj = Oracle::unbounded(4, move |x: ArrayView1<f64>| {
        gcount.fetch_add(1, Ordering::Relaxed);
        let f = 0.5 * x.iter().map(|v| v * v).sum::<f64>();
        (f, x.to_owned())
    });
    let mut solver = Solver::new(Method::NewtonCg, control(1e-8, 30), 4);
    let mut x = Array1::from_vec(vec![1.0, -1.0, 0.5, 0.25]);
    for _ in 0..30 {
        let rep = solver.step(&obj, &mut x).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }
    assert!(x.iter().all(|v| v.abs() < 1e-6), "x {x}");
    assert!(solver.gradient_calls() > 0);
    assert!(solver.hvp_calls() > 0);
    assert_eq!(
        grads.load(Ordering::Relaxed),
        solver.gradient_calls() + solver.hvp_calls()
    );
    let kept = solver.gradient_calls();
    solver.forget();
    assert_eq!(solver.gradient_calls(), kept);
    assert!(solver.hvp_calls() > 0);

    let hvps = Arc::new(AtomicUsize::new(0));
    let evals = Arc::new(AtomicUsize::new(0));
    let hcount = Arc::clone(&hvps);
    let ecount = Arc::clone(&evals);
    let analytic = HvpOracle::unbounded(
        4,
        move |x: ArrayView1<f64>| {
            ecount.fetch_add(1, Ordering::Relaxed);
            let f = 0.5 * x.iter().map(|v| v * v).sum::<f64>();
            (f, x.to_owned())
        },
        move |_, v: ArrayView1<f64>| {
            hcount.fetch_add(1, Ordering::Relaxed);
            v.to_owned()
        },
    );
    let mut solver = Solver::new(Method::NewtonCg, control(1e-8, 30), 4);
    let mut x = Array1::from_vec(vec![1.0, -1.0, 0.5, 0.25]);
    for _ in 0..30 {
        let rep = solver.step_hvp(&analytic, &mut x).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }
    assert_eq!(hvps.load(Ordering::Relaxed), solver.hvp_calls());
    assert_eq!(evals.load(Ordering::Relaxed), solver.gradient_calls());
    assert!(solver.gradient_calls() > 0 && solver.hvp_calls() > 0);

    let built = Arc::new(AtomicUsize::new(0));
    let bcount = Arc::clone(&built);
    let dense = HessianOracle::unbounded(
        4,
        |x: ArrayView1<f64>| {
            let f = 0.5 * x.iter().map(|v| v * v).sum::<f64>();
            (f, x.to_owned())
        },
        move |x: ArrayView1<f64>| {
            bcount.fetch_add(1, Ordering::Relaxed);
            Array2::<f64>::eye(x.len())
        },
    );
    let mut solver = Solver::new(Method::NewtonCg, control(1e-8, 8), 4);
    let mut x = Array1::from_vec(vec![1.0, -1.0, 0.5, 0.25]);
    let _ = solver.step_hess(&dense, &mut x).unwrap();
    assert_eq!(built.load(Ordering::Relaxed), 0);
    assert!(solver.hvp_calls() > 0);

    let named = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let f = 0.5 * x.iter().map(|v| v * v).sum::<f64>();
        (f, x.to_owned())
    });
    let rep = minimize_method(
        &named,
        Array1::from_vec(vec![0.4, -0.2]),
        &control(1e-8, 40),
        Method::from_name("newton-cg").unwrap(),
        LineSearch::default(),
    )
    .unwrap();
    assert!(rep.grad_norm < 1e-8, "|g| {}", rep.grad_norm);
}

#[test]
fn per_atom_cap_does_not_freeze_the_newton_cg_radius() {
    let start = cluster_start(32);
    let quad_fg = |x: ArrayView1<f64>| {
        let f = 0.5 * x.iter().map(|v| v * v).sum::<f64>();
        (f, x.to_owned())
    };
    let quad_hv = |_: ArrayView1<f64>, v: ArrayView1<f64>| v.to_owned();
    let capped = run_newton_cg(
        Method::NewtonCg,
        &start,
        Some(0.2),
        None,
        1e-8,
        16,
        quad_fg,
        quad_hv,
    );
    assert!(
        capped.grad_norm < 1e-8,
        "capped newton-cg stalled at |g|={} (euclid {})",
        capped.grad_norm,
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
    let uncapped = run_newton_cg(
        Method::NewtonCg,
        &start,
        None,
        None,
        1e-8,
        16,
        quad_fg,
        quad_hv,
    );
    assert!(
        uncapped.grad_norm < 1e-8,
        "uncapped newton-cg |g|={}",
        uncapped.grad_norm
    );
}

#[test]
fn maxmove_is_the_euclidean_trust_ceiling() {
    let start = cluster_start(32);
    let quad_fg = |x: ArrayView1<f64>| {
        let f = 0.5 * x.iter().map(|v| v * v).sum::<f64>();
        (f, x.to_owned())
    };
    let quad_hv = |_: ArrayView1<f64>, v: ArrayView1<f64>| v.to_owned();
    let capped = run_newton_cg(
        Method::NewtonCg,
        &start,
        None,
        Some(0.2),
        1e-8,
        16,
        quad_fg,
        quad_hv,
    );
    assert!(
        capped.max_euclid <= 0.2 + 1e-9,
        "euclidean step {} exceeded maxmove 0.2",
        capped.max_euclid
    );
    assert!(
        capped.grad_norm > 1.0,
        "a 0.2 ceiling reached the bowl in 16 steps, |g|={}",
        capped.grad_norm
    );
}

#[test]
fn morse_hvp_matches_the_dense_pair_hessian() {
    let (x, cell) = load_fixture("morse_pos_0000.con");
    let h = morse_hessian(x.view(), cell);
    let mut v = Array1::<f64>::zeros(x.len());
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = ((i * 17) % 11) as f64 - 5.0;
    }
    let hv = morse_hvp(x.view(), v.view(), cell);
    let dense = h.dot(&v);
    let mut worst = 0.0_f64;
    for (a, b) in hv.iter().zip(dense.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert!(worst < 1e-8, "product mismatch {worst}");
}

fn morse_case(name: &str, cap: Option<f64>) -> Run {
    let (start, cell) = load_fixture(name);
    let fg = move |x: ArrayView1<f64>| morse_value_grad(x, cell);
    let hv = move |x: ArrayView1<f64>, v: ArrayView1<f64>| morse_hvp(x, v, cell);
    let run = run_newton_cg(Method::NewtonCg, &start, cap, None, 1e-3, 80, fg, hv);
    let force = run.grad_calls + run.products;
    eprintln!(
        "morse {name} cap={cap:?}: grads={} hvp={} force={force} |g|={:.3e} energy={:.6} max_euclid={:.3} max_atom={:.3}",
        run.grad_calls, run.products, run.grad_norm, run.energy, run.max_euclid, run.max_atom
    );
    run
}

#[test]
fn morse_pos_0000_newton_cg_clears_the_force_stop() {
    let uncapped = morse_case("morse_pos_0000.con", None);
    let capped = morse_case("morse_pos_0000.con", Some(0.2));
    for (label, run) in [("uncapped", &uncapped), ("capped", &capped)] {
        let force = run.grad_calls + run.products;
        assert!(
            run.grad_norm <= 1e-3 && force < 10_000,
            "{label} missed the Morse stop: |g|={} force={force}",
            run.grad_norm
        );
    }
    assert!(
        capped.max_atom <= 0.2 + 1e-8,
        "atom move {}",
        capped.max_atom
    );
    assert!(
        capped.max_euclid > 0.2 + 1e-8,
        "capped newton-cg stayed inside a 0.2 Euclidean ball ({})",
        capped.max_euclid
    );
}

#[test]
fn morse_pos_0008_newton_cg_clears_the_force_stop() {
    let uncapped = morse_case("morse_pos_0008.con", None);
    let capped = morse_case("morse_pos_0008.con", Some(0.2));
    for (label, run) in [("uncapped", &uncapped), ("capped", &capped)] {
        let force = run.grad_calls + run.products;
        assert!(
            run.grad_norm <= 1e-3 && force < 10_000,
            "{label} missed the Morse stop: |g|={} force={force}",
            run.grad_norm
        );
    }
    assert!(
        capped.max_atom <= 0.2 + 1e-8,
        "atom move {}",
        capped.max_atom
    );
}

#[test]
fn newton_pcg_minimizes_a_quadratic() {
    let hvps = Arc::new(AtomicUsize::new(0));
    let hcount = Arc::clone(&hvps);
    let obj = HvpOracle::unbounded(
        24,
        |x: ArrayView1<f64>| {
            let f = x
                .iter()
                .enumerate()
                .map(|(i, v)| (i + 1) as f64 * v * v / 2.0)
                .sum();
            let g = Array1::from_iter(x.iter().enumerate().map(|(i, v)| (i + 1) as f64 * v));
            (f, g)
        },
        move |_, v: ArrayView1<f64>| {
            hcount.fetch_add(1, Ordering::Relaxed);
            Array1::from_iter(v.iter().enumerate().map(|(i, w)| (i + 1) as f64 * w))
        },
    );
    let mut solver = Solver::new(Method::NewtonPcg, control(1e-8, 40), 24);
    let mut x = Array1::from_elem(24, 0.5);
    for _ in 0..40 {
        let rep = solver.step_hvp(&obj, &mut x).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }
    assert!(euclid(&x) < 1e-6, "x norm {}", euclid(&x));
    assert_eq!(hvps.load(Ordering::Relaxed), solver.hvp_calls());
    assert!(
        solver.hvp_calls() > 16,
        "the rank-16 sketch was not counted: {}",
        solver.hvp_calls()
    );
}

fn con_dir(dir: &str) -> Vec<(String, Array1<f64>, [f64; 3])> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("read {dir}: {err}"))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("con"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "{dir} has no .con files");
    paths
        .into_iter()
        .map(|path| {
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
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
    mean_grad: f64,
    mean_products: f64,
    mean_force: f64,
}

fn summarize(rows: &[(bool, usize, usize, usize)]) -> Summary {
    let n = rows.len();
    let success = rows.iter().filter(|row| row.0).count();
    let mean = |index: usize| {
        rows.iter()
            .map(|row| match index {
                1 => row.1 as f64,
                2 => row.2 as f64,
                _ => row.3 as f64,
            })
            .sum::<f64>()
            / n.max(1) as f64
    };
    Summary {
        n,
        success,
        mean_grad: mean(1),
        mean_products: mean(2),
        mean_force: mean(3),
    }
}

fn worker_count(n: usize) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4);
    cores.min(n.max(1))
}

fn print_summary(label: &str, rows: &[(bool, usize, usize, usize)]) -> Summary {
    let summary = summarize(rows);
    eprintln!(
        "{label}: n={} success={} mean_grad={:.2} mean_products={:.2} mean_force={:.2}",
        summary.n, summary.success, summary.mean_grad, summary.mean_products, summary.mean_force
    );
    summary
}

/// Morse bulk: dense Newton and dogleg against Newton-CG, cap 0.2 and
/// uncapped. `MORSE_CON_DIR` defaults to the optbench extract.
#[test]
#[ignore]
fn benchmark_morse_newton_cg_campaign() {
    let dir =
        std::env::var("MORSE_CON_DIR").unwrap_or_else(|_| "/tmp/morse-con/morse-bulk".to_string());
    if !std::path::Path::new(&dir).is_dir() {
        eprintln!("skipping Morse campaign, {dir} is missing");
        return;
    }
    let starts = con_dir(&dir);
    let gtol = 1e-3;
    let workers = worker_count(starts.len());
    let chunk = starts.len().div_ceil(workers);
    let parts: Vec<Vec<(bool, usize, usize, usize)>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for group in starts.chunks(chunk) {
            handles.push(scope.spawn(move || {
                let mut rows = Vec::new();
                for (name, coords, box_l) in group {
                    let cell = *box_l;
                    let dim = coords.len();
                    let mut six = Vec::new();
                    for (label, cap) in [("uncapped", None), ("cap-0.2", Some(0.2))] {
                        let fg = |x: ArrayView1<f64>| morse_value_grad(x, cell);
                        let hs = |x: ArrayView1<f64>| morse_hessian(x, cell);
                        for (method_name, method) in [
                            ("dogleg", Method::Dogleg),
                            (
                                "newton",
                                Method::Newton {
                                    kind: NewtonKind::Shifted,
                                },
                            ),
                        ] {
                            let run = run_dense(method, coords, cap, gtol, 60, &fg, &hs);
                            let force = run.grad_calls + dim * run.products;
                            let ok = run.grad_norm <= gtol && force < 10_000;
                            if !ok {
                                eprintln!(
                                    "{name} {method_name}-{label} grads={} hess={} force={force} |g|={:.3e}",
                                    run.grad_calls, run.products, run.grad_norm
                                );
                            }
                            six.push((ok, run.grad_calls, run.products, force));
                        }
                        let fg = |x: ArrayView1<f64>| morse_value_grad(x, cell);
                        let hv = |x: ArrayView1<f64>, v: ArrayView1<f64>| morse_hvp(x, v, cell);
                        let run =
                            run_newton_cg(Method::NewtonCg, coords, cap, None, gtol, 60, fg, hv);
                        let force = run.grad_calls + run.products;
                        let ok = run.grad_norm <= gtol && force < 10_000;
                        if !ok {
                            eprintln!(
                                "{name} newton-cg-{label} grads={} hvp={} force={force} |g|={:.3e}",
                                run.grad_calls, run.products, run.grad_norm
                            );
                        }
                        six.push((ok, run.grad_calls, run.products, force));
                    }
                    rows.push(six);
                }
                rows
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("morse worker"))
            .collect()
    });
    let labels = [
        "morse dogleg uncapped",
        "morse newton uncapped",
        "morse newton-cg uncapped",
        "morse dogleg cap 0.2",
        "morse newton cap 0.2",
        "morse newton-cg cap 0.2",
    ];
    let mut summaries = Vec::new();
    for (index, label) in labels.iter().enumerate() {
        let rows: Vec<_> = parts.iter().map(|row| row[index]).collect();
        summaries.push(print_summary(label, &rows));
    }
    for summary in [&summaries[0], &summaries[3]] {
        assert!(
            summary.success >= 98 && summary.mean_grad <= 12.0,
            "dogleg Morse success {} mean grad {}",
            summary.success,
            summary.mean_grad
        );
    }
    assert!(
        summaries[1].success >= 80 && summaries[1].mean_grad <= 16.0,
        "uncapped Newton success {} mean grad {}",
        summaries[1].success,
        summaries[1].mean_grad
    );
    assert!(
        summaries[4].success == summaries[4].n && summaries[4].mean_grad < 20.0,
        "capped Newton success {} / {} mean grad {}",
        summaries[4].success,
        summaries[4].n,
        summaries[4].mean_grad
    );
    for summary in [&summaries[2], &summaries[5]] {
        assert!(
            summary.success == summary.n && summary.mean_force < 200.0,
            "newton-cg Morse success {} / {} mean force {}",
            summary.success,
            summary.n,
            summary.mean_force
        );
    }
}

/// LJ38 with the per-atom cap of 0.2. `LJ_CON_DIR` defaults to the
/// optbench extract.
#[test]
#[ignore]
fn benchmark_lj38_newton_cg_campaign() {
    let dir = std::env::var("LJ_CON_DIR").unwrap_or_else(|_| "/tmp/lj38/lj38-clusters".to_string());
    if !std::path::Path::new(&dir).is_dir() {
        eprintln!("skipping LJ38 campaign, {dir} is missing");
        return;
    }
    let starts = con_dir(&dir);
    let gtol = 1e-2;
    let workers = worker_count(starts.len());
    let chunk = starts.len().div_ceil(workers);
    let parts: Vec<Vec<(bool, usize, usize, usize)>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for group in starts.chunks(chunk) {
            handles.push(scope.spawn(move || {
                let mut rows = Vec::new();
                for (name, coords, _) in group {
                    let dim = coords.len();
                    let mut three = Vec::new();
                    for (method_name, method) in [
                        ("dogleg", Method::Dogleg),
                        (
                            "newton",
                            Method::Newton {
                                kind: NewtonKind::Shifted,
                            },
                        ),
                    ] {
                        let run = run_dense(
                            method,
                            coords,
                            Some(0.2),
                            gtol,
                            400,
                            lj_value_grad,
                            lj_hessian,
                        );
                        let force = run.grad_calls + dim * run.products;
                        let ok = run.grad_norm <= gtol && force < 10_000;
                        if !ok {
                            eprintln!(
                                "{name} {method_name} grads={} hess={} force={force} |g|={:.3e}",
                                run.grad_calls, run.products, run.grad_norm
                            );
                        }
                        three.push((ok, run.grad_calls, run.products, force));
                    }
                    let run = run_newton_cg(
                        Method::NewtonCg,
                        coords,
                        Some(0.2),
                        None,
                        gtol,
                        400,
                        lj_value_grad,
                        lj_hvp,
                    );
                    let force = run.grad_calls + run.products;
                    let ok = run.grad_norm <= gtol && force < 10_000;
                    if !ok {
                        eprintln!(
                            "{name} newton-cg grads={} hvp={} force={force} |g|={:.3e}",
                            run.grad_calls, run.products, run.grad_norm
                        );
                    }
                    three.push((ok, run.grad_calls, run.products, force));
                    rows.push(three);
                }
                rows
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("lj worker"))
            .collect()
    });
    let mut summaries = Vec::new();
    for (index, label) in [
        "lj38 dogleg cap 0.2",
        "lj38 newton cap 0.2",
        "lj38 newton-cg cap 0.2",
    ]
    .iter()
    .enumerate()
    {
        let rows: Vec<_> = parts.iter().map(|row| row[index]).collect();
        summaries.push(print_summary(label, &rows));
    }
    assert!(
        summaries[0].n < 1000 || (summaries[0].success >= 673 && summaries[0].mean_grad <= 82.0),
        "LJ38 dogleg success {} mean grad {}",
        summaries[0].success,
        summaries[0].mean_grad
    );
    assert!(
        summaries[1].success >= 850 && summaries[1].mean_grad <= 80.0,
        "LJ38 Newton success {} mean grad {}",
        summaries[1].success,
        summaries[1].mean_grad
    );
    assert!(
        summaries[2].success == summaries[2].n && summaries[2].mean_force < 1000.0,
        "LJ38 newton-cg success {} / {} mean force {}",
        summaries[2].success,
        summaries[2].n,
        summaries[2].mean_force
    );
}
