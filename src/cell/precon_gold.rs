//! The `μ_c I` cell block of the exponential preconditioner against ASE
//! 3.29.0 `PreconLBFGS` with `Exp` on a `FrechetCellFilter`, from
//! `tests/data/ase_cell_precon.txt` (written by
//! `tests/ase_cell_precon_gold.py`).
//!
//! Compared: `μ` and `μ_c` of the two-evaluation estimate (Cu lands on
//! the floor of 1, Si clears it), the matrix built from those rgmin
//! scales, a solve, and the L-BFGS direction from the pairs ASE stored.
//! The tolerances are derived in place from `u` and the data.

use ndarray::{Array1, ArrayView1};

use super::{Cell, CellFilter, CellFilterConfig};
use crate::{Exp, ExpParams, LbfgsH0, LbfgsPairs, Neighbours, estimate_mu_cell};

const U: f64 = f64::EPSILON / 2.0;
/// ASE keeps the strain block as `ecf logm(N F)` and `set_positions`
/// undoes it with `expm`. SciPy `logm` reaches `logm(N I)` by inverse
/// scaling and squaring, about `log2(ln N / ln 1.2)` square roots whose
/// rounding the squarings amplify, so the probe point ASE evaluates is
/// off by a few hundred `u` times the offset `ecf ln N`; the bound is
/// set at 1024 `u` times the largest extended coordinate.
const OFFSET_OPS: f64 = 1024.0;
/// The relative residual the conjugate-gradient solve stops at.
const SOLVE_RTOL: f64 = 1.0e-12;

struct Call {
    positions: Array1<f64>,
    cell: Cell,
    forces: Array1<f64>,
    stress: Cell,
}

struct Step {
    g: Array1<f64>,
    pairs: Vec<(Array1<f64>, Array1<f64>, f64)>,
    solves: Vec<(Array1<f64>, Array1<f64>)>,
    dir: Array1<f64>,
}

struct Case {
    name: String,
    natoms: usize,
    r_cut: f64,
    r_nn: f64,
    a: f64,
    c_stab: f64,
    mu: f64,
    mu_c: f64,
    lengths: [f64; 3],
    orig: Cell,
    x0: Array1<f64>,
    calls: Vec<Call>,
    neighbours: Neighbours,
    entries: Vec<(usize, usize, f64)>,
    solve: (Array1<f64>, Array1<f64>),
    steps: Vec<Step>,
}

fn row(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .skip(1)
        .map(|t| t.parse().unwrap())
        .collect()
}

fn cell3(v: &[f64]) -> Cell {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

fn load() -> Vec<Case> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/ase_cell_precon.txt"
    );
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#')).peekable();
    let mut cases = Vec::new();
    while let Some(head) = lines.next() {
        let t: Vec<&str> = head.split_whitespace().collect();
        assert_eq!(t[0], "head");
        let natoms: usize = t[2].parse().unwrap();
        let nums: Vec<f64> = t[3..].iter().map(|s| s.parse().unwrap()).collect();
        let nsteps = nums[9] as usize;
        let orig = cell3(&row(lines.next().unwrap()));
        let x0 = Array1::from(row(lines.next().unwrap()));
        let n3 = 3 * natoms;
        let mut calls = Vec::new();
        for _ in 0..2 {
            let v = row(lines.next().unwrap());
            calls.push(Call {
                positions: Array1::from(v[..n3].to_vec()),
                cell: cell3(&v[n3..n3 + 9]),
                forces: Array1::from(v[n3 + 10..2 * n3 + 10].to_vec()),
                stress: cell3(&v[2 * n3 + 10..2 * n3 + 19]),
            });
        }
        let mut neighbours = Neighbours::default();
        while lines.peek().unwrap().starts_with("pair ") {
            let v = row(lines.next().unwrap());
            neighbours.i.push(v[0] as usize);
            neighbours.j.push(v[1] as usize);
            neighbours.r.push(v[2]);
        }
        let mut entries = Vec::new();
        while lines.peek().unwrap().starts_with("p ") {
            let v = row(lines.next().unwrap());
            entries.push((v[0] as usize, v[1] as usize, v[2]));
        }
        let dim = n3 + 9;
        let v = row(lines.next().unwrap());
        let solve = (
            Array1::from(v[..dim].to_vec()),
            Array1::from(v[dim..].to_vec()),
        );
        let mut steps = Vec::new();
        for _ in 0..nsteps {
            let h: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
            let npairs: usize = h[2].parse().unwrap();
            let nsolves: usize = h[3].parse().unwrap();
            let g = Array1::from(row(lines.next().unwrap()));
            let mut pairs = Vec::new();
            for _ in 0..npairs {
                let v = row(lines.next().unwrap());
                pairs.push((
                    Array1::from(v[..dim].to_vec()),
                    Array1::from(v[dim..2 * dim].to_vec()),
                    v[2 * dim],
                ));
            }
            let mut solves = Vec::new();
            for _ in 0..nsolves {
                let v = row(lines.next().unwrap());
                solves.push((
                    Array1::from(v[..dim].to_vec()),
                    Array1::from(v[dim..].to_vec()),
                ));
            }
            let dir = Array1::from(row(lines.next().unwrap()));
            steps.push(Step {
                g,
                pairs,
                solves,
                dir,
            });
        }
        cases.push(Case {
            name: t[1].to_owned(),
            natoms,
            r_cut: nums[0],
            r_nn: nums[1],
            a: nums[2],
            c_stab: nums[3],
            mu: nums[4],
            mu_c: nums[5],
            lengths: [nums[6], nums[7], nums[8]],
            orig,
            x0,
            calls,
            neighbours,
            entries,
            solve,
            steps,
        });
    }
    cases
}

fn max_abs(v: ArrayView1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, a| m.max(a.abs()))
}

fn params(case: &Case) -> ExpParams {
    ExpParams::default()
        .with_a(case.a)
        .with_c_stab(case.c_stab)
        .with_r_cut(case.r_cut)
        .with_r_nn(case.r_nn)
        .with_lengths(case.lengths)
}

fn filter(case: &Case) -> CellFilter {
    CellFilter::new(CellFilterConfig::default(), case.orig, case.natoms).unwrap()
}

/// `x0` of rgmin: ASE's extended positions without `ecf ln N` on the
/// diagonal of the strain block.
fn start(case: &Case) -> Array1<f64> {
    let mut x = case.x0.clone();
    let base = 3 * case.natoms;
    let shift = case.natoms as f64 * (case.natoms as f64).ln();
    for d in 0..3 {
        x[base + 4 * d] -= shift;
    }
    x
}

fn gradient_of(filter: &CellFilter, call: &Call) -> Array1<f64> {
    filter
        .gradient(call.forces.view(), &call.stress, &call.cell)
        .unwrap()
}

/// `μ` and `μ_c` from [`estimate_mu_cell`] on the two evaluations ASE stored.
fn own_scales(case: &Case) -> (f64, f64) {
    let filter = filter(case);
    let x0 = start(case);
    let packed: Vec<Array1<f64>> = case
        .calls
        .iter()
        .map(|c| filter.pack(c.positions.view(), &c.cell).unwrap())
        .collect();
    let grads: Vec<Array1<f64>> = case.calls.iter().map(|c| gradient_of(&filter, c)).collect();
    let point_tol = OFFSET_OPS * U * max_abs(case.x0.view());
    let grad = |x: ArrayView1<f64>| -> Array1<f64> {
        for (p, g) in packed.iter().zip(&grads) {
            let near = (0..x.len()).all(|i| (x[i] - p[i]).abs() <= point_tol);
            if near {
                return g.clone();
            }
        }
        panic!(
            "{}: the estimate asked for a point ASE did not evaluate",
            case.name
        );
    };
    estimate_mu_cell(x0.view(), &case.neighbours, &params(case), grad).unwrap()
}

#[test]
fn mu_and_mu_c_match_the_two_evaluation_estimate() {
    for case in &load() {
        let filter = filter(case);
        let x0 = start(case);
        let packed: Vec<Array1<f64>> = case
            .calls
            .iter()
            .map(|c| filter.pack(c.positions.view(), &c.cell).unwrap())
            .collect();
        let grads: Vec<Array1<f64>> = case.calls.iter().map(|c| gradient_of(&filter, c)).collect();
        let scale = max_abs(case.x0.view());
        let point_tol = OFFSET_OPS * U * scale;
        let (mu, mu_c) = own_scales(case);

        let split = 3 * case.natoms;
        let g_scale = max_abs(grads[0].view()).max(max_abs(grads[1].view()));
        let probe = &packed[1] - &x0;
        let block = |range: std::ops::Range<usize>| {
            let mut num = 0.0;
            let mut vsum = 0.0;
            let mut vmax = 0.0_f64;
            for k in range {
                num += (grads[1][k] - grads[0][k]) * probe[k];
                vsum += probe[k].abs();
                vmax = vmax.max(probe[k].abs());
            }
            (num.abs(), vsum, vmax)
        };
        // Gradient rounding in the difference, plus the probe ASE
        // evaluated differing from the one rgmin asks for by `point_tol`.
        let cond = |(num, vsum, vmax): (f64, f64, f64)| {
            2.0 * 512.0 * U * g_scale * vsum / num + 64.0 * U + point_tol / vmax
        };
        if case.mu > 1.0 {
            let tol = case.mu * cond(block(0..split));
            assert!(
                (mu - case.mu).abs() <= tol,
                "{}: mu {mu} vs {}",
                case.name,
                case.mu
            );
        } else {
            assert_eq!(mu, 1.0, "{}: mu floors at 1", case.name);
            assert_eq!(case.mu, 1.0);
        }
        if case.mu_c > 1.0 {
            let tol = case.mu_c * cond(block(split..split + 9));
            assert!(
                (mu_c - case.mu_c).abs() <= tol,
                "{}: mu_c {mu_c} vs {}",
                case.name,
                case.mu_c
            );
        } else {
            assert_eq!(mu_c, 1.0, "{}: mu_c floors at 1", case.name);
            assert_eq!(case.mu_c, 1.0);
        }
        eprintln!(
            "{}: mu {mu} (ASE {}), mu_c {mu_c} (ASE {})",
            case.name, case.mu, case.mu_c
        );
    }
}

/// ASE's matrix, rebuilt from rgmin's own `μ` and `μ_c`.
fn assemble(case: &Case) -> Exp {
    let x0 = start(case);
    let (mu, mu_c) = own_scales(case);
    Exp::assemble_cell(
        x0.view(),
        &case.neighbours,
        &params(case),
        mu,
        mu_c,
        case.r_nn,
    )
    .unwrap()
}

#[test]
fn the_matrix_has_the_ase_entries_and_a_mu_c_cell_block() {
    for case in &load() {
        let dense = assemble(case).to_dense();
        let dim = 3 * case.natoms + 9;
        assert_eq!(dense.dim(), (dim, dim));
        let mut want = ndarray::Array2::<f64>::zeros((dim, dim));
        for &(i, j, v) in &case.entries {
            want[(i, j)] = v;
        }
        let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let tol = 64.0 * U * scale;
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (dense[(i, j)] - want[(i, j)]).abs() <= tol,
                    "{} P[{i}][{j}]: {} vs {}",
                    case.name,
                    dense[(i, j)],
                    want[(i, j)]
                );
            }
        }
        // The diagonal is the μ_c rgmin estimated. ASE's stored value
        // agrees to the same rounding bound as the rest of P.
        for k in 3 * case.natoms..dim {
            assert!(
                (dense[(k, k)] - case.mu_c).abs() <= tol,
                "{} cell diagonal: {} vs {}",
                case.name,
                dense[(k, k)],
                case.mu_c
            );
        }
    }
}

/// Bound on `||y - P^-1 q||_inf`: both solves leave a relative residual
/// of at most `SOLVE_RTOL` plus the rounding of a residual evaluation,
/// and `P` is SPD with `λ_min >= min(μ c_stab, μ_c)` (a graph Laplacian
/// plus those diagonals) and `λ_max <=` the largest absolute row sum.
fn solve_tolerance(case: &Case, q: ArrayView1<f64>) -> f64 {
    let dim = 3 * case.natoms + 9;
    let mut row_sum = vec![0.0_f64; dim];
    for &(i, _, v) in &case.entries {
        row_sum[i] += v.abs();
    }
    let lambda_max = row_sum.iter().fold(0.0_f64, |m, s| m.max(*s));
    let lambda_min = (case.mu * case.c_stab).min(case.mu_c);
    let residual = SOLVE_RTOL + 64.0 * U * lambda_max / lambda_min;
    let qn = crate::vecops::nrm2(q);
    2.0 * residual * qn / lambda_min
}

#[test]
fn a_solve_agrees_with_the_sparse_direct_solve() {
    for case in &load() {
        let exp = assemble(case);
        let (q, want) = &case.solve;
        let y = exp.solve(q.view()).unwrap();
        let tol = solve_tolerance(case, q.view());
        for i in 0..y.len() {
            assert!(
                (y[i] - want[i]).abs() <= tol,
                "{} y[{i}]: {} vs {} (tol {tol})",
                case.name,
                y[i],
                want[i]
            );
        }
    }
}

fn pairs_of(case: &Case, step: &Step) -> LbfgsPairs {
    let dim = 3 * case.natoms + 9;
    let mut pairs = LbfgsPairs::new(dim, 100).unwrap();
    for (s, y, rho) in &step.pairs {
        assert!(pairs.push_rho(s.view(), y.view(), *rho).unwrap());
    }
    pairs
}

#[test]
fn the_direction_replays_ase_with_its_own_solves() {
    for case in &load() {
        for (k, step) in case.steps.iter().enumerate() {
            let pairs = pairs_of(case, step);
            let used = std::cell::Cell::new(0_usize);
            let replay = |v: ArrayView1<f64>| -> crate::Result<Array1<f64>> {
                let (q, z) = &step.solves[used.get()];
                used.set(used.get() + 1);
                let scale = max_abs(q.view());
                for i in 0..v.len() {
                    assert!(
                        (v[i] - q[i]).abs() <= 4096.0 * U * scale,
                        "{} step {k} q",
                        case.name
                    );
                }
                Ok(z.clone())
            };
            let d = pairs
                .direction(step.g.view(), LbfgsH0::Apply(&replay))
                .unwrap();
            assert_eq!(used.get(), 1);
            let m = step.pairs.len() as f64;
            let n = d.len() as f64;
            let growth = step.pairs.iter().fold(1.0, |acc, (s, y, rho)| {
                acc * (1.0
                    + rho.abs() * crate::vecops::nrm2(s.view()) * crate::vecops::nrm2(y.view()))
            });
            let scale = max_abs(step.dir.view()).max(max_abs(step.g.view()) * growth);
            let tol = 4.0 * (m + 1.0) * (n + 2.0) * U * scale;
            for i in 0..d.len() {
                assert!(
                    (d[i] - step.dir[i]).abs() <= tol,
                    "{} step {k} d[{i}]: {} vs {} (tol {tol})",
                    case.name,
                    d[i],
                    step.dir[i]
                );
            }
        }
    }
}

#[test]
fn the_direction_with_rgmins_own_solve_agrees_to_the_solve_error() {
    for case in &load() {
        let exp = assemble(case);
        for (k, step) in case.steps.iter().enumerate() {
            let pairs = pairs_of(case, step);
            let h0 =
                |v: ArrayView1<f64>| -> crate::Result<Array1<f64>> { Ok(exp.solve(v).unwrap()) };
            let d = pairs.direction(step.g.view(), LbfgsH0::Apply(&h0)).unwrap();
            let (q, _) = &step.solves[0];
            let growth = step.pairs.iter().fold(1.0, |acc, (s, y, rho)| {
                acc * (1.0
                    + rho.abs() * crate::vecops::nrm2(s.view()) * crate::vecops::nrm2(y.view()))
            });
            let m = step.pairs.len() as f64;
            let n = d.len() as f64;
            let scale = max_abs(step.dir.view()).max(max_abs(step.g.view()) * growth);
            let tol =
                growth * solve_tolerance(case, q.view()) + 4.0 * (m + 1.0) * (n + 2.0) * U * scale;
            for i in 0..d.len() {
                assert!(
                    (d[i] - step.dir[i]).abs() <= tol,
                    "{} step {k} d[{i}]: {} vs {} (tol {tol})",
                    case.name,
                    d[i],
                    step.dir[i]
                );
            }
        }
    }
}
