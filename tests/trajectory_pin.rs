//! The capped L-BFGS step, FIRE 1.0 and the eOn conjugate-gradient step
//! stay on the trajectory recorded before the large-N workspace change.
//!
//! `tests/data/traj_pin.txt` is every force evaluation on a 7-atom
//! Lennard-Jones chunk: ten iterations, fmax left at zero so the budget
//! is what stops the run. A coordinate may differ by at most 1e-12.

use ndarray::Array1;
use rgmin::{Control, EonCg, FireKind, Method, Solver, StopNorm, minimize_eon_cg};

const CAP: f64 = 0.2;
const MEMORY: usize = 20;

#[test]
fn recorded_lj7_trajectory_stays_within_1e_12() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/traj_pin.txt");
    let gold = std::fs::read_to_string(path).unwrap();
    let got = pin_lines();
    let gold = gold
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty());
    for (g, mine) in gold.zip(got.iter()) {
        let mut gp = g.split_whitespace();
        let mut mp = mine.split_whitespace();
        assert_eq!(gp.next(), mp.next(), "arm");
        assert_eq!(gp.next(), mp.next(), "call");
        let ge: f64 = gp.next().unwrap().parse().unwrap();
        let me: f64 = mp.next().unwrap().parse().unwrap();
        let mut worst = (ge - me).abs();
        let mut ncoord = 0usize;
        for (a, b) in gp.zip(mp) {
            let ga: f64 = a.parse().unwrap();
            let ma: f64 = b.parse().unwrap();
            worst = worst.max((ga - ma).abs());
            ncoord += 1;
        }
        assert!(
            worst <= 1e-12,
            "{mine} energy {ge:.16e} vs {me:.16e} worst {worst:.3e} coords {ncoord}"
        );
    }
    assert_eq!(got.len(), 12 + 11 + 21);
}

fn pin_lines() -> Vec<String> {
    let x0 = lj7();
    let mut out = Vec::new();
    for arm in ["lbfgs", "fire", "cg"] {
        let mut x = Array1::from(x0.clone());
        let mut cells = Cells::new();
        let mut call = 0usize;
        match arm {
            "cg" => {
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
                        let e = lj(&mut cells, pos, grad);
                        call += 1;
                        out.push(line(arm, call, e, pos));
                        e
                    },
                    &opts,
                );
            }
            _ => {
                let method = if arm == "lbfgs" {
                    Method::Lbfgs { memory: MEMORY }
                } else {
                    Method::Fire { kind: FireKind::V1 }
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
                        let pos = v.as_slice().unwrap();
                        let e = lj(&mut cells, pos, grad.as_slice_mut().unwrap());
                        call += 1;
                        out.push(line(arm, call, e, pos));
                        e
                    })
                    .unwrap();
            }
        }
    }
    out
}

fn line(arm: &str, call: usize, energy: f64, x: &[f64]) -> String {
    let mut s = format!("{arm} {call} {energy:.16e}");
    for v in x {
        s.push_str(&format!(" {v:.16e}"));
    }
    s
}

fn lj7() -> Vec<f64> {
    let a = 1.122462048309373_f64 * std::f64::consts::SQRT_2;
    let mut raw = Vec::new();
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    for iz in 0..2 {
        for iy in 0..2 {
            for ix in 0..2 {
                for b in basis {
                    raw.push([
                        (ix as f64 + b[0]) * a,
                        (iy as f64 + b[1]) * a,
                        (iz as f64 + b[2]) * a,
                    ]);
                }
            }
        }
    }
    raw.truncate(16);
    let mut x = Vec::new();
    for (k, p) in raw.iter().enumerate() {
        for (c, v) in p.iter().enumerate() {
            x.push(v + 0.02 * ((k * 3 + c) as f64 * 1.7).sin());
        }
    }
    x.truncate(21);
    x
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

fn lj(cells: &mut Cells, x: &[f64], g: &mut [f64]) -> f64 {
    g.fill(0.0);
    let rc = 2.5_f64;
    let rc2 = rc * rc;
    let s6c = (1.0 / rc2).powi(3);
    let shift = 4.0 * (s6c * s6c - s6c);
    let mut e = 0.0;
    // Same cell walk as `examples/large_n.rs`, so the pair order matches
    // the recorded file.
    let (nx, ny, nz) = build_cells(cells, x, rc);
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
                                let jx = ix as isize + dx;
                                let jy = iy as isize + dy;
                                let jz = iz as isize + dz;
                                if jx < 0
                                    || jy < 0
                                    || jz < 0
                                    || jx >= nx as isize
                                    || jy >= ny as isize
                                    || jz >= nz as isize
                                {
                                    continue;
                                }
                                let slot = (jz as usize * ny + jy as usize) * nx + jx as usize;
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
                                        d[c] = x[3 * i + c] - x[3 * j + c];
                                        r2 += d[c] * d[c];
                                    }
                                    if r2 < rc2 && r2 > 1.0e-16 {
                                        let s6 = r2.powi(-3);
                                        e += 4.0 * (s6 * s6 - s6) - shift;
                                        let du = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
                                        for c in 0..3 {
                                            let t = du * d[c];
                                            g[3 * i + c] += t;
                                            g[3 * j + c] -= t;
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
    e
}

fn build_cells(cells: &mut Cells, x: &[f64], rc: f64) -> (usize, usize, usize) {
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
    for a in 0..3 {
        origin[a] = lo[a];
        let span = (hi[a] - lo[a]).max(0.0);
        ncell[a] = (span / rc).floor() as usize + 1;
    }
    let ntot = ncell[0] * ncell[1] * ncell[2];
    cells.head.clear();
    cells.head.resize(ntot, -1);
    if cells.next.len() != n {
        cells.next.resize(n, -1);
    }
    let inv = 1.0 / rc;
    for i in 0..n {
        let mut idx = [0usize; 3];
        for a in 0..3 {
            let t = (x[3 * i + a] - origin[a]) * inv;
            let mut k = t.floor() as isize;
            if k < 0 {
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
    (ncell[0], ncell[1], ncell[2])
}
