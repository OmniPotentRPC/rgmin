//! A periodic Lennard-Jones crystal with analytic forces and stress, for
//! the variable-cell tests. The pair energy is switched to zero between
//! `R_ON` and `R_OFF` with the quintic smoothstep, so the energy, the
//! forces and the stress are continuous as pairs cross the cutoff.

#![allow(dead_code)]

use ndarray::{Array1, ArrayView1};
use rgmin::{Cell, CellEvaluation, Neighbours};

pub const EPSILON: f64 = 0.5;
pub const SIGMA: f64 = 2.2717;
pub const R_ON: f64 = 4.0;
pub const R_OFF: f64 = 5.0;

pub fn fcc(cells: usize, a: f64) -> (Array1<f64>, Cell) {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut positions = Vec::new();
    for i in 0..cells {
        for j in 0..cells {
            for k in 0..cells {
                for b in &basis {
                    positions.push(a * (i as f64 + b[0]));
                    positions.push(a * (j as f64 + b[1]));
                    positions.push(a * (k as f64 + b[2]));
                }
            }
        }
    }
    let l = a * cells as f64;
    (
        Array1::from(positions),
        [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]],
    )
}

/// Deterministic displacement in `[-amplitude, amplitude]`.
pub fn rattle(positions: &mut Array1<f64>, amplitude: f64, seed: u64) {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    for v in positions.iter_mut() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        *v += amplitude * (2.0 * u - 1.0);
    }
}

/// `cell` applied to a strain `(I + e)`: rows of `cell` times `(I + e)^T`.
#[allow(clippy::needless_range_loop)]
pub fn strained(cell: &Cell, e: [[f64; 3]; 3]) -> Cell {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let m = if j == k { 1.0 } else { 0.0 } + e[j][k];
                out[i][j] += cell[i][k] * m;
            }
        }
    }
    out
}

pub fn strain_positions(positions: &Array1<f64>, e: [[f64; 3]; 3]) -> Array1<f64> {
    let mut out = positions.clone();
    for a in 0..positions.len() / 3 {
        for i in 0..3 {
            let mut s = positions[3 * a + i];
            for j in 0..3 {
                s += e[i][j] * positions[3 * a + j];
            }
            out[3 * a + i] = s;
        }
    }
    out
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn volume(cell: &Cell) -> f64 {
    dot(cell[0], cross(cell[1], cell[2])).abs()
}

/// Image ranges that cover every lattice translation within `reach`.
fn image_range(cell: &Cell, reach: f64) -> [i64; 3] {
    let v = volume(cell);
    let mut n = [0i64; 3];
    for k in 0..3 {
        let c = cross(cell[(k + 1) % 3], cell[(k + 2) % 3]);
        let width = v / dot(c, c).sqrt();
        n[k] = (reach / width).ceil() as i64;
    }
    n
}

#[allow(clippy::needless_range_loop)]
fn images(cell: &Cell, reach: f64) -> Vec<[f64; 3]> {
    let n = image_range(cell, reach);
    let mut out = Vec::new();
    for a in -n[0]..=n[0] {
        for b in -n[1]..=n[1] {
            for c in -n[2]..=n[2] {
                let mut t = [0.0; 3];
                for d in 0..3 {
                    t[d] = a as f64 * cell[0][d] + b as f64 * cell[1][d] + c as f64 * cell[2][d];
                }
                out.push(t);
            }
        }
    }
    out
}

/// Pair energy and its radial derivative.
fn pair(r: f64) -> (f64, f64) {
    let s6 = (SIGMA / r).powi(6);
    let phi = 4.0 * EPSILON * (s6 * s6 - s6);
    let dphi = 4.0 * EPSILON * (-12.0 * s6 * s6 + 6.0 * s6) / r;
    if r <= R_ON {
        return (phi, dphi);
    }
    let w = R_OFF - R_ON;
    let t = (r - R_ON) / w;
    let s = 1.0 - t * t * t * (10.0 - 15.0 * t + 6.0 * t * t);
    let ds = -30.0 * t * t * (1.0 - t) * (1.0 - t) / w;
    (phi * s, dphi * s + phi * ds)
}

pub fn evaluate(positions: ArrayView1<f64>, cell: &Cell) -> CellEvaluation {
    let n = positions.len() / 3;
    let shifts = images(cell, R_OFF);
    let mut energy = 0.0;
    let mut forces = Array1::<f64>::zeros(positions.len());
    let mut virial = [[0.0; 3]; 3];
    for i in 0..n {
        for j in 0..n {
            for t in &shifts {
                if i == j && t.iter().all(|&c| c == 0.0) {
                    continue;
                }
                let mut d = [0.0; 3];
                for a in 0..3 {
                    d[a] = positions[3 * j + a] + t[a] - positions[3 * i + a];
                }
                let r = dot(d, d).sqrt();
                if r >= R_OFF {
                    continue;
                }
                let (phi, dphi) = pair(r);
                energy += 0.5 * phi;
                for a in 0..3 {
                    forces[3 * i + a] += dphi * d[a] / r;
                    for b in 0..3 {
                        virial[a][b] += 0.5 * dphi * d[a] * d[b] / r;
                    }
                }
            }
        }
    }
    let v = volume(cell);
    let mut stress = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            stress[a][b] = virial[a][b] / v;
        }
    }
    CellEvaluation::new(energy, forces, stress)
}

/// Minimum-image pair list within `r_cut`, with the pair distances.
/// Valid while no pair has two images inside `r_cut`.
pub fn neighbours(positions: ArrayView1<f64>, cell: &Cell, r_cut: f64) -> Neighbours {
    let n = positions.len() / 3;
    let shifts = images(cell, r_cut);
    let mut nb = Neighbours::default();
    for i in 0..n {
        for j in (i + 1)..n {
            for t in &shifts {
                let mut d = [0.0; 3];
                for a in 0..3 {
                    d[a] = positions[3 * j + a] + t[a] - positions[3 * i + a];
                }
                let r = dot(d, d).sqrt();
                if r < r_cut {
                    nb.i.push(i);
                    nb.j.push(j);
                    nb.r.push(r);
                }
            }
        }
    }
    nb
}
