// Copyright (c) rgmin developers

//! Force calls and wall time of a variable-cell relaxation.
//!
//! Nearest-neighbour Lennard-Jones, `epsilon = 1`, `sigma = 1`,
//! cutoff 1.4. The fcc nearest-neighbour distance at the pair minimum
//! is `2^(1/6)`, and the cubic lattice constant is that times `sqrt(2)`.
//! Three starts: a rattled 4-atom cell expanded by 4 percent, the same
//! cell with an `xy` shear of 0.04, and a 31-atom vacancy in a 2×2×2
//! cell compressed by 3 percent. Each arm is capped L-BFGS or FIRE 2.0.
//! The stop is the largest atomic block of the generalized gradient
//! below `1e-3`, at most 300 steps. Five serial repeats. The table is
//! the median wall time. Force calls are the potential samples,
//! including the stress sample at the end.
//!
//! `cargo run --release --example bench_cell`

use std::time::Instant;

use rgmin::{
    CellMinimum, CellOptions, CellSample, Control, FireKind, LbfgsStep, Method, PeriodicPotential,
    StopNorm, relax_cell,
};

const REPEATS: usize = 5;
const EPSILON: f64 = 1.0;
const SIGMA: f64 = 1.0;
const RC: f64 = 1.4;

struct Lj {
    calls: usize,
    last_forces: Vec<f64>,
}

impl PeriodicPotential for Lj {
    fn evaluate(&mut self, positions: &[f64], cell: &[f64; 9]) -> rgmin::Result<CellSample> {
        self.calls += 1;
        let sample = lj(positions, cell);
        self.last_forces.clone_from(&sample.forces);
        Ok(sample)
    }
}

fn r_min() -> f64 {
    2.0_f64.powf(1.0 / 6.0)
}

fn a_fcc() -> f64 {
    r_min() * 2.0_f64.sqrt()
}

fn main() {
    let systems = [
        ("bulk4", bulk(1.04, 0.02)),
        ("shear4", shear(0.04)),
        ("vacancy31", vacancy()),
    ];
    let arms = [
        ("lbfgs", Method::Lbfgs { memory: 8 }, true),
        ("fire2", Method::Fire { kind: FireKind::V2 }, false),
    ];
    println!("system arm calls seconds max_force max_stress energy converged");
    for (name, (positions, cell)) in systems {
        let mut pot = Lj {
            calls: 0,
            last_forces: Vec::new(),
        };
        let sample = pot.evaluate(&positions, &cell).unwrap();
        println!(
            "# {name} start_energy {:.12} start_calls {}",
            sample.energy, pot.calls
        );
        for (arm, method, lbfgs) in &arms {
            let mut seconds = Vec::new();
            let mut calls = Vec::new();
            let mut last: Option<Row> = None;
            for _ in 0..REPEATS {
                let mut pot = Lj {
                    calls: 0,
                    last_forces: Vec::new(),
                };
                let opt = options(method.clone(), *lbfgs);
                let start = Instant::now();
                let got = relax_cell(&mut pot, &positions, &cell, &opt).unwrap();
                let elapsed = start.elapsed().as_secs_f64();
                seconds.push(elapsed);
                calls.push(pot.calls);
                last = Some(row(&got, &pot.last_forces));
            }
            seconds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = seconds[REPEATS / 2];
            let row = last.unwrap();
            let same = calls.iter().all(|count| *count == calls[0]);
            println!(
                "{name} {arm} {} {:.6e} {:.6e} {:.6e} {:.8} {} calls_stable {same}",
                calls[0], median, row.max_force, row.max_stress, row.energy, row.converged
            );
        }
    }
}

struct Row {
    max_force: f64,
    max_stress: f64,
    energy: f64,
    converged: bool,
}

fn row(got: &CellMinimum, forces: &[f64]) -> Row {
    let mut max_force = 0.0_f64;
    let mut atom = 0;
    while atom + 3 <= forces.len() {
        let norm =
            (forces[atom].powi(2) + forces[atom + 1].powi(2) + forces[atom + 2].powi(2)).sqrt();
        max_force = max_force.max(norm);
        atom += 3;
    }
    let max_stress = got
        .stress
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    Row {
        max_force,
        max_stress,
        energy: got.energy,
        converged: max_force < 1e-3 && max_stress < 1e-4,
    }
}

fn options(method: Method, lbfgs: bool) -> CellOptions {
    CellOptions {
        method,
        control: Control {
            maxiter: 300,
            gtol: 1e-3,
            istep: 0.1,
            stop_norm: StopNorm::MaxAtom,
            ..Control::default()
        },
        lbfgs_step: lbfgs.then_some(LbfgsStep::Capped { max_atom: 0.2 }),
        ..CellOptions::default()
    }
}

fn bulk(scale: f64, amp: f64) -> (Vec<f64>, [f64; 9]) {
    let (mut positions, cell) = fcc(1, a_fcc() * scale);
    for (index, value) in positions.iter_mut().enumerate() {
        let wave = ((index * 17) % 10) as f64 - 4.5;
        *value += amp * wave / 4.5;
    }
    (positions, cell)
}

fn shear(eps: f64) -> (Vec<f64>, [f64; 9]) {
    let (positions, cell) = fcc(1, a_fcc());
    let mut sheared = cell;
    sheared[1] = eps * cell[0];
    (positions, sheared)
}

fn vacancy() -> (Vec<f64>, [f64; 9]) {
    let (positions, cell) = fcc(2, a_fcc() * 0.97);
    (positions[3..].to_vec(), cell)
}

fn fcc(n_rep: usize, a: f64) -> (Vec<f64>, [f64; 9]) {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut positions = Vec::new();
    for ix in 0..n_rep {
        for iy in 0..n_rep {
            for iz in 0..n_rep {
                for site in basis {
                    positions.push((ix as f64 + site[0]) * a);
                    positions.push((iy as f64 + site[1]) * a);
                    positions.push((iz as f64 + site[2]) * a);
                }
            }
        }
    }
    let length = n_rep as f64 * a;
    (
        positions,
        [length, 0.0, 0.0, 0.0, length, 0.0, 0.0, 0.0, length],
    )
}

fn lj(positions: &[f64], cell: &[f64; 9]) -> CellSample {
    let n = positions.len() / 3;
    let nmax = image_reach(cell);
    let mut energy = 0.0;
    let mut forces = vec![0.0; positions.len()];
    let mut virial = [0.0; 9];
    for i in 0..n {
        for j in 0..n {
            for nx in -nmax..=nmax {
                for ny in -nmax..=nmax {
                    for nz in -nmax..=nmax {
                        if i == j && nx == 0 && ny == 0 && nz == 0 {
                            continue;
                        }
                        let shift = image_shift(cell, nx, ny, nz);
                        let mut d = [0.0; 3];
                        let mut r2 = 0.0;
                        for axis in 0..3 {
                            d[axis] =
                                positions[3 * j + axis] + shift[axis] - positions[3 * i + axis];
                            r2 += d[axis] * d[axis];
                        }
                        if !(1e-24..=RC * RC).contains(&r2) {
                            continue;
                        }
                        let r = r2.sqrt();
                        let inv_r = 1.0 / r;
                        let sr6 = (SIGMA * inv_r).powi(6);
                        let sr12 = sr6 * sr6;
                        energy += 0.5 * 4.0 * EPSILON * (sr12 - sr6);
                        let scale = 24.0 * EPSILON * (-2.0 * sr12 + sr6) * inv_r * inv_r;
                        for a in 0..3 {
                            forces[3 * i + a] += scale * d[a];
                            for b in 0..3 {
                                virial[a * 3 + b] += 0.5 * scale * d[a] * d[b];
                            }
                        }
                    }
                }
            }
        }
    }
    CellSample {
        energy,
        forces,
        virial,
    }
}

fn image_shift(cell: &[f64; 9], nx: i32, ny: i32, nz: i32) -> [f64; 3] {
    let n = [nx as f64, ny as f64, nz as f64];
    let mut shift = [0.0; 3];
    for axis in 0..3 {
        shift[axis] = n[0] * cell[axis] + n[1] * cell[3 + axis] + n[2] * cell[6 + axis];
    }
    shift
}

fn image_reach(cell: &[f64; 9]) -> i32 {
    let inverse = invert(cell);
    let mut reach = 0.0_f64;
    for row in 0..3 {
        let norm = (inverse[row * 3].powi(2)
            + inverse[row * 3 + 1].powi(2)
            + inverse[row * 3 + 2].powi(2))
        .sqrt();
        reach = reach.max(norm * RC);
    }
    reach.ceil() as i32
}

fn invert(m: &[f64; 9]) -> [f64; 9] {
    let d = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    [
        (m[4] * m[8] - m[5] * m[7]) / d,
        (m[2] * m[7] - m[1] * m[8]) / d,
        (m[1] * m[5] - m[2] * m[4]) / d,
        (m[5] * m[6] - m[3] * m[8]) / d,
        (m[0] * m[8] - m[2] * m[6]) / d,
        (m[2] * m[3] - m[0] * m[5]) / d,
        (m[3] * m[7] - m[4] * m[6]) / d,
        (m[1] * m[6] - m[0] * m[7]) / d,
        (m[0] * m[4] - m[1] * m[3]) / d,
    ]
}
