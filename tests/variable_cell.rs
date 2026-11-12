//! Variable-cell chart: frozen identity, finite-difference forces, and
//! a nearest-neighbour Lennard-Jones fcc crystal.

use rgmin::{
    CellOptions, CellSample, Control, FireKind, LbfgsStep, Method, PeriodicPotential, StopNorm,
    StrainMask, applied_stress, cell_jacobian, relax_cell, relax_fixed,
};

const EPSILON: f64 = 1.0;
const SIGMA: f64 = 1.0;
const RC: f64 = 1.4;

fn r_min() -> f64 {
    2.0_f64.powf(1.0 / 6.0)
}

fn a_fcc() -> f64 {
    r_min() * 2.0_f64.sqrt()
}

struct Lj;

impl PeriodicPotential for Lj {
    fn evaluate(&mut self, positions: &[f64], cell: &[f64; 9]) -> rgmin::Result<CellSample> {
        Ok(lj_sample(positions, cell))
    }
}

fn lj_sample(positions: &[f64], cell: &[f64; 9]) -> CellSample {
    let n = positions.len() / 3;
    let mut energy = 0.0;
    let mut forces = vec![0.0; positions.len()];
    let mut virial = [0.0; 9];
    let nmax = image_reach(cell, RC);
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
                        let phi = 4.0 * EPSILON * (sr12 - sr6);
                        let dphi = 24.0 * EPSILON * (-2.0 * sr12 + sr6) * inv_r;
                        energy += 0.5 * phi;
                        let scale = dphi * inv_r;
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

fn image_reach(cell: &[f64; 9], rc: f64) -> i32 {
    let inverse = invert(cell).unwrap_or([0.0; 9]);
    let mut reach = 0.0_f64;
    for row in 0..3 {
        let norm = (inverse[row * 3].powi(2)
            + inverse[row * 3 + 1].powi(2)
            + inverse[row * 3 + 2].powi(2))
        .sqrt();
        reach = reach.max(norm * rc);
    }
    reach.ceil() as i32
}

fn invert(m: &[f64; 9]) -> Option<[f64; 9]> {
    let d = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if !d.is_finite() || d.abs() < 1e-14 {
        return None;
    }
    Some([
        (m[4] * m[8] - m[5] * m[7]) / d,
        (m[2] * m[7] - m[1] * m[8]) / d,
        (m[1] * m[5] - m[2] * m[4]) / d,
        (m[5] * m[6] - m[3] * m[8]) / d,
        (m[0] * m[8] - m[2] * m[6]) / d,
        (m[2] * m[3] - m[0] * m[5]) / d,
        (m[3] * m[7] - m[4] * m[6]) / d,
        (m[1] * m[6] - m[0] * m[7]) / d,
        (m[0] * m[4] - m[1] * m[3]) / d,
    ])
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
    let cell = [length, 0.0, 0.0, 0.0, length, 0.0, 0.0, 0.0, length];
    (positions, cell)
}

fn rattle(positions: &mut [f64], amp: f64) {
    for (index, value) in positions.iter_mut().enumerate() {
        let wave = ((index * 17) % 10) as f64 - 4.5;
        *value += amp * wave / 4.5;
    }
}

struct Bowl;

impl PeriodicPotential for Bowl {
    fn evaluate(&mut self, positions: &[f64], _cell: &[f64; 9]) -> rgmin::Result<CellSample> {
        let mut energy = 0.0;
        let mut forces = vec![0.0; positions.len()];
        for (index, value) in positions.iter().enumerate() {
            energy += value * value;
            forces[index] = -2.0 * value;
        }
        Ok(CellSample {
            energy,
            forces,
            virial: [0.0; 9],
        })
    }
}

fn options(method: Method, maxiter: usize, gtol: f64) -> CellOptions {
    let lbfgs = matches!(method, Method::Lbfgs { .. });
    CellOptions {
        method,
        control: Control {
            maxiter,
            gtol,
            istep: 0.05,
            stop_norm: StopNorm::MaxAtom,
            ..Control::default()
        },
        lbfgs_step: lbfgs.then_some(LbfgsStep::Capped { max_atom: 0.2 }),
        ..CellOptions::default()
    }
}

fn bits(value: f64) -> u64 {
    value.to_bits()
}

#[test]
fn frozen_lbfgs_matches_the_fixed_cell_minimizer() {
    let cell = [6.0, 0.0, 0.0, 0.0, 5.5, 0.0, 0.0, 0.0, 5.0];
    let mut positions = vec![0.4, -0.2, 0.15, -0.3, 0.25, -0.05, 0.1, 0.05, -0.2];
    rattle(&mut positions, 0.02);
    let opt = options(Method::Lbfgs { memory: 8 }, 40, 1e-10);
    let mut frozen_opt = opt.clone();
    frozen_opt.mask = StrainMask::Frozen;
    let fixed = relax_fixed(&mut Bowl, &positions, &cell, &opt).unwrap();
    let frozen = relax_cell(&mut Bowl, &positions, &cell, &frozen_opt).unwrap();
    assert_eq!(fixed.positions.len(), frozen.positions.len());
    for (left, right) in fixed.positions.iter().zip(&frozen.positions) {
        assert_eq!(bits(*left), bits(*right));
    }
    assert_eq!(bits(fixed.energy), bits(frozen.energy));
    assert_eq!(bits(fixed.grad_norm), bits(frozen.grad_norm));
    assert_eq!(fixed.steps, frozen.steps);
    assert_eq!(fixed.evaluations, frozen.evaluations);
    assert!(frozen.energy < 1e-16);
    assert_eq!(frozen.cell, cell);
}

#[test]
fn frozen_fire_matches_the_fixed_cell_minimizer() {
    let cell = [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0];
    let positions = vec![0.3, -0.2, 0.1, -0.15, 0.25, -0.05];
    let opt = options(Method::Fire { kind: FireKind::V2 }, 25, 1e-12);
    let mut frozen_opt = opt.clone();
    frozen_opt.mask = StrainMask::Frozen;
    let fixed = relax_fixed(&mut Bowl, &positions, &cell, &opt).unwrap();
    let frozen = relax_cell(&mut Bowl, &positions, &cell, &frozen_opt).unwrap();
    for (left, right) in fixed.positions.iter().zip(&frozen.positions) {
        assert_eq!(bits(*left), bits(*right));
    }
    assert_eq!(bits(fixed.energy), bits(frozen.energy));
    assert_eq!(fixed.steps, frozen.steps);
    assert_eq!(fixed.evaluations, frozen.evaluations);
}

#[test]
fn ideal_fcc_lj_sits_at_the_known_minimum() {
    let (positions, cell) = fcc(1, a_fcc());
    let sample = lj_sample(&positions, &cell);
    assert!(
        (sample.energy - (-24.0)).abs() < 1e-10,
        "energy {}",
        sample.energy
    );
    let force = sample
        .forces
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    assert!(force < 1e-10, "force {force}");
    let stress = sample.stress(cell[0] * cell[4] * cell[8]).unwrap();
    let worst = stress.iter().map(|value| value.abs()).fold(0.0, f64::max);
    assert!(worst < 1e-10, "stress {worst}");
}

#[test]
fn virial_matches_a_finite_difference_strain() {
    let (mut positions, mut cell) = fcc(1, a_fcc() * 0.97);
    rattle(&mut positions, 0.01);
    cell[1] = 0.02;
    cell[3] = 0.02;
    let sample = lj_sample(&positions, &cell);
    let h = 1e-6;
    for index in 0..9 {
        let mut delta = [0.0; 9];
        delta[index] = h;
        let (plus_r, plus_h) = strained(&positions, &cell, &delta);
        delta[index] = -h;
        let (minus_r, minus_h) = strained(&positions, &cell, &delta);
        let derivative =
            (lj_sample(&plus_r, &plus_h).energy - lj_sample(&minus_r, &minus_h).energy) / (2.0 * h);
        let err = (derivative - sample.virial[index]).abs();
        assert!(
            err < 5e-6,
            "component {index}: fd {derivative} virial {} err {err}",
            sample.virial[index]
        );
    }
}

fn strained(positions: &[f64], cell: &[f64; 9], delta: &[f64; 9]) -> (Vec<f64>, [f64; 9]) {
    let mut deform = *delta;
    deform[0] += 1.0;
    deform[4] += 1.0;
    deform[8] += 1.0;
    let mut moved = vec![0.0; positions.len()];
    for atom in (0..positions.len()).step_by(3) {
        let x = positions[atom];
        let y = positions[atom + 1];
        let z = positions[atom + 2];
        moved[atom] = deform[0] * x + deform[1] * y + deform[2] * z;
        moved[atom + 1] = deform[3] * x + deform[4] * y + deform[5] * z;
        moved[atom + 2] = deform[6] * x + deform[7] * y + deform[8] * z;
    }
    let mut next = [0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            next[row * 3 + col] = cell[row * 3] * deform[col * 3]
                + cell[row * 3 + 1] * deform[col * 3 + 1]
                + cell[row * 3 + 2] * deform[col * 3 + 2];
        }
    }
    (moved, next)
}

#[test]
fn forces_match_a_cartesian_finite_difference() {
    let (mut positions, cell) = fcc(1, a_fcc() * 1.02);
    rattle(&mut positions, 0.02);
    let sample = lj_sample(&positions, &cell);
    let h = 1e-6;
    for index in 0..positions.len() {
        let mut plus = positions.clone();
        let mut minus = positions.clone();
        plus[index] += h;
        minus[index] -= h;
        let derivative =
            (lj_sample(&plus, &cell).energy - lj_sample(&minus, &cell).energy) / (2.0 * h);
        let err = (derivative + sample.forces[index]).abs();
        assert!(err < 5e-6, "atom component {index} err {err}");
    }
}

struct Eam;

impl PeriodicPotential for Eam {
    fn evaluate(&mut self, positions: &[f64], cell: &[f64; 9]) -> rgmin::Result<CellSample> {
        Ok(eam_sample(positions, cell))
    }
}

fn eam_sample(positions: &[f64], cell: &[f64; 9]) -> CellSample {
    let n = positions.len() / 3;
    let rc = 2.4;
    let nmax = image_reach(cell, rc);
    let mut rho = vec![0.0; n];
    let mut undirected = Vec::new();
    for i in 0..n {
        for j in i..n {
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
                        if !(1e-24..=rc * rc).contains(&r2) {
                            continue;
                        }
                        let r = r2.sqrt();
                        let psi = (-r).exp();
                        rho[i] += psi;
                        if i != j {
                            rho[j] += psi;
                        }
                        undirected.push((i, j, d, r, psi));
                    }
                }
            }
        }
    }
    let mut energy = 0.0;
    for value in &rho {
        energy += 0.5 * value * value;
    }
    let mut forces = vec![0.0; 3 * n];
    let mut virial = [0.0; 9];
    for &(i, j, d, r, psi) in &undirected {
        let dphi = -2.0 * psi * psi;
        let dpsi = -psi;
        let slope = if i == j {
            dphi + rho[i] * dpsi
        } else {
            dphi + (rho[i] + rho[j]) * dpsi
        };
        energy += psi * psi;
        let scale = slope / r;
        if i != j {
            for axis in 0..3 {
                forces[3 * i + axis] += scale * d[axis];
                forces[3 * j + axis] -= scale * d[axis];
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                virial[a * 3 + b] += scale * d[a] * d[b];
            }
        }
    }
    CellSample {
        energy,
        forces,
        virial,
    }
}

#[test]
fn embedding_virial_matches_a_finite_difference() {
    let (mut positions, cell) = fcc(1, 2.0);
    rattle(&mut positions, 0.03);
    let sample = eam_sample(&positions, &cell);
    let h = 1e-6;
    for index in [0, 4, 8, 1, 5] {
        let mut delta = [0.0; 9];
        delta[index] = h;
        let (plus_r, plus_h) = strained(&positions, &cell, &delta);
        delta[index] = -h;
        let (minus_r, minus_h) = strained(&positions, &cell, &delta);
        let derivative = (eam_sample(&plus_r, &plus_h).energy
            - eam_sample(&minus_r, &minus_h).energy)
            / (2.0 * h);
        let err = (derivative - sample.virial[index]).abs();
        assert!(
            err < 2e-5,
            "component {index}: fd {derivative} virial {} err {err}",
            sample.virial[index]
        );
    }
}

#[test]
fn generalized_cell_force_matches_a_finite_difference() {
    let (positions, cell) = fcc(1, a_fcc() * 1.03);
    let mut opt = options(Method::lbfgs(), 1, 1e-12);
    opt.pressure = 0.02;
    opt.external_stress = [0.0, 0.01, 0.0, 0.0, -0.015, 0.0, 0.0, 0.0, 0.005];
    let chart = rgmin::Chart::new(positions.len() / 3, &cell, &opt).unwrap();
    let mut y = chart.origin(&positions).unwrap();
    let n3 = positions.len();
    let jacobian = chart.jacobian();
    let strain = [
        0.01, -0.004, 0.002, 0.003, -0.02, 0.001, -0.002, 0.004, 0.015,
    ];
    for (slot, value) in strain.iter().enumerate() {
        y[n3 + slot] = jacobian * value;
    }
    let probe = chart.evaluate(&mut Lj, &y).unwrap();
    let h = 1e-7;
    for index in 0..y.len() {
        let mut plus = y.clone();
        let mut minus = y.clone();
        plus[index] += h;
        minus[index] -= h;
        let e_plus = chart.evaluate(&mut Lj, &plus).unwrap().enthalpy;
        let e_minus = chart.evaluate(&mut Lj, &minus).unwrap().enthalpy;
        let derivative = (e_plus - e_minus) / (2.0 * h);
        let err = (derivative - probe.gradient[index]).abs();
        assert!(
            err < 2e-5,
            "coordinate {index}: fd {derivative} grad {} err {err}",
            probe.gradient[index]
        );
    }
    let at_reference = chart
        .evaluate(&mut Lj, &chart.origin(&positions).unwrap())
        .unwrap();
    let volume = at_reference.geometry.volume;
    let stress = at_reference.sample.stress(volume).unwrap();
    let applied = applied_stress(opt.pressure, &opt.external_stress);
    let scale = volume / cell_jacobian(volume, positions.len() / 3);
    for index in 0..9 {
        let expect = -scale * (stress[index] - applied[index]);
        let err = (at_reference.cell_force()[index] - expect).abs();
        assert!(err < 1e-9, "reference force {index} err {err}");
    }
}

#[test]
fn fcc_relaxes_to_the_known_lattice_constant() {
    let (mut positions, cell) = fcc(1, a_fcc() * 1.04);
    rattle(&mut positions, 0.015);
    let opt = options(Method::Lbfgs { memory: 8 }, 80, 1e-6);
    let got = relax_cell(&mut Lj, &positions, &cell, &opt).unwrap();
    assert!(got.grad_norm < 1e-6, "grad {}", got.grad_norm);
    let edge = (got.cell[0].powi(2) + got.cell[1].powi(2) + got.cell[2].powi(2)).sqrt();
    assert!(
        (edge - a_fcc()).abs() < 2e-4,
        "a = {edge}, expected {}",
        a_fcc()
    );
    let worst = got
        .stress
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-5, "stress {worst}");
    assert!(got.positions.len() == 12);
}

#[test]
fn hydrostatic_mask_keeps_the_cell_cubic() {
    let (mut positions, cell) = fcc(1, a_fcc() * 1.05);
    rattle(&mut positions, 0.01);
    let mut opt = options(Method::Lbfgs { memory: 8 }, 80, 1e-6);
    opt.mask = StrainMask::Hydrostatic;
    let got = relax_cell(&mut Lj, &positions, &cell, &opt).unwrap();
    assert!(got.grad_norm < 1e-6, "grad {}", got.grad_norm);
    let lengths = [got.cell[0], got.cell[4], got.cell[8]];
    for length in lengths {
        assert!((length - a_fcc()).abs() < 5e-4, "edge {length}");
    }
    assert_eq!(bits(got.cell[1]), bits(0.0));
    assert_eq!(bits(got.strain[0]), bits(got.strain[4]));
    assert_eq!(bits(got.strain[4]), bits(got.strain[8]));
}

#[test]
fn lattice_mask_holds_the_frozen_vector() {
    let (positions, cell) = fcc(1, a_fcc() * 1.03);
    let mut free = [false; 9];
    free[0] = true;
    free[4] = true;
    let mut opt = options(Method::Lbfgs { memory: 8 }, 60, 1e-7);
    opt.mask = StrainMask::Lattice(free);
    let got = relax_cell(&mut Lj, &positions, &cell, &opt).unwrap();
    assert_eq!(bits(got.cell[6]), bits(cell[6]));
    assert_eq!(bits(got.cell[7]), bits(cell[7]));
    assert_eq!(bits(got.cell[8]), bits(cell[8]));
    assert_eq!(bits(got.cell[1]), bits(0.0));
    assert_eq!(bits(got.cell[2]), bits(0.0));
    assert!(got.grad_norm < 1e-6, "grad {}", got.grad_norm);
    assert!(
        (got.cell[0] - cell[0]).abs() > 1e-3,
        "the free edge did not move: {}",
        got.cell[0]
    );
}

#[test]
fn external_pressure_reaches_the_expected_strain() {
    let pressure = 0.05;
    let expected = pressured_nn_distance(pressure);
    let (positions, cell) = fcc(1, a_fcc());
    let mut opt = options(Method::Lbfgs { memory: 8 }, 80, 1e-8);
    opt.pressure = pressure;
    opt.mask = StrainMask::Hydrostatic;
    let got = relax_cell(&mut Lj, &positions, &cell, &opt).unwrap();
    assert!(got.grad_norm < 1e-8, "grad {}", got.grad_norm);
    let edge = got.cell[0];
    let nn = edge / 2.0_f64.sqrt();
    assert!(
        (nn - expected).abs() < 2e-4,
        "nn {nn}, expected {expected}, a {edge}"
    );
    for index in [0, 4, 8] {
        assert!(
            (got.stress[index] + pressure).abs() < 2e-5,
            "stress {} pressure {pressure}",
            got.stress[index]
        );
    }
}

fn pressured_nn_distance(pressure: f64) -> f64 {
    let n = 4.0;
    let mut r = r_min() * 0.98;
    for _ in 0..20 {
        let (f, df) = enthalpy_slope(n, r, pressure);
        r -= f / df;
    }
    r
}

fn enthalpy_slope(n: f64, r: f64, pressure: f64) -> (f64, f64) {
    let sr6 = (SIGMA / r).powi(6);
    let sr12 = sr6 * sr6;
    let d_e = n * 24.0 * EPSILON * (-12.0 * sr12 + 6.0 * sr6) / r;
    let d_v = n * 3.0 * r * r / 2.0_f64.sqrt();
    let d2_e = n * 24.0 * EPSILON * (12.0 * 13.0 * sr12 - 6.0 * 7.0 * sr6) / (r * r);
    let d2_v = n * 6.0 * r / 2.0_f64.sqrt();
    (d_e + pressure * d_v, d2_e + pressure * d2_v)
}

#[test]
fn vacancy_and_dumbbell_report_stress() {
    let a = 2.0;
    let (crystal, cell) = fcc(2, a);
    let vacancy: Vec<f64> = crystal.iter().skip(3).copied().collect();
    let bare = eam_sample(&vacancy, &cell);
    let volume = cell[0] * cell[4] * cell[8];
    let bare_stress = bare.stress(volume).unwrap();
    let off = bare_stress[1].abs()
        + bare_stress[2].abs()
        + bare_stress[3].abs()
        + bare_stress[5].abs()
        + bare_stress[6].abs()
        + bare_stress[7].abs();
    assert!(off < 1e-8, "unrelaxed vacancy shear {off}");
    assert!(
        (bare_stress[0] - bare_stress[4]).abs() < 1e-8
            && (bare_stress[4] - bare_stress[8]).abs() < 1e-8
    );
    assert!(bare_stress[0].abs() > 1e-6, "vacancy stress vanished");

    let mut opt = options(Method::Lbfgs { memory: 8 }, 30, 1e-3);
    opt.mask = StrainMask::Frozen;
    let vacancy_min = relax_cell(&mut Eam, &vacancy, &cell, &opt).unwrap();
    assert!(vacancy_min.stress.iter().all(|value| value.is_finite()));
    let dipole = vacancy_min.dipole();
    for (value, stress) in dipole.iter().zip(vacancy_min.stress) {
        let expect = vacancy_min.volume * stress;
        assert!((value - expect).abs() < 1e-8);
    }
    assert!(vacancy_min.stress[0].abs() > 1e-6);

    let mut dumb = crystal.clone();
    let split = 0.3 * a;
    dumb[2] = split;
    dumb.extend_from_slice(&[0.0, 0.0, -split]);
    let bare_dumb = eam_sample(&dumb, &cell).stress(volume).unwrap();
    assert!(
        (bare_dumb[8] - bare_dumb[0]).abs() > 1e-3,
        "unrelaxed dumbbell {bare_dumb:?}"
    );
    let dumbbell = relax_cell(&mut Eam, &dumb, &cell, &opt).unwrap();
    assert!(dumbbell.stress.iter().all(|value| value.is_finite()));
    let gap = (dumbbell.stress[8] - dumbbell.stress[0]).abs();
    let stress = dumbbell.stress;
    assert!(gap > 1e-3, "dumbbell stress {stress:?}");
}
