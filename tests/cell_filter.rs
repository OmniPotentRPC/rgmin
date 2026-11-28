//! Variable-cell relaxation against an analytic nearest-neighbour FCC crystal.
//!
//! The pair potential is the shifted Lennard-Jones potential used by the
//! other tests (`epsilon = 1`, `sigma = 1`) with a cutoff between the first
//! and second FCC shells. At zero pressure every nearest neighbour sits at
//! `2^(1/6)`, so the cubic lattice constant is `2^(2/3)`.

#![allow(clippy::needless_range_loop)]

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::Array1;
use rgmin::{
    CellCoordinates, CellFilter, CellSample, Control, FireKind, LineSearch, Method, minimize_method,
};

const EPSILON: f64 = 1.0;
const SIGMA: f64 = 1.0;
const CUTOFF: f64 = 1.40;

fn lattice_constant() -> f64 {
    // `a = sigma * 2^(1/6) * sqrt(2) = sigma * 2^(2/3)`.
    2.0_f64.powf(2.0 / 3.0)
}

fn fcc(edge: f64) -> (Vec<f64>, [f64; 9]) {
    let half = 0.5 * edge;
    let positions = vec![
        0.0, 0.0, 0.0, half, half, 0.0, half, 0.0, half, 0.0, half, half,
    ];
    let cell = [edge, 0.0, 0.0, 0.0, edge, 0.0, 0.0, 0.0, edge];
    (positions, cell)
}

fn flat_sample(positions: &[f64], _cell: &[f64; 9]) -> CellSample {
    CellSample {
        energy: 0.0,
        forces: vec![0.0; positions.len()],
        stress: [0.0; 9],
    }
}

fn rows(cell: &[f64; 9]) -> [[f64; 3]; 3] {
    [
        [cell[0], cell[1], cell[2]],
        [cell[3], cell[4], cell[5]],
        [cell[6], cell[7], cell[8]],
    ]
}

fn flatten(matrix: [[f64; 3]; 3]) -> [f64; 9] {
    [
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
    ]
}

fn mul3(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] =
                left[i][0] * right[0][j] + left[i][1] * right[1][j] + left[i][2] * right[2][j];
        }
    }
    out
}

fn transpose3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = matrix[j][i];
        }
    }
    out
}

fn det3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn inverse3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let inv_det = 1.0 / det3(matrix);
    [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det,
        ],
    ]
}

fn cell_from_deformation(orig: &[f64; 9], deform: &[[f64; 3]; 3]) -> [f64; 9] {
    flatten(mul3(rows(orig), transpose3(*deform)))
}

fn apply_deformation(reference: &[f64], deform: &[[f64; 3]; 3]) -> Vec<f64> {
    let mut out = vec![0.0; reference.len()];
    let natoms = reference.len() / 3;
    for atom in 0..natoms {
        for axis in 0..3 {
            let mut acc = 0.0;
            for source in 0..3 {
                acc += reference[3 * atom + source] * deform[axis][source];
            }
            out[3 * atom + axis] = acc;
        }
    }
    out
}

fn det(cell: &[f64; 9]) -> f64 {
    cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
        - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
        + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6])
}

fn image(cell: &[f64; 9], n: [f64; 3]) -> [f64; 3] {
    [
        n[0] * cell[0] + n[1] * cell[3] + n[2] * cell[6],
        n[0] * cell[1] + n[1] * cell[4] + n[2] * cell[7],
        n[0] * cell[2] + n[1] * cell[5] + n[2] * cell[8],
    ]
}

fn lennard_jones(positions: &[f64], cell: &[f64; 9]) -> CellSample {
    let natoms = positions.len() / 3;
    let mut energy = 0.0;
    let mut forces = vec![0.0; positions.len()];
    let mut stress = [0.0; 9];
    let cutoff2 = CUTOFF * CUTOFF;
    let inv_rc = SIGMA / CUTOFF;
    let shift6 = inv_rc.powi(6);
    let shift = 4.0 * EPSILON * (shift6 * shift6 - shift6);
    for i in 0..natoms {
        for j in (i + 1)..natoms {
            for n0 in -1..=1 {
                for n1 in -1..=1 {
                    for n2 in -1..=1 {
                        let offset = image(cell, [f64::from(n0), f64::from(n1), f64::from(n2)]);
                        let mut bond = [0.0; 3];
                        let mut distance2 = 0.0;
                        for axis in 0..3 {
                            bond[axis] =
                                positions[3 * j + axis] - positions[3 * i + axis] + offset[axis];
                            distance2 += bond[axis] * bond[axis];
                        }
                        if distance2 >= cutoff2 || distance2 < 1.0e-24 {
                            continue;
                        }
                        let inv_r2 = 1.0 / distance2;
                        let sr2 = SIGMA * SIGMA * inv_r2;
                        let sr6 = sr2 * sr2 * sr2;
                        let sr12 = sr6 * sr6;
                        energy += 4.0 * EPSILON * (sr12 - sr6) - shift;
                        // `(du/dr) / r`, with `du/dr = -24 epsilon (2 sr12 - sr6) / r`.
                        let scale = -24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
                        for a in 0..3 {
                            let force = scale * bond[a];
                            forces[3 * i + a] += force;
                            forces[3 * j + a] -= force;
                            for b in 0..3 {
                                stress[3 * a + b] += force * bond[b];
                            }
                        }
                    }
                }
            }
        }
    }
    let volume = det(cell);
    for value in &mut stress {
        *value /= volume;
    }
    CellSample {
        energy,
        forces,
        stress,
    }
}

fn pair_count(positions: &[f64], cell: &[f64; 9]) -> usize {
    let natoms = positions.len() / 3;
    let cutoff2 = CUTOFF * CUTOFF;
    let mut count = 0;
    for i in 0..natoms {
        for j in (i + 1)..natoms {
            for n0 in -1..=1 {
                for n1 in -1..=1 {
                    for n2 in -1..=1 {
                        let offset = image(cell, [f64::from(n0), f64::from(n1), f64::from(n2)]);
                        let mut distance2 = 0.0;
                        for axis in 0..3 {
                            let bond =
                                positions[3 * j + axis] - positions[3 * i + axis] + offset[axis];
                            distance2 += bond * bond;
                        }
                        if distance2 < cutoff2 && distance2 > 1.0e-24 {
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    count
}

fn mean_edge(cell: &[f64; 9]) -> f64 {
    let mut length = 0.0;
    for row in 0..3 {
        let mut norm2 = 0.0;
        for axis in 0..3 {
            norm2 += cell[3 * row + axis] * cell[3 * row + axis];
        }
        length += norm2.sqrt();
    }
    length / 3.0
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, value| acc.max(value.abs()))
}

fn fd_mismatch<F>(filter: &CellFilter<F>, point: &Array1<f64>) -> f64
where
    F: Fn(&[f64], &[f64; 9]) -> CellSample + Send + Sync,
{
    let (_value, gradient) = filter.value_and_gradient(point.view());
    let step = 1.0e-6;
    let mut mismatch = 0.0_f64;
    for index in 0..point.len() {
        let mut plus = point.clone();
        let mut minus = point.clone();
        plus[index] += step;
        minus[index] -= step;
        let forward = filter.eval(plus.view());
        let backward = filter.eval(minus.view());
        let difference = (forward - backward) / (2.0 * step);
        mismatch = mismatch.max((difference - gradient[index]).abs());
    }
    mismatch
}

fn displaced_start() -> (Vec<f64>, [f64; 9]) {
    let edge = lattice_constant() * 1.04;
    let (mut positions, cell) = fcc(edge);
    positions[0] += 0.02;
    (positions, cell)
}

fn wolfe() -> LineSearch {
    LineSearch::wolfe(1.0e-4, 0.9, 20)
}

fn relax(
    kind: CellCoordinates,
    method: Method,
    control: Control,
    pressure: f64,
    mask: [bool; 6],
    hydrostatic: bool,
) -> (rgmin::Report, Vec<f64>, [f64; 9], u64) {
    let (positions, cell) = displaced_start();
    let filter = match kind {
        CellCoordinates::Frechet => CellFilter::frechet(4, cell, lennard_jones),
        CellCoordinates::Deformation => CellFilter::deformation(4, cell, lennard_jones),
        _ => unreachable!("only the two published coordinate maps are exercised"),
    }
    .expect("filter")
    .with_scalar_pressure(pressure)
    .expect("pressure")
    .with_mask(mask)
    .with_hydrostatic_strain(hydrostatic);
    let start = filter.coordinates(&positions, &cell).expect("start");
    filter.reset_evaluations();
    let report = minimize_method(&filter, start, &control, method, wolfe()).expect("minimize");
    let (positions, cell) = filter.geometry(report.coords.view()).expect("geometry");
    (report, positions, cell, filter.evaluations())
}

#[test]
fn the_nearest_neighbour_shell_is_the_only_one_inside_the_cutoff() {
    let edge = lattice_constant();
    let (positions, cell) = fcc(edge);
    assert_eq!(pair_count(&positions, &cell), 24);
    let sample = lennard_jones(&positions, &cell);
    let distance = edge / 2.0_f64.sqrt();
    let sr6 = (SIGMA / distance).powi(6);
    assert!(
        (sr6 - 0.5).abs() < 1.0e-12,
        "nearest neighbour is not at the Lennard-Jones minimum"
    );
    let shift6 = (SIGMA / CUTOFF).powi(6);
    let pair = 4.0 * EPSILON * (sr6 * sr6 - sr6) - 4.0 * EPSILON * (shift6 * shift6 - shift6);
    assert!((sample.energy - 24.0 * pair).abs() < 1.0e-9);
    assert!(max_abs(&sample.forces) < 1.0e-8);
    assert!(max_abs(&sample.stress) < 1.0e-8);

    let filter = CellFilter::frechet(4, cell, lennard_jones).expect("filter");
    let point = filter.coordinates(&positions, &cell).expect("coordinates");
    let (value, gradient) = filter.value_and_gradient(point.view());
    assert!((value - sample.energy).abs() < 1.0e-8);
    assert!(max_abs(gradient.as_slice().expect("gradient")) < 1.0e-6);

    let (expanded, expanded_cell) = fcc(edge * 1.03);
    let filter = CellFilter::deformation(4, cell, lennard_jones).expect("deformation");
    let point = filter
        .coordinates(&expanded, &expanded_cell)
        .expect("expanded");
    let mismatch = fd_mismatch(&filter, &point);
    assert!(
        mismatch < 1.0e-6,
        "expanded shell gradient mismatch {mismatch}"
    );
}

#[test]
fn the_generalized_gradient_matches_the_enthalpy_difference() {
    let (positions, cell) = displaced_start();
    let cases = [
        (CellCoordinates::Frechet, 0.0, [true; 6], false),
        (CellCoordinates::Frechet, 0.05, [true; 6], false),
        (
            CellCoordinates::Frechet,
            0.0,
            [false, true, true, false, false, false],
            false,
        ),
        (CellCoordinates::Frechet, 0.0, [true; 6], true),
        (
            CellCoordinates::Deformation,
            0.02,
            [true, true, false, true, true, true],
            true,
        ),
    ];
    for (kind, pressure, mask, hydrostatic) in cases {
        let filter = match kind {
            CellCoordinates::Frechet => CellFilter::frechet(4, cell, lennard_jones),
            CellCoordinates::Deformation => CellFilter::deformation(4, cell, lennard_jones),
            _ => unreachable!("only the two published coordinate maps are exercised"),
        }
        .expect("filter")
        .with_scalar_pressure(pressure)
        .expect("pressure")
        .with_mask(mask)
        .with_hydrostatic_strain(hydrostatic);
        let point = filter.coordinates(&positions, &cell).expect("coordinates");
        let mismatch = fd_mismatch(&filter, &point);
        assert!(
            mismatch < 1.0e-6,
            "{kind:?} pressure={pressure} hydrostatic={hydrostatic} mismatch={mismatch}"
        );
    }

    let orig = [2.0, 0.35, 0.05, 0.2, 2.4, -0.15, 0.0, 0.25, 2.8];
    let deform = [[1.08, 0.07, 0.02], [-0.03, 0.94, 0.05], [0.01, -0.04, 1.06]];
    let cell = cell_from_deformation(&orig, &deform);
    let positions = vec![0.4, 0.2, 0.1, 1.3, 0.7, 1.6];
    let mask = [false, true, true, true, false, true];
    for kind in [CellCoordinates::Frechet, CellCoordinates::Deformation] {
        let filter = match kind {
            CellCoordinates::Frechet => CellFilter::frechet(2, orig, flat_sample),
            CellCoordinates::Deformation => CellFilter::deformation(2, orig, flat_sample),
            _ => unreachable!("only the two published coordinate maps are exercised"),
        }
        .expect("filter")
        .with_cell_factor(2.5)
        .expect("cell factor")
        .with_exp_cell_factor(1.5)
        .expect("exp factor")
        .with_scalar_pressure(0.03)
        .expect("pressure")
        .with_mask(mask)
        .with_hydrostatic_strain(true);
        let point = filter
            .coordinates(&positions, &cell)
            .expect("general deformation");
        let mismatch = fd_mismatch(&filter, &point);
        assert!(mismatch < 1.0e-6, "{kind:?} general F mismatch {mismatch}");
    }
}

fn assert_lattice(edge: f64, label: &str) {
    let a0 = lattice_constant();
    let relative = (edge - a0).abs() / a0;
    assert!(
        relative < 1.0e-6,
        "{label} edge={edge} a0={a0} relative={relative}"
    );
}

fn assert_cubic(cell: &[f64; 9]) {
    let edges = [0, 4, 8].map(|index| cell[index]);
    let mean = edges.iter().sum::<f64>() / 3.0;
    for edge in edges {
        assert!((edge - mean).abs() / mean < 1.0e-6, "edges {edges:?}");
    }
    for index in [1, 2, 3, 5, 6, 7] {
        assert!(
            cell[index].abs() < 1.0e-6,
            "off diagonal {} = {}",
            index,
            cell[index]
        );
    }
}

#[test]
fn lbfgs_and_fire_recover_the_fcc_lattice() {
    let gtol = 1.0e-9;
    let lbfgs = Control::default().with_gtol(gtol).with_maxiter(200);
    let (report, _positions, cell, evals) = relax(
        CellCoordinates::Frechet,
        Method::lbfgs(),
        lbfgs,
        0.0,
        [true; 6],
        false,
    );
    assert_cubic(&cell);
    assert_lattice(mean_edge(&cell), "lbfgs");
    assert!(report.grad_norm <= gtol, "lbfgs grad {}", report.grad_norm);
    assert_eq!(pair_count(&_positions, &cell), 24);
    println!(
        "cell-filter lbfgs_evals={evals} report_evals={}",
        report.evals
    );

    let fire = Control::default()
        .with_gtol(gtol)
        .with_maxiter(800)
        .with_istep(0.1);
    let (report, positions, cell, evals) = relax(
        CellCoordinates::Frechet,
        Method::fire(FireKind::V1),
        fire,
        0.0,
        [true; 6],
        false,
    );
    assert_cubic(&cell);
    assert_lattice(mean_edge(&cell), "fire");
    assert!(
        report.grad_norm <= gtol,
        "fire grad {} status {:?}",
        report.grad_norm,
        report.status
    );
    assert_eq!(pair_count(&positions, &cell), 24);
    println!(
        "cell-filter fire_evals={evals} report_evals={}",
        report.evals
    );

    let fire2 = Control::default()
        .with_gtol(gtol)
        .with_maxiter(800)
        .with_istep(0.1);
    let (report, positions, cell, evals) = relax(
        CellCoordinates::Frechet,
        Method::fire(FireKind::V2),
        fire2,
        0.0,
        [true; 6],
        false,
    );
    assert_cubic(&cell);
    assert_lattice(mean_edge(&cell), "fire2");
    assert!(
        report.grad_norm <= gtol,
        "fire2 grad {} status {:?}",
        report.grad_norm,
        report.status
    );
    assert_eq!(pair_count(&positions, &cell), 24);
    println!(
        "cell-filter fire2_evals={evals} report_evals={}",
        report.evals
    );
}

#[test]
fn conjugate_gradients_recover_the_fcc_lattice() {
    // Polak-Ribiere levels off near 1.6e-9 on this crystal. The lattice
    // constant is already the analytic value at that point.
    let gtol = 1.0e-8;
    let control = Control::default().with_gtol(gtol).with_maxiter(400);
    let (report, positions, cell, evals) = relax(
        CellCoordinates::Frechet,
        Method::polak_ribiere(),
        control,
        0.0,
        [true; 6],
        false,
    );
    assert_cubic(&cell);
    assert_lattice(mean_edge(&cell), "cg");
    assert!(
        report.grad_norm <= gtol,
        "cg grad {} status {:?}",
        report.grad_norm,
        report.status
    );
    assert_eq!(pair_count(&positions, &cell), 24);
    println!("cell-filter cg_evals={evals} report_evals={}", report.evals);
}

#[test]
fn deformation_coordinates_recover_the_fcc_lattice() {
    let gtol = 1.0e-9;
    let control = Control::default().with_gtol(gtol).with_maxiter(200);
    let (report, positions, cell, evals) = relax(
        CellCoordinates::Deformation,
        Method::lbfgs(),
        control,
        0.0,
        [true; 6],
        false,
    );
    assert_cubic(&cell);
    assert_lattice(mean_edge(&cell), "deformation");
    assert!(
        report.grad_norm <= gtol,
        "deformation grad {}",
        report.grad_norm
    );
    assert_eq!(pair_count(&positions, &cell), 24);
    println!("cell-filter deformation_lbfgs_evals={evals}");
}

#[test]
fn external_pressure_sets_the_stress_to_minus_pressure() {
    let pressure = 0.04;
    let gtol = 1.0e-8;
    let control = Control::default().with_gtol(gtol).with_maxiter(250);
    let (report, positions, cell, _evals) = relax(
        CellCoordinates::Frechet,
        Method::lbfgs(),
        control,
        pressure,
        [true; 6],
        false,
    );
    assert!(
        report.grad_norm <= gtol,
        "grad {} status {:?}",
        report.grad_norm,
        report.status
    );
    let sample = lennard_jones(&positions, &cell);
    let mut mismatch = 0.0_f64;
    for row in 0..3 {
        for col in 0..3 {
            let target = if row == col { -pressure } else { 0.0 };
            mismatch = mismatch.max((sample.stress[3 * row + col] - target).abs());
        }
    }
    assert!(
        mismatch <= gtol,
        "stress mismatch {mismatch} stress {:?} grad {}",
        sample.stress,
        report.grad_norm
    );
    assert_eq!(pair_count(&positions, &cell), 24);
}

#[test]
fn mask_and_hydrostatic_strain_hold_the_cell() {
    let (positions, cell) = displaced_start();
    let start_edge = cell[0];
    let gtol = 1.0e-8;
    let control = Control::default().with_gtol(gtol).with_maxiter(300);
    let filter = CellFilter::frechet(4, cell, lennard_jones)
        .expect("filter")
        .with_mask([false, true, true, false, false, false]);
    let mut start = filter.coordinates(&positions, &cell).expect("start");
    // Off the reference deformation. The masked gradient is zero, so the
    // solver never walks this coordinate back. Only the projection does.
    start[3 * 4] += 0.4;
    let (_positions, snapped) = filter.geometry(start.view()).expect("project");
    assert!(
        (snapped[0] - start_edge).abs() < 1.0e-10,
        "mask did not hold xx at the start, cell_xx {}",
        snapped[0]
    );
    let report = minimize_method(&filter, start, &control, Method::lbfgs(), wolfe()).expect("mask");
    let (_positions, relaxed) = filter.geometry(report.coords.view()).expect("geometry");
    assert!(
        (relaxed[0] - start_edge).abs() < 1.0e-10,
        "xx moved to {}",
        relaxed[0]
    );
    assert!(relaxed[1].abs() < 1.0e-8 && relaxed[2].abs() < 1.0e-8);
    assert!(relaxed[3].abs() < 1.0e-8 && relaxed[6].abs() < 1.0e-8 && relaxed[7].abs() < 1.0e-8);
    let free_edge = 0.5 * (relaxed[4] + relaxed[8]);
    assert!(
        (free_edge - start_edge).abs() / start_edge > 1.0e-3,
        "yy/zz did not move: {free_edge} start {start_edge}"
    );

    let filter = CellFilter::frechet(4, cell, lennard_jones)
        .expect("filter")
        .with_hydrostatic_strain(true);
    let mut sheared = filter.coordinates(&positions, &cell).expect("start");
    let natoms = 4;
    sheared[3 * natoms + 1] += 0.25;
    let (_positions, projected) = filter.geometry(sheared.view()).expect("project");
    for index in [1, 2, 3, 5, 6, 7] {
        assert!(projected[index].abs() < 1.0e-8, "shear survived at {index}");
    }
    let edges = [projected[0], projected[4], projected[8]];
    assert!((edges[0] - edges[1]).abs() < 1.0e-8 && (edges[1] - edges[2]).abs() < 1.0e-8);

    let start = filter.coordinates(&positions, &cell).expect("start");
    let report = minimize_method(
        &filter,
        start,
        &control,
        Method::fire(FireKind::V2),
        wolfe(),
    )
    .expect("hydrostatic");
    let (_positions, relaxed) = filter.geometry(report.coords.view()).expect("geometry");
    assert_cubic(&relaxed);
    let edge = mean_edge(&relaxed);
    assert!(
        (edge - start_edge).abs() / start_edge > 1.0e-3,
        "volume did not move"
    );
}

fn eye3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn expm3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut scaled = matrix;
    let mut norm = 0.0_f64;
    for row in scaled {
        let mut sum = 0.0;
        for value in row {
            sum += value.abs();
        }
        norm = norm.max(sum);
    }
    let mut squares = 0u32;
    while norm > 0.25 && squares < 64 {
        for row in &mut scaled {
            for value in row.iter_mut() {
                *value *= 0.5;
            }
        }
        norm *= 0.5;
        squares += 1;
    }
    let mut term = eye3();
    let mut sum = eye3();
    for k in 1..=40 {
        term = mul3(term, scaled);
        let inv = 1.0 / f64::from(k);
        for i in 0..3 {
            for j in 0..3 {
                term[i][j] *= inv;
                sum[i][j] += term[i][j];
            }
        }
    }
    for _ in 0..squares {
        sum = mul3(sum, sum);
    }
    sum
}

fn frechet_direction(log_f: [[f64; 3]; 3], row: usize, col: usize) -> [[f64; 3]; 3] {
    let step = 1.0e-7;
    let mut plus = log_f;
    let mut minus = log_f;
    plus[row][col] += step;
    minus[row][col] -= step;
    let forward = expm3(plus);
    let backward = expm3(minus);
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (forward[i][j] - backward[i][j]) / (2.0 * step);
        }
    }
    out
}

/// ASE generalized force, the negative of this filter's gradient.
///
/// Unmasked and non-hydrostatic. The Frechet cell block follows
/// `get_forces_frechet`: the derivative of `exp` at `log(F)`, divided by
/// `exp_cell_factor`, with no extra `cell_factor`.
fn ase_generalized_force(
    kind: CellCoordinates,
    positions: &[f64],
    cell: &[f64; 9],
    orig: &[f64; 9],
    log_f: [[f64; 3]; 3],
    pressure: f64,
) -> Vec<f64> {
    let cell_factor = (positions.len() / 3) as f64;
    let exp_cell_factor = cell_factor;
    let sample = lennard_jones(positions, cell);
    let deform = transpose3(mul3(inverse3(rows(orig)), rows(cell)));
    let volume = det(cell);
    let mut cauchy = rows(&sample.stress);
    for diag in 0..3 {
        cauchy[diag][diag] += pressure;
    }
    let mut virial = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            virial[i][j] = -volume * cauchy[i][j];
        }
    }
    let pulled = mul3(virial, inverse3(transpose3(deform)));
    let natoms = positions.len() / 3;
    let mut force = vec![0.0; 3 * (natoms + 3)];
    for atom in 0..natoms {
        for axis in 0..3 {
            let mut acc = 0.0;
            for source in 0..3 {
                acc += sample.forces[3 * atom + source] * deform[source][axis];
            }
            force[3 * atom + axis] = acc;
        }
    }
    let base = 3 * natoms;
    match kind {
        CellCoordinates::Deformation => {
            for row in 0..3 {
                for col in 0..3 {
                    force[base + 3 * row + col] = pulled[row][col] / cell_factor;
                }
            }
        }
        CellCoordinates::Frechet => {
            for row in 0..3 {
                for col in 0..3 {
                    let derivative = frechet_direction(log_f, row, col);
                    let mut paired = 0.0;
                    for i in 0..3 {
                        for j in 0..3 {
                            paired += derivative[i][j] * pulled[i][j];
                        }
                    }
                    force[base + 3 * row + col] = paired / exp_cell_factor;
                }
            }
        }
        _ => unreachable!("only the two published coordinate maps are exercised"),
    }
    force
}

#[test]
fn unmasked_force_matches_ase_and_masked_force_matches_the_enthalpy() {
    let (reference, orig) = fcc(lattice_constant() * 1.02);
    let log_f = [[0.04, 0.03, 0.0], [0.01, -0.02, 0.02], [0.0, -0.01, 0.03]];
    let deform = expm3(log_f);
    let positions = apply_deformation(&reference, &deform);
    let cell = cell_from_deformation(&orig, &deform);
    let pressure = 0.01;
    let natoms = 4;
    let mask = [false, true, true, true, true, false];
    for kind in [CellCoordinates::Deformation, CellCoordinates::Frechet] {
        let filter = match kind {
            CellCoordinates::Frechet => CellFilter::frechet(natoms, orig, lennard_jones),
            CellCoordinates::Deformation => CellFilter::deformation(natoms, orig, lennard_jones),
            _ => unreachable!("only the two published coordinate maps are exercised"),
        }
        .expect("filter")
        .with_scalar_pressure(pressure)
        .expect("pressure");
        let point = filter.coordinates(&positions, &cell).expect("coordinates");
        let (_value, gradient) = filter.value_and_gradient(point.view());
        let ase = ase_generalized_force(kind, &positions, &cell, &orig, log_f, pressure);
        let mut mismatch = 0.0_f64;
        for (got, want) in gradient.iter().zip(&ase) {
            mismatch = mismatch.max((got + want).abs());
        }
        println!("cell-filter ase_unmasked {kind:?} mismatch={mismatch}");
        assert!(mismatch < 1.0e-6, "{kind:?} ASE mismatch {mismatch}");

        let masked = filter.with_mask(mask);
        let masked_point = masked.coordinates(&positions, &cell).expect("masked");
        let fd = fd_mismatch(&masked, &masked_point);
        println!("cell-filter ase_masked_fd {kind:?} mismatch={fd}");
        assert!(fd < 1.0e-6, "{kind:?} masked enthalpy mismatch {fd}");
    }
}
