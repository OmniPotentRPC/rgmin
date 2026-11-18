//! The extended cell layout: round trips, the gradient against central
//! differences of the enthalpy, and the filter options.

mod cell_support;

use cell_support::{evaluate, fcc, rattle, strain_positions, strained};
use ndarray::Array1;
use rgmin::{Cell, CellFilter, CellFilterConfig, Error};

const STRAIN: [[f64; 3]; 3] = [
    [0.012, 0.004, -0.003],
    [0.004, -0.02, 0.006],
    [-0.003, 0.006, 0.015],
];

fn system() -> (Array1<f64>, Cell, Cell) {
    let (mut positions, orig) = fcc(1, 3.61);
    rattle(&mut positions, 0.04, 7);
    let cell = strained(&orig, STRAIN);
    let positions = strain_positions(&positions, STRAIN);
    (positions, orig, cell)
}

fn enthalpy(filter: &CellFilter, x: &Array1<f64>) -> f64 {
    let (positions, cell) = filter.unpack(x.view()).unwrap();
    filter.enthalpy(evaluate(positions.view(), &cell).energy, &cell)
}

fn analytic(filter: &CellFilter, x: &Array1<f64>) -> Array1<f64> {
    let (positions, cell) = filter.unpack(x.view()).unwrap();
    let ev = evaluate(positions.view(), &cell);
    filter
        .gradient(ev.forces.view(), &ev.stress, &cell)
        .unwrap()
}

fn central_difference(filter: &CellFilter, x: &Array1<f64>, k: usize) -> f64 {
    let h = 1.0e-5;
    let mut plus = x.clone();
    let mut minus = x.clone();
    plus[k] += h;
    minus[k] -= h;
    (enthalpy(filter, &plus) - enthalpy(filter, &minus)) / (2.0 * h)
}

fn max_abs(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, a| m.max(a.abs()))
}

#[test]
fn pack_and_unpack_are_inverse() {
    let (positions, orig, cell) = system();
    let n = positions.len() / 3;
    let filter = CellFilter::new(CellFilterConfig::default(), orig, n).unwrap();
    let x = filter.pack(positions.view(), &cell).unwrap();
    assert_eq!(x.len(), 3 * n + 9);
    let (back, back_cell) = filter.unpack(x.view()).unwrap();
    let scale = max_abs(&positions).max(1.0);
    for k in 0..positions.len() {
        assert!((back[k] - positions[k]).abs() <= 64.0 * f64::EPSILON * scale);
    }
    for i in 0..3 {
        for j in 0..3 {
            assert!((back_cell[i][j] - cell[i][j]).abs() <= 64.0 * f64::EPSILON * scale);
        }
    }
}

#[test]
fn the_reference_cell_packs_to_a_zero_strain_block() {
    let (positions, orig) = fcc(1, 3.61);
    let filter = CellFilter::new(CellFilterConfig::default(), orig, 4).unwrap();
    let x = filter.pack(positions.view(), &orig).unwrap();
    for k in 12..21 {
        assert_eq!(x[k], 0.0);
    }
    for k in 0..12 {
        assert_eq!(x[k], positions[k]);
    }
}

#[test]
fn the_cell_block_is_the_factor_times_the_log_of_the_deformation() {
    let (positions, orig, cell) = system();
    let small = CellFilter::new(
        CellFilterConfig::default().with_exp_cell_factor(1.0),
        orig,
        4,
    )
    .unwrap();
    let big = CellFilter::new(
        CellFilterConfig::default().with_exp_cell_factor(8.0),
        orig,
        4,
    )
    .unwrap();
    let a = small.pack(positions.view(), &cell).unwrap();
    let b = big.pack(positions.view(), &cell).unwrap();
    for k in 12..21 {
        assert!((b[k] - 8.0 * a[k]).abs() <= 64.0 * f64::EPSILON * 8.0 * a[k].abs().max(1.0));
    }
    let default = CellFilter::new(CellFilterConfig::default(), orig, 4).unwrap();
    assert_eq!(default.exp_cell_factor(), 4.0);
}

#[test]
fn the_gradient_is_the_derivative_of_the_enthalpy() {
    let (positions, orig, cell) = system();
    for pressure in [0.0, 0.004] {
        let config = CellFilterConfig::default().with_pressure(pressure);
        let filter = CellFilter::new(config, orig, 4).unwrap();
        let x = filter.pack(positions.view(), &cell).unwrap();
        let g = analytic(&filter, &x);
        let scale = max_abs(&g);
        assert!(scale > 0.0);
        for k in 0..filter.dim() {
            let fd = central_difference(&filter, &x, k);
            assert!(
                (g[k] - fd).abs() <= 1.0e-6 * scale,
                "p {pressure} component {k}: {} vs {fd}",
                g[k]
            );
        }
    }
}

#[test]
fn a_larger_factor_scales_the_cell_gradient_down_and_leaves_the_atoms() {
    let (positions, orig, cell) = system();
    let one = CellFilter::new(
        CellFilterConfig::default().with_exp_cell_factor(1.0),
        orig,
        4,
    )
    .unwrap();
    let four = CellFilter::new(CellFilterConfig::default(), orig, 4).unwrap();
    let ev = evaluate(positions.view(), &cell);
    let g1 = one.gradient(ev.forces.view(), &ev.stress, &cell).unwrap();
    let g4 = four.gradient(ev.forces.view(), &ev.stress, &cell).unwrap();
    for k in 0..12 {
        assert_eq!(g1[k], g4[k]);
    }
    for k in 12..21 {
        assert!((g4[k] * 4.0 - g1[k]).abs() <= 8.0 * f64::EPSILON * g1[k].abs().max(1.0e-3));
    }
}

#[test]
fn hydrostatic_strain_keeps_only_the_isotropic_cell_gradient() {
    let (positions, orig) = fcc(1, 3.61);
    let mut rattled = positions.clone();
    rattle(&mut rattled, 0.04, 3);
    let config = CellFilterConfig::default().with_hydrostatic(true);
    let filter = CellFilter::new(config, orig, 4).unwrap();
    let x = filter.pack(rattled.view(), &orig).unwrap();
    let g = analytic(&filter, &x);
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                assert_eq!(g[12 + 3 * i + j], 0.0);
            }
        }
    }
    assert!((g[12] - g[16]).abs() <= 8.0 * f64::EPSILON * g[12].abs());
    assert!((g[16] - g[20]).abs() <= 8.0 * f64::EPSILON * g[12].abs());
    let along: f64 = [12, 16, 20]
        .iter()
        .map(|&k| central_difference(&filter, &x, k))
        .sum();
    let predicted = g[12] + g[16] + g[20];
    assert!((predicted - along).abs() <= 1.0e-6 * predicted.abs().max(1.0e-3));
}

#[test]
fn constant_volume_removes_the_trace_of_the_cell_gradient() {
    let (positions, orig, cell) = system();
    let plain = CellFilter::new(CellFilterConfig::default(), orig, 4).unwrap();
    let held = CellFilter::new(
        CellFilterConfig::default().with_constant_volume(true),
        orig,
        4,
    )
    .unwrap();
    let ev = evaluate(positions.view(), &cell);
    let gp = plain.gradient(ev.forces.view(), &ev.stress, &cell).unwrap();
    let gh = held.gradient(ev.forces.view(), &ev.stress, &cell).unwrap();
    let trace = gh[12] + gh[16] + gh[20];
    assert!(trace.abs() <= 16.0 * f64::EPSILON * gp.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
    let third = (gp[12] + gp[16] + gp[20]) / 3.0;
    for k in 0..12 {
        assert_eq!(gp[k], gh[k]);
    }
    for k in [12, 16, 20] {
        assert!((gh[k] - (gp[k] - third)).abs() <= 16.0 * f64::EPSILON * third.abs().max(1.0e-3));
    }
}

#[test]
fn a_masked_component_gets_no_stress_force_and_no_strain() {
    let (positions, orig) = fcc(1, 3.61);
    let mut rattled = positions.clone();
    rattle(&mut rattled, 0.04, 5);
    let start = strained(&orig, [[0.0; 3]; 3]);
    let mask = [true, false, false, false, false, false];
    let config = CellFilterConfig::default().with_voigt_mask(mask);
    let filter = CellFilter::new(config, orig, 4).unwrap();
    let x = filter.pack(rattled.view(), &start).unwrap();
    let g = analytic(&filter, &x);
    for k in 1..9 {
        assert_eq!(g[12 + k], 0.0, "component {k}");
    }
    assert_ne!(g[12], 0.0);
    let fd = central_difference(&filter, &x, 12);
    assert!((g[12] - fd).abs() <= 1.0e-6 * g[12].abs());

    let mut moved = x.clone();
    for k in 1..9 {
        moved[12 + k] = 0.3;
    }
    let (_, cell) = filter.unpack(moved.view()).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            if (i, j) != (0, 0) {
                assert_eq!(cell[i][j], orig[i][j], "cell {i}{j}");
            }
        }
    }
}

#[test]
fn the_voigt_mask_fills_the_symmetric_matrix() {
    let config =
        CellFilterConfig::default().with_voigt_mask([true, false, true, true, false, true]);
    assert_eq!(
        config.mask,
        [
            [true, true, false],
            [true, false, true],
            [false, true, true]
        ]
    );
}

#[test]
fn bad_inputs_are_refused() {
    let (positions, orig) = fcc(1, 3.61);
    let singular = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(matches!(
        CellFilter::new(CellFilterConfig::default(), singular, 4),
        Err(Error::Cell { .. })
    ));
    assert!(matches!(
        CellFilter::new(CellFilterConfig::default(), orig, 0),
        Err(Error::Cell { .. })
    ));
    for factor in [0.0, -1.0, f64::NAN] {
        let config = CellFilterConfig::default().with_exp_cell_factor(factor);
        assert!(CellFilter::new(config, orig, 4).is_err());
    }
    let config = CellFilterConfig::default().with_pressure(f64::INFINITY);
    assert!(CellFilter::new(config, orig, 4).is_err());
    let filter = CellFilter::new(CellFilterConfig::default(), orig, 4).unwrap();
    assert!(matches!(
        filter.pack(positions.slice(ndarray::s![..9]), &orig),
        Err(Error::Dim { .. })
    ));
    assert!(matches!(
        filter.unpack(positions.view()),
        Err(Error::Dim { .. })
    ));
    let inverted = [[-3.61, 0.0, 0.0], [0.0, 3.61, 0.0], [0.0, 0.0, 3.61]];
    assert!(matches!(
        filter.pack(positions.view(), &inverted),
        Err(Error::Cell { .. })
    ));
}
