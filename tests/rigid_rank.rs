//! Rigid translation and rotation: rank after frozen atoms are zeroed.
//!
//! The projected gradient of one steepest step is orthogonal to the true
//! rigid motions, and a frozen atom does not move.

use ndarray::{Array1, ArrayView1};
use rgmin::{Accept, Control, Method, Oracle, Solver};

fn session(dim: usize, gtol: f64, istep: f64) -> Solver {
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            gtol,
            istep,
            ..Control::default()
        },
        dim,
    );
    solver.set_project_rigid(true);
    solver.set_accept(Accept::None);
    solver
}

/// One steepest step. `gtol` above every gradient returns the projected
/// gradient at `x` and leaves `x` where it was.
fn step_with(
    x: &Array1<f64>,
    grad: Array1<f64>,
    frozen: Option<&[bool]>,
    gtol: f64,
    istep: f64,
) -> (Array1<f64>, Array1<f64>) {
    let dim = x.len();
    let mut solver = session(dim, gtol, istep);
    if let Some(mask) = frozen {
        solver.set_frozen_atoms(mask).unwrap();
    }
    let obj = Oracle::unbounded(dim, move |_x: ArrayView1<f64>| (0.0, grad.clone()));
    let mut y = x.clone();
    let report = solver.step(&obj, &mut y).unwrap();
    (y, report.grad.expect("session gradient"))
}

fn projected(x: &Array1<f64>, grad: Array1<f64>, frozen: Option<&[bool]>) -> Array1<f64> {
    step_with(x, grad, frozen, 1.0e9, 1.0).1
}

fn norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

fn max_abs(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0, |m, c| m.max(c.abs()))
}

fn translation(nat: usize, axis: usize) -> Array1<f64> {
    let mut v = Array1::zeros(3 * nat);
    for atom in 0..nat {
        v[3 * atom + axis] = 1.0;
    }
    v
}

fn cross(omega: [f64; 3], r: [f64; 3]) -> [f64; 3] {
    [
        omega[1] * r[2] - omega[2] * r[1],
        omega[2] * r[0] - omega[0] * r[2],
        omega[0] * r[1] - omega[1] * r[0],
    ]
}

/// Infinitesimal rotation about `origin`. Zero on an atom that sits there.
fn rotation(x: &[f64], omega: [f64; 3], origin: [f64; 3]) -> Array1<f64> {
    let nat = x.len() / 3;
    let mut v = Array1::zeros(x.len());
    for atom in 0..nat {
        let r = [
            x[3 * atom] - origin[0],
            x[3 * atom + 1] - origin[1],
            x[3 * atom + 2] - origin[2],
        ];
        let w = cross(omega, r);
        v[3 * atom] = w[0];
        v[3 * atom + 1] = w[1];
        v[3 * atom + 2] = w[2];
    }
    v
}

fn centroid(x: &[f64]) -> [f64; 3] {
    let nat = x.len() / 3;
    let mut c = [0.0; 3];
    for atom in 0..nat {
        c[0] += x[3 * atom];
        c[1] += x[3 * atom + 1];
        c[2] += x[3 * atom + 2];
    }
    let s = nat as f64;
    [c[0] / s, c[1] / s, c[2] / s]
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

fn perpendicular(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let trial = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let e1 = unit(cross(axis, trial));
    let e2 = unit(cross(axis, e1));
    (e1, e2)
}

/// Component of `v` orthogonal to `modes`.
fn orthogonal_to(mut v: Array1<f64>, modes: &[Array1<f64>]) -> Array1<f64> {
    for mode in modes {
        let n2 = mode.dot(mode);
        if n2 > 0.0 {
            let coef = v.dot(mode) / n2;
            v = &v - coef * mode;
        }
    }
    v
}

fn assert_orthogonal(g: &Array1<f64>, mode: &Array1<f64>, tol: f64) {
    let scale = norm(mode) * norm(g).max(1.0);
    let dot = g.dot(mode).abs();
    assert!(
        dot <= tol * scale,
        "dot {dot:.3e} against scale {scale:.3e} (tol {tol:.1e})"
    );
}

fn assert_frozen_still(before: &Array1<f64>, after: &Array1<f64>, mask: &[bool]) {
    for (atom, &frozen) in mask.iter().enumerate() {
        if !frozen {
            continue;
        }
        for c in 0..3 {
            let i = 3 * atom + c;
            assert_eq!(
                before[i].to_bits(),
                after[i].to_bits(),
                "frozen atom {atom} coordinate {c} moved"
            );
        }
    }
}

fn assert_frozen_gradient(g: &Array1<f64>, mask: &[bool]) {
    for (atom, &frozen) in mask.iter().enumerate() {
        if !frozen {
            continue;
        }
        for c in 0..3 {
            let force = g[3 * atom + c];
            assert!(
                force.abs() < 1e-12,
                "frozen atom {atom} coordinate {c} still has a force {force}"
            );
        }
    }
}

/// CO2 on a line that is not a coordinate axis, far from the origin.
/// The axial rotation is a null mode; roundoff makes a spurious column
/// that an absolute cutoff keeps.
fn co2_far() -> (Array1<f64>, [f64; 3]) {
    let axis = unit([1.0 / 3.0, 2.0_f64.sqrt(), std::f64::consts::PI / 7.0]);
    let origin = [1.0e8, -1.0e8 / 3.0, 1.0e8 * std::f64::consts::PI];
    let bond = 1.16;
    let mut x = Array1::zeros(9);
    for (atom, t) in [0.0, bond, -bond].into_iter().enumerate() {
        for c in 0..3 {
            x[3 * atom + c] = origin[c] + t * axis[c];
        }
    }
    (x, axis)
}

#[test]
fn linear_co2_keeps_two_rotations() {
    let (x, axis) = co2_far();
    let origin = centroid(x.as_slice().unwrap());
    let (e1, e2) = perpendicular(axis);
    let rot = [
        rotation(x.as_slice().unwrap(), e1, origin),
        rotation(x.as_slice().unwrap(), e2, origin),
    ];
    let mut true_modes = vec![rot[0].clone(), rot[1].clone()];
    for axis_i in 0..3 {
        true_modes.push(translation(3, axis_i));
    }
    // Far from the origin the axial direction is a null rotation. Its
    // roundoff lies in the five real modes; the stretch must survive.
    let mut grad = orthogonal_to(rotation(x.as_slice().unwrap(), axis, origin), &true_modes);
    for mode in &true_modes {
        grad += &(mode / norm(mode));
    }
    // Symmetric stretch along the bond. It is not a rigid motion.
    let mut stretch = Array1::zeros(9);
    for c in 0..3 {
        stretch[3 + c] = axis[c];
        stretch[6 + c] = -axis[c];
    }
    grad += &stretch;

    let g = projected(&x, grad, None);
    for mode in &true_modes {
        assert_orthogonal(&g, mode, 1e-8);
    }
    let along = g.dot(&stretch) / stretch.dot(&stretch);
    assert!(
        (along - 1.0).abs() < 1e-6,
        "the CO stretch was removed, coefficient {along}"
    );
}

#[test]
fn near_linear_rotation_is_removed() {
    let delta = 1.0e-10;
    let x = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, delta, 0.0]);
    let origin = centroid(x.as_slice().unwrap());
    let mode = rotation(x.as_slice().unwrap(), [1.0, 0.0, 0.0], origin);
    let before = norm(&mode);
    assert!(before > 1e-12 && before < 1e-8, "mode norm {before}");
    let g = projected(&x, mode, None);
    assert!(
        norm(&g) < 1e-8 * before,
        "near-linear rotation survived {} -> {}",
        before,
        norm(&g)
    );
}

#[test]
fn a_single_free_atom_has_no_internal_force() {
    let x = Array1::from_vec(vec![1.5, -0.25, 4.0]);
    let grad = Array1::from_vec(vec![0.4, -0.2, 0.7]);
    let g = projected(&x, grad.clone(), None);
    assert!(norm(&g) < 1e-12, "a lone atom kept a force {g:?}");
    for axis in 0..3 {
        assert_orthogonal(&g, &translation(1, axis), 1e-12);
    }
    let (y, _) = step_with(&x, grad, None, 0.0, 1.0e-3);
    // The projected force is zero, so the step does not translate the atom.
    assert!((y[0] - x[0]).abs() < 1e-15);
    assert!((y[1] - x[1]).abs() < 1e-15);
    assert!((y[2] - x[2]).abs() < 1e-15);
}

#[test]
fn frozen_slab_leaves_no_rigid_mode() {
    // Three non-collinear slab atoms and one adsorbate.
    let x = Array1::from_vec(vec![
        0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.4, 0.6, 1.8,
    ]);
    let mask = [true, true, true, false];
    let mut grad = Array1::zeros(12);
    grad[9] = 0.2;
    grad[10] = -0.15;
    grad[11] = 0.35;
    let g = projected(&x, grad.clone(), Some(&mask));
    assert_frozen_gradient(&g, &mask);
    assert!(
        max_abs(&(&g - &grad)) < 1e-10,
        "adsorbate force changed {g:?}"
    );
    let (y, g_step) = step_with(&x, grad, Some(&mask), 0.0, 1.0e-3);
    assert_frozen_still(&x, &y, &mask);
    assert_frozen_gradient(&g_step, &mask);
    // The adsorbate did take the step: the force is real.
    assert!((y[9] - x[9]).abs() > 1e-6, "adsorbate did not move");
}

#[test]
fn one_frozen_atom_keeps_rotations_about_it() {
    let x = Array1::from_vec(vec![
        0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.1, 1.1, 0.0, 0.2, -0.3, 0.9,
    ]);
    let mask = [true, false, false, false];
    let origin = [x[0], x[1], x[2]];
    let rotations = [
        rotation(x.as_slice().unwrap(), [1.0, 0.0, 0.0], origin),
        rotation(x.as_slice().unwrap(), [0.0, 1.0, 0.0], origin),
        rotation(x.as_slice().unwrap(), [0.0, 0.0, 1.0], origin),
    ];
    // A radial push on the second atom is not a rotation about the first.
    let mut grad = rotations[2].clone();
    grad[3] += 1.0;
    let g = projected(&x, grad.clone(), Some(&mask));
    assert_frozen_gradient(&g, &mask);
    for mode in &rotations {
        assert_orthogonal(&g, mode, 1e-8);
    }
    assert!(
        (g[3] - 1.0).abs() < 1e-8,
        "radial push was removed, g[3] = {}",
        g[3]
    );
    let (y, _) = step_with(&x, grad, Some(&mask), 0.0, 1.0e-3);
    assert_frozen_still(&x, &y, &mask);
}

#[test]
fn two_frozen_atoms_keep_the_axis_rotation() {
    let x = Array1::from_vec(vec![
        0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.4, 1.2, 0.3, 1.1, -0.4, 0.8,
    ]);
    let mask = [true, true, false, false];
    let axis = rotation(x.as_slice().unwrap(), [1.0, 0.0, 0.0], [x[0], x[1], x[2]]);
    assert!(norm(&axis) > 0.5, "axis rotation vanished {}", norm(&axis));
    assert!(axis.iter().take(6).all(|c| c.abs() < 1e-12));
    let mut grad = axis.clone();
    grad[6] += 0.8; // along the axis, not the rotation
    let g = projected(&x, grad.clone(), Some(&mask));
    assert_frozen_gradient(&g, &mask);
    assert_orthogonal(&g, &axis, 1e-8);
    assert!(
        (g[6] - 0.8).abs() < 1e-8,
        "axial push was removed, g[6] = {}",
        g[6]
    );
    let (y, _) = step_with(&x, grad, Some(&mask), 0.0, 1.0e-3);
    assert_frozen_still(&x, &y, &mask);
}

#[test]
fn a_far_chain_does_not_promote_roundoff_to_a_rotation() {
    let axis = unit([1.0 / 3.0, 2.0_f64.sqrt(), std::f64::consts::PI / 5.0]);
    let origin = [8.0e6, -2.0e6, 4.0e6 * std::f64::consts::PI];
    let nat = 24;
    let mut coords = Vec::with_capacity(3 * nat);
    for atom in 0..nat {
        let t = -3.0 + 6.0 * atom as f64 / (nat - 1) as f64;
        for c in 0..3 {
            coords.push(origin[c] + t * axis[c]);
        }
    }
    let x = Array1::from(coords);
    let com = centroid(x.as_slice().unwrap());
    let (e1, e2) = perpendicular(axis);
    let real = [
        rotation(x.as_slice().unwrap(), e1, com),
        rotation(x.as_slice().unwrap(), e2, com),
    ];
    let mut true_modes = vec![real[0].clone(), real[1].clone()];
    for axis_i in 0..3 {
        true_modes.push(translation(nat, axis_i));
    }
    let axial = orthogonal_to(rotation(x.as_slice().unwrap(), axis, com), &true_modes);
    assert!(
        norm(&axial) > 1e-12,
        "expected a roundoff column, got {}",
        norm(&axial)
    );
    let mut grad = axial.clone();
    for mode in &true_modes {
        grad += &(mode / norm(mode));
    }
    let g = projected(&x, grad, None);
    for mode in &true_modes {
        assert_orthogonal(&g, mode, 1e-6);
    }
    let kept = g.dot(&axial) / axial.dot(&axial);
    assert!(
        kept > 0.5,
        "relative tolerance dropped or kept the wrong column, coefficient {kept}"
    );
}
