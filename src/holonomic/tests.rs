use ndarray::{Array1, array};

use super::lincs::derived_order;
use super::*;

const O: f64 = 15.999;
const H: f64 = 1.008;

fn water() -> Array1<f64> {
    let (d, half) = (0.9572, 104.52_f64.to_radians() / 2.0);
    array![
        0.0,
        0.0,
        0.0,
        d * half.sin(),
        d * half.cos(),
        0.0,
        -d * half.sin(),
        d * half.cos(),
        0.0
    ]
}

fn water_set() -> DistanceConstraints {
    DistanceConstraints::at(vec![(0, 1), (0, 2), (1, 2)], water().view())
        .unwrap()
        .with_masses(vec![O, H, H])
        .unwrap()
}

/// Deterministic displacement in `[-scale, scale]`.
fn kick(n: usize, scale: f64, seed: u64) -> Array1<f64> {
    let mut s = seed;
    Array1::from_shape_fn(n, |_| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        scale * (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0)
    })
}

fn centre_of_mass(x: &Array1<f64>, m: &[f64]) -> [f64; 3] {
    let total: f64 = m.iter().sum();
    let mut c = [0.0; 3];
    for (a, m) in m.iter().enumerate() {
        (0..3).for_each(|k| c[k] += m * x[3 * a + k] / total);
    }
    c
}

fn err(r: Result<impl std::fmt::Debug>) -> ConstraintError {
    match r {
        Err(Error::Constraint(e)) => e,
        other => panic!("expected a constraint error, got {other:?}"),
    }
}

#[test]
fn lone_bond_moves_by_the_mass_ratio() {
    let anchor = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    for solver in [ConstraintSolver::Shake, ConstraintSolver::Lincs] {
        let set = DistanceConstraints::new(vec![(0, 1)], vec![1.0])
            .unwrap()
            .with_masses(vec![1.0, 3.0])
            .unwrap()
            .with_solver(solver);
        let mut x = array![0.0, 0.0, 0.0, 1.2, 0.0, 0.0];
        set.constrain(anchor.view(), &mut x).unwrap();
        // Centre of mass 0.9 stays; the light atom takes 3/4 of the move.
        let want = [0.15, 0.0, 0.0, 1.15, 0.0, 0.0];
        for (got, want) in x.iter().zip(want) {
            assert!((got - want).abs() <= 8.0 * U, "{solver:?}: {x}");
        }
        assert!(set.excess(x.view()) <= 1.0);
    }
}

#[test]
fn every_solver_meets_the_water_and_keeps_its_centre_of_mass() {
    let anchor = water();
    let trial = &anchor + &kick(9, 0.05, 7);
    let mut out = Vec::new();
    for (solver, settle) in [
        (ConstraintSolver::Shake, false),
        (ConstraintSolver::Lincs, false),
        (ConstraintSolver::Auto, true),
    ] {
        let set = water_set().with_solver(solver).with_settle(settle);
        let mut x = trial.clone();
        let report = set.constrain(anchor.view(), &mut x).unwrap();
        assert_eq!(report.settle, usize::from(settle));
        assert!(set.excess(x.view()) <= 1.0, "{solver:?} {settle}");
        let (c0, c1) = (
            centre_of_mass(&trial, &[O, H, H]),
            centre_of_mass(&x, &[O, H, H]),
        );
        for k in 0..3 {
            assert!((c0[k] - c1[k]).abs() <= 64.0 * U, "{solver:?} {settle}");
        }
        out.push(x);
    }
    // One retraction: SHAKE, LINCS and SETTLE solve the same equations.
    for x in &out[1..] {
        let gap = (x - &out[0]).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(gap <= 1e3 * U, "{gap:e}");
    }
}

#[test]
fn settle_matches_converged_shake_on_a_scalene_triangle_with_three_masses() {
    let anchor = array![0.0, 0.0, 0.0, 1.3, 0.1, 0.0, 0.4, 0.9, 0.2];
    let pairs = vec![(0, 1), (2, 0), (1, 2)];
    let base = DistanceConstraints::at(pairs, anchor.view())
        .unwrap()
        .with_masses(vec![12.0, 1.0, 5.0])
        .unwrap();
    let trial = &anchor + &kick(9, 0.08, 11);
    let mut settled = trial.clone();
    base.clone().constrain(anchor.view(), &mut settled).unwrap();
    let mut shaken = trial;
    base.with_settle(false)
        .with_solver(ConstraintSolver::Shake)
        .constrain(anchor.view(), &mut shaken)
        .unwrap();
    let gap = (&settled - &shaken)
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(gap <= 1e3 * U, "{gap:e}");
}

#[test]
fn unit_masses_are_the_default() {
    let anchor = water();
    let trial = &anchor + &kick(9, 0.05, 3);
    let plain = DistanceConstraints::at(vec![(0, 1), (0, 2), (1, 2)], anchor.view()).unwrap();
    let ones = plain.clone().with_masses(vec![1.0; 3]).unwrap();
    let (mut a, mut b) = (trial.clone(), trial);
    plain.constrain(anchor.view(), &mut a).unwrap();
    ones.constrain(anchor.view(), &mut b).unwrap();
    assert_eq!(a, b);
}

#[test]
fn lincs_order_reaches_the_target_bound() {
    assert_eq!(derived_order(0.0, U), 0);
    let rho: f64 = 0.4;
    let n = derived_order(rho, U);
    assert!(rho.powi(n as i32 + 1) / (1.0 - rho) <= U);
    assert!(rho.powi(n as i32) / (1.0 - rho) > U);
}

#[test]
fn projector_removes_bond_combinations_and_keeps_rotations() {
    let x = water() + kick(9, 0.01, 5);
    let set = DistanceConstraints::at(vec![(0, 1), (0, 2), (1, 2)], x.view())
        .unwrap()
        .with_masses(vec![O, H, H])
        .unwrap();
    let w = [1.0 / O, 1.0 / H, 1.0 / H];
    // W J^T c, written out from the bonds at x.
    let mut normal = Array1::<f64>::zeros(9);
    for (&(i, j), c) in set.pairs().iter().zip([0.7, -1.3, 0.4]) {
        let b = bond(x.view(), (i, j));
        for k in 0..3 {
            normal[3 * i + k] += w[i] * c * b[k];
            normal[3 * j + k] -= w[j] * c * b[k];
        }
    }
    let scale = normal.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut v = normal.clone();
    set.project_vector(x.view(), &mut v).unwrap();
    assert!(v.iter().all(|v| v.abs() <= 64.0 * U * scale), "{v}");

    let omega = [0.3, -0.5, 0.8];
    let mut spin = Array1::<f64>::zeros(9);
    for a in 0..3 {
        let r = [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
        spin[3 * a] = omega[1] * r[2] - omega[2] * r[1];
        spin[3 * a + 1] = omega[2] * r[0] - omega[0] * r[2];
        spin[3 * a + 2] = omega[0] * r[1] - omega[1] * r[0];
    }
    let mut t = spin.clone();
    set.project_vector(x.view(), &mut t).unwrap();
    assert!((&t - &spin).iter().all(|d| d.abs() <= 64.0 * U));

    let g = kick(9, 1.0, 9);
    let mut gh = g.clone();
    let lambda = set.project_gradient(x.view(), &mut gh).unwrap();
    // g - J^T lambda, with W (g - J^T lambda) tangent.
    let mut jt = Array1::<f64>::zeros(9);
    for (&(i, j), l) in set.pairs().iter().zip(&lambda) {
        let b = bond(x.view(), (i, j));
        for k in 0..3 {
            jt[3 * i + k] += l * b[k];
            jt[3 * j + k] -= l * b[k];
        }
    }
    assert!((&g - &jt - &gh).iter().all(|d| d.abs() <= 64.0 * U));
    let wg = Array1::from_shape_fn(9, |c| w[c / 3] * gh[c]);
    for &(i, j) in set.pairs() {
        let b = bond(x.view(), (i, j));
        assert!(dot3(&bond(wg.view(), (i, j)), &b).abs() <= 64.0 * U);
    }
    // Descent along a tangent direction reads the same slope.
    let mut d = kick(9, 1.0, 13);
    set.project_vector(x.view(), &mut d).unwrap();
    assert!((g.dot(&d) - gh.dot(&d)).abs() <= 64.0 * U);
}

#[test]
fn curve_velocity_matches_central_differences() {
    let x = water();
    let set = water_set().with_settle(false);
    let tangent = Tangent::new(&set, x.view()).unwrap();
    let mut d = kick(9, 1.0, 17);
    tangent.vector(&mut d).unwrap();
    let at = |t: f64| {
        let mut r = &x + &(t * &d);
        tangent.constrain(&mut r).unwrap();
        r
    };
    let (t, h) = (0.05, 1e-5);
    let fd = (at(t + h) - at(t - h)) / (2.0 * h);
    let v = tangent.curve_velocity(at(t).view(), &d).unwrap();
    let gap = (&fd - &v).iter().fold(0.0_f64, |m, e| m.max(e.abs()));
    assert!(gap <= 1e-8, "{gap:e}");
}

#[test]
fn malformed_sets_are_refused() {
    let dup = DistanceConstraints::new(vec![(0, 1), (1, 0)], vec![1.0, 1.0]);
    assert!(matches!(err(dup), ConstraintError::Malformed { .. }));
    let flat = DistanceConstraints::new(vec![(0, 1), (0, 2), (1, 2)], vec![1.0, 1.0, 2.0]);
    assert!(matches!(err(flat), ConstraintError::Malformed { .. }));
    let set = DistanceConstraints::new(vec![(0, 3)], vec![1.0]).unwrap();
    assert!(matches!(
        err(set.fits(9)),
        ConstraintError::OutOfRange { .. }
    ));
    let heavy = set.clone().with_masses(vec![1.0, -2.0, 1.0, 1.0]);
    assert!(matches!(err(heavy), ConstraintError::Mass { atom: 1 }));
    let short = set.with_masses(vec![1.0; 3]).unwrap();
    assert!(matches!(
        err(short.fits(12)),
        ConstraintError::Malformed { .. }
    ));
}

#[test]
fn a_collinear_anchor_is_dependent() {
    let line = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let set = DistanceConstraints::new(vec![(0, 1), (0, 2), (1, 2)], vec![1.0, 1.9, 1.0]).unwrap();
    let mut v = kick(9, 1.0, 1);
    assert!(matches!(
        err(set.project_vector(line.view(), &mut v)),
        ConstraintError::Dependent { .. }
    ));
}

#[test]
fn a_flipped_bond_is_reversed() {
    let anchor = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    for solver in [ConstraintSolver::Shake, ConstraintSolver::Lincs] {
        let set = DistanceConstraints::new(vec![(0, 1)], vec![1.0])
            .unwrap()
            .with_solver(solver);
        let mut x = array![0.0, 0.0, 0.0, -1.0, 0.1, 0.0];
        let e = err(set.constrain(anchor.view(), &mut x));
        assert!(matches!(e, ConstraintError::Reversed { .. }), "{e:?}");
    }
}

/// Three bonds from a light hub to heavy ends in a narrow cone: the
/// coupling has an eigenvalue above two, so the LINCS series diverges.
fn light_hub() -> (Array1<f64>, DistanceConstraints) {
    let (s, c) = 30_f64.to_radians().sin_cos();
    let mut x = Array1::zeros(12);
    for k in 0..3 {
        let phi = (120.0 * k as f64).to_radians();
        x[3 * (k + 1)] = s * phi.cos();
        x[3 * (k + 1) + 1] = s * phi.sin();
        x[3 * (k + 1) + 2] = c;
    }
    let set = DistanceConstraints::at(vec![(0, 1), (0, 2), (0, 3)], x.view())
        .unwrap()
        .with_masses(vec![1.0, 100.0, 100.0, 100.0])
        .unwrap();
    (x, set)
}

#[test]
fn lincs_refuses_a_divergent_block_and_auto_takes_shake() {
    let (anchor, set) = light_hub();
    let trial = &anchor + &kick(12, 0.02, 21);
    let mut x = trial.clone();
    let e = err(set
        .clone()
        .with_solver(ConstraintSolver::Lincs)
        .constrain(anchor.view(), &mut x));
    match e {
        ConstraintError::LincsUnsafe { rho, .. } => assert!(rho >= 1.0),
        other => panic!("{other:?}"),
    }
    let mut x = trial;
    let report = set.constrain(anchor.view(), &mut x).unwrap();
    assert_eq!((report.shake, report.lincs), (1, 0));
    assert!(set.excess(x.view()) <= 1.0);
}

#[test]
fn settle_reports_an_impossible_tilt() {
    let anchor = water();
    let mut x = anchor.clone();
    x[5] += 3.0;
    let e = err(water_set().constrain(anchor.view(), &mut x));
    assert!(
        matches!(e, ConstraintError::SettleDegenerate { .. }),
        "{e:?}"
    );
}

/// `set` on the atoms after a unit bond on atoms 0 and 1. The bond runs
/// along x from 2 to 3 at the anchor, and the trial pulls its free end
/// to 3.4.
fn behind_a_stretched_bond(
    set: DistanceConstraints,
    anchor: &Array1<f64>,
    trial: &Array1<f64>,
) -> (DistanceConstraints, Array1<f64>, Array1<f64>) {
    let shift = |&(i, j): &(usize, usize)| (i + 2, j + 2);
    let pairs = std::iter::once((0, 1))
        .chain(set.pairs().iter().map(shift))
        .collect();
    let lengths = std::iter::once(1.0)
        .chain(set.lengths().iter().copied())
        .collect();
    let mut joined = DistanceConstraints::new(pairs, lengths)
        .unwrap()
        .with_solver(set.solver())
        .with_settle(set.settle());
    if let Some(m) = set.masses() {
        let masses = [1.0, 1.0].into_iter().chain(m.iter().copied()).collect();
        joined = joined.with_masses(masses).unwrap();
    }
    let join = |bond: [f64; 6], rest: &Array1<f64>| -> Array1<f64> {
        bond.into_iter().chain(rest.iter().copied()).collect()
    };
    (
        joined,
        join([2.0, 0.0, 0.0, 3.0, 0.0, 0.0], anchor),
        join([2.0, 0.0, 0.0, 3.4, 0.0, 0.0], trial),
    )
}

/// On its own the stretched bond closes, each end moving 0.2, so the
/// free end goes from 3.4 to 3.2. Ahead of a block that fails, the trial
/// stays exactly as it was, whichever solver fails and wherever.
#[test]
fn a_failed_block_leaves_x_as_it_was() {
    let lone = DistanceConstraints::new(vec![(0, 1)], vec![1.0]).unwrap();
    let anchor = array![2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
    let mut x = array![2.0, 0.0, 0.0, 3.4, 0.0, 0.0];
    lone.constrain(anchor.view(), &mut x).unwrap();
    assert!((x[0] - 2.2).abs() <= 8.0 * U, "{x}");
    assert!((x[3] - 3.2).abs() <= 8.0 * U, "{x}");

    let (hub, hub_set) = light_hub();
    let mut tilted = water();
    tilted[5] += 3.0;
    // A unit bond whose trial end moves 0.9999 sideways closes along the
    // anchor bond at 0.014, and SHAKE, Newton on that length, halves its
    // way there for more sweeps than its budget. With the end 1.2
    // sideways no correction along the anchor bond reaches length 1, and
    // LINCS meets its reversal test after the first projection.
    let side = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let cases = [
        (
            hub_set.with_solver(ConstraintSolver::Lincs),
            hub.clone(),
            &hub + &kick(12, 0.02, 21),
        ),
        (water_set(), water(), tilted),
        (
            lone.clone().with_solver(ConstraintSolver::Shake),
            side.clone(),
            array![0.0, 0.0, 0.0, 1.0, 0.9999, 0.0],
        ),
        (
            lone.with_solver(ConstraintSolver::Lincs),
            side,
            array![0.0, 0.0, 0.0, 1.4, 1.2, 0.0],
        ),
    ];
    let mut errors = Vec::new();
    for (set, anchor, trial) in cases {
        let (set, anchor, trial) = behind_a_stretched_bond(set, &anchor, &trial);
        let mut x = trial.clone();
        let e = err(set.constrain(anchor.view(), &mut x));
        assert_eq!(x, trial, "{e:?}");
        errors.push(e);
    }
    assert!(
        matches!(errors[0], ConstraintError::LincsUnsafe { .. }),
        "{errors:?}"
    );
    assert!(
        matches!(errors[1], ConstraintError::SettleDegenerate { .. }),
        "{errors:?}"
    );
    assert!(
        matches!(
            errors[2],
            ConstraintError::NotConverged {
                algorithm: ConstraintAlgorithm::Shake,
                ..
            }
        ),
        "{errors:?}"
    );
    assert!(
        matches!(
            errors[3],
            ConstraintError::Reversed {
                algorithm: ConstraintAlgorithm::Lincs,
                ..
            }
        ),
        "{errors:?}"
    );
}

#[test]
fn non_finite_trials_are_errors() {
    let anchor = water();
    let mut x = anchor.clone();
    x[4] = f64::NAN;
    let e = err(water_set().constrain(anchor.view(), &mut x));
    assert_eq!(e, ConstraintError::NonFinite);
}

#[test]
fn a_user_tolerance_stops_early_and_the_floor_is_the_default() {
    let anchor = water();
    let trial = &anchor + &kick(9, 0.05, 23);
    let loose = water_set()
        .with_settle(false)
        .with_solver(ConstraintSolver::Shake)
        .with_tolerance(1e-4)
        .unwrap();
    let mut x = trial.clone();
    let few = loose.constrain(anchor.view(), &mut x).unwrap().iterations;
    assert!(loose.residual(x.view()) <= 1e-4);
    let tight = loose.with_tolerance(0.0).unwrap();
    let mut y = trial;
    let many = tight.constrain(anchor.view(), &mut y).unwrap().iterations;
    assert!(few < many, "{few} {many}");
    assert!(tight.residual(y.view()) <= 4.0 * GAMMA6);
}
