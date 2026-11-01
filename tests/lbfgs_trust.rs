//! Opt-in limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) trust radius of Burdakov, Gong, Zikrin, and Yuan (2017).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use eindir_core::objectives::Rosenbrock;
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Oracle, Solver};

fn control() -> Control {
    Control {
        maxiter: 400,
        gtol: 1e-6,
        istep: 0.2,
        maxmove: None,
    }
}

fn quartic(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    (x[0].powi(4), array![4.0 * x[0].powi(3)])
}

/// Three atoms joined by springs of rest length 1.
fn springs(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mut e = 0.0;
    let mut g = Array1::zeros(9);
    for i in 0..3 {
        for j in (i + 1)..3 {
            let d: [f64; 3] = std::array::from_fn(|a| x[3 * i + a] - x[3 * j + a]);
            let r = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            e += 0.5 * (r - 1.0).powi(2);
            for a in 0..3 {
                g[3 * i + a] += (r - 1.0) * d[a] / r;
                g[3 * j + a] -= (r - 1.0) * d[a] / r;
            }
        }
    }
    (e, g)
}

fn centroid(x: &Array1<f64>) -> [f64; 3] {
    std::array::from_fn(|a| (x[a] + x[3 + a] + x[6 + a]) / 3.0)
}

/// Off, the raw first step `-g` is clipped to the box corner and the
/// session stays there. On, it reaches (1, 1). `Accept::None`, no cap.
#[test]
fn rosenbrock_accept_none_uncapped_needs_the_radius() {
    let obj = Rosenbrock::<2>::new();
    for radius in [0.0, 0.2] {
        let mut solver = Solver::new(Method::Lbfgs { memory: 20 }, control(), 2);
        solver.set_lbfgs_trust(radius);
        let mut x = array![-1.2, 1.0];
        let mut gnorm = f64::INFINITY;
        for _ in 0..400 {
            gnorm = solver.step(&obj, &mut x).unwrap().grad_norm;
            if gnorm < 1e-6 {
                break;
            }
        }
        if radius > 0.0 {
            assert!(gnorm < 1e-6, "{gnorm} {x:?}");
            assert!((x[0] - 1.0).abs() < 1e-5 && (x[1] - 1.0).abs() < 1e-5);
        } else {
            assert!(gnorm > 1.0, "{gnorm}");
            assert_eq!(x, array![2.048, 2.048]);
        }
    }
}

/// `f = |x|^2 / 2` is its own model, so `rho = 1`. The step reaches the
/// boundary and the radius doubles.
#[test]
fn exact_model_on_the_boundary_doubles_the_radius() {
    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
    let mut solver = Solver::new(Method::lbfgs(), control(), 2);
    assert_eq!(solver.lbfgs_trust_radius(), None);
    solver.set_lbfgs_trust(0.5);
    let mut x = array![10.0, 0.0];
    solver.step(&obj, &mut x).unwrap();
    assert!((x[0] - 9.5).abs() < 1e-12 && x[1] == 0.0, "{x:?}");
    assert_eq!(solver.lbfgs_trust_radius(), Some(1.0));
}

/// `x^4` from 1 with radius 1.9: the step to -0.9 drops the energy by
/// 0.3439 against a predicted 5.795, so it is taken and the radius
/// falls to `min(1.9/4, 1.9/2)`.
#[test]
fn poor_agreement_takes_the_step_and_shrinks() {
    let obj = Oracle::unbounded(1, quartic);
    let mut solver = Solver::new(Method::lbfgs(), control(), 1);
    solver.set_lbfgs_trust(1.9);
    let mut x = array![1.0];
    solver.step(&obj, &mut x).unwrap();
    assert!((x[0] + 0.9).abs() < 1e-12, "{x:?}");
    assert!((solver.lbfgs_trust_radius().unwrap() - 0.475).abs() < 1e-12);
    assert_eq!(solver.pair_count(), 1);
}

/// With radius 3 the trial at -2 rises to 16. It is refused after one
/// oracle call, `x` and the pairs stay, and the radius drops to 0.75.
/// `forget` restores the starting radius.
#[test]
fn a_rise_is_refused_and_shrinks_the_radius() {
    let calls = AtomicUsize::new(0);
    let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| {
        calls.fetch_add(1, Ordering::Relaxed);
        quartic(x)
    });
    let mut solver = Solver::new(Method::lbfgs(), control(), 1);
    solver.set_lbfgs_trust(3.0);
    let mut x = array![1.0];
    let rep = solver.step(&obj, &mut x).unwrap();
    assert_eq!(x, array![1.0]);
    assert_eq!(rep.value, 1.0);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(solver.pair_count(), 0);
    assert_eq!(solver.lbfgs_trust_radius(), Some(0.75));
    solver.step(&obj, &mut x).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert!((x[0] - 0.25).abs() < 1e-12, "{x:?}");
    solver.forget();
    assert_eq!(solver.lbfgs_trust_radius(), Some(3.0));
    solver.set_lbfgs_trust(0.0);
    assert_eq!(solver.lbfgs_trust_radius(), None);
}

/// A host move that translates the cluster stores a pair with a rigid
/// part. With `set_project_rigid`, the next trust step still leaves the
/// centroid where it was.
#[test]
fn rigid_modes_leave_the_trust_direction() {
    let obj = Oracle::unbounded(9, springs);
    let mut solver = Solver::new(Method::lbfgs(), control(), 9);
    solver.set_project_rigid(true);
    solver.set_lbfgs_trust(0.1);
    let mut x = array![0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.3, 0.0];
    solver.step(&obj, &mut x).unwrap();
    for k in 0..3 {
        x[3 * k] += 0.3;
    }
    x[3] += 0.05;
    let before = x.clone();
    solver.step(&obj, &mut x).unwrap();
    assert_eq!(solver.pair_count(), 3);
    assert!((&x - &before).dot(&(&x - &before)) > 1e-6, "{x:?}");
    let (c0, c1) = (centroid(&before), centroid(&x));
    for a in 0..3 {
        assert!((c1[a] - c0[a]).abs() < 1e-12, "{c0:?} {c1:?}");
    }
}

/// `f = x/10` with slope 1 at 0: from radius `2e-15` the step is taken
/// at `rho = 0.1` and the radius falls to `5e-16`. The session reports the
/// collapse and keeps the start.
#[test]
fn an_accepted_step_below_the_floor_collapses() {
    let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| (0.1 * x[0], array![1.0]));
    let mut solver = Solver::new(Method::lbfgs(), control(), 1);
    solver.set_lbfgs_trust(2e-15);
    let mut x = array![0.0];
    let err = solver.step(&obj, &mut x).unwrap_err();
    assert!(
        matches!(err, rgmin::Error::TrustCollapsed { steps: 1 }),
        "{err:?}"
    );
    assert_eq!(x, array![0.0]);
    assert!(solver.lbfgs_trust_radius().unwrap() < 1e-15);
}

/// A gradient of `1e200` gives `g^T d = -inf` and `|d| = inf`. The step
/// refuses it as a non-finite direction instead of accepting `tau = 0`.
#[test]
fn an_infinite_slope_is_a_non_finite_direction() {
    let obj = Oracle::unbounded(1, |_: ArrayView1<f64>| (1.0, array![1e200]));
    let mut solver = Solver::new(Method::lbfgs(), control(), 1);
    solver.set_lbfgs_trust(0.5);
    let mut x = array![3.0];
    let err = solver.step(&obj, &mut x).unwrap_err();
    assert!(
        matches!(
            err,
            rgmin::Error::Oracle {
                what: "non-finite search direction"
            }
        ),
        "{err:?}"
    );
    assert_eq!(x, array![3.0]);
    assert_eq!(solver.pair_count(), 0);
    assert_eq!(solver.lbfgs_trust_radius(), Some(0.5));
}

/// A zero gradient passes the stop test at `gtol = 0`. Its direction is
/// finite with slope 0, so the step refuses it as no descent, not as a
/// non-finite direction.
#[test]
fn a_finite_uphill_direction_is_no_descent() {
    let obj = Oracle::unbounded(1, |_: ArrayView1<f64>| (1.0, array![0.0]));
    let ctrl = Control {
        gtol: 0.0,
        ..control()
    };
    let mut solver = Solver::new(Method::lbfgs(), ctrl, 1);
    solver.set_lbfgs_trust(0.5);
    let mut x = array![3.0];
    let err = solver.step(&obj, &mut x).unwrap_err();
    assert!(
        matches!(err, rgmin::Error::NoDescent { steps: 0 }),
        "{err:?}"
    );
    assert_eq!(x, array![3.0]);
    assert_eq!(solver.pair_count(), 0);
    assert_eq!(solver.lbfgs_trust_radius(), Some(0.5));
}

/// `x^2/2` from 1 with radius 0.5 answers NaN below 0.8. The trial at 0.5
/// counts as `rho < 0`: the step reports the start, keeps the pairs empty
/// and shrinks the radius to `min(0.5/4, 0.5/2)`. The next trial, at
/// 0.875, matches the model and doubles the radius.
#[test]
fn a_non_finite_trial_reports_x_and_shrinks_the_radius() {
    let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| {
        let f = if x[0] < 0.8 {
            f64::NAN
        } else {
            0.5 * x[0] * x[0]
        };
        (f, array![x[0]])
    });
    let mut solver = Solver::new(Method::lbfgs(), control(), 1);
    solver.set_lbfgs_trust(0.5);
    let mut x = array![1.0];
    let rep = solver.step(&obj, &mut x).unwrap();
    assert_eq!(
        (rep.value, &rep.coords, &x),
        (0.5, &array![1.0], &array![1.0])
    );
    assert_eq!(solver.pair_count(), 0);
    assert_eq!(solver.lbfgs_trust_radius(), Some(0.125));
    solver.step(&obj, &mut x).unwrap();
    assert_eq!(x, array![0.875]);
    assert_eq!(solver.pair_count(), 1);
    assert_eq!(solver.lbfgs_trust_radius(), Some(0.25));
}

/// Every point but the start answers NaN. From radius `3e-15` the shrink
/// after the refused trial falls below `1e-15`, so the step returns
/// `TrustCollapsed` and keeps the start.
#[test]
fn a_non_finite_trial_below_the_floor_collapses() {
    let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| {
        let f = if x[0] == 1.0 { 0.5 } else { f64::NAN };
        (f, array![x[0]])
    });
    let mut solver = Solver::new(Method::lbfgs(), control(), 1);
    solver.set_lbfgs_trust(3e-15);
    let mut x = array![1.0];
    let err = solver.step(&obj, &mut x).unwrap_err();
    assert!(
        matches!(err, rgmin::Error::TrustCollapsed { steps: 1 }),
        "{err:?}"
    );
    assert_eq!(x, array![1.0]);
    assert_eq!(solver.pair_count(), 0);
    assert!(solver.lbfgs_trust_radius().unwrap() < 1e-15);
}

/// A NaN value at the start defines no direction. The step returns
/// `Error::Oracle` before it reaches the radius, which stays.
#[test]
fn a_non_finite_start_is_an_oracle_error() {
    let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| (f64::NAN, array![x[0]]));
    let mut solver = Solver::new(Method::lbfgs(), control(), 1);
    solver.set_lbfgs_trust(0.5);
    let mut x = array![1.0];
    let err = solver.step(&obj, &mut x).unwrap_err();
    assert!(
        matches!(
            err,
            rgmin::Error::Oracle {
                what: "non-finite value or gradient"
            }
        ),
        "{err:?}"
    );
    assert_eq!(x, array![1.0]);
    assert_eq!(solver.pair_count(), 0);
    assert_eq!(solver.lbfgs_trust_radius(), Some(0.5));
}

/// After one good step the caller moves `x`, and every trial from there
/// answers NaN until the engine opens again. The first refused step
/// records the caller's move as one pair. The second starts from the
/// remembered point and records none, and the step after the engine
/// opens adds only its own pair.
#[test]
fn a_refused_trust_trial_records_the_caller_move_once() {
    let open = AtomicBool::new(true);
    let moved = array![0.25, 0.38, 0.63];
    let fg = |x: ArrayView1<f64>| {
        let w = array![0.1, 0.2, 0.3];
        let f = (&w * &x.mapv(|v| (v - 1.0) * (v - 1.0))).sum();
        let answered = open.load(Ordering::SeqCst) || x == moved;
        let f = if answered { f } else { f64::NAN };
        (f, 2.0 * &w * &x.mapv(|v| v - 1.0))
    };
    let obj = Oracle::unbounded(3, &fg);
    let mut solver = Solver::new(Method::lbfgs(), control(), 3);
    solver.set_lbfgs_trust(0.1);
    let mut x = Array1::zeros(3);
    solver.step(&obj, &mut x).unwrap();
    let pairs = solver.pair_count();
    assert_eq!(pairs, 1);
    x.assign(&moved);
    open.store(false, Ordering::SeqCst);
    for _ in 0..2 {
        let rep = solver.step(&obj, &mut x).unwrap();
        assert_eq!((&rep.coords, &x), (&moved, &moved));
        assert_eq!(solver.pair_count(), pairs + 1);
    }
    open.store(true, Ordering::SeqCst);
    solver.step(&obj, &mut x).unwrap();
    assert_ne!(x, moved);
    assert_eq!(solver.pair_count(), pairs + 2);
}
