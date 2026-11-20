//! Integration tests for GPMin: Muller-Brown, small Lennard-Jones cluster,
//! prior strategies, hyperparameter updates, `Stepper` / `Solver` sessions,
//! and typed error handling.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, Error, GpMin, GpMinConfig, Method, Oracle, PriorStrategy, Solver, Status, Stepper,
    StopNorm,
};

/// Standard 2D Muller-Brown potential and analytic gradient.
///
/// Known local minima:
/// - M1: `(-0.55822363, 1.44172584)`, `E = -146.6995172`
/// - M2: `( 0.62349940, 0.02803776)`, `E = -108.1667241`
/// - M3: `(-0.05001082, 0.46669410)`, `E =  -80.7678181`
fn muller_brown(xy: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    const A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
    const AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
    const B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
    const C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
    const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
    const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

    let x = xy[0];
    let y = xy[1];
    let mut e = 0.0;
    let mut gx = 0.0;
    let mut gy = 0.0;
    for k in 0..4 {
        let dx = x - X0[k];
        let dy = y - Y0[k];
        let term = A[k] * (AA[k] * dx * dx + B[k] * dx * dy + C[k] * dy * dy).exp();
        e += term;
        gx += term * (2.0 * AA[k] * dx + B[k] * dy);
        gy += term * (B[k] * dx + 2.0 * C[k] * dy);
    }
    (e, array![gx, gy])
}

/// Pairwise Lennard-Jones potential (`sigma`, `epsilon`) with analytic gradient.
fn lj_cluster(coords: ArrayView1<'_, f64>, sigma: f64, epsilon: f64) -> (f64, Array1<f64>) {
    let n = coords.len() / 3;
    let sigma2 = sigma * sigma;
    let mut e = 0.0;
    let mut g = Array1::zeros(coords.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[3 * i] - coords[3 * j];
            let dy = coords[3 * i + 1] - coords[3 * j + 1];
            let dz = coords[3 * i + 2] - coords[3 * j + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let inv_r2 = 1.0 / r2;
            let sr2 = sigma2 * inv_r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            e += 4.0 * epsilon * (sr12 - sr6);
            // dV/dr * (1/r) = 24 * eps * (sr6 - 2 * sr12) / r^2
            let factor = 24.0 * epsilon * (sr6 - 2.0 * sr12) * inv_r2;
            let fx = factor * dx;
            let fy = factor * dy;
            let fz = factor * dz;
            g[3 * i] += fx;
            g[3 * i + 1] += fy;
            g[3 * i + 2] += fz;
            g[3 * j] -= fx;
            g[3 * j + 1] -= fy;
            g[3 * j + 2] -= fz;
        }
    }
    (e, g)
}

/// Perturbed 7-atom pentagonal bipyramid (global minimum of LJ7: E = -16.50538417 * eps).
fn lj7_start(sigma: f64) -> Array1<f64> {
    let r_eq = 2.0_f64.powf(1.0 / 6.0);
    let mut pts = Vec::with_capacity(21);
    let base: [[f64; 3]; 7] = [
        [0.0, 0.0, 0.55 * r_eq],
        [0.0, 0.0, -0.55 * r_eq],
        [
            0.95 * r_eq * (0.0_f64).cos(),
            0.95 * r_eq * (0.0_f64).sin(),
            0.0,
        ],
        [
            0.95 * r_eq * (2.0 * std::f64::consts::PI / 5.0).cos(),
            0.95 * r_eq * (2.0 * std::f64::consts::PI / 5.0).sin(),
            0.0,
        ],
        [
            0.95 * r_eq * (4.0 * std::f64::consts::PI / 5.0).cos(),
            0.95 * r_eq * (4.0 * std::f64::consts::PI / 5.0).sin(),
            0.0,
        ],
        [
            0.95 * r_eq * (6.0 * std::f64::consts::PI / 5.0).cos(),
            0.95 * r_eq * (6.0 * std::f64::consts::PI / 5.0).sin(),
            0.0,
        ],
        [
            0.95 * r_eq * (8.0 * std::f64::consts::PI / 5.0).cos(),
            0.95 * r_eq * (8.0 * std::f64::consts::PI / 5.0).sin(),
            0.0,
        ],
    ];
    let offsets: [[f64; 3]; 7] = [
        [0.04, -0.03, 0.05],
        [-0.03, 0.04, -0.04],
        [0.05, 0.02, -0.03],
        [-0.02, -0.05, 0.04],
        [0.03, -0.04, 0.02],
        [-0.04, 0.03, -0.02],
        [0.02, 0.05, -0.03],
    ];
    for i in 0..7 {
        for c in 0..3 {
            pts.push((base[i][c] + offsets[i][c]) * sigma);
        }
    }
    Array1::from_vec(pts)
}

#[test]
fn muller_brown_converges_with_standalone_gpmin_and_solver() {
    let obj = Oracle::unbounded(2, muller_brown);
    let ctrl = Control::default()
        .with_gtol(1e-4)
        .with_maxiter(80)
        .with_stop_norm(StopNorm::Two);

    // 1. Standalone GpMin stepper (fixed hyperparameters)
    let mut x = array![-0.3, 1.2];
    let cfg = GpMinConfig::default()
        .with_scale(0.3)
        .with_weight(50.0)
        .with_noise(0.001)
        .with_max_step(Some(0.2));
    let mut gpmin = GpMin::new(cfg, 2).unwrap();
    let report = gpmin.run(&obj, &mut x, &ctrl).unwrap();
    assert_eq!(report.status, Status::Tolerance);
    assert!(report.grad_norm <= 1e-4, "grad_norm = {}", report.grad_norm);
    assert!(
        (report.value - (-146.6995172)).abs() < 1e-5,
        "unexpected minimum energy {}",
        report.value
    );
    assert!(
        report.evals < 40,
        "GPMin should converge on Muller-Brown quickly, got {} evals",
        report.evals
    );

    // 2. Solver session with hyperparameter updates enabled
    let mut x_hp = array![-0.3, 1.2];
    let mut solver = Solver::new(Method::GpMin, ctrl, 2);
    solver
        .set_gpmin_config(
            GpMinConfig::with_hyperparam_updates()
                .with_scale(0.3)
                .with_weight(50.0)
                .with_noise(0.004)
                .with_batch_size(2),
        )
        .unwrap();
    let rep_hp = solver.run(&obj, &mut x_hp).unwrap();
    assert_eq!(rep_hp.status, Status::Tolerance);
    assert!((rep_hp.value - (-146.6995172)).abs() < 1e-5);
    let (w_after, l_after, _) = solver.gpmin_hyperparameters().unwrap();
    assert!(w_after.is_finite() && w_after > 0.0);
    assert!(l_after.is_finite() && l_after > 0.0);
}

#[test]
fn lj7_cluster_converges_fixed_and_updated_hyperparams() {
    let sigma = 2.5;
    let epsilon = 0.1;
    let obj = Oracle::unbounded(21, move |coords| lj_cluster(coords, sigma, epsilon));
    let expected_energy = -16.505384167 * epsilon;

    let ctrl = Control::default()
        .with_gtol(1e-3)
        .with_maxiter(100)
        .with_stop_norm(StopNorm::MaxAtom);

    // Fixed hyperparameters (ASE defaults: scale=0.4, weight=1.0, noise=0.001)
    let mut x_fixed = lj7_start(sigma);
    let mut solver = Solver::new(Method::GpMin, ctrl.clone(), 21);
    let rep_fixed = solver.run(&obj, &mut x_fixed).unwrap();
    assert_eq!(rep_fixed.status, Status::Tolerance);
    assert!(rep_fixed.grad_norm <= 1e-3);
    assert!(
        (rep_fixed.value - expected_energy).abs() < 1e-4,
        "fixed LJ7 energy {} vs expected {}",
        rep_fixed.value,
        expected_energy
    );

    // Updated hyperparameters (ASE update_hyperparams=True defaults)
    let mut x_upd = lj7_start(sigma);
    let mut solver_upd = Solver::new(Method::GpMin, ctrl, 21);
    solver_upd
        .set_gpmin_config(GpMinConfig::with_hyperparam_updates())
        .unwrap();
    let rep_upd = solver_upd.run(&obj, &mut x_upd).unwrap();
    assert_eq!(rep_upd.status, Status::Tolerance);
    assert!(rep_upd.grad_norm <= 1e-3);
    assert!(
        (rep_upd.value - expected_energy).abs() < 1e-4,
        "updated LJ7 energy {} vs expected {}",
        rep_upd.value,
        expected_energy
    );
}

#[test]
fn prior_strategies_update_prior_mean_as_documented() {
    let obj = Oracle::unbounded(2, |x: ArrayView1<'_, f64>| {
        (
            2.0 + x[0] * x[0] + 3.0 * x[1] * x[1],
            array![2.0 * x[0], 6.0 * x[1]],
        )
    });
    for strategy in [
        PriorStrategy::Maximum,
        PriorStrategy::Init,
        PriorStrategy::Average,
        PriorStrategy::Fixed(1.5),
    ] {
        let mut gpmin = GpMin::new(GpMinConfig::default().with_prior(strategy), 2).unwrap();
        let mut x = array![0.3, -0.2];
        let ctrl = Control::default().with_gtol(1e-5).with_maxiter(60);
        let rep = gpmin.run(&obj, &mut x, &ctrl).unwrap();
        assert_eq!(rep.status, Status::Tolerance);
        assert!((rep.value - 2.0).abs() < 1e-6);
        if let PriorStrategy::Fixed(c) = strategy {
            assert!((gpmin.prior_mean() - c).abs() <= f64::EPSILON);
        }
    }
}

#[test]
fn stepper_trait_and_error_paths_fail_closed() {
    // Invalid configs fail closed with typed Error::GpMin.
    assert!(matches!(
        GpMin::new(GpMinConfig::default().with_scale(0.0), 2),
        Err(Error::GpMin { .. })
    ));
    assert!(matches!(
        GpMin::new(GpMinConfig::default().with_weight(-1.0), 2),
        Err(Error::GpMin { .. })
    ));
    assert!(matches!(
        GpMin::new(GpMinConfig::default().with_noise(f64::NAN), 2),
        Err(Error::GpMin { .. })
    ));
    assert!(matches!(
        GpMin::new(GpMinConfig::default().with_hyperparam_bounds(Some(1.0)), 2),
        Err(Error::GpMin { .. })
    ));
    assert!(matches!(
        GpMin::new(GpMinConfig::with_hyperparam_updates().with_batch_size(0), 2),
        Err(Error::GpMin { .. })
    ));
    assert!(matches!(
        GpMin::new(GpMinConfig::default(), 0),
        Err(Error::GpMin { .. })
    ));

    // set_gpmin_config on non-GPMin session fails closed.
    let mut lbfgs = Solver::new(Method::lbfgs(), Control::default(), 2);
    assert!(matches!(
        lbfgs.set_gpmin_config(GpMinConfig::default()),
        Err(Error::GpMin { .. })
    ));

    // Stepper trait advances a quadratic and lowers energy.
    let quad = Oracle::unbounded(2, |x: ArrayView1<'_, f64>| {
        (x.dot(&x), array![2.0 * x[0], 2.0 * x[1]])
    });
    let mut gpmin = GpMin::new(GpMinConfig::default(), 2).unwrap();
    let mut x = array![0.3, -0.4];
    let r1 = Stepper::step(&mut gpmin, &quad, &mut x).unwrap();
    assert!(r1.value < 0.25);

    // Non-finite energy on a trial step fails closed with Error::Oracle.
    let bad = Oracle::unbounded(2, |x: ArrayView1<'_, f64>| {
        if x[0] < 0.2 {
            (f64::INFINITY, array![0.0, 0.0])
        } else {
            (x.dot(&x), array![2.0 * x[0], 2.0 * x[1]])
        }
    });
    let mut gpmin_bad = GpMin::new(GpMinConfig::default().with_scale(0.3), 2).unwrap();
    let mut x_bad = array![0.3, 0.0];
    assert!(matches!(
        gpmin_bad.step(&bad, &mut x_bad),
        Err(Error::Oracle { .. })
    ));
}
