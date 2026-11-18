mod common;

use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, DistanceConstraints, Error, GROMACS_STEEP_GROW, GROMACS_STEEP_SHRINK,
    GromacsSteepConfig, Method, NoiseFloor, Oracle, Solver, Status,
};

use common::morse_value_grad;

fn control(h: f64, gtol: f64, maxiter: usize) -> Control {
    Control::default()
        .with_istep(h)
        .with_gtol(gtol)
        .with_maxiter(maxiter)
        .with_maxmove(None)
}

#[test]
fn morse_dimer_has_the_exact_accept_reject_h_sequence() {
    let objective = Oracle::unbounded(6, |x: ArrayView1<f64>| {
        morse_value_grad(x, [100.0, 100.0, 100.0])
    });
    let mut x = array![0.0, 0.0, 0.0, 2.8, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsSteep, control(0.1, 0.0, 10), x.len());

    // Accept grows by 1.2, the overshoot is refused and multiplied by the
    // GROMACS 0.5 factor, and the shorter trial is accepted.
    let mut h = 0.1;
    let mut distance = 2.8;
    let accepted = [true, false, true];
    for &take in &accepted {
        let report = solver.step(&objective, &mut x).unwrap();
        assert_eq!(report.status, Status::Iterating);
        if take {
            distance += if distance < 3.0 { 2.0 * h } else { -2.0 * h };
            h *= GROMACS_STEEP_GROW;
        } else {
            h *= GROMACS_STEEP_SHRINK;
        }
        assert_eq!(solver.gromacs_step_size().unwrap().to_bits(), h.to_bits());
        assert_eq!((x[3] - x[0]).to_bits(), distance.to_bits());
    }
}

#[test]
fn with_shrink_replays_the_reference_manual_sequence() {
    let objective = Oracle::unbounded(6, |x: ArrayView1<f64>| {
        morse_value_grad(x, [100.0, 100.0, 100.0])
    });
    let mut x = array![0.0, 0.0, 0.0, 2.8, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsSteep, control(0.1, 0.0, 10), x.len());
    solver
        .set_gromacs_steep_config(GromacsSteepConfig::default().with_shrink(0.2))
        .unwrap();

    let mut h = 0.1;
    let mut distance = 2.8;
    let accepted = [true, false, true];
    for &take in &accepted {
        solver.step(&objective, &mut x).unwrap();
        if take {
            distance += if distance < 3.0 { 2.0 * h } else { -2.0 * h };
            h *= GROMACS_STEEP_GROW;
        } else {
            h *= 0.2;
        }
        assert_eq!(solver.gromacs_step_size().unwrap().to_bits(), h.to_bits());
        assert_eq!((x[3] - x[0]).to_bits(), distance.to_bits());
    }
}

#[test]
fn steep_config_is_refused_outside_a_steep_session_and_outside_the_unit_interval() {
    let mut solver = Solver::new(Method::GromacsSteep, control(0.1, 0.0, 1), 3);
    for factor in [0.0, 1.0, -0.5, 2.0, f64::NAN, f64::INFINITY] {
        let err = solver
            .set_gromacs_steep_config(GromacsSteepConfig::default().with_shrink(factor))
            .unwrap_err();
        assert!(matches!(err, Error::Steep { .. }), "{factor}: {err}");
    }
    let mut other = Solver::new(Method::Steepest, control(0.1, 0.0, 1), 3);
    let err = other
        .set_gromacs_steep_config(GromacsSteepConfig::default())
        .unwrap_err();
    assert!(matches!(err, Error::Steep { .. }), "{err}");
}

#[test]
fn tolerance_budget_and_machine_precision_are_distinct_stops() {
    let quadratic = Oracle::unbounded(3, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
    let mut at_tolerance = array![1.0, 0.0, 0.0];
    let mut solver = Solver::new(
        Method::GromacsSteep,
        control(0.1, 1.1, 5),
        at_tolerance.len(),
    );
    let report = solver.run(&quadratic, &mut at_tolerance).unwrap();
    assert_eq!(report.status, Status::Tolerance);
    assert_eq!(report.steps, 0);

    let mut budgeted = array![1.0, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsSteep, control(0.1, 0.0, 2), budgeted.len());
    let report = solver.run(&quadratic, &mut budgeted).unwrap();
    assert_eq!(report.status, Status::Budget);
    assert_eq!(report.steps, 2);

    let linear = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        (x[0], Array1::from_vec(vec![1.0, 0.0, 0.0]))
    });
    let mut rounded = array![1.0, 0.0, 0.0];
    let mut solver = Solver::new(
        Method::GromacsSteep,
        control(f64::MIN_POSITIVE, 0.0, 5),
        rounded.len(),
    );
    let report = solver.run(&linear, &mut rounded).unwrap();
    assert_eq!(report.status, Status::MachinePrecision);
    assert_eq!(rounded[0], 1.0);
}

#[test]
fn forget_resets_step_size_and_holonomic_constraints_stay_on_bond() {
    let objective = Oracle::unbounded(6, |x: ArrayView1<f64>| {
        morse_value_grad(x, [100.0, 100.0, 100.0])
    });
    let mut x = array![0.0, 0.0, 0.0, 2.8, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsSteep, control(0.1, 0.0, 10), x.len());
    solver.step(&objective, &mut x).unwrap();
    assert_ne!(solver.gromacs_step_size().unwrap(), 0.1);
    solver.forget();
    assert_eq!(solver.gromacs_step_size().unwrap(), 0.1);

    // Tethered dumbbell under a holonomic distance constraint with a noise floor.
    let quadratic = Oracle::unbounded(6, |v: ArrayView1<f64>| {
        let target = array![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let diff = &v - &target;
        (0.5 * diff.dot(&diff), diff)
    });
    let set = DistanceConstraints::new(vec![(0, 1)], vec![1.0]).unwrap();
    let mut constrained = array![0.0, 0.5, 0.0, 1.0, 0.5, 0.0];
    let ctrl = control(0.1, 1e-12, 50).with_noise_floor(NoiseFloor::new(10.0));
    let mut solver = Solver::new(Method::GromacsSteep, ctrl, constrained.len());
    solver.set_distance_constraints(set).unwrap();
    let report = solver.run(&quadratic, &mut constrained).unwrap();
    assert_eq!(report.status, Status::NoiseFloor);
    let bond = ((constrained[3] - constrained[0]).powi(2)
        + (constrained[4] - constrained[1]).powi(2)
        + (constrained[5] - constrained[2]).powi(2))
    .sqrt();
    assert!((bond - 1.0).abs() < 1e-12);
}
