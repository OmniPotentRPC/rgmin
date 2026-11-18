mod common;

use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    ConstraintSolver, Control, DistanceConstraints, Error, GROMACS_STEEP_GROW,
    GROMACS_STEEP_SHRINK, GromacsSteepConfig, Method, NoiseFloor, Oracle, Solver, Status,
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

fn bond_length(x: &Array1<f64>) -> f64 {
    let dx = x[3] - x[0];
    let dy = x[4] - x[1];
    let dz = x[5] - x[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn lincs_pair() -> DistanceConstraints {
    DistanceConstraints::new(vec![(0, 1)], vec![1.0])
        .unwrap()
        .with_solver(ConstraintSolver::Lincs)
}

/// Lateral kick that LINCS reverses on a unit bond: the trial end is
/// `(1.4, 1.2, 0)` when `h` is the length of `(0.4, 1.2, 0)`.
fn sideways_gradient() -> Array1<f64> {
    array![0.0, 0.0, 0.0, -0.4, -1.2, 0.0]
}

fn failing_step() -> f64 {
    (0.4_f64 * 0.4 + 1.2 * 1.2).sqrt()
}

#[test]
fn a_constraint_failure_shrinks_and_the_bound_then_errors() {
    let gradient = sideways_gradient();
    let objective = Oracle::unbounded(6, move |x: ArrayView1<f64>| {
        (x.dot(&gradient), gradient.clone())
    });
    let start = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let h = failing_step();

    let mut kept = start.clone();
    let mut solver = Solver::new(Method::GromacsSteep, control(h, 0.0, 20), kept.len());
    solver.set_distance_constraints(lincs_pair()).unwrap();
    let report = solver.step(&objective, &mut kept).unwrap();
    assert_eq!(report.status, Status::Iterating);
    assert_eq!(kept, start);
    assert_eq!(
        solver.gromacs_step_size().unwrap().to_bits(),
        (h * GROMACS_STEEP_SHRINK).to_bits()
    );

    let mut bounded = start.clone();
    let mut solver = Solver::new(
        Method::GromacsSteep,
        control(h * 4.0, 0.0, 20),
        bounded.len(),
    );
    solver.set_distance_constraints(lincs_pair()).unwrap();
    solver
        .set_gromacs_steep_config(GromacsSteepConfig::default().with_max_constraint_shrinks(2))
        .unwrap();
    for shrinks in 1..=2 {
        let report = solver.step(&objective, &mut bounded).unwrap();
        assert_eq!(report.status, Status::Iterating);
        assert_eq!(bounded, start);
        let shrunk = h * 4.0 * GROMACS_STEEP_SHRINK.powi(shrinks);
        assert_eq!(
            solver.gromacs_step_size().unwrap().to_bits(),
            shrunk.to_bits()
        );
    }
    let err = solver.step(&objective, &mut bounded).unwrap_err();
    assert!(matches!(err, Error::Constraint(_)), "{err}");
    assert_eq!(bounded, start);
    assert_eq!(solver.gromacs_step_size().unwrap().to_bits(), h.to_bits());

    let mut immediate = start.clone();
    let mut solver = Solver::new(Method::GromacsSteep, control(h, 0.0, 20), immediate.len());
    solver.set_distance_constraints(lincs_pair()).unwrap();
    solver
        .set_gromacs_steep_config(GromacsSteepConfig::default().with_max_constraint_shrinks(0))
        .unwrap();
    let err = solver.step(&objective, &mut immediate).unwrap_err();
    assert!(matches!(err, Error::Constraint(_)), "{err}");
    assert_eq!(immediate, start);
    assert_eq!(solver.gromacs_step_size().unwrap().to_bits(), h.to_bits());

    let mut placed = start.clone();
    let mut solver = Solver::new(Method::GromacsSteep, control(h, 0.0, 20), placed.len());
    solver.set_distance_constraints(lincs_pair()).unwrap();
    let mut moved = false;
    for _ in 0..8 {
        solver.step(&objective, &mut placed).unwrap();
        if placed != start {
            moved = true;
            break;
        }
    }
    assert!(moved);
    let length = bond_length(&placed);
    assert_eq!(length.to_bits(), 1.0_f64.to_bits());
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
