mod common;

use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    ConstraintSolver, Control, DistanceConstraints, Error, GROMACS_STEEP_GROW,
    GROMACS_STEEP_SHRINK, GromacsSteepConfig, Method, NoiseFloor, Oracle, Solver, Status, StopNorm,
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
    // LINCS enforces relative bond errors at a rounding-level floor, so
    // the post-step bond is numerically close to 1.0, not bit-exact.
    let lincs_length_tolerance = 64.0 * f64::EPSILON;
    assert!(
        (length - 1.0).abs() <= lincs_length_tolerance,
        "length {length} exceeds LINCS floor {lincs_length_tolerance}",
    );
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
fn constrained_zero_budget_halt_uses_max_atom_norm() {
    let objective = Oracle::unbounded(6, |v: ArrayView1<f64>| {
        (
            0.6 * (v[0] + v[3]),
            Array1::from_vec(vec![0.6, 0.0, 0.0, 0.6, 0.0, 0.0]),
        )
    });
    let mut x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let ctrl = control(0.1, 0.0, 0).with_noise_floor(NoiseFloor::new(10.0));
    let mut solver = Solver::new(Method::GromacsSteep, ctrl, x.len());
    solver
        .set_distance_constraints(DistanceConstraints::new(vec![(0, 1)], vec![1.0]).unwrap())
        .unwrap();
    let report = solver.step(&objective, &mut x).unwrap();
    assert_eq!(report.status, Status::NoiseFloor);
    assert_eq!(report.steps, 0);
    assert_eq!(x.to_vec(), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn constrained_stop_norm_two_matches_steepest() {
    let objective = Oracle::unbounded(6, |v: ArrayView1<f64>| {
        (
            0.6 * (v[0] + v[3]),
            Array1::from_vec(vec![0.6, 0.0, 0.0, 0.6, 0.0, 0.0]),
        )
    });
    let set = DistanceConstraints::new(vec![(0, 1)], vec![1.0]).unwrap();
    let expected = StopNorm::Two.of(Array1::from_vec(vec![0.6, 0.0, 0.0, 0.6, 0.0, 0.0]).view());

    let mut x_steepest = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut steepest = Solver::new(Method::Steepest, control(0.1, 0.7, 3), x_steepest.len());
    steepest.set_stop_norm(StopNorm::Two);
    steepest.set_distance_constraints(set.clone()).unwrap();
    let rep_steepest = steepest.step(&objective, &mut x_steepest).unwrap();
    assert_eq!(
        (
            rep_steepest.status,
            rep_steepest.grad_norm,
            rep_steepest.steps
        ),
        (Status::Iterating, expected, 1)
    );
    assert_eq!(rep_steepest.grad_norm.to_bits(), expected.to_bits());

    let mut x_gromacs = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut gromacs = Solver::new(Method::GromacsSteep, control(0.1, 0.7, 3), x_gromacs.len());
    gromacs.set_stop_norm(StopNorm::Two);
    gromacs.set_distance_constraints(set).unwrap();
    let rep_gromacs = gromacs.step(&objective, &mut x_gromacs).unwrap();
    assert_eq!(
        (rep_gromacs.status, rep_gromacs.grad_norm, rep_gromacs.steps),
        (Status::Iterating, expected, 1)
    );
    assert_eq!(rep_gromacs.grad_norm.to_bits(), expected.to_bits());
    assert_eq!(rep_gromacs.stop_norm, StopNorm::Two);
}

#[test]
fn rejected_step_reports_configured_stationarity_norm() {
    let objective = Oracle::unbounded(6, |v: ArrayView1<f64>| {
        (
            -(v[0] + v[3]),
            Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        )
    });
    let mut x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsSteep, control(0.1, 0.0, 3), x.len());
    solver.set_stop_norm(StopNorm::Two);
    let report = solver.step(&objective, &mut x).unwrap();
    let expected = StopNorm::Two.of(Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).view());
    assert_eq!(report.status, Status::Iterating);
    assert_eq!(report.grad_norm.to_bits(), expected.to_bits());
    assert_eq!(x.to_vec(), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
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

struct PrintedStep {
    dmax: f64,
    dmax_text: String,
    epot_text: String,
    fmax_text: String,
}

struct ArgonReference {
    emstep: f64,
    sigma: f64,
    epsilon: f64,
    cutoff: f64,
    final_energy: String,
    final_fmax: String,
    steps: Vec<PrintedStep>,
    start: Vec<String>,
    end: Vec<String>,
}

fn load_argon() -> ArgonReference {
    let text = include_str!("data/gromacs_steep_argon.txt");
    let mut emstep = None;
    let mut sigma = None;
    let mut epsilon = None;
    let mut cutoff = None;
    let mut final_energy = None;
    let mut final_fmax = None;
    let mut section = "";
    let mut steps = Vec::new();
    let mut start = Vec::new();
    let mut end = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match line {
            "steps" => section = "steps",
            "start" => section = "start",
            "final" => section = "final",
            _ if section == "steps" => {
                let mut parts = line.split_whitespace();
                let _count = parts.next().unwrap();
                let dmax_text = parts.next().unwrap().to_string();
                let epot_text = parts.next().unwrap().to_string();
                let fmax_text = parts.next().unwrap().to_string();
                steps.push(PrintedStep {
                    dmax: dmax_text.parse().unwrap(),
                    dmax_text,
                    epot_text,
                    fmax_text,
                });
            }
            _ if section == "start" => start.extend(line.split_whitespace().map(str::to_string)),
            _ if section == "final" => end.extend(line.split_whitespace().map(str::to_string)),
            _ => {
                let (key, value) = line.split_once(' ').unwrap();
                match key {
                    "emstep" => emstep = Some(value.parse().unwrap()),
                    "sigma" => sigma = Some(value.parse().unwrap()),
                    "epsilon" => epsilon = Some(value.parse().unwrap()),
                    "cutoff" => cutoff = Some(value.parse().unwrap()),
                    "final_energy" => final_energy = Some(value.to_string()),
                    "final_fmax" => final_fmax = Some(value.to_string()),
                    other => panic!("unknown field {other}"),
                }
            }
        }
    }
    ArgonReference {
        emstep: emstep.unwrap(),
        sigma: sigma.unwrap(),
        epsilon: epsilon.unwrap(),
        cutoff: cutoff.unwrap(),
        final_energy: final_energy.unwrap(),
        final_fmax: final_fmax.unwrap(),
        steps,
        start,
        end,
    }
}

fn printed_half_ulp(token: &str) -> f64 {
    let (mantissa, exponent) = token.split_once(['e', 'E']).expect(token);
    let digits = mantissa
        .trim_start_matches(['+', '-'])
        .split_once('.')
        .map(|(_, fraction)| fraction.len())
        .unwrap_or(0);
    let exp: i32 = exponent.parse().unwrap();
    0.5 * 10f64.powi(exp - digits as i32)
}

fn matches_print(got: f64, token: &str) -> bool {
    let printed: f64 = token.parse().unwrap();
    (got - printed).abs() <= printed_half_ulp(token)
}

fn parse_coordinates(tokens: &[String]) -> Array1<f64> {
    Array1::from_vec(tokens.iter().map(|token| token.parse().unwrap()).collect())
}

fn lennard_jones(x: ArrayView1<f64>, sigma: f64, epsilon: f64, cutoff: f64) -> (f64, Array1<f64>) {
    let atoms = x.len() / 3;
    let mut energy = 0.0;
    let mut force = Array1::zeros(x.len());
    let sigma6 = sigma.powi(6);
    let c6 = 4.0 * epsilon * sigma6;
    let c12 = c6 * sigma6;
    let cutoff_sq = cutoff * cutoff;
    for i in 0..atoms {
        for j in (i + 1)..atoms {
            let mut delta = [0.0; 3];
            let mut r2 = 0.0;
            for k in 0..3 {
                delta[k] = x[3 * i + k] - x[3 * j + k];
                r2 += delta[k] * delta[k];
            }
            if r2 >= cutoff_sq {
                continue;
            }
            let r6 = r2 * r2 * r2;
            let r12 = r6 * r6;
            energy += c12 / r12 - c6 / r6;
            let coef = (12.0 * c12 / r12 - 6.0 * c6 / r6) / r2;
            for k in 0..3 {
                let component = coef * delta[k];
                force[3 * i + k] += component;
                force[3 * j + k] -= component;
            }
        }
    }
    (energy, force.mapv(|component: f64| -component))
}

fn next_dmax_grew(current: f64, next: f64) -> bool {
    let grew = (next - current * GROMACS_STEEP_GROW).abs();
    let shrank = (next - current * GROMACS_STEEP_SHRINK).abs();
    grew < shrank
}

/// Seven-atom argon cluster against `gmx_d` 2025.4 double precision.
///
/// The reference file is that run's verbose step lines, the
/// double-precision energy footer, and the g96 coordinates. Step 0 is
/// the initial evaluation. Each later `Solver::step` is one GROMACS
/// trial. Dmax, Epot and Fmax are compared inside half a unit of the
/// last digit `minimize.cpp` prints.
#[test]
fn argon_cluster_matches_gromacs_2025_step_sequence() {
    let reference = load_argon();
    assert!(reference.steps.len() > 2);
    let mut x = parse_coordinates(&reference.start);
    let sigma = reference.sigma;
    let epsilon = reference.epsilon;
    let cutoff = reference.cutoff;
    let objective = Oracle::unbounded(x.len(), move |point: ArrayView1<f64>| {
        lennard_jones(point, sigma, epsilon, cutoff)
    });
    let (energy, gradient) = lennard_jones(x.view(), sigma, epsilon, cutoff);
    let opening = &reference.steps[0];
    assert!(
        matches_print(energy, &opening.epot_text),
        "step 0 energy {energy} vs {}",
        opening.epot_text
    );
    assert!(
        matches_print(StopNorm::MaxAtom.of(gradient.view()), &opening.fmax_text),
        "step 0 fmax"
    );

    let mut solver = Solver::new(
        Method::GromacsSteep,
        control(reference.emstep, 0.0, reference.steps.len()),
        x.len(),
    );
    let mut best = energy;
    let mut current_gradient = gradient;
    let mut saw_refusal = false;
    for index in 1..reference.steps.len() {
        let printed = &reference.steps[index];
        let step_size = solver.gromacs_step_size().unwrap();
        assert!(
            matches_print(step_size, &printed.dmax_text),
            "step {index} Dmax {step_size} vs {}",
            printed.dmax_text
        );
        let force_norm = StopNorm::MaxAtom.of(current_gradient.view());
        let mut trial = x.clone();
        if force_norm.is_finite() && force_norm > 0.0 {
            for k in 0..trial.len() {
                trial[k] -= step_size * current_gradient[k] / force_norm;
            }
        }
        let (trial_energy, trial_gradient) = lennard_jones(trial.view(), sigma, epsilon, cutoff);
        assert!(
            matches_print(trial_energy, &printed.epot_text),
            "step {index} trial energy {trial_energy} vs {}",
            printed.epot_text
        );
        assert!(
            matches_print(
                StopNorm::MaxAtom.of(trial_gradient.view()),
                &printed.fmax_text
            ),
            "step {index} trial fmax"
        );
        let before = x.clone();
        let report = solver.step(&objective, &mut x).unwrap();
        assert_eq!(report.status, Status::Iterating);
        let accepted = if index + 1 < reference.steps.len() {
            next_dmax_grew(printed.dmax, reference.steps[index + 1].dmax)
        } else {
            report.value < best
        };
        if accepted {
            assert!(
                matches_print(report.value, &printed.epot_text),
                "step {index} energy {} vs {}",
                report.value,
                printed.epot_text
            );
            assert_eq!(report.value.to_bits(), trial_energy.to_bits());
            assert!(
                matches_print(report.grad_norm, &printed.fmax_text),
                "step {index} fmax {} vs {}",
                report.grad_norm,
                printed.fmax_text
            );
            assert_ne!(x, before);
            best = report.value;
            current_gradient = trial_gradient;
        } else {
            saw_refusal = true;
            assert_eq!(x, before);
            assert_eq!(report.value.to_bits(), best.to_bits());
            assert_eq!(
                report.grad_norm.to_bits(),
                StopNorm::MaxAtom.of(current_gradient.view()).to_bits()
            );
        }
    }
    assert!(saw_refusal);
    for (got, stored) in x.iter().zip(&reference.end) {
        assert_eq!(format!("{got:.9}"), *stored, "{got}");
    }
    assert!(
        matches_print(best, &reference.final_energy),
        "final energy {best} vs {}",
        reference.final_energy
    );
    let (_final_energy, final_gradient) = lennard_jones(x.view(), sigma, epsilon, cutoff);
    let expected_final_fmax: f64 = reference.final_fmax.parse().unwrap();
    let got_final_fmax = StopNorm::MaxAtom.of(final_gradient.view());
    let final_fmax_tolerance = 1.0e-10;
    assert!(
        (got_final_fmax - expected_final_fmax).abs() <= final_fmax_tolerance,
        "final fmax {got_final_fmax} vs {}",
        reference.final_fmax
    );
}

#[test]
fn huge_but_finite_gradients_do_not_trigger_machine_precision() {
    let huge = Oracle::unbounded(3, |_x: ArrayView1<f64>| {
        (0.0, Array1::from_vec(vec![1.0e200, 0.0, 0.0]))
    });
    let mut point = array![1.0, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsSteep, control(0.1, 0.0, 1), point.len());
    let report = solver.step(&huge, &mut point).unwrap();
    assert_eq!(report.status, Status::Iterating);
    assert_eq!(point[0].to_bits(), 1.0_f64.to_bits());
}
