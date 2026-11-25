mod common;

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU8, Ordering},
};

use approx::assert_abs_diff_eq;
use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, DistanceConstraints, Error, GROMACS_LBFGS_NCORR_DEFAULT, GromacsLbfgsConfig, Method,
    Oracle, Report, Solver, Status, StopNorm,
};

fn control(emstep: f64, emtol: f64, nsteps: usize) -> Control {
    Control::default()
        .with_istep(emstep)
        .with_gtol(emtol)
        .with_maxiter(nsteps)
        .with_maxmove(None)
}

#[test]
fn correction_memory_obeys_nbfgscorr() {
    let rosenbrock = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        let u = 1.0 - x[0];
        let v = x[1] - x[0] * x[0];
        (
            u * u + 100.0 * v * v + 0.5 * x[2] * x[2],
            array![-2.0 * u - 400.0 * x[0] * v, 200.0 * v, x[2]],
        )
    });
    let mut solver = Solver::new(Method::GromacsLbfgs, control(0.01, 0.0, 100), 3);
    solver
        .set_gromacs_lbfgs_config(GromacsLbfgsConfig::default().with_nbfgscorr(3))
        .unwrap();
    let mut point = array![-1.2, 1.0, 0.5];
    for _ in 0..12 {
        let report = solver.step(&rosenbrock, &mut point).unwrap();
        if report.status != Status::Iterating {
            break;
        }
    }
    assert_eq!(solver.gromacs_lbfgs_config().unwrap().nbfgscorr, 3);
    assert_eq!(solver.pair_count(), 3);
}

#[test]
fn rebase_and_external_moves_rebuild_the_direction_from_retained_pairs() {
    let bowl = Oracle::unbounded(3, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
    for mode in ["external", "rebase", "forget_evaluation"] {
        let mut solver = Solver::new(Method::GromacsLbfgs, control(1.0, 0.0, 10), 3);
        let mut point = array![2.0, 0.0, 0.0];
        solver.step(&bowl, &mut point).unwrap();
        assert_eq!(solver.pair_count(), 1);

        point = array![-2.0, 0.0, 0.0];
        match mode {
            "rebase" => {
                solver.rebase();
                assert_eq!(solver.pair_count(), 1);
            }
            "forget_evaluation" => {
                solver.forget_evaluation();
                assert_eq!(solver.pair_count(), 1);
            }
            _ => {}
        }
        let before = 0.5 * point.dot(&point);
        let report = solver.step(&bowl, &mut point).unwrap();
        assert!(
            report.value < before,
            "mode={mode}, report={report:?}, point={point:?}"
        );
        assert!(point[0] > -2.0, "mode={mode}");
    }
}

#[test]
fn unresolved_line_search_clears_memory_and_counts_the_reset_iteration() {
    let objective = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        if x[0] >= 0.75 {
            (0.5 * x[0] * x[0], array![x[0], 0.0, 0.0])
        } else if x[0] <= 0.25 {
            (0.5, array![-1.0, 0.0, 0.0])
        } else {
            (0.5, Array1::zeros(3))
        }
    });
    let mut solver = Solver::new(Method::GromacsLbfgs, control(1.0, 0.0, 10), 3);
    let mut point = array![2.0, 0.0, 0.0];
    let accepted = solver.step(&objective, &mut point).unwrap();
    assert_eq!(accepted.steps, 1);
    assert_eq!(solver.pair_count(), 1);

    let reset = solver.step(&objective, &mut point).unwrap();
    assert_eq!(reset.status, Status::Iterating);
    assert_eq!(reset.steps, 2);
    assert_eq!(solver.pair_count(), 0);
    assert_abs_diff_eq!(point[0], 1.0);
}

#[test]
fn the_reset_line_step_is_scaled_by_the_current_force_not_the_rejected_trial() {
    // The bracket ends at x0 = 0.5, where the force is zero. The restart
    // line step is 1 / |F| at x0 = 1, where the force is (-1, 0, 0).
    let objective = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        if x[0] >= 0.75 {
            (0.5 * x[0] * x[0], array![x[0], 0.0, 0.0])
        } else if x[0] <= 0.25 {
            (0.5, array![-1.0, 0.0, 0.0])
        } else {
            (0.5, Array1::zeros(3))
        }
    });
    let mut solver = Solver::new(Method::GromacsLbfgs, control(1.0, 0.0, 10), 3);
    let mut point = array![2.0, 0.0, 0.0];
    solver.step(&objective, &mut point).unwrap();
    let reset = solver.step(&objective, &mut point).unwrap();
    assert_eq!(reset.status, Status::Iterating);
    assert_eq!(solver.pair_count(), 0);
    assert_eq!(point, array![1.0, 0.0, 0.0]);
    assert_eq!(solver.gromacs_lbfgs_step_size(), Some(1.0));
}

#[cfg(feature = "highs")]
#[test]
fn a_session_box_clip_rebuilds_the_direction_at_the_clipped_start() {
    let bowl = Oracle::unbounded(3, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
    let run = |rebase: bool| {
        let mut solver = Solver::new(Method::GromacsLbfgs, control(10.0, 0.0, 10), 3);
        solver.set_highs(true);
        let mut point = array![2.0, 1.0, 0.0];
        solver.step(&bowl, &mut point).unwrap();
        assert_eq!(solver.pair_count(), 1);
        assert!(point[1] > 0.5, "{point:?}");
        assert!(solver.set_box(Some(vec![-10.0; 3]), Some(vec![10.0, 0.1, 10.0])));
        if rebase {
            solver.rebase();
        }
        solver.step(&bowl, &mut point).unwrap();
        point
    };
    let clipped = run(false);
    let rebased = run(true);
    assert_eq!(clipped, rebased);
    // On this bowl the stored pair has `s = y`, so the rebuilt direction
    // is the force at the clipped start and the step stays on its ray.
    // The stale direction from before the clip drives `x1` below zero.
    assert!(clipped[1] > 0.0, "{clipped:?}");
}

#[test]
fn machine_precision_stops_on_a_rounded_step_or_an_empty_memory_search() {
    fn run<O: DifferentiableObjective<f64>>(objective: &O, start: Array1<f64>) -> (Report, usize) {
        let mut solver = Solver::new(Method::GromacsLbfgs, control(1.0, 0.0, 10), 3);
        let mut point = start;
        let report = solver.run(objective, &mut point).unwrap();
        assert_eq!(report.status, Status::MachinePrecision, "{report:?}");
        assert_eq!(report.stop_reason(), rgmin::StopReason::MachinePrecision);
        assert!(!report.converged);
        assert_eq!(report.coords, point);
        (report, solver.pair_count())
    }

    // The first step lands one ulp of 2^20 from the minimum. The reset
    // line step 1 along the next direction, 2^-32, is below
    // GMX_REAL_EPS / rms(p_i / |x_i|) = sqrt(3), so no trial follows.
    let centre = 2.0_f64.powi(20);
    let ulp = 2.0_f64.powi(-32);
    let bowl = Oracle::unbounded(3, move |x: ArrayView1<f64>| {
        let d = x[0] - centre;
        (0.5 * d * d, array![d, 0.0, 0.0])
    });
    let (report, pairs) = run(&bowl, array![centre + 1.0 + ulp, 0.0, 0.0]);
    assert_eq!((report.steps, report.evals, pairs), (1, 2, 1));
    assert_eq!(report.coords, array![centre + ulp, 0.0, 0.0]);

    // Every trial on the force side rises, so the bracket spends all 20
    // trials with no stored pair to discard.
    let kink = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        if x[0] > 0.0 {
            (x[0], array![1.0, 0.0, 0.0])
        } else {
            (0.0, array![-1.0, 0.0, 0.0])
        }
    });
    let (report, pairs) = run(&kink, Array1::zeros(3));
    assert_eq!((report.steps, report.evals, pairs), (0, 2 + 20, 0));
    assert_eq!(report.coords, Array1::<f64>::zeros(3));

    // A flat energy ends the bracket after one trial with an energy change
    // below GMX_REAL_EPS.
    let flat = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        if x[0] > 0.0 {
            (1.0, array![1.0, 0.0, 0.0])
        } else {
            (1.0, array![-1.0, 0.0, 0.0])
        }
    });
    let (report, pairs) = run(&flat, Array1::zeros(3));
    assert_eq!((report.steps, report.evals, pairs), (0, 2 + 1, 0));
    assert_eq!(report.coords, Array1::<f64>::zeros(3));
}

#[test]
fn error_on_unmet_counts_a_machine_precision_stop_as_unmet() {
    assert!(rgmin::StopReason::MachinePrecision.is_unmet());
    let flat = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        if x[0] > 0.0 {
            (1.0, array![1.0, 0.0, 0.0])
        } else {
            (1.0, array![-1.0, 0.0, 0.0])
        }
    });
    let mut solver = Solver::new(
        Method::GromacsLbfgs,
        control(1.0, 0.0, 10).with_error_on_unmet(true),
        3,
    );
    let mut point = Array1::zeros(3);
    let error = solver.run(&flat, &mut point).unwrap_err();
    let Error::StopUnmet { reason, report } = error else {
        panic!("expected StopUnmet, got {error:?}");
    };
    assert_eq!(reason, rgmin::StopReason::MachinePrecision);
    assert_eq!(report.status, Status::MachinePrecision);
    assert_eq!(report.coords, point);
}

/// One GROMACS L-BFGS step from the origin on a line profile `t -> (value, slope)`
/// along `x[0]`; the opening force is 1, so trial alphas equal `t`.
fn first_step_line_trials(profile: fn(f64) -> (f64, f64)) -> (Vec<f64>, f64, f64) {
    let evaluated = std::sync::Mutex::new(Vec::new());
    let objective = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        evaluated.lock().unwrap().push(x[0]);
        let (value, slope) = profile(x[0]);
        (value, array![slope, 0.0, 0.0])
    });
    let mut solver = Solver::new(Method::GromacsLbfgs, control(1.0, 0.0, 10), 3);
    let mut point = array![0.0, 0.0, 0.0];
    let report = solver.step(&objective, &mut point).unwrap();
    assert_eq!(report.status, Status::Iterating);
    assert_eq!(solver.pair_count(), 1);
    let trials = evaluated.into_inner().unwrap();
    (trials, point[0], report.value)
}

#[test]
fn bracket_interpolation_keeps_the_opening_c_derivative_after_replacing_c() {
    let (trials, accepted, value) = first_step_line_trials(|t| {
        if t <= 0.3 {
            (-t + t * t, -1.0 + 2.0 * t)
        } else if t < 0.75 {
            (1.0 + 3.0 * (t - 0.5), 3.0)
        } else {
            (1.0 + (t - 1.0), 1.0)
        }
    });
    // C = 1 is rejected and B = 0.5 replaces C; GROMACS still interpolates
    // with gpa = -1 and gpc = +1, not the slope 3 at the new C.
    assert_eq!(trials, vec![0.0, 1.0, 0.5, 0.25]);
    assert_eq!(accepted, 0.25);
    assert_abs_diff_eq!(value, -0.1875);
}

#[test]
fn bracket_interpolation_keeps_the_opening_a_derivative_after_replacing_a() {
    let (trials, accepted, value) = first_step_line_trials(|t| {
        if t <= 0.25 {
            (-t, -1.0)
        } else if t < 0.625 {
            (2.0 - 3.0 * (t - 0.5), -3.0)
        } else if t <= 0.8 {
            (-1.0 + (t - 0.75), 1.0)
        } else {
            (1.0 + (t - 1.0), 1.0)
        }
    });
    // B = 0.5 replaces A; GROMACS still interpolates with gpa = -1, not the
    // slope -3 at the new A.
    assert_eq!(trials, vec![0.0, 1.0, 0.5, 0.75]);
    assert_eq!(accepted, 0.75);
    assert_abs_diff_eq!(value, -1.0);
}

#[test]
fn rosenbrock_converges_to_emtol_and_reports_steps_and_evaluations() {
    let rosenbrock = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        let u = 1.0 - x[0];
        let v = x[1] - x[0] * x[0];
        (
            u * u + 100.0 * v * v + 0.5 * x[2] * x[2],
            array![-2.0 * u - 400.0 * x[0] * v, 200.0 * v, x[2]],
        )
    });
    let mut solver = Solver::new(Method::GromacsLbfgs, control(0.01, 1e-6, 10_000), 3);
    let mut point = array![-1.2, 1.0, 0.5];
    let report = solver.run(&rosenbrock, &mut point).unwrap();
    assert_eq!(
        report.status,
        Status::Tolerance,
        "steps={}, evals={}, f={}, Fmax={}, point={point:?}",
        report.steps,
        report.evals,
        report.value,
        report.grad_norm
    );
    assert_eq!(report.stop_norm, StopNorm::MaxAtom);
    assert!(report.grad_norm < 1e-6, "Fmax = {}", report.grad_norm);
    assert!(report.steps > 0);
    assert_eq!(report.steps, solver.steps());
    assert_eq!(report.evals, solver.evals());
    assert!(report.evals > report.steps);
    assert_abs_diff_eq!(point[0], 1.0, epsilon = 1e-4);
    assert_abs_diff_eq!(point[1], 1.0, epsilon = 1e-4);
    assert_abs_diff_eq!(point[2], 0.0, epsilon = 1e-5);
}

struct ArgonFixture {
    sigma: f64,
    epsilon: f64,
    emstep: f64,
    initial_positions: Vec<f64>,
}

fn parse_argon_fixture() -> ArgonFixture {
    let text = include_str!("data/gromacs_steep_argon.txt");
    let mut sigma = None;
    let mut epsilon = None;
    let mut emstep = None;
    let mut initial_positions = Vec::new();
    let mut section = "";
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match line {
            "steps" => section = "steps",
            "start" => section = "start",
            "final" => section = "final",
            _ if section == "start" => {
                for token in line.split_whitespace() {
                    initial_positions.push(token.parse().expect("initial coordinate"));
                }
            }
            _ if section == "steps" || section == "final" => {}
            _ => {
                let (key, value) = line.split_once(' ').expect("key value");
                match key {
                    "emstep" => emstep = Some(value.parse().expect("emstep")),
                    "sigma" => sigma = Some(value.parse().expect("sigma")),
                    "epsilon" => epsilon = Some(value.parse().expect("epsilon")),
                    _ => {}
                }
            }
        }
    }
    ArgonFixture {
        sigma: sigma.expect("sigma"),
        epsilon: epsilon.expect("epsilon"),
        emstep: emstep.expect("emstep"),
        initial_positions,
    }
}

fn lennard_jones(
    positions: ArrayView1<'_, f64>,
    sigma: f64,
    epsilon: f64,
    cutoff: f64,
) -> (f64, Array1<f64>) {
    let c6 = 4.0 * epsilon * sigma.powi(6);
    let c12 = 4.0 * epsilon * sigma.powi(12);
    let cutoff_sq = cutoff * cutoff;
    let mut energy = 0.0;
    let mut gradient = Array1::zeros(positions.len());
    for i in 0..positions.len() / 3 {
        for j in (i + 1)..positions.len() / 3 {
            let delta = array![
                positions[3 * i] - positions[3 * j],
                positions[3 * i + 1] - positions[3 * j + 1],
                positions[3 * i + 2] - positions[3 * j + 2],
            ];
            let r2 = delta.dot(&delta);
            if r2 >= cutoff_sq {
                continue;
            }
            let inv_r2 = 1.0 / r2;
            let inv_r6 = inv_r2.powi(3);
            let inv_r12 = inv_r6 * inv_r6;
            energy += c12 * inv_r12 - c6 * inv_r6;
            let force_scale = (12.0 * c12 * inv_r12 - 6.0 * c6 * inv_r6) * inv_r2;
            for axis in 0..3 {
                let component = -force_scale * delta[axis];
                gradient[3 * i + axis] += component;
                gradient[3 * j + axis] -= component;
            }
        }
    }
    (energy, gradient)
}

#[test]
fn argon_cluster_reaches_emtol_with_default_correction_count() {
    let fixture = parse_argon_fixture();
    let dim = fixture.initial_positions.len();
    let sigma = fixture.sigma;
    let epsilon = fixture.epsilon;
    let objective = Oracle::unbounded(dim, move |x| {
        lennard_jones(x, sigma, epsilon, f64::INFINITY)
    });
    let mut comparison_solver =
        Solver::new(Method::GromacsLbfgs, control(fixture.emstep, 1e-8, 12), dim);
    let mut comparison_point = Array1::from(fixture.initial_positions.clone());
    let comparison = comparison_solver
        .run(&objective, &mut comparison_point)
        .unwrap();
    assert_eq!(comparison.steps, 12);
    assert_eq!(comparison.status, Status::Budget);
    let mut solver = Solver::new(
        Method::GromacsLbfgs,
        control(fixture.emstep, 1e-3, 100),
        dim,
    );
    assert_eq!(
        solver.gromacs_lbfgs_config().unwrap().nbfgscorr,
        GROMACS_LBFGS_NCORR_DEFAULT
    );
    let mut point = Array1::from(fixture.initial_positions);
    let report = solver.run(&objective, &mut point).unwrap();
    assert_eq!(report.status, Status::Tolerance);
    assert!(report.grad_norm < 1e-3, "Fmax = {}", report.grad_norm);
    assert_eq!(report.stop_norm, StopNorm::MaxAtom);
    assert!(solver.pair_count() <= GROMACS_LBFGS_NCORR_DEFAULT);
}

#[test]
fn zero_nbfgscorr_and_wrong_method_are_rejected() {
    let mut solver = Solver::new(Method::GromacsLbfgs, control(0.01, 1e-3, 10), 3);
    let error = solver
        .set_gromacs_lbfgs_config(GromacsLbfgsConfig::default().with_nbfgscorr(0))
        .unwrap_err();
    assert!(matches!(error, Error::GromacsLbfgs { .. }));

    let mut other = Solver::new(Method::GromacsCg, control(0.01, 1e-3, 10), 3);
    assert!(other.gromacs_lbfgs_config().is_none());
    assert!(
        other
            .set_gromacs_lbfgs_config(GromacsLbfgsConfig::default())
            .is_err()
    );
}

/// Force tolerance that keeps the cubic-lattice timing runs on a fixed budget.
const LATTICE_TIMING_FORCE_TOLERANCE: f64 = 1.0e-6;

/// Wall time and force evaluations of GROMACS L-BFGS against L-BFGS and
/// GROMACS CG. Ignored by the default suite; run with `--ignored`.
#[test]
#[ignore = "records wall time; run with --ignored --nocapture"]
fn record_iteration_time_and_force_calls() {
    let fixture = parse_argon_fixture();
    let sigma = fixture.sigma;
    let epsilon = fixture.epsilon;
    let start = Array1::from(fixture.initial_positions.clone());
    println!("case\tmethod\tsteps\tforce_evals\tseconds\tseconds_per_step");
    let summary = include_str!("data/gromacs_cg_reference/summary.tsv");
    for line in summary.lines().skip(1) {
        let mut columns = line.split('\t');
        let name = columns.next().unwrap();
        let _precision = columns.next().unwrap();
        let emtol: f64 = columns.next().unwrap().parse().unwrap();
        let nsteps: usize = columns.next().unwrap().parse().unwrap();
        for method in ["gromacs-lbfgs", "lbfgs", "gromacs-cg"] {
            let (steps, evals, seconds) = median_run(
                &start,
                sigma,
                epsilon,
                fixture.emstep,
                emtol,
                nsteps,
                method,
            );
            let per_step = if steps == 0 {
                0.0
            } else {
                seconds / steps as f64
            };
            println!("{name}\t{method}\t{steps}\t{evals}\t{seconds:.6e}\t{per_step:.6e}");
        }
    }
    for side in [2usize, 3, 4] {
        let count = side * side * side;
        let mut coords = Array1::zeros(count * 3);
        let spacing = sigma * 1.2;
        let mut cursor = 0;
        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    coords[cursor] = x as f64 * spacing;
                    coords[cursor + 1] = y as f64 * spacing;
                    coords[cursor + 2] = z as f64 * spacing;
                    cursor += 3;
                }
            }
        }
        let name = format!("lj-cubic-{count}");
        for method in ["gromacs-lbfgs", "lbfgs", "gromacs-cg"] {
            let (steps, evals, seconds) = median_run(
                &coords,
                sigma,
                epsilon,
                fixture.emstep,
                LATTICE_TIMING_FORCE_TOLERANCE,
                20,
                method,
            );
            let per_step = if steps == 0 {
                0.0
            } else {
                seconds / steps as f64
            };
            println!("{name}\t{method}\t{steps}\t{evals}\t{seconds:.6e}\t{per_step:.6e}");
        }
    }
}

fn median_run(
    start: &Array1<f64>,
    sigma: f64,
    epsilon: f64,
    emstep: f64,
    emtol: f64,
    nsteps: usize,
    method: &str,
) -> (usize, usize, f64) {
    let mut samples = Vec::with_capacity(3);
    let mut last = (0usize, 0usize);
    for _ in 0..3 {
        let dim = start.len();
        let objective = Oracle::unbounded(dim, move |x: ArrayView1<f64>| {
            lennard_jones(x, sigma, epsilon, f64::INFINITY)
        });
        let selected = match method {
            "gromacs-lbfgs" => Method::GromacsLbfgs,
            "lbfgs" => Method::lbfgs_with_memory(GROMACS_LBFGS_NCORR_DEFAULT),
            "gromacs-cg" => Method::GromacsCg,
            _ => unreachable!(),
        };
        let mut solver = Solver::new(selected, control(emstep, emtol, nsteps), dim);
        let mut point = start.clone();
        let began = std::time::Instant::now();
        let report = solver.run(&objective, &mut point).unwrap();
        let elapsed = began.elapsed().as_secs_f64();
        last = (report.steps, report.evals);
        samples.push(elapsed);
    }
    samples.sort_by(|left, right| left.total_cmp(right));
    (last.0, last.1, samples[1])
}

fn rosenbrock_value(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    let u = 1.0 - x[0];
    let v = x[1] - x[0] * x[0];
    (
        u * u + 100.0 * v * v + 0.5 * x[2] * x[2],
        array![-2.0 * u - 400.0 * x[0] * v, 200.0 * v, x[2]],
    )
}

struct CleanPair {
    s: Array1<f64>,
    y: Array1<f64>,
    rho: f64,
    diag: f64,
}

/// Two-loop over a deque that holds only the pairs stored since the last
/// clear. This is the direction `apply_pairs` builds, not the GROMACS ring
/// that keeps `point` across a reset.
fn clean_direction(force: &Array1<f64>, pairs: &[CleanPair]) -> Array1<f64> {
    let mut direction = force.clone();
    if pairs.is_empty() {
        return direction;
    }
    let mut alpha = Vec::with_capacity(pairs.len());
    for pair in pairs.iter().rev() {
        let coefficient = pair.rho * pair.s.dot(&direction);
        alpha.push(coefficient);
        direction.scaled_add(-coefficient, &pair.y);
    }
    direction *= pairs.last().expect("pairs").diag;
    for (pair, coefficient) in pairs.iter().zip(alpha.into_iter().rev()) {
        let beta = pair.rho * pair.y.dot(&direction);
        direction.scaled_add(coefficient - beta, &pair.s);
    }
    direction
}

fn same_ray(displacement: &Array1<f64>, direction: &Array1<f64>) {
    let disp_norm = displacement.dot(displacement).sqrt();
    let dir_norm = direction.dot(direction).sqrt();
    assert!(
        disp_norm > 0.0 && dir_norm > 0.0,
        "disp {displacement:?} dir {direction:?}"
    );
    let cosine = displacement.dot(direction) / (disp_norm * dir_norm);
    assert!(
        cosine > 1.0 - 1e-8,
        "cosine {cosine}, displacement {displacement:?}, direction {direction:?}"
    );
}

/// After a failed line search the correction deque is empty, and later
/// displacements follow a two-loop over only the pairs stored since that
/// clear. GROMACS leaves `point` unchanged and can reread older slots.
#[test]
fn line_search_reset_continues_on_a_clean_correction_ring() {
    let phase = Arc::new(AtomicU8::new(0));
    let trap = Arc::new(Mutex::new(Array1::zeros(3)));
    let objective = Oracle::unbounded(3, {
        let phase = Arc::clone(&phase);
        let trap = Arc::clone(&trap);
        move |x: ArrayView1<f64>| {
            if phase.load(Ordering::SeqCst) == 1 {
                let origin = trap.lock().expect("trap").clone();
                let (value, gradient) = rosenbrock_value(origin.view());
                if x.iter()
                    .zip(origin.iter())
                    .all(|(left, right)| left.to_bits() == right.to_bits())
                {
                    return (value, gradient);
                }
                return (value + 1.0, &x.to_owned() - &origin);
            }
            rosenbrock_value(x)
        }
    });
    let nbfgscorr = 2;
    let mut solver = Solver::new(Method::GromacsLbfgs, control(0.1, 0.0, 40), 3);
    solver
        .set_gromacs_lbfgs_config(GromacsLbfgsConfig::default().with_nbfgscorr(nbfgscorr))
        .unwrap();
    let mut point = array![-1.2, 1.0, 0.5];
    for _ in 0..4 {
        let report = solver.step(&objective, &mut point).unwrap();
        assert_eq!(report.status, Status::Iterating, "{report:?}");
    }
    assert_eq!(solver.pair_count(), nbfgscorr);

    *trap.lock().expect("trap") = point.clone();
    phase.store(1, Ordering::SeqCst);
    let trapped = point.clone();
    let reset = solver.step(&objective, &mut point).unwrap();
    assert_eq!(reset.status, Status::Iterating, "{reset:?}");
    assert_eq!(solver.pair_count(), 0);
    assert_eq!(point, trapped);
    phase.store(0, Ordering::SeqCst);

    let mut pairs = Vec::new();
    let mut gradient = reset.grad.expect("gradient");
    for step in 0..5 {
        let force = gradient.mapv(|component| -component);
        let predicted = clean_direction(&force, &pairs);
        let previous = point.clone();
        let previous_gradient = gradient.clone();
        let report = solver.step(&objective, &mut point).unwrap();
        assert_eq!(
            report.status,
            Status::Iterating,
            "post-reset step {step}: {report:?}"
        );
        same_ray(&(&point - &previous), &predicted);
        let displacement = &point - &previous;
        let gradient_change = &report.grad.clone().expect("gradient") - &previous_gradient;
        let curvature = displacement.dot(&gradient_change);
        let yy = gradient_change.dot(&gradient_change);
        assert!(curvature.abs() > 0.0 && yy > 0.0, "step {step}");
        if pairs.len() == nbfgscorr {
            pairs.remove(0);
        }
        pairs.push(CleanPair {
            s: displacement,
            y: gradient_change,
            rho: 1.0 / curvature,
            diag: curvature / yy,
        });
        assert_eq!(solver.pair_count(), pairs.len());
        gradient = report.grad.expect("gradient");
    }
    assert_eq!(pairs.len(), nbfgscorr);
}

/// A frozen atom still moves: the L-BFGS force keeps that atom's
/// component. GROMACS zeros the stored search direction there instead.
#[test]
fn frozen_atom_force_stays_in_the_lbfgs_step() {
    let start = array![0.0, 0.0, 0.0, 1.2, 0.2, -0.4];
    let target = array![0.4, -0.3, 0.2, 1.0, 0.8, 0.1];
    let harmonic = Oracle::unbounded(6, {
        let target = target.clone();
        move |x: ArrayView1<f64>| {
            let gradient = &x.to_owned() - &target;
            (0.5 * gradient.dot(&gradient), gradient)
        }
    });
    let zeroed_gradient = Oracle::unbounded(6, {
        let target = target.clone();
        move |x: ArrayView1<f64>| {
            let mut gradient = &x.to_owned() - &target;
            let value = 0.5 * gradient.dot(&gradient);
            gradient[0] = 0.0;
            gradient[1] = 0.0;
            gradient[2] = 0.0;
            (value, gradient)
        }
    });

    let mut frozen = Solver::new(Method::GromacsLbfgs, control(0.2, 0.0, 4), 6);
    frozen.set_frozen(&[1, 1, 1, 0, 0, 0]);
    let mut masked = start.clone();
    frozen.step(&harmonic, &mut masked).unwrap();
    // The mask does not zero the frozen atom's force, so atom 0 moves on
    // the first iteration. These coordinates pin that step.
    assert_abs_diff_eq!(masked[0], 0.067_377_981_474_241_66, epsilon = 1e-12);
    assert_abs_diff_eq!(masked[1], -0.050_533_486_105_681_25, epsilon = 1e-12);
    assert_abs_diff_eq!(masked[2], 0.033_688_990_737_120_83, epsilon = 1e-12);
    assert_abs_diff_eq!(masked[3], 1.160_559_230_356_541_5, epsilon = 1e-12);
    assert_abs_diff_eq!(masked[4], 0.193_426_538_392_756_92, epsilon = 1e-12);
    assert_abs_diff_eq!(masked[5], -0.386_853_076_785_513_84, epsilon = 1e-12);

    let mut cleared = Solver::new(Method::GromacsLbfgs, control(0.2, 0.0, 4), 6);
    let mut cleared_point = start.clone();
    cleared.step(&zeroed_gradient, &mut cleared_point).unwrap();
    assert_eq!(cleared_point[0], 0.0);
    assert!((masked[3] - cleared_point[3]).abs() > 1e-3);

    let mut raw = Solver::new(Method::GromacsLbfgs, control(0.2, 0.0, 4), 6);
    let mut raw_point = start;
    raw.step(&harmonic, &mut raw_point).unwrap();
    assert!((masked[0] - raw_point[0]).abs() > 1e-3);
    assert!((masked[3] - raw_point[3]).abs() > 1e-3);

    frozen.step(&harmonic, &mut masked).unwrap();
    cleared.step(&zeroed_gradient, &mut cleared_point).unwrap();
    assert!((masked[3] - cleared_point[3]).abs() > 1e-3);
}

struct LbfgsArgonRow {
    k: usize,
    evals: usize,
    energy: f64,
    fmax: f64,
}

struct LbfgsArgonCase {
    nbfgscorr: usize,
    emstep: f64,
    rows: Vec<LbfgsArgonRow>,
}

struct LbfgsArgonReference {
    sigma: f64,
    epsilon: f64,
    cutoff: f64,
    emtol: f64,
    cases: Vec<LbfgsArgonCase>,
}

fn load_lbfgs_argon() -> LbfgsArgonReference {
    let text = include_str!("data/gromacs_lbfgs_argon.txt");
    let mut sigma = None;
    let mut epsilon = None;
    let mut cutoff = None;
    let mut emtol = None;
    let mut cases: Vec<LbfgsArgonCase> = Vec::new();
    let mut rows: Option<Vec<LbfgsArgonRow>> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("case ") {
            if let Some(finished) = rows.take() {
                cases.last_mut().expect("case").rows = finished;
            }
            let mut parts = rest.split_whitespace();
            let nbfgscorr = parts.next().expect("nbfgscorr").parse().expect("nbfgscorr");
            let emstep = parts.next().expect("emstep").parse().expect("emstep");
            let _nsteps: usize = parts.next().expect("nsteps").parse().expect("nsteps");
            cases.push(LbfgsArgonCase {
                nbfgscorr,
                emstep,
                rows: Vec::new(),
            });
            rows = Some(Vec::new());
            continue;
        }
        if rows.is_some() {
            let mut parts = line.split_whitespace();
            let k = parts.next().expect("k").parse().expect("k");
            let evals = parts.next().expect("evals").parse().expect("evals");
            let energy = parts.next().expect("energy").parse().expect("energy");
            let fmax = parts.next().expect("fmax").parse().expect("fmax");
            rows.as_mut().expect("rows").push(LbfgsArgonRow {
                k,
                evals,
                energy,
                fmax,
            });
            continue;
        }
        let (key, value) = line.split_once(' ').expect("field");
        match key {
            "sigma" => sigma = Some(value.parse().expect("sigma")),
            "epsilon" => epsilon = Some(value.parse().expect("epsilon")),
            "cutoff" => cutoff = Some(value.parse().expect("cutoff")),
            "emtol" => emtol = Some(value.parse().expect("emtol")),
            other => panic!("unknown field {other}"),
        }
    }
    if let Some(finished) = rows.take() {
        cases.last_mut().expect("case").rows = finished;
    }
    LbfgsArgonReference {
        sigma: sigma.expect("sigma"),
        epsilon: epsilon.expect("epsilon"),
        cutoff: cutoff.expect("cutoff"),
        emtol: emtol.expect("emtol"),
        cases,
    }
}

/// Seven-atom argon against `gmx_d` 2023.3 double precision.
///
/// `tests/data/gromacs_lbfgs_argon.txt` is that build. Row `k` is GROMACS
/// `nsteps = k`, which runs the inclusive iterations `step = 0..=k`.
/// rgmin's step `k + 1` matches the evaluation count and stays within
/// `1e-11` of the footer energy. The maximum atomic force is compared
/// to `1e-6`: near the minimum a last-bit coordinate difference moves
/// `Fmax` by more than one unit in the last place of the footer.
#[test]
fn argon_matches_double_precision_gmx_lbfgs() {
    let reference = load_lbfgs_argon();
    assert!(reference.cases.len() >= 4);
    let start = Array1::from(parse_argon_fixture().initial_positions);
    let sigma = reference.sigma;
    let epsilon = reference.epsilon;
    let cutoff = reference.cutoff;
    for case in &reference.cases {
        let objective = Oracle::unbounded(start.len(), move |x: ArrayView1<f64>| {
            lennard_jones(x, sigma, epsilon, cutoff)
        });
        let mut solver = Solver::new(
            Method::GromacsLbfgs,
            control(case.emstep, reference.emtol, case.rows.len()),
            start.len(),
        );
        solver
            .set_gromacs_lbfgs_config(GromacsLbfgsConfig::default().with_nbfgscorr(case.nbfgscorr))
            .unwrap();
        let mut point = start.clone();
        for row in &case.rows {
            let report = solver.step(&objective, &mut point).unwrap();
            assert_eq!(
                report.status,
                Status::Iterating,
                "nbfgscorr={} emstep={} k={}",
                case.nbfgscorr,
                case.emstep,
                row.k
            );
            assert_eq!(report.steps, row.k + 1);
            assert_eq!(
                report.evals, row.evals,
                "nbfgscorr={} emstep={} k={}",
                case.nbfgscorr, case.emstep, row.k
            );
            assert!(
                (report.value - row.energy).abs() <= 1e-11,
                "nbfgscorr={} emstep={} k={} energy {} vs {}",
                case.nbfgscorr,
                case.emstep,
                row.k,
                report.value,
                row.energy
            );
            assert!(
                (report.grad_norm - row.fmax).abs() <= 1e-6,
                "nbfgscorr={} emstep={} k={} Fmax {} vs {}",
                case.nbfgscorr,
                case.emstep,
                row.k,
                report.grad_norm,
                row.fmax
            );
        }
    }
}

#[test]
fn holonomic_constraints_are_refused() {
    let constraints = DistanceConstraints::new(vec![(0, 1)], vec![1.0]).unwrap();
    let mut solver = Solver::new(Method::GromacsLbfgs, control(0.01, 1e-3, 10), 6);
    let err = solver.set_distance_constraints(constraints).unwrap_err();
    assert!(
        err.to_string().contains("GROMACS L-BFGS"),
        "unexpected error: {err}"
    );
}
