mod common;

use approx::assert_abs_diff_eq;
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, DistanceConstraints, Error, GROMACS_LBFGS_NCORR_DEFAULT, GromacsLbfgsConfig, Method,
    Oracle, Solver, Status, StopNorm,
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

fn lennard_jones(positions: ArrayView1<'_, f64>, sigma: f64, epsilon: f64) -> (f64, Array1<f64>) {
    let c6 = 4.0 * epsilon * sigma.powi(6);
    let c12 = 4.0 * epsilon * sigma.powi(12);
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
    let objective = Oracle::unbounded(dim, move |x| lennard_jones(x, sigma, epsilon));
    let mut comparison_solver =
        Solver::new(Method::GromacsLbfgs, control(fixture.emstep, 1e-8, 12), dim);
    let mut comparison_point = Array1::from(fixture.initial_positions.clone());
    let comparison = comparison_solver
        .run(&objective, &mut comparison_point)
        .unwrap();
    eprintln!(
        "argon l-bfgs comparison: rgmin steps={} force_evaluations={} Epot={:.14e} Fmax={:.14e}; \
         gmx 2023.3 steps_reported=-1 accepted_step_lines=12 force_evaluations=13 \
         Epot=-1.2548940e1 Fmax=2.0878734e-4",
        comparison.steps, comparison.evals, comparison.value, comparison.grad_norm
    );
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
            lennard_jones(x, sigma, epsilon)
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
