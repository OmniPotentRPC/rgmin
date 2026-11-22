mod common;

use approx::assert_abs_diff_eq;
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Accept, ConstraintError, ConstraintSolver, Control, DistanceConstraints, Error,
    GROMACS_CG_NSTCGSTEEP_DEFAULT, GromacsCgConfig, LineSearch, ManifoldKind, Method, NoiseFloor,
    Oracle, Solver, Status, StopNorm, minimize_method,
};

fn control(h: f64, gtol: f64, maxiter: usize) -> Control {
    Control::default()
        .with_istep(h)
        .with_gtol(gtol)
        .with_maxiter(maxiter)
        .with_maxmove(None)
}

fn ellipse() -> Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    Oracle::unbounded(3, |x: ArrayView1<f64>| {
        let energy = x[0] * x[0] + 5.0 * x[1] * x[1];
        let grad = array![2.0 * x[0], 10.0 * x[1], 0.0];
        (energy, grad)
    })
}

fn exact_cg_solver(nstcgsteep: u32) -> Solver {
    let mut solver = Solver::new(Method::GromacsCg, control(0.1, 1e-10, 10), 3);
    solver.set_linesearch(LineSearch::brent(80, 1e-15));
    solver
        .set_gromacs_cg_config(GromacsCgConfig::default().with_nstcgsteep(nstcgsteep))
        .unwrap();
    assert_eq!(solver.gromacs_cg_config().unwrap().nstcgsteep, nstcgsteep);
    solver
}

// f(x, y, z) = x^2 + 5 y^2 from (1, 1, 0) with an exact line search.
// Step 1 is steepest descent along -g0 = [-2, -10, 0] with alpha = 13/126,
// landing at x1 = [50/63, -2/63, 0]. GROMACS `do_cg` tests
// `nstcg && (step % nstcg) == 0` after step 0, so any positive nstcgsteep
// makes step 2 steepest descent as well: along -g1 = [-100/63, 20/63, 0]
// with alpha = 13/30 it lands at x2 = [20/189, 20/189, 0].
const X1: [f64; 2] = [50.0 / 63.0, -2.0 / 63.0];
const X2: [f64; 2] = [20.0 / 189.0, 20.0 / 189.0];

#[test]
fn quadratic_2d_restarts_after_step_zero_then_follows_polak_ribiere() {
    let objective = ellipse();
    let mut x = array![1.0, 1.0, 0.0];
    let mut solver = exact_cg_solver(GROMACS_CG_NSTCGSTEEP_DEFAULT);

    let step1 = solver.step(&objective, &mut x).unwrap();
    assert_eq!(step1.steps, 1);
    assert_eq!(step1.status, Status::Iterating);
    assert_abs_diff_eq!(x[0], X1[0], epsilon = 1e-9);
    assert_abs_diff_eq!(x[1], X1[1], epsilon = 1e-9);
    assert_abs_diff_eq!(x[2], 0.0, epsilon = 1e-15);

    let step2 = solver.step(&objective, &mut x).unwrap();
    assert_eq!(step2.status, Status::Iterating);
    assert_abs_diff_eq!(x[0], X2[0], epsilon = 1e-8);
    assert_abs_diff_eq!(x[1], X2[1], epsilon = 1e-8);

    // Step 3 is the first Polak-Ribiere step: two conjugate directions from
    // x1 reach the minimum of a two-dimensional quadratic.
    let step3 = solver.step(&objective, &mut x).unwrap();
    assert_eq!(step3.steps, 3);
    assert_eq!(step3.status, Status::Tolerance);
    assert_abs_diff_eq!(x[0], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(x[1], 0.0, epsilon = 1e-7);
}

#[test]
fn nstcgsteep_zero_follows_the_exact_polak_ribiere_path() {
    // Without periodic restarts, step 2 uses beta1 = 100/3969 and lands on
    // the minimum.
    let objective = ellipse();
    let mut x = array![1.0, 1.0, 0.0];
    let mut solver = exact_cg_solver(0);

    solver.step(&objective, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], X1[0], epsilon = 1e-9);
    assert_abs_diff_eq!(x[1], X1[1], epsilon = 1e-9);

    let step2 = solver.step(&objective, &mut x).unwrap();
    assert_eq!(step2.steps, 2);
    assert_eq!(step2.status, Status::Tolerance);
    assert_abs_diff_eq!(x[0], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(x[1], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(step2.value, 0.0, epsilon = 1e-13);
}

#[test]
fn nstcgsteep_one_restarts_every_step_to_steepest_descent() {
    let objective = ellipse();
    let mut x = array![1.0, 1.0, 0.0];
    let mut solver = exact_cg_solver(1);

    solver.step(&objective, &mut x).unwrap();
    solver.step(&objective, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], X2[0], epsilon = 1e-8);
    assert_abs_diff_eq!(x[1], X2[1], epsilon = 1e-8);

    // Step 3 is steepest descent again: along -g2 = -(40/189) [1, 5, 0]
    // with alpha = 13/126 it lands at (20/189) [100/126, -4/126, 0].
    let step3 = solver.step(&objective, &mut x).unwrap();
    assert_eq!(step3.status, Status::Iterating);
    assert_abs_diff_eq!(x[0], (20.0 / 189.0) * (100.0 / 126.0), epsilon = 1e-8);
    assert_abs_diff_eq!(x[1], (20.0 / 189.0) * (-4.0 / 126.0), epsilon = 1e-8);
}

#[test]
fn forget_restarts_gromacs_cg_as_a_fresh_session() {
    let objective = ellipse();
    let mut x = array![1.0, 1.0, 0.0];
    let mut used = exact_cg_solver(3);
    used.step(&objective, &mut x).unwrap();
    used.step(&objective, &mut x).unwrap();
    used.forget();

    let mut y = x.clone();
    let mut fresh = exact_cg_solver(3);
    for _ in 0..3 {
        let a = used.step(&objective, &mut x).unwrap();
        let b = fresh.step(&objective, &mut y).unwrap();
        assert_eq!(a.status, b.status);
        assert_eq!(x, y);
        if b.status != Status::Iterating {
            break;
        }
    }
}

#[test]
fn rosenbrock_converges_under_gromacs_cg_session_and_minimize_method() {
    let rosenbrock = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        let u = 1.0 - x[0];
        let v = x[1] - x[0] * x[0];
        let f = u * u + 100.0 * v * v + 0.5 * x[2] * x[2];
        let gx = -2.0 * u - 400.0 * x[0] * v;
        let gy = 200.0 * v;
        let gz = x[2];
        (f, array![gx, gy, gz])
    });

    let start = array![-1.2, 1.0, 0.5];
    let ctrl = control(0.01, 1e-6, 500);
    let report = minimize_method(
        &rosenbrock,
        start.clone(),
        &ctrl,
        Method::GromacsCg,
        LineSearch::wolfe(1e-4, 0.1, 20),
    )
    .unwrap();
    assert_eq!(report.status, Status::Tolerance);
    assert_eq!(report.stop_norm, StopNorm::MaxAtom);
    assert!(report.grad_norm < 1e-6, "grad_norm = {}", report.grad_norm);
    assert_abs_diff_eq!(report.coords[0], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(report.coords[1], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(report.coords[2], 0.0, epsilon = 1e-6);
}

#[test]
fn gromacs_cg_refuses_holonomic_constraints_with_a_clear_error() {
    let mut solver = Solver::new(Method::GromacsCg, control(0.01, 1e-4, 10), 6);
    assert_eq!(
        solver.gromacs_cg_config().unwrap().nstcgsteep,
        GROMACS_CG_NSTCGSTEEP_DEFAULT
    );
    let constraints = DistanceConstraints::new(vec![(0, 1)], vec![1.0])
        .unwrap()
        .with_solver(ConstraintSolver::Lincs);
    let err = solver.set_distance_constraints(constraints).unwrap_err();
    assert!(
        matches!(
            err,
            Error::Constraint(ConstraintError::Unsupported { what })
                if what.contains("GROMACS conjugate gradient")
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn gromacs_cg_config_accepts_zero_period_and_rejects_other_methods() {
    let mut cg = Solver::new(Method::GromacsCg, control(0.01, 1e-4, 10), 3);
    cg.set_gromacs_cg_config(GromacsCgConfig::default().with_nstcgsteep(0))
        .unwrap();
    assert_eq!(cg.gromacs_cg_config().unwrap().nstcgsteep, 0);

    let mut steep = Solver::new(Method::GromacsSteep, control(0.01, 1e-4, 10), 3);
    assert!(steep.gromacs_cg_config().is_none());
    let err = steep
        .set_gromacs_cg_config(GromacsCgConfig::default())
        .unwrap_err();
    assert!(matches!(err, Error::GromacsCg { .. }), "{err:?}");
}

#[test]
fn gromacs_cg_stops_at_machine_precision_when_step_rounds_away_or_energy_is_flat() {
    let linear_slope = Oracle::unbounded(3, |x: ArrayView1<f64>| (x[0], array![1.0, 0.0, 0.0]));
    let mut x = array![1.0, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsCg, control(f64::MIN_POSITIVE, 0.0, 5), 3);
    let report = solver.step(&linear_slope, &mut x).unwrap();
    assert_eq!(report.status, Status::MachinePrecision);
    assert_eq!(x[0], 1.0);

    let flat_energy = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        if x[0] == 0.0 {
            (1.0, array![-1.0, 0.0, 0.0])
        } else {
            (1.0, array![1.0, 0.0, 0.0])
        }
    });
    let mut x_flat = array![0.0, 0.0, 0.0];
    let mut flat_solver = Solver::new(Method::GromacsCg, control(0.01, 0.0, 5), 3);
    let flat_report = flat_solver.step(&flat_energy, &mut x_flat).unwrap();
    assert_eq!(flat_report.status, Status::MachinePrecision);
    assert_eq!(flat_report.steps, 1);
}

#[test]
fn gromacs_cg_handles_huge_finite_gradients_and_non_finite_trials() {
    let huge = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        if x[0] == 1.0 {
            (1.0, array![1.0e200, 0.0, 0.0])
        } else {
            (0.0, array![0.0, 0.0, 0.0])
        }
    });
    let mut x = array![1.0, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsCg, control(0.1, 1.0, 5), 3);
    let report = solver.step(&huge, &mut x).unwrap();
    assert_eq!(report.status, Status::Tolerance);
    assert_abs_diff_eq!(x[0], 0.9, epsilon = 1e-15);

    let start = array![0.0, 0.0, 0.0];
    let start_check = start.clone();
    let bad_trial = Oracle::unbounded(3, move |x: ArrayView1<f64>| {
        if x == start_check.view() {
            (1.0, array![1.0, 0.0, 0.0])
        } else {
            (f64::NAN, array![0.0, 0.0, 0.0])
        }
    });
    let mut x_bad = start;
    let mut bad_solver = Solver::new(Method::GromacsCg, control(0.1, 0.0, 5), 3);
    let err = bad_solver.step(&bad_trial, &mut x_bad).unwrap_err();
    assert!(matches!(err, Error::Oracle { .. }), "{err:?}");
}

#[test]
fn gromacs_cg_respects_noise_floor_and_stop_norm_override() {
    let bowl = Oracle::unbounded(6, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
    let mut x = array![0.04, 0.0, 0.0, 0.04, 0.0, 0.0];
    let ctrl = control(0.01, 0.05, 5);
    let mut max_atom_solver = Solver::new(Method::GromacsCg, ctrl.clone(), 6);
    assert_eq!(max_atom_solver.stop_norm(), StopNorm::MaxAtom);
    let r1 = max_atom_solver.step(&bowl, &mut x).unwrap();
    assert_eq!(r1.status, Status::Tolerance);

    let mut x2 = array![0.04, 0.0, 0.0, 0.04, 0.0, 0.0];
    let mut two_norm_solver = Solver::new(Method::GromacsCg, ctrl, 6);
    two_norm_solver.set_stop_norm(StopNorm::Two);
    let r2 = two_norm_solver.step(&bowl, &mut x2).unwrap();
    assert_eq!(r2.stop_norm, StopNorm::Two);

    let noisy = Oracle::unbounded(3, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
    let noisy_ctrl = control(0.01, 1e-12, 5).with_noise_floor(NoiseFloor::new(0.1));
    let mut x_noisy = array![0.05, 0.0, 0.0];
    let mut noisy_solver = Solver::new(Method::GromacsCg, noisy_ctrl, 3);
    let r_noisy = noisy_solver.step(&noisy, &mut x_noisy).unwrap();
    assert_eq!(r_noisy.status, Status::NoiseFloor);
}

fn ill_conditioned() -> Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    Oracle::unbounded(3, |x: ArrayView1<f64>| {
        let energy = 0.5 * (x[0] * x[0] + 100.0 * x[1] * x[1]);
        (energy, array![x[0], 100.0 * x[1], 0.0])
    })
}

// A steepest-descent step of a quadratic with exact line search leaves
// g1 = g0 - alpha A g0. On diag(1, 100) from (1, 1e-3) the Polak-Ribiere
// factor |g1|^2 / |g0|^2 is about 24, well above the cap of 5.
#[test]
fn a_beta_above_the_cap_restarts_on_steepest_descent() {
    let objective = ill_conditioned();
    let start = array![1.0, 1.0e-3, 0.0];
    let mut x = start.clone();
    let mut solver = Solver::new(Method::GromacsCg, control(0.1, 1e-14, 10), 3);
    solver.set_linesearch(LineSearch::brent(80, 1e-15));
    solver
        .set_gromacs_cg_config(GromacsCgConfig::default().with_nstcgsteep(0))
        .unwrap();

    solver.step(&objective, &mut x).unwrap();
    let x1 = x.clone();
    let g0 = array![start[0], 100.0 * start[1]];
    let h_g0 = array![g0[0], 100.0 * g0[1]];
    let alpha = g0.dot(&g0) / g0.dot(&h_g0);
    let g1 = &g0 - &(alpha * &h_g0);
    let beta = g1.dot(&(&g1 - &g0)) / g0.dot(&g0);
    assert!(beta > 20.0, "scenario needs a large beta, got {beta}");

    let report = solver.step(&objective, &mut x).unwrap();
    assert_eq!(report.status, Status::Iterating);
    let h_g1 = array![g1[0], 100.0 * g1[1]];
    let alpha1 = g1.dot(&g1) / g1.dot(&h_g1);
    let expected = array![x1[0] - alpha1 * g1[0], x1[1] - alpha1 * g1[1]];
    assert_abs_diff_eq!(x[0], expected[0], epsilon = 1e-9);
    assert_abs_diff_eq!(x[1], expected[1], epsilon = 1e-9);
    assert!(x[0].abs() > 1e-3, "an uncapped step reaches the minimum");
}

// A first step of 1.5 along -g0 = -1 overshoots the minimum of x^2 / 2.
// Then g1 = -0.5, beta = 0.75 and the Polak-Ribiere direction
// -g1 + beta d0 = -0.25 has g1 . d = 0.125 > 0, so the session takes
// steepest descent and counts a fallback when it forms that direction at
// the end of step 1.
#[test]
fn a_conjugate_direction_that_does_not_descend_falls_back_to_steepest_descent() {
    let bowl = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        (0.5 * x[0] * x[0], array![x[0], 0.0, 0.0])
    });
    let mut x = array![1.0, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsCg, control(1.5, 1e-14, 10), 3);
    solver.set_linesearch(LineSearch::backtracking(1e-4, 0.5, 20));
    solver
        .set_gromacs_cg_config(GromacsCgConfig::default().with_nstcgsteep(0))
        .unwrap();

    solver.step(&bowl, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], -0.5, epsilon = 1e-15);
    assert_eq!(solver.nlcg_descent_fallbacks(), 1);

    let report = solver.step(&bowl, &mut x).unwrap();
    assert!(x[0] > -0.5, "step 2 followed the ascent direction: {x:?}");
    assert!(report.value < 0.125);
}

#[test]
fn accept_step_takes_the_session_step_length_not_a_unit_step() {
    let bowl = Oracle::unbounded(3, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
    let mut x = array![5.0, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsCg, control(0.1, 1e-14, 10), 3);
    solver.set_accept(Accept::Step);
    solver.step(&bowl, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], 4.9, epsilon = 1e-12);
    assert_abs_diff_eq!(x[1], 0.0, epsilon = 1e-15);
}

#[test]
fn accept_step_follows_unscaled_polak_ribiere_direction() {
    let quad = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        let energy = 0.5 * (x[0] * x[0] + 400.0 * x[1] * x[1]);
        (energy, array![x[0], 400.0 * x[1], 0.0])
    });
    let istep = 0.01_f64;
    let mut x = array![1.0_f64, 0.005, 0.0];
    let mut solver = Solver::new(Method::GromacsCg, control(istep, 1e-14, 10), 3);
    solver
        .set_gromacs_cg_config(GromacsCgConfig::default().with_nstcgsteep(0))
        .unwrap();
    solver.set_accept(Accept::Step);

    let g0 = array![1.0_f64, 2.0, 0.0];
    let d0 = -&g0;
    let u0 = &d0 / d0.dot(&d0).sqrt();

    solver.step(&quad, &mut x).unwrap();
    let x1 = array![1.0_f64, 0.005, 0.0] + istep * &u0;
    assert_abs_diff_eq!(x[0], x1[0], epsilon = 1e-12);
    assert_abs_diff_eq!(x[1], x1[1], epsilon = 1e-12);

    let g1 = array![x1[0], 400.0 * x1[1], 0.0];
    let beta = g1.dot(&(&g1 - &g0)) / g0.dot(&g0);
    let d1 = -&g1 + beta * &d0;
    let u1 = &d1 / d1.dot(&d1).sqrt();
    let x2 = &x1 + &(istep * &u1);

    solver.step(&quad, &mut x).unwrap();
    assert_abs_diff_eq!(x[0], x2[0], epsilon = 1e-12);
    assert_abs_diff_eq!(x[1], x2[1], epsilon = 1e-12);
    assert_abs_diff_eq!(x[2], 0.0, epsilon = 1e-15);
}

#[test]
fn a_curved_manifold_takes_the_opening_step_length_not_a_unit_step() {
    let height = Oracle::unbounded(3, |x: ArrayView1<f64>| (x[2], array![0.0, 0.0, 1.0]));
    let mut x = array![1.0, 0.0, 0.0];
    let mut solver = Solver::new(Method::GromacsCg, control(0.1, 1e-14, 10), 3);
    solver.set_manifold(ManifoldKind::Sphere);
    solver.step(&height, &mut x).unwrap();
    assert_abs_diff_eq!(x.dot(&x), 1.0, epsilon = 1e-12);
    // The retraction of a tangent step of length 0.1 turns the point by
    // atan(0.1) = 0.0997 rad. A unit step would turn it by pi / 4.
    let angle = x[0].clamp(-1.0, 1.0).acos();
    assert_abs_diff_eq!(angle, 0.1_f64.atan(), epsilon = 1e-9);
}

struct Box3 {
    bounds: Bounds<f64>,
}

impl Box3 {
    fn new(lower: Array1<f64>, upper: Array1<f64>) -> Self {
        Self {
            bounds: Bounds::new(lower, upper, 0.0),
        }
    }
}

// E = x / 2 + 15 y^2 - y has g = (0.5, 30 y - 1, 0). The box is x >= 0,
// y >= 0, z free, and the minimum over it is (0, 1/30, 0).
fn box_energy(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    (
        0.5 * x[0] + 15.0 * x[1] * x[1] - x[1],
        array![0.5, 30.0 * x[1] - 1.0, 0.0],
    )
}

impl Objective<f64> for Box3 {
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        box_energy(x).0
    }
}

impl Gradient<f64> for Box3 {
    fn dim(&self) -> usize {
        3
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        box_energy(x).1
    }
}

impl DifferentiableObjective<f64> for Box3 {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        box_energy(x)
    }
}

// From (0, 0.1, 0) the first backtracking trial that satisfies Armijo
// leaves the box through both lower faces, and the clip puts the point on
// the corner (0, 0, 0). There g1 = (0.5, -1, 0) and beta = 0.71, so the
// conjugate direction -g1 + beta d0 = (-0.85, -0.41, 0) still descends
// (g1 . d = -0.01) but points out of the box in both free coordinates, and
// its opening trial clips back onto the start. Steepest descent
// -g1 = (-0.5, 1, 0) moves along y.
#[test]
fn a_conjugate_trial_clipped_to_the_start_retries_steepest_descent() {
    let objective = Box3::new(array![0.0, 0.0, -1e12], array![1e12, 1e12, 1e12]);
    let mut x = array![0.0, 0.1, 0.0];
    let mut solver = Solver::new(Method::GromacsCg, control(1.0, 1e-14, 10), 3);
    solver.set_linesearch(LineSearch::backtracking(1e-4, 0.5, 30));
    solver
        .set_gromacs_cg_config(GromacsCgConfig::default().with_nstcgsteep(0))
        .unwrap();

    solver.step(&objective, &mut x).unwrap();
    assert_eq!(x.to_vec(), vec![0.0, 0.0, 0.0]);

    assert_eq!(solver.nlcg_descent_fallbacks(), 0);
    let report = solver.step(&objective, &mut x).unwrap();
    assert_ne!(report.status, Status::MachinePrecision);
    assert_eq!(x[0], 0.0);
    assert!(
        x[1] > 0.0,
        "the steepest-descent retry did not move y: {x:?}"
    );
    assert!(report.value < 0.0);
}

#[test]
fn a_bounded_minimum_on_the_box_edge_is_reached_inside_the_box() {
    let objective = Box3::new(array![0.0, 0.0, -1e12], array![1e12, 1e12, 1e12]);
    let mut x = array![0.05, 0.05, 0.0];
    let mut solver = Solver::new(Method::GromacsCg, control(0.01, 1e-9, 200), 3);
    solver.set_linesearch(LineSearch::backtracking(1e-4, 0.5, 30));
    let report = solver.run(&objective, &mut x).unwrap();
    assert!(x[0] >= 0.0 && x[1] >= 0.0, "left the box: {x:?}");
    assert_eq!(x[0], 0.0);
    assert!(report.value < 0.0);
    assert_abs_diff_eq!(x[1], 1.0 / 30.0, epsilon = 1e-4);
}

struct ArgonFixture {
    sigma: f64,
    epsilon: f64,
    emstep: f64,
    nsteps: usize,
    initial_positions: Vec<f64>,
}

fn parse_argon_fixture() -> ArgonFixture {
    let text = include_str!("data/gromacs_steep_argon.txt");
    let mut sigma = None;
    let mut epsilon = None;
    let mut emstep = None;
    let mut nsteps = 0usize;
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
            _ if section == "steps" => {
                let step_index: usize = line
                    .split_whitespace()
                    .next()
                    .expect("step index")
                    .parse()
                    .expect("step index usize");
                nsteps = step_index;
            }
            _ if section == "start" => {
                for token in line.split_whitespace() {
                    initial_positions.push(token.parse().expect("initial coordinate"));
                }
            }
            _ if section == "final" => {}
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
        nsteps,
        initial_positions,
    }
}

fn lennard_jones_energy_and_gradient(
    positions: ArrayView1<'_, f64>,
    sigma: f64,
    epsilon: f64,
) -> (f64, Array1<f64>) {
    let c6 = 4.0 * epsilon * sigma.powi(6);
    let c12 = 4.0 * epsilon * sigma.powi(12);
    let atoms = positions.len() / 3;
    let mut energy = 0.0;
    let mut gradient = Array1::zeros(positions.len());
    for i in 0..atoms {
        for j in (i + 1)..atoms {
            let dx = positions[3 * i] - positions[3 * j];
            let dy = positions[3 * i + 1] - positions[3 * j + 1];
            let dz = positions[3 * i + 2] - positions[3 * j + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let inv_r2 = 1.0 / r2;
            let inv_r6 = inv_r2 * inv_r2 * inv_r2;
            let inv_r12 = inv_r6 * inv_r6;
            energy += c12 * inv_r12 - c6 * inv_r6;
            let f_scalar = (12.0 * c12 * inv_r12 - 6.0 * c6 * inv_r6) * inv_r2;
            let gx = -f_scalar * dx;
            let gy = -f_scalar * dy;
            let gz = -f_scalar * dz;
            gradient[3 * i] += gx;
            gradient[3 * i + 1] += gy;
            gradient[3 * i + 2] += gz;
            gradient[3 * j] -= gx;
            gradient[3 * j + 1] -= gy;
            gradient[3 * j + 2] -= gz;
        }
    }
    (energy, gradient)
}

#[test]
fn argon_cluster_matches_gromacs_cg_reference_energy_and_force_scales() {
    // Reference values from `gmx_d mdrun` (GROMACS 2023.3 double precision) with
    // `integrator = cg`, `emstep = 0.02`, `nstcgsteep = 1000` on the 7-atom argon
    // fixture (`tests/data/gromacs_steep_argon.*`):
    //   - Initial (Step 0): Epot = -1.23140162769774e+01, Fmax = 7.58904e+00
    //   - At nsteps = 12, emtol = 1e-8:
    //       gmx_d reaches the 12-step cap with Epot = -1.25489376279604e+01,
    //       Fmax = 3.54506056338870e-03 (whereas steep is at Fmax = 4.087015e-01).
    //   - At emtol = 1e-2, nsteps = 100:
    //       gmx_d converges in 9 steps with Epot = -1.25489364597334e+01, Fmax < 1e-2.
    //   - At emtol = 1e-3, nsteps = 100:
    //       gmx_d converges in 13 steps with Epot = -1.25489376639539e+01, Fmax < 1e-3.
    //   - At emtol = 1e-6, nsteps = 100:
    //       gmx_d converges in 57 steps with Epot = -1.25489377088303e+01, Fmax < 1e-6.
    let fixture = parse_argon_fixture();
    let sigma = fixture.sigma;
    let epsilon = fixture.epsilon;
    let dim = fixture.initial_positions.len();
    let objective = Oracle::unbounded(dim, move |x: ArrayView1<f64>| {
        lennard_jones_energy_and_gradient(x, sigma, epsilon)
    });

    // Check 0-step assessment matches Step 0 of gmx_d mdrun.
    let mut x0 = Array1::from(fixture.initial_positions.clone());
    let mut zero_solver = Solver::new(Method::GromacsCg, control(fixture.emstep, 1e-8, 0), dim);
    let zero_report = zero_solver.run(&objective, &mut x0).unwrap();
    assert_eq!(zero_report.steps, 0);
    assert_abs_diff_eq!(zero_report.value, -1.23140162769774e+01, epsilon = 1e-12);
    assert_abs_diff_eq!(zero_report.grad_norm, 7.58904e+00, epsilon = 1e-5);

    // 12-step budget run (`nsteps = 12`, `emtol = 1e-8`, `emstep = 0.02`).
    let mut x_12 = Array1::from(fixture.initial_positions.clone());
    let mut solver_12 = Solver::new(
        Method::GromacsCg,
        control(fixture.emstep, 1e-8, fixture.nsteps),
        dim,
    );
    let report_12 = solver_12.run(&objective, &mut x_12).unwrap();
    assert_eq!(report_12.status, Status::Budget);
    assert_eq!(report_12.steps, fixture.nsteps);
    assert_eq!(report_12.stop_norm, StopNorm::MaxAtom);
    // Epot matches gmx_d (-12.5489376279604) to < 1e-6 kJ/mol, and Fmax is at the
    // same ~1e-3 kJ/(mol nm) order (over 100x lower than steep's 0.4087 after 12 steps).
    assert_abs_diff_eq!(report_12.value, -1.25489376279604e+01, epsilon = 1e-5);
    assert!(
        report_12.grad_norm < 5.0e-3,
        "expected Fmax < 5e-3 after 12 CG steps, got {}",
        report_12.grad_norm
    );

    // Convergence to emtol = 1e-2, 1e-3, and 1e-6, compared against gmx_d.
    let mut x_tol2 = Array1::from(fixture.initial_positions.clone());
    let mut solver_tol2 = Solver::new(Method::GromacsCg, control(fixture.emstep, 1e-2, 100), dim);
    let report_tol2 = solver_tol2.run(&objective, &mut x_tol2).unwrap();
    assert_eq!(report_tol2.status, Status::Tolerance);
    assert!(report_tol2.grad_norm < 1e-2);
    assert_eq!(report_tol2.steps, 6);
    assert_abs_diff_eq!(report_tol2.value, -1.25489364597334e+01, epsilon = 1e-5);

    let mut x_tol3 = Array1::from(fixture.initial_positions.clone());
    let mut solver_tol3 = Solver::new(Method::GromacsCg, control(fixture.emstep, 1e-3, 100), dim);
    let report_tol3 = solver_tol3.run(&objective, &mut x_tol3).unwrap();
    assert_eq!(report_tol3.status, Status::Tolerance);
    assert!(report_tol3.grad_norm < 1e-3);
    assert_eq!(report_tol3.steps, 7);
    assert_abs_diff_eq!(report_tol3.value, -1.25489376639539e+01, epsilon = 1e-6);

    let mut x_tol6 = Array1::from(fixture.initial_positions);
    let mut solver_tol6 = Solver::new(Method::GromacsCg, control(fixture.emstep, 1e-6, 100), dim);
    let report_tol6 = solver_tol6.run(&objective, &mut x_tol6).unwrap();
    assert_eq!(report_tol6.status, Status::Tolerance);
    assert!(report_tol6.grad_norm < 1e-6);
    assert_eq!(report_tol6.steps, 9);
    assert_abs_diff_eq!(report_tol6.value, -1.25489377088304e+01, epsilon = 1e-12);
}

#[test]
fn gromacs_cg_accepts_small_downhill_energy_increase_near_machine_precision() {
    // Match GROMACS `do_cg` (`src/gromacs/mdrun/minimize.cpp`):
    // accept `f < f0 + sqrt(GMX_REAL_EPS) * |f0|` when `gpc < 0` (downhill),
    // and reject at or above that threshold or when `gpc >= 0`.
    let e0 = 100.0_f64;
    let tol = f64::EPSILON.sqrt() * e0.abs();
    let accepted_rise = 0.25 * tol;
    let rejected_rise = 1.5 * tol;

    let downhill_small_rise = Oracle::unbounded(3, move |x: ArrayView1<f64>| {
        if x[0] == 0.0 {
            (e0, array![-1.0, 0.0, 0.0])
        } else if x[0] > 0.0 && x[0] <= 0.1 {
            (e0 + accepted_rise, array![-0.5, 0.0, 0.0])
        } else {
            (e0 + rejected_rise, array![-0.5, 0.0, 0.0])
        }
    });

    let mut x_cg = array![0.0, 0.0, 0.0];
    let mut cg_solver = Solver::new(Method::GromacsCg, control(0.1, 1e-6, 5), 3);
    let r_cg = cg_solver.step(&downhill_small_rise, &mut x_cg).unwrap();
    assert_eq!(r_cg.status, Status::Iterating);
    assert!(x_cg[0] > 0.0, "expected accepted step, got x={x_cg:?}");
    assert_eq!(r_cg.value, e0 + accepted_rise);

    // Without the GROMACS downhill-rise tolerance, standard NLCG rejects this rise.
    let mut x_pr = array![0.0, 0.0, 0.0];
    let mut pr_solver = Solver::new(Method::polak_ribiere(), control(0.1, 1e-6, 5), 3);
    let _r_pr = pr_solver.step(&downhill_small_rise, &mut x_pr).unwrap();
    assert_eq!(x_pr[0], 0.0);

    // A rise above sqrt(eps) * |e0| is rejected even when downhill.
    let too_high = Oracle::unbounded(3, move |x: ArrayView1<f64>| {
        if x[0] == 0.0 {
            (e0, array![-1.0, 0.0, 0.0])
        } else {
            (e0 + rejected_rise, array![-0.5, 0.0, 0.0])
        }
    });
    let mut x_high = array![0.0, 0.0, 0.0];
    let mut high_solver = Solver::new(Method::GromacsCg, control(0.1, 1e-6, 5), 3);
    let r_high = high_solver.step(&too_high, &mut x_high).unwrap();
    assert_eq!(r_high.status, Status::MachinePrecision);
    assert_eq!(x_high[0], 0.0);

    // An uphill directional derivative (gpc >= 0) is rejected even for a tiny rise.
    let uphill = Oracle::unbounded(3, move |x: ArrayView1<f64>| {
        if x[0] == 0.0 {
            (e0, array![-1.0, 0.0, 0.0])
        } else {
            (e0 + accepted_rise, array![0.5, 0.0, 0.0])
        }
    });
    let mut x_up = array![0.0, 0.0, 0.0];
    let mut up_solver = Solver::new(Method::GromacsCg, control(0.1, 1e-6, 5), 3);
    let r_up = up_solver.step(&uphill, &mut x_up).unwrap();
    assert_eq!(r_up.status, Status::MachinePrecision);
    assert_eq!(x_up[0], 0.0);
}

#[test]
fn gromacs_cg_keeps_downhill_rise_tolerance_after_set_objective_roundoff() {
    let e0 = 100.0_f64;
    let tol = f64::EPSILON.sqrt() * e0.abs();
    let accepted_rise = 0.25 * tol;
    let downhill_small_rise = Oracle::unbounded(3, move |x: ArrayView1<f64>| {
        if x[0] == 0.0 {
            (e0, array![-1.0, 0.0, 0.0])
        } else {
            (e0 + accepted_rise, array![-0.5, 0.0, 0.0])
        }
    });

    for roundoff in [1e-12, 0.0] {
        let mut x = array![0.0, 0.0, 0.0];
        let mut solver = Solver::new(Method::GromacsCg, control(0.1, 1e-6, 5), 3);
        assert!(solver.set_objective_roundoff(roundoff));
        assert!(!solver.set_objective_roundoff(-1.0));
        let report = solver.step(&downhill_small_rise, &mut x).unwrap();
        assert_eq!(report.status, Status::Iterating, "roundoff={roundoff}");
        assert!(
            x[0] > 0.0,
            "expected accepted step for roundoff={roundoff}, got x={x:?}"
        );
        assert_eq!(report.value, e0 + accepted_rise);
    }
}

#[test]
fn gromacs_cg_step_count_matches_maxiter_on_analytic_quadratic() {
    // f(x, y, z) = x^2 + 5 y^2 from (1, 1, 0) with exact line search converges
    // in 3 steps (step 1 SD, step 2 SD per nstcgsteep > 0, step 3 PR).
    // Pin exact step counts across maxiter = 0, 1, 2, 3, and 10 via both
    // minimize_method and Solver::run.
    let objective = ellipse();
    let start = array![1.0, 1.0, 0.0];
    let ls = LineSearch::brent(80, 1e-15);

    for (budget, want_steps, want_status) in [
        (0usize, 0usize, Status::Budget),
        (1, 1, Status::Budget),
        (2, 2, Status::Budget),
        (3, 3, Status::Tolerance),
        (10, 3, Status::Tolerance),
    ] {
        let ctrl = control(0.1, 1e-10, budget);
        let rep = minimize_method(&objective, start.clone(), &ctrl, Method::GromacsCg, ls).unwrap();
        assert_eq!(rep.steps, want_steps, "minimize_method budget={budget}");
        assert_eq!(rep.status, want_status, "minimize_method budget={budget}");

        let mut x = start.clone();
        let mut solver = Solver::new(Method::GromacsCg, ctrl, 3);
        solver.set_linesearch(ls);
        let rep_s = solver.run(&objective, &mut x).unwrap();
        assert_eq!(rep_s.steps, want_steps, "Solver::run budget={budget}");
        assert_eq!(rep_s.status, want_status, "Solver::run budget={budget}");
    }

    // Starting at a converged point takes 0 steps even with a positive budget.
    let zero_start = array![0.0, 0.0, 0.0];
    let rep_done = minimize_method(
        &objective,
        zero_start,
        &control(0.1, 1e-10, 5),
        Method::GromacsCg,
        ls,
    )
    .unwrap();
    assert_eq!(rep_done.steps, 0);
    assert_eq!(rep_done.status, Status::Tolerance);
}
