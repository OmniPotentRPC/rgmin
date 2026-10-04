//! Exact quadratic gradients with a bounded error in the reported value.
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, LineSearch, LineSearchOptions, Method, Solver};
use std::sync::atomic::{AtomicUsize, Ordering};

const START: f64 = 1.0 / 4096.0;
const NOISE: f64 = 1.0 / 16777216.0;
struct NoisyBowl {
    center: f64,
    start: f64,
    bounds: Bounds<f64>,
    calls: AtomicUsize,
}
impl NoisyBowl {
    fn new() -> Self {
        Self {
            center: 0.0,
            start: START,
            bounds: Bounds::new(array![-1.0], array![1.0], 0.0),
            calls: AtomicUsize::new(0),
        }
    }
}
impl Objective<f64> for NoisyBowl {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if x[0] == self.start {
            100.0
        } else {
            100.0 + NOISE + 0.5 * (x[0] - self.center).powi(2)
        }
    }
}
impl Gradient<f64> for NoisyBowl {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![x[0] - self.center]
    }
}
impl DifferentiableObjective<f64> for NoisyBowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        (self.eval(x), self.grad(x))
    }
}
fn wolfe() -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 20,
    }
}
fn options() -> LineSearchOptions {
    LineSearchOptions::with_objective_roundoff(1e-8).unwrap()
}
fn solver(slack: Option<f64>) -> Solver {
    Solver::new(
        Method::lbfgs(),
        Control {
            gtol: 1e-10,
            ftol_rel: slack,
            ..Control::default()
        },
        1,
    )
}

#[test]
fn configured_window_reaches_exact_stationarity_on_the_same_noisy_oracle() {
    for configured in [false, true] {
        let objective = NoisyBowl::new();
        let mut s = solver(None);
        let mut x = array![START];
        if configured {
            assert!(s.set_objective_roundoff(1e-8));
        }
        let r = s.step(&objective, &mut x).unwrap();
        if configured {
            assert_eq!(x, array![0.0]);
            assert_eq!(r.grad_norm, 0.0);
            assert_eq!(r.value, 100.0 + NOISE);
            assert_eq!(objective.calls.load(Ordering::Relaxed), 2);
            assert_eq!(s.pair_count(), 1);
        } else {
            assert_eq!(x, array![START]);
            assert_eq!(r.grad_norm, START);
            assert_eq!(r.value, 100.0);
            assert_eq!(s.pair_count(), 0);
        }
    }
}

#[test]
fn explicit_zero_slack_and_physical_policies_retain_their_value_test() {
    for (slack, policy) in [
        (Some(0.0), Accept::None),
        (None, Accept::Energy),
        (None, Accept::Nonmonotone),
    ] {
        let objective = NoisyBowl::new();
        let mut s = solver(slack);
        let mut x = array![START];
        s.set_accept(policy);
        assert!(s.set_objective_roundoff(1e-8));
        let r = s.step(&objective, &mut x).unwrap();
        assert_eq!(x, array![START]);
        assert_eq!(r.value, 100.0);
        assert_eq!(r.grad_norm, START);
        assert_eq!(s.pair_count(), 0);
    }
}

#[test]
fn invalid_options_leave_a_valid_solver_policy_intact() {
    let mut s = solver(None);
    assert!(s.set_objective_roundoff(1e-8));
    for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!s.set_objective_roundoff(value));
        assert!(LineSearchOptions::with_objective_roundoff(value).is_none());
    }
    let objective = NoisyBowl::new();
    let mut x = array![START];
    assert_eq!(s.step(&objective, &mut x).unwrap().grad_norm, 0.0);
    assert_eq!(x, array![0.0]);
}

#[test]
fn reversed_direction_and_zoom_keep_the_configured_window() {
    for (direction, initial) in [(START, 1.0), (-START, 2.0)] {
        let objective = NoisyBowl::new();
        let result = wolfe()
            .search_from_with_options(
                |x| objective.value_and_gradient(x),
                array![START].view(),
                100.0,
                array![START].view(),
                array![direction].view(),
                initial,
                2.0,
                options(),
            )
            .unwrap();
        assert!(result.g[0].abs() <= 0.9 * START);
        assert!(result.f - 100.0 <= 1e-6);
        assert!(result.alpha > 0.0 && result.alpha <= 2.0);
        if initial == 2.0 {
            assert!(result.evals > 1);
        }
    }
}

#[test]
fn clipped_trial_must_pass_curvature_along_its_actual_displacement() {
    let objective = NoisyBowl {
        center: START,
        start: 0.0,
        bounds: Bounds::new(array![-1.0], array![START / 100.0], 0.0),
        calls: AtomicUsize::new(0),
    };
    let mut s = solver(None);
    assert!(s.set_objective_roundoff(1e-8));
    let mut x = array![0.0];
    let r = s.step(&objective, &mut x).unwrap();
    assert_eq!(x, array![0.0]);
    assert_eq!(r.grad_norm, START);
    assert_eq!(r.value, 100.0);
    assert_eq!(s.pair_count(), 0);
}

#[test]
fn value_window_never_substitutes_for_finite_descent_and_curvature() {
    use rgmin::linesearch::conditions::approximate_strong_wolfe as accept;
    assert!(accept(100.0 + NOISE, 100.0, 0.0, -1.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(100.0 + 2e-6, 100.0, 0.0, -1.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(100.0 + NOISE, 100.0, -1.0, -1.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(100.0 + NOISE, 100.0, 1.0, -1.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(100.0, 100.0, 0.0, 0.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(100.0, 100.0, 0.0, 1.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(NOISE, 0.0, 0.0, -1.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(100.0 + NOISE, 100.0, 0.0, -1.0, 1e-4, 0.9, 0.0));
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!accept(invalid, 100.0, 0.0, -1.0, 1e-4, 0.9, 1e-8));
        assert!(!accept(100.0, 100.0, invalid, -1.0, 1e-4, 0.9, 1e-8));
        assert!(!accept(100.0, 100.0, 0.0, invalid, 1e-4, 0.9, 1e-8));
    }
}

#[cfg(feature = "capi")]
#[test]
fn c_setter_rejects_bad_inputs_without_replacing_the_session() {
    use rgmin::ffi::*;
    unsafe {
        assert_eq!(
            rgmin_solver_set_objective_roundoff(std::ptr::null_mut(), 1e-8),
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        );
        let control = rgmin_control_t {
            maxiter: 10,
            gtol: 1e-10,
            istep: 1.0,
            memory: 8,
            maxmove: 0.0,
        };
        let session = rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &control, 1);
        assert!(!session.is_null());
        assert_eq!(
            rgmin_solver_set_objective_roundoff(session, 1e-8),
            rgmin_status_t::RGMIN_SUCCESS
        );
        assert_eq!(
            rgmin_solver_set_objective_roundoff(session, f64::NAN),
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        );
        assert_eq!(
            rgmin_solver_set_objective_roundoff(session, 0.0),
            rgmin_status_t::RGMIN_SUCCESS
        );
        rgmin_solver_free(session);
    }
}

#[test]
fn a_decreasing_value_cannot_accept_a_nonfinite_gradient() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for configured in [false, true] {
            for slack in [None, Some(0.0)] {
                let objective = rgmin::Oracle::unbounded(1, move |x: ArrayView1<f64>| {
                    if x[0] == 0.0 { (1.0, array![1.0]) } else { (0.0, array![invalid]) }
                });
                let mut s = solver(slack);
                if configured { assert!(s.set_objective_roundoff(1e-8)); }
                let mut x = array![0.0];
                let result = s.step(&objective, &mut x).unwrap();
                assert_eq!(x, array![0.0]);
                assert_eq!(result.value, 1.0);
                assert_eq!(result.grad_norm, 1.0);
                assert_eq!(s.pair_count(), 0);
            }
        }
    }
}
