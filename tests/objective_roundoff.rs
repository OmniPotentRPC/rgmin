//! Exact quadratic gradients with a bounded error in the reported value.
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, LbfgsStep, LineSearch, LineSearchOptions, Method, Solver};
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
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            gtol: 1e-10,
            ftol_rel: slack,
            ..Control::default()
        },
        1,
    );
    // The window belongs to the strong Wolfe search. The default step is
    // the eOn cap, which does not consult it.
    solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
    solver
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

/// A C session opens under `Accept::Energy`. The strong Wolfe search of
/// every energy policy takes a rise inside the window its caller set,
/// with the two oracle calls `Accept::None` needs.
#[test]
fn energy_policies_take_a_rise_inside_the_configured_window() {
    for policy in [Accept::None, Accept::Energy, Accept::Nonmonotone] {
        let objective = NoisyBowl::new();
        let mut s = solver(None);
        s.set_accept(policy);
        assert!(s.set_objective_roundoff(1e-6));
        let mut x = array![START];
        let r = s.step(&objective, &mut x).unwrap();
        assert_eq!(x, array![0.0], "{policy:?}");
        assert_eq!(r.grad_norm, 0.0, "{policy:?}");
        assert_eq!(r.value, 100.0 + NOISE, "{policy:?}");
        assert_eq!(objective.calls.load(Ordering::Relaxed), 2, "{policy:?}");
        assert_eq!(s.pair_count(), 1, "{policy:?}");
    }
}

#[test]
fn explicit_zero_slack_and_the_default_window_keep_their_value_test() {
    for (slack, policy, configured) in [
        (Some(0.0), Accept::None, true),
        (Some(0.0), Accept::Energy, true),
        (Some(0.0), Accept::Nonmonotone, true),
        (None, Accept::Energy, false),
        (None, Accept::Nonmonotone, false),
    ] {
        let objective = NoisyBowl::new();
        let mut s = solver(slack);
        let mut x = array![START];
        s.set_accept(policy);
        if configured {
            assert!(s.set_objective_roundoff(1e-8));
        }
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
            noise_eps: 0.0,
            noise_c: 0.0,
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

#[cfg(feature = "capi")]
mod capi {
    use super::{NOISE, START};
    use dlpk::sys::DLManagedTensorVersioned;
    use rgmin::ffi::*;
    use std::os::raw::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};

    unsafe fn first(t: *const DLManagedTensorVersioned) -> *mut f64 {
        let dl = unsafe { &(*t).dl_tensor };
        unsafe { (dl.data as *mut u8).add(dl.byte_offset as usize) as *mut f64 }
    }

    /// `NoisyBowl` through the C callbacks. `user` counts evaluations.
    unsafe extern "C" fn eval(
        user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        value: *mut f64,
    ) -> rgmin_status_t {
        unsafe { (*(user as *const AtomicUsize)).fetch_add(1, Ordering::Relaxed) };
        let v = unsafe { *first(x) };
        let f = if v == START {
            100.0
        } else {
            100.0 + NOISE + 0.5 * v * v
        };
        unsafe { *value = f };
        rgmin_status_t::RGMIN_SUCCESS
    }

    unsafe extern "C" fn grad(
        _user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        g: *mut DLManagedTensorVersioned,
    ) -> rgmin_status_t {
        unsafe { *first(g) = *first(x) };
        rgmin_status_t::RGMIN_SUCCESS
    }

    /// A strong Wolfe session from `rgmin_solver_create` with a 1e-6 value
    /// window takes the rise of 2^-24 under its default acceptance, under
    /// `RGMIN_ACCEPT_ENERGY`, and under the 0.5 default `RGMIN_ACCEPT_NONE`.
    #[test]
    fn c_session_takes_a_rise_inside_the_configured_window() {
        for accept in [
            None,
            Some(rgmin_accept_t::RGMIN_ACCEPT_ENERGY),
            Some(rgmin_accept_t::RGMIN_ACCEPT_NONE),
        ] {
            let calls = AtomicUsize::new(0);
            let control = rgmin_control_t {
                maxiter: 10,
                gtol: 1e-10,
                istep: 1.0,
                memory: 8,
                maxmove: 0.0,
                noise_eps: 0.0,
                noise_c: 0.0,
            };
            let mut x = [START];
            let mut out = rgmin_report_t {
                value: 0.0,
                steps: 0,
                grad_norm: 0.0,
                stop: rgmin_stop_t::RGMIN_STOP_BUDGET,
            };
            let st = unsafe {
                let session = rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &control, 1);
                assert!(!session.is_null());
                assert_eq!(rgmin_solver_set_lbfgs_wolfe(session), 0);
                if let Some(accept) = accept {
                    rgmin_solver_set_accept(session, accept);
                }
                assert_eq!(
                    rgmin_solver_set_objective_roundoff(session, 1e-6),
                    rgmin_status_t::RGMIN_SUCCESS
                );
                let xt = rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 1);
                let st = rgmin_solver_step(
                    session,
                    Some(eval),
                    Some(grad),
                    &calls as *const AtomicUsize as *mut c_void,
                    xt,
                    &mut out,
                );
                rgmin_tensor_free(xt);
                rgmin_solver_free(session);
                st
            };
            assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS, "{accept:?}");
            assert_eq!(x, [0.0], "{accept:?}");
            assert_eq!(out.value, 100.0 + NOISE, "{accept:?}");
            assert_eq!(out.grad_norm, 0.0, "{accept:?}");
            assert_eq!(out.stop, rgmin_stop_t::RGMIN_STOP_TOLERANCE, "{accept:?}");
            assert!(calls.load(Ordering::Relaxed) <= 4, "{accept:?}: {calls:?}");
        }
    }
}

#[test]
fn a_decreasing_value_cannot_accept_a_nonfinite_gradient() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for configured in [false, true] {
            for slack in [None, Some(0.0)] {
                let objective = rgmin::Oracle::unbounded(1, move |x: ArrayView1<f64>| {
                    if x[0] == 0.0 {
                        (1.0, array![1.0])
                    } else {
                        (0.0, array![invalid])
                    }
                });
                let mut s = solver(slack);
                if configured {
                    assert!(s.set_objective_roundoff(1e-8));
                }
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
