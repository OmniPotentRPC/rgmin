use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::nlcg::{Conjugacy, Restart};
use rgmin::{
    Control, DirectionalCurvature, Report, ScgOptions, ScgParams, ScgStepTolerance, minimize_scg,
    minimize_scg_exact, minimize_scg_exact_with_options, minimize_scg_with_options,
};
use std::sync::Mutex;

struct ShiftedBowl {
    center: f64,
    bounds: Bounds<f64>,
    visits: Mutex<Vec<Vec<f64>>>,
}
impl ShiftedBowl {
    fn new(center: f64) -> Self {
        Self {
            center,
            bounds: Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0),
            visits: Mutex::new(Vec::new()),
        }
    }
}
impl Objective<f64> for ShiftedBowl {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x.iter().map(|v| 0.5 * (v - self.center).powi(2)).sum()
    }
}
impl Gradient<f64> for ShiftedBowl {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        &x - self.center
    }
}
impl DifferentiableObjective<f64> for ShiftedBowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.visits.lock().unwrap().push(x.to_vec());
        (self.eval(x), self.grad(x))
    }
}
impl DirectionalCurvature for ShiftedBowl {
    fn directional_curvature(&self, _x: ArrayView1<f64>, d: ArrayView1<f64>) -> Option<f64> {
        Some(d.dot(&d))
    }
}
fn control() -> Control {
    Control {
        maxiter: 2,
        gtol: 0.0,
        ..Control::default()
    }
}
fn params(tol: f64) -> ScgParams {
    ScgParams {
        tol_sol: tol,
        tol_func: 1.0,
        ..ScgParams::default()
    }
}
fn run(center: f64, tol: f64, exact: bool, options: Option<ScgOptions>) -> (Report, Vec<Vec<f64>>) {
    let objective = ShiftedBowl::new(center);
    let x = array![center + 1.0, center + 1.0];
    let report = match (exact, options) {
        (false, None) => minimize_scg(
            &objective,
            x,
            &control(),
            &params(tol),
            Conjugacy::LiuStorey,
            Restart::Never,
        ),
        (true, None) => minimize_scg_exact(
            &objective,
            x,
            &control(),
            &params(tol),
            Conjugacy::LiuStorey,
            Restart::Never,
        ),
        (false, Some(o)) => minimize_scg_with_options(
            &objective,
            x,
            &control(),
            &params(tol),
            Conjugacy::LiuStorey,
            Restart::Never,
            &o,
        ),
        (true, Some(o)) => minimize_scg_exact_with_options(
            &objective,
            x,
            &control(),
            &params(tol),
            Conjugacy::LiuStorey,
            Restart::Never,
            &o,
        ),
    }
    .unwrap();
    (report, objective.visits.into_inner().unwrap())
}
fn relative() -> ScgOptions {
    ScgOptions {
        step_tolerance: ScgStepTolerance::RelativeEuclidean,
    }
}

#[test]
fn default_options_preserve_coordinates_values_steps_and_every_visit() {
    for exact in [false, true] {
        let (legacy, old_visits) = run(100.0, 0.01, exact, None);
        let (explicit, new_visits) = run(100.0, 0.01, exact, Some(ScgOptions::default()));
        assert_eq!(legacy.coords, explicit.coords);
        assert_eq!(legacy.value, explicit.value);
        assert_eq!(legacy.grad_norm, explicit.grad_norm);
        assert_eq!(legacy.steps, explicit.steps);
        assert_eq!(old_visits, new_visits);
        assert_eq!(legacy.steps, 2);
        assert_eq!(old_visits.len(), if exact { 3 } else { 5 });
    }
}

#[test]
fn relative_rule_uses_the_accepted_point_and_stops_without_an_extra_probe() {
    for exact in [false, true] {
        let (report, visits) = run(100.0, 0.01, exact, Some(relative()));
        assert_eq!(report.steps, 1);
        for x in report.coords {
            assert!((x - 100.5).abs() < 1e-7);
        }
        assert_eq!(visits.len(), if exact { 2 } else { 3 });
        assert_eq!(visits.last().unwrap().len(), 2);
        assert!((visits.last().unwrap()[0] - 100.5).abs() < 1e-7);
    }
}

#[test]
fn relative_rule_uses_euclidean_step_norm_and_accepted_coordinate_scale() {
    let (report, visits) = run(0.0, 0.35, true, Some(relative()));
    // First step (-1/2,-1/2) has length sqrt(1/2), above
    // 0.35*(1+sqrt(1/2)); either an infinity norm or the initial
    // coordinate scale would terminate at that first accepted point.
    assert_eq!(report.steps, 2);
    assert_eq!(visits.len(), 3);
    assert_eq!(visits[1], vec![0.5, 0.5]);
    for x in report.coords {
        assert!((x - 1.0 / 6.0).abs() < 1e-14);
    }
}

#[cfg(feature = "capi")]
mod capi {
    use super::*;
    use dlpk::sys::DLManagedTensorVersioned;
    use rgmin::ffi::*;
    use std::ffi::c_void;
    use std::mem::{offset_of, size_of};
    #[derive(Default)]
    struct Calls {
        eval: Vec<Vec<f64>>,
        grad: usize,
        curvature: usize,
    }
    unsafe fn point<'a>(x: *const DLManagedTensorVersioned) -> &'a [f64] {
        unsafe { std::slice::from_raw_parts((*x).dl_tensor.data.cast(), 2) }
    }
    unsafe extern "C" fn eval(
        user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        out: *mut f64,
    ) -> rgmin_status_t {
        let x = unsafe { point(x) };
        unsafe {
            (&mut *user.cast::<Calls>()).eval.push(x.to_vec());
            *out = x.iter().map(|v| 0.5 * (v - 100.0).powi(2)).sum();
        }
        rgmin_status_t::RGMIN_SUCCESS
    }
    unsafe extern "C" fn grad(
        user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        out: *mut DLManagedTensorVersioned,
    ) -> rgmin_status_t {
        let x = unsafe { point(x) };
        unsafe {
            (&mut *user.cast::<Calls>()).grad += 1;
            let dest = (*out).dl_tensor.data.cast::<f64>();
            for i in 0..2 {
                *dest.add(i) = x[i] - 100.0;
            }
        }
        rgmin_status_t::RGMIN_SUCCESS
    }
    unsafe extern "C" fn curvature(
        user: *mut c_void,
        _x: *const DLManagedTensorVersioned,
        d: *const DLManagedTensorVersioned,
        out: *mut f64,
    ) -> rgmin_status_t {
        let d = unsafe { point(d) };
        unsafe {
            (&mut *user.cast::<Calls>()).curvature += 1;
            *out = d.iter().map(|v| v * v).sum();
        }
        rgmin_status_t::RGMIN_SUCCESS
    }
    fn solve(kind: i32) -> (rgmin_status_t, [f64; 2], rgmin_report_t, Calls) {
        let mut x = [101.0; 2];
        let mut calls = Calls::default();
        let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
        let control = rgmin_control_t {
            maxiter: 2,
            gtol: 0.0,
            istep: 1.0,
            memory: 5,
            maxmove: 0.0,
        };
        let params = rgmin_scg_params_t {
            sigma0: 1e-4,
            lambda: 1.0,
            lambda_limit: 1e60,
            tol_sol: 0.01,
            tol_func: 1.0,
            conjugacy: rgmin_conjugacy_t::RGMIN_CONJUGACY_LIU_STOREY as i32,
        };
        let options = rgmin_scg_options_t {
            step_tolerance: kind,
        };
        let mut report = rgmin_report_t {
            value: 42.0,
            steps: 43,
            grad_norm: 44.0,
        };
        let status = unsafe {
            rgmin_minimize_scg_with_options(
                Some(eval),
                Some(grad),
                Some(curvature),
                (&mut calls as *mut Calls).cast(),
                tensor,
                &control,
                &params,
                &options,
                &mut report,
            )
        };
        unsafe { rgmin_tensor_free(tensor) };
        (status, x, report, calls)
    }
    #[test]
    fn c_policy_matches_rust_visits_and_retains_the_base_parameter_layout() {
        assert_eq!(size_of::<rgmin_scg_params_t>(), 48);
        assert_eq!(offset_of!(rgmin_scg_params_t, conjugacy), 40);
        assert_eq!(size_of::<rgmin_scg_options_t>(), 4);
        for (kind, options, expected_calls) in [(0, ScgOptions::default(), 3), (1, relative(), 2)] {
            let (rust, visits) = run(100.0, 0.01, true, Some(options));
            let (status, x, c, calls) = solve(kind);
            assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
            assert_eq!(x.as_slice(), rust.coords.as_slice().unwrap());
            assert_eq!(c.value, rust.value);
            assert_eq!(c.steps, rust.steps);
            assert_eq!(c.grad_norm, rust.grad_norm);
            assert_eq!(calls.eval, visits);
            assert_eq!(calls.grad, expected_calls);
            assert_eq!(calls.curvature, expected_calls - 1);
        }
    }
    #[test]
    fn unknown_policy_keeps_inputs_and_outputs_without_any_callback() {
        for kind in [-1, 2, 256] {
            let (status, x, report, calls) = solve(kind);
            assert_eq!(status, rgmin_status_t::RGMIN_INVALID_PARAMETER);
            assert_eq!(x, [101.0; 2]);
            assert_eq!(report.value, 42.0);
            assert_eq!(report.steps, 43);
            assert_eq!(report.grad_norm, 44.0);
            assert!(calls.eval.is_empty());
            assert_eq!(calls.grad, 0);
            assert_eq!(calls.curvature, 0);
        }
    }
}
