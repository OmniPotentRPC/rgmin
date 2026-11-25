//! Huang, Absil, and Gallivan, SIAM J. Optim. 28 (2018) 470–495,
//! <https://doi.org/10.1137/17M1127582>.
//!
//! Section 3: on Euclidean space the transport is the identity and
//! Algorithm 2 is cautious BFGS. The quadratic `1/2 (x^2 + 100 y^2)` has
//! minimizer value 0. Section 7.2 stops at `||g|| / ||g_0|| < 10^{-6}`.
//!
//! Section 7.1 with `N = 1`, `n = 3`, `p = 1` and `R = 0`, so
//! `C = diag(3, 2, 1)` and the cost is `-(x^T C x)^2` on the sphere
//! (Stiefel `St(3, 1)`). A diagonalizer is the first basis vector and
//! the cost there is `-9`. The same section 7.2 stop applies.

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, HUANG_GRADIENT_RATIO, Manifold, ManifoldKind, Oracle, Rlbfgs, Status, StopReason,
    minimize_rlbfgs,
};

fn quadratic() -> Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let value = 0.5 * (x[0] * x[0] + 100.0 * x[1] * x[1]);
        let grad = array![x[0], 100.0 * x[1]];
        (value, grad)
    })
}

fn paper_control(maxiter: usize) -> Control {
    Control::default().with_maxiter(maxiter).with_gtol(0.0)
}

#[test]
fn euclidean_quadratic_meets_the_paper_stop() {
    let obj = quadratic();
    let start = array![1.0, 1.0];
    let (value0, grad0) = obj.value_and_gradient(start.view());
    let g0 = grad0.dot(&grad0).sqrt();
    let report = minimize_rlbfgs(
        &ManifoldKind::Euclidean,
        &obj,
        start,
        &paper_control(12),
        &Rlbfgs::default(),
    )
    .expect("rlbfgs");
    assert_eq!(report.status, Status::Tolerance, "{report:?}");
    assert!(
        report.value.abs() < 1e-8,
        "value {} from {value0}",
        report.value
    );
    assert!(
        report.grad_norm / g0 < HUANG_GRADIENT_RATIO,
        "ratio {} / {g0}",
        report.grad_norm
    );
}

fn joint_diagonalization() -> Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    Oracle::unbounded(3, |x: ArrayView1<f64>| {
        let diag = [3.0, 2.0, 1.0];
        let mut q = 0.0;
        for i in 0..3 {
            q += diag[i] * x[i] * x[i];
        }
        let value = -(q * q);
        let mut grad = Array1::zeros(3);
        for i in 0..3 {
            grad[i] = -4.0 * q * diag[i] * x[i];
        }
        (value, grad)
    })
}

#[test]
fn stiefel_joint_diagonalization_reaches_the_diagonalizer() {
    let obj = joint_diagonalization();
    let start = {
        let raw = array![1.0, 1.0, 1.0];
        let squared: f64 = raw.dot(&raw);
        raw / squared.sqrt()
    };
    let (_value0, grad0) = obj.value_and_gradient(start.view());
    let g0 = {
        let rgrad = ManifoldKind::Sphere.egrad2rgrad(&start, &grad0);
        ManifoldKind::Sphere.norm(&start, &rgrad)
    };
    let report = minimize_rlbfgs(
        &ManifoldKind::Sphere,
        &obj,
        start,
        &paper_control(20),
        &Rlbfgs::default(),
    )
    .expect("rlbfgs");
    assert_eq!(report.status, Status::Tolerance, "{report:?}");
    assert!(
        (report.value + 9.0).abs() < 1e-6,
        "cost {} is not -9",
        report.value
    );
    assert!(
        report.grad_norm / g0 < HUANG_GRADIENT_RATIO,
        "ratio {} / {g0}",
        report.grad_norm
    );
    let norm = report.coords.dot(&report.coords).sqrt();
    assert!((norm - 1.0).abs() < 1e-8, "left the sphere at {norm}");
}

struct ContractingMetric;

impl Manifold for ContractingMetric {
    fn project(&self, _x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        v.clone()
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        x + v
    }

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        v.clone()
    }

    fn inner(&self, x: &Array1<f64>, u: &Array1<f64>, v: &Array1<f64>) -> f64 {
        let metric = if x[0] <= 1.0 { 1e-12 } else { 1.0 };
        metric * u.dot(v)
    }

    fn exp(&self, x: &Array1<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
        Some(x + v)
    }

    fn parallel_transport_along(
        &self,
        _x: &Array1<f64>,
        _d: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        Some(v.clone())
    }
}

#[test]
fn riemannian_gradient_tolerance_outranks_the_evaluation_budget() {
    let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]));
    let control = Control::default()
        .with_maxiter(2)
        .with_gtol(1e-4)
        .with_maxmove(1.0)
        .with_max_evals(2);
    let params = Rlbfgs::default().with_grad_ratio(0.0);
    let report =
        minimize_rlbfgs(&ContractingMetric, &obj, array![2.0], &control, &params).expect("rlbfgs");
    assert_eq!(report.status, Status::Tolerance, "{report:?}");
    assert!(report.grad_norm < control.gtol, "{report:?}");
    assert_eq!(report.evals, 2);
}

#[test]
fn rlbfgs_checks_the_evaluation_budget_when_the_iteration_ends() {
    let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| {
        (1.5 * x[0] * x[0], array![3.0 * x[0]])
    });
    let control = Control::default()
        .with_maxiter(10)
        .with_gtol(1e-12)
        .with_max_evals(2);
    let params = Rlbfgs::default().with_grad_ratio(0.0);
    let report = minimize_rlbfgs(
        &ManifoldKind::Euclidean,
        &obj,
        array![1.0],
        &control,
        &params,
    )
    .expect("rlbfgs");
    assert_eq!(report.status, Status::EvalBudget, "{report:?}");
    assert_eq!(report.steps, 1, "{report:?}");
    assert_eq!(report.evals, 3, "{report:?}");
    assert!((report.coords[0] + 0.5).abs() < 1e-12, "{report:?}");
    assert!(report.value < 1.5, "{report:?}");
}

/// A start that is the only point with a low value, so no trial can
/// satisfy the Armijo test and the line search ends without a step.
fn uphill_everywhere() -> Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    Oracle::unbounded(1, |x: ArrayView1<f64>| {
        let value = if x[0] == 1.0 { 0.5 } else { 10.0 };
        (value, array![x[0]])
    })
}

#[test]
fn rlbfgs_names_the_evaluation_budget_when_the_line_search_fails_with_it_spent() {
    let control = Control::default()
        .with_maxiter(10)
        .with_gtol(1e-12)
        .with_max_evals(3);
    let params = Rlbfgs::default().with_grad_ratio(0.0);
    let report = minimize_rlbfgs(
        &ManifoldKind::Euclidean,
        &uphill_everywhere(),
        array![1.0],
        &control,
        &params,
    )
    .expect("rlbfgs");
    assert_eq!(report.status, Status::EvalBudget, "{report:?}");
    assert_eq!(report.stop_reason(), StopReason::EvalBudget, "{report:?}");
    assert_eq!(report.steps, 1, "{report:?}");
    assert!(report.evals >= 3, "{report:?}");
    assert_eq!(report.coords, array![1.0], "{report:?}");
    assert_eq!(report.value, 0.5, "{report:?}");
}

#[test]
fn rlbfgs_reports_a_failed_line_search_while_the_evaluation_budget_holds() {
    let control = Control::default()
        .with_maxiter(10)
        .with_gtol(1e-12)
        .with_max_evals(1_000);
    let params = Rlbfgs::default().with_grad_ratio(0.0);
    let report = minimize_rlbfgs(
        &ManifoldKind::Euclidean,
        &uphill_everywhere(),
        array![1.0],
        &control,
        &params,
    )
    .expect("rlbfgs");
    assert_eq!(
        report.stop_reason(),
        StopReason::LineSearchFailed,
        "{report:?}"
    );
    assert_eq!(report.steps, 0, "{report:?}");
    assert_eq!(report.coords, array![1.0], "{report:?}");
}

#[cfg(feature = "capi")]
mod capi {
    use std::os::raw::c_void;

    use dlpk::sys::DLManagedTensorVersioned;
    use rgmin::ffi::{
        rgmin_control_t, rgmin_manifold_t, rgmin_minimize_rlbfgs, rgmin_report_t, rgmin_rlbfgs_t,
        rgmin_status_t, rgmin_stop_t, rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
    };

    unsafe extern "C" fn eval(
        _user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        value_out: *mut f64,
    ) -> rgmin_status_t {
        let p = unsafe { (*x).dl_tensor.data as *const f64 };
        unsafe {
            let x0 = *p;
            let x1 = *p.add(1);
            *value_out = 0.5 * (x0 * x0 + 100.0 * x1 * x1);
        }
        rgmin_status_t::RGMIN_SUCCESS
    }

    unsafe extern "C" fn grad(
        _user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        g: *mut DLManagedTensorVersioned,
    ) -> rgmin_status_t {
        let p = unsafe { (*x).dl_tensor.data as *const f64 };
        let gp = unsafe { (*g).dl_tensor.data as *mut f64 };
        unsafe {
            *gp = *p;
            *gp.add(1) = 100.0 * *p.add(1);
        }
        rgmin_status_t::RGMIN_SUCCESS
    }

    #[test]
    fn c_abi_euclidean_quadratic_meets_the_paper_stop() {
        let mut x = [1.0, 1.0];
        let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
        let ctrl = rgmin_control_t {
            maxiter: 12,
            gtol: 0.0,
            istep: 1.0,
            memory: 8,
            maxmove: 0.0,
            noise_eps: 0.0,
            noise_c: 0.0,
            require_convergence: 0,
            observer: None,
            observer_user: std::ptr::null_mut(),
        };
        let params = rgmin_rlbfgs_t {
            memory: 4,
            cautious_slope: 1.0 / 10_000.0,
            armijo: 1.0 / 10_000.0,
            backtrack: 0.5,
            linesearch_max: 48,
            grad_ratio: 1.0 / 1_000_000.0,
        };
        let mut out = rgmin_report_t {
            value: 1.0,
            steps: 0,
            grad_norm: 1.0,
            stop: rgmin_stop_t::RGMIN_STOP_BUDGET,
            converged: 0,
        };
        let status = unsafe {
            rgmin_minimize_rlbfgs(
                Some(eval),
                Some(grad),
                std::ptr::null_mut(),
                tensor,
                &ctrl,
                &params,
                rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(tensor) };
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
        assert_eq!(out.stop, rgmin_stop_t::RGMIN_STOP_TOLERANCE);
        assert!(out.value.abs() < 1e-8, "value {}", out.value);
        let g0 = (1.0_f64 + 100.0 * 100.0).sqrt();
        assert!(
            out.grad_norm / g0 < 1.0 / 1_000_000.0,
            "grad {}",
            out.grad_norm
        );
    }
}
