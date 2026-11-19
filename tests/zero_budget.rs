//! A zero iteration budget takes no step, and its report is the objective
//! evaluated at the start: [`Status::Budget`] there, or
//! [`Status::Tolerance`] only when that gradient is below `gtol`.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::{
    Control, FireKind, HessianObjective, HessianVector, LineSearch, Method, NewtonKind, Report,
    Solver, Status, minimize_method, minimize_method_hess, minimize_method_hvp,
};

const GTOL: f64 = 1e-8;

/// `f = 7 + 1/2 (x^2 + 2 y^2 + 3 z^2)`.
struct Bowl {
    bounds: Bounds<f64>,
}

fn bowl() -> Bowl {
    Bowl {
        bounds: Bounds::new(Array1::from_elem(3, -1e6), Array1::from_elem(3, 1e6), 0.0),
    }
}

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        3
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let g = array![x[0], 2.0 * x[1], 3.0 * x[2]];
        (7.0 + 0.5 * x.dot(&g), g)
    }
}

impl HessianObjective for Bowl {
    fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
        Array2::from_diag(&array![1.0, 2.0, 3.0])
    }
}

impl HessianVector for Bowl {
    fn hessian_vector(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        array![v[0], 2.0 * v[1], 3.0 * v[2]]
    }
}

fn moving() -> Array1<f64> {
    array![1.0, -1.0, 0.5]
}

fn stationary() -> Array1<f64> {
    array![0.0, 0.0, 0.0]
}

fn zero_budget() -> Control {
    Control::default().with_maxiter(0).with_gtol(GTOL)
}

/// Every first-order method a session or `minimize_method` runs.
fn first_order() -> Vec<Method> {
    vec![
        Method::polak_ribiere(),
        Method::Bfgs,
        Method::lbfgs(),
        Method::Sr1,
        Method::Sr2,
        Method::adam(),
        Method::Steepest,
        Method::pso(),
        Method::fire(FireKind::V1),
        Method::fire(FireKind::V2),
        Method::Bb,
        Method::QuickMin,
        Method::Ode12r,
        Method::diis(),
        Method::NewtonCg,
        Method::NewtonPcg,
    ]
}

/// The report is `f` and `||g||` at `x0` with no step, `Budget` away
/// from the minimum and `Tolerance` at it.
fn assert_judged_at_start(what: &str, x0: &Array1<f64>, report: &Report) {
    let (f, g) = bowl().value_and_gradient(x0.view());
    let gnorm = g.dot(&g).sqrt();
    assert_eq!(report.steps, 0, "{what}: {report:?}");
    assert_eq!(report.value, f, "{what}: {report:?}");
    assert!(
        (report.grad_norm - gnorm).abs() <= 1e-12 * gnorm.max(1.0),
        "{what}: {report:?}"
    );
    let expected = if gnorm < GTOL {
        Status::Tolerance
    } else {
        Status::Budget
    };
    assert_eq!(report.status, expected, "{what}: {report:?}");
}

#[test]
fn a_zero_budget_session_run_judges_the_evaluated_start() {
    for method in first_order() {
        for x0 in [moving(), stationary()] {
            let mut solver = Solver::new(method.clone(), zero_budget(), 3);
            let mut x = x0.clone();
            let report = solver.run(&bowl(), &mut x).unwrap();
            assert_eq!(x, x0, "{method:?}");
            assert_eq!(solver.steps(), 0, "{method:?}");
            assert_judged_at_start(&format!("run {method:?}"), &x0, &report);
        }
    }
}

/// `minimize_method` and its Hessian and Hessian-vector forms. The
/// session-backed methods (FIRE, BB, QuickMin, ODE12r, DIIS, Newton-CG,
/// dogleg) reported value 0, `||g|| = inf` and `Budget` without an
/// evaluation, even at the minimum.
#[test]
fn zero_budget_one_shots_judge_the_evaluated_start() {
    for x0 in [moving(), stationary()] {
        for method in first_order() {
            let report = minimize_method(
                &bowl(),
                x0.clone(),
                &zero_budget(),
                method.clone(),
                LineSearch::default(),
            )
            .unwrap();
            assert_eq!(report.coords, x0, "{method:?}");
            assert_judged_at_start(&format!("minimize_method {method:?}"), &x0, &report);
        }
        let second_order = [
            Method::newton(NewtonKind::Shifted),
            Method::newton(NewtonKind::Rfo),
            Method::Dogleg,
        ];
        for method in first_order().into_iter().chain(second_order) {
            let report = minimize_method_hess(
                &bowl(),
                x0.clone(),
                &zero_budget(),
                method.clone(),
                LineSearch::default(),
            )
            .unwrap();
            assert_judged_at_start(&format!("minimize_method_hess {method:?}"), &x0, &report);
        }
        for method in first_order() {
            let report = minimize_method_hvp(
                &bowl(),
                x0.clone(),
                &zero_budget(),
                method.clone(),
                LineSearch::default(),
            )
            .unwrap();
            assert_judged_at_start(&format!("minimize_method_hvp {method:?}"), &x0, &report);
        }
    }
}

#[cfg(feature = "capi")]
mod capi {
    use super::GTOL;
    use dlpk::sys::DLManagedTensorVersioned;
    use eindir_core::ffi::{eindir_core_abi_stamp, eindir_objective_t, eindir_status_t};
    use rgmin::ffi::*;
    use std::os::raw::c_void;

    const N: usize = 3;
    const CURVATURE: [f64; N] = [1.0, 2.0, 3.0];

    unsafe fn data(t: *const DLManagedTensorVersioned) -> *mut f64 {
        let dl = unsafe { &(*t).dl_tensor };
        unsafe { (dl.data as *mut u8).add(dl.byte_offset as usize) as *mut f64 }
    }

    fn value_at(x: *const f64) -> f64 {
        7.0 + 0.5
            * (0..N)
                .map(|i| CURVATURE[i] * unsafe { *x.add(i) * *x.add(i) })
                .sum::<f64>()
    }

    unsafe extern "C" fn eval(
        _user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        value: *mut f64,
    ) -> rgmin_status_t {
        unsafe { *value = value_at(data(x)) };
        rgmin_status_t::RGMIN_SUCCESS
    }

    unsafe extern "C" fn grad(
        _user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        g: *mut DLManagedTensorVersioned,
    ) -> rgmin_status_t {
        let (x, g) = unsafe { (data(x), data(g)) };
        for (i, c) in CURVATURE.iter().enumerate() {
            unsafe { *g.add(i) = c * *x.add(i) };
        }
        rgmin_status_t::RGMIN_SUCCESS
    }

    unsafe extern "C" fn hess(
        _user: *mut c_void,
        _x: *const DLManagedTensorVersioned,
        h: *mut DLManagedTensorVersioned,
    ) -> rgmin_status_t {
        let h = unsafe { data(h) };
        for (i, c) in CURVATURE.iter().enumerate() {
            for j in 0..N {
                unsafe { *h.add(i * N + j) = if i == j { *c } else { 0.0 } };
            }
        }
        rgmin_status_t::RGMIN_SUCCESS
    }

    unsafe extern "C" fn hvp(
        _user: *mut c_void,
        _x: *const DLManagedTensorVersioned,
        v: *const DLManagedTensorVersioned,
        hv: *mut DLManagedTensorVersioned,
    ) -> rgmin_status_t {
        let (v, hv) = unsafe { (data(v), data(hv)) };
        for (i, c) in CURVATURE.iter().enumerate() {
            unsafe { *hv.add(i) = c * *v.add(i) };
        }
        rgmin_status_t::RGMIN_SUCCESS
    }

    unsafe extern "C" fn eindir_eval(
        _user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        value: *mut f64,
    ) -> eindir_status_t {
        unsafe { *value = value_at(data(x)) };
        eindir_status_t::EINDIR_SUCCESS
    }

    unsafe extern "C" fn eindir_grad(
        user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        g: *mut DLManagedTensorVersioned,
    ) -> eindir_status_t {
        unsafe { grad(user, x, g) };
        eindir_status_t::EINDIR_SUCCESS
    }

    fn control() -> rgmin_control_t {
        rgmin_control_t {
            maxiter: 0,
            gtol: GTOL,
            istep: 0.0,
            memory: 8,
            maxmove: 0.0,
            noise_eps: 0.0,
            noise_c: 0.0,
            require_convergence: 0,
        }
    }

    fn blank() -> rgmin_report_t {
        rgmin_report_t {
            value: -1.0,
            steps: 99,
            grad_norm: -1.0,
            stop: rgmin_stop_t::RGMIN_STOP_NOISE_FLOOR,
            converged: 0,
        }
    }

    enum Entry {
        Plain,
        Hess,
        Hvp,
        Eindir,
    }

    fn one_shot(
        entry: &Entry,
        method: rgmin_method_t,
        start: [f64; N],
    ) -> ([f64; N], rgmin_report_t) {
        let mut x = start;
        let mut out = blank();
        let ctrl = control();
        let objective = eindir_objective_t {
            dim: N,
            low: std::ptr::null_mut(),
            high: std::ptr::null_mut(),
            eval_fn: eindir_eval,
            grad_fn: Some(eindir_grad),
            user_data: std::ptr::null_mut(),
            free_fn: None,
        };
        let st = unsafe {
            let xt = rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), N);
            let user = std::ptr::null_mut();
            let st = match entry {
                Entry::Plain => {
                    rgmin_minimize(Some(eval), Some(grad), user, xt, &ctrl, method, &mut out)
                }
                Entry::Hess => rgmin_minimize_hess(
                    Some(eval),
                    Some(grad),
                    Some(hess),
                    user,
                    xt,
                    &ctrl,
                    method,
                    &mut out,
                ),
                Entry::Hvp => rgmin_minimize_hvp(
                    Some(eval),
                    Some(grad),
                    Some(hvp),
                    user,
                    xt,
                    &ctrl,
                    method,
                    &mut out,
                ),
                Entry::Eindir => rgmin_minimize_eindir(
                    &objective,
                    &eindir_core_abi_stamp(),
                    xt,
                    &ctrl,
                    method,
                    &mut out,
                ),
            };
            rgmin_tensor_free(xt);
            st
        };
        assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
        (x, out)
    }

    /// Every C one-shot with a zero budget reports `f` and `||g||` at the
    /// start and takes no step. The session-backed methods wrote value 0,
    /// `grad_norm = inf` and `RGMIN_STOP_BUDGET`, even at the minimum.
    #[test]
    fn zero_budget_c_one_shots_judge_the_evaluated_start() {
        use rgmin_method_t::*;
        let plain = [
            RGMIN_LBFGS,
            RGMIN_FIRE,
            RGMIN_FIRE2,
            RGMIN_BB,
            RGMIN_QUICKMIN,
            RGMIN_DIIS,
            RGMIN_ODE12R,
            RGMIN_NEWTON_CG,
        ];
        let cases: Vec<(Entry, rgmin_method_t)> = plain
            .iter()
            .map(|m| (Entry::Plain, *m))
            .chain(plain.iter().map(|m| (Entry::Eindir, *m)))
            .chain([RGMIN_NEWTON, RGMIN_RFO, RGMIN_DOGLEG, RGMIN_FIRE].map(|m| (Entry::Hess, m)))
            .chain([RGMIN_NEWTON_CG, RGMIN_NEWTON_PCG, RGMIN_BB].map(|m| (Entry::Hvp, m)))
            .collect();
        for (entry, method) in &cases {
            for start in [[1.0, -1.0, 0.5], [0.0; N]] {
                let (x, out) = one_shot(entry, *method, start);
                let gnorm = (0..N)
                    .map(|i| (CURVATURE[i] * start[i]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let what = format!(
                    "{method:?} via {}",
                    match entry {
                        Entry::Plain => "rgmin_minimize",
                        Entry::Hess => "rgmin_minimize_hess",
                        Entry::Hvp => "rgmin_minimize_hvp",
                        Entry::Eindir => "rgmin_minimize_eindir",
                    }
                );
                assert_eq!(x, start, "{what}");
                assert_eq!(out.steps, 0, "{what}");
                assert_eq!(out.value, value_at(start.as_ptr()), "{what}");
                assert!(
                    (out.grad_norm - gnorm).abs() <= 1e-12,
                    "{what}: {}",
                    out.grad_norm
                );
                let stop = if gnorm < GTOL {
                    rgmin_stop_t::RGMIN_STOP_TOLERANCE
                } else {
                    rgmin_stop_t::RGMIN_STOP_BUDGET
                };
                assert_eq!(out.stop, stop, "{what}");
            }
        }
    }
}
