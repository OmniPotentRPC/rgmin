//! Evaluations per second through the C ABI on a trivial objective.
//!
//! The objective is a quadratic bowl, so the callback itself is a few
//! flops and the measurement is dominated by the marshalling waist:
//! whatever the FFI layer allocates, copies, and frees per evaluation.
//! Run before and after a waist change; the printed rate is the number.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use dlpk::sys::DLManagedTensorVersioned;
use rgmin::ffi::{
    rgmin_control_t, rgmin_method_t, rgmin_minimize, rgmin_report_t, rgmin_status_t, rgmin_stop_t,
    rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
};

static EVALS: AtomicUsize = AtomicUsize::new(0);

unsafe fn slice_of<'a>(t: *const DLManagedTensorVersioned) -> &'a [f64] {
    let dl = unsafe { &(*t).dl_tensor };
    let n = unsafe { *dl.shape as usize };
    unsafe { std::slice::from_raw_parts(dl.data as *const f64, n) }
}

unsafe fn slice_of_mut<'a>(t: *mut DLManagedTensorVersioned) -> &'a mut [f64] {
    let dl = unsafe { &(*t).dl_tensor };
    let n = unsafe { *dl.shape as usize };
    unsafe { std::slice::from_raw_parts_mut(dl.data as *mut f64, n) }
}

unsafe extern "C" fn bowl_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    EVALS.fetch_add(1, Ordering::Relaxed);
    let xs = unsafe { slice_of(x) };
    unsafe { *value_out = xs.iter().map(|v| v * v).sum() };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn bowl_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    grad_out: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let xs = unsafe { slice_of(x) };
    let gs = unsafe { slice_of_mut(grad_out) };
    for (g, v) in gs.iter_mut().zip(xs) {
        *g = 2.0 * v;
    }
    rgmin_status_t::RGMIN_SUCCESS
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let rounds: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let ctrl = rgmin_control_t {
        maxiter: 60,
        gtol: 1e-10,
        istep: 1e-3,
        memory: 0,
        maxmove: 0.0,
        noise_eps: 0.0,
        noise_c: 0.0,
    };

    let start = Instant::now();
    for round in 0..rounds {
        let mut x: Vec<f64> = (0..n)
            .map(|i| 1.0 + ((i + round) % 7) as f64 * 0.1)
            .collect();
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), n) };
        let mut report = rgmin_report_t {
            value: 0.0,
            steps: 0,
            grad_norm: 0.0,
            stop: rgmin_stop_t::RGMIN_STOP_TOLERANCE,
        };
        let st = unsafe {
            rgmin_minimize(
                Some(bowl_eval),
                Some(bowl_grad),
                std::ptr::null_mut(),
                xt,
                &ctrl,
                rgmin_method_t::RGMIN_LBFGS,
                &mut report,
            )
        };
        unsafe { rgmin_tensor_free(xt) };
        assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS, "round {round}");
    }
    let dt = start.elapsed().as_secs_f64();
    let evals = EVALS.load(Ordering::Relaxed);
    println!(
        "n={n} rounds={rounds} evals={evals} wall={dt:.3}s rate={:.0} evals/s",
        evals as f64 / dt
    );
}
