//! An off-diagonal entry of 1e-160 next to a unit gap, in the
//! transition-state Broyden-Fletcher-Goldfarb-Shanno (TS-BFGS) update
//! and in the rational function optimization (RFO) step. Both
//! diagonalize with cyclic Jacobi, where `theta = (a_qq - a_pp) / (2 a_pq)`
//! reaches 5e159 for this 3-by-3 model and `theta^2` overflowed. With
//! overflow trapped, the host stopped there with the floating-point
//! exception signal (SIGFPE).
#![cfg(all(
    target_os = "linux",
    any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
))]

use ndarray::{Array1, Array2, array};
use rgmin::{rfo_get_s, ts_bfgs_update};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const FE_OVERFLOW: i32 = 0x08;
#[cfg(target_arch = "aarch64")]
const FE_OVERFLOW: i32 = 0x04;

unsafe extern "C" {
    fn feclearexcept(excepts: i32) -> i32;
    fn fetestexcept(excepts: i32) -> i32;
}

/// Whether `f` raises the floating-point overflow flag of this thread.
fn raises_overflow(f: impl FnOnce()) -> bool {
    unsafe { feclearexcept(FE_OVERFLOW) };
    f();
    unsafe { fetestexcept(FE_OVERFLOW) != 0 }
}

fn coupled_model() -> Array2<f64> {
    array![[1.0, 1e-160, 0.0], [1e-160, 2.0, 0.5], [0.0, 0.5, 3.0]]
}

#[test]
fn ts_bfgs_update_with_a_1e_160_coupling_raises_no_overflow() {
    let mut b = coupled_model();
    let s = array![1e-2, 0.0, 0.0];
    let y = array![1e-2, 0.0, 0.0];
    let overflow = raises_overflow(|| ts_bfgs_update(&mut b, &s, &y));
    assert!(!overflow, "TS-BFGS raised FE_OVERFLOW: {b:?}");
    assert!(b.iter().all(|v| v.is_finite()), "{b:?}");
    let bs = b.dot(&s);
    for i in 0..3 {
        assert!((bs[i] - y[i]).abs() <= 1e-15, "secant B s = {bs:?}");
        for k in 0..3 {
            assert_eq!(b[(i, k)], b[(k, i)], "{b:?}");
        }
    }
}

#[test]
fn rfo_step_with_a_1e_160_gradient_raises_no_overflow() {
    let h = Array2::from_diag(&array![1.0, 2.0, 3.0]);
    let g = array![1e-160, 0.5, 0.0];
    let mut s = Array1::zeros(3);
    let overflow = raises_overflow(|| s = rfo_get_s(&h, &g, 0, 1.0));
    assert!(!overflow, "RFO raised FE_OVERFLOW: {s:?}");
    // The lowest eigenvalue of [[2, 0.5], [0.5, 0]] is 1 - sqrt(1.25),
    // so the step along the second mode is -0.5 / (1 + sqrt(1.25)).
    let want = -0.5 / (1.0 + 1.25_f64.sqrt());
    assert!((s[1] - want).abs() <= 1e-15, "RFO step {s:?}");
    assert!(s[0].abs() <= 1e-159 && s[2] == 0.0, "RFO step {s:?}");
}
