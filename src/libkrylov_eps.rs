//! Feature-gated libkrylov arm. Linked only when `rgmin_has_libkrylov`.

use ndarray::Array1;
#[cfg(rgmin_has_libkrylov)]
use std::cell::Cell;
#[cfg(rgmin_has_libkrylov)]
use std::ffi::c_void;
#[cfg(rgmin_has_libkrylov)]
use std::os::raw::c_int;

use crate::error::{Error, Result};
#[cfg(rgmin_has_libkrylov)]
use crate::vecops::nrm2;

#[cfg(rgmin_has_libkrylov)]
type HessApply = unsafe extern "C" fn(*mut c_void, i64, *const f64, *mut f64) -> c_int;

#[cfg(rgmin_has_libkrylov)]
unsafe extern "C" {
    fn rgmin_libkrylov_lowest(
        n: i64,
        seed: *const f64,
        nev: i64,
        maxit: i64,
        krylov: i64,
        tol: f64,
        out_vec: *mut f64,
        out_value: *mut f64,
        out_actions: *mut i64,
        user: *mut c_void,
        apply: HessApply,
    ) -> c_int;
}

#[cfg(rgmin_has_libkrylov)]
struct ApplyCtx<'a> {
    apply: &'a dyn Fn(&[f64]) -> Array1<f64>,
    n: usize,
    actions: Cell<usize>,
}

#[cfg(rgmin_has_libkrylov)]
unsafe extern "C" fn apply_cb(user: *mut c_void, n: i64, v: *const f64, hv: *mut f64) -> c_int {
    if user.is_null() || v.is_null() || hv.is_null() || n <= 0 {
        return 1;
    }
    let ctx = unsafe { &*(user as *const ApplyCtx<'_>) };
    let n = n as usize;
    if ctx.n != n {
        return 1;
    }
    let vv = unsafe { std::slice::from_raw_parts(v, n) };
    ctx.actions.set(ctx.actions.get() + 1);
    let out = (ctx.apply)(vv);
    if out.len() != n {
        return 1;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(out.as_ptr(), hv, n);
    }
    0
}

/// libkrylov on a frozen Hessian action. Unlinked builds stay unavailable.
pub(crate) fn solve<F>(
    seed: &[f64],
    nev: usize,
    maxit: usize,
    krylov: usize,
    tol: f64,
    apply: F,
) -> Result<(Array1<f64>, f64, usize)>
where
    F: Fn(&[f64]) -> Array1<f64>,
{
    #[cfg(not(rgmin_has_libkrylov))]
    {
        let _ = (seed, nev, maxit, krylov, tol, apply);
        Err(Error::EigenUnavailable { kind: "libkrylov" })
    }
    #[cfg(rgmin_has_libkrylov)]
    {
        linked_solve(seed, nev, maxit, krylov, tol, apply)
    }
}

#[cfg(rgmin_has_libkrylov)]
fn linked_solve<F>(
    seed: &[f64],
    nev: usize,
    maxit: usize,
    krylov: usize,
    tol: f64,
    apply: F,
) -> Result<(Array1<f64>, f64, usize)>
where
    F: Fn(&[f64]) -> Array1<f64>,
{
    let n = seed.len();
    if n == 0 {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    let ctx = ApplyCtx {
        apply: &apply,
        n,
        actions: Cell::new(0),
    };
    let mut out = Array1::<f64>::zeros(n);
    let mut value = 0.0_f64;
    let mut actions: i64 = 0;
    let rc = unsafe {
        rgmin_libkrylov_lowest(
            n as i64,
            seed.as_ptr(),
            nev.max(1) as i64,
            maxit as i64,
            krylov.max(1) as i64,
            tol,
            out.as_mut_ptr(),
            &mut value,
            &mut actions,
            (&raw const ctx).cast::<c_void>().cast_mut(),
            apply_cb,
        )
    };
    match rc {
        0 => {
            let actions = if actions > 0 {
                actions as usize
            } else {
                ctx.actions.get()
            };
            let nrm = nrm2(out.view());
            if nrm > 1e-14 {
                out.mapv_inplace(|c| c / nrm);
            }
            Ok((out, value, actions))
        }
        1 => Err(Error::Dim { got: n, dim: n }),
        2 => Err(Error::Libkrylov {
            what: "no converged pair",
        }),
        _ => Err(Error::Libkrylov {
            what: "ckrylov_solve_real_equation failed",
        }),
    }
}
