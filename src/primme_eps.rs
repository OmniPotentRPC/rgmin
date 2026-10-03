//! Feature-gated PRIMME arm. Linked only when `rgmin_has_primme`.

use ndarray::Array1;
#[cfg(rgmin_has_primme)]
use std::cell::Cell;
#[cfg(rgmin_has_primme)]
use std::ffi::c_void;
#[cfg(rgmin_has_primme)]
use std::os::raw::c_int;

use crate::error::{Error, Result};
#[cfg(rgmin_has_primme)]
use crate::vecops::nrm2;

#[cfg(rgmin_has_primme)]
type HessApply = unsafe extern "C" fn(*mut c_void, i64, *const f64, *mut f64) -> c_int;

#[cfg(rgmin_has_primme)]
unsafe extern "C" {
    fn rgmin_primme_lowest(
        n: i64,
        seed: *const f64,
        nev: i64,
        maxit: i64,
        tol: f64,
        out_vec: *mut f64,
        out_value: *mut f64,
        out_actions: *mut i64,
        user: *mut c_void,
        apply: HessApply,
        precond: Option<HessApply>,
    ) -> c_int;
}

#[cfg(rgmin_has_primme)]
struct ApplyCtx<'a> {
    apply: &'a dyn Fn(&[f64]) -> Array1<f64>,
    precond: Option<&'a dyn Fn(&[f64]) -> Array1<f64>>,
    n: usize,
    actions: Cell<usize>,
}

#[cfg(rgmin_has_primme)]
unsafe extern "C" fn apply_cb(user: *mut c_void, n: i64, v: *const f64, hv: *mut f64) -> c_int {
    unsafe { call_ctx(user, n, v, hv, false) }
}

#[cfg(rgmin_has_primme)]
unsafe extern "C" fn precond_cb(user: *mut c_void, n: i64, v: *const f64, hv: *mut f64) -> c_int {
    unsafe { call_ctx(user, n, v, hv, true) }
}

#[cfg(rgmin_has_primme)]
unsafe fn call_ctx(
    user: *mut c_void,
    n: i64,
    v: *const f64,
    hv: *mut f64,
    precond: bool,
) -> c_int {
    if user.is_null() || v.is_null() || hv.is_null() || n <= 0 {
        return 1;
    }
    let ctx = unsafe { &*(user as *const ApplyCtx<'_>) };
    let n = n as usize;
    if ctx.n != n {
        return 1;
    }
    let vv = unsafe { std::slice::from_raw_parts(v, n) };
    let out = if precond {
        match ctx.precond {
            Some(f) => f(vv),
            None => return 1,
        }
    } else {
        ctx.actions.set(ctx.actions.get() + 1);
        (ctx.apply)(vv)
    };
    if out.len() != n {
        return 1;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(out.as_ptr(), hv, n);
    }
    0
}

/// PRIMME on a frozen Hessian action. Unlinked builds stay unavailable.
pub(crate) fn solve<F>(
    seed: &[f64],
    nev: usize,
    maxit: usize,
    tol: f64,
    apply: F,
    precond: Option<&dyn Fn(&[f64]) -> Array1<f64>>,
) -> Result<(Array1<f64>, f64, usize)>
where
    F: Fn(&[f64]) -> Array1<f64>,
{
    #[cfg(not(rgmin_has_primme))]
    {
        let _ = (seed, nev, maxit, tol, apply, precond);
        Err(Error::EigenUnavailable { kind: "primme" })
    }
    #[cfg(rgmin_has_primme)]
    {
        linked_solve(seed, nev, maxit, tol, apply, precond)
    }
}

#[cfg(rgmin_has_primme)]
fn linked_solve<F>(
    seed: &[f64],
    nev: usize,
    maxit: usize,
    tol: f64,
    apply: F,
    precond: Option<&dyn Fn(&[f64]) -> Array1<f64>>,
) -> Result<(Array1<f64>, f64, usize)>
where
    F: Fn(&[f64]) -> Array1<f64>,
{
    let n = seed.len();
    if n == 0 {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    let use_t = precond.is_some();
    let ctx = ApplyCtx {
        apply: &apply,
        precond,
        n,
        actions: Cell::new(0),
    };
    let mut out = Array1::<f64>::zeros(n);
    let mut value = 0.0_f64;
    let mut actions: i64 = 0;
    let precond_ptr: Option<HessApply> = if use_t { Some(precond_cb) } else { None };
    let rc = unsafe {
        rgmin_primme_lowest(
            n as i64,
            seed.as_ptr(),
            nev.max(1) as i64,
            maxit as i64,
            tol,
            out.as_mut_ptr(),
            &mut value,
            &mut actions,
            (&raw const ctx).cast::<c_void>().cast_mut(),
            apply_cb,
            precond_ptr,
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
        2 => Err(Error::Primme {
            what: "no converged pair",
        }),
        _ => Err(Error::Primme {
            what: "dprimme failed",
        }),
    }
}
