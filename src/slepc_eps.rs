//! Feature-gated SLEPc EPS arm. Linked only when `rgmin_has_slepc`.

use ndarray::Array1;
#[cfg(rgmin_has_slepc)]
use std::cell::Cell;
#[cfg(rgmin_has_slepc)]
use std::ffi::c_void;
#[cfg(rgmin_has_slepc)]
use std::os::raw::c_int;
#[cfg(rgmin_has_slepc)]
use std::sync::Mutex;

use crate::error::{Error, Result};
use crate::slepc_kind::SlepcParams;
#[cfg(rgmin_has_slepc)]
use crate::vecops::nrm2;

#[cfg(rgmin_has_slepc)]
static SLEPC_LOCK: Mutex<()> = Mutex::new(());

#[cfg(rgmin_has_slepc)]
type HessApply = unsafe extern "C" fn(*mut c_void, i64, *const f64, *mut f64) -> c_int;

#[cfg(rgmin_has_slepc)]
unsafe extern "C" {
    fn rgmin_slepc_lowest(
        n: i64,
        seed: *const f64,
        nev: i64,
        ncv: i64,
        maxit: i64,
        tol: f64,
        pmat: *mut c_void,
        st_kind: i32,
        out_vec: *mut f64,
        out_value: *mut f64,
        out_actions: *mut i64,
        user: *mut c_void,
        apply: HessApply,
    ) -> c_int;
}

#[cfg(rgmin_has_slepc)]
struct ApplyCtx<'a> {
    apply: &'a dyn Fn(&[f64]) -> Array1<f64>,
    n: usize,
    actions: Cell<usize>,
}

#[cfg(rgmin_has_slepc)]
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
    let out = (ctx.apply)(vv);
    if out.len() != n {
        return 1;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(out.as_ptr(), hv, n);
    }
    ctx.actions.set(ctx.actions.get() + 1);
    0
}

/// SLEPc EPS on a Hessian MatShell. Unlinked builds stay unavailable.
pub(crate) fn solve<F>(
    seed: &[f64],
    nev: usize,
    ncv: usize,
    maxit: usize,
    tol: f64,
    slepc: &SlepcParams,
    apply: F,
) -> Result<(Array1<f64>, f64, usize)>
where
    F: Fn(&[f64]) -> Array1<f64>,
{
    #[cfg(not(rgmin_has_slepc))]
    {
        let _ = (seed, nev, ncv, maxit, tol, slepc, apply);
        Err(Error::EigenUnavailable { kind: "slepc" })
    }
    #[cfg(rgmin_has_slepc)]
    {
        if slepc.st.needs_pmat() && slepc.pmat.is_none() {
            return Err(Error::Slepc {
                what: "sinvert/cayley needs a host Pmat",
            });
        }
        linked_solve(seed, nev, ncv, maxit, tol, slepc, apply)
    }
}

#[cfg(rgmin_has_slepc)]
fn linked_solve<F>(
    seed: &[f64],
    nev: usize,
    ncv: usize,
    maxit: usize,
    tol: f64,
    slepc: &SlepcParams,
    apply: F,
) -> Result<(Array1<f64>, f64, usize)>
where
    F: Fn(&[f64]) -> Array1<f64>,
{
    let n = seed.len();
    if n == 0 {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    let _guard = SLEPC_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let ctx = ApplyCtx {
        apply: &apply,
        n,
        actions: Cell::new(0),
    };
    let mut out = Array1::<f64>::zeros(n);
    let mut value = 0.0_f64;
    let mut actions: i64 = 0;
    let pmat = slepc
        .pmat
        .map(|p| p.as_raw())
        .unwrap_or(std::ptr::null_mut());
    let rc = unsafe {
        rgmin_slepc_lowest(
            n as i64,
            seed.as_ptr(),
            nev.max(1) as i64,
            ncv as i64,
            maxit as i64,
            tol,
            pmat,
            i32::from(slepc.st as u8),
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
        2 => Err(Error::Slepc {
            what: "no converged pair",
        }),
        4 => Err(Error::EigenUnavailable { kind: "slepc" }),
        _ => Err(Error::Slepc {
            what: "EPSSolve failed",
        }),
    }
}
