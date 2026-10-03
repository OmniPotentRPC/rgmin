//! Feature-gated ChASE arm. Linked only when `rgmin_has_chase`.
//! Assembled dense H only. ApplyHessian never enters this file.

use ndarray::{Array1, ArrayView2};

use crate::error::{Error, Result};
#[cfg(rgmin_has_chase)]
use crate::vecops::nrm2;

#[cfg(rgmin_has_chase)]
unsafe extern "C" {
    fn rgmin_chase_lowest(
        n: i64,
        h: *const f64,
        seed: *const f64,
        nev: i64,
        extra: i64,
        degree: i64,
        tol: f64,
        out_vec: *mut f64,
        out_value: *mut f64,
    ) -> i32;
}

/// ChASE on an assembled symmetric `H`. Unlinked builds stay unavailable.
pub(crate) fn solve(
    h: ArrayView2<f64>,
    seed: &[f64],
    nev: usize,
    extra: usize,
    degree: usize,
    tol: f64,
) -> Result<(Array1<f64>, f64, usize)> {
    #[cfg(not(rgmin_has_chase))]
    {
        let _ = (h, seed, nev, extra, degree, tol);
        Err(Error::EigenUnavailable { kind: "chase" })
    }
    #[cfg(rgmin_has_chase)]
    {
        linked_solve(h, seed, nev, extra, degree, tol)
    }
}

#[cfg(rgmin_has_chase)]
fn linked_solve(
    h: ArrayView2<f64>,
    seed: &[f64],
    nev: usize,
    extra: usize,
    degree: usize,
    tol: f64,
) -> Result<(Array1<f64>, f64, usize)> {
    let n = seed.len();
    if n == 0 || h.nrows() != n || h.ncols() != n {
        return Err(Error::Dim {
            got: seed.len(),
            dim: h.nrows(),
        });
    }
    let Some(h_slice) = h.as_slice() else {
        return Err(Error::Chase {
            what: "H is not contiguous",
        });
    };
    let mut out = Array1::<f64>::zeros(n);
    let mut value = 0.0_f64;
    let rc = unsafe {
        rgmin_chase_lowest(
            n as i64,
            h_slice.as_ptr(),
            seed.as_ptr(),
            nev.max(1) as i64,
            extra.max(1) as i64,
            degree.max(1) as i64,
            tol,
            out.as_mut_ptr(),
            &mut value,
        )
    };
    if rc != 0 {
        return Err(Error::Chase {
            what: "dchase failed",
        });
    }
    if !value.is_finite() || !out.iter().all(|c| c.is_finite()) {
        return Err(Error::Chase {
            what: "no finite pair",
        });
    }
    let nrm = nrm2(out.view());
    if nrm > 1e-14 {
        out.mapv_inplace(|c| c / nrm);
    }
    Ok((out, value, 0))
}
