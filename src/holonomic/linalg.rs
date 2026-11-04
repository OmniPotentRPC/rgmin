//! Dense kernels for one block of coupled constraints. Blocks are the
//! connected components of the constraint graph, so these matrices are
//! as small as the molecules they come from.

use super::U;

/// Extreme eigenvalues of the symmetric row-major `n x n` matrix `a` by
/// classical Jacobi rotations, largest off-diagonal entry first.
///
/// Each rotation removes `2 a_pq^2` from `off(a)^2`, and the largest entry
/// is at least the mean, so `off^2` shrinks by `1 - 1 / N` per rotation
/// with `N = n (n - 1) / 2` (Golub and Van Loan, Matrix Computations,
/// section 8.5). Stopping at `off <= U ||a||_F` leaves each eigenvalue
/// within `U ||a||_F` (Weyl), which `N ln(off_0^2 / (U ||a||_F)^2)`
/// rotations reach.
pub(super) fn extreme_eigenvalues(mut a: Vec<f64>, n: usize) -> (f64, f64) {
    let off = |a: &[f64]| {
        let mut s = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                s += 2.0 * a[p * n + q] * a[p * n + q];
            }
        }
        s
    };
    let frob = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let target = (U * frob).powi(2);
    let pairs = (n * n.saturating_sub(1) / 2).max(1) as f64;
    let off0 = off(&a);
    let budget = if off0 > target {
        (pairs * (off0 / target).ln()).ceil() as usize
    } else {
        0
    };
    for _ in 0..budget {
        if off(&a) <= target {
            break;
        }
        let (mut p, mut q, mut big) = (0, 1, -1.0);
        for i in 0..n {
            for j in i + 1..n {
                if a[i * n + j].abs() > big {
                    (p, q, big) = (i, j, a[i * n + j].abs());
                }
            }
        }
        rotate(&mut a, n, p, q);
    }
    let diag = (0..n).map(|k| a[k * n + k]);
    diag.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), d| {
        (lo.min(d), hi.max(d))
    })
}

/// One Jacobi rotation that zeroes `a[p, q]` (Golub and Van Loan,
/// algorithm 8.5.1, the stable choice of the tangent).
fn rotate(a: &mut [f64], n: usize, p: usize, q: usize) {
    let apq = a[p * n + q];
    if apq == 0.0 {
        return;
    }
    let tau = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
    let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
    let t = if tau == 0.0 { 1.0 } else { t };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    for k in 0..n {
        let (akp, akq) = (a[k * n + p], a[k * n + q]);
        a[k * n + p] = c * akp - s * akq;
        a[k * n + q] = s * akp + c * akq;
    }
    for k in 0..n {
        let (apk, aqk) = (a[p * n + k], a[q * n + k]);
        a[p * n + k] = c * apk - s * aqk;
        a[q * n + k] = s * apk + c * aqk;
    }
}

/// Lower Cholesky factor of the symmetric positive definite row-major
/// `a`, or `None` at a non-positive pivot.
pub(super) fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let s: f64 = (0..j).map(|k| l[j * n + k] * l[j * n + k]).sum();
        let pivot = a[j * n + j] - s;
        if pivot.is_nan() || pivot <= 0.0 {
            return None;
        }
        let d = pivot.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            l[i * n + j] = (a[i * n + j] - s) / d;
        }
    }
    Some(l)
}

/// Solve `L L^T x = b` in place.
pub(super) fn cholesky_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i * n + k] * b[k]).sum();
        b[i] = (b[i] - s) / l[i * n + i];
    }
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k * n + i] * b[k]).sum();
        b[i] = (b[i] - s) / l[i * n + i];
    }
}

/// Gaussian elimination with partial pivoting on the row-major `a`.
/// A zero pivot leaves a non-finite solution, returned as `None`.
pub(super) fn lu_solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))?;
        (0..n).for_each(|c| a.swap(k * n + c, p * n + c));
        b.swap(k, p);
        for i in k + 1..n {
            let f = a[i * n + k] / a[k * n + k];
            (k..n).for_each(|c| a[i * n + c] -= f * a[k * n + c]);
            b[i] -= f * b[k];
        }
    }
    for k in (0..n).rev() {
        let s: f64 = (k + 1..n).map(|c| a[k * n + c] * b[c]).sum();
        b[k] = (b[k] - s) / a[k * n + k];
    }
    b.iter().all(|v| v.is_finite()).then_some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_matches_closed_form_two_by_two() {
        let (lo, hi) = extreme_eigenvalues(vec![2.0, 1.0, 1.0, 2.0], 2);
        assert!((lo - 1.0).abs() <= 4.0 * U && (hi - 3.0).abs() <= 8.0 * U);
    }

    #[test]
    fn jacobi_on_the_water_coupling() {
        // Unit diagonal with every off-diagonal c: eigenvalues 1 + 2c and
        // 1 - c (twice).
        let c = -0.3;
        let a = vec![1.0, c, c, c, 1.0, c, c, c, 1.0];
        let (lo, hi) = extreme_eigenvalues(a, 3);
        assert!((lo - (1.0 + 2.0 * c)).abs() <= 16.0 * U);
        assert!((hi - (1.0 - c)).abs() <= 16.0 * U);
    }

    #[test]
    fn cholesky_solves_and_refuses_indefinite() {
        let a = [4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2).unwrap();
        let mut b = [2.0, 1.0];
        cholesky_solve(&l, 2, &mut b);
        assert!((4.0 * b[0] + 2.0 * b[1] - 2.0).abs() <= 8.0 * U);
        assert!((2.0 * b[0] + 3.0 * b[1] - 1.0).abs() <= 8.0 * U);
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
    }

    #[test]
    fn lu_reports_a_singular_matrix() {
        assert!(lu_solve(vec![1.0, 2.0, 2.0, 4.0], vec![1.0, 1.0]).is_none());
        let x = lu_solve(vec![0.0, 1.0, 1.0, 0.0], vec![2.0, 3.0]).unwrap();
        assert_eq!(x, vec![3.0, 2.0]);
    }
}
