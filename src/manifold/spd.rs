//! Symmetric positive definite matrices, affine-invariant metric.
//!
//! manopt `sympositivedefinitefactory`: an `n x n` point packed
//! row-major as length `n^2`. Tangent vectors are symmetric. Projection
//! is symmetrization. The second-order retraction is
//! `Y = symm(X + U + (1/2) U X^{-1} U)` (Boumal second-order). Transport
//! is projection at the arrival point; on a tangent that is the
//! identity used by the MATLAB factory.
//!
//! A 3N cluster is not this packing. Length must be a square.

use ndarray::{Array1, ArrayView1};

use crate::vecops;

use super::Manifold;

/// Affine-invariant SPD cone. Packed row-major, length `n^2`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Spd;

/// Side length if `len` is a positive perfect square.
pub fn side(len: usize) -> Option<usize> {
    matrix_order(len)
}

fn matrix_order(len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let n = (len as f64).sqrt().round() as usize;
    if n.checked_mul(n) == Some(len) {
        Some(n)
    } else {
        None
    }
}

/// Split a length-n² ambient vector into (n, row-major entries).
pub fn unpack(x: &Array1<f64>) -> Option<(usize, Vec<f64>)> {
    let n = matrix_order(x.len())?;
    Some((n, x.iter().copied().collect()))
}

/// Flatten a row-major n-by-n matrix into the ambient vector.
pub fn pack(n: usize, a: Vec<f64>) -> Array1<f64> {
    Array1::from_shape_vec(n * n, a).unwrap()
}

fn symm(n: usize, a: &[f64]) -> Vec<f64> {
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            s[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }
    s
}

fn prefix_dot(a: &[f64], b: &[f64]) -> f64 {
    vecops::dot(ArrayView1::from(a), ArrayView1::from(b))
}

fn cholesky(n: usize, a: &[f64]) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            if j > 0 {
                s -= prefix_dot(&l[i * n..i * n + j], &l[j * n..j * n + j]);
            }
            if i == j {
                if !(s > 1e-18) {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

fn factor_spd(n: usize, x: &[f64]) -> Option<Vec<f64>> {
    let xs = symm(n, x);
    let mut ridge = 0.0;
    for _ in 0..40 {
        let mut a = xs.clone();
        if ridge > 0.0 {
            for i in 0..n {
                a[i * n + i] += ridge;
            }
        }
        if let Some(l) = cholesky(n, &a) {
            return Some(l);
        }
        ridge = if ridge == 0.0 { 1e-14 } else { ridge * 4.0 };
    }
    None
}

fn solve_chol(n: usize, l: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for col in 0..n {
        let mut y: Vec<f64> = (0..n).map(|i| rhs[i * n + col]).collect();
        for i in 0..n {
            let s = if i == 0 {
                y[0]
            } else {
                y[i] - prefix_dot(&l[i * n..i * n + i], &y[..i])
            };
            y[i] = s / l[i * n + i];
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            if i + 1 < n {
                let mut lcol = Vec::with_capacity(n - i - 1);
                let mut xtail = Vec::with_capacity(n - i - 1);
                for k in (i + 1)..n {
                    lcol.push(l[k * n + i]);
                    xtail.push(x[k]);
                }
                s -= prefix_dot(&lcol, &xtail);
            }
            x[i] = s / l[i * n + i];
        }
        for i in 0..n {
            out[i * n + col] = x[i];
        }
    }
    out
}

fn matmul(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for j in 0..n {
        let col: Vec<f64> = (0..n).map(|k| b[k * n + j]).collect();
        let col_view = ArrayView1::from(col.as_slice());
        for i in 0..n {
            c[i * n + j] = vecops::dot(ArrayView1::from(&a[i * n..(i + 1) * n]), col_view);
        }
    }
    c
}

/// `true` when the packed matrix is symmetric and Cholesky succeeds.
pub fn is_spd(x: &Array1<f64>) -> bool {
    match unpack(x) {
        Some((n, a)) => is_spd_matrix(n, &a),
        None => false,
    }
}

fn is_spd_matrix(n: usize, a: &[f64]) -> bool {
    for i in 0..n {
        for j in 0..n {
            if (a[i * n + j] - a[j * n + i]).abs() > 1e-10 {
                return false;
            }
        }
    }
    cholesky(n, a).is_some()
}

impl Manifold for Spd {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if side(n).is_some() {
            Ok(())
        } else {
            Err(n)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let Some((n, eta)) = unpack(v) else {
            return v.clone();
        };
        if x.len() != v.len() {
            return v.clone();
        }
        pack(n, symm(n, &eta))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let Some((n, xm)) = unpack(x) else {
            return x + v;
        };
        if v.len() != x.len() {
            return x + v;
        }
        let eta = v.iter().copied().collect::<Vec<_>>();
        let Some(l) = factor_spd(n, &xm) else {
            let fallback: Vec<f64> = x.iter().zip(v.iter()).map(|(a, b)| a + b).collect();
            return pack(n, symm(n, &fallback));
        };
        let z = solve_chol(n, &l, &eta);
        let mid = matmul(n, &eta, &z);
        let mut y_arr = pack(n, xm);
        vecops::axpy(1.0, v.view(), &mut y_arr);
        vecops::axpy(0.5, ArrayView1::from(mid.as_slice()), &mut y_arr);
        let y_flat: Vec<f64> = y_arr.iter().copied().collect();
        pack(n, symm(n, &y_flat))
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_stays_on_the_spd_set() {
        let x = array![2.0, 0.2, 0.2, 3.0];
        let v = array![0.0, 0.15, 0.15, -0.1];
        let y = Spd.retract(&x, &v);
        assert_eq!(y.len(), 4);
        assert!((y[1] - y[2]).abs() < 1e-12, "not symmetric {y:?}");
        assert!(is_spd(&y), "left the SPD set {y:?}");
        assert_eq!(side(y.len()), Some(2));
    }

    #[test]
    fn identity_plus_symmetric_step_matches_closed_form() {
        let x = array![1.0, 0.0, 0.0, 1.0];
        let v = array![0.0, 0.2, 0.2, 0.0];
        let y = Spd.retract(&x, &v);
        // Y = I + U + (1/2) U^2, U = [[0, 0.2], [0.2, 0]]
        assert!((y[0] - 1.02).abs() < 1e-12, "{y:?}");
        assert!((y[1] - 0.2).abs() < 1e-12, "{y:?}");
        assert!((y[2] - 0.2).abs() < 1e-12, "{y:?}");
        assert!((y[3] - 1.02).abs() < 1e-12, "{y:?}");
        let (n, ym) = unpack(&y).unwrap();
        assert!(is_spd_matrix(n, &ym));
    }

    #[test]
    fn scalar_spd_stays_positive() {
        let x = array![2.0];
        let v = array![0.4];
        let y = Spd.retract(&x, &v);
        // 2 + 0.4 + 0.5 * 0.4 * (0.4 / 2) = 2.44
        assert!((y[0] - 2.44).abs() < 1e-12, "{y:?}");
        assert!(y[0] > 0.0);
    }

    #[test]
    fn project_is_symmetric() {
        let x = array![1.0, 0.0, 0.0, 2.0];
        let v = array![0.3, 1.0, -0.4, 0.5];
        let t = Spd.project(&x, &v);
        assert!((t[1] - t[2]).abs() < 1e-15);
        assert!((t[1] - 0.3).abs() < 1e-15);
        assert!((t[0] - 0.3).abs() < 1e-15);
        assert!((t[3] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn transport_of_a_tangent_is_itself() {
        let x = array![1.0, 0.1, 0.1, 2.0];
        let y = array![2.0, 0.0, 0.0, 2.0];
        let v = array![0.0, 0.25, 0.25, -0.1];
        let t = Spd.transport(&x, &y, &v);
        for i in 0..4 {
            assert!((t[i] - v[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn three_by_three_stays_spd_not_so3() {
        let x = array![2.0, 0.1, 0.0, 0.1, 3.0, 0.2, 0.0, 0.2, 4.0];
        let v = array![0.0, 0.05, 0.0, 0.05, 0.0, 0.1, 0.0, 0.1, 0.0];
        let y = Spd.retract(&x, &v);
        assert_eq!(y.len(), 9);
        assert!(is_spd(&y), "left the SPD set {y:?}");
        // Not a rotation: Frobenius ||Y||^2 is far from 3.
        let fro2: f64 = y.iter().map(|a| a * a).sum();
        assert!((fro2 - 3.0).abs() > 1.0, "must not be SO(3) {y:?}");
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let x = array![4.0, 1.0, 2.0, 1.0, 5.0, 0.0, 2.0, 0.0, 6.0];
        let (n, m) = unpack(&x).unwrap();
        assert_eq!(n, 3);
        let y = pack(n, m);
        for i in 0..9 {
            assert!((x[i] - y[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn wrong_dim_does_not_shrink() {
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = Spd.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(Spd.project(&x, &v).len(), 114);
        assert!(Spd.required_dim(114).is_err());
        assert!(Spd.required_dim(6).is_err());
        assert!(Spd.required_dim(4).is_ok());
        assert!(Spd.required_dim(9).is_ok());
        assert!(Spd.required_dim(1).is_ok());
    }
}
