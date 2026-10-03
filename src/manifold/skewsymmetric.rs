//! Euclidean space of real skew-symmetric matrices.
//! manopt `skewsymmetricfactory`.
//!
//! A point is an `n x n` skew-symmetric matrix packed row-major as
//! length `n^2` (manopt stores the full matrix, not the `n(n-1)/2`
//! unique entries). The geometry is the linear subspace of
//! skew-symmetric matrices in the Frobenius metric. Projection is
//! `multiskew` (`0.5 (A - A^T)`). Retraction is `X + U`. Transport
//! is the identity. A 3N cluster is not this packing. `n >= 2`.
//!
//! `k > 1` (a product of blocks) is a different factory.

use ndarray::Array1;

use crate::vecops;

use super::Manifold;

/// Real skew-symmetric `n x n` matrices. Packed row-major, length `n^2`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SkewSymmetric;

/// Side length if `len` is a perfect square of an integer `n >= 2`.
pub fn side(len: usize) -> Option<usize> {
    if len < 4 {
        return None;
    }
    let n = (len as f64).sqrt().round() as usize;
    if n >= 2 && n.checked_mul(n) == Some(len) {
        Some(n)
    } else {
        None
    }
}

/// Split a length-n² ambient vector into (n, row-major entries).
pub fn unpack(x: &Array1<f64>) -> Option<(usize, Vec<f64>)> {
    let n = side(x.len())?;
    Some((n, x.iter().copied().collect()))
}

/// Flatten a row-major n-by-n matrix into the ambient vector.
pub fn pack(n: usize, a: Vec<f64>) -> Array1<f64> {
    Array1::from_shape_vec(n * n, a).unwrap()
}

/// Frobenius inner product. manopt `M.inner = d1(:).'*d2(:)`.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt `M.typicaldist = sqrt(k)*n` with `k = 1`.
pub fn typical_dist(n: usize) -> f64 {
    n as f64
}

/// `true` when the packed matrix is square (`n >= 2`) and skew-symmetric.
pub fn is_skewsymmetric(x: &Array1<f64>) -> bool {
    match unpack(x) {
        Some((n, a)) => is_skew_matrix(n, &a),
        None => false,
    }
}

fn is_skew_matrix(n: usize, a: &[f64]) -> bool {
    for i in 0..n {
        for j in 0..n {
            if (a[i * n + j] + a[j * n + i]).abs() > 1e-10 {
                return false;
            }
        }
    }
    true
}

/// manopt `multiskew`: `0.5 (A - A^T)`.
fn multiskew(n: usize, a: &[f64]) -> Vec<f64> {
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            s[i * n + j] = 0.5 * (a[i * n + j] - a[j * n + i]);
        }
    }
    s
}

impl Manifold for SkewSymmetric {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if side(n).is_some() { Ok(()) } else { Err(n) }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let Some((n, eta)) = unpack(v) else {
            return v.clone();
        };
        if x.len() != v.len() {
            return v.clone();
        }
        pack(n, multiskew(n, &eta))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let Some((n, _)) = unpack(x) else {
            return x + v;
        };
        if v.len() != x.len() {
            return x + v;
        }
        let mut y = x.clone();
        vecops::axpy(1.0, v.view(), &mut y);
        let flat: Vec<f64> = y.iter().copied().collect();
        pack(n, multiskew(n, &flat))
    }

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        v.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_stays_on_the_skew_set() {
        let x = array![0.0, 0.3, -0.3, 0.0];
        let v = array![0.1, 0.4, -0.2, 0.05];
        let y = SkewSymmetric.retract(&x, &v);
        assert_eq!(y.len(), 4);
        assert!(is_skewsymmetric(&y), "left the skew-symmetric set {y:?}");
        assert!((y[0]).abs() < 1e-15, "nonzero diagonal {y:?}");
        assert!((y[3]).abs() < 1e-15, "nonzero diagonal {y:?}");
        assert!((y[1] + y[2]).abs() < 1e-15, "not skew {y:?}");
        assert_eq!(side(y.len()), Some(2));
    }

    #[test]
    fn retract_stays_on_so3_algebra() {
        // n=3 packing: row-major so(3). Start off the set.
        let x = array![0.1, 0.2, 0.3, -0.2, 0.0, 0.4, -0.3, -0.4, 0.05];
        let v = array![0.2, 0.1, -0.3, 0.4, 0.1, 0.2, 0.0, -0.1, -0.2];
        let y = SkewSymmetric.retract(&x, &v);
        assert_eq!(y.len(), 9);
        assert!(is_skewsymmetric(&y), "left so(3) {y:?}");
        assert_eq!(side(y.len()), Some(3));
        for i in 0..3 {
            assert!(y[i * 3 + i].abs() < 1e-15, "diag {y:?}");
        }
    }

    #[test]
    fn retract_is_translation_then_skew() {
        let x = array![0.0, 1.0, -1.0, 0.0];
        let v = array![0.0, 0.2, -0.2, 0.0];
        let y = SkewSymmetric.retract(&x, &v);
        assert!((y[0]).abs() < 1e-15, "{y:?}");
        assert!((y[1] - 1.2).abs() < 1e-15, "{y:?}");
        assert!((y[2] + 1.2).abs() < 1e-15, "{y:?}");
        assert!((y[3]).abs() < 1e-15, "{y:?}");
    }

    #[test]
    fn project_kills_the_symmetric_part() {
        let x = array![0.0, 1.0, -1.0, 0.0];
        let v = array![0.3, 1.0, -0.4, 0.5];
        let t = SkewSymmetric.project(&x, &v);
        assert!((t[0]).abs() < 1e-15);
        assert!((t[3]).abs() < 1e-15);
        assert!((t[1] + t[2]).abs() < 1e-15);
        assert!((t[1] - 0.7).abs() < 1e-15);
        assert!((t[2] + 0.7).abs() < 1e-15);
    }

    #[test]
    fn transport_of_a_tangent_is_itself() {
        let x = array![0.0, 0.1, -0.1, 0.0];
        let y = array![0.0, 2.0, -2.0, 0.0];
        let v = array![0.0, 0.25, -0.25, 0.0];
        let t = SkewSymmetric.transport(&x, &y, &v);
        for i in 0..4 {
            assert!((t[i] - v[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn frobenius_inner_and_typical_dist() {
        let u = array![0.0, 0.5, -0.5, 0.0];
        let v = array![0.0, 1.0, -1.0, 0.0];
        assert!((inner(&u, &v) - 1.0).abs() < 1e-15);
        assert!((typical_dist(3) - 3.0).abs() < 1e-15);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let x = array![0.0, 1.0, 2.0, -1.0, 0.0, 3.0, -2.0, -3.0, 0.0];
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
        let y = SkewSymmetric.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(SkewSymmetric.project(&x, &v).len(), 114);
        assert!(SkewSymmetric.required_dim(114).is_err());
        assert!(SkewSymmetric.required_dim(6).is_err());
        assert!(SkewSymmetric.required_dim(1).is_err());
        assert!(SkewSymmetric.required_dim(0).is_err());
        assert!(SkewSymmetric.required_dim(4).is_ok());
        assert!(SkewSymmetric.required_dim(9).is_ok());
    }

    #[test]
    fn not_the_sphere_and_not_symmetric() {
        let x = array![0.0, 2.0, -2.0, 0.0];
        let v = array![0.0, 0.5, -0.5, 0.0];
        let y = SkewSymmetric.retract(&x, &v);
        let fro2: f64 = y.iter().map(|a| a * a).sum();
        assert!((fro2 - 1.0).abs() > 1.0, "must not be a unit sphere {y:?}");
        assert!(
            (y[1] - y[2]).abs() > 1.0,
            "must not land on the symmetric set {y:?}"
        );
        assert_ne!(
            crate::manifold::ManifoldKind::SkewSymmetric,
            crate::manifold::ManifoldKind::Symmetric
        );
        assert_ne!(
            crate::manifold::ManifoldKind::SkewSymmetric,
            crate::manifold::ManifoldKind::Sphere
        );
    }
}
