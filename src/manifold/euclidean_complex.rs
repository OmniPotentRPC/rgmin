//! Complex Euclidean space \(\mathbb{C}^n\) identified with \(\mathbb{R}^{2n}\).
//! manopt `euclideancomplexfactory`.
//!
//! A point is `n` complex entries packed as interleaved `(re, im)`
//! pairs, length `2 n`. The geometry is the real identification of
//! \(\mathbb{C}^n\) with \(\mathbb{R}^{2n}\): the inner product is
//! manopt `real(d1(:)'*d2(:))`, which on this packing is the real
//! Euclidean product. Projection is the identity. Retraction is
//! `x + v`. Transport is the identity. This is not the sphere
//! \(S^{2n-1}\), not [`super::ComplexCircle`] (no unit-modulus
//! constraint), and not a 3N cluster.
//!
//! manopt also accepts a second size (`C^{m x n}`) and a size
//! vector. Here `n` is the number of complex entries (`C^n`, or
//! `C^{m x k}` flattened so `n = m k`). Pair reductions go through
//! [`crate::vecops`].

use ndarray::{Array1, ArrayView1};

use crate::vecops;

use super::Manifold;

/// \(\mathbb{C}^n\) as `n` interleaved real-imaginary pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EuclideanComplex {
    /// Number of complex entries. manopt `m` at `n = 1`.
    pub n: usize,
}

impl Default for EuclideanComplex {
    fn default() -> Self {
        Self { n: 1 }
    }
}

impl EuclideanComplex {
    /// `n` complex entries. Illegal `n == 0` fails [`Manifold::required_dim`].
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Packed length `2 n`, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.n.checked_mul(2)
    }

    fn fits(self, len: usize) -> bool {
        self.n >= 1 && self.packed_len() == Some(len)
    }

    /// Interleaved pack of real and imaginary parts (equal length).
    pub fn pack(re: ArrayView1<f64>, im: ArrayView1<f64>) -> Array1<f64> {
        let n = re.len().min(im.len());
        let mut out = Array1::zeros(2 * n);
        for k in 0..n {
            out[2 * k] = re[k];
            out[2 * k + 1] = im[k];
        }
        out
    }

    /// Split a packed vector into `(re, im)`. `None` if the length is odd.
    pub fn unpack(x: &Array1<f64>) -> Option<(Array1<f64>, Array1<f64>)> {
        if x.len() % 2 != 0 {
            return None;
        }
        let n = x.len() / 2;
        let mut re = Array1::zeros(n);
        let mut im = Array1::zeros(n);
        for k in 0..n {
            re[k] = x[2 * k];
            im[k] = x[2 * k + 1];
        }
        Some((re, im))
    }
}

/// Interleaved pack of real and imaginary parts.
pub fn pack(re: ArrayView1<f64>, im: ArrayView1<f64>) -> Array1<f64> {
    EuclideanComplex::pack(re, im)
}

/// Split a packed vector into `(re, im)`. `None` if the length is odd.
pub fn unpack(x: &Array1<f64>) -> Option<(Array1<f64>, Array1<f64>)> {
    EuclideanComplex::unpack(x)
}

/// manopt `M.inner = real(d1(:)'*d2(:))` on the interleaved packing.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt `M.typicaldist = sqrt(prod(dimensions))` for `C^n`.
pub fn typical_dist(n: usize) -> f64 {
    (n as f64).sqrt()
}

/// `true` when the vector is a legal interleaved packing (`2 n`, `n >= 1`).
pub fn is_euclidean_complex(x: &Array1<f64>) -> bool {
    x.len() >= 2 && x.len() % 2 == 0
}

impl Manifold for EuclideanComplex {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        match self.packed_len() {
            Some(want) if self.n >= 1 && n == want => Ok(()),
            Some(want) => Err(want),
            None => Err(n),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return v.clone();
        }
        v.clone()
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return x.clone();
        }
        let mut y = x.clone();
        vecops::axpy(1.0, v.view(), &mut y);
        y
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
    fn retract_stays_on_the_set() {
        let m = EuclideanComplex { n: 2 };
        let x = array![1.0, 0.5, -0.25, 2.0];
        let v = array![0.3, -0.1, 0.2, 0.4];
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 4);
        assert!(is_euclidean_complex(&y), "left C^2 {y:?}");
        assert!((y[0] - 1.3).abs() < 1e-15);
        assert!((y[1] - 0.4).abs() < 1e-15);
        assert!((y[2] + 0.05).abs() < 1e-15);
        assert!((y[3] - 2.4).abs() < 1e-15);
        let fro = vecops::nrm2(y.view());
        assert!((fro - 1.0).abs() > 1.0, "must not be the sphere {y:?}");
    }

    #[test]
    fn project_and_transport_are_identity() {
        let m = EuclideanComplex { n: 2 };
        let x = array![1.0, 0.0, 0.0, 1.0];
        let y = array![0.5, -0.5, 1.0, 0.0];
        let v = array![0.2, 0.3, -0.1, 0.4];
        let t = m.project(&x, &v);
        assert!((&t - &v).mapv(f64::abs).sum() < 1e-15);
        let w = m.transport(&x, &y, &v);
        assert!((&w - &v).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn pack_unpack_round_trips() {
        let re = array![1.0, 0.0, -1.0];
        let im = array![0.0, 1.0, 0.5];
        let x = EuclideanComplex::pack(re.view(), im.view());
        assert_eq!(x, array![1.0, 0.0, 0.0, 1.0, -1.0, 0.5]);
        let (r2, i2) = EuclideanComplex::unpack(&x).unwrap();
        assert!((r2 - re).mapv(f64::abs).sum() < 1e-15);
        assert!((i2 - im).mapv(f64::abs).sum() < 1e-15);
        assert!(EuclideanComplex::unpack(&array![1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn inner_is_the_real_product() {
        let u = array![1.0, 2.0, -0.5, 0.5];
        let v = array![0.5, -1.0, 2.0, 4.0];
        // 1*0.5 + 2*(-1) + (-0.5)*2 + 0.5*4 = 0.5 - 2 - 1 + 2 = -0.5
        assert!((inner(&u, &v) + 0.5).abs() < 1e-15);
        assert!((typical_dist(4) - 2.0).abs() < 1e-15);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn not_the_sphere_and_not_complex_circle() {
        let m = EuclideanComplex { n: 2 };
        let x = array![2.0, 0.0, 0.0, 2.0];
        let y = m.retract(&x, &Array1::zeros(4));
        let n0 = (y[0] * y[0] + y[1] * y[1]).sqrt();
        let n1 = (y[2] * y[2] + y[3] * y[3]).sqrt();
        assert!((n0 - 1.0).abs() > 0.5, "must not force S^1 {y:?}");
        assert!((n1 - 1.0).abs() > 0.5, "must not force S^1 {y:?}");
        let fro = vecops::nrm2(y.view());
        assert!((fro - 1.0).abs() > 1.0, "must not be S^3 {y:?}");
        assert_ne!(
            crate::manifold::ManifoldKind::EuclideanComplex { n: 2 },
            crate::manifold::ManifoldKind::Sphere
        );
        assert_ne!(
            crate::manifold::ManifoldKind::EuclideanComplex { n: 2 },
            crate::manifold::ManifoldKind::ComplexCircle { n: 2 }
        );
        assert_ne!(
            crate::manifold::ManifoldKind::EuclideanComplex { n: 2 },
            crate::manifold::ManifoldKind::Euclidean
        );
    }

    #[test]
    fn wrong_dim_rejects_a_3n_cluster() {
        let m = EuclideanComplex { n: 2 };
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(m.project(&x, &v).len(), 114);
        assert_eq!(m.required_dim(114), Err(4));
        assert!(m.required_dim(4).is_ok());
        assert!(EuclideanComplex::new(1).required_dim(2).is_ok());
        assert!(EuclideanComplex::new(0).required_dim(0).is_err());
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = EuclideanComplex { n: 2 };
        let x = array![1.0, -0.5, 0.25, 2.0];
        let y = m.retract(&x, &Array1::zeros(4));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
    }
}
