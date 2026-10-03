//! Interleaved complex Euclidean space \(\mathbb{C}^n\).
//!
//! manopt `euclideancomplexfactory(n)` (and `(m, n)` as \(n = m n\)).
//! The complex plane is identified with \(\mathbb{R}^2\). A point is
//! `n` complex numbers packed as interleaved `(re, im)` pairs, length
//! `2 n`. The inner product is manopt `real(u(:)'*v(:))`, which is the
//! real Euclidean inner product on the packed vector.
//!
//! Projection is the identity. Retraction is `x + v`. Transport is
//! the identity. This is not the sphere, not \((S^1)^n\), and not a
//! 3N cluster. Isolated molecules use [`super::RigidQuotient`].
//!
//! Reductions go through [`crate::vecops`].

use ndarray::{Array1, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// Complex Euclidean \(\mathbb{C}^n\) as `n` interleaved real-imaginary pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EuclideanComplex {
    /// Number of complex entries. manopt `n` at `m = 1`, or `m*n` for a matrix.
    pub n: usize,
}

impl Default for EuclideanComplex {
    fn default() -> Self {
        Self { n: 1 }
    }
}

impl EuclideanComplex {
    /// \(\mathbb{C}^n\). Illegal `n == 0` fails [`Manifold::required_dim`].
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

/// Frobenius / real inner product. manopt `real(d1(:)'*d2(:))`.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt `M.typicaldist = sqrt(prod(size))` for \(\mathbb{C}^n\).
pub fn typical_dist(n: usize) -> f64 {
    (n as f64).sqrt()
}

/// `true` when `x` is a legal interleaved packing (`length = 2 n`, `n >= 1`).
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
        let mut y = Vector::from_host(x.clone());
        vecops::vaxpy(1.0, &Vector::from_host(v.clone()), &mut y);
        y.into_host()
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
        assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
        let n0 = (y[0] * y[0] + y[1] * y[1]).sqrt();
        let n1 = (y[2] * y[2] + y[3] * y[3]).sqrt();
        assert!((n0 - 1.0).abs() > 0.1, "must not force S^1 pair 0 {y:?}");
        assert!((n1 - 1.0).abs() > 0.1, "must not force S^1 pair 1 {y:?}");
    }

    #[test]
    fn project_is_the_identity() {
        let m = EuclideanComplex { n: 2 };
        let x = array![1.0, 0.0, 0.0, 1.0];
        let v = array![0.5, -0.25, 0.1, 0.8];
        let t = m.project(&x, &v);
        for i in 0..4 {
            assert!((t[i] - v[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn transport_of_a_tangent_is_itself() {
        let m = EuclideanComplex { n: 2 };
        let x = array![1.0, 0.0, 0.0, 1.0];
        let y = array![0.0, 2.0, -1.0, 0.5];
        let v = array![0.2, 0.3, -0.1, 0.4];
        let t = m.transport(&x, &y, &v);
        for i in 0..4 {
            assert!((t[i] - v[i]).abs() < 1e-15);
        }
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
    fn inner_is_real_dot_and_typical_dist_is_sqrt_n() {
        let u = array![1.0, 0.0, 0.0, 1.0];
        let v = array![0.0, 1.0, 1.0, 0.0];
        assert!(inner(&u, &v).abs() < 1e-15);
        let w = array![1.0, 2.0, 3.0, 4.0];
        assert!((inner(&w, &w) - 30.0).abs() < 1e-15);
        assert!((typical_dist(4) - 2.0).abs() < 1e-15);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
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
        assert!(EuclideanComplex::new(3).required_dim(5).is_err());
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = EuclideanComplex { n: 2 };
        let x = array![1.0, -0.5, 0.25, 2.0];
        let y = m.retract(&x, &Array1::zeros(4));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn kind_is_not_sphere_or_complex_circle() {
        use crate::manifold::ManifoldKind;
        assert_ne!(
            ManifoldKind::EuclideanComplex { n: 2 },
            ManifoldKind::Sphere
        );
        assert_ne!(
            ManifoldKind::EuclideanComplex { n: 2 },
            ManifoldKind::ComplexCircle { n: 2 }
        );
        assert_ne!(
            ManifoldKind::EuclideanComplex { n: 1 },
            ManifoldKind::Stiefel
        );
        assert_ne!(
            ManifoldKind::EuclideanComplex { n: 2 },
            ManifoldKind::Euclidean
        );
    }
}
