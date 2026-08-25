//! Complex unit sphere \(\{z \in \mathbb{C}^n : z^* z = 1\}\).
//!
//! manopt `spherecomplexfactory` (default \(m = 1\)): unit-Frobenius
//! complex vectors, identified with the real sphere in
//! \(\mathbb{R}^{2n}\) under the interleaved packing `(re, im)`.
//! The real inner product \(\mathrm{Re}(x^* u)\) is the Euclidean
//! product on the packed vector. Projection is
//! \(u - \mathrm{Re}(x^* u)\, x\). Retraction is
//! \((x+v)/\|x+v\|\). Transport is projection at the arrival point.
//! This token is not [`super::Sphere`] (real \(S^{n-1}\) with a
//! different packing contract) and not [`super::ComplexCircle`]
//! (product of \(S^1\)). Reserved tokens 7-10 stay unused.
//!
//! Reductions go through [`crate::vecops`].

use ndarray::{Array1, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// Unit sphere in \(\mathbb{C}^n\), packed as `2 n` reals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SphereComplex {
    /// Complex dimension. Packed length is `2 n`.
    pub n: usize,
}

impl Default for SphereComplex {
    fn default() -> Self {
        Self { n: 1 }
    }
}

impl SphereComplex {
    /// \(\mathbb{C}^n\) unit sphere. Illegal `n == 0` fails
    /// [`Manifold::required_dim`].
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

/// Real inner product. manopt `real(d1(:)'*d2(:))`.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt `M.typicaldist = pi`.
pub fn typical_dist() -> f64 {
    std::f64::consts::PI
}

/// `true` when `x` is a legal interleaved packing (`length = 2 n`, `n >= 1`).
pub fn is_sphere_complex(x: &Array1<f64>) -> bool {
    x.len() >= 2 && x.len() % 2 == 0
}

impl Manifold for SphereComplex {
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
        let s = vecops::dot(x.view(), v.view());
        let mut out = Vector::from_host(v.clone());
        vecops::vaxpy(-s, &Vector::from_host(x.clone()), &mut out);
        out.into_host()
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return x.clone();
        }
        let mut y = Vector::from_host(x.clone());
        vecops::vaxpy(1.0, &Vector::from_host(v.clone()), &mut y);
        let nrm = vecops::vnrm2(&y);
        if nrm > 1e-16 {
            let ys = y.host_mut();
            *ys /= nrm;
            return y.into_host();
        }
        let n0 = vecops::nrm2(x.view());
        if n0 <= 1e-16 {
            return x.clone();
        }
        x / n0
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
    fn retract_stays_on_the_complex_sphere() {
        let m = SphereComplex { n: 2 };
        let x = array![1.0, 0.0, 0.0, 0.0];
        let v = array![0.0, 0.1, 0.2, -0.1];
        let y = m.retract(&x, &v);
        let n2 = vecops::nrm2(y.view());
        assert!((n2 - 1.0).abs() < 1e-14, "norm {n2}");
        assert!(is_sphere_complex(&y));
    }

    #[test]
    fn project_is_hermitian_tangent() {
        let m = SphereComplex { n: 2 };
        let x = array![0.6, 0.8, 0.0, 0.0];
        let v = array![1.0, 2.0, 3.0, 4.0];
        let t = m.project(&x, &v);
        let ip = inner(&x, &t);
        assert!(ip.abs() < 1e-14, "Re(x^* t) = {ip}");
        assert!((typical_dist() - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn required_dim_is_twice_complex_n() {
        let m = SphereComplex { n: 3 };
        assert_eq!(m.required_dim(6), Ok(()));
        assert_eq!(m.required_dim(3), Err(6));
        assert!(SphereComplex::new(1).required_dim(2).is_ok());
        assert!(SphereComplex::new(0).required_dim(0).is_err());
    }

    #[test]
    fn not_the_real_sphere_token() {
        assert_ne!(
            crate::manifold::ManifoldKind::SphereComplex { n: 2 },
            crate::manifold::ManifoldKind::Sphere
        );
        assert_ne!(
            crate::manifold::ManifoldKind::SphereComplex { n: 2 },
            crate::manifold::ManifoldKind::ComplexCircle { n: 2 }
        );
        assert_ne!(
            crate::manifold::ManifoldKind::SphereComplex { n: 2 },
            crate::manifold::ManifoldKind::EuclideanComplex { n: 2 }
        );
        assert_ne!(
            crate::manifold::ManifoldKind::SphereComplex { n: 1 },
            crate::manifold::ManifoldKind::Stiefel
        );
    }

    #[test]
    fn pack_unpack_round_trips() {
        let re = array![1.0, 0.0, -1.0];
        let im = array![0.0, 1.0, 0.5];
        let x = SphereComplex::pack(re.view(), im.view());
        assert_eq!(x, array![1.0, 0.0, 0.0, 1.0, -1.0, 0.5]);
        let (r2, i2) = SphereComplex::unpack(&x).unwrap();
        assert!((r2 - re).mapv(f64::abs).sum() < 1e-15);
        assert!((i2 - im).mapv(f64::abs).sum() < 1e-15);
        assert!(SphereComplex::unpack(&array![1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let m = SphereComplex { n: 2 };
        let x = array![1.0, 0.0, 0.0, 0.0];
        let y = array![0.0, 0.0, 1.0, 0.0];
        let v = array![0.2, 0.3, -0.1, 0.4];
        let t = m.transport(&x, &y, &v);
        let p = m.project(&y, &v);
        assert!((&t - &p).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn wrong_dim_rejects_a_3n_cluster() {
        let m = SphereComplex { n: 2 };
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert!(
            (&y - &x).mapv(f64::abs).sum() < 1e-15,
            "must not sphere-normalize a cluster"
        );
        assert_eq!(m.project(&x, &v).len(), 114);
        assert_eq!(m.required_dim(114), Err(4));
        assert!(m.required_dim(4).is_ok());
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = SphereComplex { n: 2 };
        let x = array![0.6, 0.8, 0.0, 0.0];
        let y = m.retract(&x, &Array1::zeros(4));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
        assert!((vecops::nrm2(y.view()) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn product_of_circles_is_not_this_geometry() {
        // Two unit-modulus pairs: ||.|| = sqrt(2), not 1.
        let m = SphereComplex { n: 2 };
        let x = array![1.0, 0.0, 0.0, 1.0];
        let y = m.retract(&x, &Array1::zeros(4));
        let fro = vecops::nrm2(y.view());
        assert!(
            (fro - 1.0).abs() < 1e-14,
            "must be the complex sphere {y:?}"
        );
        let n0 = (y[0] * y[0] + y[1] * y[1]).sqrt();
        let n1 = (y[2] * y[2] + y[3] * y[3]).sqrt();
        assert!(
            (n0 - 1.0).abs() > 0.2,
            "must not force each pair onto S^1 {y:?}"
        );
        assert!(
            (n1 - 1.0).abs() > 0.2,
            "must not force each pair onto S^1 {y:?}"
        );
    }
}
