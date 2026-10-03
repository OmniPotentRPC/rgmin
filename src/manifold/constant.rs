//! Singleton \(\{A\}\) as a zero-dimensional manifold.
//!
//! manopt `constantfactory(A)`. The only point is the packed vector
//! `A` of length `n`. The only tangent is the zero vector of that
//! length. Projection and transport are the zero array. Retraction
//! is the fixed point. This is a helper geometry for holding a
//! block constant (manopt `productmanifold`), not the sphere and
//! not a 3N cluster packing.
//!
//! Reductions go through [`crate::vecops`].

use ndarray::{Array1, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// Singleton \(\{A\}\) of packed length `n`. manopt `constantfactory`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Constant {
    /// Packed length of the singleton. manopt `numel(A)`.
    pub n: usize,
}

impl Default for Constant {
    fn default() -> Self {
        Self { n: 1 }
    }
}

impl Constant {
    /// Singleton of packed length `n`. Illegal `n == 0` fails [`Manifold::required_dim`].
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    fn fits(self, len: usize) -> bool {
        self.n >= 1 && self.n == len
    }

    /// Pack the singleton point. Identity on the ambient vector.
    pub fn pack(a: ArrayView1<f64>) -> Array1<f64> {
        a.to_owned()
    }

    /// Unpack the singleton. `None` if the length is not `n`.
    pub fn unpack(self, x: &Array1<f64>) -> Option<Array1<f64>> {
        if self.fits(x.len()) {
            Some(x.clone())
        } else {
            None
        }
    }
}

/// Zero inner product. manopt `M.inner = 0` on a 0-dim tangent.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    let n = u.len().min(v.len());
    vecops::vdot(&Vector::zeros_cpu(n), &Vector::zeros_cpu(n))
}

/// manopt `M.typicaldist = 0`.
pub fn typical_dist() -> f64 {
    0.0
}

/// `true` when `x` is a legal singleton packing of length `n`.
pub fn is_constant(x: &Array1<f64>, n: usize) -> bool {
    n >= 1 && x.len() == n
}

fn zeros_of(n: usize) -> Array1<f64> {
    Vector::zeros_cpu(n).into_host()
}

impl Manifold for Constant {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if self.n >= 1 && n == self.n {
            Ok(())
        } else {
            Err(self.n)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return v.clone();
        }
        zeros_of(v.len())
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return x.clone();
        }
        x.clone()
    }

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        zeros_of(v.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_stays_on_the_set() {
        let m = Constant { n: 3 };
        let x = array![1.25, -0.5, 2.0];
        let v = array![0.3, -0.1, 4.0];
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 3);
        assert!(is_constant(&y, 3), "left the singleton {y:?}");
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15, "moved off A {y:?}");
        let fro = vecops::nrm2(y.view());
        assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
    }

    #[test]
    fn project_is_the_zero_tangent() {
        let m = Constant { n: 3 };
        let x = array![1.0, 2.0, 3.0];
        let v = array![0.5, -0.25, 0.8];
        let t = m.project(&x, &v);
        for i in 0..3 {
            assert!(t[i].abs() < 1e-15, "nonzero tangent {t:?}");
        }
        assert!((inner(&t, &t)).abs() < 1e-15);
        assert!(typical_dist().abs() < 1e-15);
    }

    #[test]
    fn transport_of_a_tangent_is_zero() {
        let m = Constant { n: 2 };
        let x = array![1.0, 0.0];
        let y = array![1.0, 0.0];
        let v = array![0.2, 0.3];
        let t = m.transport(&x, &y, &v);
        assert!(t.iter().all(|a| a.abs() < 1e-15), "nonzero transport {t:?}");
    }

    #[test]
    fn pack_unpack_round_trips() {
        let a = array![1.0, -2.0, 0.5];
        let x = Constant::pack(a.view());
        assert_eq!(x, a);
        let back = Constant { n: 3 }.unpack(&x).unwrap();
        assert!((back - a).mapv(f64::abs).sum() < 1e-15);
        assert!(Constant { n: 2 }.unpack(&x).is_none());
        assert!(Constant { n: 0 }.unpack(&array![1.0]).is_none());
    }

    #[test]
    fn wrong_dim_rejects_a_3n_cluster() {
        let m = Constant { n: 2 };
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(m.project(&x, &v).len(), 114);
        assert_eq!(m.required_dim(114), Err(2));
        assert!(m.required_dim(2).is_ok());
        assert!(Constant::new(1).required_dim(1).is_ok());
        assert!(Constant::new(0).required_dim(0).is_err());
        assert!(Constant::new(3).required_dim(5).is_err());
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = Constant { n: 3 };
        let x = array![1.0, -0.5, 0.25];
        let y = m.retract(&x, &Array1::zeros(3));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn kind_is_not_sphere_or_stiefel() {
        use crate::manifold::ManifoldKind;
        assert_ne!(ManifoldKind::Constant { n: 3 }, ManifoldKind::Sphere);
        assert_ne!(ManifoldKind::Constant { n: 1 }, ManifoldKind::Stiefel);
        assert_ne!(ManifoldKind::Constant { n: 3 }, ManifoldKind::Euclidean);
        assert_ne!(
            ManifoldKind::Constant { n: 4 },
            ManifoldKind::EuclideanComplex { n: 2 }
        );
    }
}
