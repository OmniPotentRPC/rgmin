//! Element-wise positive orthant \(\{x \in \mathbb{R}^n : x_i > 0\}\).
//!
//! manopt `positivefactory` (default second size 1): the open
//! positive orthant, identified with a packed length-`n` vector.
//! MATLAB `positivefactory(m, k)` is the same geometry at
//! `n = m k`. The metric is the product of bi-invariant metrics
//! on each positive scalar: \(\langle \eta, \zeta \rangle_x =
//! (\eta \oslash x)^\top (\zeta \oslash x)\). Projection is the
//! identity (open subset of \(\mathbb{R}^n\)). Retraction is the
//! exponential \(x \odot \exp(v \oslash x)\). Transport is the
//! identity (manopt default). This token is not [`super::Sphere`],
//! not the simplex, and not a 3N cluster packing. Reserved
//! tokens 7-10 stay unused.
//!
//! Reductions go through [`crate::vecops`].

use ndarray::{Array1, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// Strictly positive orthant of packed length `n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Positive {
    /// Packed length. manopt `m*n` with default `n = 1`.
    pub n: usize,
}

impl Default for Positive {
    fn default() -> Self {
        Self { n: 1 }
    }
}

impl Positive {
    /// Positive orthant of packed length `n`. Illegal `n == 0`
    /// fails [`Manifold::required_dim`].
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    fn fits(self, len: usize) -> bool {
        self.n >= 1 && self.n == len
    }

    /// Pack a positive vector. Identity on the ambient array.
    pub fn pack(x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }

    /// Unpack a packed point. `None` if the length is not `n`.
    pub fn unpack(self, x: &Array1<f64>) -> Option<Array1<f64>> {
        if self.fits(x.len()) {
            Some(x.clone())
        } else {
            None
        }
    }
}

/// Scale-invariant inner product. manopt ` (eta./X)'*(zeta./X) `.
pub fn inner(x: &Array1<f64>, eta: &Array1<f64>, zeta: &Array1<f64>) -> f64 {
    let n = x.len().min(eta.len()).min(zeta.len());
    if n == 0 {
        return 0.0;
    }
    let mut u = Vector::from_host(eta.clone());
    let mut w = Vector::from_host(zeta.clone());
    {
        let us = u.host_mut();
        for i in 0..n {
            us[i] /= x[i];
        }
    }
    {
        let ws = w.host_mut();
        for i in 0..n {
            ws[i] /= x[i];
        }
    }
    vecops::vdot(&u, &w)
}

/// manopt `M.typicaldist = sqrt(m*n)`.
pub fn typical_dist(n: usize) -> f64 {
    (n as f64).sqrt()
}

/// `true` when every entry is finite and strictly positive.
pub fn is_positive(x: &Array1<f64>) -> bool {
    !x.is_empty() && x.iter().all(|xi| xi.is_finite() && *xi > 0.0)
}

impl Manifold for Positive {
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
        Vector::from_host(v.clone()).into_host()
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != egrad.len() {
            return egrad.clone();
        }
        let mut r = Vector::from_host(egrad.clone());
        {
            let rs = r.host_mut();
            for (ri, xi) in rs.iter_mut().zip(x.iter()) {
                *ri *= *xi * *xi;
            }
        }
        r.into_host()
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() || !is_positive(x) {
            return x.clone();
        }
        let mut y = Vector::from_host(x.clone());
        {
            let ys = y.host_mut();
            for (yi, vi) in ys.iter_mut().zip(v.iter()) {
                let step = (*vi / *yi).exp();
                *yi *= step;
                // Overflow stays large (MAX); underflow/NaN stay at EPSILON.
                if *yi > 0.0 && yi.is_finite() {
                    continue;
                }
                if yi.is_infinite() && *yi > 0.0 {
                    *yi = f64::MAX;
                } else {
                    *yi = f64::EPSILON;
                }
            }
        }
        y.into_host()
    }

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        Vector::from_host(v.clone()).into_host()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_stays_on_the_set() {
        let m = Positive { n: 3 };
        let x = array![1.0, 2.0, 0.5];
        let v = array![0.2, -0.4, 0.1];
        let y = m.retract(&x, &v);
        assert!(is_positive(&y), "left the positive orthant {y:?}");
        for i in 0..3 {
            let expect = x[i] * (v[i] / x[i]).exp();
            assert!((y[i] - expect).abs() < 1e-14, "retr {y:?}");
        }
        let fro = vecops::nrm2(y.view());
        assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
    }

    #[test]
    fn project_is_the_identity() {
        let m = Positive { n: 3 };
        let x = array![1.0, 2.0, 0.5];
        let v = array![0.3, -1.0, 4.0];
        let t = m.project(&x, &v);
        assert!((&t - &v).mapv(f64::abs).sum() < 1e-15, "proj {t:?}");
        assert!((typical_dist(3) - 3.0_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn transport_is_the_identity() {
        let m = Positive { n: 2 };
        let x = array![1.0, 2.0];
        let y = array![1.5, 0.25];
        let v = array![0.2, -0.3];
        let t = m.transport(&x, &y, &v);
        assert!((&t - &v).mapv(f64::abs).sum() < 1e-15, "transp {t:?}");
    }

    #[test]
    fn egrad2rgrad_scales_by_x_squared() {
        let m = Positive { n: 3 };
        let x = array![2.0, 0.5, 4.0];
        let g = array![1.0, -2.0, 0.25];
        let r = m.egrad2rgrad(&x, &g);
        assert!((r[0] - 4.0).abs() < 1e-15);
        assert!((r[1] + 0.5).abs() < 1e-15);
        assert!((r[2] - 4.0).abs() < 1e-15);
    }

    #[test]
    fn inner_is_scale_invariant() {
        let x = array![2.0, 0.5];
        let eta = array![2.0, 1.0];
        let zeta = array![4.0, -0.5];
        let ip = inner(&x, &eta, &zeta);
        // (eta./x) · (zeta./x) = [1, 2] · [2, -1] = 0
        assert!(ip.abs() < 1e-15, "inner {ip}");
        let nrm = inner(&x, &eta, &eta).sqrt();
        assert!((nrm - 5.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn pack_unpack_round_trips() {
        let a = array![1.0, 0.25, 4.0];
        let x = Positive::pack(a.view());
        assert_eq!(x, a);
        let back = Positive { n: 3 }.unpack(&x).unwrap();
        assert!((back - a).mapv(f64::abs).sum() < 1e-15);
        assert!(Positive { n: 2 }.unpack(&x).is_none());
        assert!(Positive { n: 0 }.unpack(&array![1.0]).is_none());
    }

    #[test]
    fn wrong_dim_rejects_a_3n_cluster() {
        let m = Positive { n: 2 };
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert!(
            (&y - &x).mapv(f64::abs).sum() < 1e-15,
            "must not exp-retract a cluster"
        );
        assert_eq!(m.project(&x, &v).len(), 114);
        assert_eq!(m.required_dim(114), Err(2));
        assert!(m.required_dim(2).is_ok());
        assert!(Positive::new(1).required_dim(1).is_ok());
        assert!(Positive::new(0).required_dim(0).is_err());
        assert!(Positive::new(3).required_dim(5).is_err());
    }

    #[test]
    fn large_negative_step_stays_strictly_positive() {
        let m = Positive { n: 1 };
        let x = array![1.0];
        let v = array![-800.0];
        let y = m.retract(&x, &v);
        assert!(is_positive(&y), "underflow left the open orthant {y:?}");
        assert!(y[0] >= f64::EPSILON);
    }

    #[test]
    fn large_positive_step_stays_large() {
        let m = Positive { n: 1 };
        let x = array![1.0];
        let v = array![800.0];
        let y = m.retract(&x, &v);
        assert!(is_positive(&y), "overflow left the open orthant {y:?}");
        assert!(y[0] >= 1.0, "overflow must not snap toward 0 {y:?}");
        assert_eq!(y[0], f64::MAX);
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = Positive { n: 3 };
        let x = array![1.0, 0.5, 2.0];
        let y = m.retract(&x, &Array1::zeros(3));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
        assert!(is_positive(&y));
    }

    #[test]
    fn kind_is_not_sphere_or_stiefel() {
        use crate::manifold::ManifoldKind;
        assert_ne!(ManifoldKind::Positive { n: 3 }, ManifoldKind::Sphere);
        assert_ne!(ManifoldKind::Positive { n: 1 }, ManifoldKind::Stiefel);
        assert_ne!(ManifoldKind::Positive { n: 3 }, ManifoldKind::Euclidean);
        assert_ne!(ManifoldKind::Positive { n: 3 }, ManifoldKind::Multinomial);
        assert_ne!(
            ManifoldKind::Positive { n: 3 },
            ManifoldKind::Constant { n: 3 }
        );
    }
}
