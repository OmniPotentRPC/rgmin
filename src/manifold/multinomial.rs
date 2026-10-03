//! Multinomial simplex \(\{x > 0,\ 1^\top x = 1\}\) with the Fisher metric.
//!
//! manopt `multinomialfactory` at \(m = 1\). The point is a rank-1 f64
//! vector of length \(n \ge 2\). Projection is Fisher-orthogonal onto
//! \(\{1^\top v = 0\}\): \(v - (1^\top v)\, x\). Retraction is the
//! first-order map \(x \odot \exp(v \oslash x)\), renormalized onto the
//! simplex. Transport is projection at the arrival point.
//!
//! Sun, Gao, Hong, Mishra, Yin, *IEEE Trans. Pattern Anal. Mach. Intell.*
//! 38(3) 476-489 (2016), <https://doi.org/10.1109/TPAMI.2015.2465901>.
//! Column-stochastic \(m > 1\) and doubly-stochastic factories are
//! different tokens.

use ndarray::{Array1, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// Relative interior of the simplex, Fisher information metric.
#[derive(Clone, Copy, Debug, Default)]
pub struct Multinomial;

impl Multinomial {
    /// Flat f64 pack of a simplex point (manopt `M.vec`, \(m = 1\)).
    pub fn pack(x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }

    /// Inverse of [`Self::pack`] (manopt `M.mat` with \(m = 1\)).
    pub fn unpack(packed: &Array1<f64>) -> Array1<f64> {
        packed.clone()
    }
}

fn onto_simplex(x: ArrayView1<f64>) -> Array1<f64> {
    let s = vecops::sum(x);
    if s.is_finite() && s > 0.0 {
        let mut y = Array1::from_iter(x.iter().map(|xi| (*xi / s).max(f64::EPSILON)));
        let s2 = vecops::sum(y.view());
        if s2.is_finite() && s2 > 0.0 {
            y.mapv_inplace(|yi| yi / s2);
        }
        return y;
    }
    let n = x.len().max(1) as f64;
    Array1::from_elem(x.len(), 1.0 / n)
}

impl Manifold for Multinomial {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n >= 2 {
            Ok(())
        } else {
            Err(2)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != v.len() {
            return v.clone();
        }
        let vx = Vector::from_host(x.clone());
        let mut eta = Vector::from_host(v.clone());
        let alpha = vecops::sum(eta.host_view());
        vecops::vaxpy(-alpha, &vx, &mut eta);
        eta.into_host()
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != v.len() {
            return x + v;
        }
        let mut y = Array1::from_iter(x.iter().zip(v.iter()).map(|(xi, vi)| {
            let den = xi.max(f64::EPSILON);
            xi.max(0.0) * (*vi / den).exp()
        }));
        let s = vecops::sum(y.view());
        if !s.is_finite() || s <= 0.0 {
            return onto_simplex(x.view());
        }
        for yi in y.iter_mut() {
            *yi = (*yi / s).max(f64::EPSILON);
        }
        onto_simplex(y.view())
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
    fn retract_stays_on_the_simplex() {
        let x = array![0.2, 0.3, 0.5];
        let v = array![0.1, -0.05, -0.05];
        let y = Multinomial.retract(&x, &v);
        assert!(y.iter().all(|&yi| yi > 0.0), "left the interior {y:?}");
        let s = vecops::sum(y.view());
        assert!((s - 1.0).abs() < 1e-14, "sum {s} y={y:?}");
    }

    #[test]
    fn project_is_tangent_to_the_simplex() {
        let x = array![0.2, 0.3, 0.5];
        let v = array![1.0, 2.0, 3.0];
        let t = Multinomial.project(&x, &v);
        let s = vecops::sum(t.view());
        assert!(s.abs() < 1e-14, "not tangent, 1^T t = {s}");
        // v - (1^T v) x = [1, 2, 3] - 6 [0.2, 0.3, 0.5] = [-0.2, 0.2, 0.0]
        assert!((t[0] + 0.2).abs() < 1e-15);
        assert!((t[1] - 0.2).abs() < 1e-15);
        assert!(t[2].abs() < 1e-15);
    }

    #[test]
    fn pack_unpack_round_trip() {
        let x = array![0.1, 0.2, 0.7];
        let p = Multinomial::pack(x.view());
        let u = Multinomial::unpack(&p);
        assert_eq!(p, x);
        assert_eq!(u, x);
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let x = array![0.2, 0.3, 0.5];
        let y = array![0.4, 0.4, 0.2];
        let v = array![0.1, 0.0, -0.1];
        let t = Multinomial.transport(&x, &y, &v);
        let p = Multinomial.project(&y, &v);
        assert!((&t - &p).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn required_dim_needs_two_categories() {
        assert_eq!(Multinomial.required_dim(0), Err(2));
        assert_eq!(Multinomial.required_dim(1), Err(2));
        assert!(Multinomial.required_dim(2).is_ok());
        assert!(Multinomial.required_dim(114).is_ok());
    }
}
