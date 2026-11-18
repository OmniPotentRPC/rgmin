//! Affine hyperplane \(\{x : n^\top (x - p) = 0\}\).
//!
//! The geometry is flat: geodesics are straight lines and parallel
//! transport is the identity on the tangent space. The Euclidean
//! projector is \(v - n(n^\top v)\) with a unit normal. The kinetic
//! drift of a constrained integrator is the straight step followed by
//! the mass-metric restoration
//! \(x - w\, n^\top(x - p)\), \(w = M^{-1} n / (n^\top M^{-1} n)\),
//! which is the position half of RATTLE on this constraint.

use ndarray::{Array1, Array2};

use super::Manifold;
use super::mass::{self, MassMetric};
use crate::error::{Error, Result};

/// The plane through `point` with unit Euclidean normal `normal`.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineHyperplane {
    point: Array1<f64>,
    normal: Array1<f64>,
}

impl AffineHyperplane {
    /// Normalize `normal`. The point is not moved.
    pub fn new(point: Array1<f64>, normal: Array1<f64>) -> Result<Self> {
        if point.is_empty() || point.len() != normal.len() {
            return Err(Error::MassMetric {
                what: "hyperplane point and normal need one positive length",
            });
        }
        if !point.iter().chain(normal.iter()).all(|v| v.is_finite()) {
            return Err(Error::MassMetric {
                what: "hyperplane point and normal must be finite",
            });
        }
        let norm = normal.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return Err(Error::MassMetric {
                what: "hyperplane normal must be nonzero",
            });
        }
        let inv = 1.0 / norm;
        let mut unit = normal;
        for value in unit.iter_mut() {
            *value *= inv;
        }
        Ok(Self {
            point,
            normal: unit,
        })
    }

    /// A point of the plane.
    pub fn point(&self) -> &Array1<f64> {
        &self.point
    }

    /// Unit Euclidean normal.
    pub fn normal(&self) -> &Array1<f64> {
        &self.normal
    }

    /// Ambient dimension.
    pub fn dim(&self) -> usize {
        self.point.len()
    }

    /// Signed Euclidean distance \(n^\top (x - p)\).
    pub fn signed_distance(&self, x: &Array1<f64>) -> Result<f64> {
        if x.len() != self.dim() {
            return Err(Error::MassMetric {
                what: "hyperplane distance has the wrong length",
            });
        }
        let mut distance = 0.0;
        for i in 0..self.dim() {
            distance += self.normal[i] * (x[i] - self.point[i]);
        }
        Ok(distance)
    }

    /// Euclidean foot of the perpendicular from `x`.
    pub fn restore_euclidean(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        let distance = self.signed_distance(x)?;
        Ok(x - &self.normal * distance)
    }
}

impl Manifold for AffineHyperplane {
    fn required_dim(&self, n: usize) -> std::result::Result<(), usize> {
        if n == self.dim() {
            Ok(())
        } else {
            Err(self.dim())
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != self.dim() || v.len() != self.dim() {
            return v.clone();
        }
        let mut nv = 0.0;
        for i in 0..self.dim() {
            nv += self.normal[i] * v[i];
        }
        let mut out = v.clone();
        for i in 0..self.dim() {
            out[i] -= self.normal[i] * nv;
        }
        out
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != self.dim() || v.len() != self.dim() {
            return x.clone();
        }
        let step = self.project(x, v);
        x + step
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let _ = x_from;
        self.project(x_to, v)
    }

    fn exp(&self, x: &Array1<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
        (x.len() == self.dim() && v.len() == self.dim()).then(|| self.retract(x, v))
    }

    fn log(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
        (x.len() == self.dim() && y.len() == self.dim()).then(|| self.project(x, &(y - x)))
    }

    fn parallel_transport_along(
        &self,
        x: &Array1<f64>,
        d: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        (x.len() == self.dim() && d.len() == self.dim() && v.len() == self.dim())
            .then(|| self.project(x, v))
    }

    fn transport_is_isometric(&self) -> bool {
        true
    }

    fn drift_is_straight(&self) -> bool {
        true
    }

    fn euclidean_normals(&self, x: &Array1<f64>) -> Option<Array2<f64>> {
        if x.len() != self.dim() {
            return None;
        }
        let mut normals = Array2::zeros((self.dim(), 1));
        normals.column_mut(0).assign(&self.normal);
        Some(normals)
    }

    fn snap_mass(&self, metric: &MassMetric, x: &Array1<f64>) -> Result<Array1<f64>> {
        let normals = self.euclidean_normals(x).ok_or(Error::MassMetric {
            what: "hyperplane snap has the wrong length",
        })?;
        mass::restore_mass(normals.view(), metric, &self.point, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mass_restore_matches_the_inverse_mass_step() {
        let plane = AffineHyperplane::new(array![1.0, 2.0, 0.0], array![0.0, 2.0, 0.0]).unwrap();
        let metric = MassMetric::new(array![4.0, 1.0, 9.0]).unwrap();
        let x = array![1.0, 5.0, 0.5];
        let y = plane.snap_mass(&metric, &x).unwrap();
        let normal = plane.normal();
        let alpha = normal[1] * normal[1] / 1.0;
        let nx = normal[1] * (5.0 - 2.0);
        let w1 = normal[1] * (1.0 / (alpha * 1.0));
        assert_eq!(y[0].to_bits(), 1.0_f64.to_bits());
        assert_eq!(y[1].to_bits(), (5.0 - w1 * nx).to_bits());
        assert_eq!(y[2].to_bits(), 0.5_f64.to_bits());
        let distance = plane.signed_distance(&y).unwrap();
        assert_eq!(distance.to_bits(), 0.0_f64.to_bits());
    }
}
