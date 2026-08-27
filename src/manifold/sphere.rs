//! Unit sphere \(S^{n-1}\). manopt_cpp `Sphere`.
//!
//! Projection \(v - (x\cdot v)x\). Retraction \((x+v)/\|x+v\|\).
//! Transport is projection at the arrival point.
//! Reductions go through [`crate::vecops`].

use ndarray::Array1;

use crate::vecops::{self, Vector};

use super::Manifold;

/// Unit sphere in the ambient Euclidean metric.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sphere;

impl Manifold for Sphere {
    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let s = vecops::dot(x.view(), v.view());
        let mut out = Vector::from_host(v.clone());
        vecops::vaxpy(-s, &Vector::from_host(x.clone()), &mut out);
        out.into_host()
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut y = Vector::from_host(x.clone());
        vecops::vaxpy(1.0, &Vector::from_host(v.clone()), &mut y);
        let n = vecops::vnrm2(&y);
        if n <= 1e-16 {
            let n0 = vecops::nrm2(x.view());
            if n0 <= 1e-16 {
                return x.clone();
            }
            let mut xn = Vector::from_host(x.clone());
            vecops::scale(1.0 / n0, xn.host_mut());
            return xn.into_host();
        }
        vecops::scale(1.0 / n, y.host_mut());
        y.into_host()
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
    fn project_is_orthogonal_to_x() {
        let x = array![1.0, 0.0, 0.0];
        let v = array![2.0, 3.0, 4.0];
        let t = Sphere.project(&x, &v);
        assert!(vecops::dot(x.view(), t.view()).abs() < 1e-15);
        assert!((t[1] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn retract_stays_on_the_sphere() {
        let x = array![0.0, 1.0, 0.0];
        let v = array![0.1, 0.0, -0.2];
        let y = Sphere.retract(&x, &v);
        assert!((vecops::nrm2(y.view()) - 1.0).abs() < 1e-14);
    }

    #[cfg(feature = "par")]
    #[test]
    fn par_retract_stays_on_the_sphere() {
        retract_stays_on_the_sphere();
    }
}
