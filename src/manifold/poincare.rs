//! Open unit ball with the Poincare metric. manopt `poincareballfactory`.
//!
//! Packed as a length-`k` vector with Euclidean 2-norm strictly less
//! than 1. This is not the hyperboloid model and not the sphere: the
//! tangent space is the ambient space, the metric is conformal with
//! factor \(2 / (1 - \|x\|^2)\), and the retraction is the Riemannian
//! exponential via Mobius addition.
//!
//! [`PoincareBall::project`] converts a Euclidean ambient vector to a
//! Riemannian gradient, \(v \cdot (1 - \|x\|^2)^2 / 4\).

use ndarray::Array1;

use crate::vecops::{dot, nrm2};

use super::Manifold;

/// Open unit ball \(B^k = \{ x \in R^k : \|x\| < 1 \}\) with curvature \(-1\).
#[derive(Clone, Copy, Debug, Default)]
pub struct PoincareBall;

impl PoincareBall {
    /// Pack a Poincare-ball point: the ambient length-`k` vector itself.
    pub fn pack(coords: Array1<f64>) -> Array1<f64> {
        coords
    }

    /// Unpack a packed Poincare-ball point to ambient coordinates.
    pub fn unpack(x: &Array1<f64>) -> Array1<f64> {
        x.clone()
    }
}

fn n2(x: &Array1<f64>) -> f64 {
    dot(x.view(), x.view())
}

/// Conformal factor \(2 / (1 - \|x\|^2)\). Clamped so the pole at the
/// boundary does not invert the metric.
fn conformal_factor(x: &Array1<f64>) -> f64 {
    2.0 / (1.0 - n2(x)).max(1e-15)
}

/// Radial pullback onto the open ball. The exponential is only defined
/// for \(\|x\| < 1\).
fn interiorize(x: &Array1<f64>) -> Array1<f64> {
    let n = nrm2(x.view());
    if n < 1.0 {
        x.clone()
    } else if n <= 1e-16 {
        x.clone()
    } else {
        x * ((1.0 - 1e-12) / n)
    }
}

/// Gyrovector (Mobius) addition on the ball.
fn mobius_add(x: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let sp = dot(x.view(), y.view());
    let nx = n2(x);
    let ny = n2(y);
    let denom = 1.0 + 2.0 * sp + nx * ny;
    if denom.abs() < 1e-18 {
        return interiorize(x);
    }
    let ax = 1.0 + 2.0 * sp + ny;
    let ay = 1.0 - nx;
    Array1::from_iter(
        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (ax * xi + ay * yi) / denom),
    )
}

/// Riemannian exponential: Mobius addition of \(x\) with a tanh-scaled
/// tangent, matching manopt `poincareballfactory` `exponential`.
fn exp_map(x: &Array1<f64>, d: &Array1<f64>) -> Array1<f64> {
    let x = interiorize(x);
    let n = nrm2(d.view());
    let factor = (1.0 - n2(&x)).max(1e-15);
    let scale = if n < 1e-16 {
        1.0 / factor
    } else {
        let th = (n / factor).tanh().min(1.0 - 1e-15);
        th / n
    };
    interiorize(&mobius_add(&x, &(d * scale)))
}

impl Manifold for PoincareBall {
    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != v.len() {
            return v.clone();
        }
        // egrad2rgrad: g^{-1} flattens the Euclidean gradient.
        let factor = conformal_factor(x);
        v * (1.0 / (factor * factor))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != v.len() {
            return interiorize(x);
        }
        exp_map(x, v)
    }

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        // manopt: not a parallel transport; the embedding is the tangent.
        v.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::Sphere;
    use ndarray::array;

    #[test]
    fn pack_unpack_is_the_ambient_vector() {
        let x = array![0.3, -0.2, 0.1];
        let p = PoincareBall::pack(x.clone());
        assert_eq!(p, x);
        assert_eq!(PoincareBall::unpack(&p), x);
        assert!(nrm2(p.view()) < 1.0);
    }

    #[test]
    fn project_uses_the_poincare_metric() {
        let x = array![0.3, 0.0, 0.0];
        let v = array![1.0, 2.0, 3.0];
        let p = PoincareBall.project(&x, &v);
        let scale = (1.0 - 0.09) * (1.0 - 0.09) / 4.0;
        for i in 0..3 {
            assert!(
                (p[i] - v[i] * scale).abs() < 1e-14,
                "p[{i}] = {} want {}",
                p[i],
                v[i] * scale
            );
        }
        // Not the identity and not the sphere's Euclidean projection.
        assert!((p[0] - v[0]).abs() > 1e-6);
        let s = Sphere.project(&x, &v);
        assert!((&p - &s).mapv(f64::abs).sum() > 1e-6);
    }

    #[test]
    fn retract_stays_inside_the_open_unit_ball() {
        let x = array![0.4, -0.2, 0.1];
        let v = array![0.3, 0.5, -0.4];
        let y = PoincareBall.retract(&x, &v);
        let n = nrm2(y.view());
        assert!(n < 1.0, "left the open ball, ||y|| = {n}, y = {y:?}");
    }

    #[test]
    fn retract_of_a_huge_step_stays_inside() {
        let x = array![0.8, 0.0];
        let v = array![20.0, 0.0];
        let y = PoincareBall.retract(&x, &v);
        let n = nrm2(y.view());
        assert!(n < 1.0, "huge step left the ball, ||y|| = {n}");
        // Euclidean translation would sit at 20.8.
        assert!((y[0] - 20.8).abs() > 1.0);
    }

    #[test]
    fn retract_at_the_origin_is_tanh_scaled() {
        let x = array![0.0, 0.0];
        let v = array![0.5, 0.0];
        let y = PoincareBall.retract(&x, &v);
        let want = 0.5_f64.tanh();
        assert!((y[0] - want).abs() < 1e-14, "y = {y:?}");
        assert!(y[1].abs() < 1e-15);
    }

    #[test]
    fn zero_tangent_is_a_fixed_point() {
        let x = array![0.2, -0.3, 0.1];
        let y = PoincareBall.retract(&x, &Array1::zeros(3));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-14);
    }

    #[test]
    fn transport_is_the_embedding_identity() {
        let x = array![0.1, 0.2];
        let z = array![-0.3, 0.0];
        let v = array![0.4, -0.1];
        assert_eq!(PoincareBall.transport(&x, &z, &v), v);
    }

    #[test]
    fn geometry_is_not_the_sphere() {
        let x = array![0.3, 0.4, 0.0];
        let v = array![0.1, -0.2, 0.05];
        let yp = PoincareBall.retract(&x, &v);
        let ys = Sphere.retract(&x, &v);
        assert!((&yp - &ys).mapv(f64::abs).sum() > 1e-6);
        assert!(nrm2(yp.view()) < 1.0);
        assert!((nrm2(ys.view()) - 1.0).abs() < 1e-12);
    }
}
