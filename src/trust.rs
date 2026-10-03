//! Powell dogleg trust-region step on a dense host Hessian.
//!
//! Nocedal and Wright, *Numerical Optimization*, algorithm 4.1,
//! <https://doi.org/10.1007/978-0-387-40065-5>.
//! Dennis and Schnabel, *Numerical Methods for Unconstrained
//! Optimization and Nonlinear Equations*,
//! <https://doi.org/10.1137/1.9781611971200>.

use ndarray::{Array1, Array2};

use crate::newton::shifted_newton;
use crate::step::l2;

pub use crate::restricted_trust::{RestrictedStep, TrustRegion};

const RHO_BAD: f64 = 0.25;
const RHO_GOOD: f64 = 0.75;
const BOUNDARY: f64 = 0.8;

/// Cauchy / Newton dogleg point inside radius `delta`.
pub fn dogleg_direction(hess: &Array2<f64>, grad: &Array1<f64>, delta: f64) -> Array1<f64> {
    let radius = delta.max(1e-16);
    let p_b = shifted_newton(hess, grad);
    let nb = l2(&p_b);
    if nb <= radius {
        return p_b;
    }
    let hg = hess.dot(grad);
    let ghg = grad.dot(&hg);
    let gg = grad.dot(grad);
    if gg <= 0.0 {
        return Array1::zeros(grad.len());
    }
    let p_u = if ghg > 1e-16 {
        grad.mapv(|v| -(gg / ghg) * v)
    } else {
        let ng = gg.sqrt();
        return grad.mapv(|v| -radius * v / ng);
    };
    let nu = l2(&p_u);
    if nu >= radius {
        return p_u.mapv(|v| v * (radius / nu));
    }
    let diff = &p_b - &p_u;
    let a = diff.dot(&diff);
    if a <= 1e-32 {
        return p_u;
    }
    let b = 2.0 * p_u.dot(&diff);
    let c = p_u.dot(&p_u) - radius * radius;
    let disc = (b * b - 4.0 * a * c).max(0.0);
    let tau = ((-b + disc.sqrt()) / (2.0 * a)).clamp(0.0, 1.0);
    &p_u + &(&diff * tau)
}

/// Predicted reduction `-g·p - 1/2 p·H p`.
pub fn predicted_reduction(hess: &Array2<f64>, grad: &Array1<f64>, p: &Array1<f64>) -> f64 {
    let hp = hess.dot(p);
    -grad.dot(p) - 0.5 * p.dot(&hp)
}

/// Trust-region ratio `ared / pred`.
pub fn reduction_ratio(ared: f64, pred: f64) -> f64 {
    if pred.abs() <= 1e-16 {
        if ared >= 0.0 { 1.0 } else { -1.0 }
    } else {
        ared / pred
    }
}

/// Nocedal-Wright radius update. Returns the new radius.
pub fn update_radius(radius: f64, rho: f64, pnorm: f64, rmax: f64) -> f64 {
    let r = radius.max(1e-16);
    if rho < RHO_BAD {
        (0.25 * r).max(1e-16)
    } else if rho > RHO_GOOD && pnorm >= BOUNDARY * r {
        (2.0 * r).min(rmax.max(r))
    } else {
        r
    }
}

/// True when the trial point is accepted (`ρ > 0`).
pub fn accept_ratio(rho: f64) -> bool {
    rho > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn dogleg_takes_the_newton_point_inside_a_large_radius() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let p = dogleg_direction(&h, &g, 10.0);
        assert!((p[0] + 1.0).abs() < 1e-10);
        assert!(p[1].abs() < 1e-10);
    }

    #[test]
    fn dogleg_stays_on_the_trust_sphere() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let p = dogleg_direction(&h, &g, 0.1);
        let n = l2(&p);
        assert!((n - 0.1).abs() < 1e-12);
        assert!(p[0] < 0.0);
    }
}
