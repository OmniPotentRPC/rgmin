//! Riemannian trust region with truncated conjugate gradients (RTR-tCG).
//!
//! Absil, Baker, Gallivan, "Trust-region methods on Riemannian manifolds",
//! Found. Comput. Math. 7, 303 (2007). At the current point `x` on the
//! manifold the model
//!
//! `m(eta) = f(x) + <grad f(x), eta> + 1/2 <Hess f(x)[eta], eta>`,
//! for `eta` in the tangent space `T_x M`,
//!
//! is minimised inside the ball `||eta|| <= Delta` by the Steihaug–Toint
//! truncated CG iteration run entirely in the tangent space: every
//! Hessian–vector product is projected back with [`Manifold::project`], so
//! for an embedded submanifold with the Euclidean metric the iteration sees
//! `proj_x (H eta)`, which is the Riemannian Hessian when the tangent
//! projector is constant along the retraction (linear tangent spaces) and
//! its Gauss–Newton part otherwise. The caller retracts the step, measures
//! the actual decrease and feeds the ratio to [`RtrRadius::update`].
//!
//! The constants in the radius update are the theorem's: accept when
//! `rho > rho' = 0.1`, shrink by 1/4 below `rho = 1/4`, grow by 2 above
//! `rho = 3/4` at the boundary, never past `Delta_bar`. Nothing here is a
//! per-fixture number.
//!
//! For `f(x) = x^2` at `x = 1`, the model has its minimizer at `eta = -1`:
//! ```
//! use ndarray::array;
//! use rgmin::rtr::truncated_cg_projected;
//! let result = truncated_cg_projected(
//!     |v| v.clone(), array![2.0].view(), |v| v * 2.0, 2.0, 1.0, 0.1, 10,
//! );
//! assert!((result.eta[0] + 1.0).abs() < 1e-12);
//! assert!((result.model_decrease - 1.0).abs() < 1e-12);
//! ```

use ndarray::{Array1, ArrayView1};

use crate::manifold::Manifold;
use crate::vecops::{dot, nrm2};

/// Why the truncated CG iteration stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcgStop {
    /// The model has negative curvature along the search direction; the step
    /// was followed to the trust boundary.
    NegativeCurvature = 0,
    /// The CG iterate left the trust ball; the step was cut at the boundary.
    Boundary = 1,
    /// The residual met the inner stopping rule
    /// `||r_j|| <= ||r_0|| min(||r_0||^theta, kappa)`.
    Residual = 2,
    /// The iteration budget ran out.
    MaxIter = 3,
    /// The projected gradient is zero: nothing to do.
    ZeroGradient = 4,
}

/// Result of one tangent-space subproblem.
#[derive(Clone, Debug)]
pub struct TcgResult {
    /// Tangent step `eta`.
    pub eta: Array1<f64>,
    /// Model decrease `m(0) - m(eta) = -(<g, eta> + 1/2 <H eta, eta>)`,
    /// evaluated with one extra Hessian-vector product on the returned step.
    pub model_decrease: f64,
    /// Condition terminating the conjugate-gradient subproblem.
    pub stop: TcgStop,
    /// Number of conjugate-gradient iterations performed.
    pub iterations: usize,
}

/// Positive root `tau` of `||eta + tau delta|| = radius`.
fn boundary_tau(eta: &Array1<f64>, delta: &Array1<f64>, radius: f64) -> f64 {
    let dd = dot(delta.view(), delta.view());
    if dd <= 0.0 {
        return 0.0;
    }
    let ed = dot(eta.view(), delta.view());
    let ee = dot(eta.view(), eta.view());
    let disc = (ed * ed - dd * (ee - radius * radius)).max(0.0);
    (-ed + disc.sqrt()) / dd
}

/// Truncated CG for the trust-region subproblem on a tangent space given by
/// its projector `project`.
///
/// `grad` is the ambient gradient and `hvp(v)` the ambient Hessian-vector
/// product; both are projected here. `theta` and `kappa` set the
/// superlinear inner stopping rule (`theta = 1`, `kappa = 0.1` are the
/// reference defaults).
pub fn truncated_cg_projected<P, F>(
    mut project: P,
    grad: ArrayView1<f64>,
    mut hvp: F,
    radius: f64,
    theta: f64,
    kappa: f64,
    maxiter: usize,
) -> TcgResult
where
    P: FnMut(&Array1<f64>) -> Array1<f64>,
    F: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let n = grad.len();
    let g = project(&grad.to_owned());
    let mut eta = Array1::<f64>::zeros(n);
    let mut r = g.clone();
    let mut delta = -&r;
    let r0 = nrm2(r.view());
    if !(r0 > 0.0) {
        return TcgResult {
            eta,
            model_decrease: 0.0,
            stop: TcgStop::ZeroGradient,
            iterations: 0,
        };
    }
    let tol = r0 * r0.powf(theta).min(kappa);
    let mut rr = r0 * r0;
    let mut stop = TcgStop::MaxIter;
    let mut iterations = 0;
    for _ in 0..maxiter {
        iterations += 1;
        let hd = project(&hvp(&delta));
        let kap = dot(delta.view(), hd.view());
        if kap <= 0.0 {
            let tau = boundary_tau(&eta, &delta, radius);
            eta = &eta + &(&delta * tau);
            stop = TcgStop::NegativeCurvature;
            break;
        }
        let alpha = rr / kap;
        let eta_new = &eta + &(&delta * alpha);
        if nrm2(eta_new.view()) >= radius {
            let tau = boundary_tau(&eta, &delta, radius);
            eta = &eta + &(&delta * tau);
            stop = TcgStop::Boundary;
            break;
        }
        eta = eta_new;
        r = &r + &(&hd * alpha);
        let rr_new = dot(r.view(), r.view());
        if rr_new.sqrt() <= tol {
            stop = TcgStop::Residual;
            break;
        }
        let beta = rr_new / rr;
        rr = rr_new;
        delta = &(-&r) + &(&delta * beta);
    }
    let heta = project(&hvp(&eta));
    let model_decrease = -(dot(g.view(), eta.view()) + 0.5 * dot(eta.view(), heta.view()));
    TcgResult {
        eta,
        model_decrease,
        stop,
        iterations,
    }
}

/// Trust radius state with the reference update rule.
#[derive(Clone, Copy, Debug)]
pub struct RtrRadius {
    /// Radius of the current tangent-space trust region.
    pub radius: f64,
    /// Upper bound on the trust radius.
    pub radius_max: f64,
}

impl RtrRadius {
    /// Reference initialisation: `Delta_0 = Delta_bar / 8`.
    pub fn new(radius_max: f64) -> Self {
        let rm = radius_max.max(f64::MIN_POSITIVE);
        Self {
            radius: rm / 8.0,
            radius_max: rm,
        }
    }

    /// Reduction ratio `rho = actual / predicted` and the acceptance and
    /// radius update of Absil-Baker-Gallivan Algorithm 1. Returns whether
    /// the step is accepted (`rho > 1/10`). A non-positive predicted
    /// decrease is treated as a failed model (`rho = -inf`).
    pub fn update(&mut self, actual_decrease: f64, model_decrease: f64, eta_norm: f64) -> bool {
        let rho = if model_decrease > 0.0 {
            actual_decrease / model_decrease
        } else {
            f64::NEG_INFINITY
        };
        if rho < 0.25 {
            self.radius = (0.25 * self.radius).max(f64::MIN_POSITIVE);
        } else if rho > 0.75 && eta_norm >= self.radius * (1.0 - 1e-12) {
            self.radius = (2.0 * self.radius).min(self.radius_max);
        }
        rho > 0.1
    }
}

/// Solve a tangent-space trust-region subproblem on an embedded manifold.
/// The Hessian callback supplies the ambient action, projected at `x`.
pub fn truncated_cg<M, F>(
    manifold: &M,
    x: ArrayView1<f64>,
    grad: ArrayView1<f64>,
    hvp: F,
    radius: f64,
    theta: f64,
    kappa: f64,
    maxiter: usize,
) -> TcgResult
where
    M: Manifold + ?Sized,
    F: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let anchor = x.to_owned();
    truncated_cg_projected(
        |v| manifold.project(&anchor, v),
        grad,
        hvp,
        radius,
        theta,
        kappa,
        maxiter,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{Euclidean, Sphere};
    use ndarray::{Array2, array};

    fn quad_hvp(h: &Array2<f64>) -> impl Fn(&Array1<f64>) -> Array1<f64> + '_ {
        move |v| h.dot(v)
    }

    #[test]
    fn euclidean_tcg_reaches_the_newton_point_inside_the_ball() {
        let h = Array2::from_diag(&array![1.0, 2.0, 3.0]);
        let g = array![1.0, -2.0, 0.5];
        let x = Array1::zeros(3);
        let r = truncated_cg(
            &Euclidean,
            x.view(),
            g.view(),
            quad_hvp(&h),
            100.0,
            1.0,
            0.1,
            50,
        );
        let newton = array![-1.0, 1.0, -0.5 / 3.0];
        assert!(
            (r.eta.clone() - newton.clone()).mapv(f64::abs).sum() < 1e-10,
            "{:?}",
            r.eta
        );
        let expected = -(g.dot(&newton) + 0.5 * newton.dot(&h.dot(&newton)));
        assert!((r.model_decrease - expected).abs() < 1e-10);
        assert_eq!(r.stop, TcgStop::Residual);
    }

    #[test]
    fn negative_curvature_runs_to_the_boundary() {
        let h = Array2::from_diag(&array![-1.0, 2.0]);
        let g = array![1.0, 0.0];
        let x = Array1::zeros(2);
        let r = truncated_cg(
            &Euclidean,
            x.view(),
            g.view(),
            quad_hvp(&h),
            0.5,
            1.0,
            0.1,
            50,
        );
        assert_eq!(r.stop, TcgStop::NegativeCurvature);
        assert!((nrm2(r.eta.view()) - 0.5).abs() < 1e-12);
        assert!(r.eta[0] < 0.0);
    }

    #[test]
    fn sphere_rtr_finds_the_smallest_eigenvector() {
        // f(x) = x^T A x on S^2: minimiser is the eigenvector of the smallest
        // eigenvalue. Euclidean gradient 2 A x, Hessian 2 A; the projection
        // in truncated_cg supplies the Riemannian versions up to the
        // Weingarten term, which the rho test absorbs.
        let a = Array2::from_diag(&array![3.0, 1.0, 2.0]);
        let sphere = Sphere;
        let mut x = array![0.6, 0.48, 0.64];
        x /= nrm2(x.view());
        let f = |x: &Array1<f64>| x.dot(&a.dot(x));
        let mut radius = RtrRadius::new(1.0);
        for _ in 0..100 {
            let g = &a.dot(&x) * 2.0;
            let r = truncated_cg(
                &sphere,
                x.view(),
                g.view(),
                |v| &a.dot(v) * 2.0,
                radius.radius,
                1.0,
                0.1,
                20,
            );
            if nrm2(sphere.project(&x, &g).view()) < 1e-10 {
                break;
            }
            let x_try = sphere.retract(&x, &r.eta);
            let accept = radius.update(f(&x) - f(&x_try), r.model_decrease, nrm2(r.eta.view()));
            if accept {
                x = x_try;
            }
        }
        assert!((f(&x) - 1.0).abs() < 1e-8, "f = {}", f(&x));
        assert!(x[1].abs() > 1.0 - 1e-6);
    }

    #[test]
    fn radius_update_follows_the_reference_rule() {
        let mut r = RtrRadius::new(8.0);
        assert_eq!(r.radius, 1.0);
        assert!(!r.update(0.1, 1.0, 0.5)); // rho = 0.1: not accepted, shrink
        assert!((r.radius - 0.25).abs() < 1e-15);
        assert!(r.update(0.9, 1.0, 0.25)); // rho = 0.9 at the boundary: grow
        assert!((r.radius - 0.5).abs() < 1e-15);
        assert!(r.update(0.5, 1.0, 0.1)); // rho = 0.5 interior: unchanged
        assert!((r.radius - 0.5).abs() < 1e-15);
        assert!(!r.update(1.0, 0.0, 0.1)); // no predicted decrease: reject
    }
}
