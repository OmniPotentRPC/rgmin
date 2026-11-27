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
//! the actual decrease and feeds the ratio to [`crate::rtr::RtrRadius::update`].
//!
//! The constants in the radius update are the theorem's: accept when
//! `rho > rho' = 0.1`, shrink by 1/4 below `rho = 1/4`, grow by 2 above
//! `rho = 3/4` at the boundary, never past `Delta_bar`. Nothing here is a
//! per-fixture number.
//!
//! [`crate::rtr::minimize_rtr`] runs the whole loop in the
//! metric of the manifold: the gradient through
//! [`Manifold::egrad2rgrad`], the Hessian through
//! [`Manifold::ehess2rhess`] with its Weingarten term where the kind
//! supplies one, and every inner product and the trust ball in
//! [`Manifold::inner`].
//! [`crate::rtr::truncated_cg_riemannian`] is one
//! of its subproblems, for a driver that measures the decrease itself.
//! [`crate::rtr::minimize_rtr_gauged`] moves to
//! another representation of the same point after every accepted step.
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
use crate::vecops::dot;

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
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TcgResult {
    /// Tangent step `eta`.
    pub eta: Array1<f64>,
    /// Model decrease `m(0) - m(eta) = -(<g, eta> + 1/2 <H eta, eta>)`,
    /// evaluated with one extra Hessian-vector product on the returned step.
    ///
    /// NaN when a Hessian action was not finite. The iteration stops at the
    /// first such action, `eta` is the last finite iterate, and `stop` is
    /// [`TcgStop::MaxIter`].
    pub model_decrease: f64,
    /// Condition terminating the conjugate-gradient subproblem.
    pub stop: TcgStop,
    /// Number of conjugate-gradient iterations performed.
    pub iterations: usize,
}

/// Positive root `tau` of `||eta + tau delta|| = radius` in `inner`.
fn boundary_tau<I>(inner: &I, eta: &Array1<f64>, delta: &Array1<f64>, radius: f64) -> f64
where
    I: Fn(&Array1<f64>, &Array1<f64>) -> f64,
{
    let dd = inner(delta, delta);
    if dd <= 0.0 {
        return 0.0;
    }
    let ed = inner(eta, delta);
    let ee = inner(eta, eta);
    let disc = (ed * ed - dd * (ee - radius * radius)).max(0.0);
    (-ed + disc.sqrt()) / dd
}

fn euclidean(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    dot(a.view(), b.view())
}

/// Trust radius and the inner stopping rule of one subproblem.
#[derive(Clone, Copy)]
struct TcgLimits {
    radius: f64,
    theta: f64,
    kappa: f64,
    maxiter: usize,
}

/// Truncated CG for the trust-region subproblem on a tangent space given by
/// its projector `project`.
///
/// `grad` is the ambient gradient and `hvp(v)` the ambient Hessian-vector
/// product; both are projected here. `theta` and `kappa` set the
/// superlinear inner stopping rule (`theta = 1`, `kappa = 0.1` are the
/// reference defaults).
pub fn truncated_cg_projected<P, F>(
    project: P,
    grad: ArrayView1<f64>,
    hvp: F,
    radius: f64,
    theta: f64,
    kappa: f64,
    maxiter: usize,
) -> TcgResult
where
    P: FnMut(&Array1<f64>) -> Array1<f64>,
    F: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let limits = TcgLimits {
        radius,
        theta,
        kappa,
        maxiter,
    };
    tcg(euclidean, project, grad, hvp, limits)
}

/// Steihaug–Toint truncated CG with every inner product and the trust
/// ball measured in `inner`.
fn tcg<I, P, F>(
    inner: I,
    mut project: P,
    grad: ArrayView1<f64>,
    mut hvp: F,
    limits: TcgLimits,
) -> TcgResult
where
    I: Fn(&Array1<f64>, &Array1<f64>) -> f64,
    P: FnMut(&Array1<f64>) -> Array1<f64>,
    F: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let TcgLimits {
        radius,
        theta,
        kappa,
        maxiter,
    } = limits;
    let n = grad.len();
    let g = project(&grad.to_owned());
    let mut eta = Array1::<f64>::zeros(n);
    let mut r = g.clone();
    let mut delta = -&r;
    let r0 = inner(&r, &r).sqrt();
    if r0.is_nan() || r0 <= 0.0 {
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
        let kap = inner(&delta, &hd);
        if !kap.is_finite() || hd.iter().any(|c| !c.is_finite()) {
            return TcgResult {
                eta,
                model_decrease: f64::NAN,
                stop: TcgStop::MaxIter,
                iterations,
            };
        }
        if kap <= 0.0 {
            let tau = boundary_tau(&inner, &eta, &delta, radius);
            eta = &eta + &(&delta * tau);
            stop = TcgStop::NegativeCurvature;
            break;
        }
        let alpha = rr / kap;
        let eta_new = &eta + &(&delta * alpha);
        if inner(&eta_new, &eta_new).sqrt() >= radius {
            let tau = boundary_tau(&inner, &eta, &delta, radius);
            eta = &eta + &(&delta * tau);
            stop = TcgStop::Boundary;
            break;
        }
        eta = eta_new;
        r = &r + &(&hd * alpha);
        let rr_new = inner(&r, &r);
        if rr_new.sqrt() <= tol {
            stop = TcgStop::Residual;
            break;
        }
        let beta = rr_new / rr;
        rr = rr_new;
        delta = &(-&r) + &(&delta * beta);
    }
    let heta = project(&hvp(&eta));
    let model_decrease = if heta.iter().all(|c| c.is_finite()) {
        -(inner(&g, &eta) + 0.5 * inner(&eta, &heta))
    } else {
        f64::NAN
    };
    TcgResult {
        eta,
        model_decrease,
        stop,
        iterations,
    }
}

/// Trust radius state with the reference update rule.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct RtrRadius {
    /// Radius of the current tangent-space trust region.
    pub radius: f64,
    /// Upper bound on the trust radius.
    pub radius_max: f64,
}

with_setters!(RtrRadius {
    with_radius => radius: val f64,
    with_radius_max => radius_max: val f64,
});

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
    ///
    /// A radius above [`Self::radius_max`], which the public fields allow,
    /// is clamped to the maximum before the update, so the radius left
    /// behind never exceeds it whatever the ratio is. The growth test then
    /// compares `eta_norm` with the clamped radius.
    pub fn update(&mut self, actual_decrease: f64, model_decrease: f64, eta_norm: f64) -> bool {
        self.radius = self.radius.min(self.radius_max);
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
#[expect(
    clippy::too_many_arguments,
    reason = "public signature; grouping the arguments would break callers"
)]
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

/// Options for [`minimize_rtr`]. The acceptance and radius constants are
/// those of [`RtrRadius`]. The defaults are manopt `trustregions`'.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RtrOptions {
    /// Upper bound \(\bar\Delta\) on the trust radius; the first radius is
    /// one eighth of it. `None` takes [`Manifold::typical_dist`].
    pub radius_max: Option<f64>,
    /// Stop when the Riemannian gradient is at most this long.
    pub grad_tol: f64,
    /// Outer iteration budget.
    pub max_iter: usize,
    /// Inner CG budget. `None` takes the packed length, which bounds the
    /// dimension of the tangent space.
    pub cg_max: Option<usize>,
    /// Exponent \(\theta\) of the inner stopping rule.
    pub theta: f64,
    /// Factor \(\kappa\) of the inner stopping rule.
    pub kappa: f64,
    /// Both decreases in the ratio \(\rho\) gain \(\max(1, |f|)\,
    /// \varepsilon\) times this, so steps whose decrease is below the
    /// rounding of `f` read \(\rho \approx 1\) instead of noise.
    pub rho_regularization: f64,
}

impl Default for RtrOptions {
    fn default() -> Self {
        Self {
            radius_max: None,
            grad_tol: 1e-6,
            max_iter: 1000,
            cg_max: None,
            theta: 1.0,
            kappa: 0.1,
            rho_regularization: 1e3,
        }
    }
}

impl RtrOptions {
    /// Set [`Self::radius_max`].
    #[must_use]
    pub fn with_radius_max(mut self, radius_max: f64) -> Self {
        self.radius_max = Some(radius_max);
        self
    }

    /// Set [`Self::cg_max`].
    #[must_use]
    pub fn with_cg_max(mut self, cg_max: usize) -> Self {
        self.cg_max = Some(cg_max);
        self
    }
}

with_setters!(RtrOptions {
    with_grad_tol => grad_tol: val f64,
    with_max_iter => max_iter: val usize,
    with_theta => theta: val f64,
    with_kappa => kappa: val f64,
    with_rho_regularization => rho_regularization: val f64,
});

/// Result of [`minimize_rtr`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RtrMin {
    /// Point on the manifold.
    pub x: Array1<f64>,
    /// Objective value at `x`.
    pub value: f64,
    /// Length of the Riemannian gradient at `x` in the manifold metric.
    pub grad_norm: f64,
    /// Outer iterations performed.
    pub iterations: usize,
    /// Steps the radius update accepted.
    pub accepted: usize,
    /// Conjugate-gradient iterations over all subproblems.
    pub inner_iterations: usize,
}

/// Solve the trust-region subproblem at `x` in the metric of `manifold`.
///
/// The model is the one [`minimize_rtr`] builds: the gradient is
/// [`Manifold::egrad2rgrad`] of the Euclidean gradient `egrad`, the
/// Hessian is [`Manifold::ehess2rhess`] of the Euclidean action
/// `ehess(v)` or, for a kind without the Weingarten term, its
/// Gauss–Newton part, and every inner product and the ball of radius
/// `radius` are [`Manifold::inner`]. On a product with a
/// [`crate::manifold::TangentMetric`] this is the preconditioned
/// subproblem. `opts` supplies [`RtrOptions::theta`],
/// [`RtrOptions::kappa`] and [`RtrOptions::cg_max`].
///
/// A driver that measures the decrease itself, for example from sampled
/// free-energy differences, retracts `eta`, measures, and passes
/// `manifold.norm(x, &eta)` to [`RtrRadius::update`].
pub fn truncated_cg_riemannian<M, H>(
    manifold: &M,
    x: &Array1<f64>,
    egrad: &Array1<f64>,
    ehess: H,
    radius: f64,
    opts: &RtrOptions,
) -> TcgResult
where
    M: Manifold + ?Sized,
    H: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let rgrad = manifold.egrad2rgrad(x, egrad);
    tcg_in_metric(manifold, x, egrad, &rgrad, ehess, radius, opts)
}

/// [`truncated_cg_riemannian`] with the Riemannian gradient `rgrad` of
/// `egrad` already formed.
fn tcg_in_metric<M, H>(
    manifold: &M,
    x: &Array1<f64>,
    egrad: &Array1<f64>,
    rgrad: &Array1<f64>,
    mut ehess: H,
    radius: f64,
    opts: &RtrOptions,
) -> TcgResult
where
    M: Manifold + ?Sized,
    H: FnMut(&Array1<f64>) -> Array1<f64>,
{
    tcg(
        |a, b| manifold.inner(x, a, b),
        |v| manifold.project(x, v),
        rgrad.view(),
        |v| {
            let hv = ehess(v);
            manifold
                .ehess2rhess(x, egrad, &hv, v)
                .unwrap_or_else(|| manifold.egrad2rgrad(x, &hv))
        },
        TcgLimits {
            radius,
            theta: opts.theta,
            kappa: opts.kappa,
            maxiter: opts.cg_max.unwrap_or(x.len()).max(1),
        },
    )
}

/// Minimise `value` on `manifold` by the Absil–Baker–Gallivan trust
/// region.
///
/// `egrad(x)` is the Euclidean gradient and `ehess(x, v)` the Euclidean
/// Hessian action, both in the packed ambient coordinates. The model is
/// built in the Riemannian metric: the gradient by
/// [`Manifold::egrad2rgrad`], the Hessian by [`Manifold::ehess2rhess`]
/// or, for a kind without the Weingarten term, by
/// [`Manifold::egrad2rgrad`] of the ambient action, its Gauss–Newton part.
/// A weighted [`crate::manifold::ProductManifold`] therefore runs in its
/// own metric, and [`RtrOptions::radius_max`] defaults to its typical
/// distance.
pub fn minimize_rtr<M, FV, GG, HH>(
    manifold: &M,
    x0: Array1<f64>,
    value: FV,
    egrad: GG,
    ehess: HH,
    opts: RtrOptions,
) -> RtrMin
where
    M: Manifold + ?Sized,
    FV: FnMut(&Array1<f64>) -> f64,
    GG: FnMut(&Array1<f64>) -> Array1<f64>,
    HH: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
{
    rtr(manifold, x0, value, egrad, ehess, |_| false, opts)
}

/// [`minimize_rtr`] with a gauge move after every accepted step.
///
/// `gauge(x)` replaces the iterate with another representation of the
/// same point: a map that leaves `value` unchanged, such as re-centring
/// the pivot of a hyperplane on the plane. The step is accepted on the
/// value before the move. The value is evaluated again when the gauge
/// moved `x`, and the gradient always is, so the next model, the stopping
/// test and the returned [`RtrMin`] are those of the moved point. A gauge
/// that changes the value still runs, without the monotone decrease.
pub fn minimize_rtr_gauged<M, FV, GG, HH, GA>(
    manifold: &M,
    x0: Array1<f64>,
    value: FV,
    egrad: GG,
    ehess: HH,
    mut gauge: GA,
    opts: RtrOptions,
) -> RtrMin
where
    M: Manifold + ?Sized,
    FV: FnMut(&Array1<f64>) -> f64,
    GG: FnMut(&Array1<f64>) -> Array1<f64>,
    HH: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    GA: FnMut(&mut Array1<f64>),
{
    let moved = |x: &mut Array1<f64>| {
        let before = x.clone();
        gauge(x);
        *x != before
    };
    rtr(manifold, x0, value, egrad, ehess, moved, opts)
}

/// The trust-region loop. `moved(x)` applies the gauge after an accepted
/// step and reports whether it changed `x`.
fn rtr<M, FV, GG, HH, GA>(
    manifold: &M,
    x0: Array1<f64>,
    mut value: FV,
    mut egrad: GG,
    mut ehess: HH,
    mut moved: GA,
    opts: RtrOptions,
) -> RtrMin
where
    M: Manifold + ?Sized,
    FV: FnMut(&Array1<f64>) -> f64,
    GG: FnMut(&Array1<f64>) -> Array1<f64>,
    HH: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    GA: FnMut(&mut Array1<f64>) -> bool,
{
    let radius_max = opts
        .radius_max
        .unwrap_or_else(|| manifold.typical_dist(x0.len()));
    let radius_max = if radius_max.is_finite() && radius_max > 0.0 {
        radius_max
    } else {
        1.0
    };
    let mut radius = RtrRadius::new(radius_max);
    let mut x = x0;
    let mut f = value(&x);
    let mut g = egrad(&x);
    let mut rgrad = manifold.egrad2rgrad(&x, &g);
    let mut gnorm = manifold.norm(&x, &rgrad);
    let mut iterations = 0;
    let mut accepted = 0;
    let mut inner_iterations = 0;
    while iterations < opts.max_iter && gnorm.is_finite() && gnorm > opts.grad_tol {
        iterations += 1;
        let step = tcg_in_metric(
            manifold,
            &x,
            &g,
            &rgrad,
            |v| ehess(&x, v),
            radius.radius,
            &opts,
        );
        inner_iterations += step.iterations;
        let x_try = manifold.retract(&x, &step.eta);
        let f_try = value(&x_try);
        let shift = f.abs().max(1.0) * f64::EPSILON * opts.rho_regularization;
        let actual = if f_try.is_finite() {
            f - f_try + shift
        } else {
            f64::NEG_INFINITY
        };
        let predicted = step.model_decrease + shift;
        if radius.update(actual, predicted, manifold.norm(&x, &step.eta)) {
            x = x_try;
            f = if moved(&mut x) { value(&x) } else { f_try };
            g = egrad(&x);
            rgrad = manifold.egrad2rgrad(&x, &g);
            gnorm = manifold.norm(&x, &rgrad);
            accepted += 1;
        }
    }
    RtrMin {
        x,
        value: f,
        grad_norm: gnorm,
        iterations,
        accepted,
        inner_iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{
        Euclidean, ManifoldKind, ProductFactor, ProductManifold, Sphere, TangentMetric,
    };
    use crate::vecops::nrm2;
    use ndarray::{Array2, array};
    use std::sync::Arc;

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
    fn minimize_rtr_finds_the_smallest_eigenvector() {
        let a = Array2::from_diag(&array![3.0, 1.0, 2.0]);
        let mut x = array![0.6, 0.48, 0.64];
        x /= nrm2(x.view());
        let out = minimize_rtr(
            &Sphere,
            x,
            |z| z.dot(&a.dot(z)),
            |z| &a.dot(z) * 2.0,
            |_z, v| &a.dot(v) * 2.0,
            RtrOptions::default().with_grad_tol(1e-12),
        );
        assert!(out.grad_norm <= 1e-12, "grad {}", out.grad_norm);
        assert!((out.value - 1.0).abs() < 1e-12, "f = {}", out.value);
        assert!(out.x[1].abs() > 1.0 - 1e-12);
        assert!(out.iterations < 20, "{} iterations", out.iterations);
    }

    #[test]
    fn the_riemannian_step_runs_in_the_tangent_metric() {
        struct Curvature;
        impl TangentMetric for Curvature {
            fn apply(&self, _x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
                v * &array![1.0, 10.0, 100.0]
            }
            fn solve(&self, _x: &Array1<f64>, r: &Array1<f64>) -> Array1<f64> {
                r / &array![1.0, 10.0, 100.0]
            }
        }
        let a = array![1.0, 10.0, 100.0];
        let factor = ProductFactor::new(ManifoldKind::Euclidean, 3);
        let plain = ProductManifold::new([factor.clone()]).unwrap();
        let newton = ProductManifold::new([factor.with_metric(Arc::new(Curvature))]).unwrap();
        let x = array![1.0, 1.0, 1.0];
        let g = &a * &x;
        let opts = RtrOptions::default().with_kappa(1e-10);
        let step = truncated_cg_riemannian(&newton, &x, &g, |v| &a * v, 100.0, &opts);
        assert_eq!((step.stop, step.iterations), (TcgStop::Residual, 1));
        assert!((&step.eta + &x).iter().all(|e| e.abs() < 1e-14));
        assert!((step.model_decrease - 0.5 * g.dot(&x)).abs() < 1e-12);
        let unscaled = truncated_cg_riemannian(&plain, &x, &g, |v| &a * v, 100.0, &opts);
        assert_eq!(unscaled.iterations, 3, "{unscaled:?}");
    }

    #[test]
    fn the_gauge_moves_after_every_accepted_step() {
        // f depends on x0 + x1 only, so averaging the two leaves it unchanged.
        let f = |z: &Array1<f64>| (z[0] + z[1] - 1.0).powi(2) + z[2] * z[2];
        let g = |z: &Array1<f64>| {
            let r = 2.0 * (z[0] + z[1] - 1.0);
            array![r, r, 2.0 * z[2]]
        };
        let h = |_z: &Array1<f64>, v: &Array1<f64>| {
            let s = 2.0 * (v[0] + v[1]);
            array![s, s, 2.0 * v[2]]
        };
        let mut moves = 0;
        let gauge = |z: &mut Array1<f64>| {
            moves += 1;
            let mean = 0.5 * (z[0] + z[1]);
            z[0] = mean;
            z[1] = mean;
        };
        let opts = RtrOptions::default().with_grad_tol(1e-12);
        let out = minimize_rtr_gauged(&Euclidean, array![3.0, -1.0, 2.0], f, g, h, gauge, opts);
        assert_eq!(moves, out.accepted);
        assert!(out.grad_norm <= 1e-12, "{out:?}");
        assert_eq!(out.x[0], out.x[1]);
        assert!((out.x[0] - 0.5).abs() < 1e-12 && out.x[2].abs() < 1e-12);
    }

    #[test]
    fn truncated_cg_known_answers_on_small_quadratics() {
        // 1D positive curvature inside a large ball: the Newton step.
        // H = 2, g = 2, eta = -1, model decrease = 1, residual stop.
        let residual = truncated_cg_projected(
            |v| v.clone(),
            array![2.0].view(),
            |v| v * 2.0,
            10.0,
            1.0,
            0.1,
            5,
        );
        assert_eq!(residual.stop, TcgStop::Residual);
        assert_eq!(residual.iterations, 1);
        assert!((residual.eta[0] + 1.0).abs() < 1e-12);
        assert!((residual.model_decrease - 1.0).abs() < 1e-12);

        // Same model, trust radius 0.25. The full step has length 0.5
        // when H = 4 and g = 2, so the step stops at eta = -0.25.
        // Model decrease is 0.375.
        let boundary = truncated_cg_projected(
            |v| v.clone(),
            array![2.0].view(),
            |v| v * 4.0,
            0.25,
            1.0,
            0.1,
            5,
        );
        assert_eq!(boundary.stop, TcgStop::Boundary);
        assert!((boundary.eta[0] + 0.25).abs() < 1e-12);
        assert!((boundary.model_decrease - 0.375).abs() < 1e-12);

        // Negative curvature runs to the boundary along -g.
        // H = -2, g = 1, radius 0.4, eta = -0.4, model decrease 0.56.
        let curved = truncated_cg_projected(
            |v| v.clone(),
            array![1.0].view(),
            |v| v * -2.0,
            0.4,
            1.0,
            0.1,
            5,
        );
        assert_eq!(curved.stop, TcgStop::NegativeCurvature);
        assert!((curved.eta[0] + 0.4).abs() < 1e-12);
        assert!((curved.model_decrease - 0.56).abs() < 1e-12);

        let flat = truncated_cg_projected(
            |v| v.clone(),
            array![0.0, 0.0].view(),
            |v| v.clone(),
            1.0,
            1.0,
            0.1,
            4,
        );
        assert_eq!(flat.stop, TcgStop::ZeroGradient);
        assert_eq!(flat.iterations, 0);
        assert_eq!(flat.model_decrease, 0.0);
    }

    #[test]
    fn radius_update_follows_the_absil_baker_gallivan_rules() {
        let mut radius = RtrRadius {
            radius: 1.0,
            radius_max: 8.0,
        };
        assert!(!radius.update(0.1, 1.0, 0.5));
        assert!((radius.radius - 0.25).abs() < 1e-15);
        assert!(radius.update(0.9, 1.0, 0.25));
        assert!((radius.radius - 0.5).abs() < 1e-15);
        assert!(radius.update(0.5, 1.0, 0.1));
        assert!((radius.radius - 0.5).abs() < 1e-15);
        assert!(!radius.update(1.0, 0.0, 0.1));
        assert!((radius.radius - 0.125).abs() < 1e-15);
        radius.radius = 3.0;
        radius.radius_max = 4.0;
        assert!(radius.update(0.9, 1.0, 3.0));
        assert!((radius.radius - 4.0).abs() < 1e-15);
        assert!(radius.update(0.75, 1.0, 4.0));
        assert!((radius.radius - 4.0).abs() < 1e-15);
        assert!(radius.update(0.25, 1.0, 1.0));
        assert!((radius.radius - 4.0).abs() < 1e-15);
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

    #[test]
    fn radius_above_the_maximum_is_clamped() {
        let mut radius = RtrRadius {
            radius: 8.0,
            radius_max: 1.0,
        };
        assert!(radius.update(0.5, 1.0, 0.0));
        assert_eq!(radius.radius, 1.0);

        let mut radius = RtrRadius {
            radius: 8.0,
            radius_max: 1.0,
        };
        assert!(!radius.update(0.1, 1.0, 0.0));
        assert!((radius.radius - 0.25).abs() < 1e-15);

        let mut radius = RtrRadius {
            radius: 8.0,
            radius_max: 1.0,
        };
        assert!(radius.update(0.9, 1.0, 1.0));
        assert_eq!(radius.radius, 1.0);
    }

    #[test]
    fn tcg_stops_on_a_non_finite_hessian_action() {
        let calls = std::cell::Cell::new(0usize);
        let result = truncated_cg_projected(
            |v| v.clone(),
            ndarray::array![1.0, 1.0].view(),
            |v| {
                calls.set(calls.get() + 1);
                v.mapv(|_| f64::NAN)
            },
            10.0,
            1.0,
            0.1,
            50,
        );
        assert_eq!(calls.get(), 1, "tCG kept calling a poisoned Hessian");
        assert!(result.model_decrease.is_nan());
        assert!(result.eta.iter().all(|c| c.is_finite()));
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn tcg_stops_when_the_action_turns_infinite_midway() {
        let calls = std::cell::Cell::new(0usize);
        let result = truncated_cg_projected(
            |v| v.clone(),
            ndarray::array![1.0, 0.5, 0.25].view(),
            |v| {
                calls.set(calls.get() + 1);
                if calls.get() >= 2 {
                    v.mapv(|_| f64::INFINITY)
                } else {
                    v * 0.1
                }
            },
            100.0,
            1.0,
            1e-12,
            50,
        );
        assert_eq!(calls.get(), 2);
        assert!(result.model_decrease.is_nan());
        assert!(result.eta.iter().all(|c| c.is_finite()));
    }
}
