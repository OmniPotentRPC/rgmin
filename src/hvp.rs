//! Matrix-free Newton: Hessian action instead of a Hessian matrix.
//!
//! The dense path in [`crate::newton`] factors an explicit `n x n`
//! Hessian, which prices every step at O(n^3) flops and O(n^2) memory.
//! TAO's BNK family never forms the matrix: it asks the objective for
//! the action `H(x) v` and runs Steihaug-Toint conjugate gradients
//! inside a trust region, so cost scales with the number of actions,
//! not with n^3. This module is that path for eindir objectives.
//!
//! Nocedal and Wright, *Numerical Optimization*, algorithms 4.1 and
//! 7.2, <https://doi.org/10.1007/978-0-387-40065-5>.
//! Steihaug, *The Conjugate Gradient Method and Trust Regions in Large
//! Scale Optimization*, <https://doi.org/10.1137/0720042>.

use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, Array2, ArrayView1};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::report::Report;
use crate::rigid::project_out_rot_trans;
use crate::scg::DirectionalCurvature;
use crate::step::{scale_step, scale_step_atom};
use crate::trust::{reduction_ratio, update_radius};
use crate::vecops::{axpy, dot, nrm2};

/// Hessian action `H(x) v` without forming the matrix.
pub trait HessianVector: DifferentiableObjective<f64> {
    /// `grad^2 f(x) . v`.
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64>;
}

/// Every Hessian action supplies a directional curvature for free:
/// `d . H d` is one action and one dot, so any [`HessianVector`]
/// objective can drive [`crate::minimize_scg_exact`] as well.
impl<T: HessianVector + ?Sized> DirectionalCurvature for T {
    fn directional_curvature(&self, x: ArrayView1<f64>, d: ArrayView1<f64>) -> Option<f64> {
        let hd = self.hessian_vector(x, d);
        Some(dot(d, hd.view()))
    }
}

/// Closure adapter: fused `(f, g)` plus a Hessian action.
pub struct HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    f: F,
    hvp: H,
    bounds: eindir_core::Bounds<f64>,
}

impl<F, H> HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    /// Wide-box oracle of dimension `dim`.
    pub fn unbounded(dim: usize, f: F, hvp: H) -> Self {
        const LO: f64 = -1e12;
        const HI: f64 = 1e12;
        Self {
            f,
            hvp,
            bounds: eindir_core::Bounds::new(
                Array1::from_elem(dim, LO),
                Array1::from_elem(dim, HI),
                0.0,
            ),
        }
    }
}

impl<F, H> eindir_core::Objective<f64> for HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        (self.f)(x).0
    }
}

impl<F, H> eindir_core::Gradient<f64> for HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        (self.f)(x).1
    }
}

impl<F, H> DifferentiableObjective<f64> for HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.f)(x)
    }
}

impl<F, H> HessianVector for HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        (self.hvp)(x, v)
    }
}

/// A Hessian action from central gradient differences, for objectives
/// with no analytic action: `H v ~ (g(x + e v) - g(x - e v)) / 2e`.
/// Two gradient calls per action.
pub struct FdHvp<'a, O: DifferentiableObjective<f64> + ?Sized> {
    obj: &'a O,
    eps: f64,
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> FdHvp<'a, O> {
    /// Wrap `obj`; `eps` scales the probe (1e-6 suits doubles at unit
    /// curvature).
    pub fn new(obj: &'a O, eps: f64) -> Self {
        Self { obj, eps }
    }
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> eindir_core::Objective<f64> for FdHvp<'a, O> {
    fn dim(&self) -> usize {
        eindir_core::Objective::dim(self.obj)
    }
    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        self.obj.bounds()
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.obj.eval(x)
    }
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> eindir_core::Gradient<f64> for FdHvp<'a, O> {
    fn dim(&self) -> usize {
        eindir_core::Objective::dim(self.obj)
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.obj.value_and_gradient(x).1
    }
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> DifferentiableObjective<f64> for FdHvp<'a, O> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.obj.value_and_gradient(x)
    }
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> HessianVector for FdHvp<'a, O> {
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let vn = nrm2(v);
        if vn <= 0.0 {
            return Array1::zeros(v.len());
        }
        let h = self.eps / vn;
        let mut xp = x.to_owned();
        axpy(h, v, &mut xp);
        let mut xm = x.to_owned();
        axpy(-h, v, &mut xm);
        let gp = self.obj.value_and_gradient(xp.view()).1;
        let gm = self.obj.value_and_gradient(xm.view()).1;
        (gp - gm) / (2.0 * h)
    }
}

/// Approximate inverse applied to a residual: `z = P^{-1} r`. CG's
/// answers stay exact under any symmetric positive definite `P`; only
/// the iteration count changes, so a randomized preconditioner leaves
/// nothing stochastic in the result.
pub trait Preconditioner {
    /// `P^{-1} r`.
    fn solve(&self, r: ArrayView1<f64>) -> Array1<f64>;
}

/// No preconditioning: `z = r`.
pub struct IdentityPrecond;

impl Preconditioner for IdentityPrecond {
    fn solve(&self, r: ArrayView1<f64>) -> Array1<f64> {
        r.to_owned()
    }
}

/// Steihaug-Toint CG on the trust-region subproblem: approximately
/// minimize `g.p + p.Hp/2` subject to `||p|| <= radius`. Returns the
/// step and the model reduction `m(0) - m(p)`; every Hessian access is
/// one action.
pub fn steihaug_cg<O>(
    obj: &O,
    x: ArrayView1<f64>,
    grad: &Array1<f64>,
    radius: f64,
    rtol: f64,
    maxiter: usize,
) -> (Array1<f64>, f64)
where
    O: HessianVector + ?Sized,
{
    steihaug_pcg(obj, x, grad, radius, rtol, maxiter, &IdentityPrecond)
}

/// [`steihaug_cg`] under a preconditioner. The trust boundary lives in
/// the preconditioner's metric, tracked by the standard recurrences
/// (Conn, Gould, Toint, *Trust-Region Methods*, section 5.1), so no
/// extra applications of `P` are spent on norms.
pub fn steihaug_pcg<O, P>(
    obj: &O,
    x: ArrayView1<f64>,
    grad: &Array1<f64>,
    radius: f64,
    rtol: f64,
    maxiter: usize,
    precond: &P,
) -> (Array1<f64>, f64)
where
    O: HessianVector + ?Sized,
    P: Preconditioner + ?Sized,
{
    let n = grad.len();
    let mut p = Array1::<f64>::zeros(n);
    let mut hp = Array1::<f64>::zeros(n);
    let mut r = grad.clone();
    let mut z = precond.solve(r.view());
    let mut d = z.mapv(|v| -v);
    let mut rz = dot(r.view(), z.view());
    let gnorm = nrm2(grad.view());
    let stop = (rtol * gnorm).max(f64::MIN_POSITIVE);

    // M-metric bookkeeping: with M z = r, every needed inner product
    // reduces to Euclidean dots already on hand.
    let mut p_mp = 0.0_f64;
    let mut p_md = 0.0_f64;
    let mut d_md = rz;
    let r2 = radius * radius;

    let model_drop = |p: &Array1<f64>, hp: &Array1<f64>| {
        -(dot(grad.view(), p.view()) + 0.5 * dot(p.view(), hp.view()))
    };
    let boundary_tau = |p_mp: f64, p_md: f64, d_md: f64| -> f64 {
        if d_md <= 0.0 {
            return 0.0;
        }
        let disc = (p_md * p_md + d_md * (r2 - p_mp)).max(0.0);
        (-p_md + disc.sqrt()) / d_md
    };

    if !(rz.is_finite() && rz > 0.0) {
        // A broken or indefinite preconditioner forfeits its metric;
        // fall back to the steepest boundary step.
        let p = grad.mapv(|v| -radius * v / gnorm.max(f64::MIN_POSITIVE));
        let hp = obj.hessian_vector(x, p.view());
        let drop = model_drop(&p, &hp);
        return (p, drop);
    }

    for _ in 0..maxiter {
        let hd = obj.hessian_vector(x, d.view());
        let dhd = dot(d.view(), hd.view());
        if !dhd.is_finite() {
            // A broken action cannot price curvature; take the
            // steepest boundary step and let the ratio test judge it.
            let p = grad.mapv(|v| -radius * v / gnorm.max(f64::MIN_POSITIVE));
            let hp = obj.hessian_vector(x, p.view());
            let drop = model_drop(&p, &hp);
            return (p, drop);
        }
        if dhd <= 0.0 {
            // Negative curvature: follow d to the boundary.
            let tau = boundary_tau(p_mp, p_md, d_md);
            axpy(tau, d.view(), &mut p);
            axpy(tau, hd.view(), &mut hp);
            let drop = model_drop(&p, &hp);
            return (p, drop);
        }
        let alpha = rz / dhd;
        let p_mp_next = p_mp + 2.0 * alpha * p_md + alpha * alpha * d_md;
        if p_mp_next >= r2 {
            let tau = boundary_tau(p_mp, p_md, d_md);
            axpy(tau, d.view(), &mut p);
            axpy(tau, hd.view(), &mut hp);
            let drop = model_drop(&p, &hp);
            return (p, drop);
        }
        axpy(alpha, d.view(), &mut p);
        axpy(alpha, hd.view(), &mut hp);
        p_mp = p_mp_next;
        axpy(alpha, hd.view(), &mut r);
        if nrm2(r.view()) < stop {
            break;
        }
        z = precond.solve(r.view());
        let rz_next = dot(r.view(), z.view());
        if !(rz_next.is_finite() && rz_next > 0.0) {
            break;
        }
        let beta = rz_next / rz;
        p_md = beta * (p_md + alpha * d_md);
        d_md = rz_next + beta * beta * d_md;
        rz = rz_next;
        for (di, zi) in d.iter_mut().zip(z.iter()) {
            *di = -zi + beta * *di;
        }
    }
    let drop = model_drop(&p, &hp);
    (p, drop)
}

/// Randomized Nystrom preconditioner (Frangella, Tropp, Udell,
/// *Randomized Nystrom Preconditioning*, SIAM J. Matrix Anal. Appl.
/// 44 (2023), <https://doi.org/10.1137/21M1466244>).
///
/// `rank` Hessian actions on a Rademacher block sketch the dominant
/// eigenspace; the preconditioner equalizes those modes down to the
/// smallest captured eigenvalue and leaves the orthogonal complement
/// alone. Cluster and kernel Hessians carry a few stiff modes over a
/// soft bulk, which is exactly the spectrum this flattens. The
/// randomness lives only here: CG under any SPD preconditioner
/// returns the same step, in fewer actions when the sketch captures
/// the stiffness.
pub struct NystromPrecond {
    u: Array2<f64>,
    lam: Array1<f64>,
    mu: f64,
}

impl NystromPrecond {
    /// Sketch `H(x)` with `rank` actions. `seed` fixes the Rademacher
    /// draw so a rebuild at the same point is reproducible.
    pub fn build<O>(obj: &O, x: ArrayView1<f64>, rank: usize, seed: u64) -> Self
    where
        O: HessianVector + ?Sized,
    {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let n = x.len();
        let k = rank.clamp(1, n);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut omega = Array2::<f64>::zeros((n, k));
        for v in omega.iter_mut() {
            *v = if rng.random::<bool>() { 1.0 } else { -1.0 };
        }
        let mut y = Array2::<f64>::zeros((n, k));
        for j in 0..k {
            let col = omega.column(j).to_owned();
            let hcol = obj.hessian_vector(x, col.view());
            y.column_mut(j).assign(&hcol);
        }
        let ynorm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nu = 1e-12 * ynorm.max(1.0);
        let ynu = &y + &(nu * &omega);
        let mut m = omega.t().dot(&ynu);
        for i in 0..k {
            for j in (i + 1)..k {
                let s = 0.5 * (m[(i, j)] + m[(j, i)]);
                m[(i, j)] = s;
                m[(j, i)] = s;
            }
        }
        let (svals, v) = sym_eig_jacobi(m);
        // B = Ynu V diag(s^{-1/2}) over the safely positive s, so
        // A_nys = B B^T.
        let floor = nu.max(1e-14 * svals.iter().cloned().fold(0.0, f64::max));
        let kept: Vec<usize> = (0..k).filter(|&i| svals[i] > floor).collect();
        if kept.is_empty() {
            return Self {
                u: Array2::zeros((n, 0)),
                lam: Array1::zeros(0),
                mu: 1.0,
            };
        }
        let mut b = Array2::<f64>::zeros((n, kept.len()));
        for (bj, &i) in kept.iter().enumerate() {
            let scale = 1.0 / svals[i].sqrt();
            let col = ynu.dot(&v.column(i).to_owned()) * scale;
            b.column_mut(bj).assign(&col);
        }
        // Thin eigenfactorization of B B^T through the small Gram
        // matrix: B^T B = W S^2 W^T gives U = B W S^{-1}.
        let g = b.t().dot(&b);
        let (s2, w) = sym_eig_jacobi(g);
        let s2floor = 1e-14 * s2.iter().cloned().fold(0.0, f64::max).max(1.0);
        let idx: Vec<usize> = (0..s2.len()).filter(|&i| s2[i] > s2floor).collect();
        let mut u = Array2::<f64>::zeros((n, idx.len()));
        let mut lam = Array1::<f64>::zeros(idx.len());
        for (uj, &i) in idx.iter().enumerate() {
            let sig = s2[i].sqrt();
            let col = b.dot(&w.column(i).to_owned()) / sig;
            u.column_mut(uj).assign(&col);
            lam[uj] = (s2[i] - nu).max(0.0);
        }
        let mu = lam
            .iter()
            .cloned()
            .filter(|&l| l > 0.0)
            .fold(f64::INFINITY, f64::min);
        let mu = if mu.is_finite() { mu } else { 1.0 };
        Self { u, lam, mu }
    }

    /// Captured eigenvalue estimates, unordered.
    pub fn spectrum(&self) -> ArrayView1<'_, f64> {
        self.lam.view()
    }
}

impl Preconditioner for NystromPrecond {
    fn solve(&self, r: ArrayView1<f64>) -> Array1<f64> {
        if self.u.ncols() == 0 {
            return r.to_owned();
        }
        // P^{-1} r = r + U (diag(mu / (lam + mu)) - I) U^T r, scaled
        // so the unsketched complement passes through unchanged.
        let t = self.u.t().dot(&r);
        let adj = Array1::from_iter(
            t.iter()
                .zip(self.lam.iter())
                .map(|(ti, li)| ti * (self.mu / (li + self.mu) - 1.0)),
        );
        let mut z = r.to_owned();
        z += &self.u.dot(&adj);
        z
    }
}

/// Above this `|theta|` the Jacobi tangent is `1 / (2 theta)`.
///
/// Here `theta^2 + 1` rounds to `theta^2`, so `1 / (2 theta)` is the
/// value the full formula returns, and the branch changes no result
/// that was finite before. The Linear Algebra Package (LAPACK) starts
/// the same branch at `1 / sqrt(eps)` = `2^26`. Between `2^26` and
/// `2^27` the two can still differ by one or two units in the last
/// place.
const JACOBI_BIG_THETA: f64 = 1.0 / f64::EPSILON;

/// Tangent of the Jacobi rotation that zeroes `a_pq`, with
/// `theta = (a_qq - a_pp) / (2 a_pq)`.
///
/// The full formula `t = sign(theta) / (|theta| + sqrt(theta^2 + 1))`
/// is the 2-by-2 symmetric Schur decomposition of Golub and Van Loan,
/// *Matrix Computations*, 4th ed., algorithm 8.5.1,
/// <https://doi.org/10.56021/9781421407944>. When `|a_pq|` falls below
/// about 3.7e-155 times the gap `|a_qq - a_pp|`, `|theta|` passes
/// 1.3e154 and `theta^2` overflows. With overflow trapped, the host
/// stops there; with the trap masked, `t` is 0 and the solver skips
/// the rotation. Above
/// [`JACOBI_BIG_THETA`] the tangent is `1 / (2 theta)`, the limit of
/// the full formula, as in LAPACK's one-sided Jacobi routines `dgesvj`
/// and `dgsvj0` (Drmac and Veselic, *New Fast and Accurate Jacobi SVD
/// Algorithm. I*, <https://doi.org/10.1137/050639193>).
fn jacobi_tangent(theta: f64) -> f64 {
    if theta.abs() > JACOBI_BIG_THETA {
        return 0.5 / theta;
    }
    let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
    sign / (theta.abs() + (theta * theta + 1.0).sqrt())
}

/// Cyclic Jacobi eigendecomposition of a small symmetric matrix.
/// Returns eigenvalues and the column eigenvectors.
///
/// Each rotation is the 2-by-2 symmetric Schur decomposition of Golub
/// and Van Loan, *Matrix Computations*, 4th ed., algorithm 8.5.1, with
/// the tangent from `jacobi_tangent`.
pub(crate) fn sym_eig_jacobi(mut a: Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let k = a.nrows();
    let mut v = Array2::<f64>::eye(k);
    for _ in 0..64 {
        let mut off = 0.0;
        for p in 0..k {
            for q in (p + 1)..k {
                off += a[(p, q)] * a[(p, q)];
            }
        }
        let scale = a.diag().iter().map(|d| d.abs()).fold(1.0, f64::max);
        if off.sqrt() <= 1e-15 * scale {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = a[(p, q)];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let theta = (a[(q, q)] - a[(p, p)]) / (2.0 * apq);
                let t = jacobi_tangent(theta);
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for i in 0..k {
                    let aip = a[(i, p)];
                    let aiq = a[(i, q)];
                    a[(i, p)] = c * aip - s * aiq;
                    a[(i, q)] = s * aip + c * aiq;
                }
                for i in 0..k {
                    let api = a[(p, i)];
                    let aqi = a[(q, i)];
                    a[(p, i)] = c * api - s * aqi;
                    a[(q, i)] = s * api + c * aqi;
                }
                for i in 0..k {
                    let vip = v[(i, p)];
                    let viq = v[(i, q)];
                    v[(i, p)] = c * vip - s * viq;
                    v[(i, q)] = s * vip + c * viq;
                }
            }
        }
    }
    (a.diag().to_owned(), v)
}

const ETA_ACCEPT: f64 = 0.1;
const RADIUS_FLOOR: f64 = 1e-14;
/// Nyström sketch width for [`crate::Method::NewtonPcg`].
const NYSTROM_RANK: usize = 16;

/// No analytic action: the session builds a forward-difference product.
pub(crate) struct FiniteDifference;

/// Analytic `H v` when the caller has one. [`FiniteDifference`] has none.
pub(crate) trait OptionalAction {
    /// `Some(Hv)` from an analytic action, or `None` to finite-difference.
    fn apply(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Option<Array1<f64>>;
}

impl OptionalAction for FiniteDifference {
    fn apply(&self, _x: ArrayView1<f64>, _v: ArrayView1<f64>) -> Option<Array1<f64>> {
        None
    }
}

impl<F> OptionalAction for F
where
    F: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64>,
{
    fn apply(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Option<Array1<f64>> {
        Some(self(x, v))
    }
}

/// One Steihaug outer step. The radius ceiling is Euclidean.
pub(crate) struct NewtonCgRequest {
    /// Current trust radius.
    pub radius: f64,
    /// Euclidean ceiling. The helper also refuses to shrink below `radius`.
    pub rmax: f64,
    /// Per-atom clip on the trial, after the trust step is chosen.
    pub atom_cap: Option<f64>,
    /// Euclidean clip on the trial when no per-atom cap is set.
    pub euclid_cap: Option<f64>,
    /// Project the step out of rigid translation and rotation.
    pub project_rigid: bool,
    /// Nyström-preconditioned CG. The trust ceiling stays Euclidean.
    pub precondition: bool,
    /// Rademacher seed for the sketch. Rebuilt every outer step.
    pub seed: u64,
}

/// Point, radius, and the products this attempt spent.
pub(crate) struct NewtonCgTaken {
    /// Accepted point, or the incoming point when the trial is refused.
    pub pos: Array1<f64>,
    /// Value at [`Self::pos`].
    pub value: f64,
    /// Raw gradient at [`Self::pos`].
    pub grad: Array1<f64>,
    /// Updated trust radius.
    pub radius: f64,
    /// The trial replaced the current point.
    pub moved: bool,
    /// Trial `(f, g)` evaluations. Zero when the model step is refused
    /// before the objective is touched.
    pub trial_gradients: usize,
    /// Hessian-vector products, including a Nyström sketch and a
    /// reprice of a clipped step. A forward-difference product counts
    /// here, not as a gradient.
    pub hvps: usize,
    /// The radius hit [`RADIUS_FLOOR`].
    pub collapsed: bool,
}

/// Counts products and serves either an analytic action or one
/// forward difference of the known gradient.
struct SessionAction<'a, O: ?Sized, A> {
    obj: &'a O,
    /// Raw `g(x)` at the point CG is linearizing. The forward
    /// difference reuses it, so a product costs one gradient.
    fd_grad: &'a Array1<f64>,
    hvps: &'a AtomicUsize,
    action: &'a A,
}

impl<'a, O, A> eindir_core::Objective<f64> for SessionAction<'a, O, A>
where
    O: DifferentiableObjective<f64> + ?Sized,
    A: OptionalAction + Send + Sync,
{
    fn dim(&self) -> usize {
        eindir_core::Objective::dim(self.obj)
    }
    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        self.obj.bounds()
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.obj.eval(x)
    }
}

impl<'a, O, A> eindir_core::Gradient<f64> for SessionAction<'a, O, A>
where
    O: DifferentiableObjective<f64> + ?Sized,
    A: OptionalAction + Send + Sync,
{
    fn dim(&self) -> usize {
        eindir_core::Objective::dim(self.obj)
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.obj.value_and_gradient(x).1
    }
}

impl<'a, O, A> DifferentiableObjective<f64> for SessionAction<'a, O, A>
where
    O: DifferentiableObjective<f64> + ?Sized,
    A: OptionalAction + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.obj.value_and_gradient(x)
    }
}

impl<'a, O, A> HessianVector for SessionAction<'a, O, A>
where
    O: DifferentiableObjective<f64> + ?Sized,
    A: OptionalAction + Send + Sync,
{
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        if let Some(hv) = self.action.apply(x, v) {
            self.hvps.fetch_add(1, Ordering::Relaxed);
            return hv;
        }
        let vn = nrm2(v);
        if vn <= 0.0 {
            return Array1::zeros(v.len());
        }
        self.hvps.fetch_add(1, Ordering::Relaxed);
        forward_difference(self.obj, self.fd_grad, x, v)
    }
}

/// `Hv ≈ (g(x + h v) - g(x)) / h` with `h = 1e-8 / ||v||`.
/// One gradient. The known `g(x)` is not evaluated again.
fn forward_difference<O>(
    obj: &O,
    grad: &Array1<f64>,
    x: ArrayView1<f64>,
    v: ArrayView1<f64>,
) -> Array1<f64>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let vn = nrm2(v);
    let h = 1e-8 / vn;
    let mut xp = x.to_owned();
    axpy(h, v, &mut xp);
    let gp = obj.value_and_gradient(xp.view()).1;
    (gp - grad) / h
}

fn steps_differ(a: ArrayView1<f64>, b: ArrayView1<f64>) -> bool {
    a.len() != b.len()
        || a.iter().zip(b.iter()).any(|(u, v)| {
            let scale = 1.0 + u.abs().max(v.abs());
            (u - v).abs() > 1e-12 * scale
        })
}

/// One Steihaug-Toint attempt. Hessian access is only `H v`.
///
/// `maxmove` is the Euclidean ceiling on the radius. A per-atom cap
/// clips the trial after the direction is chosen; the boundary test
/// uses the pre-clip length, matching [`crate::Solver`]'s dogleg step.
/// Steihaug, <https://doi.org/10.1137/0720042>.
/// Nocedal and Wright, chapter 7 and algorithm 4.1,
/// <https://doi.org/10.1007/978-0-387-40065-5>.
pub(crate) fn newton_cg_step<O, A>(
    obj: &O,
    x: &Array1<f64>,
    value: f64,
    grad: &Array1<f64>,
    fd_grad: &Array1<f64>,
    req: &NewtonCgRequest,
    action: &A,
) -> NewtonCgTaken
where
    O: DifferentiableObjective<f64> + ?Sized,
    A: OptionalAction + Send + Sync,
{
    let hvps = AtomicUsize::new(0);
    let adapter = SessionAction {
        obj,
        fd_grad,
        hvps: &hvps,
        action,
    };
    let n = x.len();
    let gnorm = nrm2(grad.view());
    let rtol = gnorm.sqrt().min(0.5);
    let radius = req.radius;
    let rmax = req.rmax.max(radius);
    let (p_cg, mut pred) = if req.precondition {
        let pre = NystromPrecond::build(&adapter, x.view(), NYSTROM_RANK, req.seed);
        steihaug_pcg(&adapter, x.view(), grad, radius, rtol, 2 * n, &pre)
    } else {
        steihaug_cg(&adapter, x.view(), grad, radius, rtol, 2 * n)
    };
    // PCG stops on the preconditioner sphere. Pull a long step back
    // onto the Euclidean trust sphere before the displacement cap.
    let mut p = p_cg.clone();
    if req.project_rigid {
        project_out_rot_trans(&mut p, x.view());
    }
    let pnorm = nrm2(p.view());
    if radius.is_finite() && pnorm > radius * (1.0 + 1e-10) && pnorm > 0.0 {
        let scale = radius / pnorm;
        p.mapv_inplace(|v| v * scale);
    }
    let trust_norm = nrm2(p.view());
    let mut trial = x.clone();
    axpy(1.0, p.view(), &mut trial);
    if let Some(cap) = req.atom_cap {
        scale_step_atom(x, &mut trial, cap);
    } else if let Some(cap) = req.euclid_cap {
        scale_step(x, &mut trial, cap);
    }
    trial = obj.bounds().clip(trial.view());
    let step = &trial - x;
    if steps_differ(step.view(), p_cg.view()) {
        let hp = adapter.hessian_vector(x.view(), step.view());
        pred = -(dot(grad.view(), step.view()) + 0.5 * dot(step.view(), hp.view()));
    }
    let taken = nrm2(step.view());
    let boundary_norm = if req.atom_cap.is_some() || req.euclid_cap.is_some() {
        trust_norm
    } else {
        taken
    };
    let refuse = || NewtonCgTaken {
        pos: x.clone(),
        value,
        grad: grad.clone(),
        radius: update_radius(radius, -1.0, 0.0, rmax),
        moved: false,
        trial_gradients: 0,
        hvps: hvps.load(Ordering::Relaxed),
        collapsed: false,
    };
    if trust_norm <= 0.0 || !(pred.is_finite() && pred > 0.0) {
        let mut out = refuse();
        out.collapsed = out.radius < RADIUS_FLOOR;
        return out;
    }
    let (ft, gt) = obj.value_and_gradient(trial.view());
    let rho = if ft.is_finite() {
        reduction_ratio(value - ft, pred)
    } else {
        -1.0
    };
    let radius = update_radius(radius, rho, boundary_norm, rmax);
    let moved = rho > ETA_ACCEPT && ft.is_finite();
    let collapsed = radius < RADIUS_FLOOR;
    if moved {
        NewtonCgTaken {
            pos: trial,
            value: ft,
            grad: gt,
            radius,
            moved: true,
            trial_gradients: 1,
            hvps: hvps.load(Ordering::Relaxed),
            collapsed,
        }
    } else {
        NewtonCgTaken {
            pos: x.clone(),
            value,
            grad: grad.clone(),
            radius,
            moved: false,
            trial_gradients: 1,
            hvps: hvps.load(Ordering::Relaxed),
            collapsed,
        }
    }
}

/// Matrix-free Newton: Steihaug-Toint CG inside a Nocedal-Wright trust
/// region. `control.maxmove` caps the trust radius, so the region
/// plays the role the step clip plays elsewhere; `control.istep` seeds
/// the initial radius when positive.
pub fn minimize_newton_cg<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
) -> Result<Report>
where
    O: HessianVector + ?Sized,
{
    let mut pos = init.into();
    if pos.len() != eindir_core::Objective::dim(obj) {
        return Err(Error::Dim {
            got: pos.len(),
            dim: eindir_core::Objective::dim(obj),
        });
    }
    pos = obj.bounds().clip(pos.view());
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());

    let rmax = control.maxmove.unwrap_or(f64::INFINITY).max(RADIUS_FLOOR);
    let mut radius = if control.istep > 0.0 {
        control.istep.min(rmax)
    } else {
        1.0_f64.min(rmax)
    };
    let action = |xv: ArrayView1<f64>, v: ArrayView1<f64>| obj.hessian_vector(xv, v);

    for step in 0..control.maxiter {
        let gnorm = nrm2(grad.view());
        if !gnorm.is_finite() {
            return Err(Error::TrustCollapsed { steps: step });
        }
        if gnorm < control.gtol {
            return Ok(Report {
                value,
                coords: pos,
                steps: step,
                grad_norm: gnorm,
                grad: None,
            });
        }
        // Inexact-Newton forcing keeps early CG cheap and the tail
        // superlinear (Nocedal-Wright eq 7.3). The shared step also
        // applies `maxmove` as a Euclidean clip; with the radius
        // already inside that ceiling the clip does not move `p`.
        let taken = newton_cg_step(
            obj,
            &pos,
            value,
            &grad,
            &grad,
            &NewtonCgRequest {
                radius,
                rmax,
                atom_cap: None,
                euclid_cap: control.maxmove,
                project_rigid: false,
                precondition: false,
                seed: step as u64,
            },
            &action,
        );
        radius = taken.radius;
        if taken.collapsed {
            return Err(Error::TrustCollapsed { steps: step });
        }
        if taken.moved {
            pos = taken.pos;
            value = taken.value;
            grad = taken.grad;
        }
    }
    Ok(Report {
        value,
        coords: pos,
        steps: control.maxiter,
        grad_norm: nrm2(grad.view()),
        grad: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn full_tangent(theta: f64) -> f64 {
        let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
        sign / (theta.abs() + (theta * theta + 1.0).sqrt())
    }

    #[test]
    fn jacobi_tangent_matches_the_full_formula_above_the_threshold() {
        let mut rng = StdRng::seed_from_u64(11);
        let (lo, hi) = (JACOBI_BIG_THETA.log2(), 1.3e154_f64.log2());
        for _ in 0..100_000 {
            let mut theta = 2f64.powf(rng.random_range(lo..hi)).max(JACOBI_BIG_THETA);
            if rng.random::<bool>() {
                theta = -theta;
            }
            assert_eq!(
                jacobi_tangent(theta).to_bits(),
                full_tangent(theta).to_bits(),
                "theta = {theta:e}"
            );
        }
    }

    #[test]
    fn jacobi_rotates_a_1e_160_coupling_next_to_a_unit_gap() {
        // theta for the (0, 1) pair is (2 - 1) / 2e-160 = 5e159.
        let a = array![[1.0, 1e-160, 0.0], [1e-160, 2.0, 0.5], [0.0, 0.5, 3.0]];
        let (evals, evecs) = sym_eig_jacobi(a);
        let r = 0.5_f64.sqrt();
        for (got, want) in evals.iter().zip([1.0, 2.5 - r, 2.5 + r]) {
            assert!((got - want).abs() <= 1e-15, "eigenvalues {evals:?}");
        }
        // The first rotation tilts the unit eigenvector by -1e-160,
        // against a first-order tilt of -8/7 * 1e-160.
        let tilt = evecs[(1, 0)] / evecs[(0, 0)];
        assert!(tilt < -5e-161 && tilt > -2e-160, "tilt = {tilt:e}");
    }
}
