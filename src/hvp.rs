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

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, Array2, ArrayView1};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::report::Report;
use crate::scg::DirectionalCurvature;
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

/// Cyclic Jacobi eigendecomposition of a small symmetric matrix.
/// Returns eigenvalues and the column eigenvectors.
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
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
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
    let n = pos.len();

    let rmax = control.maxmove.unwrap_or(f64::INFINITY).max(RADIUS_FLOOR);
    let mut radius = if control.istep > 0.0 {
        control.istep.min(rmax)
    } else {
        1.0_f64.min(rmax)
    };

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
            });
        }
        // Inexact-Newton forcing keeps early CG cheap and the tail
        // superlinear (Nocedal-Wright eq 7.3).
        let rtol = gnorm.sqrt().min(0.5);
        let (p, pred) = steihaug_cg(obj, pos.view(), &grad, radius, rtol, 2 * n);
        let pnorm = nrm2(p.view());
        if pnorm <= 0.0 || pred <= 0.0 {
            radius *= 0.25;
            if radius < RADIUS_FLOOR {
                return Err(Error::TrustCollapsed { steps: step });
            }
            continue;
        }
        let mut trial = pos.clone();
        axpy(1.0, p.view(), &mut trial);
        let trial = obj.bounds().clip(trial.view());
        let (ft, gt) = obj.value_and_gradient(trial.view());
        let rho = if ft.is_finite() {
            reduction_ratio(value - ft, pred)
        } else {
            -1.0
        };
        radius = update_radius(radius, rho, pnorm, rmax);
        if radius < RADIUS_FLOOR {
            return Err(Error::TrustCollapsed { steps: step });
        }
        if rho > ETA_ACCEPT {
            pos = trial;
            value = ft;
            grad = gt;
        }
    }
    Ok(Report {
        value,
        coords: pos,
        steps: control.maxiter,
        grad_norm: nrm2(grad.view()),
    })
}
