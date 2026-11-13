//! Matrix-free lowest eigenpair, and the block of k lowest pairs, of a
//! symmetric Hessian action.
//!
//! IRC kick and the `lambda_min` sign check need one extremal pair,
//! not a full ELPA / SLATE spectrum. Dispatch is a closed
//! [`EigensolverKind`]: Lanczos, Rayleigh-Ritz, Jacobi-Davidson,
//! LOBPCG, and the Jónsson dimer with Heyden plane rotations run
//! here; every other named backend fail-closes with
//! [`Error::EigenUnavailable`]. Integers match `schema/eigen.capnp`.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::error::{Error, Result};
use crate::hvp::HessianVector;
use crate::vecops::{axpy, dot, nrm2};

/// Cutoff used by gpr_optim `kMinDistributedSymmetricEigenOrder`.
/// Dense distributed backends (ELPA, SLATE) sit at or above this.
pub const DENSE_EIGEN_CUTOFF: usize = 512;

/// Closed eigensolver tag. Ordinals match `schema/eigen.capnp`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EigensolverKind {
    /// Lanczos + tiny Jacobi on the tridiagonal. Default IRC kick.
    Lanczos = 0,
    /// Residual-expanded Rayleigh-Ritz (gpr_optim `lowestEigenpairRayleighRitz`).
    RayleighRitz = 1,
    /// Jacobi-Davidson correction, matrix-free inner CG
    /// (Davidson 1975, Sleijpen-Van der Vorst 1996).
    JacobiDavidson = 2,
    /// LOBPCG, nev = 1 (Knyazev 2001).
    Lobpcg = 3,
    /// PRIMME. Not linked.
    Primme = 4,
    /// SLEPc EPS. Not linked.
    Slepc = 5,
    /// ChASE Chebyshev filter. Not linked.
    Chase = 6,
    /// ELPA dense distributed. Not linked.
    Elpa = 7,
    /// ELPA2 GPU. Not linked.
    Elpa2 = 8,
    /// SLATE heev. Not linked.
    Slate = 9,
    /// MAGMA syev. Not linked.
    Magma = 10,
    /// cuSOLVER dense / batched. Not linked.
    Cusolver = 11,
    /// DLA-Future. Not linked.
    DlaFuture = 12,
    /// EigenExa. Not linked.
    EigenExa = 13,
    /// Jónsson dimer with Heyden plane rotations. Linked. Matrix-free.
    Dimer = 14,
}

impl EigensolverKind {
    /// Schema / C ABI name. Never a free-form string key.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Lanczos => "lanczos",
            Self::RayleighRitz => "rayleighRitz",
            Self::JacobiDavidson => "jacobiDavidson",
            Self::Lobpcg => "lobpcg",
            Self::Primme => "primme",
            Self::Slepc => "slepc",
            Self::Chase => "chase",
            Self::Elpa => "elpa",
            Self::Elpa2 => "elpa2",
            Self::Slate => "slate",
            Self::Magma => "magma",
            Self::Cusolver => "cusolver",
            Self::DlaFuture => "dlaFuture",
            Self::EigenExa => "eigenExa",
            Self::Dimer => "dimer",
        }
    }

    /// Built into this crate. Unlinked kinds return [`Error::EigenUnavailable`].
    pub const fn is_linked(self) -> bool {
        matches!(
            self,
            Self::Lanczos | Self::RayleighRitz | Self::JacobiDavidson | Self::Lobpcg | Self::Dimer
        )
    }

    /// Works from Hessian actions, no assembled matrix.
    pub const fn is_matrix_free(self) -> bool {
        matches!(
            self,
            Self::Lanczos
                | Self::RayleighRitz
                | Self::JacobiDavidson
                | Self::Lobpcg
                | Self::Primme
                | Self::Slepc
                | Self::Dimer
        )
    }

    /// Decode a schema / C ordinal. Unknown integers are `None`.
    pub const fn from_ordinal(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Lanczos),
            1 => Some(Self::RayleighRitz),
            2 => Some(Self::JacobiDavidson),
            3 => Some(Self::Lobpcg),
            4 => Some(Self::Primme),
            5 => Some(Self::Slepc),
            6 => Some(Self::Chase),
            7 => Some(Self::Elpa),
            8 => Some(Self::Elpa2),
            9 => Some(Self::Slate),
            10 => Some(Self::Magma),
            11 => Some(Self::Cusolver),
            12 => Some(Self::DlaFuture),
            13 => Some(Self::EigenExa),
            14 => Some(Self::Dimer),
            _ => None,
        }
    }
}

/// Typed parameters for [`lowest_mode`]. No string fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EigenParams {
    /// Which backend.
    pub kind: EigensolverKind,
    /// Extremal pairs requested. The IRC kick uses 1.
    pub nev: usize,
    /// Krylov / subspace cap. 0 selects `min(n, 12)`.
    pub krylov: usize,
    /// Outer iterations. 0 selects `n`.
    pub max_iter: usize,
    /// Residual tolerance. Non-positive selects `1e-8`.
    pub tol: f64,
}

impl Default for EigenParams {
    fn default() -> Self {
        Self {
            kind: EigensolverKind::Lanczos,
            nev: 1,
            krylov: 0,
            max_iter: 0,
            tol: 0.0,
        }
    }
}

impl EigenParams {
    fn krylov_dim(self, n: usize) -> usize {
        let k = if self.krylov == 0 {
            12.min(n)
        } else {
            self.krylov
        };
        k.clamp(1, n.max(1))
    }

    fn tolerance(self) -> f64 {
        if self.tol > 0.0 { self.tol } else { 1e-8 }
    }

    fn iterations(self, n: usize) -> usize {
        if self.max_iter == 0 {
            n.max(8)
        } else {
            self.max_iter
        }
    }
}

/// Hessian action used by the lowest-mode waist.
///
/// [`HessianVector`] implements this. Closures `Fn(x, v) -> H v` do too,
/// so the C ABI does not have to fake a full objective.
pub trait ApplyHessian {
    /// `H(x) v` without forming `H`.
    fn apply_hessian(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64>;
}

impl<T: HessianVector + ?Sized> ApplyHessian for T {
    fn apply_hessian(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        self.hessian_vector(x, v)
    }
}

/// Result of [`lowest_eigenpair`] / [`lowest_mode`].
#[derive(Clone, Debug)]
pub struct LowestMode {
    /// Approximate eigenvector, Euclidean-normalized.
    pub vector: Array1<f64>,
    /// Approximate eigenvalue (Rayleigh quotient).
    pub value: f64,
    /// Number of Hessian actions.
    pub actions: usize,
}

/// Lanczos for the lowest eigenpair of `H(x)`.
///
/// Convenience wrapper around [`lowest_mode`] with
/// [`EigensolverKind::Lanczos`]. An empty seed is [`Error::Dim`],
/// the same answer [`lowest_mode`] gives. It does not panic.
pub fn lowest_eigenpair<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    krylov: usize,
) -> Result<LowestMode> {
    let params = EigenParams {
        kind: EigensolverKind::Lanczos,
        krylov,
        ..EigenParams::default()
    };
    lowest_mode(h, x, seed, &params)
}

/// Dispatch on [`EigenParams::kind`]. Unlinked backends return
/// [`Error::EigenUnavailable`].
pub fn lowest_mode<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
) -> Result<LowestMode> {
    if seed.is_empty() {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    match params.kind {
        EigensolverKind::Lanczos => Ok(lanczos(h, x, seed, params.krylov_dim(seed.len()))),
        EigensolverKind::RayleighRitz => Ok(rayleigh_ritz(h, x, seed, params)),
        EigensolverKind::JacobiDavidson => Ok(jacobi_davidson(h, x, seed, params)),
        EigensolverKind::Lobpcg => Ok(lobpcg(h, x, seed, params)),
        EigensolverKind::Dimer => Ok(dimer(h, x, seed, params)),
        other => Err(Error::EigenUnavailable { kind: other.name() }),
    }
}

fn lanczos<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    krylov: usize,
) -> LowestMode {
    let n = seed.len();
    let m = krylov.min(n).max(1);
    let mut q: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut alpha = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m);
    q.push(normalize(seed.to_owned()));

    let mut actions = 0;
    for j in 0..m {
        let hv = h.apply_hessian(x, q[j].view());
        actions += 1;
        let a = dot(hv.view(), q[j].view());
        alpha.push(a);
        if j + 1 == m {
            break;
        }
        let mut w = hv;
        axpy(-a, q[j].view(), &mut w);
        if j > 0 {
            axpy(-beta[j - 1], q[j - 1].view(), &mut w);
        }
        for qi in q.iter() {
            let overlap = dot(w.view(), qi.view());
            axpy(-overlap, qi.view(), &mut w);
        }
        let b = nrm2(w.view());
        if b <= 1e-12 {
            break;
        }
        beta.push(b);
        q.push(w / b);
    }

    let k = alpha.len();
    let mut t = vec![vec![0.0; k]; k];
    for i in 0..k {
        t[i][i] = alpha[i];
        if i + 1 < k && i < beta.len() {
            t[i][i + 1] = beta[i];
            t[i + 1][i] = beta[i];
        }
    }
    let (evals, evecs) = jacobi_eigen(&mut t);
    let lowest = argmin(&evals);
    let mut mode = Array1::zeros(n);
    for (i, qi) in q.iter().enumerate().take(k) {
        axpy(evecs[i][lowest], qi.view(), &mut mode);
    }
    LowestMode {
        vector: normalize(mode),
        value: evals[lowest],
        actions,
    }
}

/// Jónsson dimer with Heyden plane rotations.
///
/// The dimer axis is the current mode. One Hessian action gives the
/// curvature `C = N·HN` and the rotational force `F_rot = HN − C N`.
/// A Rayleigh-Ritz solve selects the lower-curvature axis in the plane
/// `{N, Θ}`, with `Θ = F_rot/‖F_rot‖`. The Hessian is fixed at `x`, so
/// its action rotates with the axis and each plane needs one new action.
fn dimer<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
) -> LowestMode {
    let mut nvec = normalize(seed.to_owned());
    let max_rot = if params.max_iter == 0 {
        20
    } else {
        params.max_iter
    };
    let tol = params.tolerance();
    let mut hn = h.apply_hessian(x, nvec.view());
    let mut actions = 1;
    let mut curvature = 0.0;
    for _ in 0..max_rot {
        curvature = dot(hn.view(), nvec.view());
        let mut frot = hn.clone();
        axpy(-curvature, nvec.view(), &mut frot);
        let frn = nrm2(frot.view());
        if frn <= tol {
            break;
        }
        let theta = &frot / frn;
        let ht = h.apply_hessian(x, theta.view());
        actions += 1;
        let b = dot(nvec.view(), ht.view());
        let c_th = dot(theta.view(), ht.view());
        let phi = 0.5 * (2.0 * b).atan2(curvature - c_th);
        let (co, si) = (phi.cos(), phi.sin());
        let c1 = co * co * curvature + 2.0 * co * si * b + si * si * c_th;
        let c2 = si * si * curvature - 2.0 * co * si * b + co * co * c_th;
        let (cuse, suse, cval) = if c1 <= c2 {
            (co, si, c1)
        } else {
            (-si, co, c2)
        };
        let mut next = nvec.mapv(|v| v * cuse);
        axpy(suse, theta.view(), &mut next);
        let mut hnext = hn.mapv(|v| v * cuse);
        axpy(suse, ht.view(), &mut hnext);
        hn = hnext / nrm2(next.view());
        nvec = normalize(next);
        curvature = cval;
    }
    LowestMode {
        vector: nvec,
        value: curvature,
        actions,
    }
}

/// Lowest Ritz value with its unit vector, residual and Hessian action.
type RitzPair = (f64, Array1<f64>, Array1<f64>, Array1<f64>);

struct Subspace {
    q: Vec<Array1<f64>>,
    aq: Vec<Array1<f64>>,
    actions: usize,
}

impl Subspace {
    fn with_capacity(m: usize) -> Self {
        Self {
            q: Vec::with_capacity(m),
            aq: Vec::with_capacity(m),
            actions: 0,
        }
    }

    fn try_append<H: ApplyHessian + ?Sized>(
        &mut self,
        h: &H,
        x: ArrayView1<f64>,
        mut v: Array1<f64>,
    ) -> bool {
        for qi in &self.q {
            let overlap = dot(v.view(), qi.view());
            axpy(-overlap, qi.view(), &mut v);
        }
        let b = nrm2(v.view());
        if b <= 1e-12 {
            return false;
        }
        v.mapv_inplace(|c| c / b);
        let av = h.apply_hessian(x, v.view());
        self.actions += 1;
        self.q.push(v);
        self.aq.push(av);
        true
    }

    fn ritz(&self, n: usize) -> Option<RitzPair> {
        let k = self.q.len();
        if k == 0 {
            return None;
        }
        let mut t = vec![vec![0.0; k]; k];
        for (i, qi) in self.q.iter().enumerate() {
            for (j, aqj) in self.aq.iter().enumerate().take(i + 1) {
                let value = dot(qi.view(), aqj.view());
                t[i][j] = value;
                t[j][i] = value;
            }
        }
        let (evals, evecs) = jacobi_eigen(&mut t);
        let lowest = argmin(&evals);
        let mut mode = Array1::zeros(n);
        let mut amode = Array1::zeros(n);
        for ((weights, qi), aqi) in evecs.iter().zip(&self.q).zip(&self.aq) {
            axpy(weights[lowest], qi.view(), &mut mode);
            axpy(weights[lowest], aqi.view(), &mut amode);
        }
        let nrm = nrm2(mode.view());
        if nrm <= 1e-14 {
            return None;
        }
        mode.mapv_inplace(|c| c / nrm);
        amode.mapv_inplace(|c| c / nrm);
        let theta = evals[lowest];
        let mut residual = amode.clone();
        axpy(-theta, mode.view(), &mut residual);
        for qi in &self.q {
            let overlap = dot(residual.view(), qi.view());
            axpy(-overlap, qi.view(), &mut residual);
        }
        Some((theta, mode, residual, amode))
    }
}

fn rayleigh_ritz<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
) -> LowestMode {
    let n = seed.len();
    let m = params.krylov_dim(n);
    let tol = params.tolerance();
    let mut space = Subspace::with_capacity(m);
    if !space.try_append(h, x, seed.to_owned()) {
        let mut unit = Array1::zeros(n);
        unit[0] = 1.0;
        space.try_append(h, x, unit);
    }
    let mut last = LowestMode {
        vector: normalize(seed.to_owned()),
        value: 0.0,
        actions: space.actions,
    };
    while space.q.len() <= m {
        let Some((theta, mode, residual, _)) = space.ritz(n) else {
            break;
        };
        last = LowestMode {
            vector: mode,
            value: theta,
            actions: space.actions,
        };
        let rnorm = nrm2(residual.view());
        if rnorm <= tol * (1.0 + theta.abs()) {
            break;
        }
        if space.q.len() >= m {
            break;
        }
        if !space.try_append(h, x, residual) {
            break;
        }
    }
    last
}

fn project_against(u: ArrayView1<f64>, v: &mut Array1<f64>) {
    let overlap = dot(v.view(), u);
    axpy(-overlap, u, v);
}

fn apply_jd<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    u: ArrayView1<f64>,
    theta: f64,
    p: ArrayView1<f64>,
) -> Array1<f64> {
    let mut w = p.to_owned();
    project_against(u, &mut w);
    let mut hw = h.apply_hessian(x, w.view());
    axpy(-theta, w.view(), &mut hw);
    project_against(u, &mut hw);
    hw
}

/// Matrix-free Jacobi-Davidson correction: CG on
/// `(I-uu^T)(H-θI)(I-uu^T) t = -r`, `t ⊥ u`.
fn jd_correction<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    u: ArrayView1<f64>,
    theta: f64,
    residual: ArrayView1<f64>,
    max_inner: usize,
    actions: &mut usize,
) -> Option<Array1<f64>> {
    let n = residual.len();
    let mut b = residual.to_owned();
    b.mapv_inplace(|c| -c);
    project_against(u, &mut b);
    let mut sol = Array1::zeros(n);
    let mut r = b;
    let mut p = r.clone();
    let mut rsold = dot(r.view(), r.view());
    if rsold <= 1e-30 {
        return None;
    }
    for _ in 0..max_inner {
        let ap = apply_jd(h, x, u, theta, p.view());
        *actions += 1;
        let pap = dot(p.view(), ap.view());
        if pap.abs() <= 1e-30 {
            break;
        }
        let alpha = rsold / pap;
        axpy(alpha, p.view(), &mut sol);
        axpy(-alpha, ap.view(), &mut r);
        let rsnew = dot(r.view(), r.view());
        if rsnew.sqrt() <= 1e-10 {
            break;
        }
        let beta = rsnew / rsold;
        let mut p_new = r.clone();
        axpy(beta, p.view(), &mut p_new);
        p = p_new;
        rsold = rsnew;
    }
    project_against(u, &mut sol);
    if nrm2(sol.view()) <= 1e-14 {
        None
    } else {
        Some(sol)
    }
}

fn jacobi_davidson<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
) -> LowestMode {
    let n = seed.len();
    let m = params.krylov_dim(n);
    let tol = params.tolerance();
    let inner = (n / 4).clamp(4, 16);
    let mut space = Subspace::with_capacity(m);
    if !space.try_append(h, x, seed.to_owned()) {
        let mut unit = Array1::zeros(n);
        unit[0] = 1.0;
        space.try_append(h, x, unit);
    }
    let mut last = LowestMode {
        vector: normalize(seed.to_owned()),
        value: 0.0,
        actions: space.actions,
    };
    let mut extra_actions = 0;
    while space.q.len() <= m {
        let Some((theta, mode, residual, _)) = space.ritz(n) else {
            break;
        };
        last = LowestMode {
            vector: mode.clone(),
            value: theta,
            actions: space.actions + extra_actions,
        };
        let rnorm = nrm2(residual.view());
        if rnorm <= tol * (1.0 + theta.abs()) {
            break;
        }
        if space.q.len() >= m {
            break;
        }
        let next = jd_correction(
            h,
            x,
            mode.view(),
            theta,
            residual.view(),
            inner,
            &mut extra_actions,
        )
        .unwrap_or(residual);
        if !space.try_append(h, x, next) {
            break;
        }
    }
    last.actions = space.actions + extra_actions;
    last
}

fn lobpcg<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
) -> LowestMode {
    let n = seed.len();
    let tol = params.tolerance();
    let max_iter = params.iterations(n);
    let mut vec = normalize(seed.to_owned());
    let mut avec = h.apply_hessian(x, vec.view());
    let mut actions = 1;
    let mut p: Option<Array1<f64>> = None;
    let mut theta = dot(vec.view(), avec.view());
    for _ in 0..max_iter {
        let mut residual = avec.clone();
        axpy(-theta, vec.view(), &mut residual);
        if nrm2(residual.view()) <= tol * (1.0 + theta.abs()) {
            break;
        }
        let mut space = Subspace::with_capacity(3);
        space.q.push(vec.clone());
        space.aq.push(avec.clone());
        if !space.try_append(h, x, residual) {
            break;
        }
        if let Some(dir) = p.take() {
            space.try_append(h, x, dir);
        }
        actions += space.actions;
        let Some((new_theta, new_vec, _, new_avec)) = space.ritz(n) else {
            break;
        };
        let mut dir = new_vec.clone();
        axpy(-1.0, vec.view(), &mut dir);
        vec = new_vec;
        avec = new_avec;
        theta = new_theta;
        if nrm2(dir.view()) > 1e-14 {
            p = Some(dir);
        }
    }
    LowestMode {
        vector: vec,
        value: theta,
        actions,
    }
}

/// One Ritz pair from [`lowest_eigenpairs`].
///
/// Vectors from one solve are orthonormal. `converged` is the final
/// residual test `||Hv − θv|| ≤ tol (1 + |θ|)`.
#[derive(Clone, Debug)]
pub struct Eigenpair {
    /// Euclidean-normalized Ritz vector.
    pub vector: Array1<f64>,
    /// Ritz value, the Rayleigh quotient.
    pub value: f64,
    /// `||Hv − θv||₂`.
    pub residual: f64,
    /// Whether `residual` met the tolerance passed to [`lowest_eigenpairs`].
    pub converged: bool,
}

/// The k lowest pairs of a symmetric product, lowest value first.
#[derive(Clone, Debug)]
pub struct Eigenpairs {
    /// Length `k`, values ascending.
    pub pairs: Vec<Eigenpair>,
    /// Hessian-vector products used by the solve.
    pub actions: usize,
    /// Block expansions performed. The seed Rayleigh–Ritz is not one.
    pub iterations: usize,
}

const BLOCK_DROP: f64 = 1e-12;

fn eigen_input(what: &'static str) -> Error {
    Error::EigenInput { what }
}

/// k lowest eigenpairs of a symmetric product, matrix-free.
///
/// Block LOBPCG with an identity preconditioner (Knyazev, SIAM J. Sci.
/// Comput. 23 (2001), 517–541). `seed` has shape `(n, k)`, one column
/// per starting vector, and is the warm start: the previous block is a
/// valid seed. Each iteration orthonormalizes the residual block against
/// the current Ritz block and the previous conjugate direction, then
/// takes the lowest k Ritz pairs of that basis. A column that is
/// numerically in the span already is dropped.
///
/// `tol` is the residual tolerance `||Hv − θv|| ≤ tol (1 + |θ|)`.
/// `max_iter` caps the number of block expansions. A solve that stops
/// on the cap still returns its best pairs; `Eigenpair::converged`
/// says which residuals met `tol`.
///
/// # Errors
///
/// [`Error::EigenBlock`] when `k = 0` or `k > n`. [`Error::EigenInput`]
/// when `x` or `seed` is non-finite, `tol` is non-positive or
/// non-finite, `max_iter` is zero, or a product is non-finite or the
/// wrong length. [`Error::Dim`] when `x.len()` is not `n`. The solve
/// does not panic on these inputs.
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, Array2, ArrayView1};
/// use rgmin::{ApplyHessian, lowest_eigenpairs};
///
/// struct Diag;
/// impl ApplyHessian for Diag {
///     fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
///         let lam = [1.0, -2.0, 4.0];
///         Array1::from_iter(lam.iter().zip(v.iter()).map(|(l, vi)| l * vi))
///     }
/// }
///
/// let x = Array1::zeros(3);
/// let mut seed = Array2::zeros((3, 2));
/// seed[(0, 0)] = 1.0;
/// seed[(1, 0)] = 0.2;
/// seed[(1, 1)] = 0.2;
/// seed[(2, 1)] = 1.0;
/// let found = lowest_eigenpairs(&Diag, x.view(), seed.view(), 1e-8, 20).unwrap();
/// assert!((found.pairs[0].value + 2.0).abs() < 1e-6);
/// assert!((found.pairs[1].value - 1.0).abs() < 1e-6);
/// assert!(found.pairs.iter().all(|pair| pair.converged));
/// ```
pub fn lowest_eigenpairs<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView2<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<Eigenpairs> {
    let (n, k) = seed.dim();
    if k == 0 || k > n {
        return Err(Error::EigenBlock { k, dim: n });
    }
    if x.len() != n {
        return Err(Error::Dim {
            got: x.len(),
            dim: n,
        });
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(eigen_input("tolerance must be positive and finite"));
    }
    if max_iter == 0 {
        return Err(eigen_input("iteration cap must be positive"));
    }
    if x.iter().any(|v| !v.is_finite()) || seed.iter().any(|v| !v.is_finite()) {
        return Err(eigen_input("non-finite value"));
    }
    let cols: Vec<Array1<f64>> = (0..k).map(|j| seed.column(j).to_owned()).collect();
    let mut block = orthonormal_block(cols, n, k)?;
    let mut search: Vec<Array1<f64>> = Vec::new();
    let mut search_h: Vec<Array1<f64>> = Vec::new();
    let mut actions = 0usize;
    let mut iterations = 0usize;
    let (_ax, evals, residuals) = loop {
        let mut ax = apply_columns(h, x, &block, &mut actions)?;
        let evals = align_block(&mut block, &mut ax)?;
        let residuals = column_residuals(&ax, &block, &evals)?;
        let all_converged = residuals.iter().zip(&evals).all(|(r, theta)| {
            let norm = nrm2(r.view());
            norm.is_finite() && norm <= tol * (1.0 + theta.abs())
        });
        if all_converged || iterations == max_iter {
            break (ax, evals, residuals);
        }
        iterations += 1;
        let mut width = Vec::new();
        absorb(&block, &mut width, residuals);
        let width_h = apply_columns(h, x, &width, &mut actions)?;
        let mut locked = block.clone();
        locked.extend(width.iter().cloned());
        let mut locked_h = ax.clone();
        locked_h.extend(width_h.iter().cloned());
        let (search_use, search_h_use) = deflate_directions(&locked, &locked_h, search, search_h)?;
        if width.is_empty() && search_use.is_empty() {
            let residuals = column_residuals(&ax, &block, &evals)?;
            break (ax, evals, residuals);
        }
        let mut basis = block.clone();
        basis.extend(width);
        basis.extend(search_use);
        let mut images = ax.clone();
        images.extend(width_h);
        images.extend(search_h_use);
        let (ritz, vecs) = ritz_sorted(&basis, &images)?;
        if ritz.len() < k {
            return Err(eigen_input("Ritz block is short"));
        }
        block = mix(&basis, &vecs, k)?;
        let mut mixed_h = mix(&images, &vecs, k)?;
        rescale_pair(&mut block, &mut mixed_h)?;
        let tail: Vec<Array1<f64>> = basis.into_iter().skip(k).collect();
        let tail_h: Vec<Array1<f64>> = images.into_iter().skip(k).collect();
        let raw_p = mix_rows(&tail, &vecs, k, k)?;
        let raw_ap = mix_rows(&tail_h, &vecs, k, k)?;
        search = Vec::new();
        search_h = Vec::new();
        for (dir, image) in raw_p.into_iter().zip(raw_ap) {
            let norm = nrm2(dir.view());
            if norm > BLOCK_DROP && norm.is_finite() {
                search.push(dir);
                search_h.push(image);
            }
        }
    };
    let mut pairs = Vec::with_capacity(k);
    for (vector, (theta, residual_vec)) in block.into_iter().zip(evals.iter().zip(&residuals)) {
        let residual = nrm2(residual_vec.view());
        if !residual.is_finite() || !theta.is_finite() {
            return Err(eigen_input("non-finite residual"));
        }
        pairs.push(Eigenpair {
            vector,
            value: *theta,
            residual,
            converged: residual <= tol * (1.0 + theta.abs()),
        });
    }
    Ok(Eigenpairs {
        pairs,
        actions,
        iterations,
    })
}

fn orthonormal_block(cols: Vec<Array1<f64>>, n: usize, k: usize) -> Result<Vec<Array1<f64>>> {
    let mut basis = Vec::with_capacity(k);
    absorb(&[], &mut basis, cols);
    let mut axis = 0;
    while basis.len() < k && axis < n {
        let mut unit = Array1::zeros(n);
        unit[axis] = 1.0;
        axis += 1;
        absorb(&[], &mut basis, vec![unit]);
    }
    if basis.len() != k {
        return Err(Error::EigenBlock { k, dim: n });
    }
    Ok(basis)
}

fn absorb(locked: &[Array1<f64>], into: &mut Vec<Array1<f64>>, incoming: Vec<Array1<f64>>) {
    for mut v in incoming {
        for _ in 0..2 {
            for q in locked.iter().chain(into.iter()) {
                let overlap = dot(v.view(), q.view());
                axpy(-overlap, q.view(), &mut v);
            }
        }
        let norm = nrm2(v.view());
        if norm > BLOCK_DROP && norm.is_finite() {
            v.mapv_inplace(|c| c / norm);
            into.push(v);
        }
    }
}

fn apply_columns<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    cols: &[Array1<f64>],
    actions: &mut usize,
) -> Result<Vec<Array1<f64>>> {
    let n = x.len();
    let mut out = Vec::with_capacity(cols.len());
    for col in cols {
        if col.len() != n {
            return Err(eigen_input(
                "block column length does not match the dimension",
            ));
        }
        let image = h.apply_hessian(x, col.view());
        *actions += 1;
        if image.len() != n {
            return Err(eigen_input(
                "hessian product length does not match the dimension",
            ));
        }
        if image.iter().any(|v| !v.is_finite()) {
            return Err(eigen_input("hessian product was non-finite"));
        }
        out.push(image);
    }
    Ok(out)
}

fn ritz_sorted(q: &[Array1<f64>], aq: &[Array1<f64>]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let m = q.len();
    if m == 0 || aq.len() != m {
        return Err(eigen_input("empty trial subspace"));
    }
    let mut gram = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..=i {
            let value = dot(q[i].view(), aq[j].view());
            if !value.is_finite() {
                return Err(eigen_input("non-finite Rayleigh entry"));
            }
            gram[i][j] = value;
            gram[j][i] = value;
        }
    }
    let (evals, evecs) = jacobi_eigen(&mut gram);
    if evals.iter().any(|v| !v.is_finite()) {
        return Err(eigen_input("non-finite Ritz value"));
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| evals[a].total_cmp(&evals[b]));
    let evals: Vec<f64> = order.iter().map(|&i| evals[i]).collect();
    let mut sorted = vec![vec![0.0; m]; m];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..m {
            let coeff = evecs[row][old_col];
            if !coeff.is_finite() {
                return Err(eigen_input("non-finite Ritz vector"));
            }
            sorted[row][new_col] = coeff;
        }
    }
    Ok((evals, sorted))
}

fn align_block(cols: &mut Vec<Array1<f64>>, images: &mut Vec<Array1<f64>>) -> Result<Vec<f64>> {
    let (evals, vecs) = ritz_sorted(cols, images)?;
    let k = cols.len();
    let new_cols = mix(cols, &vecs, k)?;
    let new_images = mix(images, &vecs, k)?;
    *cols = new_cols;
    *images = new_images;
    rescale_pair(cols, images)?;
    Ok(evals.into_iter().take(k).collect())
}

fn mix(basis: &[Array1<f64>], evecs: &[Vec<f64>], cols: usize) -> Result<Vec<Array1<f64>>> {
    mix_rows(basis, evecs, 0, cols)
}

fn mix_rows(
    basis: &[Array1<f64>],
    evecs: &[Vec<f64>],
    row0: usize,
    cols: usize,
) -> Result<Vec<Array1<f64>>> {
    let n = basis.first().map_or(0, |v| v.len());
    let mut out = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut y = Array1::zeros(n);
        for (i, vector) in basis.iter().enumerate() {
            let coeff = evecs
                .get(row0 + i)
                .and_then(|row| row.get(col))
                .copied()
                .ok_or_else(|| eigen_input("Ritz coefficient is missing"))?;
            axpy(coeff, vector.view(), &mut y);
        }
        out.push(y);
    }
    Ok(out)
}

fn rescale_pair(cols: &mut [Array1<f64>], images: &mut [Array1<f64>]) -> Result<()> {
    if cols.len() != images.len() {
        return Err(eigen_input("paired block lengths differ"));
    }
    for (vector, image) in cols.iter_mut().zip(images.iter_mut()) {
        let norm = nrm2(vector.view());
        if norm <= BLOCK_DROP || !norm.is_finite() {
            return Err(eigen_input("Ritz vector vanished"));
        }
        if (norm - 1.0).abs() > 1e-12 {
            vector.mapv_inplace(|c| c / norm);
            image.mapv_inplace(|c| c / norm);
        }
    }
    Ok(())
}

fn column_residuals(
    images: &[Array1<f64>],
    cols: &[Array1<f64>],
    evals: &[f64],
) -> Result<Vec<Array1<f64>>> {
    if images.len() != cols.len() || evals.len() < cols.len() {
        return Err(eigen_input("residual block is incomplete"));
    }
    let mut out = Vec::with_capacity(cols.len());
    for i in 0..cols.len() {
        let mut residual = images[i].clone();
        axpy(-evals[i], cols[i].view(), &mut residual);
        out.push(residual);
    }
    Ok(out)
}

type PairBlock = (Vec<Array1<f64>>, Vec<Array1<f64>>);

fn deflate_directions(
    locked: &[Array1<f64>],
    locked_h: &[Array1<f64>],
    dirs: Vec<Array1<f64>>,
    images: Vec<Array1<f64>>,
) -> Result<PairBlock> {
    if locked.len() != locked_h.len() || dirs.len() != images.len() {
        return Err(eigen_input("paired block lengths differ"));
    }
    let mut kept = Vec::new();
    let mut kept_h = Vec::new();
    for (mut dir, mut image) in dirs.into_iter().zip(images) {
        if dir.len() != image.len() {
            return Err(eigen_input("paired block lengths differ"));
        }
        for _ in 0..2 {
            for (basis, basis_h) in locked
                .iter()
                .zip(locked_h.iter())
                .chain(kept.iter().zip(kept_h.iter()))
            {
                let overlap = dot(dir.view(), basis.view());
                axpy(-overlap, basis.view(), &mut dir);
                axpy(-overlap, basis_h.view(), &mut image);
            }
        }
        let norm = nrm2(dir.view());
        if norm > BLOCK_DROP && norm.is_finite() {
            dir.mapv_inplace(|c| c / norm);
            image.mapv_inplace(|c| c / norm);
            if image.iter().any(|c| !c.is_finite()) {
                return Err(eigen_input("non-finite search direction"));
            }
            kept.push(dir);
            kept_h.push(image);
        }
    }
    Ok((kept, kept_h))
}

fn argmin(vals: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..vals.len() {
        if vals[i] < vals[best] {
            best = i;
        }
    }
    best
}

fn normalize(mut v: Array1<f64>) -> Array1<f64> {
    let n = nrm2(v.view());
    if n > 1e-14 {
        v.mapv_inplace(|c| c / n);
    }
    v
}

fn jacobi_eigen(a: &mut [Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let aij = a[i][j];
                off += aij * aij;
                if aij.abs() <= 1e-15 {
                    continue;
                }
                let tau = (a[j][j] - a[i][i]) / (2.0 * aij);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let aii = a[i][i];
                let ajj = a[j][j];
                a[i][i] = c * c * aii - 2.0 * s * c * aij + s * s * ajj;
                a[j][j] = s * s * aii + 2.0 * s * c * aij + c * c * ajj;
                a[i][j] = 0.0;
                a[j][i] = 0.0;
                for k in 0..n {
                    if k != i && k != j {
                        let aik = a[i][k];
                        let ajk = a[j][k];
                        a[i][k] = c * aik - s * ajk;
                        a[k][i] = a[i][k];
                        a[j][k] = s * aik + c * ajk;
                        a[k][j] = a[j][k];
                    }
                    let vki = v[k][i];
                    let vkj = v[k][j];
                    v[k][i] = c * vki - s * vkj;
                    v[k][j] = s * vki + c * vkj;
                }
            }
        }
        if off.sqrt() <= 1e-14 {
            break;
        }
    }
    let evals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (evals, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use crate::hvp::HvpOracle;
    use ndarray::{Array2, ArrayView1, array};

    fn gapped_diag(n: usize) -> impl HessianVector {
        HvpOracle::unbounded(
            n,
            move |x| {
                let t = x[0];
                let mut g = Array1::zeros(n);
                g[0] = 4.0 * t * (t * t - 1.0);
                for i in 1..n {
                    g[i] = 4.0 * x[i];
                }
                let e = (t * t - 1.0) * (t * t - 1.0)
                    + 2.0 * x.iter().skip(1).map(|c| c * c).sum::<f64>();
                (e, g)
            },
            move |_x, v| {
                let mut hv = Array1::zeros(n);
                hv[0] = -8.0 * v[0];
                for i in 1..n {
                    hv[i] = 4.0 * v[i];
                }
                hv
            },
        )
    }

    fn dense_column_actions<H: ApplyHessian>(h: &H, x: ArrayView1<f64>, n: usize) -> LowestMode {
        let mut mat = vec![vec![0.0; n]; n];
        let mut actions = 0;
        for i in 0..n {
            let mut e = Array1::zeros(n);
            e[i] = 1.0;
            let he = h.apply_hessian(x, e.view());
            actions += 1;
            for j in 0..n {
                mat[j][i] = he[j];
            }
        }
        let (evals, evecs) = jacobi_eigen(&mut mat);
        let lowest = argmin(&evals);
        let mut mode = Array1::zeros(n);
        for i in 0..n {
            mode[i] = evecs[i][lowest];
        }
        LowestMode {
            vector: normalize(mode),
            value: evals[lowest],
            actions,
        }
    }

    #[test]
    fn recovers_the_downhill_axis_of_a_double_well_hessian() {
        let h = gapped_diag(6);
        let x = Array1::zeros(6);
        let seed = array![0.2, 0.7, 0.1, 0.0, 0.0, 0.0];
        let mode = lowest_eigenpair(&h, x.view(), seed.view(), 6).unwrap();
        assert!(mode.value < 0.0, "saddle curvature {}", mode.value);
        assert!(
            mode.vector[0].abs() > 0.9,
            "mode should lie on x, got {:?}",
            mode.vector
        );
        assert!(mode.actions <= 6);
    }

    #[test]
    fn an_empty_seed_is_a_dimension_error() {
        let h = gapped_diag(4);
        let x = Array1::zeros(4);
        let err = lowest_eigenpair(&h, x.view(), Array1::zeros(0).view(), 4).unwrap_err();
        assert!(matches!(err, Error::Dim { got: 0, .. }));
    }

    #[test]
    fn schema_ordinals_are_the_closed_enum() {
        for raw in 0u8..=14 {
            let kind = EigensolverKind::from_ordinal(raw).expect("ordinal in range");
            assert_eq!(kind as u8, raw);
        }
        assert!(EigensolverKind::from_ordinal(15).is_none());
        assert_eq!(EigensolverKind::Lanczos.name(), "lanczos");
        assert_eq!(EigensolverKind::Dimer.name(), "dimer");
        assert_eq!(EigensolverKind::EigenExa.name(), "eigenExa");
        assert_eq!(DENSE_EIGEN_CUTOFF, 512);
    }

    struct DiagH(Array1<f64>);
    impl ApplyHessian for DiagH {
        fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
            Array1::from_iter(self.0.iter().zip(v.iter()).map(|(l, vi)| l * vi))
        }
    }

    #[test]
    fn dimer_linear_diag_six_is_a_handful_of_actions() {
        let mut lam = Array1::from_elem(6, 0.5);
        lam[0] = -2.5;
        lam[1] = 0.4;
        let h = DiagH(lam);
        let x = Array1::zeros(6);
        let seed = array![0.3, 0.37, 0.44, 0.51, 0.58, 0.65];
        let mode = lowest_mode(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Dimer,
                max_iter: 20,
                tol: 1e-8,
                ..EigenParams::default()
            },
        )
        .unwrap();
        assert!((mode.value + 2.5).abs() < 1e-10, "C {}", mode.value);
        assert!(mode.vector[0].abs() > 0.999);
        assert!(
            mode.actions <= 12,
            "Jacobi planes on a 6-D diag should be cheap, got {}",
            mode.actions
        );
    }

    #[test]
    fn dimer_linear_diag_matches_cpp_bench_spectrum() {
        let lam = array![-2.5, 0.4, 0.55, 0.7, 0.85, 1.0];
        let h = DiagH(lam);
        let x = Array1::zeros(6);
        let seed = array![0.3, 0.37, 0.44, 0.51, 0.58, 0.65];
        let mode = lowest_mode(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Dimer,
                max_iter: 20,
                tol: 1e-8,
                ..EigenParams::default()
            },
        )
        .unwrap();
        assert!((mode.value + 2.5).abs() < 1e-10, "C {}", mode.value);
        eprintln!("cpp-spectrum actions={}", mode.actions);
        assert!(
            mode.actions <= 16,
            "C++ bench spectrum should not need 29 actions, got {}",
            mode.actions
        );
    }

    #[test]
    fn dimer_in_plane_rr_finishes_in_two_actions_on_a_2d_well() {
        let h = gapped_diag(2);
        let x = Array1::zeros(2);
        let seed = array![0.2, 0.7];
        let mode = lowest_mode(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Dimer,
                max_iter: 20,
                tol: 1e-8,
                ..EigenParams::default()
            },
        )
        .unwrap();
        assert!(mode.value < 0.0, "C {}", mode.value);
        assert!(mode.vector[0].abs() > 0.999);
        assert!(
            mode.actions <= 4,
            "one plane, two Hessian actions, got {}",
            mode.actions
        );
    }

    #[test]
    fn jonsson_heyden_dimer_recovers_the_gapped_mode() {
        let h = gapped_diag(6);
        let x = Array1::zeros(6);
        let seed = array![0.2, 0.7, 0.1, 0.0, 0.0, 0.0];
        let params = EigenParams {
            kind: EigensolverKind::Dimer,
            max_iter: 20,
            tol: 1e-8,
            ..EigenParams::default()
        };
        let mode = lowest_mode(&h, x.view(), seed.view(), &params).unwrap();
        assert!(mode.value < 0.0, "dimer curvature {}", mode.value);
        assert!(
            mode.vector[0].abs() > 0.9,
            "dimer mode should lie on x, got {:?}",
            mode.vector
        );
        assert!(
            mode.actions <= 4,
            "in-plane RR should finish in two actions per plane, got {}",
            mode.actions
        );
        assert!(EigensolverKind::Dimer.is_linked());
        assert_ne!(
            EigensolverKind::Dimer as u8,
            EigensolverKind::JacobiDavidson as u8
        );
    }

    #[test]
    fn matrix_free_kinds_recover_the_gapped_mode_with_fewer_actions_than_dense() {
        let n = 32;
        let h = gapped_diag(n);
        let x = Array1::zeros(n);
        let mut seed = Array1::zeros(n);
        seed[0] = 0.3;
        seed[3] = 0.7;
        seed[11] = 0.2;
        let dense = dense_column_actions(&h, x.view(), n);
        assert_eq!(dense.actions, n);
        assert!(dense.value < 0.0);
        assert!(dense.vector[0].abs() > 0.9);

        for kind in [
            EigensolverKind::Lanczos,
            EigensolverKind::RayleighRitz,
            EigensolverKind::JacobiDavidson,
            EigensolverKind::Lobpcg,
            EigensolverKind::Dimer,
        ] {
            let params = EigenParams {
                kind,
                krylov: 8,
                max_iter: 16,
                tol: 1e-6,
                nev: 1,
            };
            let mode = lowest_mode(&h, x.view(), seed.view(), &params).unwrap();
            assert!(mode.value < 0.0, "{:?} curvature {}", kind, mode.value);
            assert!(
                mode.vector[0].abs() > 0.9,
                "{:?} mode {:?}",
                kind,
                mode.vector
            );
            assert!(
                mode.actions < n,
                "{:?} used {} actions, dense uses {}",
                kind,
                mode.actions,
                n
            );
            let cos = mode
                .vector
                .iter()
                .zip(dense.vector.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .abs();
            assert!(cos > 0.9, "{kind:?} |cos| = {cos}");
        }
    }

    struct Mat(Vec<Vec<f64>>);
    impl ApplyHessian for Mat {
        fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
            let mut image = Array1::zeros(v.len());
            for (out, row) in image.iter_mut().zip(&self.0) {
                *out = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            }
            image
        }
    }

    fn dense_lowest(mut mat: Vec<Vec<f64>>, k: usize) -> (Vec<f64>, Vec<Array1<f64>>) {
        let n = mat.len();
        let (evals, evecs) = jacobi_eigen(&mut mat);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| evals[a].total_cmp(&evals[b]));
        let values: Vec<f64> = order.iter().take(k).map(|&i| evals[i]).collect();
        let mut vectors = Vec::with_capacity(k);
        for &col in order.iter().take(k) {
            let mut mode = Array1::zeros(n);
            for row in 0..n {
                mode[row] = evecs[row][col];
            }
            vectors.push(normalize(mode));
        }
        (values, vectors)
    }

    fn assert_subspace(found: &Eigenpairs, truth: &[Array1<f64>], values: &[f64], tol: f64) {
        assert_eq!(found.pairs.len(), truth.len());
        let k = truth.len();
        let mut overlap = 0.0;
        for (pair, value) in found.pairs.iter().zip(values) {
            assert!(
                (pair.value - value).abs() <= tol,
                "value {} vs {value}",
                pair.value
            );
            assert!(pair.converged, "residual {}", pair.residual);
            assert!((nrm2(pair.vector.view()) - 1.0).abs() < tol);
        }
        for left in truth {
            for pair in &found.pairs {
                let cos = dot(left.view(), pair.vector.view());
                overlap += cos * cos;
            }
        }
        assert!(
            (overlap - k as f64).abs() < tol,
            "subspace overlap {overlap} for k={k}"
        );
        for i in 0..k {
            for j in 0..i {
                let cos = dot(found.pairs[i].vector.view(), found.pairs[j].vector.view());
                assert!(cos.abs() < tol, "pair {i},{j} overlap {cos}");
            }
        }
    }

    fn random_symmetric(n: usize, seed: u64, spd: bool) -> Vec<Vec<f64>> {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let mut raw = vec![vec![0.0; n]; n];
        for row in &mut raw {
            for value in row.iter_mut() {
                *value = rng.random_range(-1.0..1.0);
            }
        }
        if spd {
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| raw.iter().map(|row| row[i] * row[j]).sum())
                        .collect()
                })
                .collect()
        } else {
            (0..n)
                .map(|i| (0..n).map(|j| 0.5 * (raw[i][j] + raw[j][i])).collect())
                .collect()
        }
    }

    fn identity_seed(n: usize, k: usize) -> Array2<f64> {
        let mut seed = Array2::zeros((n, k));
        for j in 0..k {
            seed[(j, j)] = 1.0;
            if j + 1 < n {
                seed[(j + 1, j)] = 0.3;
            }
        }
        seed
    }

    #[test]
    fn block_k1_matches_lowest_eigenpair_to_tolerance() {
        let lam = array![-2.5, 0.4, 0.55, 0.7, 0.85, 1.0];
        let h = DiagH(lam);
        let x = Array1::zeros(6);
        let seed_col = array![0.3, 0.37, 0.44, 0.51, 0.58, 0.65];
        let mut seed = Array2::zeros((6, 1));
        seed.column_mut(0).assign(&seed_col);
        let tol = 1e-8;
        let single = lowest_eigenpair(&h, x.view(), seed_col.view(), 6);
        let single = single.as_ref().unwrap();
        let block = lowest_eigenpairs(&h, x.view(), seed.view(), tol, 30).unwrap();
        assert_eq!(block.pairs.len(), 1);
        assert!((block.pairs[0].value - single.value).abs() <= tol);
        let cos = dot(block.pairs[0].vector.view(), single.vector.view()).abs();
        assert!(1.0 - cos <= tol, "|cos| = {cos}");
        assert!(block.pairs[0].converged);
    }

    #[test]
    fn block_diagonal_matches_dense_eigh() {
        let lam = array![-4.0, -0.5, 0.2, 1.5, 3.0, 8.0];
        let h = DiagH(lam.clone());
        let n = lam.len();
        let k = 3;
        let x = Array1::zeros(n);
        let found = lowest_eigenpairs(&h, x.view(), identity_seed(n, k).view(), 1e-10, 40).unwrap();
        let mut truth = Vec::new();
        let mut values = Vec::new();
        for j in 0..k {
            let mut mode = Array1::zeros(n);
            mode[j] = 1.0;
            truth.push(mode);
            values.push(lam[j]);
        }
        assert_subspace(&found, &truth, &values, 1e-6);
    }

    #[test]
    fn block_random_spd_and_indefinite_match_dense_eigh() {
        let xdim = 6;
        let k = 3;
        let x = Array1::zeros(xdim);
        for (spd, seed) in [(true, 11u64), (false, 19u64)] {
            let mat = random_symmetric(xdim, seed, spd);
            let h = Mat(mat.clone());
            let (values, truth) = dense_lowest(mat, k);
            let found =
                lowest_eigenpairs(&h, x.view(), identity_seed(xdim, k).view(), 1e-10, 40).unwrap();
            assert_subspace(&found, &truth, &values, 1e-5);
        }
    }

    #[test]
    fn block_degenerate_eigenvalues_match_the_invariant_subspace() {
        let v1 = normalize(array![1.0, 1.0, 1.0]);
        let mut v2 = array![1.0, -1.0, 0.0];
        axpy(-dot(v2.view(), v1.view()), v1.view(), &mut v2);
        let v2 = normalize(v2);
        let v3 = normalize(array![
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ]);
        let evals = [1.0, 1.0, 4.0];
        let modes = [v1.clone(), v2.clone(), v3];
        let apply = |v: ArrayView1<f64>| {
            let mut image = Array1::zeros(3);
            for (lam, mode) in evals.iter().zip(&modes) {
                axpy(*lam * dot(mode.view(), v), mode.view(), &mut image);
            }
            image
        };
        struct Outer<F>(F);
        impl<F> ApplyHessian for Outer<F>
        where
            F: Fn(ArrayView1<f64>) -> Array1<f64>,
        {
            fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
                (self.0)(v)
            }
        }
        let x = Array1::zeros(3);
        let mut seed = Array2::zeros((3, 2));
        seed[(0, 0)] = 0.2;
        seed[(1, 0)] = -0.4;
        seed[(2, 0)] = 0.7;
        seed[(0, 1)] = 0.9;
        seed[(1, 1)] = 0.1;
        seed[(2, 1)] = -0.3;
        let found = lowest_eigenpairs(&Outer(apply), x.view(), seed.view(), 1e-10, 30).unwrap();
        assert_subspace(&found, &[v1, v2], &[1.0, 1.0], 1e-5);
    }

    #[test]
    fn block_rejects_empty_too_wide_and_non_finite_input() {
        let h = DiagH(array![1.0, 2.0, 3.0]);
        let x = Array1::zeros(3);
        let empty = Array2::zeros((3, 0));
        assert!(matches!(
            lowest_eigenpairs(&h, x.view(), empty.view(), 1e-8, 4),
            Err(Error::EigenBlock { k: 0, dim: 3 })
        ));
        let wide = Array2::zeros((3, 4));
        assert!(matches!(
            lowest_eigenpairs(&h, x.view(), wide.view(), 1e-8, 4),
            Err(Error::EigenBlock { k: 4, dim: 3 })
        ));
        let mut nan_seed = Array2::zeros((3, 1));
        nan_seed[(1, 0)] = f64::NAN;
        assert!(matches!(
            lowest_eigenpairs(&h, x.view(), nan_seed.view(), 1e-8, 4),
            Err(Error::EigenInput { .. })
        ));
        let mut nan_x = x.clone();
        nan_x[0] = f64::INFINITY;
        let seed = Array2::ones((3, 1));
        assert!(matches!(
            lowest_eigenpairs(&h, nan_x.view(), seed.view(), 1e-8, 4),
            Err(Error::EigenInput { .. })
        ));
        assert!(matches!(
            lowest_eigenpairs(&h, x.view(), seed.view(), f64::NAN, 4),
            Err(Error::EigenInput { .. })
        ));
        assert!(matches!(
            lowest_eigenpairs(&h, x.view(), seed.view(), 1e-8, 0),
            Err(Error::EigenInput { .. })
        ));
        let short = Array1::zeros(2);
        assert!(matches!(
            lowest_eigenpairs(&h, short.view(), seed.view(), 1e-8, 4),
            Err(Error::Dim { got: 2, dim: 3 })
        ));
    }

    #[test]
    fn block_refuses_a_non_finite_product_without_panicking() {
        struct Boom;
        impl ApplyHessian for Boom {
            fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
                Array1::from_elem(v.len(), f64::NAN)
            }
        }
        let x = Array1::zeros(2);
        let seed = Array2::eye(2);
        let err = lowest_eigenpairs(&Boom, x.view(), seed.view(), 1e-8, 3).unwrap_err();
        assert!(matches!(err, Error::EigenInput { .. }), "{err}");
    }

    #[test]
    fn unlinked_kinds_fail_closed() {
        let h = gapped_diag(4);
        let x = Array1::zeros(4);
        let seed = array![1.0, 0.0, 0.0, 0.0];
        for raw in 4u8..=13 {
            let kind = EigensolverKind::from_ordinal(raw).unwrap();
            assert!(!kind.is_linked());
            let err = lowest_mode(
                &h,
                x.view(),
                seed.view(),
                &EigenParams {
                    kind,
                    ..EigenParams::default()
                },
            )
            .unwrap_err();
            match err {
                Error::EigenUnavailable { kind: name } => assert_eq!(name, kind.name()),
                other => panic!("expected unavailable, got {other}"),
            }
        }
    }
}
