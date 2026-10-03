//! Matrix-free lowest eigenpair of a symmetric Hessian action.
//!
//! IRC kick and the `lambda_min` sign check need one extremal pair,
//! not a full ELPA / SLATE spectrum. Dispatch is a closed
//! [`EigensolverKind`]: Lanczos, Rayleigh-Ritz, Jacobi-Davidson,
//! LOBPCG, and the Jónsson dimer with Heyden plane rotations run
//! here; every other named backend fail-closes with
//! [`Error::EigenUnavailable`]. Integers match `schema/eigen.capnp`.

use ndarray::{Array1, ArrayView1};

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
/// [`EigensolverKind::Lanczos`].
pub fn lowest_eigenpair<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    krylov: usize,
) -> LowestMode {
    let params = EigenParams {
        kind: EigensolverKind::Lanczos,
        krylov,
        ..EigenParams::default()
    };
    lowest_mode(h, x, seed, &params).expect("Lanczos is linked")
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

    fn ritz(&self, n: usize) -> Option<(f64, Array1<f64>, Array1<f64>, Array1<f64>)> {
        let k = self.q.len();
        if k == 0 {
            return None;
        }
        let mut t = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..=i {
                let value = dot(self.q[i].view(), self.aq[j].view());
                t[i][j] = value;
                t[j][i] = value;
            }
        }
        let (evals, evecs) = jacobi_eigen(&mut t);
        let lowest = argmin(&evals);
        let mut mode = Array1::zeros(n);
        let mut amode = Array1::zeros(n);
        for i in 0..k {
            axpy(evecs[i][lowest], self.q[i].view(), &mut mode);
            axpy(evecs[i][lowest], self.aq[i].view(), &mut amode);
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
    use ndarray::array;

    fn gapped_diag(
        n: usize,
    ) -> HvpOracle<
        impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
        impl Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    > {
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
        let mode = lowest_eigenpair(&h, x.view(), seed.view(), 6);
        assert!(mode.value < 0.0, "saddle curvature {}", mode.value);
        assert!(
            mode.vector[0].abs() > 0.9,
            "mode should lie on x, got {:?}",
            mode.vector
        );
        assert!(mode.actions <= 6);
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
            assert!(cos > 0.9, "{:?} |cos| = {cos}", kind);
        }
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
