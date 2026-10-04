//! Matrix-free lowest eigenpair of a symmetric Hessian action.
//!
//! IRC kick and the `lambda_min` sign check need one extremal pair,
//! not a full ELPA / SLATE spectrum. Dispatch is a closed
//! [`EigensolverKind`]: Lanczos, Rayleigh-Ritz, Jacobi-Davidson,
//! LOBPCG, and the Jónsson dimer with Heyden plane rotations run
//! here; PRIMME `dprimme` runs when the `primme` feature links
//! libprimme; SLEPc EPS runs when the `slepc` feature links PETSc/SLEPc;
//! every other named backend fail-closes with
//! [`Error::EigenUnavailable`]. Integers match `schema/eigen.capnp`.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::dlaf_kind::DlaFutureParams;
use crate::eigenexa_kind::EigenExaParams;
use crate::error::{Error, Result};
use crate::hvp::HessianVector;
use crate::slepc_kind::SlepcParams;
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
    /// PRIMME `dprimme`. Linked with the `primme` feature when libprimme is present.
    Primme = 4,
    /// SLEPc EPS. Linked with the `slepc` feature when PETSc/SLEPc are present.
    Slepc = 5,
    /// ChASE Chebyshev filter on assembled dense H. Linked with the `chase` feature when libchase is present.
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
    /// DLA-Future partial spectrum. `begin` fixed at 0. Not linked.
    DlaFuture = 12,
    /// EigenExa. Full spectrum only. Not linked.
    EigenExa = 13,
    /// Jónsson dimer with Heyden plane rotations. Linked. Matrix-free.
    Dimer = 14,
    /// libkrylov Davidson / JD. Linked with the `libkrylov` feature when
    /// libkrylov is present. Matrix-free Hessian-vector only.
    Libkrylov = 15,
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
            Self::Libkrylov => "libkrylov",
        }
    }

    /// Built into this crate. Unlinked kinds return [`Error::EigenUnavailable`].
    pub const fn is_linked(self) -> bool {
        matches!(
            self,
            Self::Lanczos | Self::RayleighRitz | Self::JacobiDavidson | Self::Lobpcg | Self::Dimer
        ) || (cfg!(rgmin_has_primme) && matches!(self, Self::Primme))
            || (cfg!(rgmin_has_slepc) && matches!(self, Self::Slepc))
            || (cfg!(rgmin_has_chase) && matches!(self, Self::Chase))
            || (cfg!(rgmin_has_libkrylov) && matches!(self, Self::Libkrylov))
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
                | Self::Libkrylov
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
            15 => Some(Self::Libkrylov),
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
    /// ChASE filter degree. 0 selects 20 with per-vector optimization.
    pub degree: usize,
    /// ChASE search-space extra (`nex`). 0 selects `max(8, ceil(0.2 * nev))`.
    pub extra: usize,
}

impl Default for EigenParams {
    fn default() -> Self {
        Self {
            kind: EigensolverKind::Lanczos,
            nev: 1,
            krylov: 0,
            max_iter: 0,
            tol: 0.0,
            degree: 0,
            extra: 0,
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

    /// ChASE initial degree. 0 selects 20.
    pub fn chase_degree(self) -> usize {
        if self.degree == 0 { 20 } else { self.degree }
    }

    /// ChASE `nex`. 0 selects `max(8, ceil(0.2 * nev))`, never 0.2 at `nev = 1`.
    pub fn chase_extra(self) -> usize {
        if self.extra == 0 {
            let frac = (0.2 * self.nev.max(1) as f64).ceil() as usize;
            frac.max(8)
        } else {
            self.extra
        }
    }

    /// ChASE outer iterations. 0 selects 25, not `n`.
    pub fn chase_iterations(self) -> usize {
        if self.max_iter == 0 {
            25
        } else {
            self.max_iter
        }
    }
}

/// Closed left-preconditioner on the lowest-mode waist.
///
/// Ordinals are dest tokens, not PETSc strings. [`crate::hvp::Preconditioner`]
/// implements this as `T r = P^{-1} r`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PreconditionerKind {
    /// `T = I`.
    None = 0,
    /// Inverse diagonal (Jacobi).
    Diagonal = 1,
    /// Inverse 3-by-3 blocks on a 3N packing.
    Block3 = 2,
    /// Host-supplied [`ApplyPreconditioner`].
    User = 3,
}

impl PreconditionerKind {
    /// Token name.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Diagonal => "diagonal",
            Self::Block3 => "block3",
            Self::User => "user",
        }
    }

    /// Closed ordinal, or `None`.
    pub fn from_ordinal(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::None),
            1 => Some(Self::Diagonal),
            2 => Some(Self::Block3),
            3 => Some(Self::User),
            _ => None,
        }
    }
}

/// `T r` at the current point. LOBPCG, Jacobi-Davidson, and PRIMME
/// apply this to the residual. Lanczos, dimer, and Rayleigh-Ritz
/// ignore it.
pub trait ApplyPreconditioner {
    /// Left preconditioner `T r`.
    fn apply_preconditioner(&self, x: ArrayView1<f64>, r: ArrayView1<f64>) -> Array1<f64>;
    /// Closed kind for the C waist.
    fn kind(&self) -> PreconditionerKind {
        PreconditionerKind::User
    }
}

impl ApplyPreconditioner for crate::hvp::IdentityPrecond {
    fn apply_preconditioner(&self, _x: ArrayView1<f64>, r: ArrayView1<f64>) -> Array1<f64> {
        r.to_owned()
    }
    fn kind(&self) -> PreconditionerKind {
        PreconditionerKind::None
    }
}

/// Inverse-diagonal Jacobi. `T r_i = r_i / max(|d_i|, floor)`.
#[derive(Clone, Debug)]
pub struct DiagonalJacobi {
    /// Inverse diagonal entries.
    pub inv: Array1<f64>,
}

impl DiagonalJacobi {
    /// Build `T` from a stored diagonal.
    pub fn from_diag(diag: ArrayView1<f64>) -> Self {
        Self {
            inv: Array1::from_iter(diag.iter().map(|d| 1.0 / d.abs().max(1e-8))),
        }
    }
}

impl ApplyPreconditioner for DiagonalJacobi {
    fn apply_preconditioner(&self, _x: ArrayView1<f64>, r: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter(r.iter().zip(self.inv.iter()).map(|(ri, ti)| ri * ti))
    }
    fn kind(&self) -> PreconditionerKind {
        PreconditionerKind::Diagonal
    }
}

/// Inverse 3-by-3 Jacobi on a 3N packing. Remainder entries pass through.
#[derive(Clone, Debug)]
pub struct Block3Jacobi {
    /// Inverse 3-by-3 blocks, row-major.
    pub inv_blocks: Vec<[f64; 9]>,
}

impl ApplyPreconditioner for Block3Jacobi {
    fn apply_preconditioner(&self, _x: ArrayView1<f64>, r: ArrayView1<f64>) -> Array1<f64> {
        let n = r.len();
        let mut z = r.to_owned();
        let atoms = self.inv_blocks.len().min(n / 3);
        for a in 0..atoms {
            let b = &self.inv_blocks[a];
            let i0 = 3 * a;
            let r0 = r[i0];
            let r1 = r[i0 + 1];
            let r2 = r[i0 + 2];
            z[i0] = b[0] * r0 + b[1] * r1 + b[2] * r2;
            z[i0 + 1] = b[3] * r0 + b[4] * r1 + b[5] * r2;
            z[i0 + 2] = b[6] * r0 + b[7] * r1 + b[8] * r2;
        }
        z
    }
    fn kind(&self) -> PreconditionerKind {
        PreconditionerKind::Block3
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
    lowest_mode_precond(h, x, seed, params, &crate::hvp::IdentityPrecond)
}

/// SLEPc EPS arm of [`lowest_mode`]. `params.kind` is ignored.
///
/// Unbuilt `slepc` (or the feature on without PETSc/SLEPc) returns
/// [`Error::EigenUnavailable`]. A host that already lives in PETSc
/// passes a Pmat through [`SlepcParams`].
pub fn lowest_mode_slepc<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
    slepc: &SlepcParams,
) -> Result<LowestMode> {
    if seed.is_empty() {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    #[cfg(feature = "slepc")]
    {
        let n = seed.len();
        let seed_owned = seed.to_owned();
        let (vector, value, actions) = crate::slepc_eps::solve(
            seed_owned.as_slice().unwrap(),
            params.nev.max(1),
            params.krylov_dim(n),
            params.iterations(n),
            params.tolerance(),
            slepc,
            |v| h.apply_hessian(x, ArrayView1::from(v)),
        )?;
        Ok(LowestMode {
            vector,
            value,
            actions,
        })
    }
    #[cfg(not(feature = "slepc"))]
    {
        let _ = (h, x, params, slepc);
        Err(Error::EigenUnavailable {
            kind: EigensolverKind::Slepc.name(),
        })
    }
}

/// PRIMME `dprimme` arm of [`lowest_mode`]. `params.kind` is ignored.
///
/// Unbuilt `primme` (or the feature on without libprimme) returns
/// [`Error::EigenUnavailable`]. `T` is `primme.applyPreconditioner`
/// when `t.kind()` is not [`PreconditionerKind::None`].
pub fn lowest_mode_primme<H, P>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
    t: &P,
) -> Result<LowestMode>
where
    H: ApplyHessian + ?Sized,
    P: ApplyPreconditioner + ?Sized,
{
    if seed.is_empty() {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    #[cfg(feature = "primme")]
    {
        let n = seed.len();
        let seed_owned = seed.to_owned();
        let apply = |v: &[f64]| h.apply_hessian(x, ArrayView1::from(v));
        let (vector, value, actions) = if t.kind() != PreconditionerKind::None {
            let pre = |v: &[f64]| t.apply_preconditioner(x, ArrayView1::from(v));
            crate::primme_eps::solve(
                seed_owned.as_slice().unwrap(),
                params.nev.max(1),
                params.iterations(n),
                params.tolerance(),
                apply,
                Some(&pre),
            )?
        } else {
            crate::primme_eps::solve(
                seed_owned.as_slice().unwrap(),
                params.nev.max(1),
                params.iterations(n),
                params.tolerance(),
                apply,
                None,
            )?
        };
        Ok(LowestMode {
            vector,
            value,
            actions,
        })
    }
    #[cfg(not(feature = "primme"))]
    {
        let _ = (h, x, params, t);
        Err(Error::EigenUnavailable {
            kind: EigensolverKind::Primme.name(),
        })
    }
}

/// libkrylov arm of [`lowest_mode`]. `params.kind` is ignored.
///
/// Unbuilt `libkrylov` (or the feature on without the library) returns
/// [`Error::EigenUnavailable`]. ApplyHessian is `ckrylov` `real_multiply`.
/// String keys stay inside the C shim.
pub fn lowest_mode_libkrylov<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
) -> Result<LowestMode> {
    if seed.is_empty() {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    #[cfg(feature = "libkrylov")]
    {
        let n = seed.len();
        let seed_owned = seed.to_owned();
        let (vector, value, actions) = crate::libkrylov_eps::solve(
            seed_owned.as_slice().unwrap(),
            params.nev.max(1),
            params.iterations(n),
            params.krylov_dim(n),
            params.tolerance(),
            |v| h.apply_hessian(x, ArrayView1::from(v)),
        )?;
        Ok(LowestMode {
            vector,
            value,
            actions,
        })
    }
    #[cfg(not(feature = "libkrylov"))]
    {
        let _ = (h, x, params);
        Err(Error::EigenUnavailable {
            kind: EigensolverKind::Libkrylov.name(),
        })
    }
}

/// EigenExa arm of [`lowest_mode`]. `params.kind` is ignored.
///
/// EigenExa has no supported partial `nvec`. `nev < n` is
/// [`Error::EigenFullSpectrum`]. This ApplyHessian entry never
/// assembles `H` with `n` actions; a linked dense wrap is a
/// separate entry. Unlinked full-spectrum stays
/// [`Error::EigenUnavailable`].
pub fn lowest_mode_eigenexa<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
    eigenexa: &EigenExaParams,
) -> Result<LowestMode> {
    if seed.is_empty() {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    let n = seed.len();
    let nev = params.nev;
    if nev < n {
        let _ = (h, x, eigenexa);
        return Err(Error::EigenFullSpectrum {
            kind: EigensolverKind::EigenExa.name(),
            nev,
            n,
        });
    }
    let _ = (h, x, eigenexa);
    Err(Error::EigenUnavailable {
        kind: EigensolverKind::EigenExa.name(),
    })
}

/// DLA-Future partial-spectrum arm. This is the assembled-H entry.
///
/// `dlaf.begin` must be 0 (`hermitian_eigensolver(..., 0, nev)` /
/// `il = 1`). `n` must be at least [`DENSE_EIGEN_CUTOFF`]. The
/// ApplyHessian [`lowest_mode`] path never calls this and stays
/// [`Error::EigenUnavailable`] so the waist cannot form `H` with
/// `n` actions. Unlinked full-window stays unavailable.
pub fn lowest_mode_dlaf(
    n: usize,
    params: &EigenParams,
    dlaf: &DlaFutureParams,
) -> Result<LowestMode> {
    if n == 0 {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    if dlaf.begin != 0 {
        return Err(Error::EigenBegin {
            kind: EigensolverKind::DlaFuture.name(),
            begin: dlaf.begin,
        });
    }
    if n < DENSE_EIGEN_CUTOFF {
        return Err(Error::EigenDenseCutoff {
            kind: EigensolverKind::DlaFuture.name(),
            n,
            cutoff: DENSE_EIGEN_CUTOFF,
        });
    }
    let _ = params;
    Err(Error::EigenUnavailable {
        kind: EigensolverKind::DlaFuture.name(),
    })
}

/// ChASE arm on an assembled symmetric `H`. `params.kind` is ignored.
///
/// [`lowest_mode`] on `ApplyHessian` never calls this and stays
/// [`Error::EigenUnavailable`], so the waist cannot form `H` with
/// `n` actions and does not run a matrix-free Chebyshev recurrence
/// under [`EigensolverKind::Chase`]. `extra == 0` maps through
/// [`EigenParams::chase_extra`]. A previous [`LowestMode`] seed is
/// the only approximation; there is no `char` mode flag.
/// Unbuilt `chase` stays [`Error::EigenUnavailable`].
pub fn lowest_mode_chase(
    h: ArrayView2<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
) -> Result<LowestMode> {
    if h.nrows() != h.ncols() {
        return Err(Error::Dim {
            got: h.nrows(),
            dim: h.ncols(),
        });
    }
    let n = h.nrows();
    if n == 0 {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    if seed.len() != n {
        return Err(Error::Dim {
            got: seed.len(),
            dim: n,
        });
    }
    let Some(seed_sl) = seed.as_slice() else {
        return Err(Error::Dim {
            got: seed.len(),
            dim: n,
        });
    };
    let (vector, value, actions) = crate::chase_eps::solve(
        h,
        seed_sl,
        params.nev.max(1),
        params.chase_extra(),
        params.chase_degree(),
        params.tolerance(),
    )?;
    let _ = params.chase_iterations();
    Ok(LowestMode {
        vector,
        value,
        actions,
    })
}

/// Assembled symmetric `H` entry for ELPA / ELPA2 / SLATE.
///
/// [`lowest_mode`] on `ApplyHessian` never calls this: those kinds
/// stay [`Error::EigenUnavailable`] so the waist cannot form `H`
/// with `n` actions. `n` must be at least [`DENSE_EIGEN_CUTOFF`].
/// Unlinked builds stay unavailable. The C `rgmin_lowest_eigenpair`
/// waist is Hessian-vector only.
pub fn lowest_mode_dense(h: ArrayView2<f64>, params: &EigenParams) -> Result<LowestMode> {
    if h.nrows() != h.ncols() {
        return Err(Error::Dim {
            got: h.nrows(),
            dim: h.ncols(),
        });
    }
    let n = h.nrows();
    if n == 0 {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    match params.kind {
        EigensolverKind::Elpa | EigensolverKind::Elpa2 | EigensolverKind::Slate => {
            if n < DENSE_EIGEN_CUTOFF {
                return Err(Error::EigenDenseCutoff {
                    kind: params.kind.name(),
                    n,
                    cutoff: DENSE_EIGEN_CUTOFF,
                });
            }
            let _ = h;
            Err(Error::EigenUnavailable {
                kind: params.kind.name(),
            })
        }
        other => Err(Error::EigenUnavailable { kind: other.name() }),
    }
}

/// [`lowest_mode`] with a typed left preconditioner on LOBPCG, JD, and PRIMME.
pub fn lowest_mode_precond<H, P>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
    t: &P,
) -> Result<LowestMode>
where
    H: ApplyHessian + ?Sized,
    P: ApplyPreconditioner + ?Sized,
{
    if seed.is_empty() {
        return Err(Error::Dim { got: 0, dim: 0 });
    }
    match params.kind {
        EigensolverKind::Lanczos => Ok(lanczos(h, x, seed, params.krylov_dim(seed.len()))),
        EigensolverKind::RayleighRitz => Ok(rayleigh_ritz(h, x, seed, params)),
        EigensolverKind::JacobiDavidson => Ok(jacobi_davidson(h, x, seed, params, t)),
        EigensolverKind::Lobpcg => Ok(lobpcg(h, x, seed, params, t)),
        EigensolverKind::Dimer => Ok(dimer(h, x, seed, params)),
        EigensolverKind::Primme => lowest_mode_primme(h, x, seed, params, t),
        EigensolverKind::Libkrylov => lowest_mode_libkrylov(h, x, seed, params),
        EigensolverKind::Slepc => lowest_mode_slepc(h, x, seed, params, &SlepcParams::default()),
        EigensolverKind::EigenExa => {
            lowest_mode_eigenexa(h, x, seed, params, &EigenExaParams::default())
        }
        other => Err(Error::EigenUnavailable { kind: other.name() }),
    }
}

/// eOn `client/Lanczos.cpp`: `beta < eps` on the start vector returns.
const LANCZOS_START_EPS: f64 = f64::EPSILON;
/// eOn `client/Lanczos.cpp`: `beta <= 1e-10 * |alpha|` is linear dependence.
const LANCZOS_LINDEP: f64 = 1e-10;

fn lanczos<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    krylov: usize,
) -> LowestMode {
    let (q, alpha, beta, actions) = lanczos_basis(h, x, seed, krylov);
    let n = seed.len();
    if q.is_empty() || alpha.is_empty() {
        return LowestMode {
            vector: Array1::zeros(n),
            value: 0.0,
            actions,
        };
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

/// Real-symmetric Lanczos. Two-pass full reorthogonalization
/// (SLEPc `EPS_ORTH_FULL`). Start-vector and residual tests follow
/// eOn `client/Lanczos.cpp`: refuse `||q0|| < eps`, and stop with
/// linear dependence when `beta <= 1e-10 |alpha|` without emitting
/// another column. SLEPc likewise tests breakdown before the
/// residual divide and shortens the factorization.
fn lanczos_basis<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    krylov: usize,
) -> (Vec<Array1<f64>>, Vec<f64>, Vec<f64>, usize) {
    let n = seed.len();
    let start = nrm2(seed);
    if !start.is_finite() || start < LANCZOS_START_EPS {
        return (Vec::new(), Vec::new(), Vec::new(), 0);
    }
    let m = krylov.min(n).max(1);
    let mut q: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut alpha = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m);
    q.push(seed.to_owned() / start);

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
        reorthogonalize(&mut w, &q);
        let b = nrm2(w.view());
        if !b.is_finite() || b <= LANCZOS_LINDEP * a.abs() {
            break;
        }
        beta.push(b);
        q.push(w / b);
    }
    (q, alpha, beta, actions)
}

fn reorthogonalize(w: &mut Array1<f64>, q: &[Array1<f64>]) {
    for _ in 0..2 {
        for qi in q {
            let overlap = dot(w.view(), qi.view());
            axpy(-overlap, qi.view(), w);
        }
    }
}

fn gram_inf_error(q: &[Array1<f64>]) -> f64 {
    let k = q.len();
    let mut worst: f64 = 0.0;
    for i in 0..k {
        for j in i..k {
            let g = dot(q[i].view(), q[j].view());
            let t = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((g - t).abs());
        }
    }
    worst
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
fn jd_correction<H, P>(
    h: &H,
    x: ArrayView1<f64>,
    u: ArrayView1<f64>,
    theta: f64,
    residual: ArrayView1<f64>,
    max_inner: usize,
    actions: &mut usize,
    t: &P,
) -> Option<Array1<f64>>
where
    H: ApplyHessian + ?Sized,
    P: ApplyPreconditioner + ?Sized,
{
    let n = residual.len();
    let mut b = residual.to_owned();
    b.mapv_inplace(|c| -c);
    project_against(u, &mut b);
    let mut sol = Array1::zeros(n);
    let mut r = b;
    let mut z = t.apply_preconditioner(x, r.view());
    if t.kind() != PreconditionerKind::None {
        project_against(u, &mut z);
    }
    let mut p = z.clone();
    let mut rsold = dot(r.view(), z.view());
    if rsold.abs() <= 1e-30 {
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
        if nrm2(r.view()) <= 1e-10 {
            break;
        }
        z = t.apply_preconditioner(x, r.view());
        if t.kind() != PreconditionerKind::None {
            project_against(u, &mut z);
        }
        let rsnew = dot(r.view(), z.view());
        let beta = rsnew / rsold;
        let mut p_new = z;
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

fn jacobi_davidson<H, P>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
    t: &P,
) -> LowestMode
where
    H: ApplyHessian + ?Sized,
    P: ApplyPreconditioner + ?Sized,
{
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
            t,
        )
        .unwrap_or(residual);
        if !space.try_append(h, x, next) {
            break;
        }
    }
    last.actions = space.actions + extra_actions;
    last
}

fn lobpcg<H, P>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
    t: &P,
) -> LowestMode
where
    H: ApplyHessian + ?Sized,
    P: ApplyPreconditioner + ?Sized,
{
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
        residual = t.apply_preconditioner(x, residual.view());
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
        for raw in 0u8..=15 {
            let kind = EigensolverKind::from_ordinal(raw).expect("ordinal in range");
            assert_eq!(kind as u8, raw);
        }
        assert!(EigensolverKind::from_ordinal(16).is_none());
        assert_eq!(EigensolverKind::Lanczos.name(), "lanczos");
        assert_eq!(EigensolverKind::Dimer.name(), "dimer");
        assert_eq!(EigensolverKind::EigenExa.name(), "eigenExa");
        assert_eq!(DENSE_EIGEN_CUTOFF, 512);
    }

    #[test]
    fn chase_degree_and_extra_zero_select_library_defaults() {
        let kick = EigenParams {
            kind: EigensolverKind::Chase,
            nev: 1,
            ..EigenParams::default()
        };
        assert_eq!(kick.chase_degree(), 20);
        assert_eq!(kick.chase_extra(), 8);
        assert_eq!(kick.chase_iterations(), 25);
        assert_ne!(kick.chase_extra(), kick.krylov_dim(32));
        let wide = EigenParams {
            kind: EigensolverKind::Chase,
            nev: 100,
            ..EigenParams::default()
        };
        assert_eq!(wide.chase_extra(), 20);
        let set = EigenParams {
            kind: EigensolverKind::Chase,
            nev: 1,
            degree: 12,
            extra: 4,
            max_iter: 7,
            ..EigenParams::default()
        };
        assert_eq!(set.chase_degree(), 12);
        assert_eq!(set.chase_extra(), 4);
        assert_eq!(set.chase_iterations(), 7);
        let schema = include_str!("../schema/eigen.capnp");
        assert!(schema.contains("degree @5"));
        assert!(schema.contains("extra @6"));
        assert!(!schema.contains("chase_set"));
        #[cfg(not(rgmin_has_chase))]
        assert!(!EigensolverKind::Chase.is_linked());
        #[cfg(rgmin_has_chase)]
        assert!(EigensolverKind::Chase.is_linked());
        assert!(!EigensolverKind::Chase.is_matrix_free());
    }

    #[test]
    fn chase_applyhessian_stays_unavailable_and_does_not_assemble() {
        use std::cell::Cell;
        struct CountH<'a>(&'a Cell<usize>);
        impl ApplyHessian for CountH<'_> {
            fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
                self.0.set(self.0.get() + 1);
                v.to_owned()
            }
        }
        let actions = Cell::new(0);
        let h = CountH(&actions);
        let x = Array1::zeros(4);
        let seed = array![1.0, 0.0, 0.0, 0.0];
        let err = lowest_mode(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Chase,
                ..EigenParams::default()
            },
        )
        .unwrap_err();
        match err {
            Error::EigenUnavailable { kind } => assert_eq!(kind, "chase"),
            other => panic!("expected unavailable, got {other}"),
        }
        assert_eq!(actions.get(), 0, "Chase must not assemble H from actions");
        if !EigensolverKind::Chase.is_linked() {
            let dense = ndarray::Array2::<f64>::zeros((8, 8));
            let seed8 = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            let err = lowest_mode_chase(
                dense.view(),
                seed8.view(),
                &EigenParams {
                    kind: EigensolverKind::Chase,
                    nev: 1,
                    ..EigenParams::default()
                },
            )
            .unwrap_err();
            match err {
                Error::EigenUnavailable { kind } => assert_eq!(kind, "chase"),
                other => panic!("expected unavailable, got {other}"),
            }
        }
        let kick = EigenParams {
            kind: EigensolverKind::Chase,
            nev: 1,
            ..EigenParams::default()
        };
        assert_eq!(kick.chase_extra(), 8);
        assert_ne!(kick.chase_extra(), 0);
    }

    struct DiagH(Array1<f64>);
    impl ApplyHessian for DiagH {
        fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
            Array1::from_iter(self.0.iter().zip(v.iter()).map(|(l, vi)| l * vi))
        }
    }

    #[test]
    fn preconditioner_ordinals_are_closed() {
        assert_eq!(
            PreconditionerKind::from_ordinal(0),
            Some(PreconditionerKind::None)
        );
        assert_eq!(
            PreconditionerKind::from_ordinal(1),
            Some(PreconditionerKind::Diagonal)
        );
        assert_eq!(
            PreconditionerKind::from_ordinal(2),
            Some(PreconditionerKind::Block3)
        );
        assert_eq!(
            PreconditionerKind::from_ordinal(3),
            Some(PreconditionerKind::User)
        );
        assert!(PreconditionerKind::from_ordinal(4).is_none());
        assert_eq!(crate::hvp::IdentityPrecond.kind(), PreconditionerKind::None);
    }

    #[test]
    fn lobpcg_none_recovers_gapped_32() {
        let n = 32;
        let mut lam = Array1::from_elem(n, 1.0);
        lam[0] = -2.5;
        let h = DiagH(lam);
        let x = Array1::zeros(n);
        let mut seed = Array1::zeros(n);
        seed[0] = 0.3;
        seed[3] = 0.7;
        seed[11] = 0.2;
        let mode = lowest_mode(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Lobpcg,
                max_iter: 16,
                tol: 1e-6,
                ..EigenParams::default()
            },
        )
        .unwrap();
        assert!(mode.value < 0.0);
        assert!(mode.vector[0].abs() > 0.9);
    }

    #[test]
    fn lobpcg_diagonal_uses_fewer_actions_than_none() {
        let n = 32;
        let lam = Array1::from_iter((0..n).map(|i| ((i + 1) as f64).powi(2)));
        let h = DiagH(lam.clone());
        let x = Array1::zeros(n);
        let seed = Array1::ones(n);
        let params = EigenParams {
            kind: EigensolverKind::Lobpcg,
            max_iter: 80,
            tol: 1e-8,
            ..EigenParams::default()
        };
        let none = lowest_mode(&h, x.view(), seed.view(), &params).unwrap();
        let t = DiagonalJacobi::from_diag(lam.view());
        let diag = lowest_mode_precond(&h, x.view(), seed.view(), &params, &t).unwrap();
        assert!((none.value - 1.0).abs() < 1e-4, "None Ritz {}", none.value);
        assert!(
            (diag.value - 1.0).abs() < 1e-4,
            "Diagonal Ritz {}",
            diag.value
        );
        assert!(none.vector[0].abs() > 0.9 && diag.vector[0].abs() > 0.9);
        assert!(
            diag.actions < none.actions,
            "Diagonal {} actions, None {}",
            diag.actions,
            none.actions
        );
    }

    #[test]
    fn lanczos_identity_stops_on_breakdown_without_nan() {
        let h = DiagH(Array1::ones(4));
        let x = Array1::zeros(4);
        let seed = array![1.0, 0.0, 0.0, 0.0];
        let (q, alpha, beta, actions) = lanczos_basis(&h, x.view(), seed.view(), 4);
        assert_eq!(q.len(), 1);
        assert_eq!(alpha.len(), 1);
        assert!(beta.is_empty());
        assert!((alpha[0] - 1.0).abs() < 1e-14);
        assert!(q[0].iter().all(|v| v.is_finite()));
        assert_eq!(actions, 1);
        let mode = lanczos(&h, x.view(), seed.view(), 4);
        assert!(mode.value.is_finite());
        assert!((mode.value - 1.0).abs() < 1e-12);
        assert!(mode.vector.iter().all(|v| v.is_finite()));
    }

    fn hdr_diag(n: usize) -> DiagH {
        let mut lam = Array1::zeros(n);
        let lo = 1.0_f64.ln();
        let hi = 1.0e12_f64.ln();
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            lam[i] = (lo + t * (hi - lo)).exp();
        }
        DiagH(lam)
    }

    #[test]
    fn lanczos_two_pass_reortho_holds_the_hdr_basis() {
        let n = 80;
        let h = hdr_diag(n);
        let x = Array1::zeros(n);
        let seed = Array1::from_iter((0..n).map(|i| 1.0 + i as f64 / (n as f64 - 1.0)));
        let (q, _alpha, _beta, _actions) = lanczos_basis(&h, x.view(), seed.view(), 55);
        let err = gram_inf_error(&q);
        assert!(err < 1e-10, "two-pass full reortho Gram inf-error {err}");
    }

    #[test]
    fn lanczos_recovers_the_hdr_lowest_pair() {
        let n = 80;
        let h = hdr_diag(n);
        let x = Array1::zeros(n);
        let mut seed = Array1::from_elem(n, 1e-6);
        seed[0] = 1.0;
        let mode = lanczos(&h, x.view(), seed.view(), 16);
        assert!(mode.vector.iter().all(|v| v.is_finite()));
        assert!(
            (mode.value - 1.0).abs() < 2e-3,
            "lowest Ritz {}",
            mode.value
        );
        assert!(mode.vector[0].abs() > 0.99);
    }

    #[test]
    fn lanczos_zero_start_matches_eon_eps_return() {
        let h = DiagH(Array1::ones(4));
        let x = Array1::zeros(4);
        let seed = Array1::zeros(4);
        let (q, alpha, beta, actions) = lanczos_basis(&h, x.view(), seed.view(), 4);
        assert!(q.is_empty());
        assert!(alpha.is_empty());
        assert!(beta.is_empty());
        assert_eq!(actions, 0);
        let mode = lanczos(&h, x.view(), seed.view(), 4);
        assert_eq!(mode.value, 0.0);
        assert!(mode.vector.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn lanczos_lindep_uses_eon_beta_over_alpha() {
        let h = DiagH(Array1::ones(3));
        let x = Array1::zeros(3);
        let seed = array![1.0, 0.0, 0.0];
        let (q, alpha, beta, _) = lanczos_basis(&h, x.view(), seed.view(), 3);
        assert_eq!(q.len(), 1);
        assert!((alpha[0] - 1.0).abs() < 1e-14);
        assert!(beta.is_empty());
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
            let result = lowest_mode(
                &h,
                x.view(),
                seed.view(),
                &EigenParams {
                    kind,
                    ..EigenParams::default()
                },
            );
            if kind.is_linked() && kind.is_matrix_free() {
                let mode = result.unwrap();
                let residual =
                    h.apply_hessian(x.view(), mode.vector.view()) - &mode.vector * mode.value;
                assert!(nrm2(residual.view()) < 1e-6, "{} residual", kind.name());
                assert!(mode.value < 0.0, "{} lowest mode", kind.name());
                continue;
            }
            assert!(!kind.is_linked() || kind == EigensolverKind::Chase);
            let err = result.unwrap_err();
            match err {
                Error::EigenUnavailable { kind: name } => assert_eq!(name, kind.name()),
                other => panic!("expected unavailable, got {other}"),
            }
        }
    }

    #[test]
    fn slepc_unbuilt_is_unavailable() {
        if EigensolverKind::Slepc.is_linked() {
            return;
        }
        let h = gapped_diag(4);
        let x = Array1::zeros(4);
        let seed = array![1.0, 0.0, 0.0, 0.0];
        let err = lowest_mode_slepc(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Slepc,
                ..EigenParams::default()
            },
            &SlepcParams::default(),
        )
        .unwrap_err();
        match err {
            Error::EigenUnavailable { kind } => assert_eq!(kind, "slepc"),
            other => panic!("expected unavailable, got {other}"),
        }
        assert_eq!(EigensolverKind::Slepc as u8, 5);
        assert_eq!(EigensolverKind::Slepc.name(), "slepc");
    }

    #[test]
    fn slepc_shim_typed_setters_only() {
        let shim = include_str!("slepc_shim.c");
        assert!(shim.contains("MatCreateShell"));
        assert!(shim.contains("EPSSetOperators"));
        assert!(shim.contains("EPSSetProblemType"));
        assert!(shim.contains("EPSSetType"));
        assert!(shim.contains("EPSSetWhichEigenpairs"));
        assert!(shim.contains("EPSSetDimensions"));
        assert!(shim.contains("EPSSetTolerances"));
        assert!(shim.contains("STSetType"));
        assert!(shim.contains("STSetPreconditionerMat"));
        assert!(!shim.contains("SlepcInitializeNoArguments"));
        assert!(!shim.contains("EPSSetFromOptions"));
        assert!(!shim.contains("STSetFromOptions"));
        assert!(!shim.contains("PetscOptions"));
        assert!(!shim.contains("PetscInitialize("));
        assert!(shim.contains("SlepcInitialized"));
    }

    #[test]
    fn slepc_st_ordinals_are_closed() {
        for raw in 0u8..=4 {
            let kind = crate::SlepcStKind::from_ordinal(raw).expect("ordinal in range");
            assert_eq!(kind as u8, raw);
        }
        assert!(crate::SlepcStKind::from_ordinal(5).is_none());
        assert!(crate::SlepcStKind::Sinvert.needs_pmat());
        assert!(crate::SlepcStKind::Cayley.needs_pmat());
        assert!(!crate::SlepcStKind::Default.needs_pmat());
    }

    #[cfg(rgmin_has_slepc)]
    #[test]
    fn slepc_recovers_the_gapped_mode() {
        let n = 32;
        let h = gapped_diag(n);
        let x = Array1::zeros(n);
        let mut seed = Array1::zeros(n);
        seed[0] = 0.3;
        seed[3] = 0.7;
        seed[11] = 0.2;
        let mode = lowest_mode(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Slepc,
                krylov: 8,
                max_iter: 32,
                tol: 1e-6,
                nev: 1,
            },
        );
        let mode = match mode {
            Ok(m) => m,
            Err(Error::EigenUnavailable { kind }) => {
                assert_eq!(kind, "slepc");
                return;
            }
            Err(other) => panic!("expected pair or unavailable, got {other}"),
        };
        assert!(mode.value < 0.0, "SLEPc curvature {}", mode.value);
        assert!(mode.vector[0].abs() > 0.9, "SLEPc mode {:?}", mode.vector);
        assert!(EigensolverKind::Slepc.is_linked());
    }

    #[test]
    fn primme_unbuilt_is_unavailable() {
        if EigensolverKind::Primme.is_linked() {
            return;
        }
        let h = gapped_diag(4);
        let x = Array1::zeros(4);
        let seed = array![1.0, 0.0, 0.0, 0.0];
        let err = lowest_mode_primme(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Primme,
                ..EigenParams::default()
            },
            &crate::hvp::IdentityPrecond,
        )
        .unwrap_err();
        match err {
            Error::EigenUnavailable { kind } => assert_eq!(kind, "primme"),
            other => panic!("expected unavailable, got {other}"),
        }
        assert_eq!(EigensolverKind::Primme as u8, 4);
        assert_eq!(EigensolverKind::Primme.name(), "primme");
    }

    #[test]
    fn libkrylov_unbuilt_is_unavailable() {
        if EigensolverKind::Libkrylov.is_linked() {
            return;
        }
        let h = gapped_diag(4);
        let x = Array1::zeros(4);
        let seed = array![1.0, 0.0, 0.0, 0.0];
        let err = lowest_mode_libkrylov(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Libkrylov,
                ..EigenParams::default()
            },
        )
        .unwrap_err();
        match err {
            Error::EigenUnavailable { kind } => assert_eq!(kind, "libkrylov"),
            other => panic!("expected unavailable, got {other}"),
        }
        assert_eq!(EigensolverKind::Libkrylov as u8, 15);
        assert_eq!(EigensolverKind::Libkrylov.name(), "libkrylov");
        assert!(EigensolverKind::Libkrylov.is_matrix_free());
    }

    #[test]
    fn libkrylov_shim_keeps_string_keys_private() {
        let shim = include_str!("libkrylov_shim.c");
        assert!(shim.contains("ckrylov_solve_real_equation"));
        assert!(shim.contains("real_multiply") || shim.contains("rgmin_ckrylov_multiply"));
        assert!(shim.contains("CKRYLOV_REAL_KIND"));
        let public = include_str!("ffi.rs");
        assert!(!public.contains("CKRYLOV_"));
        assert!(!public.contains("ckrylov_set_enum_option"));
        let schema = include_str!("../schema/eigen.capnp");
        assert!(schema.contains("libkrylov @15"));
        assert!(!schema.contains("ckrylov"));
    }

    #[cfg(rgmin_has_libkrylov)]
    #[test]
    fn libkrylov_recovers_the_double_well_vs_jacobi_davidson() {
        let h = gapped_diag(6);
        let x = Array1::zeros(6);
        let seed = array![0.2, 0.7, 0.1, 0.0, 0.0, 0.0];
        let jd = lowest_mode(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::JacobiDavidson,
                krylov: 8,
                max_iter: 32,
                tol: 1e-10,
                nev: 1,
                ..EigenParams::default()
            },
        )
        .unwrap();
        let mode = lowest_mode(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Libkrylov,
                krylov: 8,
                max_iter: 64,
                tol: 1e-10,
                nev: 1,
                ..EigenParams::default()
            },
        );
        let mode = match mode {
            Ok(m) => m,
            Err(Error::EigenUnavailable { kind }) => {
                assert_eq!(kind, "libkrylov");
                return;
            }
            Err(other) => panic!("expected pair or unavailable, got {other}"),
        };
        assert!(
            (mode.value - jd.value).abs() < 1e-8,
            "libkrylov {} vs JD {}",
            mode.value,
            jd.value
        );
        let cos = mode
            .vector
            .iter()
            .zip(jd.vector.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .abs();
        assert!(cos > 1.0 - 1e-6, "libkrylov |cos| = {cos}");
        assert!(EigensolverKind::Libkrylov.is_linked());
    }

    #[test]
    fn primme_shim_typed_fields_only() {
        let shim = include_str!("primme_shim.c");
        assert!(shim.contains("dprimme"));
        assert!(shim.contains("primme_initialize"));
        assert!(shim.contains("primme_smallest"));
        assert!(shim.contains("matrixMatvec"));
        assert!(shim.contains("applyPreconditioner"));
        assert!(shim.contains("initSize"));
        assert!(shim.contains("numEvals"));
        assert!(shim.contains("correctionParams.precondition"));
        assert!(!shim.contains("primme_params_set"));
        assert!(!shim.contains("primme_set_member"));
        assert!(!shim.contains("primme_get_member"));
        assert!(!shim.contains("primme_set_method"));
        assert!(shim.contains("PRIMME_MAIN_ITER_FAILURE"));
        assert!(!shim.contains("PRIMME_MAX_ITERATIONS_REACHED"));
    }

    #[cfg(rgmin_has_primme)]
    #[test]
    fn primme_recovers_the_gapped_mode() {
        let n = 32;
        let h = gapped_diag(n);
        let x = Array1::zeros(n);
        let mut seed = Array1::zeros(n);
        seed[0] = 0.3;
        seed[3] = 0.7;
        seed[11] = 0.2;
        let mode = lowest_mode(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Primme,
                krylov: 8,
                max_iter: 64,
                tol: 1e-6,
                nev: 1,
            },
        );
        let mode = match mode {
            Ok(m) => m,
            Err(Error::EigenUnavailable { kind }) => {
                assert_eq!(kind, "primme");
                return;
            }
            Err(other) => panic!("expected pair or unavailable, got {other}"),
        };
        assert!(mode.value < 0.0, "PRIMME curvature {}", mode.value);
        assert!(mode.vector[0].abs() > 0.9, "PRIMME mode {:?}", mode.vector);
        assert!(EigensolverKind::Primme.is_linked());
    }

    #[cfg(rgmin_has_primme)]
    #[test]
    fn primme_diagonal_t_recovers_the_gapped_mode() {
        let n = 32;
        let h = gapped_diag(n);
        let x = Array1::zeros(n);
        let mut seed = Array1::zeros(n);
        seed[0] = 0.3;
        seed[3] = 0.7;
        seed[11] = 0.2;
        let diag = Array1::from_iter((0..n).map(|i| if i == 0 { -8.0 } else { 4.0 }));
        let t = DiagonalJacobi::from_diag(diag.view());
        let mode = lowest_mode_primme(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Primme,
                krylov: 8,
                max_iter: 64,
                tol: 1e-6,
                nev: 1,
            },
            &t,
        );
        let mode = match mode {
            Ok(m) => m,
            Err(Error::EigenUnavailable { kind }) => {
                assert_eq!(kind, "primme");
                return;
            }
            Err(other) => panic!("expected pair or unavailable, got {other}"),
        };
        assert!(mode.value < 0.0, "PRIMME+T curvature {}", mode.value);
        assert!(
            mode.vector[0].abs() > 0.9,
            "PRIMME+T mode {:?}",
            mode.vector
        );
    }

    #[test]
    fn eigenexa_nev_less_than_n_refuses() {
        let h = gapped_diag(4);
        let x = Array1::zeros(4);
        let seed = array![1.0, 0.0, 0.0, 0.0];
        let err = lowest_mode(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::EigenExa,
                nev: 1,
                ..EigenParams::default()
            },
        )
        .unwrap_err();
        match err {
            Error::EigenFullSpectrum { kind, nev, n } => {
                assert_eq!(kind, "eigenExa");
                assert_eq!(nev, 1);
                assert_eq!(n, 4);
            }
            other => panic!("expected full-spectrum refuse, got {other}"),
        }
    }

    #[test]
    fn eigenexa_full_n_stays_unavailable_and_does_not_assemble() {
        use std::cell::Cell;
        struct CountH<'a>(&'a Cell<usize>);
        impl ApplyHessian for CountH<'_> {
            fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
                self.0.set(self.0.get() + 1);
                v.to_owned()
            }
        }
        let actions = Cell::new(0);
        let h = CountH(&actions);
        let x = Array1::zeros(4);
        let seed = array![1.0, 0.0, 0.0, 0.0];
        let err = lowest_mode_eigenexa(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::EigenExa,
                nev: 4,
                ..EigenParams::default()
            },
            &EigenExaParams::default(),
        )
        .unwrap_err();
        match err {
            Error::EigenUnavailable { kind } => assert_eq!(kind, "eigenExa"),
            other => panic!("expected unavailable, got {other}"),
        }
        assert_eq!(
            actions.get(),
            0,
            "EigenExa must not assemble H from actions"
        );
        assert!(!EigensolverKind::EigenExa.is_linked());
        assert_eq!(EigensolverKind::EigenExa as u8, 13);
    }

    #[test]
    fn eigenexa_algo_ordinals_are_closed() {
        assert_eq!(crate::EigenExaAlgo::S as u8, 0);
        assert_eq!(crate::EigenExaAlgo::Sx as u8, 1);
        assert_eq!(crate::EigenExaAlgo::S.name(), "eigen_s");
        assert_eq!(crate::EigenExaAlgo::Sx.name(), "eigen_sx");
        assert!(crate::EigenExaAlgo::from_ordinal(2).is_none());
        assert_eq!(
            crate::EigenExaParams::default().algo,
            crate::EigenExaAlgo::Sx
        );
    }

    #[test]
    fn dlaf_begin_nonzero_is_rejected() {
        let err = lowest_mode_dlaf(
            DENSE_EIGEN_CUTOFF,
            &EigenParams {
                kind: EigensolverKind::DlaFuture,
                nev: 1,
                ..EigenParams::default()
            },
            &DlaFutureParams { begin: 3 },
        )
        .unwrap_err();
        match err {
            Error::EigenBegin { kind, begin } => {
                assert_eq!(kind, "dlaFuture");
                assert_eq!(begin, 3);
            }
            other => panic!("expected begin refuse, got {other}"),
        }
    }

    #[test]
    fn dlaf_below_cutoff_is_rejected() {
        let err = lowest_mode_dlaf(
            32,
            &EigenParams {
                kind: EigensolverKind::DlaFuture,
                nev: 1,
                ..EigenParams::default()
            },
            &DlaFutureParams::default(),
        )
        .unwrap_err();
        match err {
            Error::EigenDenseCutoff { kind, n, cutoff } => {
                assert_eq!(kind, "dlaFuture");
                assert_eq!(n, 32);
                assert_eq!(cutoff, DENSE_EIGEN_CUTOFF);
            }
            other => panic!("expected cutoff refuse, got {other}"),
        }
    }

    #[test]
    fn dlaf_applyhessian_stays_unavailable_and_does_not_assemble() {
        use std::cell::Cell;
        struct CountH<'a>(&'a Cell<usize>);
        impl ApplyHessian for CountH<'_> {
            fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
                self.0.set(self.0.get() + 1);
                v.to_owned()
            }
        }
        let actions = Cell::new(0);
        let h = CountH(&actions);
        let x = Array1::zeros(4);
        let seed = array![1.0, 0.0, 0.0, 0.0];
        let err = lowest_mode(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::DlaFuture,
                ..EigenParams::default()
            },
        )
        .unwrap_err();
        match err {
            Error::EigenUnavailable { kind } => assert_eq!(kind, "dlaFuture"),
            other => panic!("expected unavailable, got {other}"),
        }
        assert_eq!(
            actions.get(),
            0,
            "DLA-Future must not assemble H from actions"
        );
        let err = lowest_mode_dlaf(
            DENSE_EIGEN_CUTOFF,
            &EigenParams {
                kind: EigensolverKind::DlaFuture,
                nev: 1,
                ..EigenParams::default()
            },
            &DlaFutureParams::default(),
        )
        .unwrap_err();
        match err {
            Error::EigenUnavailable { kind } => assert_eq!(kind, "dlaFuture"),
            other => panic!("expected unavailable, got {other}"),
        }
        assert!(!EigensolverKind::DlaFuture.is_linked());
        assert_eq!(EigensolverKind::DlaFuture as u8, 12);
        assert_eq!(DlaFutureParams::default().begin, 0);
    }

    #[test]
    fn dense_applyhessian_elpa_slate_stay_unavailable_and_do_not_assemble() {
        use std::cell::Cell;
        struct CountH<'a>(&'a Cell<usize>);
        impl ApplyHessian for CountH<'_> {
            fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
                self.0.set(self.0.get() + 1);
                v.to_owned()
            }
        }
        let x = Array1::zeros(4);
        let seed = array![1.0, 0.0, 0.0, 0.0];
        for kind in [
            EigensolverKind::Elpa,
            EigensolverKind::Elpa2,
            EigensolverKind::Slate,
        ] {
            let actions = Cell::new(0);
            let h = CountH(&actions);
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
            assert_eq!(actions.get(), 0, "{kind:?} must not assemble H");
            assert!(!kind.is_linked());
        }
        assert_eq!(EigenParams::default().kind, EigensolverKind::Lanczos);
    }

    #[test]
    fn dense_entry_below_cutoff_is_rejected() {
        let h = ndarray::Array2::<f64>::zeros((32, 32));
        let err = lowest_mode_dense(
            h.view(),
            &EigenParams {
                kind: EigensolverKind::Elpa,
                nev: 1,
                ..EigenParams::default()
            },
        )
        .unwrap_err();
        match err {
            Error::EigenDenseCutoff { kind, n, cutoff } => {
                assert_eq!(kind, "elpa");
                assert_eq!(n, 32);
                assert_eq!(cutoff, DENSE_EIGEN_CUTOFF);
            }
            other => panic!("expected cutoff, got {other}"),
        }
    }

    #[test]
    fn dense_entry_at_cutoff_stays_unavailable() {
        let h = ndarray::Array2::<f64>::zeros((DENSE_EIGEN_CUTOFF, DENSE_EIGEN_CUTOFF));
        for kind in [
            EigensolverKind::Elpa,
            EigensolverKind::Elpa2,
            EigensolverKind::Slate,
        ] {
            let err = lowest_mode_dense(
                h.view(),
                &EigenParams {
                    kind,
                    nev: 1,
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
