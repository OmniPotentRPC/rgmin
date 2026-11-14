//! Narrow C waist for rgmin. Every vector is a DLPack tensor (dlpk),
//! the same contract as eindir_objective_t, so a later device kernel
//! does not change the ABI.

use std::cell::{Cell, RefCell};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::slice;

use dlpk::sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensorVersioned, DLPackVersion,
    DLTensor,
};
use eindir_core::ffi::{
    eindir_abi_stamp_t, eindir_core_abi_compatible, eindir_objective_eval, eindir_objective_grad,
    eindir_objective_has_grad, eindir_objective_t, eindir_status_t,
};
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::holonomic::{ConstraintError, ConstraintSolver, DistanceConstraints};
use crate::precon::{ExpParams, ExpState, Neighbours};
use crate::{
    Accept, ApplyBlockPreconditioner, ApplyHessian, Conjugacy, Control, DirectionalCurvature,
    EigenParams, EigensolverKind, Error, HessianObjective, HessianOracle, HessianVector, HvpOracle,
    LbfgsH0, LbfgsPairs, LineSearch, ManifoldKind, Method, NewtonKind, Oracle, QnStep, Restart,
    RiemannianManifold, ScgOptions, ScgParams, ScgStepTolerance, SearchStep, Solver, SpdMetric,
    StiefelRetraction, StopNorm, SuppliedSearch, lowest_eigenpairs, lowest_eigenpairs_precond,
    lowest_mode, minimize_method, minimize_method_hess, minimize_method_hvp,
    minimize_scg_exact_with_options, minimize_scg_with_options,
};

/// Status codes. 0 is success, matching metatensor / eindir.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_status_t {
    /// Completed.
    RGMIN_SUCCESS = 0,
    /// Null pointer or inconsistent length.
    RGMIN_INVALID_PARAMETER = 1,
    /// Panic or internal failure behind the C boundary.
    RGMIN_INTERNAL_ERROR = 2,
    /// Tensor is not on a device this build can evaluate (GPU later).
    RGMIN_UNSUPPORTED_DEVICE = 3,
    /// Named eigensolver is not linked in this build.
    RGMIN_UNAVAILABLE = 4,
    /// A distance constraint set was refused or could not be met.
    /// [`rgmin_last_constraint_error`] names the kind. ABI minor 31.
    RGMIN_CONSTRAINT_ERROR = 5,
}

fn status_from_error(e: &Error) -> rgmin_status_t {
    set_last_error(&e.to_string());
    match e {
        Error::Oracle { .. } | Error::FireStalled { .. } | Error::AcceptStalled { .. } => {
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
        Error::EigenUnavailable { .. } => rgmin_status_t::RGMIN_UNAVAILABLE,
        Error::Constraint(c) => {
            LAST_CONSTRAINT.with(|kind| kind.set(constraint_kind(c)));
            rgmin_status_t::RGMIN_CONSTRAINT_ERROR
        }
        _ => rgmin_status_t::RGMIN_INVALID_PARAMETER,
    }
}

/// Compatibility identity for the rgmin C ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct rgmin_abi_stamp_t {
    /// Incompatible changes increment this value.
    pub abi_major: u16,
    /// Additive compatible changes increment this value.
    pub abi_minor: u16,
    /// Struct and function-layout revision for this ABI major/minor.
    pub layout_revision: u16,
}

pub const RGMIN_ABI_VERSION_MAJOR: u16 = 1;
pub const RGMIN_ABI_VERSION_MINOR: u16 = 43;
pub const RGMIN_ABI_LAYOUT_REVISION: u16 = 5;

/// Method tag. Keep this a closed C enum; Rust [`Method`] is the source.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_method_t {
    /// Polak-Ribiere NLCG + Brent.
    RGMIN_POLAK_RIBIERE = 0,
    /// Fletcher-Reeves NLCG + Brent.
    RGMIN_FLETCHER_REEVES = 1,
    /// Dense inverse-BFGS.
    RGMIN_BFGS = 2,
    /// Limited-memory BFGS.
    RGMIN_LBFGS = 3,
    /// Inverse SR1.
    RGMIN_SR1 = 4,
    /// Adam + Brent.
    RGMIN_ADAM = 5,
    /// Steepest descent.
    RGMIN_STEEPEST = 6,
    /// SR2 Hessian update.
    RGMIN_SR2 = 7,
    /// Particle swarm.
    RGMIN_PSO = 8,
    /// Hestenes-Stiefel NLCG + Brent.
    RGMIN_HESTENES_STIEFEL = 9,
    /// Dai-Yuan NLCG + Brent.
    RGMIN_DAI_YUAN = 10,
    /// Fletcher conjugate-descent NLCG + Brent.
    RGMIN_CONJUGATE_DESCENT = 11,
    /// Hager-Zhang NLCG + Brent.
    RGMIN_HAGER_ZHANG = 12,
    /// Liu-Storey NLCG + Brent.
    RGMIN_LIU_STOREY = 13,
    /// Gilbert-Nocedal FR-PR hybrid NLCG + Brent.
    RGMIN_FR_PR = 14,
    /// Shifted Newton on a caller-supplied Hessian.
    RGMIN_NEWTON = 15,
    /// Banerjee / Baker RFO on a caller-supplied Hessian.
    RGMIN_RFO = 16,
    /// FIRE (Bitzek 2006).
    RGMIN_FIRE = 17,
    /// Barzilai-Borwein spectral steepest descent.
    RGMIN_BB = 18,
    /// Powell dogleg on a caller-supplied Hessian.
    RGMIN_DOGLEG = 19,
    /// FIRE 2.0 (Guénolé 2020).
    RGMIN_FIRE2 = 20,
    /// Quick-min. Released 0.5.2 discriminant.
    RGMIN_QUICKMIN = 21,
    /// Pulay residual subspace. `memory` 0 means 6 stored pairs.
    RGMIN_DIIS = 22,
    /// ODE12r: adaptive Euler on the (preconditioned) force.
    RGMIN_ODE12R = 23,
    /// Steihaug-Toint Newton-CG. Released 0.5.2 discriminant.
    RGMIN_NEWTON_CG = 24,
    /// Newton-CG with a Nyström preconditioner. Released 0.5.2 discriminant.
    RGMIN_NEWTON_PCG = 25,
    /// Polak-Ribiere-Polyak, `max(0, β)`. Develop surface, past the released slots.
    RGMIN_POLAK_RIBIERE_PLUS = 26,
    /// eOn Polak-Ribiere with the Powell restart at 0.5. Develop surface.
    RGMIN_EON_CG = 27,
}

/// Closed leaf conjugacy. Integers match dest [`Conjugacy`] declaration
/// order. Not [`rgmin_method_t`] (that enum is the solver axis).
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_conjugacy_t {
    RGMIN_CONJUGACY_FLETCHER_REEVES = 0,
    RGMIN_CONJUGACY_POLAK_RIBIERE = 1,
    RGMIN_CONJUGACY_HESTENES_STIEFEL = 2,
    RGMIN_CONJUGACY_DAI_YUAN = 3,
    RGMIN_CONJUGACY_CONJUGATE_DESCENT = 4,
    RGMIN_CONJUGACY_HAGER_ZHANG = 5,
    RGMIN_CONJUGACY_LIU_STOREY = 6,
    RGMIN_CONJUGACY_FR_PR = 7,
    RGMIN_CONJUGACY_POLAK_RIBIERE_PLUS = 8,
}

/// HiGHS `solver`. Integers match [`crate::HighsSolverKind`].
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_highs_solver_t {
    RGMIN_HIGHS_CHOOSE = 0,
    RGMIN_HIGHS_SIMPLEX = 1,
    RGMIN_HIGHS_IPM = 2,
    RGMIN_HIGHS_IPX = 3,
    RGMIN_HIGHS_HIPO = 4,
    RGMIN_HIGHS_PDLP = 5,
    RGMIN_HIGHS_HIPDLP = 6,
    RGMIN_HIGHS_QPASM = 7,
}

/// HiGHS `run_crossover`. Integers match [`crate::HighsCrossover`].
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_highs_crossover_t {
    RGMIN_HIGHS_CROSSOVER_CHOOSE = 0,
    RGMIN_HIGHS_CROSSOVER_ON = 1,
    RGMIN_HIGHS_CROSSOVER_OFF = 2,
}

/// Why a finished minimization stopped. `Iterating` is not a C result.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_stop_t {
    /// `||g||` fell below `gtol`.
    RGMIN_STOP_TOLERANCE = 0,
    /// The stationarity measure fell below `c * sqrt(n_free) * noise_eps`.
    RGMIN_STOP_NOISE_FLOOR = 1,
    /// SCG stopped on its step or objective test with `||g||` still above `gtol`.
    RGMIN_STOP_STEP_TOLERANCE = 2,
    /// The iteration budget was spent above every stop.
    RGMIN_STOP_BUDGET = 3,
}

/// Iteration controls. `memory` is used only by L-BFGS (0 means 10).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rgmin_control_t {
    /// Maximum iterations.
    pub maxiter: usize,
    /// Stop when `||g||_2 < gtol`.
    pub gtol: f64,
    /// Initial line-search step. Non-positive selects 1. An energy-tested
    /// steepest, BFGS, SR1, SR2, BB or unpreconditioned CG session then
    /// takes `min(1, 1/||g||)` at its first step as the step scale, until
    /// `forget` or `rebase`.
    pub istep: f64,
    /// L-BFGS correction pairs; 0 selects 10.
    pub memory: usize,
    /// Euclidean cap. Positive is the cap. Zero keeps the defaults,
    /// including the default per-atom cap. Negative leaves every cap off
    /// (layout revision 5; the released 0.5.2 meaning).
    pub maxmove: f64,
    /// Gradient-projection noise. Non-positive leaves the floor off.
    pub noise_eps: f64,
    /// Floor factor. Non-positive selects 3.
    pub noise_c: f64,
}

/// Result written by [`rgmin_minimize`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rgmin_report_t {
    /// `f(x)` at the accepted point.
    pub value: f64,
    /// Outer iterations.
    pub steps: usize,
    /// Gradient norm at the accepted point.
    pub grad_norm: f64,
    /// Which stop produced this report.
    pub stop: rgmin_stop_t,
}

/// `f(x)` callback. `x` is a rank-1 f64 DLPack tensor.
pub type rgmin_eval_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t;

/// `∇f(x)` callback. Writes into the pre-allocated `grad_out` tensor.
pub type rgmin_grad_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    grad_out: *mut DLManagedTensorVersioned,
) -> rgmin_status_t;

/// `∇²f(x)` callback. Writes a length-`n²` row-major Hessian into `hess_out`.
pub type rgmin_hess_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    hess_out: *mut DLManagedTensorVersioned,
) -> rgmin_status_t;

/// Fused `(f, ∇f)` callback. One geometry, one host potential call.
pub type rgmin_evalgrad_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
    grad_out: *mut DLManagedTensorVersioned,
) -> rgmin_status_t;

/// Directional curvature `dᵀ ∇²f(x) d`. Return [`rgmin_status_t::RGMIN_SUCCESS`]
/// and write the scalar, or any other status to fall back to the SCG probe.
pub type rgmin_curv_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    d: *const DLManagedTensorVersioned,
    curv_out: *mut f64,
) -> rgmin_status_t;

/// Møller SCG damping / tolerances. Null at the C entry selects
/// [`ScgParams::default`] plus [`Conjugacy::PolakRibiere`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rgmin_scg_params_t {
    /// Base finite-difference curvature probe (`sigma_0`).
    pub sigma0: f64,
    /// Initial Levenberg-Marquardt damping.
    pub lambda: f64,
    /// Stall when `lambda` reaches this.
    pub lambda_limit: f64,
    /// `||α d||_∞` solution tolerance.
    pub tol_sol: f64,
    /// Relative objective-change tolerance.
    pub tol_func: f64,
    /// Leaf conjugacy as [`xts_conjugacy_t`]. Stored as `i32` so an
    /// unknown C enumerant is not UB on a closed Rust enum.
    pub conjugacy: i32,
}

/// Closed eigensolver tag. Integers match `schema/eigen.capnp`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_eigen_kind_t {
    RGMIN_EIGEN_LANCZOS = 0,
    RGMIN_EIGEN_RAYLEIGH_RITZ = 1,
    RGMIN_EIGEN_JACOBI_DAVIDSON = 2,
    RGMIN_EIGEN_LOBPCG = 3,
    RGMIN_EIGEN_PRIMME = 4,
    RGMIN_EIGEN_SLEPC = 5,
    RGMIN_EIGEN_CHASE = 6,
    RGMIN_EIGEN_ELPA = 7,
    RGMIN_EIGEN_ELPA2 = 8,
    RGMIN_EIGEN_SLATE = 9,
    RGMIN_EIGEN_MAGMA = 10,
    RGMIN_EIGEN_CUSOLVER = 11,
    RGMIN_EIGEN_DLA_FUTURE = 12,
    RGMIN_EIGEN_EIGENEXA = 13,
    RGMIN_EIGEN_DIMER = 14,
    RGMIN_EIGEN_LIBKRYLOV = 15,
}

/// Typed lowest-mode parameters. No string fields. Null at the C
/// entry selects Lanczos defaults (`nev = 1`, `krylov = 0`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rgmin_eigen_params_t {
    /// [`rgmin_eigen_kind_t`] stored as `i32` so an unknown enumerant
    /// is not UB on the closed Rust enum.
    pub kind: i32,
    /// Extremal pairs. IRC kick uses 1.
    pub nev: u32,
    /// Krylov / subspace cap. 0 selects `min(n, 12)`.
    pub krylov: u32,
    /// Outer iterations. 0 selects `n`.
    pub max_iter: u32,
    /// Residual tolerance. Non-positive selects `1e-8`.
    pub tol: f64,
}

/// Result of [`rgmin_lowest_eigenpair`]. The vector is written to
/// the caller tensor.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rgmin_lowest_mode_t {
    /// Rayleigh quotient.
    pub value: f64,
    /// Hessian actions consumed.
    pub actions: usize,
}

/// `H(x) v` callback. Writes into the pre-allocated `hv_out` tensor.
pub type rgmin_hvp_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    v: *const DLManagedTensorVersioned,
    hv_out: *mut DLManagedTensorVersioned,
) -> rgmin_status_t;

/// Apply preconditioner `T` to one block. `block` and `out` are row-major
/// n-by-m f64 tensors, rank 2 with shape `(n, m)`. A null function pointer
/// at [`rgmin_lowest_eigenpairs_ex`] is the identity and is not called.
pub type rgmin_precond_fn = unsafe extern "C" fn(
    user: *mut c_void,
    block: *const DLManagedTensorVersioned,
    out: *mut DLManagedTensorVersioned,
) -> rgmin_status_t;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
    static LAST_CONSTRAINT: Cell<rgmin_constraint_error_t> =
        const { Cell::new(rgmin_constraint_error_t::RGMIN_CONSTRAINT_NONE) };
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        let c = CString::new(msg).unwrap_or_else(|_| CString::new("(interior NUL)").unwrap());
        *cell.borrow_mut() = c;
    });
    LAST_CONSTRAINT.with(|kind| kind.set(rgmin_constraint_error_t::RGMIN_CONSTRAINT_NONE));
}

/// Last error on this thread. Valid until the next rgmin C call.
#[unsafe(no_mangle)]
pub extern "C" fn rgmin_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
}

/// The same borrowed string as [`rgmin_last_error`].
///
/// The pointer stays valid until the next rgmin call on this thread.
/// The caller does not free it.
#[unsafe(no_mangle)]
pub extern "C" fn rgmin_last_error_message() -> *const c_char {
    rgmin_last_error()
}

/// Package version, NUL-terminated.
#[unsafe(no_mangle)]
pub extern "C" fn rgmin_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Return the compatibility identity for this C ABI build.
#[unsafe(no_mangle)]
pub extern "C" fn rgmin_abi_stamp() -> rgmin_abi_stamp_t {
    rgmin_abi_stamp_t {
        abi_major: RGMIN_ABI_VERSION_MAJOR,
        abi_minor: RGMIN_ABI_VERSION_MINOR,
        layout_revision: RGMIN_ABI_LAYOUT_REVISION,
    }
}

/// Return nonzero when a caller's ABI identity is accepted by this build.
///
/// # Safety
/// `stamp` is null or points to a readable [`rgmin_abi_stamp_t`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_abi_compatible(stamp: *const rgmin_abi_stamp_t) -> i32 {
    if stamp.is_null() {
        return 0;
    }
    let stamp = unsafe { &*stamp };
    i32::from(
        stamp.abi_major == RGMIN_ABI_VERSION_MAJOR
            && stamp.layout_revision == RGMIN_ABI_LAYOUT_REVISION,
    )
}

fn conjugacy_from_c(raw: i32) -> Result<Conjugacy, rgmin_status_t> {
    match raw {
        0 => Ok(Conjugacy::FletcherReeves),
        1 => Ok(Conjugacy::PolakRibiere),
        2 => Ok(Conjugacy::HestenesStiefel),
        3 => Ok(Conjugacy::DaiYuan),
        4 => Ok(Conjugacy::ConjugateDescent),
        5 => Ok(Conjugacy::HagerZhang),
        6 => Ok(Conjugacy::LiuStorey),
        7 => Ok(Conjugacy::FrPr),
        8 => Ok(Conjugacy::PolakRibierePlus),
        other => {
            set_last_error(&format!("rgmin_minimize_scg: unknown conjugacy {other}"));
            Err(rgmin_status_t::RGMIN_INVALID_PARAMETER)
        }
    }
}

fn method_from_c(m: rgmin_method_t, memory: usize) -> Method {
    match m {
        rgmin_method_t::RGMIN_POLAK_RIBIERE => Method::polak_ribiere(),
        rgmin_method_t::RGMIN_FLETCHER_REEVES => Method::nlcg(crate::Conjugacy::FletcherReeves),
        rgmin_method_t::RGMIN_BFGS => Method::Bfgs,
        rgmin_method_t::RGMIN_LBFGS => Method::Lbfgs {
            memory: if memory == 0 { 10 } else { memory },
        },
        rgmin_method_t::RGMIN_SR1 => Method::Sr1,
        rgmin_method_t::RGMIN_ADAM => Method::adam(),
        rgmin_method_t::RGMIN_STEEPEST => Method::Steepest,
        rgmin_method_t::RGMIN_SR2 => Method::Sr2,
        rgmin_method_t::RGMIN_PSO => Method::pso(),
        rgmin_method_t::RGMIN_HESTENES_STIEFEL => Method::nlcg(crate::Conjugacy::HestenesStiefel),
        rgmin_method_t::RGMIN_DAI_YUAN => Method::nlcg(crate::Conjugacy::DaiYuan),
        rgmin_method_t::RGMIN_CONJUGATE_DESCENT => Method::nlcg(crate::Conjugacy::ConjugateDescent),
        rgmin_method_t::RGMIN_HAGER_ZHANG => Method::nlcg(crate::Conjugacy::HagerZhang),
        rgmin_method_t::RGMIN_LIU_STOREY => Method::nlcg(crate::Conjugacy::LiuStorey),
        rgmin_method_t::RGMIN_FR_PR => Method::nlcg(crate::Conjugacy::FrPr),
        rgmin_method_t::RGMIN_NEWTON => Method::Newton {
            kind: NewtonKind::Shifted,
        },
        rgmin_method_t::RGMIN_RFO => Method::Newton {
            kind: NewtonKind::Rfo,
        },
        rgmin_method_t::RGMIN_FIRE => Method::Fire {
            kind: crate::FireKind::V1,
        },
        rgmin_method_t::RGMIN_BB => Method::Bb,
        rgmin_method_t::RGMIN_DOGLEG => Method::Dogleg,
        rgmin_method_t::RGMIN_FIRE2 => Method::Fire {
            kind: crate::FireKind::V2,
        },
        rgmin_method_t::RGMIN_QUICKMIN => Method::QuickMin,
        rgmin_method_t::RGMIN_ODE12R => Method::Ode12r,
        rgmin_method_t::RGMIN_DIIS => Method::Diis {
            memory: if memory == 0 { 6 } else { memory },
        },
        rgmin_method_t::RGMIN_NEWTON_CG => Method::NewtonCg,
        rgmin_method_t::RGMIN_NEWTON_PCG => Method::NewtonPcg,
        rgmin_method_t::RGMIN_POLAK_RIBIERE_PLUS => Method::polak_ribiere_plus(),
        rgmin_method_t::RGMIN_EON_CG => Method::eon_cg(),
    }
}

fn method_to_c(method: &Method) -> Result<rgmin_method_t, ()> {
    let tag = match method {
        Method::Nlcg { conjugacy, restart } => match (conjugacy, restart) {
            (Conjugacy::PolakRibiere, crate::Restart::Powell { threshold })
                if *threshold == 0.5 =>
            {
                rgmin_method_t::RGMIN_EON_CG
            }
            (Conjugacy::FletcherReeves, _) => rgmin_method_t::RGMIN_FLETCHER_REEVES,
            (Conjugacy::PolakRibiere, _) => rgmin_method_t::RGMIN_POLAK_RIBIERE,
            (Conjugacy::PolakRibierePlus, _) => rgmin_method_t::RGMIN_POLAK_RIBIERE_PLUS,
            (Conjugacy::HestenesStiefel, _) => rgmin_method_t::RGMIN_HESTENES_STIEFEL,
            (Conjugacy::DaiYuan, _) => rgmin_method_t::RGMIN_DAI_YUAN,
            (Conjugacy::ConjugateDescent, _) => rgmin_method_t::RGMIN_CONJUGATE_DESCENT,
            (Conjugacy::HagerZhang, _) => rgmin_method_t::RGMIN_HAGER_ZHANG,
            (Conjugacy::LiuStorey, _) => rgmin_method_t::RGMIN_LIU_STOREY,
            (Conjugacy::FrPr, _) => rgmin_method_t::RGMIN_FR_PR,
            (Conjugacy::Hybrid { .. }, _) => return Err(()),
        },
        Method::Bfgs => rgmin_method_t::RGMIN_BFGS,
        Method::Lbfgs { .. } => rgmin_method_t::RGMIN_LBFGS,
        Method::Sr1 => rgmin_method_t::RGMIN_SR1,
        Method::Adam { .. } => rgmin_method_t::RGMIN_ADAM,
        Method::Steepest => rgmin_method_t::RGMIN_STEEPEST,
        Method::Sr2 => rgmin_method_t::RGMIN_SR2,
        Method::Pso { .. } => rgmin_method_t::RGMIN_PSO,
        Method::Newton { kind } => match kind {
            NewtonKind::Shifted => rgmin_method_t::RGMIN_NEWTON,
            NewtonKind::Rfo => rgmin_method_t::RGMIN_RFO,
        },
        Method::Fire { kind } => match kind {
            crate::FireKind::V1 => rgmin_method_t::RGMIN_FIRE,
            crate::FireKind::V2 => rgmin_method_t::RGMIN_FIRE2,
        },
        Method::Bb => rgmin_method_t::RGMIN_BB,
        Method::Dogleg => rgmin_method_t::RGMIN_DOGLEG,
        Method::QuickMin => rgmin_method_t::RGMIN_QUICKMIN,
        Method::Ode12r => rgmin_method_t::RGMIN_ODE12R,
        Method::Diis { .. } => rgmin_method_t::RGMIN_DIIS,
        Method::NewtonCg => rgmin_method_t::RGMIN_NEWTON_CG,
        Method::NewtonPcg => rgmin_method_t::RGMIN_NEWTON_PCG,
    };
    Ok(tag)
}

/// Parse a method token into [`rgmin_method_t`]. Case is ignored.
/// Hyphens and spaces match underscores, so `newton-cg` and `newton_cg`
/// are the same method.
///
/// # Safety
/// `name` is a NUL-terminated UTF-8 string. `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_method_from_name(
    name: *const c_char,
    out: *mut rgmin_method_t,
) -> rgmin_status_t {
    if name.is_null() || out.is_null() {
        set_last_error("rgmin_method_from_name: null");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let text = unsafe { CStr::from_ptr(name) };
    let text = match text.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("rgmin_method_from_name: name is not utf-8");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
    };
    match Method::from_name(text) {
        Ok(method) => match method_to_c(&method) {
            Ok(tag) => {
                unsafe { *out = tag };
                rgmin_status_t::RGMIN_SUCCESS
            }
            Err(()) => {
                set_last_error("rgmin_method_from_name: method has no C token");
                rgmin_status_t::RGMIN_INVALID_PARAMETER
            }
        },
        Err(e) => status_from_error(&e),
    }
}

/// Borrow a 1-D CPU f64 buffer as a DLPack tensor. Caller must
/// [`rgmin_tensor_free`] it. The buffer must outlive the tensor.
///
/// # Safety
/// `data` points to `len` writable f64s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_tensor_borrow_cpu_f64(
    data: *mut f64,
    len: usize,
) -> *mut DLManagedTensorVersioned {
    if data.is_null() || len == 0 {
        set_last_error("rgmin_tensor_borrow_cpu_f64: null or empty");
        return std::ptr::null_mut();
    }
    unsafe { create_borrowed_f64_1d(data, len, DLDeviceType::kDLCPU, 0) }
}

/// Release a tensor created by [`rgmin_tensor_borrow_cpu_f64`].
///
/// # Safety
/// `tensor` is null or a pointer from this crate's borrow helper.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_tensor_free(tensor: *mut DLManagedTensorVersioned) {
    if tensor.is_null() {
        return;
    }
    if let Some(deleter) = unsafe { (*tensor).deleter } {
        unsafe { deleter(tensor) };
    }
}

fn eigen_params_from_c(raw: *const rgmin_eigen_params_t) -> Result<EigenParams, rgmin_status_t> {
    if raw.is_null() {
        return Ok(EigenParams::default());
    }
    let p = unsafe { *raw };
    let kind = u8::try_from(p.kind)
        .ok()
        .and_then(EigensolverKind::from_ordinal)
        .ok_or_else(|| {
            set_last_error(&format!("rgmin_eigen_kind_t unknown ordinal {}", p.kind));
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        })?;
    Ok(EigenParams {
        kind,
        nev: p.nev as usize,
        krylov: p.krylov as usize,
        max_iter: p.max_iter as usize,
        tol: p.tol,
        ..EigenParams::default()
    })
}

struct CHvp {
    hvp: rgmin_hvp_fn,
    user: *mut c_void,
}

impl ApplyHessian for CHvp {
    fn apply_hessian(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let n = x.len();
        let mut xbuf = x.to_owned();
        let mut vbuf = v.to_owned();
        let mut hv = Array1::zeros(n);
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(xbuf.as_mut_ptr(), n) };
        let vt = unsafe { rgmin_tensor_borrow_cpu_f64(vbuf.as_mut_ptr(), n) };
        let ht = unsafe { rgmin_tensor_borrow_cpu_f64(hv.as_mut_ptr(), n) };
        let st = unsafe { (self.hvp)(self.user, xt, vt, ht) };
        unsafe {
            rgmin_tensor_free(xt);
            rgmin_tensor_free(vt);
            rgmin_tensor_free(ht);
        }
        if st != rgmin_status_t::RGMIN_SUCCESS {
            return Array1::from_elem(n, f64::NAN);
        }
        hv
    }
}

struct CPrecond<'a> {
    precond: rgmin_precond_fn,
    user: *mut c_void,
    failed: &'a Cell<bool>,
}

impl ApplyBlockPreconditioner for CPrecond<'_> {
    fn apply_block_preconditioner(&self, block: ArrayView2<f64>) -> Array2<f64> {
        let (n, m) = block.dim();
        let mut input = block.to_owned();
        let mut output = Array2::zeros((n, m));
        let (Some(input_slice), Some(output_slice)) = (input.as_slice_mut(), output.as_slice_mut())
        else {
            self.failed.set(true);
            return Array2::from_elem((n, m), f64::NAN);
        };
        let inn = unsafe { borrow_f64_matrix(input_slice.as_mut_ptr(), n, m) };
        let out = unsafe { borrow_f64_matrix(output_slice.as_mut_ptr(), n, m) };
        let status = unsafe { (self.precond)(self.user, inn, out) };
        unsafe {
            rgmin_tensor_free(inn);
            rgmin_tensor_free(out);
        }
        if status != rgmin_status_t::RGMIN_SUCCESS {
            self.failed.set(true);
            return Array2::from_elem((n, m), f64::NAN);
        }
        output
    }
}

/// Matrix-free lowest Hessian eigenpair. `params == NULL` is Lanczos.
/// Unlinked kinds return [`rgmin_status_t::RGMIN_UNAVAILABLE`].
///
/// # Safety
/// `hvp` is callable for the lifetime of this call. `x`, `seed`, and
/// `mode_out` are rank-1 f64 CPU tensors of equal length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_lowest_eigenpair(
    hvp: Option<rgmin_hvp_fn>,
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    seed: *const DLManagedTensorVersioned,
    mode_out: *mut DLManagedTensorVersioned,
    params: *const rgmin_eigen_params_t,
    out: *mut rgmin_lowest_mode_t,
) -> rgmin_status_t {
    unsafe {
        rgmin_lowest_eigenpair_with_options(
            hvp,
            user,
            x,
            seed,
            mode_out,
            params,
            std::ptr::null(),
            out,
        )
    }
}

unsafe fn create_borrowed_f64_1d(
    data: *mut f64,
    len: usize,
    device_type: DLDeviceType,
    device_id: i32,
) -> *mut DLManagedTensorVersioned {
    struct Ctx {
        shape: [i64; 1],
        strides: [i64; 1],
    }

    unsafe extern "C" fn deleter(ptr: *mut DLManagedTensorVersioned) {
        if ptr.is_null() {
            return;
        }
        let ctx = unsafe { (*ptr).manager_ctx.cast::<Ctx>() };
        if !ctx.is_null() {
            drop(unsafe { Box::from_raw(ctx) });
        }
        drop(unsafe { Box::from_raw(ptr) });
    }

    let mut ctx = Box::new(Ctx {
        shape: [len as i64],
        strides: [1],
    });
    let dl_tensor = DLTensor {
        data: data.cast(),
        device: DLDevice {
            device_type,
            device_id,
        },
        ndim: 1,
        dtype: DLDataType {
            code: DLDataTypeCode::kDLFloat,
            bits: 64,
            lanes: 1,
        },
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    };
    let managed = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: Box::into_raw(ctx).cast(),
        deleter: Some(deleter),
        flags: 0,
        dl_tensor,
    });
    Box::into_raw(managed)
}

/// A reusable DLPack shell for handing borrowed CPU f64 buffers across
/// the callback boundary without per-call allocation. Shape and strides
/// live beside the managed struct behind one `Box`, so the tensor's
/// self-referential pointers stay valid for the shell's lifetime; only
/// the data pointer and length change between calls. `deleter` is
/// `None` because the callee never owns the tensor.
struct StandingShell(Box<ShellInner>);

struct ShellInner {
    managed: DLManagedTensorVersioned,
    shape: [i64; 1],
    strides: [i64; 1],
}

// SAFETY: the raw pointers inside the shell reference only the shell's
// own boxed fields and, transiently, the buffer handed to `point_at`
// for one callback under the owning Mutex.
unsafe impl Send for StandingShell {}

impl StandingShell {
    fn new() -> Self {
        let mut inner = Box::new(ShellInner {
            managed: DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
                flags: 0,
                dl_tensor: DLTensor {
                    data: std::ptr::null_mut(),
                    device: DLDevice {
                        device_type: DLDeviceType::kDLCPU,
                        device_id: 0,
                    },
                    ndim: 1,
                    dtype: DLDataType {
                        code: DLDataTypeCode::kDLFloat,
                        bits: 64,
                        lanes: 1,
                    },
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
            },
            shape: [0],
            strides: [1],
        });
        inner.managed.dl_tensor.shape = inner.shape.as_mut_ptr();
        inner.managed.dl_tensor.strides = inner.strides.as_mut_ptr();
        StandingShell(inner)
    }

    fn point_at(&mut self, data: *mut f64, len: usize) -> *mut DLManagedTensorVersioned {
        self.0.shape[0] = len as i64;
        self.0.managed.dl_tensor.data = data.cast();
        &mut self.0.managed
    }
}

/// Per-oracle standing storage: one shell for the iterate, one for the
/// output tensor, and a copy buffer for the rare non-contiguous view.
struct Scratch {
    x: StandingShell,
    out: StandingShell,
    xbuf: Vec<f64>,
}

impl Scratch {
    fn new() -> std::sync::Mutex<Self> {
        std::sync::Mutex::new(Scratch {
            x: StandingShell::new(),
            out: StandingShell::new(),
            xbuf: Vec::new(),
        })
    }

    /// The iterate tensor: pointed straight at contiguous view storage
    /// (the callback side receives it const), through the standing
    /// copy buffer otherwise.
    fn x_tensor(&mut self, xv: ndarray::ArrayView1<f64>) -> *mut DLManagedTensorVersioned {
        match xv.as_slice() {
            Some(s) => self.x.point_at(s.as_ptr() as *mut f64, s.len()),
            None => {
                self.xbuf.clear();
                self.xbuf.extend(xv.iter());
                self.x.point_at(self.xbuf.as_mut_ptr(), self.xbuf.len())
            }
        }
    }
}

struct ScgFfiOracle<F>
where
    F: Fn(ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    inner: Oracle<F>,
    curv: Option<usize>,
    user: usize,
    scratch: std::sync::Mutex<Scratch>,
}

impl<F> Objective<f64> for ScgFfiOracle<F>
where
    F: Fn(ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }
    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }
    fn eval(&self, x: ndarray::ArrayView1<f64>) -> f64 {
        self.inner.eval(x)
    }
}

impl<F> Gradient<f64> for ScgFfiOracle<F>
where
    F: Fn(ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    fn dim(&self) -> usize {
        Gradient::dim(&self.inner)
    }
    fn grad(&self, x: ndarray::ArrayView1<f64>) -> Array1<f64> {
        self.inner.grad(x)
    }
}

impl<F> DifferentiableObjective<f64> for ScgFfiOracle<F>
where
    F: Fn(ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    fn value_and_gradient(&self, x: ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.inner.value_and_gradient(x)
    }
}

impl<F> DirectionalCurvature for ScgFfiOracle<F>
where
    F: Fn(ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    fn directional_curvature(
        &self,
        x: ndarray::ArrayView1<f64>,
        d: ndarray::ArrayView1<f64>,
    ) -> Option<f64> {
        let curv_ptr = self.curv?;
        let curv_fn: rgmin_curv_fn = unsafe { std::mem::transmute(curv_ptr) };
        let user = self.user as *mut c_void;
        let mut s = self.scratch.lock().expect("ffi scratch");
        let xt = s.x_tensor(x);
        let dt = match d.as_slice() {
            Some(sl) => s.out.point_at(sl.as_ptr() as *mut f64, sl.len()),
            None => {
                // rare non-contiguous direction: reuse xbuf then retarget out
                let mut tmp = d.to_owned();
                let p = tmp.as_mut_ptr();
                let n = tmp.len();
                let dt = s.out.point_at(p, n);
                let mut curv = 0.0;
                let st = unsafe { curv_fn(user, xt, dt, &mut curv) };
                return (st == rgmin_status_t::RGMIN_SUCCESS).then_some(curv);
            }
        };
        let mut curv = 0.0;
        let st = unsafe { curv_fn(user, xt, dt, &mut curv) };
        (st == rgmin_status_t::RGMIN_SUCCESS).then_some(curv)
    }
}

fn cpu_f64_slice<'a>(
    t: *const DLManagedTensorVersioned,
    name: &str,
) -> Result<&'a [f64], rgmin_status_t> {
    if t.is_null() {
        set_last_error(&format!("{name}: null tensor"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    let t = unsafe { &*t };
    let dl = &t.dl_tensor;
    if dl.device.device_type != DLDeviceType::kDLCPU {
        set_last_error(&format!(
            "{name}: device {:?} not supported in this build (CPU only)",
            dl.device.device_type as i32
        ));
        return Err(rgmin_status_t::RGMIN_UNSUPPORTED_DEVICE);
    }
    if dl.ndim != 1
        || dl.dtype.code != DLDataTypeCode::kDLFloat
        || dl.dtype.bits != 64
        || dl.dtype.lanes != 1
        || dl.shape.is_null()
        || dl.data.is_null()
    {
        set_last_error(&format!("{name}: need rank-1 f64 contiguous DLPack"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    let n = unsafe { *dl.shape as usize };
    if n == 0 {
        set_last_error(&format!("{name}: empty"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    if !dl.strides.is_null() && unsafe { *dl.strides } != 1 {
        set_last_error(&format!("{name}: non-unit stride"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    let ptr = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
    Ok(unsafe { slice::from_raw_parts(ptr, n) })
}

fn cpu_f64_slice_mut<'a>(
    t: *mut DLManagedTensorVersioned,
    name: &str,
) -> Result<&'a mut [f64], rgmin_status_t> {
    let s = cpu_f64_slice(t as *const _, name)?;
    let n = s.len();
    let p = s.as_ptr() as *mut f64;
    Ok(unsafe { slice::from_raw_parts_mut(p, n) })
}

/// Minimize `f` from the 1-D f64 DLPack iterate `x`. On success, `x` is
/// overwritten with the accepted point.
///
/// # Safety
///
/// `eval` and `grad` must be callable for the lifetime of this call.
/// `x` must be a writable rank-1 f64 tensor. Non-CPU devices return
/// [`rgmin_status_t::RGMIN_UNSUPPORTED_DEVICE`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_minimize(
    eval: Option<rgmin_eval_fn>,
    grad: Option<rgmin_grad_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    ctrl: *const rgmin_control_t,
    method: rgmin_method_t,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eval = match eval {
            Some(f) => f,
            None => {
                set_last_error("rgmin_minimize: eval is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let grad = match grad {
            Some(f) => f,
            None => {
                set_last_error("rgmin_minimize: grad is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        if matches!(
            method,
            rgmin_method_t::RGMIN_NEWTON | rgmin_method_t::RGMIN_RFO | rgmin_method_t::RGMIN_DOGLEG
        ) {
            set_last_error("rgmin_minimize: Newton/RFO/dogleg needs rgmin_minimize_hess");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        if ctrl.is_null() || out.is_null() {
            set_last_error("rgmin_minimize: ctrl/out null");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let n = init.len();
        let obj = c_oracle(eval, grad, user, n);
        let c = unsafe { &*ctrl };
        let control = control_from_c(c);
        match minimize_method(
            &obj,
            Array1::from(init),
            &control,
            method_from_c(method, c.memory),
            LineSearch::default(),
        ) {
            Ok(rep) => {
                let dest = match cpu_f64_slice_mut(x, "x") {
                    Ok(s) => s,
                    Err(st) => return st,
                };
                dest.copy_from_slice(rep.coords.as_slice().expect("contiguous"));
                unsafe {
                    *out = rgmin_report_t {
                        value: rep.value,
                        steps: rep.steps,
                        grad_norm: rep.grad_norm,
                        stop: stop_from_status(rep.status),
                    };
                }
                rgmin_status_t::RGMIN_SUCCESS
            }
            Err(e) => status_from_error(&e),
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("rgmin_minimize: panic");
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

/// Møller SCG. `curv` is optional: null uses the finite-difference probe;
/// a callback that returns success supplies `dᵀ ∇²f(x) d`.
///
/// # Safety
///
/// Same as [`rgmin_minimize`]. `curv`, when non-null, must be callable
/// for the lifetime of this call. `params` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_minimize_scg(
    eval: Option<rgmin_eval_fn>,
    grad: Option<rgmin_grad_fn>,
    curv: Option<rgmin_curv_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    ctrl: *const rgmin_control_t,
    params: *const rgmin_scg_params_t,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    unsafe {
        rgmin_minimize_scg_with_options(
            eval,
            grad,
            curv,
            user,
            x,
            ctrl,
            params,
            std::ptr::null(),
            out,
        )
    }
}

/// Minimize with a Newton / RFO direction. `hess` writes a length-`n²`
/// row-major Hessian at the current `x`.
///
/// # Safety
///
/// Same as [`rgmin_minimize`]. `hess` must be callable for the lifetime
/// of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_minimize_hess(
    eval: Option<rgmin_eval_fn>,
    grad: Option<rgmin_grad_fn>,
    hess: Option<rgmin_hess_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    ctrl: *const rgmin_control_t,
    method: rgmin_method_t,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eval = match eval {
            Some(f) => f,
            None => {
                set_last_error("rgmin_minimize_hess: eval is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let grad = match grad {
            Some(f) => f,
            None => {
                set_last_error("rgmin_minimize_hess: grad is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let hess = match hess {
            Some(f) => f,
            None => {
                set_last_error("rgmin_minimize_hess: hess is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        if ctrl.is_null() || out.is_null() {
            set_last_error("rgmin_minimize_hess: ctrl/out null");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let n = init.len();
        let obj = c_oracle_hess(eval, grad, hess, user, n);
        let c = unsafe { &*ctrl };
        let control = control_from_c(c);
        let rust_method = method_from_c(method, c.memory);
        match minimize_method_hess(
            &obj,
            Array1::from(init),
            &control,
            rust_method,
            LineSearch::default(),
        ) {
            Ok(rep) => {
                let dest = match cpu_f64_slice_mut(x, "x") {
                    Ok(s) => s,
                    Err(st) => return st,
                };
                dest.copy_from_slice(rep.coords.as_slice().expect("contiguous"));
                unsafe {
                    *out = rgmin_report_t {
                        value: rep.value,
                        steps: rep.steps,
                        grad_norm: rep.grad_norm,
                        stop: stop_from_status(rep.status),
                    };
                }
                rgmin_status_t::RGMIN_SUCCESS
            }
            Err(e) => status_from_error(&e),
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("rgmin_minimize_hess: panic");
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

/// Steihaug-Toint Newton-CG from an analytic Hessian-vector product.
/// `method` is [`rgmin_method_t::RGMIN_NEWTON_CG`] or
/// [`rgmin_method_t::RGMIN_NEWTON_PCG`]. The product is never expanded
/// into a dense Hessian.
///
/// # Safety
/// Same as [`rgmin_minimize`]. `hvp` must be callable for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_minimize_hvp(
    eval: Option<rgmin_eval_fn>,
    grad: Option<rgmin_grad_fn>,
    hvp: Option<rgmin_hvp_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    ctrl: *const rgmin_control_t,
    method: rgmin_method_t,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eval = match eval {
            Some(f) => f,
            None => {
                set_last_error("rgmin_minimize_hvp: eval is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let grad = match grad {
            Some(f) => f,
            None => {
                set_last_error("rgmin_minimize_hvp: grad is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let hvp = match hvp {
            Some(f) => f,
            None => {
                set_last_error("rgmin_minimize_hvp: hvp is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        if ctrl.is_null() || out.is_null() {
            set_last_error("rgmin_minimize_hvp: ctrl/out null");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let n = init.len();
        let obj = c_oracle_hvp(eval, grad, hvp, user, n);
        let c = unsafe { &*ctrl };
        let control = control_from_c(c);
        match minimize_method_hvp(
            &obj,
            Array1::from(init),
            &control,
            method_from_c(method, c.memory),
            LineSearch::default(),
        ) {
            Ok(rep) => {
                let dest = match cpu_f64_slice_mut(x, "x") {
                    Ok(s) => s,
                    Err(st) => return st,
                };
                dest.copy_from_slice(rep.coords.as_slice().expect("contiguous"));
                unsafe {
                    *out = rgmin_report_t {
                        value: rep.value,
                        steps: rep.steps,
                        grad_norm: rep.grad_norm,
                        stop: stop_from_status(rep.status),
                    };
                }
                rgmin_status_t::RGMIN_SUCCESS
            }
            Err(e) => status_from_error(&e),
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("rgmin_minimize_hvp: panic");
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

/// Minimize an eindir-compatible objective without taking ownership of it.
///
/// The objective must provide an analytic gradient and the producer's ABI
/// stamp must be compatible with this build. The objective remains owned by
/// the caller for the full duration of the call.
///
/// # Safety
///
/// `objective`, `stamp`, `x`, `ctrl`, and `out` must be valid for the duration
/// of this call. `objective` and `stamp` must come from the same compatible
/// eindir ABI family. `x` must be a writable rank-1 CPU f64 tensor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_minimize_eindir(
    objective: *const eindir_objective_t,
    stamp: *const eindir_abi_stamp_t,
    x: *mut DLManagedTensorVersioned,
    ctrl: *const rgmin_control_t,
    method: rgmin_method_t,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if objective.is_null() || stamp.is_null() || ctrl.is_null() || out.is_null() {
            set_last_error("rgmin_minimize_eindir: null argument");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        if unsafe { eindir_core_abi_compatible(stamp) } == 0 {
            set_last_error("rgmin_minimize_eindir: incompatible eindir ABI stamp");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        if unsafe { eindir_objective_has_grad(objective) } == 0 {
            set_last_error("rgmin_minimize_eindir: objective has no gradient");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let dim = unsafe { (*objective).dim };
        if dim != init.len() {
            set_last_error(&format!(
                "rgmin_minimize_eindir: x length {} != objective dim {}",
                init.len(),
                dim
            ));
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let objective_addr = objective as usize;
        let scratch = Scratch::new();
        let obj = Oracle::unbounded(dim, move |xv| {
            let objective = objective_addr as *const eindir_objective_t;
            let m = xv.len();
            let mut s = scratch.lock().expect("ffi scratch");
            let xt = s.x_tensor(xv);
            let mut value = 0.0;
            let eval_status = unsafe { eindir_objective_eval(objective, xt, &mut value) };
            let mut gradient = Array1::zeros(m);
            let gt = s.out.point_at(gradient.as_mut_ptr(), m);
            let grad_status = unsafe { eindir_objective_grad(objective, xt, gt) };
            if eval_status != eindir_status_t::EINDIR_SUCCESS
                || grad_status != eindir_status_t::EINDIR_SUCCESS
            {
                set_last_error("rgmin_minimize_eindir: eindir evaluation failed");
                return (f64::INFINITY, Array1::from_elem(m, f64::NAN));
            }
            (value, gradient)
        });
        let c = unsafe { &*ctrl };
        let control = control_from_c(c);
        match minimize_method(
            &obj,
            Array1::from(init),
            &control,
            method_from_c(method, c.memory),
            LineSearch::default(),
        ) {
            Ok(rep) => {
                let dest = match cpu_f64_slice_mut(x, "x") {
                    Ok(s) => s,
                    Err(st) => return st,
                };
                dest.copy_from_slice(rep.coords.as_slice().expect("contiguous"));
                unsafe {
                    *out = rgmin_report_t {
                        value: rep.value,
                        steps: rep.steps,
                        grad_norm: rep.grad_norm,
                        stop: stop_from_status(rep.status),
                    };
                }
                rgmin_status_t::RGMIN_SUCCESS
            }
            Err(e) => status_from_error(&e),
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("rgmin_minimize_eindir: panic");
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

/// Opaque session. Algorithm memory lives here; `x` stays a DLPack tensor.
pub struct rgmin_solver_t {
    solver: Solver,
    /// Gradient behind the last successful step report.
    grad: Option<Array1<f64>>,
}

/// Allocate a session. `dim` is the length of `x`. Null on bad arguments.
///
/// Since 0.6.0 the session opens with [`Accept::Energy`]
/// (`RGMIN_ACCEPT_ENERGY`), where 0.5 opened with `RGMIN_ACCEPT_NONE`.
/// The default L-BFGS step is the eOn cap, which runs no energy test
/// under either. Line searches take the same value window under both.
/// Under the energy policy a step that is not line-searched backtracks on
/// a rise, a non-positive `istep` opens steepest descent, unpreconditioned
/// nonlinear CG, BFGS, SR1, SR2 and BB with a unit-length step, and BFGS,
/// SR1 and SR2 rescale their first step.
/// [`rgmin_solver_set_accept`] with `RGMIN_ACCEPT_NONE` restores the 0.5
/// behaviour.
///
/// # Safety
/// `ctrl` is null or points to a readable [`rgmin_control_t`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_create(
    method: rgmin_method_t,
    ctrl: *const rgmin_control_t,
    dim: usize,
) -> *mut rgmin_solver_t {
    if ctrl.is_null() || dim == 0 {
        set_last_error("rgmin_solver_create: null ctrl or dim=0");
        return std::ptr::null_mut();
    }
    if matches!(
        method,
        rgmin_method_t::RGMIN_NEWTON | rgmin_method_t::RGMIN_RFO
    ) {
        // Allowed: step_hess is the verb. Create still succeeds.
    }
    let c = unsafe { &*ctrl };
    let control = control_from_c(c);
    let mut solver = Solver::new(method_from_c(method, c.memory), control, dim).with_gtol(c.gtol);
    solver.set_accept(Accept::Energy);
    if c.istep <= 0.0 {
        solver.set_unit_first_step();
    }
    Box::into_raw(Box::new(rgmin_solver_t { solver, grad: None }))
}

/// Release a session from [`rgmin_solver_create`].
///
/// # Safety
/// `solver` is null or a session from [`rgmin_solver_create`] that is not
/// yet freed. The handle is dangling after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_free(solver: *mut rgmin_solver_t) {
    if !solver.is_null() {
        drop(unsafe { Box::from_raw(solver) });
    }
}

/// Drop method memory. The next step is a cold start from the current `x`.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_forget(solver: *mut rgmin_solver_t) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.forget() };
}

/// Keep the method memory, drop the point: the next step evaluates the
/// objective afresh at the current `x` (initial step scale, empty
/// acceptance window) and the retained curvature preconditions it. The
/// restart for a sequence of related objectives.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_rebase(solver: *mut rgmin_solver_t) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.rebase() };
}

/// Set the Euclidean step cap used by the next [`rgmin_solver_step`].
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_maxmove(solver: *mut rgmin_solver_t, maxmove: f64) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_maxmove(maxmove) };
}

/// How an L-BFGS session uses a caller Hessian (eOn `lbfgs_step`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_qn_step_t {
    /// Two-loop. A supplied Hessian is \(H_0 = P^{-1}\).
    RGMIN_QN_LBFGS = 0,
    /// Regularized Newton on the caller Hessian.
    RGMIN_QN_NEWTON = 1,
    /// Banerjee RFO on the caller Hessian.
    RGMIN_QN_RFO = 2,
}

/// eOn `lbfgs_step`. Legal on an `RGMIN_LBFGS` session with [`rgmin_solver_step_hess`].
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_qn_step(
    solver: *mut rgmin_solver_t,
    step: rgmin_qn_step_t,
) {
    if solver.is_null() {
        return;
    }
    let qn = match step {
        rgmin_qn_step_t::RGMIN_QN_NEWTON => QnStep::Newton,
        rgmin_qn_step_t::RGMIN_QN_RFO => QnStep::Rfo,
        rgmin_qn_step_t::RGMIN_QN_LBFGS => QnStep::TwoLoop,
    };
    unsafe { (*solver).solver.set_qn_step(qn) };
}

/// Parameters of the exponential preconditioner. ABI minor 28.
///
/// `mu` and `r_nn` are NaN when the session should compute them.
/// `a` and `c_stab` are NaN for the paper defaults (3 and 0.1).
/// `length_*` are 0 to use the span of the positions.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rgmin_exp_t {
    /// Exponent A. NaN selects 3.
    pub a: f64,
    /// Diagonal stabilization. NaN selects 0.1. Must be positive otherwise.
    pub c_stab: f64,
    /// Energy scale. NaN estimates it on the next step.
    pub mu: f64,
    /// Nearest-neighbour distance. NaN or nonpositive selects the paper's rule.
    pub r_nn: f64,
    /// Cutoff. Nonpositive keeps every supplied pair.
    pub r_cut: f64,
    /// Lattice length along x. Nonpositive selects the position span.
    pub length_x: f64,
    /// Lattice length along y.
    pub length_y: f64,
    /// Lattice length along z.
    pub length_z: f64,
    /// Nonzero reestimates μ whenever the matrix is rebuilt.
    pub recalc_mu: i32,
}

/// Install the exponential preconditioner on a session.
///
/// `pair_i` and `pair_j` have length `npair` and index atoms in the
/// session vector (`dim = 3 N`, C order). `distance` is null to take
/// Cartesian distances from the point on each step, or length
/// `npair` to keep those distances. `anchor_i` and `anchor_r` are bonds
/// to atoms with no coordinate in the vector; both are null when
/// `nanchor` is 0.
///
/// # Safety
///
/// Every pointer argument must be null or valid for the type and
/// length documented above; a session handle must come from
/// [`rgmin_solver_create`] and not yet be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_exp(
    solver: *mut rgmin_solver_t,
    params: *const rgmin_exp_t,
    pair_i: *const u32,
    pair_j: *const u32,
    distance: *const f64,
    npair: usize,
    anchor_i: *const u32,
    anchor_r: *const f64,
    nanchor: usize,
) -> rgmin_status_t {
    if solver.is_null() || params.is_null() {
        set_last_error("rgmin_solver_set_exp: null solver or params");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    if npair > 0 && (pair_i.is_null() || pair_j.is_null()) {
        set_last_error("rgmin_solver_set_exp: null pair list");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    if nanchor > 0 && (anchor_i.is_null() || anchor_r.is_null()) {
        set_last_error("rgmin_solver_set_exp: null anchor list");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let p = unsafe { &*params };
    let dim = unsafe { (*solver).solver.dim() };
    if dim % 3 != 0 {
        set_last_error("rgmin_solver_set_exp: dimension is not a multiple of 3");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let i = if npair == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(pair_i, npair) }
    };
    let j = if npair == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(pair_j, npair) }
    };
    let r = if distance.is_null() {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(distance, npair) }.to_vec()
    };
    let mut anchor = Vec::with_capacity(nanchor);
    if nanchor > 0 {
        let ai = unsafe { slice::from_raw_parts(anchor_i, nanchor) };
        let ar = unsafe { slice::from_raw_parts(anchor_r, nanchor) };
        for k in 0..nanchor {
            anchor.push((ai[k] as usize, ar[k]));
        }
    }
    let neighbours = Neighbours {
        i: i.iter().map(|&v| v as usize).collect(),
        j: j.iter().map(|&v| v as usize).collect(),
        r,
        anchor,
    };
    let finite_pos = |v: f64| v.is_finite() && v > 0.0;
    let lengths = if finite_pos(p.length_x) || finite_pos(p.length_y) || finite_pos(p.length_z) {
        Some([p.length_x, p.length_y, p.length_z])
    } else {
        None
    };
    let exp_params = ExpParams {
        a: if p.a.is_finite() { p.a } else { 3.0 },
        c_stab: if p.c_stab.is_finite() { p.c_stab } else { 0.1 },
        mu: if p.mu.is_finite() { Some(p.mu) } else { None },
        r_nn: if finite_pos(p.r_nn) {
            Some(p.r_nn)
        } else {
            None
        },
        r_cut: if finite_pos(p.r_cut) {
            Some(p.r_cut)
        } else {
            None
        },
        recalc_mu: p.recalc_mu != 0,
        lengths,
    };
    // Supplied distances do not need the geometry. A Cartesian
    // list is placed on the next step, when the session holds `x`.
    if neighbours.r.is_empty() {
        match ExpState::deferred(neighbours, exp_params) {
            Ok(state) => {
                unsafe { (*solver).solver.set_exp_deferred(state) };
                rgmin_status_t::RGMIN_SUCCESS
            }
            Err(e) => {
                set_last_error(&e.to_string());
                rgmin_status_t::RGMIN_INVALID_PARAMETER
            }
        }
    } else {
        let positions = vec![0.0; dim];
        match unsafe {
            (*solver).solver.set_exp(
                ndarray::ArrayView1::from(positions.as_slice()),
                &neighbours,
                &exp_params,
            )
        } {
            Ok(()) => rgmin_status_t::RGMIN_SUCCESS,
            Err(e) => {
                set_last_error(&e.to_string());
                rgmin_status_t::RGMIN_INVALID_PARAMETER
            }
        }
    }
}

/// Drop the exponential preconditioner installed by [`rgmin_solver_set_exp`].
///
/// # Safety
///
/// Every pointer argument must be null or valid for the type and
/// length documented above; a session handle must come from
/// [`rgmin_solver_create`] and not yet be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_clear_exp(solver: *mut rgmin_solver_t) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.clear_exp() };
}

/// Energy scale of the installed exponential preconditioner, or NaN.
///
/// # Safety
///
/// Every pointer argument must be null or valid for the type and
/// length documented above; a session handle must come from
/// [`rgmin_solver_create`] and not yet be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_exp_mu(solver: *const rgmin_solver_t) -> f64 {
    if solver.is_null() {
        return f64::NAN;
    }
    unsafe { (*solver).solver.exp_mu().unwrap_or(f64::NAN) }
}

/// How a session takes a proposed step (eOn `lbfgs_accept`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_accept_t {
    /// Take the maxmove-clipped step.
    RGMIN_ACCEPT_NONE = 0,
    /// Refuse an energy rise (up to 10 halvings).
    RGMIN_ACCEPT_ENERGY = 1,
    /// Grippo window of the last five accepted values.
    RGMIN_ACCEPT_NONMONOTONE = 2,
    /// Clipped quasi-Newton step, one oracle, no line search. Released 0.5.2.
    RGMIN_ACCEPT_STEP = 3,
}

/// eOn `lbfgs_accept`. Legal on any session.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_accept(
    solver: *mut rgmin_solver_t,
    accept: rgmin_accept_t,
) {
    if solver.is_null() {
        return;
    }
    let a = match accept {
        rgmin_accept_t::RGMIN_ACCEPT_ENERGY => Accept::Energy,
        rgmin_accept_t::RGMIN_ACCEPT_NONMONOTONE => Accept::Nonmonotone,
        rgmin_accept_t::RGMIN_ACCEPT_NONE => Accept::None,
        rgmin_accept_t::RGMIN_ACCEPT_STEP => Accept::Step,
    };
    unsafe { (*solver).solver.set_accept(a) };
}

/// Gradient norm a session compares to `gtol` and writes to
/// [`rgmin_report_t::grad_norm`]. Integers match [`StopNorm`].
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_stop_norm_t {
    /// `||g||_2` (default).
    RGMIN_STOP_NORM_TWO = 0,
    /// Largest per-atom norm: ASE `fmax`, eOn `max_atom`, LAMMPS `max`.
    RGMIN_STOP_NORM_MAX_ATOM = 1,
    /// Largest absolute component: eOn `max_component`, LAMMPS `inf`.
    RGMIN_STOP_NORM_INF = 2,
    /// `||g||_2 / sqrt(n)`: eOn `rms`. ABI minor 32.
    RGMIN_STOP_NORM_RMS = 3,
}

/// Select the session stop norm. ABI minor 29. Returns 0, or 1 on a
/// null session or an unknown integer.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_stop_norm(solver: *mut rgmin_solver_t, norm: i32) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_stop_norm: null solver");
        return 1;
    }
    let norm = match norm {
        0 => StopNorm::Two,
        1 => StopNorm::MaxAtom,
        2 => StopNorm::Inf,
        3 => StopNorm::Rms,
        _ => {
            set_last_error("rgmin_solver_set_stop_norm: unknown norm");
            return 1;
        }
    };
    unsafe { (*solver).solver.set_stop_norm(norm) };
    0
}

/// FIRE 2.0 parameter set. Integers match [`crate::FirePreset`].
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_fire_preset_t {
    /// What a new `RGMIN_FIRE2` session runs.
    RGMIN_FIRE_PRESET_DEFAULT = 0,
    /// Guénolé 2020 Table 1.
    RGMIN_FIRE_PRESET_GUENOLE2020 = 1,
}

/// Select the FIRE 2.0 parameter set and restart the integrator.
/// ABI minor 30. Returns 0, or 1 on a null session, an unknown integer
/// or a session that is not `RGMIN_FIRE2`.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_fire_preset(
    solver: *mut rgmin_solver_t,
    preset: i32,
) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_fire_preset: null solver");
        return 1;
    }
    let preset = match preset {
        0 => crate::FirePreset::Default,
        1 => crate::FirePreset::Guenole2020,
        _ => {
            set_last_error("rgmin_solver_set_fire_preset: unknown preset");
            return 1;
        }
    };
    match unsafe { (*solver).solver.set_fire_preset(preset) } {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            1
        }
    }
}

/// ABC-FIRE on (`on != 0`) or off for a FIRE 2.0 session. ABI minor 30.
/// Returns 0, or 1 on a null session or a session that is not
/// `RGMIN_FIRE2`.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_fire_abc(solver: *mut rgmin_solver_t, on: i32) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_fire_abc: null solver");
        return 1;
    }
    match unsafe { (*solver).solver.set_fire_abc(on != 0) } {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            1
        }
    }
}

/// Per-atom cap of the default eOn L-BFGS step.
/// `max_atom > 0` sets that cap. A non-positive value restores
/// [`crate::EON_MAX_MOVE`]. Returns 0, or 1 on a null session or a
/// session that cannot run the capped step. ABI minor 35.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_lbfgs_capped(
    solver: *mut rgmin_solver_t,
    max_atom: f64,
) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_lbfgs_capped: null solver");
        return 1;
    }
    let step = crate::LbfgsStep::Capped {
        max_atom: if max_atom > 0.0 {
            max_atom
        } else {
            crate::EON_MAX_MOVE
        },
    };
    match unsafe { (*solver).solver.set_lbfgs_step(step) } {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            1
        }
    }
}

/// ASE length control on an L-BFGS session: shorten the two-loop step
/// and keep the pairs. `max_atom > 0` sets the cap. A non-positive
/// value restores [`crate::EON_MAX_MOVE`]. Returns 0, or 1 on a null
/// session or a session that cannot run the scaled step. ABI minor 35.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_lbfgs_scaled(
    solver: *mut rgmin_solver_t,
    max_atom: f64,
) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_lbfgs_scaled: null solver");
        return 1;
    }
    let step = crate::LbfgsStep::Scaled {
        max_atom: if max_atom > 0.0 {
            max_atom
        } else {
            crate::EON_MAX_MOVE
        },
    };
    match unsafe { (*solver).solver.set_lbfgs_step(step) } {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            1
        }
    }
}

/// Opt into the strong Wolfe line search on an L-BFGS session.
/// The default session does not search a line. Returns 0, or 1 on a
/// null session or a session that is not L-BFGS. ABI minor 34.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_lbfgs_wolfe(solver: *mut rgmin_solver_t) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_lbfgs_wolfe: null solver");
        return 1;
    }
    match unsafe { (*solver).solver.set_lbfgs_step(crate::LbfgsStep::Wolfe) } {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            1
        }
    }
}

/// Per-atom cap (eOn `maxAtomMotionAppliedV`). While the L-BFGS step is
/// the default cap, this replaces that cap. After
/// [`rgmin_solver_set_lbfgs_wolfe`], and without a distance constraint
/// set, this selects the geometry step and ignores
/// [`rgmin_solver_set_accept`]. With a set, L-BFGS keeps its
/// two-loop step under the acceptance mode and the cap scales each trial
/// before it returns to the set.
/// A nominal steepest trial at least 64 times the cap takes one
/// finite-difference curvature probe, clipped to the objective bounds.
/// Ordinary starts use `0.1 I` and a five-value nonmonotone window.
/// An installed exponential preconditioner replaces that scalar seed,
/// including the steepest fallbacks. Newton and dogleg scale a saddle
/// kick to this length. Non-positive clears it.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_atom_maxmove(solver: *mut rgmin_solver_t, maxmove: f64) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_atom_maxmove(maxmove) };
}

/// eOn `lbfgs_project_rigid`.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_project_rigid(solver: *mut rgmin_solver_t, enabled: i32) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_project_rigid(enabled != 0) };
}

/// Al-Baali extra-updates on the newest L-BFGS pair.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_extra_updates(solver: *mut rgmin_solver_t, extra: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_extra_updates(extra) };
}

/// Li-Fukushima cautious pair filter. `eps <= 0` disables it.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_cautious(
    solver: *mut rgmin_solver_t,
    eps: f64,
    alpha: f64,
) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_cautious(eps, alpha) };
}

/// HiGHS feasible-set step. Returns 1 when this build has no `highs` feature.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_highs(solver: *mut rgmin_solver_t, enabled: i32) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_highs: null solver");
        return 1;
    }
    #[cfg(feature = "highs")]
    {
        unsafe { (*solver).solver.set_highs(enabled != 0) };
        0
    }
    #[cfg(not(feature = "highs"))]
    {
        let _ = enabled;
        set_last_error("rgmin_solver_set_highs: build has no highs feature");
        1
    }
}

/// Per-coordinate box on `x + p`. A NULL side is unbounded.
/// `n` is the length of each non-NULL side and must match the session
/// dimension. Returns 0, or 1 if this build has no `highs` feature.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
/// `lower` and `upper` are each null or point to `n` readable f64s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_box(
    solver: *mut rgmin_solver_t,
    lower: *const f64,
    upper: *const f64,
    n: usize,
) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_box: null solver");
        return 1;
    }
    #[cfg(feature = "highs")]
    {
        let dim = unsafe { (*solver).solver.dim() };
        if (!lower.is_null() || !upper.is_null()) && n != dim {
            set_last_error("rgmin_solver_set_box: n does not match session dim");
            return 1;
        }
        let lo = if lower.is_null() {
            None
        } else {
            let mut dest = vec![0.0; n];
            dest.copy_from_slice(unsafe { slice::from_raw_parts(lower, n) });
            Some(dest)
        };
        let hi = if upper.is_null() {
            None
        } else {
            let mut dest = vec![0.0; n];
            dest.copy_from_slice(unsafe { slice::from_raw_parts(upper, n) });
            Some(dest)
        };
        let _ = unsafe { (*solver).solver.set_box(lo, hi) };
        0
    }
    #[cfg(not(feature = "highs"))]
    {
        let _ = (lower, upper, n);
        set_last_error("rgmin_solver_set_box: build has no highs feature");
        1
    }
}

/// L_inf trust radius on the HiGHS step. Non-positive clears it.
/// Returns 0, or 1 if this build has no `highs` feature.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_trust(solver: *mut rgmin_solver_t, radius: f64) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_trust: null solver");
        return 1;
    }
    #[cfg(feature = "highs")]
    {
        let _ = unsafe { (*solver).solver.set_trust(radius) };
        0
    }
    #[cfg(not(feature = "highs"))]
    {
        let _ = radius;
        set_last_error("rgmin_solver_set_trust: build has no highs feature");
        1
    }
}

/// Append one sparse equality `a · p = rhs`. `idx` / `coef` have length `nnz`.
/// Returns 0, or 1 if this build has no `highs` feature.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
/// `idx` and `coef` are each null or point to `nnz` readable values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_add_equality(
    solver: *mut rgmin_solver_t,
    idx: *const usize,
    coef: *const f64,
    nnz: usize,
    rhs: f64,
) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_add_equality: null solver");
        return 1;
    }
    #[cfg(feature = "highs")]
    {
        if nnz == 0 {
            return 0;
        }
        if idx.is_null() || coef.is_null() {
            set_last_error("rgmin_solver_add_equality: null idx or coef");
            return 1;
        }
        let dim = unsafe { (*solver).solver.dim() };
        let i_src = unsafe { slice::from_raw_parts(idx, nnz) };
        let a_src = unsafe { slice::from_raw_parts(coef, nnz) };
        let mut dest = Vec::with_capacity(nnz);
        for k in 0..nnz {
            if i_src[k] >= dim {
                set_last_error("rgmin_solver_add_equality: index out of range");
                return 1;
            }
            dest.push((i_src[k], a_src[k]));
        }
        let _ = unsafe { (*solver).solver.add_equality(dest, rhs) };
        0
    }
    #[cfg(not(feature = "highs"))]
    {
        let _ = (idx, coef, nnz, rhs);
        set_last_error("rgmin_solver_add_equality: build has no highs feature");
        1
    }
}

/// Drop every stored HiGHS equality. Returns 0, or 1 without `highs`.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_clear_equalities(solver: *mut rgmin_solver_t) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_clear_equalities: null solver");
        return 1;
    }
    #[cfg(feature = "highs")]
    {
        let _ = unsafe { (*solver).solver.clear_equalities() };
        0
    }
    #[cfg(not(feature = "highs"))]
    {
        set_last_error("rgmin_solver_clear_equalities: build has no highs feature");
        1
    }
}

/// HiGHS `solver` token. Unknown integers return 1.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_highs_solver(
    solver: *mut rgmin_solver_t,
    kind: i32,
) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_highs_solver: null solver");
        return 1;
    }
    let Some(tok) = crate::HighsSolverKind::from_ordinal(kind) else {
        set_last_error("rgmin_solver_set_highs_solver: unknown token");
        return 1;
    };
    #[cfg(feature = "highs")]
    {
        let _ = unsafe { (*solver).solver.set_highs_solver(tok) };
        0
    }
    #[cfg(not(feature = "highs"))]
    {
        let _ = tok;
        set_last_error("rgmin_solver_set_highs_solver: build has no highs feature");
        1
    }
}

/// HiGHS `run_crossover`. Unknown integers return 1.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_highs_crossover(
    solver: *mut rgmin_solver_t,
    kind: i32,
) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_highs_crossover: null solver");
        return 1;
    }
    let Some(tok) = crate::HighsCrossover::from_ordinal(kind) else {
        set_last_error("rgmin_solver_set_highs_crossover: unknown token");
        return 1;
    };
    #[cfg(feature = "highs")]
    {
        let _ = unsafe { (*solver).solver.set_highs_crossover(tok) };
        0
    }
    #[cfg(not(feature = "highs"))]
    {
        let _ = tok;
        set_last_error("rgmin_solver_set_highs_crossover: build has no highs feature");
        1
    }
}

/// HiGHS user callback. NULL `cb` clears it.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
/// A non-null `cb` must stay callable with `user` until it is cleared or
/// the session is freed, from any thread that steps the session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_highs_callback(
    solver: *mut rgmin_solver_t,
    cb: Option<crate::HighsCCallback>,
    user: *mut c_void,
) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_highs_callback: null solver");
        return 1;
    }
    #[cfg(feature = "highs")]
    {
        let callback = cb.map(|function| unsafe { crate::HighsCallback::new(function, user) });
        let _ = unsafe { (*solver).solver.set_highs_callback(callback) };
        0
    }
    #[cfg(not(feature = "highs"))]
    {
        let _ = (cb, user);
        set_last_error("rgmin_solver_set_highs_callback: build has no highs feature");
        1
    }
}

/// Embedded manifold.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_manifold_t {
    RGMIN_MANIFOLD_EUCLIDEAN = 0,
    RGMIN_MANIFOLD_SPHERE = 1,
    RGMIN_MANIFOLD_SO3 = 2,
    RGMIN_MANIFOLD_STIEFEL = 3,
    RGMIN_MANIFOLD_SE3 = 4,
    /// Sella Cartesian \(R^{3N}/\mathrm{SE}(3)\). 3N, N >= 2.
    RGMIN_MANIFOLD_RIGID_QUOTIENT = 5,
    /// Mass-weighted Eckart (Sella IRC / Page–McIver). 3N, N >= 2.
    RGMIN_MANIFOLD_MW_RIGID = 6,
    /// Reserved: 7 SPD, 8 Grassmann, 9 Hyperbolic, 10 Poincare.
    /// Product of unit spheres. Shape from `rgmin_solver_set_oblique`.
    RGMIN_MANIFOLD_OBLIQUE = 11,
    /// Simplex with the Fisher metric. manopt `multinomialfactory` (m = 1).
    RGMIN_MANIFOLD_MULTINOMIAL = 12,
    /// Product of unit circles (S^1)^n. Packed length 2n.
    /// Token defaults to n = 1; use rgmin_solver_set_complex_circle.
    RGMIN_MANIFOLD_COMPLEX_CIRCLE = 13,
    /// Real symmetric n-by-n, row-major n². manopt `symmetricfactory`.
    RGMIN_MANIFOLD_SYMMETRIC = 14,
    /// Real skew-symmetric n-by-n, row-major n², n >= 2.
    /// manopt `skewsymmetricfactory`.
    RGMIN_MANIFOLD_SKEWSYMMETRIC = 15,
    /// Complex Euclidean C^n. Packed interleaved, length 2n.
    /// Token defaults to n = 1; use rgmin_solver_set_euclidean_complex.
    RGMIN_MANIFOLD_EUCLIDEAN_COMPLEX = 16,
    /// Singleton {A} of packed length n. manopt `constantfactory`.
    /// Token defaults to n = 1; use rgmin_solver_set_constant.
    RGMIN_MANIFOLD_CONSTANT = 17,
    /// Doubly-stochastic n-by-n, packed n². Token defaults to n = 2;
    /// use rgmin_solver_set_multinomial_ds. Reserved 7-10 unused.
    RGMIN_MANIFOLD_MULTINOMIAL_DS = 18,
    /// Symmetric doubly-stochastic n-by-n. Token defaults to n = 2;
    /// use rgmin_solver_set_multinomial_sym. Reserved 7-10 unused.
    RGMIN_MANIFOLD_MULTINOMIAL_SYM = 19,
    /// Complex unit sphere C^n, packed 2n. Token defaults to n = 1;
    /// use rgmin_solver_set_sphere_complex. Reserved 7-10 unused.
    RGMIN_MANIFOLD_SPHERE_COMPLEX = 20,
    /// Positive orthant of packed length n. manopt `positivefactory`.
    /// Token defaults to n = 1; use rgmin_solver_set_positive.
    /// Reserved 7-10 unused.
    RGMIN_MANIFOLD_POSITIVE = 21,
    /// Centered m-by-n matrices, packed m*n. Token defaults to
    /// 2-by-2 centered columns; use rgmin_solver_set_centered_matrix.
    /// Symmetric positive definite matrices. Released 0.5.2 token 7.
    RGMIN_MANIFOLD_SPD = 7,
    /// Real Grassmann. Released 0.5.2 token 8.
    RGMIN_MANIFOLD_GRASSMANN = 8,
    /// Hyperboloid. Released 0.5.2 token 9.
    RGMIN_MANIFOLD_HYPERBOLIC = 9,
    /// Poincare ball. Released 0.5.2 token 10.
    RGMIN_MANIFOLD_POINCARE = 10,
    /// Centered m-by-n matrices.
    RGMIN_MANIFOLD_CENTERED_MATRIX = 22,
    /// Complex unitary matrices. Released 0.5.2 token 23.
    RGMIN_MANIFOLD_UNITARY = 23,
}

/// Select the session manifold by token.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_manifold(
    solver: *mut rgmin_solver_t,
    manifold: rgmin_manifold_t,
) {
    if solver.is_null() {
        return;
    }
    let kind = match manifold {
        rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE => ManifoldKind::Sphere,
        rgmin_manifold_t::RGMIN_MANIFOLD_SO3 => ManifoldKind::So3,
        rgmin_manifold_t::RGMIN_MANIFOLD_STIEFEL => ManifoldKind::Stiefel,
        rgmin_manifold_t::RGMIN_MANIFOLD_SE3 => ManifoldKind::Se3,
        rgmin_manifold_t::RGMIN_MANIFOLD_RIGID_QUOTIENT => ManifoldKind::RigidQuotient,
        rgmin_manifold_t::RGMIN_MANIFOLD_MW_RIGID => ManifoldKind::MwRigid,
        rgmin_manifold_t::RGMIN_MANIFOLD_SPD => ManifoldKind::Spd,
        rgmin_manifold_t::RGMIN_MANIFOLD_GRASSMANN => ManifoldKind::Grassmann,
        rgmin_manifold_t::RGMIN_MANIFOLD_HYPERBOLIC => ManifoldKind::Hyperbolic,
        rgmin_manifold_t::RGMIN_MANIFOLD_POINCARE => ManifoldKind::PoincareBall,
        rgmin_manifold_t::RGMIN_MANIFOLD_UNITARY => ManifoldKind::Unitary { n: 1 },
        rgmin_manifold_t::RGMIN_MANIFOLD_OBLIQUE => ManifoldKind::Oblique { n: 0, m: 0 },
        rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL => ManifoldKind::Multinomial,
        rgmin_manifold_t::RGMIN_MANIFOLD_COMPLEX_CIRCLE => ManifoldKind::ComplexCircle { n: 1 },
        rgmin_manifold_t::RGMIN_MANIFOLD_SYMMETRIC => ManifoldKind::Symmetric,
        rgmin_manifold_t::RGMIN_MANIFOLD_SKEWSYMMETRIC => ManifoldKind::SkewSymmetric,
        rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN_COMPLEX => {
            ManifoldKind::EuclideanComplex { n: 1 }
        }
        rgmin_manifold_t::RGMIN_MANIFOLD_CONSTANT => ManifoldKind::Constant { n: 1 },
        rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_DS => {
            ManifoldKind::MultinomialDoublyStochastic { n: 2 }
        }
        rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_SYM => {
            ManifoldKind::MultinomialSymmetric { n: 2 }
        }
        rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE_COMPLEX => ManifoldKind::SphereComplex { n: 1 },
        rgmin_manifold_t::RGMIN_MANIFOLD_POSITIVE => ManifoldKind::Positive { n: 1 },
        rgmin_manifold_t::RGMIN_MANIFOLD_CENTERED_MATRIX => ManifoldKind::CenteredMatrix {
            m: 2,
            n: 2,
            rows: false,
        },
        rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN => ManifoldKind::Euclidean,
    };
    unsafe { (*solver).solver.set_manifold(kind) };
}

/// Oblique \(\mathrm{OB}(n,m)\): product of `m` unit spheres in `R^n`.
/// Packed column-major, length `n*m`. Not a 3N cluster.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_oblique(solver: *mut rgmin_solver_t, n: usize, m: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_oblique(n, m) };
}

/// Stiefel \(\mathrm{St}(n,p)\). `p = 1` is the sphere packing.
/// `p > 1` is packed column-major, length `n*p`.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_stiefel(solver: *mut rgmin_solver_t, n: usize, p: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_stiefel(n, p) };
}

/// `n` unit-modulus complex numbers. Packed interleaved, length `2 n`.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_complex_circle(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_complex_circle(n) };
}

/// Complex Euclidean \(\mathbb{C}^n\). Packed interleaved, length `2 n`.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_euclidean_complex(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_euclidean_complex(n) };
}

/// Singleton of packed length `n`. manopt `constantfactory`.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_constant(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_constant(n) };
}

/// Doubly-stochastic n-by-n, packed `n^2`. manopt
/// `multinomialdoublystochasticfactory`. Token 18 defaults to n = 2.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_multinomial_ds(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_multinomial_ds(n) };
}

/// Symmetric doubly-stochastic n-by-n, packed `n^2`. manopt
/// `multinomialsymmetricfactory`. Token 19 defaults to n = 2.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_multinomial_sym(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_multinomial_sym(n) };
}

/// Complex unit sphere in \(\mathbb{C}^n\). Packed interleaved, length `2 n`.
/// manopt `spherecomplexfactory`. Token 20 defaults to n = 1.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_sphere_complex(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_sphere_complex(n) };
}

/// Positive orthant of packed length `n`. manopt `positivefactory`.
/// Token 21 defaults to n = 1. Reserved tokens 7-10 unused.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_positive(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_positive(n) };
}

/// Centered `m x n` matrices, packed row-major `m n`.
/// manopt `centeredmatrixfactory`. Token 22 defaults to 2-by-2
/// centered columns. `center_rows != 0` centers rows (`1^T X = 0`).
/// Reserved tokens 7-10 unused.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_centered_matrix(
    solver: *mut rgmin_solver_t,
    m: usize,
    n: usize,
    center_rows: i32,
) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_centered_matrix(m, n, center_rows != 0) };
}

/// Per-atom masses for `RGMIN_MANIFOLD_MW_RIGID`. `n_atoms == 0` or a
/// null pointer clears them (unit mass).
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
/// `masses` is null or points to `n_atoms` readable f64s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_masses(
    solver: *mut rgmin_solver_t,
    masses: *const f64,
    n_atoms: usize,
) {
    if solver.is_null() {
        return;
    }
    if masses.is_null() || n_atoms == 0 {
        unsafe { (*solver).solver.set_masses(Array1::zeros(0)) };
        return;
    }
    let slice = unsafe { slice::from_raw_parts(masses, n_atoms) };
    unsafe { (*solver).solver.set_masses(Array1::from(slice.to_vec())) };
}

/// Periodic cell. Nonzero: Sella `proj_rot = false`, quotient is \(R^{3N}/T(3)\).
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_periodic(solver: *mut rgmin_solver_t, enabled: i32) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_periodic(enabled != 0) };
}

/// Why a distance constraint set was refused or could not be met.
/// Integers follow the variants of [`ConstraintError`].
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_constraint_error_t {
    /// The last error on this thread was not a constraint error.
    RGMIN_CONSTRAINT_NONE = 0,
    /// [`ConstraintError::Malformed`].
    RGMIN_CONSTRAINT_MALFORMED = 1,
    /// [`ConstraintError::OutOfRange`].
    RGMIN_CONSTRAINT_OUT_OF_RANGE = 2,
    /// [`ConstraintError::Mass`].
    RGMIN_CONSTRAINT_MASS = 3,
    /// [`ConstraintError::Unsupported`].
    RGMIN_CONSTRAINT_UNSUPPORTED = 4,
    /// [`ConstraintError::NotConverged`].
    RGMIN_CONSTRAINT_NOT_CONVERGED = 5,
    /// [`ConstraintError::Reversed`].
    RGMIN_CONSTRAINT_REVERSED = 6,
    /// [`ConstraintError::Dependent`].
    RGMIN_CONSTRAINT_DEPENDENT = 7,
    /// [`ConstraintError::LincsUnsafe`].
    RGMIN_CONSTRAINT_LINCS_UNSAFE = 8,
    /// [`ConstraintError::SettleDegenerate`].
    RGMIN_CONSTRAINT_SETTLE_DEGENERATE = 9,
    /// [`ConstraintError::Bound`].
    RGMIN_CONSTRAINT_BOUND = 10,
    /// [`ConstraintError::NonFinite`].
    RGMIN_CONSTRAINT_NON_FINITE = 11,
}

fn constraint_kind(e: &ConstraintError) -> rgmin_constraint_error_t {
    use rgmin_constraint_error_t::*;
    match e {
        ConstraintError::Malformed { .. } => RGMIN_CONSTRAINT_MALFORMED,
        ConstraintError::OutOfRange { .. } => RGMIN_CONSTRAINT_OUT_OF_RANGE,
        ConstraintError::Mass { .. } => RGMIN_CONSTRAINT_MASS,
        ConstraintError::Unsupported { .. } => RGMIN_CONSTRAINT_UNSUPPORTED,
        ConstraintError::NotConverged { .. } => RGMIN_CONSTRAINT_NOT_CONVERGED,
        ConstraintError::Reversed { .. } => RGMIN_CONSTRAINT_REVERSED,
        ConstraintError::Dependent { .. } => RGMIN_CONSTRAINT_DEPENDENT,
        ConstraintError::LincsUnsafe { .. } => RGMIN_CONSTRAINT_LINCS_UNSAFE,
        ConstraintError::SettleDegenerate { .. } => RGMIN_CONSTRAINT_SETTLE_DEGENERATE,
        ConstraintError::Bound { .. } => RGMIN_CONSTRAINT_BOUND,
        ConstraintError::NonFinite => RGMIN_CONSTRAINT_NON_FINITE,
    }
}

/// Kind of the last error on this thread when it was a constraint
/// error, else `RGMIN_CONSTRAINT_NONE`. ABI minor 31.
#[unsafe(no_mangle)]
pub extern "C" fn rgmin_last_constraint_error() -> rgmin_constraint_error_t {
    LAST_CONSTRAINT.with(Cell::get)
}

/// Solver for the blocks SETTLE does not take. Integers match
/// [`ConstraintSolver`].
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_constraint_solver_t {
    /// LINCS where its expansion converges at least as fast per pass as
    /// the Gauss-Seidel bound of SHAKE, SHAKE elsewhere.
    RGMIN_CONSTRAINT_SOLVER_AUTO = 0,
    /// Gauss-Seidel SHAKE on every block.
    RGMIN_CONSTRAINT_SOLVER_SHAKE = 1,
    /// LINCS on every block.
    RGMIN_CONSTRAINT_SOLVER_LINCS = 2,
}

/// Opaque distance constraint set. A session copies it.
pub struct rgmin_constraints_t {
    set: DistanceConstraints,
}

/// `(pair_i[k], pair_j[k])` for `k < npair`, null when a side is null
/// or an index does not fit.
unsafe fn read_pairs(
    pair_i: *const u32,
    pair_j: *const u32,
    npair: usize,
) -> Option<Vec<(usize, usize)>> {
    if npair == 0 {
        return Some(Vec::new());
    }
    if pair_i.is_null() || pair_j.is_null() {
        return None;
    }
    let (i, j) = unsafe {
        (
            slice::from_raw_parts(pair_i, npair),
            slice::from_raw_parts(pair_j, npair),
        )
    };
    let pairs = i
        .iter()
        .zip(j)
        .map(|(&a, &b)| Some((usize::try_from(a).ok()?, usize::try_from(b).ok()?)));
    pairs.collect()
}

fn boxed_constraints(made: crate::Result<DistanceConstraints>) -> *mut rgmin_constraints_t {
    match made {
        Ok(set) => Box::into_raw(Box::new(rgmin_constraints_t { set })),
        Err(e) => {
            status_from_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Fix `|x_i - x_j| = length[k]` for each pair `k`, with unit masses,
/// the automatic solver, SETTLE on rigid triangles and the rounding
/// floor as tolerance. Null on bad arguments, with
/// [`rgmin_last_error`] and [`rgmin_last_constraint_error`] set.
/// ABI minor 31.
///
/// # Safety
/// Each non-null pointer is valid for `npair` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_constraints_create(
    pair_i: *const u32,
    pair_j: *const u32,
    lengths: *const f64,
    npair: usize,
) -> *mut rgmin_constraints_t {
    let Some(pairs) = (unsafe { read_pairs(pair_i, pair_j, npair) }) else {
        set_last_error("rgmin_constraints_create: null pair array");
        return std::ptr::null_mut();
    };
    if lengths.is_null() && npair > 0 {
        set_last_error("rgmin_constraints_create: null lengths");
        return std::ptr::null_mut();
    }
    let lengths = if npair == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(lengths, npair) }.to_vec()
    };
    boxed_constraints(DistanceConstraints::new(pairs, lengths))
}

/// Fix each pair at its distance in `x`, packed `3 N` of length `n`.
/// Null on bad arguments. ABI minor 31.
///
/// # Safety
/// The pair arrays are valid for `npair` reads and `x` for `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_constraints_create_at(
    pair_i: *const u32,
    pair_j: *const u32,
    npair: usize,
    x: *const f64,
    n: usize,
) -> *mut rgmin_constraints_t {
    let Some(pairs) = (unsafe { read_pairs(pair_i, pair_j, npair) }) else {
        set_last_error("rgmin_constraints_create_at: null pair array");
        return std::ptr::null_mut();
    };
    if x.is_null() {
        set_last_error("rgmin_constraints_create_at: null x");
        return std::ptr::null_mut();
    }
    let x = unsafe { slice::from_raw_parts(x, n) };
    boxed_constraints(DistanceConstraints::at(pairs, ArrayView1::from(x)))
}

/// Release a set from [`rgmin_constraints_create`] or
/// [`rgmin_constraints_create_at`]. Null is a no-op.
///
/// # Safety
/// `set` is null or a live set from this ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_constraints_free(set: *mut rgmin_constraints_t) {
    if !set.is_null() {
        drop(unsafe { Box::from_raw(set) });
    }
}

/// Apply `change` to the set behind `set`, or report a null handle.
unsafe fn update_constraints(
    set: *mut rgmin_constraints_t,
    what: &str,
    change: impl FnOnce(DistanceConstraints) -> crate::Result<DistanceConstraints>,
) -> rgmin_status_t {
    if set.is_null() {
        set_last_error(&format!("{what}: null set"));
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let handle = unsafe { &mut *set };
    match change(handle.set.clone()) {
        Ok(next) => {
            handle.set = next;
            rgmin_status_t::RGMIN_SUCCESS
        }
        Err(e) => status_from_error(&e),
    }
}

/// One mass per atom of the coordinates the set is used with; the
/// corrections and projections then use the metric `M^-1`. ABI minor 31.
///
/// # Safety
/// `set` is null or live, and `masses` is null or valid for `n_atoms`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_constraints_set_masses(
    set: *mut rgmin_constraints_t,
    masses: *const f64,
    n_atoms: usize,
) -> rgmin_status_t {
    if masses.is_null() {
        set_last_error("rgmin_constraints_set_masses: null masses");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let masses = unsafe { slice::from_raw_parts(masses, n_atoms) }.to_vec();
    unsafe {
        update_constraints(set, "rgmin_constraints_set_masses", |c| {
            c.with_masses(masses)
        })
    }
}

/// Solver for the blocks SETTLE does not take. An unknown integer is
/// `RGMIN_INVALID_PARAMETER`. ABI minor 31.
///
/// # Safety
/// `set` is null or a live set from this ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_constraints_set_solver(
    set: *mut rgmin_constraints_t,
    solver: i32,
) -> rgmin_status_t {
    let solver = match solver {
        0 => ConstraintSolver::Auto,
        1 => ConstraintSolver::Shake,
        2 => ConstraintSolver::Lincs,
        _ => {
            set_last_error("rgmin_constraints_set_solver: unknown solver");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
    };
    unsafe {
        update_constraints(set, "rgmin_constraints_set_solver", |c| {
            Ok(c.with_solver(solver))
        })
    }
}

/// SETTLE on rigid triangles when `on` is nonzero (the default), else
/// the solver takes them. ABI minor 31.
///
/// # Safety
/// `set` is null or a live set from this ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_constraints_set_settle(
    set: *mut rgmin_constraints_t,
    on: i32,
) -> rgmin_status_t {
    unsafe {
        update_constraints(set, "rgmin_constraints_set_settle", |c| {
            Ok(c.with_settle(on != 0))
        })
    }
}

/// Relative tolerance on `|d^2 - r^2| / (2 d^2)`, the GROMACS
/// `shake-tol` measure. Zero asks for the rounding floor of each
/// constraint, and a smaller tolerance is raised to it. ABI minor 31.
///
/// # Safety
/// `set` is null or a live set from this ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_constraints_set_tolerance(
    set: *mut rgmin_constraints_t,
    tolerance: f64,
) -> rgmin_status_t {
    unsafe {
        update_constraints(set, "rgmin_constraints_set_tolerance", |c| {
            c.with_tolerance(tolerance)
        })
    }
}

/// Fix the LINCS expansion order (GROMACS `lincs-order`). Zero derives
/// it per block from the coupling and the tolerance. ABI minor 31.
///
/// # Safety
/// `set` is null or a live set from this ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_constraints_set_lincs_order(
    set: *mut rgmin_constraints_t,
    order: usize,
) -> rgmin_status_t {
    let order = (order > 0).then_some(order);
    unsafe {
        update_constraints(set, "rgmin_constraints_set_lincs_order", |c| {
            Ok(c.with_lincs_order(order))
        })
    }
}

/// Largest ratio of a relative residual to its tolerance at `x`: at
/// most one when every constraint is met. NaN on a null argument, a
/// length the set does not fit, or a non-finite residual. ABI minor 31.
///
/// # Safety
/// `set` is null or live, and `x` is null or valid for `n` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_constraints_excess(
    set: *const rgmin_constraints_t,
    x: *const f64,
    n: usize,
) -> f64 {
    if set.is_null() || x.is_null() {
        return f64::NAN;
    }
    let set = unsafe { &(*set).set };
    if set.fits(n).is_err() {
        return f64::NAN;
    }
    set.excess(ArrayView1::from(unsafe { slice::from_raw_parts(x, n) }))
}

/// Return `x` to the set by corrections along the bonds of `anchor`,
/// the point `x` was stepped from. Both have length `n`. On an error
/// `x` is unchanged. ABI minor 31.
///
/// # Safety
/// `set` is null or live; `anchor` and `x` are null or valid for `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_constraints_constrain(
    set: *const rgmin_constraints_t,
    anchor: *const f64,
    x: *mut f64,
    n: usize,
) -> rgmin_status_t {
    if set.is_null() || anchor.is_null() || x.is_null() {
        set_last_error("rgmin_constraints_constrain: null argument");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let set = unsafe { &(*set).set };
    let anchor = ArrayView1::from(unsafe { slice::from_raw_parts(anchor, n) });
    let out = unsafe { slice::from_raw_parts_mut(x, n) };
    let mut moved = Array1::from(out.to_vec());
    match set.constrain(anchor, &mut moved) {
        Ok(_) => {
            out.copy_from_slice(moved.as_slice().expect("contiguous"));
            rgmin_status_t::RGMIN_SUCCESS
        }
        Err(e) => status_from_error(&e),
    }
}

/// Copy `set` into the session. Every later step keeps each constraint
/// to its tolerance; a setting the set cannot run with is
/// `RGMIN_CONSTRAINT_ERROR` here or at the next step. ABI minor 31.
///
/// # Safety
/// `solver` and `set` are null or live handles from this ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_distance_constraints(
    solver: *mut rgmin_solver_t,
    set: *const rgmin_constraints_t,
) -> rgmin_status_t {
    if solver.is_null() || set.is_null() {
        set_last_error("rgmin_solver_set_distance_constraints: null argument");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let set = unsafe { (*set).set.clone() };
    match unsafe { (*solver).solver.set_distance_constraints(set) } {
        Ok(()) => rgmin_status_t::RGMIN_SUCCESS,
        Err(e) => status_from_error(&e),
    }
}

/// Drop the session's constraint set and its method memory. Null and a
/// session without a set are no-ops. ABI minor 31.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_clear_distance_constraints(solver: *mut rgmin_solver_t) {
    if !solver.is_null() {
        unsafe { (*solver).solver.clear_distance_constraints() };
    }
}

/// One outer iteration. `x` is in/out. Callbacks live for this call only.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
/// `eval` and `grad` must be callable with `user` for the duration of this call.
/// `x` is null or a writable rank-1 f64 tensor, and `out` is null or
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_step(
    solver: *mut rgmin_solver_t,
    eval: Option<rgmin_eval_fn>,
    grad: Option<rgmin_grad_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if solver.is_null() {
            set_last_error("rgmin_solver_step: null solver");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let eval = match eval {
            Some(f) => f,
            None => {
                set_last_error("rgmin_solver_step: eval is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let grad = match grad {
            Some(f) => f,
            None => {
                set_last_error("rgmin_solver_step: grad is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        if out.is_null() {
            set_last_error("rgmin_solver_step: out is NULL");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let n = init.len();
        let obj = c_oracle(eval, grad, user, n);
        let mut pos = Array1::from(init);
        match unsafe { (*solver).solver.step(&obj, &mut pos) } {
            Ok(rep) => write_report(solver, x, out, rep),
            Err(e) => status_from_error(&e),
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("rgmin_solver_step: panic");
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

/// One Newton / RFO iteration. `hess` writes a length-`n*n` row-major Hessian.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
/// `eval`, `grad` and `hess` must be callable with `user` for the duration of this call.
/// `x` is null or a writable rank-1 f64 tensor, and `out` is null or
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_step_hess(
    solver: *mut rgmin_solver_t,
    eval: Option<rgmin_eval_fn>,
    grad: Option<rgmin_grad_fn>,
    hess: Option<rgmin_hess_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if solver.is_null() {
            set_last_error("rgmin_solver_step_hess: null solver");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let eval = match eval {
            Some(f) => f,
            None => {
                set_last_error("rgmin_solver_step_hess: eval is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let grad = match grad {
            Some(f) => f,
            None => {
                set_last_error("rgmin_solver_step_hess: grad is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let hess = match hess {
            Some(f) => f,
            None => {
                set_last_error("rgmin_solver_step_hess: hess is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        if out.is_null() {
            set_last_error("rgmin_solver_step_hess: out is NULL");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let n = init.len();
        let obj = c_oracle_hess(eval, grad, hess, user, n);
        let mut pos = Array1::from(init);
        match unsafe { (*solver).solver.step_hess(&obj, &mut pos) } {
            Ok(rep) => write_report(solver, x, out, rep),
            Err(e) => status_from_error(&e),
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("rgmin_solver_step_hess: panic");
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

/// One Steihaug-Toint iteration. `hvp` writes `H(x) v` and is not asked
/// for a dense matrix. Legal on `RGMIN_NEWTON_CG` and `RGMIN_NEWTON_PCG`.
///
/// # Safety
/// Callbacks live for this call only. `x` is a writable rank-1 CPU f64 tensor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_step_hvp(
    solver: *mut rgmin_solver_t,
    eval: Option<rgmin_eval_fn>,
    grad: Option<rgmin_grad_fn>,
    hvp: Option<rgmin_hvp_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if solver.is_null() {
            set_last_error("rgmin_solver_step_hvp: null solver");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let eval = match eval {
            Some(f) => f,
            None => {
                set_last_error("rgmin_solver_step_hvp: eval is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let grad = match grad {
            Some(f) => f,
            None => {
                set_last_error("rgmin_solver_step_hvp: grad is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let hvp = match hvp {
            Some(f) => f,
            None => {
                set_last_error("rgmin_solver_step_hvp: hvp is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        if out.is_null() {
            set_last_error("rgmin_solver_step_hvp: out is NULL");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let n = init.len();
        let obj = c_oracle_hvp(eval, grad, hvp, user, n);
        let mut pos = Array1::from(init);
        match unsafe { (*solver).solver.step_hvp(&obj, &mut pos) } {
            Ok(rep) => write_report(solver, x, out, rep),
            Err(e) => status_from_error(&e),
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("rgmin_solver_step_hvp: panic");
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

/// One outer iteration with a fused `(f, g)` callback.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
/// `evalgrad` must be callable with `user` for the duration of this call.
/// `x` is null or a writable rank-1 f64 tensor, and `out` is null or
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_step_fg(
    solver: *mut rgmin_solver_t,
    evalgrad: Option<rgmin_evalgrad_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if solver.is_null() {
            set_last_error("rgmin_solver_step_fg: null solver");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let evalgrad = match evalgrad {
            Some(f) => f,
            None => {
                set_last_error("rgmin_solver_step_fg: evalgrad is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        if out.is_null() {
            set_last_error("rgmin_solver_step_fg: out is NULL");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let n = init.len();
        let obj = c_oracle_fg(evalgrad, user, n);
        let mut pos = Array1::from(init);
        match unsafe { (*solver).solver.step(&obj, &mut pos) } {
            Ok(rep) => write_report(solver, x, out, rep),
            Err(e) => status_from_error(&e),
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("rgmin_solver_step_fg: panic");
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

/// Record `s`, `y` from the caller's previous outer. No evaluation.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
/// `s` and `y` are each null or point to `n` readable f64s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_push_pair(
    solver: *mut rgmin_solver_t,
    s: *const f64,
    y: *const f64,
    n: usize,
) -> i32 {
    if solver.is_null() || s.is_null() || y.is_null() {
        set_last_error("rgmin_solver_push_pair: null argument");
        return 1;
    }
    let dim = unsafe { (*solver).solver.dim() };
    if n != dim {
        set_last_error("rgmin_solver_push_pair: n does not match session dim");
        return 1;
    }
    let sv = unsafe { std::slice::from_raw_parts(s, n) };
    let yv = unsafe { std::slice::from_raw_parts(y, n) };
    let ok = unsafe {
        (*solver)
            .solver
            .push_pair(ndarray::ArrayView1::from(sv), ndarray::ArrayView1::from(yv))
    };
    i32::from(!ok)
}

/// Packed length of `x`. A null session returns zero.
///
/// # Safety
/// A non-null `solver` must be a live session from `rgmin_solver_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_dim(solver: *const rgmin_solver_t) -> usize {
    if solver.is_null() {
        return 0;
    }
    unsafe { (*solver).solver.dim() }
}

/// Outer steps completed. A null session returns zero.
///
/// # Safety
/// A non-null `solver` must be a live session from `rgmin_solver_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_steps(solver: *const rgmin_solver_t) -> usize {
    if solver.is_null() {
        return 0;
    }
    unsafe { (*solver).solver.steps() }
}

/// Gradient evaluations on this session. A null solver returns zero.
/// Finite-difference products are [`rgmin_solver_hvp_calls`], not gradients.
///
/// # Safety
/// A non-null `solver` must be a live session from `rgmin_solver_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_gradient_calls(solver: *const rgmin_solver_t) -> usize {
    if solver.is_null() {
        return 0;
    }
    unsafe { (*solver).solver.gradient_calls() }
}

/// Hessian-vector products on this session. A null solver returns zero.
/// Each product, analytic or forward-difference, counts as one.
///
/// # Safety
/// A non-null `solver` must be a live session from `rgmin_solver_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_hvp_calls(solver: *const rgmin_solver_t) -> usize {
    if solver.is_null() {
        return 0;
    }
    unsafe { (*solver).solver.hvp_calls() }
}

/// Number of accepted L-BFGS pairs. Null and other methods return zero.
///
/// # Safety
/// A non-null `solver` must be a live session from `rgmin_solver_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_pair_count(solver: *const rgmin_solver_t) -> usize {
    if solver.is_null() {
        return 0;
    }
    unsafe { (*solver).solver.pair_count() }
}

/// Copy the gradient behind the last successful step report into `out`.
/// `n` must equal the session dimension. ABI minor 29.
///
/// # Safety
/// A non-null `out` must hold `n` writable doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_report_gradient(
    solver: *const rgmin_solver_t,
    out: *mut f64,
    n: usize,
) -> rgmin_status_t {
    if solver.is_null() || out.is_null() {
        set_last_error("rgmin_solver_report_gradient: null argument");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let Some(grad) = (unsafe { &(*solver).grad }) else {
        set_last_error("rgmin_solver_report_gradient: no step report yet");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    };
    if n != grad.len() {
        set_last_error("rgmin_solver_report_gradient: n does not match session dim");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    unsafe { slice::from_raw_parts_mut(out, n) }
        .copy_from_slice(grad.as_slice().expect("contiguous"));
    rgmin_status_t::RGMIN_SUCCESS
}

/// Line-search kind. Integers match the C enum `rgmin_linesearch_kind_t`.
/// `rgmin_linesearch_create` takes the integer. [`rgmin_solver_set_linesearch`]
/// takes this enum. The C typedef `rgmin_linesearch_t` is the opaque handle.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_linesearch_kind_t {
    /// Brent on a golden bracket.
    RGMIN_LINESEARCH_BRENT = 0,
    /// Armijo backtracking.
    RGMIN_LINESEARCH_BACKTRACKING = 1,
    /// Goldstein two-sided condition.
    RGMIN_LINESEARCH_GOLDSTEIN = 2,
    /// Strong Wolfe with zoom.
    RGMIN_LINESEARCH_WOLFE = 3,
}

/// Same discriminants as [`rgmin_linesearch_kind_t`]. Tests and the 0.5.2
/// setter spell the kind this way. The C header reserves `rgmin_linesearch_t`
/// for the opaque search handle.
pub type rgmin_linesearch_t = rgmin_linesearch_kind_t;

/// Caller-driven line search. It does not call an objective.
pub struct rgmin_linesearch {
    search: SuppliedSearch,
    n: usize,
}

fn linesearch_kind(
    kind: i32,
    c1: f64,
    c2: f64,
    beta: f64,
    maxiter: usize,
    tol: f64,
) -> Option<LineSearch> {
    match kind {
        0 => Some(LineSearch::Brent { maxiter, tol }),
        1 => Some(LineSearch::Backtracking {
            c: c1,
            beta,
            maxiter,
        }),
        2 => Some(LineSearch::Goldstein {
            c: c1,
            beta,
            maxiter,
        }),
        3 => Some(LineSearch::Wolfe { c1, c2, maxiter }),
        _ => None,
    }
}

/// Start a line search. The first [`rgmin_linesearch`] call supplies `f` and `g` at `x`.
///
/// # Safety
/// `x` and `dir` point at `n` doubles. The returned pointer is freed with
/// [`rgmin_linesearch_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_linesearch_create(
    kind: i32,
    x: *const f64,
    dir: *const f64,
    n: usize,
    istep: f64,
    c1: f64,
    c2: f64,
    beta: f64,
    maxiter: usize,
    tol: f64,
) -> *mut rgmin_linesearch {
    if x.is_null() || dir.is_null() || n == 0 {
        set_last_error("rgmin_linesearch_create: null pointer or n = 0");
        return std::ptr::null_mut();
    }
    let Some(rule) = linesearch_kind(kind, c1, c2, beta, maxiter, tol) else {
        set_last_error("rgmin_linesearch_create: unknown kind");
        return std::ptr::null_mut();
    };
    let x = unsafe { slice::from_raw_parts(x, n) };
    let dir = unsafe { slice::from_raw_parts(dir, n) };
    let search = SuppliedSearch::new(rule, ArrayView1::from(x), ArrayView1::from(dir), istep);
    Box::into_raw(Box::new(rgmin_linesearch { search, n }))
}

/// Advance with the caller's energy and gradient. Never calls an objective.
///
/// # Safety
/// `search` comes from [`rgmin_linesearch_create`]. `g` and `x_out` have the
/// search length. `alpha`, `value`, and `need_eval` are writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_linesearch(
    search: *mut rgmin_linesearch,
    f: f64,
    g: *const f64,
    x_out: *mut f64,
    alpha: *mut f64,
    value: *mut f64,
    need_eval: *mut i32,
) -> i32 {
    if search.is_null()
        || g.is_null()
        || x_out.is_null()
        || alpha.is_null()
        || value.is_null()
        || need_eval.is_null()
    {
        set_last_error("rgmin_linesearch: null argument");
        return 1;
    }
    let n = unsafe { (*search).n };
    let grad = unsafe { slice::from_raw_parts(g, n) };
    let step = unsafe { (*search).search.offer(f, ArrayView1::from(grad)) };
    let out = unsafe { slice::from_raw_parts_mut(x_out, n) };
    match step {
        SearchStep::Evaluate { alpha: a, x } => {
            out.copy_from_slice(x.as_slice().expect("contiguous"));
            unsafe {
                *alpha = a;
                *value = 0.0;
                *need_eval = 1;
            }
        }
        SearchStep::Accepted {
            x,
            value: v,
            alpha: a,
        } => {
            out.copy_from_slice(x.as_slice().expect("contiguous"));
            unsafe {
                *alpha = a;
                *value = v;
                *need_eval = 0;
            }
        }
    }
    0
}

/// Release a search from [`rgmin_linesearch_create`].
///
/// # Safety
/// `search` is null or a pointer from [`rgmin_linesearch_create`] that has
/// not been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_linesearch_free(search: *mut rgmin_linesearch) {
    if !search.is_null() {
        drop(unsafe { Box::from_raw(search) });
    }
}

/// Two-loop `d = -H g`. No evaluation and no push.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
/// `grad` is null or points to `n` readable f64s, and `dir` is null or
/// points to `n` writable f64s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_search_direction(
    solver: *mut rgmin_solver_t,
    grad: *const f64,
    dir: *mut f64,
    n: usize,
) -> rgmin_status_t {
    if solver.is_null() || grad.is_null() || dir.is_null() {
        set_last_error("rgmin_solver_search_direction: null argument");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let dim = unsafe { (*solver).solver.dim() };
    if n != dim {
        set_last_error("rgmin_solver_search_direction: n does not match session dim");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let gv = unsafe { std::slice::from_raw_parts(grad, n) };
    match unsafe {
        (*solver)
            .solver
            .search_direction(ndarray::ArrayView1::from(gv))
    } {
        Ok(d) => {
            let out = unsafe { std::slice::from_raw_parts_mut(dir, n) };
            out.copy_from_slice(d.as_slice().unwrap());
            rgmin_status_t::RGMIN_SUCCESS
        }
        Err(e) => status_from_error(&e),
    }
}

/// One Newton / RFO iteration with a fused `(f, g)` callback.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
/// `evalgrad` and `hess` must be callable with `user` for the duration of this call.
/// `x` is null or a writable rank-1 f64 tensor, and `out` is null or
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_step_hess_fg(
    solver: *mut rgmin_solver_t,
    evalgrad: Option<rgmin_evalgrad_fn>,
    hess: Option<rgmin_hess_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if solver.is_null() {
            set_last_error("rgmin_solver_step_hess_fg: null solver");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let evalgrad = match evalgrad {
            Some(f) => f,
            None => {
                set_last_error("rgmin_solver_step_hess_fg: evalgrad is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let hess = match hess {
            Some(f) => f,
            None => {
                set_last_error("rgmin_solver_step_hess_fg: hess is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        if out.is_null() {
            set_last_error("rgmin_solver_step_hess_fg: out is NULL");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let n = init.len();
        let obj = c_oracle_hess_fg(evalgrad, hess, user, n);
        let mut pos = Array1::from(init);
        match unsafe { (*solver).solver.step_hess(&obj, &mut pos) } {
            Ok(rep) => write_report(solver, x, out, rep),
            Err(e) => status_from_error(&e),
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("rgmin_solver_step_hess_fg: panic");
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

fn c_oracle(
    eval: rgmin_eval_fn,
    grad: rgmin_grad_fn,
    user: *mut c_void,
    n: usize,
) -> Oracle<impl Fn(ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    let eval_ptr = eval as usize;
    let grad_ptr = grad as usize;
    let user_addr = user as usize;
    let scratch = Scratch::new();
    Oracle::unbounded(n, move |xv| {
        let eval_fn: rgmin_eval_fn = unsafe { std::mem::transmute(eval_ptr) };
        let grad_fn: rgmin_grad_fn = unsafe { std::mem::transmute(grad_ptr) };
        let user = user_addr as *mut c_void;
        let m = xv.len();
        let mut s = scratch.lock().expect("ffi scratch");
        let xt = s.x_tensor(xv);
        let mut value = 0.0;
        let ev_st = unsafe { eval_fn(user, xt, &mut value) };
        let mut g = Array1::zeros(m);
        let gt = s.out.point_at(g.as_mut_ptr(), m);
        let gr_st = unsafe { grad_fn(user, xt, gt) };
        if ev_st != rgmin_status_t::RGMIN_SUCCESS || gr_st != rgmin_status_t::RGMIN_SUCCESS {
            // A failed callback yields no gradient: the NaN fill keeps
            // any consumer from mistaking half-written storage for a
            // small (or converged) gradient.
            return (f64::INFINITY, Array1::from_elem(m, f64::NAN));
        }
        (value, g)
    })
}

fn c_oracle_hess(
    eval: rgmin_eval_fn,
    grad: rgmin_grad_fn,
    hess: rgmin_hess_fn,
    user: *mut c_void,
    n: usize,
) -> impl HessianObjective {
    let eval_ptr = eval as usize;
    let grad_ptr = grad as usize;
    let hess_ptr = hess as usize;
    let user_addr = user as usize;
    let scratch = Scratch::new();
    let hscratch = Scratch::new();
    HessianOracle::unbounded(
        n,
        move |xv| {
            let eval_fn: rgmin_eval_fn = unsafe { std::mem::transmute(eval_ptr) };
            let grad_fn: rgmin_grad_fn = unsafe { std::mem::transmute(grad_ptr) };
            let user = user_addr as *mut c_void;
            let m = xv.len();
            let mut s = scratch.lock().expect("ffi scratch");
            let xt = s.x_tensor(xv);
            let mut value = 0.0;
            let ev_st = unsafe { eval_fn(user, xt, &mut value) };
            let mut g = Array1::zeros(m);
            let gt = s.out.point_at(g.as_mut_ptr(), m);
            let gr_st = unsafe { grad_fn(user, xt, gt) };
            if ev_st != rgmin_status_t::RGMIN_SUCCESS || gr_st != rgmin_status_t::RGMIN_SUCCESS {
                return (f64::INFINITY, Array1::from_elem(m, f64::NAN));
            }
            (value, g)
        },
        move |xv| {
            let hess_fn: rgmin_hess_fn = unsafe { std::mem::transmute(hess_ptr) };
            let user = user_addr as *mut c_void;
            let mut s = hscratch.lock().expect("ffi scratch");
            let xt = s.x_tensor(xv);
            let mut h = Array2::zeros((n, n));
            let ht = s
                .out
                .point_at(h.as_slice_mut().expect("contiguous").as_mut_ptr(), n * n);
            let st = unsafe { hess_fn(user, xt, ht) };
            if st != rgmin_status_t::RGMIN_SUCCESS {
                // A failed Hessian is no Hessian: NaN poisons the
                // Newton solve into a refused step instead of quietly
                // substituting a plausible matrix.
                return Array2::from_elem((n, n), f64::NAN);
            }
            h
        },
    )
}

fn c_oracle_hvp(
    eval: rgmin_eval_fn,
    grad: rgmin_grad_fn,
    hvp: rgmin_hvp_fn,
    user: *mut c_void,
    n: usize,
) -> impl HessianVector {
    let eval_ptr = eval as usize;
    let grad_ptr = grad as usize;
    let hvp_ptr = hvp as usize;
    let user_addr = user as usize;
    let fg_scratch = Scratch::new();
    let hvp_x = Scratch::new();
    let hvp_v = Scratch::new();
    HvpOracle::unbounded(
        n,
        move |xv| {
            let eval_fn: rgmin_eval_fn = unsafe { std::mem::transmute(eval_ptr) };
            let grad_fn: rgmin_grad_fn = unsafe { std::mem::transmute(grad_ptr) };
            let user = user_addr as *mut c_void;
            let m = xv.len();
            let mut s = fg_scratch.lock().expect("ffi scratch");
            let xt = s.x_tensor(xv);
            let mut value = 0.0;
            let ev_st = unsafe { eval_fn(user, xt, &mut value) };
            let mut g = Array1::zeros(m);
            let gt = s.out.point_at(g.as_mut_ptr(), m);
            let gr_st = unsafe { grad_fn(user, xt, gt) };
            if ev_st != rgmin_status_t::RGMIN_SUCCESS || gr_st != rgmin_status_t::RGMIN_SUCCESS {
                return (f64::INFINITY, Array1::from_elem(m, f64::NAN));
            }
            (value, g)
        },
        move |xv, vv| {
            let hvp_fn: rgmin_hvp_fn = unsafe { std::mem::transmute(hvp_ptr) };
            let user = user_addr as *mut c_void;
            let mut s = hvp_x.lock().expect("ffi scratch");
            let xt = s.x_tensor(xv);
            let mut vs = hvp_v.lock().expect("ffi scratch");
            let vt = vs.x_tensor(vv);
            let mut hv = Array1::zeros(vv.len());
            let ht = vs.out.point_at(hv.as_mut_ptr(), hv.len());
            let st = unsafe { hvp_fn(user, xt, vt, ht) };
            if st != rgmin_status_t::RGMIN_SUCCESS {
                return Array1::from_elem(hv.len(), f64::NAN);
            }
            hv
        },
    )
}

fn c_oracle_fg(
    evalgrad: rgmin_evalgrad_fn,
    user: *mut c_void,
    n: usize,
) -> Oracle<impl Fn(ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    let fg_ptr = evalgrad as usize;
    let user_addr = user as usize;
    let scratch = Scratch::new();
    Oracle::unbounded(n, move |xv| {
        let fg_fn: rgmin_evalgrad_fn = unsafe { std::mem::transmute(fg_ptr) };
        let user = user_addr as *mut c_void;
        let m = xv.len();
        let mut s = scratch.lock().expect("ffi scratch");
        let xt = s.x_tensor(xv);
        let mut value = 0.0;
        let mut g = Array1::zeros(m);
        let gt = s.out.point_at(g.as_mut_ptr(), m);
        let st = unsafe { fg_fn(user, xt, &mut value, gt) };
        if st != rgmin_status_t::RGMIN_SUCCESS {
            return (f64::INFINITY, Array1::from_elem(m, f64::NAN));
        }
        (value, g)
    })
}

fn c_oracle_hess_fg(
    evalgrad: rgmin_evalgrad_fn,
    hess: rgmin_hess_fn,
    user: *mut c_void,
    n: usize,
) -> impl HessianObjective {
    let fg_ptr = evalgrad as usize;
    let hess_ptr = hess as usize;
    let user_addr = user as usize;
    let scratch = Scratch::new();
    let hscratch = Scratch::new();
    HessianOracle::unbounded(
        n,
        move |xv| {
            let fg_fn: rgmin_evalgrad_fn = unsafe { std::mem::transmute(fg_ptr) };
            let user = user_addr as *mut c_void;
            let m = xv.len();
            let mut s = scratch.lock().expect("ffi scratch");
            let xt = s.x_tensor(xv);
            let mut value = 0.0;
            let mut g = Array1::zeros(m);
            let gt = s.out.point_at(g.as_mut_ptr(), m);
            let st = unsafe { fg_fn(user, xt, &mut value, gt) };
            if st != rgmin_status_t::RGMIN_SUCCESS {
                return (f64::INFINITY, Array1::from_elem(m, f64::NAN));
            }
            (value, g)
        },
        move |xv| {
            let hess_fn: rgmin_hess_fn = unsafe { std::mem::transmute(hess_ptr) };
            let user = user_addr as *mut c_void;
            let mut s = hscratch.lock().expect("ffi scratch");
            let xt = s.x_tensor(xv);
            let mut h = Array2::zeros((n, n));
            let ht = s
                .out
                .point_at(h.as_slice_mut().expect("contiguous").as_mut_ptr(), n * n);
            let st = unsafe { hess_fn(user, xt, ht) };
            if st != rgmin_status_t::RGMIN_SUCCESS {
                return Array2::from_elem((n, n), f64::NAN);
            }
            h
        },
    )
}

fn write_report(
    solver: *mut rgmin_solver_t,
    x: *mut DLManagedTensorVersioned,
    out: *mut rgmin_report_t,
    rep: crate::Report,
) -> rgmin_status_t {
    let dest = match cpu_f64_slice_mut(x, "x") {
        Ok(s) => s,
        Err(st) => return st,
    };
    dest.copy_from_slice(rep.coords.as_slice().expect("contiguous"));
    unsafe {
        *out = rgmin_report_t {
            value: rep.value,
            steps: rep.steps,
            grad_norm: rep.grad_norm,
            stop: stop_from_status(rep.status),
        };
        (*solver).grad = rep.grad;
    }
    rgmin_status_t::RGMIN_SUCCESS
}

/// Stored `(s, y)` pairs for [`rgmin_lbfgs_direction`]. The handle does not
/// evaluate an objective.
pub struct rgmin_lbfgs_pairs {
    pairs: LbfgsPairs,
}

/// Apply `H0` to `v`, writing `n` doubles to `out`. Return 0 on success.
///
/// The callback is not an objective. It must not call back into the same
/// pair handle.
pub type rgmin_h0_fn =
    unsafe extern "C" fn(user: *mut c_void, v: *const f64, out: *mut f64, n: usize) -> i32;

/// Create a pair handle of dimension `n`. `capacity` 0 keeps ten pairs.
/// Null when `n` is 0.
///
/// # Safety
/// The returned pointer is freed with [`rgmin_lbfgs_pairs_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_lbfgs_pairs_create(
    n: usize,
    capacity: usize,
) -> *mut rgmin_lbfgs_pairs {
    match LbfgsPairs::new(n, capacity) {
        Ok(pairs) => Box::into_raw(Box::new(rgmin_lbfgs_pairs { pairs })),
        Err(err) => {
            set_last_error(&err.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Append one pair. `stored`, when non-null, is 1 if the pair was kept.
/// A non-positive curvature pair returns success and sets `stored` to 0.
///
/// # Safety
/// `pairs` is null or a live handle from [`rgmin_lbfgs_pairs_create`].
/// `s` and `y` are null or point at `n` readable doubles. `stored` is
/// null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_lbfgs_pairs_push(
    pairs: *mut rgmin_lbfgs_pairs,
    s: *const f64,
    y: *const f64,
    n: usize,
    stored: *mut i32,
) -> rgmin_status_t {
    unsafe { lbfgs_pairs_push_c("rgmin_lbfgs_pairs_push", pairs, s, y, None, n, stored) }
}

/// Append one pair with the caller's `rho` in place of `1 / (s · y)`.
/// `stored`, when non-null, is 1 if the pair was kept. The pair passes
/// the curvature gate of [`rgmin_lbfgs_pairs_push`] first, and a `rho`
/// that is not finite and positive is declined. A declined pair returns
/// success and sets `stored` to 0. ABI minor 43.
///
/// # Safety
/// `pairs` is null or a live handle from [`rgmin_lbfgs_pairs_create`].
/// `s` and `y` are null or point at `n` readable doubles. `stored` is
/// null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_lbfgs_pairs_push_rho(
    pairs: *mut rgmin_lbfgs_pairs,
    s: *const f64,
    y: *const f64,
    rho: f64,
    n: usize,
    stored: *mut i32,
) -> rgmin_status_t {
    unsafe {
        lbfgs_pairs_push_c(
            "rgmin_lbfgs_pairs_push_rho",
            pairs,
            s,
            y,
            Some(rho),
            n,
            stored,
        )
    }
}

/// # Safety
/// The pointer contract of [`rgmin_lbfgs_pairs_push`].
unsafe fn lbfgs_pairs_push_c(
    name: &str,
    pairs: *mut rgmin_lbfgs_pairs,
    s: *const f64,
    y: *const f64,
    rho: Option<f64>,
    n: usize,
    stored: *mut i32,
) -> rgmin_status_t {
    if pairs.is_null() || s.is_null() || y.is_null() {
        set_last_error(&format!("{name}: null argument"));
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let s = ArrayView1::from(unsafe { slice::from_raw_parts(s, n) });
    let y = ArrayView1::from(unsafe { slice::from_raw_parts(y, n) });
    let handle = unsafe { &mut (*pairs).pairs };
    let pushed = match rho {
        Some(rho) => handle.push_rho(s, y, rho),
        None => handle.push(s, y),
    };
    match pushed {
        Ok(kept) => {
            if !stored.is_null() {
                unsafe { *stored = i32::from(kept) };
            }
            rgmin_status_t::RGMIN_SUCCESS
        }
        Err(err) => status_from_error(&err),
    }
}

/// Number of stored pairs. A null handle returns zero.
///
/// # Safety
/// `pairs` is null or a live handle from [`rgmin_lbfgs_pairs_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_lbfgs_pairs_count(pairs: *const rgmin_lbfgs_pairs) -> usize {
    if pairs.is_null() {
        return 0;
    }
    unsafe { (*pairs).pairs.len() }
}

/// Drop every stored pair. A null handle is a no-op.
///
/// # Safety
/// `pairs` is null or a live handle from [`rgmin_lbfgs_pairs_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_lbfgs_pairs_clear(pairs: *mut rgmin_lbfgs_pairs) {
    if !pairs.is_null() {
        unsafe { (*pairs).pairs.clear() };
    }
}

/// Release a handle from [`rgmin_lbfgs_pairs_create`]. Null is a no-op.
///
/// # Safety
/// `pairs` is null or a pointer from [`rgmin_lbfgs_pairs_create`] that
/// has not been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_lbfgs_pairs_free(pairs: *mut rgmin_lbfgs_pairs) {
    if !pairs.is_null() {
        drop(unsafe { Box::from_raw(pairs) });
    }
}

/// Direction `-H g`. Pass either `h0_diag` or `apply_h0`, not both.
/// An empty memory writes `-H0 g`. The call does not evaluate an
/// objective. ABI minor 38.
///
/// A diagonal whose entries are all `γ` is `H0 = γ I`. `apply_h0` is
/// called once, on the residual of the backward loop that starts from
/// `grad`; the direction is `-H g` when that map is linear. Pairs
/// stored by [`rgmin_lbfgs_pairs_push_rho`] enter with their own `rho`.
/// The result is [`LbfgsPairs::direction`] for the same pairs and `H0`.
///
/// # Safety
/// `pairs` is null or a live handle from [`rgmin_lbfgs_pairs_create`].
/// `grad` is null or points at `n` readable doubles, and `dir` is null
/// or points at `n` writable doubles. `h0_diag`, when non-null, points
/// at `n` readable doubles. `apply_h0`, when non-null, is callable with
/// `user` for the duration of this call and must not call back into
/// `pairs`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_lbfgs_direction(
    pairs: *const rgmin_lbfgs_pairs,
    grad: *const f64,
    h0_diag: *const f64,
    apply_h0: Option<rgmin_h0_fn>,
    user: *mut c_void,
    dir: *mut f64,
    n: usize,
) -> rgmin_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lbfgs_direction_c(pairs, grad, h0_diag, apply_h0, user, dir, n)
    })) {
        Ok(status) => status,
        Err(_) => {
            set_last_error("rgmin_lbfgs_direction: panic");
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

fn lbfgs_direction_c(
    pairs: *const rgmin_lbfgs_pairs,
    grad: *const f64,
    h0_diag: *const f64,
    apply_h0: Option<rgmin_h0_fn>,
    user: *mut c_void,
    dir: *mut f64,
    n: usize,
) -> rgmin_status_t {
    if pairs.is_null() || grad.is_null() || dir.is_null() {
        set_last_error("rgmin_lbfgs_direction: null argument");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    match (h0_diag.is_null(), apply_h0.is_none()) {
        (false, false) => {
            set_last_error("rgmin_lbfgs_direction: pass a diagonal or a callback");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        (true, true) => {
            set_last_error("rgmin_lbfgs_direction: H0 is missing");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        _ => {}
    }
    let grad = unsafe { slice::from_raw_parts(grad, n) };
    let outcome = if let Some(apply) = apply_h0 {
        let callback = |v: ArrayView1<'_, f64>| {
            let Some(input) = v.as_slice() else {
                return Err(Error::LbfgsDirection {
                    what: "H0 input is not contiguous",
                });
            };
            let mut out = vec![0.0; input.len()];
            let code = unsafe { apply(user, input.as_ptr(), out.as_mut_ptr(), input.len()) };
            if code != 0 {
                return Err(Error::LbfgsDirection {
                    what: "H0 callback failed",
                });
            }
            Ok(Array1::from_vec(out))
        };
        unsafe {
            (*pairs)
                .pairs
                .direction(ArrayView1::from(grad), LbfgsH0::Apply(&callback))
        }
    } else {
        let diag = unsafe { slice::from_raw_parts(h0_diag, n) };
        unsafe {
            (*pairs).pairs.direction(
                ArrayView1::from(grad),
                LbfgsH0::Diagonal(ArrayView1::from(diag)),
            )
        }
    };
    match outcome {
        Ok(direction) => {
            let out = unsafe { slice::from_raw_parts_mut(dir, n) };
            out.copy_from_slice(direction.as_slice().expect("contiguous"));
            rgmin_status_t::RGMIN_SUCCESS
        }
        Err(err) => status_from_error(&err),
    }
}

/// `rgmin_control_t.maxmove`: positive is the Euclidean cap, zero leaves
/// the caps at their defaults, and negative leaves both the Euclidean cap
/// and the default per-atom cap off.
fn maxmove_from_c(maxmove: f64) -> Option<f64> {
    if maxmove > 0.0 {
        Some(maxmove)
    } else if maxmove < 0.0 {
        Some(f64::INFINITY)
    } else {
        None
    }
}

fn noise_floor_from_c(c: &rgmin_control_t) -> Option<crate::NoiseFloor> {
    if !(c.noise_eps.is_finite() && c.noise_eps > 0.0) {
        return None;
    }
    let mut floor = crate::NoiseFloor::new(c.noise_eps);
    if c.noise_c.is_finite() && c.noise_c > 0.0 {
        floor = floor.with_c(c.noise_c);
    }
    Some(floor)
}

fn control_from_c(c: &rgmin_control_t) -> Control {
    Control {
        maxiter: c.maxiter,
        gtol: c.gtol,
        istep: if c.istep > 0.0 { c.istep } else { 1.0 },
        maxmove: maxmove_from_c(c.maxmove),
        stop_norm: StopNorm::Two,
        ftol_rel: None,
        noise_floor: noise_floor_from_c(c),
    }
}

fn stop_from_status(status: crate::Status) -> rgmin_stop_t {
    match status {
        crate::Status::Tolerance => rgmin_stop_t::RGMIN_STOP_TOLERANCE,
        crate::Status::NoiseFloor => rgmin_stop_t::RGMIN_STOP_NOISE_FLOOR,
        crate::Status::StepTolerance => rgmin_stop_t::RGMIN_STOP_STEP_TOLERANCE,
        crate::Status::Budget | crate::Status::Iterating => rgmin_stop_t::RGMIN_STOP_BUDGET,
    }
}

/// Retraction of a Stiefel handle from [`rgmin_manifold_stiefel`].
/// The C entry takes the integer. ABI minor 41.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_stiefel_retraction_t {
    /// Q factor of `X + V` with a positive diagonal R.
    RGMIN_STIEFEL_QR = 0,
    /// Polar factor of `X + V`.
    RGMIN_STIEFEL_POLAR = 1,
}

/// Metric of an SPD handle from [`rgmin_manifold_spd`].
/// The C entry takes the integer. ABI minor 41.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_spd_metric_t {
    /// `<U, V>_X = tr(X^-1 U X^-1 V)`.
    RGMIN_SPD_AFFINE_INVARIANT = 0,
    /// `<U, V>_X = <Dlog(X)[U], Dlog(X)[V]>`.
    RGMIN_SPD_LOG_EUCLIDEAN = 1,
}

/// Opaque Stiefel, SPD or product manifold. Every buffer it reads or
/// writes is caller-owned, column-major, of length
/// [`rgmin_manifold_len`]. ABI minor 41.
pub struct rgmin_manifold_handle {
    manifold: RiemannianManifold,
}

fn boxed_manifold(made: crate::Result<RiemannianManifold>) -> *mut rgmin_manifold_handle {
    match made {
        Ok(manifold) => Box::into_raw(Box::new(rgmin_manifold_handle { manifold })),
        Err(err) => {
            set_last_error(&err.to_string());
            std::ptr::null_mut()
        }
    }
}

/// \(\mathrm{St}(n, p)\), `1 <= p <= n`, packed column-major `n x p`.
/// `retraction` is an [`rgmin_stiefel_retraction_t`] value. Null on a bad
/// shape or an unknown retraction. ABI minor 41.
#[unsafe(no_mangle)]
pub extern "C" fn rgmin_manifold_stiefel(
    n: usize,
    p: usize,
    retraction: i32,
) -> *mut rgmin_manifold_handle {
    let retraction = match retraction {
        0 => StiefelRetraction::Qr,
        1 => StiefelRetraction::Polar,
        other => {
            set_last_error(&format!(
                "rgmin_manifold_stiefel: unknown retraction {other}"
            ));
            return std::ptr::null_mut();
        }
    };
    boxed_manifold(RiemannianManifold::stiefel(n, p, retraction))
}

/// \(\mathrm{SPD}(p)\), `p >= 1`, packed column-major `p x p`. `metric` is
/// an [`rgmin_spd_metric_t`] value. The retraction is the exponential of
/// the metric. Null on `p = 0` or an unknown metric. ABI minor 41.
#[unsafe(no_mangle)]
pub extern "C" fn rgmin_manifold_spd(p: usize, metric: i32) -> *mut rgmin_manifold_handle {
    let metric = match metric {
        0 => SpdMetric::AffineInvariant,
        1 => SpdMetric::LogEuclidean,
        other => {
            set_last_error(&format!("rgmin_manifold_spd: unknown metric {other}"));
            return std::ptr::null_mut();
        }
    };
    boxed_manifold(RiemannianManifold::spd(p, metric))
}

/// Product of `count` handles in order. The factors are copied; the caller
/// still frees each one. A factor that is itself a product contributes its
/// factors. Null on a null array, a null entry or `count = 0`.
/// ABI minor 41.
///
/// # Safety
/// `factors` is null or points at `count` pointers, each null or a live
/// handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_manifold_product(
    factors: *const *const rgmin_manifold_handle,
    count: usize,
) -> *mut rgmin_manifold_handle {
    if factors.is_null() || count == 0 {
        set_last_error("rgmin_manifold_product: need at least one factor");
        return std::ptr::null_mut();
    }
    let handles = unsafe { slice::from_raw_parts(factors, count) };
    let mut parts = Vec::with_capacity(count);
    for (i, h) in handles.iter().enumerate() {
        if h.is_null() {
            set_last_error(&format!("rgmin_manifold_product: factor {i} is null"));
            return std::ptr::null_mut();
        }
        parts.push(unsafe { (**h).manifold.clone() });
    }
    boxed_manifold(RiemannianManifold::product(parts))
}

/// Release a manifold handle. Null is a no-op. ABI minor 41.
///
/// # Safety
/// `manifold` is null or a handle that has not been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_manifold_free(manifold: *mut rgmin_manifold_handle) {
    if !manifold.is_null() {
        drop(unsafe { Box::from_raw(manifold) });
    }
}

/// Packed length of one point: `n p` for Stiefel, `p^2` for SPD, the sum
/// for a product. A null handle returns zero. ABI minor 41.
///
/// # Safety
/// `manifold` is null or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_manifold_len(manifold: *const rgmin_manifold_handle) -> usize {
    if manifold.is_null() {
        0
    } else {
        unsafe { (*manifold).manifold.packed_len() }
    }
}

type ManifoldOp<'a> =
    dyn FnOnce(&RiemannianManifold, &[Vec<f64>], &mut [Vec<f64>]) -> crate::Result<()> + 'a;

/// Copy the inputs, run `op`, and write the outputs only on success, so an
/// output may alias an input.
fn manifold_call(
    func: &str,
    manifold: *const rgmin_manifold_handle,
    len: usize,
    inputs: &[(&str, *const f64)],
    outputs: &[(&str, *mut f64)],
    op: Box<ManifoldOp<'_>>,
) -> rgmin_status_t {
    if manifold.is_null() {
        set_last_error(&format!("{func}: null manifold"));
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let m = unsafe { &(*manifold).manifold };
    if len != m.packed_len() {
        set_last_error(&format!(
            "{func}: len {len} is not the packed length {}",
            m.packed_len()
        ));
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let named = inputs
        .iter()
        .map(|(name, ptr)| (*name, ptr.is_null()))
        .chain(outputs.iter().map(|(name, ptr)| (*name, ptr.is_null())));
    for (name, null) in named {
        if null {
            set_last_error(&format!("{func}: null {name}"));
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
    }
    let copies: Vec<Vec<f64>> = inputs
        .iter()
        .map(|(_, ptr)| unsafe { slice::from_raw_parts(*ptr, len) }.to_vec())
        .collect();
    let mut outs = vec![vec![0.0; len]; outputs.len()];
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op(m, &copies, &mut outs))) {
        Ok(Ok(())) => {
            for ((_, ptr), values) in outputs.iter().zip(&outs) {
                unsafe { slice::from_raw_parts_mut(*ptr, len) }.copy_from_slice(values);
            }
            rgmin_status_t::RGMIN_SUCCESS
        }
        Ok(Err(err)) => {
            let status = status_from_error(&err);
            set_last_error(&format!("{func}: {err}"));
            status
        }
        Err(_) => {
            set_last_error(&format!("{func}: panic"));
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

/// Tangent projection of the ambient `v` at `x` into `out`. Stiefel:
/// `V - X sym(X^T V)`. SPD: `sym(V)`. ABI minor 41.
///
/// # Safety
/// `manifold` is null or a live handle. `x`, `v` and `out` are null or
/// point at `len` doubles; `out` is writable and may alias an input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_manifold_project(
    manifold: *const rgmin_manifold_handle,
    x: *const f64,
    v: *const f64,
    out: *mut f64,
    len: usize,
) -> rgmin_status_t {
    manifold_call(
        "rgmin_manifold_project",
        manifold,
        len,
        &[("x", x), ("v", v)],
        &[("out", out)],
        Box::new(|m, i, o| m.project(&i[0], &i[1], &mut o[0])),
    )
}

/// Riemannian gradient at `x` from the Euclidean gradient `egrad`.
/// Stiefel: the projection. SPD affine-invariant: `X sym(G) X`. SPD
/// log-Euclidean: `Dlog(X)^-2 [sym(G)]`. ABI minor 41.
///
/// # Safety
/// As [`rgmin_manifold_project`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_manifold_egrad_to_rgrad(
    manifold: *const rgmin_manifold_handle,
    x: *const f64,
    egrad: *const f64,
    out: *mut f64,
    len: usize,
) -> rgmin_status_t {
    manifold_call(
        "rgmin_manifold_egrad_to_rgrad",
        manifold,
        len,
        &[("x", x), ("egrad", egrad)],
        &[("out", out)],
        Box::new(|m, i, o| m.egrad_to_rgrad(&i[0], &i[1], &mut o[0])),
    )
}

/// Retraction of the step `v` at `x` into `out`. Stiefel uses the
/// handle's QR or polar retraction; SPD uses the exponential of its
/// metric. ABI minor 41.
///
/// # Safety
/// As [`rgmin_manifold_project`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_manifold_retract(
    manifold: *const rgmin_manifold_handle,
    x: *const f64,
    v: *const f64,
    out: *mut f64,
    len: usize,
) -> rgmin_status_t {
    manifold_call(
        "rgmin_manifold_retract",
        manifold,
        len,
        &[("x", x), ("v", v)],
        &[("out", out)],
        Box::new(|m, i, o| m.retract(&i[0], &i[1], &mut o[0])),
    )
}

/// Riemannian exponential at `x`; `v` is projected to the tangent space
/// first. ABI minor 41.
///
/// # Safety
/// As [`rgmin_manifold_project`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_manifold_exp(
    manifold: *const rgmin_manifold_handle,
    x: *const f64,
    v: *const f64,
    out: *mut f64,
    len: usize,
) -> rgmin_status_t {
    manifold_call(
        "rgmin_manifold_exp",
        manifold,
        len,
        &[("x", x), ("v", v)],
        &[("out", out)],
        Box::new(|m, i, o| m.exp(&i[0], &i[1], &mut o[0])),
    )
}

/// Riemannian logarithm of `y` at `x`. A handle with a Stiefel factor
/// returns `RGMIN_INVALID_PARAMETER`. ABI minor 41.
///
/// # Safety
/// As [`rgmin_manifold_project`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_manifold_log(
    manifold: *const rgmin_manifold_handle,
    x: *const f64,
    y: *const f64,
    out: *mut f64,
    len: usize,
) -> rgmin_status_t {
    manifold_call(
        "rgmin_manifold_log",
        manifold,
        len,
        &[("x", x), ("y", y)],
        &[("out", out)],
        Box::new(|m, i, o| m.log(&i[0], &i[1], &mut o[0])),
    )
}

/// Vector transport of `v` from `x_from` to `x_to`. Stiefel: projection
/// at `x_to`. SPD: parallel transport of the metric. ABI minor 41.
///
/// # Safety
/// As [`rgmin_manifold_project`], with `x_from` and `x_to` in place of `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_manifold_transport(
    manifold: *const rgmin_manifold_handle,
    x_from: *const f64,
    x_to: *const f64,
    v: *const f64,
    out: *mut f64,
    len: usize,
) -> rgmin_status_t {
    manifold_call(
        "rgmin_manifold_transport",
        manifold,
        len,
        &[("x_from", x_from), ("x_to", x_to), ("v", v)],
        &[("out", out)],
        Box::new(|m, i, o| m.transport(&i[0], &i[1], &i[2], &mut o[0])),
    )
}

/// Riemannian inner product of the tangents `u` and `v` at `x`, written to
/// `*out`. ABI minor 41.
///
/// # Safety
/// `manifold` is null or a live handle. `x`, `u` and `v` are null or point
/// at `len` readable doubles. `out` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_manifold_inner(
    manifold: *const rgmin_manifold_handle,
    x: *const f64,
    u: *const f64,
    v: *const f64,
    out: *mut f64,
    len: usize,
) -> rgmin_status_t {
    if out.is_null() {
        set_last_error("rgmin_manifold_inner: null out");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let mut value = 0.0;
    let status = manifold_call(
        "rgmin_manifold_inner",
        manifold,
        len,
        &[("x", x), ("u", u), ("v", v)],
        &[],
        Box::new(|m, i, _| {
            value = m.inner(&i[0], &i[1], &i[2])?;
            Ok(())
        }),
    );
    if status == rgmin_status_t::RGMIN_SUCCESS {
        unsafe { *out = value };
    }
    status
}

/// One Riemannian gradient step: `rgrad_out` is the Riemannian gradient of
/// `egrad` at `x`, and `x_out` is the retraction of `-t rgrad_out` at `x`.
/// `t` must be finite. ABI minor 41.
///
/// # Safety
/// `manifold` is null or a live handle. `x` and `egrad` are null or point
/// at `len` readable doubles. `rgrad_out` and `x_out` are null or point at
/// `len` writable doubles and may alias the inputs, not each other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_manifold_gradient_step(
    manifold: *const rgmin_manifold_handle,
    x: *const f64,
    egrad: *const f64,
    t: f64,
    rgrad_out: *mut f64,
    x_out: *mut f64,
    len: usize,
) -> rgmin_status_t {
    manifold_call(
        "rgmin_manifold_gradient_step",
        manifold,
        len,
        &[("x", x), ("egrad", egrad)],
        &[("rgrad_out", rgrad_out), ("x_out", x_out)],
        Box::new(move |m, i, o| {
            let (rgrad, rest) = o.split_at_mut(1);
            m.gradient_step(&i[0], &i[1], t, &mut rgrad[0], &mut rest[0])
        }),
    )
}

#[cfg(test)]
mod device_tests {
    use super::*;

    #[test]
    fn cuda_tag_is_unsupported() {
        let mut buf = [0.0_f64; 2];
        let t = unsafe { create_borrowed_f64_1d(buf.as_mut_ptr(), 2, DLDeviceType::kDLCUDA, 0) };
        let err = cpu_f64_slice(t, "cuda").unwrap_err();
        unsafe { rgmin_tensor_free(t) };
        assert_eq!(err, rgmin_status_t::RGMIN_UNSUPPORTED_DEVICE);
    }
}

#[cfg(test)]
mod conjugacy_abi_tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn scg_params_layout_is_five_doubles_then_i32() {
        assert_eq!(size_of::<rgmin_conjugacy_t>(), 4);
        assert_eq!(size_of::<rgmin_highs_solver_t>(), 4);
        assert_eq!(size_of::<rgmin_highs_crossover_t>(), 4);
        assert_eq!(size_of::<rgmin_scg_params_t>(), 48);
        assert_eq!(offset_of!(rgmin_scg_params_t, sigma0), 0);
        assert_eq!(offset_of!(rgmin_scg_params_t, lambda), 8);
        assert_eq!(offset_of!(rgmin_scg_params_t, lambda_limit), 16);
        assert_eq!(offset_of!(rgmin_scg_params_t, tol_sol), 24);
        assert_eq!(offset_of!(rgmin_scg_params_t, tol_func), 32);
        assert_eq!(offset_of!(rgmin_scg_params_t, conjugacy), 40);
    }

    #[test]
    fn eigenpair_ex_extends_the_minor_39_record() {
        assert_eq!(offset_of!(rgmin_eigenpair_t, value), 0);
        assert_eq!(offset_of!(rgmin_eigenpair_t, actions), 8);
        assert_eq!(size_of::<rgmin_eigenpair_t>(), 16);
        assert_eq!(offset_of!(rgmin_eigenpair_ex_t, value), 0);
        assert_eq!(offset_of!(rgmin_eigenpair_ex_t, actions), 8);
        assert_eq!(offset_of!(rgmin_eigenpair_ex_t, residual), 16);
        assert_eq!(offset_of!(rgmin_eigenpair_ex_t, converged), 24);
        assert_eq!(size_of::<rgmin_eigenpair_ex_t>(), 32);
    }

    #[test]
    fn eigen_params_layout_is_i32_then_three_u32_then_f64() {
        assert_eq!(size_of::<rgmin_eigen_kind_t>(), 4);
        assert_eq!(size_of::<rgmin_eigen_params_t>(), 24);
        assert_eq!(offset_of!(rgmin_eigen_params_t, kind), 0);
        assert_eq!(offset_of!(rgmin_eigen_params_t, nev), 4);
        assert_eq!(offset_of!(rgmin_eigen_params_t, krylov), 8);
        assert_eq!(offset_of!(rgmin_eigen_params_t, max_iter), 12);
        assert_eq!(offset_of!(rgmin_eigen_params_t, tol), 16);
        assert_eq!(rgmin_eigen_kind_t::RGMIN_EIGEN_LANCZOS as i32, 0);
        assert_eq!(rgmin_eigen_kind_t::RGMIN_EIGEN_EIGENEXA as i32, 13);
        assert_eq!(rgmin_eigen_kind_t::RGMIN_EIGEN_DIMER as i32, 14);
    }

    #[test]
    fn conjugacy_from_c_is_dest_leaf_order() {
        assert_eq!(conjugacy_from_c(0).unwrap(), Conjugacy::FletcherReeves);
        assert_eq!(conjugacy_from_c(1).unwrap(), Conjugacy::PolakRibiere);
        assert_eq!(conjugacy_from_c(2).unwrap(), Conjugacy::HestenesStiefel);
        assert_eq!(conjugacy_from_c(3).unwrap(), Conjugacy::DaiYuan);
        assert_eq!(conjugacy_from_c(4).unwrap(), Conjugacy::ConjugateDescent);
        assert_eq!(conjugacy_from_c(5).unwrap(), Conjugacy::HagerZhang);
        assert_eq!(conjugacy_from_c(6).unwrap(), Conjugacy::LiuStorey);
        assert_eq!(conjugacy_from_c(7).unwrap(), Conjugacy::FrPr);
        assert_eq!(conjugacy_from_c(8).unwrap(), Conjugacy::PolakRibierePlus);
        assert_ne!(conjugacy_from_c(7).unwrap(), {
            Conjugacy::hybrid(Conjugacy::FletcherReeves, Conjugacy::PolakRibiere, true)
        });
    }

    #[test]
    fn conjugacy_from_c_rejects_unknown_and_method_t_codes() {
        for raw in [-1, 9, 99, 13] {
            assert_eq!(
                conjugacy_from_c(raw),
                Err(rgmin_status_t::RGMIN_INVALID_PARAMETER)
            );
            let msg = unsafe { std::ffi::CStr::from_ptr(rgmin_last_error()) };
            assert!(
                msg.to_string_lossy().contains("conjugacy"),
                "last_error for {raw}: {msg:?}"
            );
        }
    }
}

// --- 0.5.2 surface joined at ABI minor 40. Minor 39 is rgmin_lowest_eigenpairs. ---

pub const RGMIN_PRECON_NONE: u32 = 0;
/// Packwood exponential metric.
pub const RGMIN_PRECON_EXP: u32 = 1;
/// Lindh model Hessian.
pub const RGMIN_PRECON_LINDH: u32 = 2;
/// Bonded metric.
pub const RGMIN_PRECON_FF: u32 = 3;
/// [`crate::select_precon`].
pub const RGMIN_PRECON_AUTO: u32 = 4;

/// Periodic bulk, for [`rgmin_precon_t::geometry_kind`].
pub const RGMIN_GEOM_BULK: i32 = 0;
/// Slab or surface.
pub const RGMIN_GEOM_SURFACE: i32 = 1;
/// Isolated molecule.
pub const RGMIN_GEOM_MOLECULE: i32 = 2;
/// Finite cluster.
pub const RGMIN_GEOM_CLUSTER: i32 = 3;

/// One description of every geometry preconditioner. ABI minor 32.
///
/// `struct_size` is `sizeof` on the caller side. A shorter value is
/// rejected. Bytes past the fields this build knows are ignored.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct rgmin_precon_t {
    /// `sizeof(rgmin_precon_t)` as the caller compiled it.
    pub struct_size: u32,
    /// [`RGMIN_PRECON_NONE`] through [`RGMIN_PRECON_AUTO`].
    pub kind: u32,
    /// Diagonal stabilization. NaN or <= 0 selects the default.
    pub c_stab: f64,
    /// Energy scale. NaN or <= 0 estimates it.
    pub mu: f64,
    /// Cutoff. NaN or <= 0 leaves the cutoff open.
    pub r_cut: f64,
    /// Orthorhombic lengths. A component <= 0 leaves that axis open.
    pub cell: [f64; 3],
    /// Atomic numbers, length `natoms`. Lindh and AUTO.
    pub numbers: *const i32,
    /// Free atoms. Lindh and AUTO require `3 * natoms == dim`.
    pub natoms: usize,
    /// Ghost atomic numbers, length `nghost`.
    pub ghost_z: *const i32,
    /// Ghost coordinates, length `3 * nghost`.
    pub ghost_pos: *const f64,
    /// Frozen atoms.
    pub nghost: usize,
    /// Exp neighbour index, length `npairs`.
    pub nbr_i: *const usize,
    /// Exp neighbour index, length `npairs`.
    pub nbr_j: *const usize,
    /// Exp pair distance. Null takes Cartesian distances on the next step.
    pub nbr_r: *const f64,
    /// Exp pairs.
    pub npairs: usize,
    /// Free atom bonded to a frozen atom.
    pub anchor_i: *const usize,
    /// Distance of that bond.
    pub anchor_r: *const f64,
    /// Anchor bonds.
    pub nanchor: usize,
    /// Exp exponent A. NaN or <= 0 selects 3.
    pub exp_a: f64,
    /// Harmonic bonds.
    pub bonds: *const rgmin_bond_t,
    /// Bond count.
    pub nbonds: usize,
    /// Harmonic angles.
    pub angles: *const rgmin_angle_t,
    /// Angle count.
    pub nangles: usize,
    /// Nonzero adds the Lindh torsion term.
    pub dihedrals: i32,
    /// [`RGMIN_GEOM_BULK`] and the three companions. AUTO only.
    pub geometry_kind: i32,
}

/// Optional SCG policies, separate from the stable parameter record.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct rgmin_scg_options_t {
    /// Literal [`rgmin_scg_step_tolerance_t`]. Unknown values are rejected.
    pub step_tolerance: i32,
}

/// Accepted-step SCG convergence rule.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_scg_step_tolerance_t {
    RGMIN_SCG_ABSOLUTE_INFINITY = 0,
    RGMIN_SCG_RELATIVE_EUCLIDEAN = 1,
}

/// Optional backend settings, separate from the stable base parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct rgmin_eigen_options_t {
    /// ChASE filter degree. Zero selects the backend default.
    pub degree: u32,
    /// ChASE extra search dimensions. Zero selects the backend default.
    pub extra: u32,
}

/// Lowest Hessian eigenpair with separate optional backend settings.
/// Matrix-free backends ignore the ChASE degree and extra dimensions.
/// A null options pointer selects the backend defaults.
///
/// # Safety
/// The tensor and callback requirements are those of [`rgmin_lowest_eigenpair`].
/// Non-null parameter and options pointers refer to their declared C records.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_lowest_eigenpair_with_options(
    hvp: Option<rgmin_hvp_fn>,
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    seed: *const DLManagedTensorVersioned,
    mode_out: *mut DLManagedTensorVersioned,
    params: *const rgmin_eigen_params_t,
    options: *const rgmin_eigen_options_t,
    out: *mut rgmin_lowest_mode_t,
) -> rgmin_status_t {
    let Some(hvp) = hvp else {
        set_last_error("rgmin_lowest_eigenpair: null hvp");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    };
    if out.is_null() {
        set_last_error("rgmin_lowest_eigenpair: null out");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let xs = match cpu_f64_slice(x, "x") {
        Ok(s) => s,
        Err(st) => return st,
    };
    let seed_s = match cpu_f64_slice(seed, "seed") {
        Ok(s) => s,
        Err(st) => return st,
    };
    let x_arr = Array1::from(xs.to_vec());
    let seed_arr = Array1::from(seed_s.to_vec());
    let mode_s = match cpu_f64_slice_mut(mode_out, "mode_out") {
        Ok(s) => s,
        Err(st) => return st,
    };
    if x_arr.len() != seed_arr.len() || x_arr.len() != mode_s.len() {
        set_last_error("rgmin_lowest_eigenpair: x/seed/mode length mismatch");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let mut typed = match eigen_params_from_c(params) {
        Ok(p) => p,
        Err(st) => return st,
    };
    if let Some(options) = unsafe { options.as_ref() } {
        typed.degree = options.degree as usize;
        typed.extra = options.extra as usize;
    }
    let apply = CHvp { hvp, user };
    match lowest_mode(&apply, x_arr.view(), seed_arr.view(), &typed) {
        Ok(mode) => {
            mode_s.copy_from_slice(mode.vector.as_slice().unwrap_or(&[]));
            unsafe {
                *out = rgmin_lowest_mode_t {
                    value: mode.value,
                    actions: mode.actions,
                };
            }
            rgmin_status_t::RGMIN_SUCCESS
        }
        Err(e) => status_from_error(&e),
    }
}

/// Møller SCG with an explicit accepted-step convergence rule.
/// A null options pointer selects the absolute infinity-norm rule.
///
/// # Safety
/// The tensor and callback requirements are those of [`rgmin_minimize_scg`].
/// A non-null options pointer refers to an initialized C options record.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_minimize_scg_with_options(
    eval: Option<rgmin_eval_fn>,
    grad: Option<rgmin_grad_fn>,
    curv: Option<rgmin_curv_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    ctrl: *const rgmin_control_t,
    params: *const rgmin_scg_params_t,
    options: *const rgmin_scg_options_t,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eval = match eval {
            Some(f) => f,
            None => {
                set_last_error("rgmin_minimize_scg: eval is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let grad = match grad {
            Some(f) => f,
            None => {
                set_last_error("rgmin_minimize_scg: grad is NULL");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        if ctrl.is_null() || out.is_null() {
            set_last_error("rgmin_minimize_scg: ctrl/out null");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let n = init.len();
        let c = unsafe { &*ctrl };
        let control = control_from_c(c);
        let (scg, conjugacy) = if params.is_null() {
            (ScgParams::default(), Conjugacy::PolakRibiere)
        } else {
            let p = unsafe { &*params };
            match conjugacy_from_c(p.conjugacy) {
                Ok(c) => (
                    ScgParams {
                        sigma0: p.sigma0,
                        lambda: p.lambda,
                        lambda_limit: p.lambda_limit,
                        tol_sol: p.tol_sol,
                        tol_func: p.tol_func,
                    },
                    c,
                ),
                Err(st) => return st,
            }
        };
        let step_tolerance = match unsafe { options.as_ref() }.map_or(0, |o| o.step_tolerance) {
            0 => ScgStepTolerance::AbsoluteInfinity,
            1 => ScgStepTolerance::RelativeEuclidean,
            other => {
                set_last_error(&format!(
                    "rgmin_minimize_scg: unknown step tolerance {other}"
                ));
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
        };
        let options = ScgOptions { step_tolerance };
        let restart = Restart::Never;
        let obj = ScgFfiOracle {
            inner: c_oracle(eval, grad, user, n),
            curv: curv.map(|f| f as usize),
            user: user as usize,
            scratch: Scratch::new(),
        };
        let report = if obj.curv.is_some() {
            minimize_scg_exact_with_options(
                &obj,
                Array1::from(init),
                &control,
                &scg,
                conjugacy,
                restart,
                &options,
            )
        } else {
            minimize_scg_with_options(
                &obj.inner,
                Array1::from(init),
                &control,
                &scg,
                conjugacy,
                restart,
                &options,
            )
        };
        match report {
            Ok(rep) => {
                let dest = match cpu_f64_slice_mut(x, "x") {
                    Ok(s) => s,
                    Err(st) => return st,
                };
                dest.copy_from_slice(rep.coords.as_slice().expect("contiguous"));
                unsafe {
                    *out = rgmin_report_t {
                        value: rep.value,
                        steps: rep.steps,
                        grad_norm: rep.grad_norm,
                        stop: stop_from_status(rep.status),
                    };
                }
                rgmin_status_t::RGMIN_SUCCESS
            }
            Err(e) => status_from_error(&e),
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("rgmin_minimize_scg: panic");
            rgmin_status_t::RGMIN_INTERNAL_ERROR
        }
    }
}

/// Geometry preconditioner. ABI minor 32.
///
/// Field names that already existed on the old exponential parameters
/// keep those names: `c_stab`, `mu`, `r_cut`. `exp_a` is the old `a`.
/// `cell` replaces the three lattice lengths.
///
/// Morse bonds and FF harmonic torsions are not on this struct. They
/// stay on [`crate::PreconSpec::Ff`]. Lindh's torsion term is
/// `dihedrals`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct rgmin_bond_t {
    /// Free-atom index.
    pub i: u32,
    /// Free-atom index.
    pub j: u32,
    /// Harmonic curvature.
    pub k: f64,
}

/// One harmonic angle. `vertex` is the centre atom.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct rgmin_angle_t {
    /// Free-atom index.
    pub i: u32,
    /// Free-atom index of the vertex.
    pub vertex: u32,
    /// Free-atom index.
    pub k: u32,
    /// Harmonic angle curvature.
    pub k_theta: f64,
}

fn precon_blank(kind: u32) -> rgmin_precon_t {
    rgmin_precon_t {
        struct_size: std::mem::size_of::<rgmin_precon_t>() as u32,
        kind,
        c_stab: f64::NAN,
        mu: f64::NAN,
        r_cut: f64::NAN,
        cell: [0.0, 0.0, 0.0],
        numbers: std::ptr::null(),
        natoms: 0,
        ghost_z: std::ptr::null(),
        ghost_pos: std::ptr::null(),
        nghost: 0,
        nbr_i: std::ptr::null(),
        nbr_j: std::ptr::null(),
        nbr_r: std::ptr::null(),
        npairs: 0,
        anchor_i: std::ptr::null(),
        anchor_r: std::ptr::null(),
        nanchor: 0,
        exp_a: f64::NAN,
        bonds: std::ptr::null(),
        nbonds: 0,
        angles: std::ptr::null(),
        nangles: 0,
        dihedrals: 0,
        geometry_kind: 0,
    }
}

/// Fill `p` with defaults for `kind`.
///
/// # Safety
///
/// `p` is null or valid for a write of [`rgmin_precon_t`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_precon_init(p: *mut rgmin_precon_t, kind: u32) {
    if p.is_null() {
        return;
    }
    unsafe { p.write(precon_blank(kind)) };
}

fn load_precon(p: *const rgmin_precon_t) -> std::result::Result<rgmin_precon_t, &'static str> {
    if p.is_null() {
        return Err("rgmin_solver_set_precon: null params");
    }
    let got = unsafe { (*p).struct_size } as usize;
    let full = std::mem::size_of::<rgmin_precon_t>();
    if got < full {
        return Err("rgmin_solver_set_precon: struct_size is too small");
    }
    let mut local = precon_blank(RGMIN_PRECON_NONE);
    let n = full;
    unsafe {
        std::ptr::copy_nonoverlapping(
            p.cast::<u8>(),
            (&mut local as *mut rgmin_precon_t).cast::<u8>(),
            n,
        );
    }
    Ok(local)
}

/// Install the geometry preconditioner described by `params`.
///
/// Every neighbour, anchor, bond, and angle index is checked against
/// `dim / 3` before the session stores the list. A bad index does not
/// wait for the next step.
///
/// # Safety
///
/// `solver` is null or a live session. `params` is null or readable
/// for `params->struct_size` bytes, and every pointer it holds is null
/// or valid for the count beside it.
#[unsafe(no_mangle)]
fn install_auto(solver: *mut rgmin_solver_t, p: &rgmin_precon_t, atoms: usize) -> rgmin_status_t {
    if p.numbers.is_null() || p.natoms != atoms {
        set_last_error("rgmin_solver_set_precon: AUTO needs one atomic number per free atom");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let z = unsafe { slice::from_raw_parts(p.numbers, p.natoms) };
    let kind = match p.geometry_kind {
        RGMIN_GEOM_BULK => crate::GeometryKind::Bulk,
        RGMIN_GEOM_SURFACE => crate::GeometryKind::Surface,
        RGMIN_GEOM_MOLECULE => crate::GeometryKind::Molecule,
        RGMIN_GEOM_CLUSTER => crate::GeometryKind::Cluster,
        _ => {
            set_last_error("rgmin_solver_set_precon: unknown geometry_kind");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
    };
    let has_ff = p.nbonds > 0 || p.nangles > 0;
    let family = unsafe { (*solver).solver.optimizer_family() };
    match crate::select_precon(kind, z, has_ff, family) {
        crate::PreconChoice::Identity => {
            unsafe { (*solver).solver.clear_precon() };
            rgmin_status_t::RGMIN_SUCCESS
        }
        crate::PreconChoice::Exp => {
            if p.npairs == 0 && p.nanchor == 0 {
                set_last_error(
                    "rgmin_solver_set_precon: AUTO selected Exp and the neighbour list is empty",
                );
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
            install_exp(solver, p, atoms)
        }
        crate::PreconChoice::Lindh => install_lindh(solver, p, atoms),
        crate::PreconChoice::Ff => install_ff(solver, p, atoms),
    }
}

fn install_exp(solver: *mut rgmin_solver_t, p: &rgmin_precon_t, atoms: usize) -> rgmin_status_t {
    if p.npairs > 0 && (p.nbr_i.is_null() || p.nbr_j.is_null()) {
        set_last_error("rgmin_solver_set_precon: null neighbour list");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    if p.nanchor > 0 && (p.anchor_i.is_null() || p.anchor_r.is_null()) {
        set_last_error("rgmin_solver_set_precon: null anchor list");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    if p.natoms != 0 && p.natoms != atoms {
        set_last_error("rgmin_solver_set_precon: natoms is not dim / 3");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let i = if p.npairs == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(p.nbr_i, p.npairs) }.to_vec()
    };
    let j = if p.npairs == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(p.nbr_j, p.npairs) }.to_vec()
    };
    for (&a, &b) in i.iter().zip(&j) {
        if !index_ok(a, atoms) || !index_ok(b, atoms) {
            set_last_error("rgmin_solver_set_precon: neighbour index is out of range");
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
    }
    let r = if p.nbr_r.is_null() || p.npairs == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(p.nbr_r, p.npairs) }.to_vec()
    };
    let mut anchor = Vec::with_capacity(p.nanchor);
    if p.nanchor > 0 {
        let ai = unsafe { slice::from_raw_parts(p.anchor_i, p.nanchor) };
        let ar = unsafe { slice::from_raw_parts(p.anchor_r, p.nanchor) };
        for k in 0..p.nanchor {
            if !index_ok(ai[k], atoms) {
                set_last_error("rgmin_solver_set_precon: anchor index is out of range");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
            anchor.push((ai[k], ar[k]));
        }
    }
    let neighbours = crate::Neighbours { i, j, r, anchor };
    let mut params = ExpParams::default();
    if let Some(a) = optional_positive(p.exp_a) {
        params.a = a;
    }
    if let Some(c) = optional_positive(p.c_stab) {
        params.c_stab = c;
    }
    params.mu = optional_positive(p.mu);
    params.r_cut = optional_positive(p.r_cut);
    params.lengths = cell_of(p.cell);
    if neighbours.r.is_empty() {
        match crate::precon::ExpState::deferred(neighbours, params) {
            Ok(state) => {
                unsafe { (*solver).solver.set_exp_deferred(state) };
                rgmin_status_t::RGMIN_SUCCESS
            }
            Err(e) => {
                set_last_error(&e.to_string());
                rgmin_status_t::RGMIN_INVALID_PARAMETER
            }
        }
    } else {
        let positions = vec![0.0; atoms * 3];
        match unsafe {
            (*solver)
                .solver
                .set_exp(ndarray::ArrayView1::from(&positions), &neighbours, &params)
        } {
            Ok(()) => rgmin_status_t::RGMIN_SUCCESS,
            Err(e) => {
                set_last_error(&e.to_string());
                rgmin_status_t::RGMIN_INVALID_PARAMETER
            }
        }
    }
}

fn install_lindh(solver: *mut rgmin_solver_t, p: &rgmin_precon_t, atoms: usize) -> rgmin_status_t {
    if p.numbers.is_null() || p.natoms == 0 {
        set_last_error("rgmin_solver_set_precon: null atomic numbers");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    if p.natoms != atoms {
        set_last_error("rgmin_solver_set_precon: dimension is not 3 * natoms");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    if p.nghost > 0 && (p.ghost_z.is_null() || p.ghost_pos.is_null()) {
        set_last_error("rgmin_solver_set_precon: null ghost list");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let z = unsafe { slice::from_raw_parts(p.numbers, p.natoms) }.to_vec();
    let mut ghosts = Vec::with_capacity(p.nghost);
    if p.nghost > 0 {
        let gz = unsafe { slice::from_raw_parts(p.ghost_z, p.nghost) };
        let gp = unsafe { slice::from_raw_parts(p.ghost_pos, 3 * p.nghost) };
        for i in 0..p.nghost {
            ghosts.push(crate::Ghost {
                z: gz[i],
                position: [gp[3 * i], gp[3 * i + 1], gp[3 * i + 2]],
            });
        }
    }
    let mut params = crate::LindhParams::default();
    if let Some(c) = optional_positive(p.c_stab) {
        params.c_stab = c;
    }
    params.r_cut = optional_positive(p.r_cut);
    params.mu = optional_positive(p.mu);
    params.lengths = cell_of(p.cell);
    params.dihedrals = p.dihedrals != 0;
    match crate::precon::LindhState::deferred(z, ghosts, params) {
        Ok(state) => {
            unsafe { (*solver).solver.set_lindh_deferred(state) };
            rgmin_status_t::RGMIN_SUCCESS
        }
        Err(e) => {
            set_last_error(&e.to_string());
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        }
    }
}

fn install_ff(solver: *mut rgmin_solver_t, p: &rgmin_precon_t, atoms: usize) -> rgmin_status_t {
    if p.nbonds > 0 && p.bonds.is_null() {
        set_last_error("rgmin_solver_set_precon: null bond list");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    if p.nangles > 0 && p.angles.is_null() {
        set_last_error("rgmin_solver_set_precon: null angle list");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let mut bonds = Vec::with_capacity(p.nbonds);
    if p.nbonds > 0 {
        let raw = unsafe { slice::from_raw_parts(p.bonds, p.nbonds) };
        for b in raw {
            if !index_ok(b.i as usize, atoms) || !index_ok(b.j as usize, atoms) {
                set_last_error("rgmin_solver_set_precon: bond index is out of range");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
            bonds.push(crate::BondTerm {
                i: b.i as usize,
                j: b.j as usize,
                k: b.k,
            });
        }
    }
    let mut angles = Vec::with_capacity(p.nangles);
    if p.nangles > 0 {
        let raw = unsafe { slice::from_raw_parts(p.angles, p.nangles) };
        for a in raw {
            if !index_ok(a.i as usize, atoms)
                || !index_ok(a.vertex as usize, atoms)
                || !index_ok(a.k as usize, atoms)
            {
                set_last_error("rgmin_solver_set_precon: angle index is out of range");
                return rgmin_status_t::RGMIN_INVALID_PARAMETER;
            }
            angles.push(crate::AngleTerm {
                i: a.i as usize,
                vertex: a.vertex as usize,
                k: a.k as usize,
                k_theta: a.k_theta,
            });
        }
    }
    let mut params = crate::FfParams::default();
    if let Some(c) = optional_positive(p.c_stab) {
        params.c_stab = c;
    }
    params.r_cut = optional_positive(p.r_cut);
    params.mu = optional_positive(p.mu);
    params.lengths = cell_of(p.cell);
    match crate::precon::FfState::deferred(bonds, angles, params) {
        Ok(state) => {
            unsafe { (*solver).solver.set_ff_deferred(state) };
            rgmin_status_t::RGMIN_SUCCESS
        }
        Err(e) => {
            set_last_error(&e.to_string());
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        }
    }
}

fn rigid_masses<'a>(
    masses: *const f64,
    n_mass: usize,
    n: usize,
) -> Result<&'a [f64], rgmin_status_t> {
    if masses.is_null() || n_mass == 0 {
        Ok(&[])
    } else if n_mass != n / 3 {
        Err(rgmin_status_t::RGMIN_INVALID_PARAMETER)
    } else {
        Ok(unsafe { slice::from_raw_parts(masses, n_mass) })
    }
}

/// Install a geometry preconditioner.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
/// `params` is null or a `rgmin_precon_t` the caller initialized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_precon(
    solver: *mut rgmin_solver_t,
    params: *const rgmin_precon_t,
) -> rgmin_status_t {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_precon: null solver");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let p = match load_precon(params) {
        Ok(p) => p,
        Err(e) => {
            set_last_error(e);
            return rgmin_status_t::RGMIN_INVALID_PARAMETER;
        }
    };
    let dim = unsafe { (*solver).solver.dim() };
    if dim % 3 != 0 || dim == 0 {
        set_last_error("rgmin_solver_set_precon: dimension is not a positive multiple of 3");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let atoms = dim / 3;
    match p.kind {
        RGMIN_PRECON_NONE => {
            unsafe { (*solver).solver.clear_precon() };
            rgmin_status_t::RGMIN_SUCCESS
        }
        RGMIN_PRECON_EXP => install_exp(solver, &p, atoms),
        RGMIN_PRECON_LINDH => install_lindh(solver, &p, atoms),
        RGMIN_PRECON_FF => install_ff(solver, &p, atoms),
        RGMIN_PRECON_AUTO => install_auto(solver, &p, atoms),
        _ => {
            set_last_error("rgmin_solver_set_precon: unknown kind");
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        }
    }
}

/// Drop whichever geometry preconditioner is installed.
///
/// # Safety
///
/// `solver` is null or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_clear_precon(solver: *mut rgmin_solver_t) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.clear_precon() };
}

/// Energy scale of the installed geometry preconditioner, or NaN.
///
/// # Safety
///
/// `solver` is null or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_precon_scale(solver: *const rgmin_solver_t) -> f64 {
    if solver.is_null() {
        return f64::NAN;
    }
    unsafe { (*solver).solver.precon_scale().unwrap_or(f64::NAN) }
}

/// FIRE schedule of a FIRE session (see [`crate::FireVariant`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_fire_variant_t {
    /// The session's FIRE kind with rgmin's parameters.
    RGMIN_FIRE_RGMIN = 0,
    /// FIRE 2.0 as Guénolé et al. 2020 algorithm 2 publishes it.
    RGMIN_FIRE_GUENOLE2020 = 1,
}

/// Select the FIRE schedule; a no-op on a non-FIRE session.
///
/// # Safety
///
/// `solver` must be null or a live handle from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_fire_variant(
    solver: *mut rgmin_solver_t,
    variant: rgmin_fire_variant_t,
) {
    if solver.is_null() {
        return;
    }
    let v = match variant {
        rgmin_fire_variant_t::RGMIN_FIRE_RGMIN => crate::FireVariant::Rgmin,
        rgmin_fire_variant_t::RGMIN_FIRE_GUENOLE2020 => crate::FireVariant::Guenole2020,
    };
    unsafe { (*solver).solver.set_fire_variant(v) };
}

/// Split a quick-min session into atoms `[0, at)` and cell `[at, n)`.
///
/// # Safety
///
/// `solver` must be null or a live handle from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_quickmin_cell(
    solver: *mut rgmin_solver_t,
    at: usize,
) -> rgmin_status_t {
    if solver.is_null() {
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    match unsafe { (*solver).solver.set_quickmin_cell(at) } {
        Ok(()) => rgmin_status_t::RGMIN_SUCCESS,
        Err(_) => rgmin_status_t::RGMIN_INVALID_PARAMETER,
    }
}

/// Quick-min options of a quick-min session (see [`crate::QuickMinOptions`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rgmin_quickmin_variant_t {
    /// [`crate::QuickMinOptions::rgmin`]: FIRE time step, clamp rescale.
    RGMIN_QUICKMIN_RGMIN = 0,
    /// [`crate::QuickMinOptions::eon`]: fixed time step, no rescale.
    RGMIN_QUICKMIN_EON = 1,
    /// [`crate::QuickMinOptions::per_atom`]: one projection per atom.
    RGMIN_QUICKMIN_PER_ATOM = 2,
}

/// Select the quick-min options; a no-op on any other session. The
/// velocity and the time step restart; the cell split stays.
///
/// # Safety
///
/// `solver` must be null or a live handle from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_quickmin_variant(
    solver: *mut rgmin_solver_t,
    variant: rgmin_quickmin_variant_t,
) {
    if solver.is_null() {
        return;
    }
    let options = match variant {
        rgmin_quickmin_variant_t::RGMIN_QUICKMIN_RGMIN => crate::QuickMinOptions::rgmin(),
        rgmin_quickmin_variant_t::RGMIN_QUICKMIN_EON => crate::QuickMinOptions::eon(),
        rgmin_quickmin_variant_t::RGMIN_QUICKMIN_PER_ATOM => crate::QuickMinOptions::per_atom(),
    };
    unsafe { (*solver).solver.set_quickmin_options(options) };
}

/// Set the session's line search (see [`crate::Solver::set_linesearch`]).
///
/// Returns `RGMIN_INVALID_PARAMETER` and leaves the session unchanged
/// when `maxiter` is zero or the constants leave their ranges: Wolfe
/// needs `0 < c1 < c2 < 1`, backtracking `0 < c1 < 1` and
/// `0 < c2 < 1`, Goldstein `0 < c1 < 0.5` and `0 < c2 < 1`, Brent a
/// positive tolerance `c1`.
///
/// # Safety
///
/// `solver` must be null or a live handle from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_linesearch(
    solver: *mut rgmin_solver_t,
    kind: rgmin_linesearch_t,
    c1: f64,
    c2: f64,
    maxiter: usize,
) -> rgmin_status_t {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_linesearch: null solver");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let unit = |v: f64| v > 0.0 && v < 1.0;
    let ls = match kind {
        rgmin_linesearch_t::RGMIN_LINESEARCH_WOLFE if unit(c1) && unit(c2) && c1 < c2 => {
            Some(LineSearch::Wolfe { c1, c2, maxiter })
        }
        rgmin_linesearch_t::RGMIN_LINESEARCH_BACKTRACKING if unit(c1) && unit(c2) => {
            Some(LineSearch::Backtracking {
                c: c1,
                beta: c2,
                maxiter,
            })
        }
        rgmin_linesearch_t::RGMIN_LINESEARCH_GOLDSTEIN if c1 > 0.0 && c1 < 0.5 && unit(c2) => {
            Some(LineSearch::Goldstein {
                c: c1,
                beta: c2,
                maxiter,
            })
        }
        rgmin_linesearch_t::RGMIN_LINESEARCH_BRENT if c1 > 0.0 && c1.is_finite() => {
            Some(LineSearch::Brent { maxiter, tol: c1 })
        }
        _ => None,
    };
    match ls {
        Some(ls) if maxiter > 0 => {
            unsafe { (*solver).solver.set_linesearch(ls) };
            rgmin_status_t::RGMIN_SUCCESS
        }
        _ => {
            set_last_error("rgmin_solver_set_linesearch: constants out of range");
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        }
    }
}

/// Configure approximate Wolfe values for the session's Euclidean line
/// searches under every [`rgmin_accept_t`].
/// Negative or nonfinite relative windows leave the session unchanged.
///
/// # Safety
/// `solver` must be null or a live handle from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_objective_roundoff(
    solver: *mut rgmin_solver_t,
    relative: f64,
) -> rgmin_status_t {
    let Some(solver) = (unsafe { solver.as_mut() }) else {
        set_last_error("rgmin_solver_set_objective_roundoff: null solver");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    };
    if solver.solver.set_objective_roundoff(relative) {
        rgmin_status_t::RGMIN_SUCCESS
    } else {
        set_last_error("rgmin_solver_set_objective_roundoff: invalid relative window");
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    }
}

/// Capped Cartesian L-BFGS takes the geometry step (eOn's L-BFGS with a
/// finite-difference or `sy/yy` initial curvature, resets, and an energy
/// test). Nonzero turns it on. Off, the per-atom cap only clips the
/// two-loop trial. Needs a per-atom cap.
///
/// # Safety
///
/// A non-null `solver` must be a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_geometry_step(solver: *mut rgmin_solver_t, enabled: i32) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_geometry_step(enabled != 0) };
}

/// `0` off, `1` auto, `2` translation, `3` translation and rotation.
///
/// # Safety
///
/// A non-null `solver` must be a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_rigid_mode(solver: *mut rgmin_solver_t, mode: i32) {
    if solver.is_null() {
        return;
    }
    let Some(mode) = crate::RigidMode::from_i32(mode) else {
        set_last_error("rgmin_solver_set_rigid_mode: mode is not 0, 1, 2, or 3");
        return;
    };
    unsafe { (*solver).solver.set_rigid_mode(mode) };
}

/// Cartesian mass inner product. A raw force leaves this at zero.
///
/// # Safety
///
/// A non-null `solver` must be a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_rigid_mass_metric(
    solver: *mut rgmin_solver_t,
    enabled: i32,
) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_rigid_mass_metric(enabled != 0) };
}

/// Coordinates store `q_i = sqrt(m_i) x_i`. Not both this and the mass metric.
///
/// # Safety
///
/// A non-null `solver` must be a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_rigid_mass_weighted(
    solver: *mut rgmin_solver_t,
    enabled: i32,
) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_rigid_mass_weighted(enabled != 0) };
}

/// Periodic axes. A slab is `(1, 1, 0)`.
///
/// # Safety
///
/// A non-null `solver` must be a live session from [`rgmin_solver_create`]; pointer arguments are null or valid for the documented length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_pbc_axes(
    solver: *mut rgmin_solver_t,
    x: i32,
    y: i32,
    z: i32,
) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_pbc_axes(x != 0, y != 0, z != 0) };
}

/// Length `3N`. A nonzero byte freezes that coordinate. Null or `len == 0` clears it.
///
/// # Safety
///
/// A non-null `solver` must be a live session from [`rgmin_solver_create`]; pointer arguments are null or valid for the documented length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_frozen(
    solver: *mut rgmin_solver_t,
    frozen: *const u8,
    len: usize,
) {
    if solver.is_null() {
        return;
    }
    if frozen.is_null() || len == 0 {
        unsafe { (*solver).solver.set_frozen(&[]) };
        return;
    }
    let slice = unsafe { slice::from_raw_parts(frozen, len) };
    unsafe { (*solver).solver.set_frozen(slice) };
}

/// Broken translations, then broken rotations. The host sets both.
///
/// # Safety
///
/// A non-null `solver` must be a live session from [`rgmin_solver_create`]; pointer arguments are null or valid for the documented length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_broken_axes(
    solver: *mut rgmin_solver_t,
    tx: i32,
    ty: i32,
    tz: i32,
    rx: i32,
    ry: i32,
    rz: i32,
) {
    if solver.is_null() {
        return;
    }
    unsafe {
        (*solver)
            .solver
            .set_broken_axes(tx != 0, ty != 0, tz != 0, rx != 0, ry != 0, rz != 0);
    }
}

/// Alias for [`rgmin_solver_set_trust`].
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_highs_trust(
    solver: *mut rgmin_solver_t,
    radius: f64,
) -> i32 {
    unsafe { rgmin_solver_set_trust(solver, radius) }
}

/// Configure the factor shape geometry.
///
/// # Safety
/// `solver` must be null or a live handle from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_factor_shape(
    solver: *mut rgmin_solver_t,
    n: usize,
    p: usize,
) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_factor_shape(n, p) };
}

/// Configure the unitary geometry.
///
/// # Safety
/// `solver` must be null or a live handle from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_unitary(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_unitary(n) };
}

/// Caller-owned Eckart buffers. One per image. Not thread-local.
pub struct rgmin_rigid_workspace_t {
    inner: crate::RigidWorkspace,
}

/// Empty workspace.
///
/// # Safety
///
/// The returned workspace is owned by the caller and released with [`rgmin_rigid_workspace_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_rigid_workspace_new() -> *mut rgmin_rigid_workspace_t {
    Box::into_raw(Box::new(rgmin_rigid_workspace_t {
        inner: crate::RigidWorkspace::new(),
    }))
}

/// Free a workspace from [`rgmin_rigid_workspace_new`].
///
/// # Safety
///
/// `ws` is null or a workspace from [`rgmin_rigid_workspace_new`] that has not been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_rigid_workspace_free(ws: *mut rgmin_rigid_workspace_t) {
    if !ws.is_null() {
        drop(unsafe { Box::from_raw(ws) });
    }
}

/// Rank of the last project or project-hessian on this workspace.
///
/// # Safety
///
/// `ws` is null or a live workspace from [`rgmin_rigid_workspace_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_rigid_workspace_rank(ws: *const rgmin_rigid_workspace_t) -> usize {
    if ws.is_null() {
        0
    } else {
        unsafe { (*ws).inner.rank() }
    }
}

/// `P v` in place. `n_mass == 0` is unit mass. A null `frozen` is no freeze.
///
/// # Safety
///
/// `ws` is a live workspace from [`rgmin_rigid_workspace_new`]; every pointer argument is null or valid for the documented length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_rigid_project(
    ws: *mut rgmin_rigid_workspace_t,
    v: *mut f64,
    n: usize,
    pos: *const f64,
    masses: *const f64,
    n_mass: usize,
    mode: i32,
    mass_weighted_coords: i32,
    mass_metric: i32,
    pbc_x: i32,
    pbc_y: i32,
    pbc_z: i32,
    frozen: *const u8,
    broken_tx: i32,
    broken_ty: i32,
    broken_tz: i32,
    broken_rx: i32,
    broken_ry: i32,
    broken_rz: i32,
) -> rgmin_status_t {
    if ws.is_null() || v.is_null() || pos.is_null() || n == 0 || !n.is_multiple_of(3) {
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let Some(mode) = crate::RigidMode::from_i32(mode) else {
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    };
    let mass = match rigid_masses(masses, n_mass, n) {
        Ok(m) => m,
        Err(st) => return st,
    };
    let v = unsafe { slice::from_raw_parts_mut(v, n) };
    let pos = unsafe { slice::from_raw_parts(pos, n) };
    let frozen = if frozen.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts(frozen, n) })
    };
    let req = crate::RigidRequest {
        pos,
        masses: if mass.is_empty() { None } else { Some(mass) },
        mass_weighted_coords: mass_weighted_coords != 0,
        mass_metric: mass_metric != 0 && mass_weighted_coords == 0,
        mode,
        pbc: [pbc_x != 0, pbc_y != 0, pbc_z != 0],
        frozen,
        broken_translation: [broken_tx != 0, broken_ty != 0, broken_tz != 0],
        broken_rotation: [broken_rx != 0, broken_ry != 0, broken_rz != 0],
    };
    unsafe { (*ws).inner.project(v, &req) };
    rgmin_status_t::RGMIN_SUCCESS
}

/// Replace row-major `H` (`n * n`) with `P H P`.
///
/// # Safety
///
/// `ws` is a live workspace from [`rgmin_rigid_workspace_new`]; every pointer argument is null or valid for the documented length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_rigid_project_hessian(
    ws: *mut rgmin_rigid_workspace_t,
    h: *mut f64,
    n: usize,
    pos: *const f64,
    masses: *const f64,
    n_mass: usize,
    mode: i32,
    mass_weighted_coords: i32,
    mass_metric: i32,
    pbc_x: i32,
    pbc_y: i32,
    pbc_z: i32,
    frozen: *const u8,
    broken_tx: i32,
    broken_ty: i32,
    broken_tz: i32,
    broken_rx: i32,
    broken_ry: i32,
    broken_rz: i32,
) -> rgmin_status_t {
    if ws.is_null() || h.is_null() || pos.is_null() || n == 0 || !n.is_multiple_of(3) {
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let Some(mode) = crate::RigidMode::from_i32(mode) else {
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    };
    let mass = match rigid_masses(masses, n_mass, n) {
        Ok(m) => m,
        Err(st) => return st,
    };
    let h = unsafe { slice::from_raw_parts_mut(h, n * n) };
    let pos = unsafe { slice::from_raw_parts(pos, n) };
    let frozen = if frozen.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts(frozen, n) })
    };
    let req = crate::RigidRequest {
        pos,
        masses: if mass.is_empty() { None } else { Some(mass) },
        mass_weighted_coords: mass_weighted_coords != 0,
        mass_metric: mass_metric != 0 && mass_weighted_coords == 0,
        mode,
        pbc: [pbc_x != 0, pbc_y != 0, pbc_z != 0],
        frozen,
        broken_translation: [broken_tx != 0, broken_ty != 0, broken_tz != 0],
        broken_rotation: [broken_rx != 0, broken_ry != 0, broken_rz != 0],
    };
    unsafe { (*ws).inner.project_hessian(h, &req) };
    rgmin_status_t::RGMIN_SUCCESS
}

/// Horn / Coutsias alignment. Any periodic axis skips the rotation.
/// Writes the RMSD when `rmsd` is non-null.
///
/// # Safety
///
/// Every pointer argument is null or valid for the documented length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_rigid_align(
    mobile: *mut f64,
    reference: *const f64,
    n: usize,
    masses: *const f64,
    n_mass: usize,
    pbc_x: i32,
    pbc_y: i32,
    pbc_z: i32,
    rmsd: *mut f64,
) -> rgmin_status_t {
    if mobile.is_null() || reference.is_null() || n == 0 || !n.is_multiple_of(3) {
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let mass = match rigid_masses(masses, n_mass, n) {
        Ok(m) => m,
        Err(st) => return st,
    };
    let mobile = unsafe { slice::from_raw_parts_mut(mobile, n) };
    let reference = unsafe { slice::from_raw_parts(reference, n) };
    let value = crate::align_rigid(
        mobile,
        reference,
        if mass.is_empty() { None } else { Some(mass) },
        [pbc_x != 0, pbc_y != 0, pbc_z != 0],
    );
    if !rmsd.is_null() {
        unsafe { *rmsd = value };
    }
    rgmin_status_t::RGMIN_SUCCESS
}

/// Drop only the cached evaluation (`Solver::forget_evaluation`): the
/// next step evaluates the objective at the current `x`, and the method
/// memory, the step scale and the acceptance window stay. For a host
/// whose oracle changes between steps at the same point, such as a
/// min-mode effective gradient after the mode moves or a band that arms
/// its climbing image. ABI minor 28.
///
/// # Safety
///
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_forget_evaluation(solver: *mut rgmin_solver_t) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.forget_evaluation() };
}

/// Opt-in L-BFGS NEB guards (`Solver::set_lbfgs_neb_guards`): distance,
/// angle and curvature resets of the pair memory, and an empty-memory
/// scale of 0.01. Nonzero enables them; zero restores scale one. Other
/// methods ignore the setting. ABI minor 28.
///
/// # Safety
///
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_lbfgs_neb_guards(
    solver: *mut rgmin_solver_t,
    enabled: i32,
) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_lbfgs_neb_guards(enabled != 0) };
}

fn optional_positive(v: f64) -> Option<f64> {
    if v.is_finite() && v > 0.0 {
        Some(v)
    } else {
        None
    }
}

fn cell_of(cell: [f64; 3]) -> Option<[f64; 3]> {
    if cell.iter().any(|v| v.is_finite() && *v > 0.0) {
        Some(cell)
    } else {
        None
    }
}

fn index_ok(i: usize, natoms: usize) -> bool {
    i < natoms
}

/// One pair from [`rgmin_lowest_eigenpairs`]. `actions` is the product
/// count of the whole solve. ABI minor 39.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rgmin_eigenpair_t {
    /// Ritz value.
    pub value: f64,
    /// Hessian-vector products used by the solve.
    pub actions: usize,
}

/// One pair from [`rgmin_lowest_eigenpairs_ex`]. The first two fields
/// match [`rgmin_eigenpair_t`]. `residual` is `||Hv − θv||₂` and
/// `converged` is 1 when that residual met the tolerance. ABI minor 42.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rgmin_eigenpair_ex_t {
    /// Ritz value.
    pub value: f64,
    /// Hessian-vector products used by the solve.
    pub actions: usize,
    /// `||Hv − θv||₂` for this pair.
    pub residual: f64,
    /// 1 when `residual` met the tolerance, otherwise 0.
    pub converged: i32,
}

fn cpu_dl<'a>(
    tensor: *const DLManagedTensorVersioned,
    name: &str,
) -> Result<&'a DLTensor, rgmin_status_t> {
    if tensor.is_null() {
        set_last_error(&format!("{name}: null tensor"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    let dl = unsafe { &(*tensor).dl_tensor };
    if dl.device.device_type != DLDeviceType::kDLCPU {
        set_last_error(&format!(
            "{name}: device {:?} not supported in this build (CPU only)",
            dl.device.device_type as i32
        ));
        return Err(rgmin_status_t::RGMIN_UNSUPPORTED_DEVICE);
    }
    if dl.dtype.code != DLDataTypeCode::kDLFloat
        || dl.dtype.bits != 64
        || dl.dtype.lanes != 1
        || dl.shape.is_null()
        || dl.data.is_null()
        || dl.ndim < 1
    {
        set_last_error(&format!("{name}: need an f64 DLPack tensor"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    Ok(dl)
}

fn block_strides(
    dl: &DLTensor,
    n: usize,
    k: usize,
    name: &str,
) -> Result<(isize, isize), rgmin_status_t> {
    let shape = unsafe { slice::from_raw_parts(dl.shape, dl.ndim as usize) };
    if shape.iter().any(|dim| *dim < 0) {
        set_last_error(&format!("{name}: negative dimension"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    let strides = if dl.strides.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts(dl.strides, dl.ndim as usize) })
    };
    if let Some(strides) = strides
        && strides.iter().any(|stride| *stride < 0)
    {
        set_last_error(&format!("{name}: negative stride"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    if dl.ndim == 1 {
        let len = shape[0] as usize;
        let expect = match n.checked_mul(k) {
            Some(expect) => expect,
            None => {
                set_last_error(&format!("{name}: block is too large"));
                return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
            }
        };
        if len != expect {
            set_last_error(&format!("{name}: length {len} is not n*k = {expect}"));
            return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
        }
        if let Some(strides) = strides
            && strides[0] != 1
        {
            set_last_error(&format!("{name}: non-unit stride"));
            return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
        }
        return Ok((k as isize, 1));
    }
    if dl.ndim == 2 {
        if shape[0] as usize != n || shape[1] as usize != k {
            set_last_error(&format!(
                "{name}: shape ({}, {}) is not ({n}, {k})",
                shape[0], shape[1]
            ));
            return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
        }
        let row = strides.map_or(k as i64, |s| s[0]);
        let col = strides.map_or(1, |s| s[1]);
        return Ok((row as isize, col as isize));
    }
    set_last_error(&format!("{name}: need rank 1 or rank 2"));
    Err(rgmin_status_t::RGMIN_INVALID_PARAMETER)
}

fn f64_base(dl: &DLTensor) -> *const f64 {
    unsafe { (dl.data as *const u8).add(dl.byte_offset as usize).cast() }
}

fn read_block(
    tensor: *const DLManagedTensorVersioned,
    n: usize,
    k: usize,
    name: &str,
) -> Result<Array2<f64>, rgmin_status_t> {
    let dl = cpu_dl(tensor, name)?;
    let (row_stride, col_stride) = block_strides(dl, n, k, name)?;
    let base = f64_base(dl);
    let mut out = Array2::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let offset = (i as isize * row_stride + j as isize * col_stride) as usize;
            let value = unsafe { *base.add(offset) };
            out[(i, j)] = value;
        }
    }
    Ok(out)
}

fn write_block(
    tensor: *mut DLManagedTensorVersioned,
    block: &Array2<f64>,
    name: &str,
) -> Result<(), rgmin_status_t> {
    let (n, k) = block.dim();
    let dl = cpu_dl(tensor.cast(), name)?;
    let (row_stride, col_stride) = block_strides(dl, n, k, name)?;
    let base = f64_base(dl) as *mut f64;
    for i in 0..n {
        for j in 0..k {
            let offset = (i as isize * row_stride + j as isize * col_stride) as usize;
            unsafe { *base.add(offset) = block[(i, j)] };
        }
    }
    Ok(())
}

unsafe fn borrow_f64_matrix(data: *mut f64, n: usize, m: usize) -> *mut DLManagedTensorVersioned {
    struct Ctx {
        shape: [i64; 2],
        strides: [i64; 2],
    }

    unsafe extern "C" fn deleter(ptr: *mut DLManagedTensorVersioned) {
        if ptr.is_null() {
            return;
        }
        let ctx = unsafe { (*ptr).manager_ctx.cast::<Ctx>() };
        if !ctx.is_null() {
            drop(unsafe { Box::from_raw(ctx) });
        }
        drop(unsafe { Box::from_raw(ptr) });
    }

    let mut ctx = Box::new(Ctx {
        shape: [n as i64, m as i64],
        strides: [m as i64, 1],
    });
    let dl_tensor = DLTensor {
        data: data.cast(),
        device: DLDevice {
            device_type: DLDeviceType::kDLCPU,
            device_id: 0,
        },
        ndim: 2,
        dtype: DLDataType {
            code: DLDataTypeCode::kDLFloat,
            bits: 64,
            lanes: 1,
        },
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    };
    let managed = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: Box::into_raw(ctx).cast(),
        deleter: Some(deleter),
        flags: 0,
        dl_tensor,
    });
    Box::into_raw(managed)
}

#[allow(clippy::too_many_arguments)]
unsafe fn run_lowest_eigenpairs(
    label: &str,
    hvp: Option<rgmin_hvp_fn>,
    user: *mut c_void,
    precond: Option<rgmin_precond_fn>,
    precond_user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    seed: *const DLManagedTensorVersioned,
    modes_out: *mut DLManagedTensorVersioned,
    values_out: *mut DLManagedTensorVersioned,
    params: *const rgmin_eigen_params_t,
    options: *const rgmin_eigen_options_t,
    k: usize,
) -> Result<crate::Eigenpairs, rgmin_status_t> {
    let Some(hvp) = hvp else {
        set_last_error(&format!("{label}: null hvp"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    };
    if modes_out.is_null() || values_out.is_null() {
        set_last_error(&format!("{label}: null output"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    if k == 0 {
        set_last_error(&format!("{label}: k = 0 is outside 1..=n"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    let xs = cpu_f64_slice(x, "x")?;
    let n = xs.len();
    if k > n {
        set_last_error(&format!("{label}: k = {k} is outside 1..={n}"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    let seed_block = read_block(seed, n, k, "seed")?;
    let values = cpu_f64_slice_mut(values_out, "values_out")?;
    if values.len() != k {
        set_last_error(&format!("{label}: values_out length is not k"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    let typed = eigen_params_from_c(params)?;
    if !typed.kind.is_linked() {
        return Err(status_from_error(&Error::EigenUnavailable {
            kind: typed.kind.name(),
        }));
    }
    if !options.is_null() {
        let opts = unsafe { *options };
        let _ = (opts.degree, opts.extra);
    }
    if !typed.tol.is_finite() {
        set_last_error(&format!("{label}: tolerance is non-finite"));
        return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    let tol = if typed.tol > 0.0 { typed.tol } else { 1e-8 };
    let max_iter = if typed.max_iter == 0 {
        n.max(8)
    } else {
        typed.max_iter
    };
    let apply = CHvp { hvp, user };
    let x_arr = Array1::from(xs.to_vec());
    let failed = Cell::new(false);
    let solved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if let Some(callback) = precond {
            let op = CPrecond {
                precond: callback,
                user: precond_user,
                failed: &failed,
            };
            lowest_eigenpairs_precond(
                &apply,
                Some(&op),
                x_arr.view(),
                seed_block.view(),
                tol,
                max_iter,
            )
        } else {
            lowest_eigenpairs(&apply, x_arr.view(), seed_block.view(), tol, max_iter)
        }
    }));
    let found = match solved {
        Ok(Ok(found)) => found,
        Ok(Err(err)) => {
            if failed.get() {
                set_last_error(&format!("{label}: preconditioner failed"));
                return Err(rgmin_status_t::RGMIN_INVALID_PARAMETER);
            }
            return Err(status_from_error(&err));
        }
        Err(_) => {
            set_last_error(&format!("{label}: panicked"));
            return Err(rgmin_status_t::RGMIN_INTERNAL_ERROR);
        }
    };
    if found.pairs.len() != k {
        set_last_error(&format!("{label}: solve returned the wrong block width"));
        return Err(rgmin_status_t::RGMIN_INTERNAL_ERROR);
    }
    let mut modes = Array2::zeros((n, k));
    for (j, pair) in found.pairs.iter().enumerate() {
        if pair.vector.len() != n {
            set_last_error(&format!("{label}: mode length does not match n"));
            return Err(rgmin_status_t::RGMIN_INTERNAL_ERROR);
        }
        for i in 0..n {
            modes[(i, j)] = pair.vector[i];
        }
        values[j] = pair.value;
    }
    write_block(modes_out, &modes, "modes_out")?;
    Ok(found)
}

/// k lowest eigenpairs of a symmetric product. Block LOBPCG
/// (Knyazev 2001). `seed` and `modes_out` are row-major n-by-k: rank 2
/// with shape `(n, k)`, or rank 1 of length `n * k`. `values_out` and
/// `pairs_out` have length k, values ascending. Each `actions` field is
/// the product count of the solve. `params == NULL` selects tol `1e-8`
/// and `max_iter = max(n, 8)`. `options == NULL` is accepted; this
/// matrix-free solve ignores ChASE `degree` and `extra`. An unlinked
/// `params->kind` returns [`rgmin_status_t::RGMIN_UNAVAILABLE`].
/// `k == 0`, `k > n`, and non-finite input return
/// [`rgmin_status_t::RGMIN_INVALID_PARAMETER`].
/// [`rgmin_eigenpair_t`] keeps this layout. Residuals and a preconditioner
/// are [`rgmin_lowest_eigenpairs_ex`].
///
/// # Safety
/// `hvp` is callable for this call. `x` is a rank-1 f64 CPU tensor of
/// length n. `seed` and `modes_out` describe the same n-by-k block and
/// may alias. `values_out` is rank-1 length k. `pairs_out` points at k
/// writable records. `params` and `options` are null or readable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_lowest_eigenpairs(
    hvp: Option<rgmin_hvp_fn>,
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    seed: *const DLManagedTensorVersioned,
    modes_out: *mut DLManagedTensorVersioned,
    values_out: *mut DLManagedTensorVersioned,
    params: *const rgmin_eigen_params_t,
    options: *const rgmin_eigen_options_t,
    pairs_out: *mut rgmin_eigenpair_t,
    k: usize,
) -> rgmin_status_t {
    if pairs_out.is_null() {
        set_last_error("rgmin_lowest_eigenpairs: null output");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let found = match unsafe {
        run_lowest_eigenpairs(
            "rgmin_lowest_eigenpairs",
            hvp,
            user,
            None,
            std::ptr::null_mut(),
            x,
            seed,
            modes_out,
            values_out,
            params,
            options,
            k,
        )
    } {
        Ok(found) => found,
        Err(status) => return status,
    };
    for (j, pair) in found.pairs.iter().enumerate() {
        unsafe {
            *pairs_out.add(j) = rgmin_eigenpair_t {
                value: pair.value,
                actions: found.actions,
            };
        }
    }
    rgmin_status_t::RGMIN_SUCCESS
}

/// [`rgmin_lowest_eigenpairs`] plus each pair's residual and a converged
/// flag, and an optional preconditioner. `precond == NULL` is the identity
/// and is not called; the values and vectors then match
/// [`rgmin_lowest_eigenpairs`] bit for bit. `precond` applies `T` to one
/// row-major n-by-m block per expansion. `pairs_out` is
/// [`rgmin_eigenpair_ex_t`], length k. ABI minor 42. The minor-39 record
/// [`rgmin_eigenpair_t`] is unchanged.
///
/// # Safety
/// Same tensor rules as [`rgmin_lowest_eigenpairs`]. `precond` is null or
/// callable for this solve. `pairs_out` points at k writable extended records.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_lowest_eigenpairs_ex(
    hvp: Option<rgmin_hvp_fn>,
    user: *mut c_void,
    precond: Option<rgmin_precond_fn>,
    precond_user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    seed: *const DLManagedTensorVersioned,
    modes_out: *mut DLManagedTensorVersioned,
    values_out: *mut DLManagedTensorVersioned,
    params: *const rgmin_eigen_params_t,
    options: *const rgmin_eigen_options_t,
    pairs_out: *mut rgmin_eigenpair_ex_t,
    k: usize,
) -> rgmin_status_t {
    if pairs_out.is_null() {
        set_last_error("rgmin_lowest_eigenpairs_ex: null output");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let found = match unsafe {
        run_lowest_eigenpairs(
            "rgmin_lowest_eigenpairs_ex",
            hvp,
            user,
            precond,
            precond_user,
            x,
            seed,
            modes_out,
            values_out,
            params,
            options,
            k,
        )
    } {
        Ok(found) => found,
        Err(status) => return status,
    };
    for (j, pair) in found.pairs.iter().enumerate() {
        unsafe {
            *pairs_out.add(j) = rgmin_eigenpair_ex_t {
                value: pair.value,
                actions: found.actions,
                residual: pair.residual,
                converged: i32::from(pair.converged),
            };
        }
    }
    rgmin_status_t::RGMIN_SUCCESS
}
