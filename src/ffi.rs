//! Narrow C waist for rgmin. Every vector is a DLPack tensor (dlpk),
//! the same contract as eindir_objective_t, so a later device kernel
//! does not change the ABI.

use std::cell::RefCell;
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
use ndarray::{Array1, Array2, ArrayView1};

use crate::precon::{ExpParams, ExpState, Neighbours};
use crate::{
    Accept, ApplyHessian, Conjugacy, Control, DirectionalCurvature, EigenParams, EigensolverKind,
    Error, HessianOracle, HvpOracle, LineSearch, ManifoldKind, Method, NewtonKind, Oracle, QnStep,
    Restart, ScgParams, Solver, StopNorm, lowest_mode, minimize_method, minimize_method_hess,
    minimize_method_hvp, minimize_scg, minimize_scg_exact,
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
}

fn status_from_error(e: &Error) -> rgmin_status_t {
    set_last_error(&e.to_string());
    match e {
        Error::Oracle { .. }
        | Error::FireStalled { .. }
        | Error::AcceptStalled { .. }
        | Error::NoDescent { .. } => rgmin_status_t::RGMIN_INTERNAL_ERROR,
        Error::EigenUnavailable { .. } => rgmin_status_t::RGMIN_UNAVAILABLE,
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
pub const RGMIN_ABI_VERSION_MINOR: u16 = 37;
pub const RGMIN_ABI_LAYOUT_REVISION: u16 = 4;

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
    /// Steihaug-Toint Newton-CG. Hessian access is a product, not a matrix.
    RGMIN_NEWTON_CG = 21,
    /// Steihaug-Toint Newton-CG with a Nyström preconditioner.
    RGMIN_NEWTON_PCG = 22,
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
    /// Euclidean cap on each proposed step; non-positive disables the cap.
    pub maxmove: f64,
}

/// Result written by [`rgmin_minimize`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rgmin_report_t {
    /// `f(x)` at the accepted point.
    pub value: f64,
    /// Outer iterations.
    pub steps: usize,
    /// `||∇f||_2`.
    pub grad_norm: f64,
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

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        let c = CString::new(msg).unwrap_or_else(|_| CString::new("(interior NUL)").unwrap());
        *cell.borrow_mut() = c;
    });
}

/// Last error on this thread. Valid until the next rgmin C call.
#[unsafe(no_mangle)]
pub extern "C" fn rgmin_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
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
        rgmin_method_t::RGMIN_NEWTON_CG => Method::NewtonCg,
        rgmin_method_t::RGMIN_NEWTON_PCG => Method::NewtonPcg,
    }
}

fn method_to_c(method: &Method) -> Result<rgmin_method_t, ()> {
    let tag = match method {
        Method::Nlcg { conjugacy, .. } => match conjugacy {
            Conjugacy::FletcherReeves => rgmin_method_t::RGMIN_FLETCHER_REEVES,
            Conjugacy::PolakRibiere => rgmin_method_t::RGMIN_POLAK_RIBIERE,
            Conjugacy::HestenesStiefel => rgmin_method_t::RGMIN_HESTENES_STIEFEL,
            Conjugacy::DaiYuan => rgmin_method_t::RGMIN_DAI_YUAN,
            Conjugacy::ConjugateDescent => rgmin_method_t::RGMIN_CONJUGATE_DESCENT,
            Conjugacy::HagerZhang => rgmin_method_t::RGMIN_HAGER_ZHANG,
            Conjugacy::LiuStorey => rgmin_method_t::RGMIN_LIU_STOREY,
            Conjugacy::FrPr => rgmin_method_t::RGMIN_FR_PR,
            Conjugacy::Hybrid { .. } => return Err(()),
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
    let kind = EigensolverKind::from_ordinal(p.kind as u8).ok_or_else(|| {
        set_last_error(&format!("rgmin_eigen_kind_t unknown ordinal {}", p.kind));
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    })?;
    Ok(EigenParams {
        kind,
        nev: p.nev as usize,
        krylov: p.krylov as usize,
        max_iter: p.max_iter as usize,
        tol: p.tol,
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
    let mode_s = match cpu_f64_slice_mut(mode_out, "mode_out") {
        Ok(s) => s,
        Err(st) => return st,
    };
    if xs.len() != seed_s.len() || xs.len() != mode_s.len() {
        set_last_error("rgmin_lowest_eigenpair: x/seed/mode length mismatch");
        return rgmin_status_t::RGMIN_INVALID_PARAMETER;
    }
    let typed = match eigen_params_from_c(params) {
        Ok(p) => p,
        Err(st) => return st,
    };
    let apply = CHvp { hvp, user };
    let x_arr = Array1::from(xs.to_vec());
    let seed_arr = Array1::from(seed_s.to_vec());
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
        let control = Control {
            maxiter: c.maxiter,
            gtol: c.gtol,
            istep: if c.istep > 0.0 { c.istep } else { 1.0 },
            maxmove: if c.maxmove > 0.0 {
                Some(c.maxmove)
            } else {
                None
            },
        };
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
        let control = Control {
            maxiter: c.maxiter,
            gtol: c.gtol,
            istep: if c.istep > 0.0 { c.istep } else { 1.0 },
            maxmove: if c.maxmove > 0.0 {
                Some(c.maxmove)
            } else {
                None
            },
        };
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
        let restart = Restart::Never;
        let obj = ScgFfiOracle {
            inner: c_oracle(eval, grad, user, n),
            curv: curv.map(|f| f as usize),
            user: user as usize,
            scratch: Scratch::new(),
        };
        let report = if obj.curv.is_some() {
            minimize_scg_exact(&obj, Array1::from(init), &control, &scg, conjugacy, restart)
        } else {
            minimize_scg(
                &obj.inner,
                Array1::from(init),
                &control,
                &scg,
                conjugacy,
                restart,
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
        let control = Control {
            maxiter: c.maxiter,
            gtol: c.gtol,
            istep: if c.istep > 0.0 { c.istep } else { 1.0 },
            maxmove: if c.maxmove > 0.0 {
                Some(c.maxmove)
            } else {
                None
            },
        };
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
        let control = Control {
            maxiter: c.maxiter,
            gtol: c.gtol,
            istep: if c.istep > 0.0 { c.istep } else { 1.0 },
            maxmove: if c.maxmove > 0.0 {
                Some(c.maxmove)
            } else {
                None
            },
        };
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
        let control = Control {
            maxiter: c.maxiter,
            gtol: c.gtol,
            istep: if c.istep > 0.0 { c.istep } else { 1.0 },
            maxmove: if c.maxmove > 0.0 {
                Some(c.maxmove)
            } else {
                None
            },
        };
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
    let control = Control {
        maxiter: c.maxiter,
        gtol: c.gtol,
        istep: if c.istep > 0.0 { c.istep } else { 1.0 },
        maxmove: if c.maxmove > 0.0 {
            Some(c.maxmove)
        } else {
            None
        },
    };
    let mut solver = Solver::new(method_from_c(method, c.memory), control, dim).with_gtol(c.gtol);
    solver.set_accept(Accept::Energy);
    if c.istep <= 0.0 {
        solver.set_unit_first_step();
    }
    Box::into_raw(Box::new(rgmin_solver_t { solver, grad: None }))
}

/// Release a session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_free(solver: *mut rgmin_solver_t) {
    if !solver.is_null() {
        drop(unsafe { Box::from_raw(solver) });
    }
}

/// Drop method memory. The next step is a cold start from the current `x`.
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_rebase(solver: *mut rgmin_solver_t) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.rebase() };
}

/// Set the Euclidean step cap used by the next [`rgmin_solver_step`].
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
}

/// eOn `lbfgs_accept`. Legal on any session.
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
}

/// Select the session stop norm. ABI minor 29. Returns 0, or 1 on a
/// null session or an unknown integer.
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
        _ => {
            set_last_error("rgmin_solver_set_stop_norm: unknown norm");
            return 1;
        }
    };
    unsafe { (*solver).solver.set_stop_norm(norm) };
    0
}

/// Per-atom cap (eOn `maxAtomMotionAppliedV`). On Euclidean L-BFGS this
/// selects the geometry step and ignores [`rgmin_solver_set_accept`].
/// A nominal steepest trial at least 64 times the cap takes one
/// finite-difference curvature probe, clipped to the objective bounds.
/// Ordinary starts use `0.1 I` and a five-value nonmonotone window.
/// An installed exponential preconditioner replaces that scalar seed,
/// including the steepest fallbacks. Newton and dogleg scale a saddle
/// kick to this length. Non-positive clears it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_atom_maxmove(solver: *mut rgmin_solver_t, maxmove: f64) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_atom_maxmove(maxmove) };
}

/// eOn `lbfgs_project_rigid`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_project_rigid(solver: *mut rgmin_solver_t, enabled: i32) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_project_rigid(enabled != 0) };
}

/// Al-Baali extra-updates on the newest L-BFGS pair.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_extra_updates(solver: *mut rgmin_solver_t, extra: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_extra_updates(extra) };
}

/// Li-Fukushima cautious pair filter. `eps <= 0` disables it.
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

/// Adaptive trust radius for limited-memory
/// Broyden-Fletcher-Goldfarb-Shanno (L-BFGS), starting at
/// `radius`. A non-positive, NaN or infinite radius is not an error: it
/// turns the radius off, the default, and returns 0. The radius update
/// is Algorithm 1 of Burdakov, Gong, Zikrin, and Yuan 2017. The step is
/// a ray truncation along the quasi-Newton direction; see
/// [`Solver::set_lbfgs_trust`]. On Euclidean L-BFGS without HiGHS it
/// replaces [`rgmin_solver_set_accept`] and the geometry step of
/// [`rgmin_solver_set_atom_maxmove`]. A step whose direction is not
/// downhill or not finite returns `RGMIN_INTERNAL_ERROR`. Application binary
/// interface (ABI) minor 37. Returns 0, or 1 on a null session.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_lbfgs_trust(
    solver: *mut rgmin_solver_t,
    radius: f64,
) -> i32 {
    if solver.is_null() {
        set_last_error("rgmin_solver_set_lbfgs_trust: null solver");
        return 1;
    }
    unsafe { (*solver).solver.set_lbfgs_trust(radius) };
    0
}

/// Current L-BFGS trust radius, or 0 when it is off or the session is
/// null. A method, manifold or HiGHS step that ignores the radius still
/// reports the stored value. ABI minor 37.
///
/// # Safety
/// `solver` is null or a live session from [`rgmin_solver_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_lbfgs_trust_radius(solver: *const rgmin_solver_t) -> f64 {
    if solver.is_null() {
        return 0.0;
    }
    unsafe { (*solver).solver.lbfgs_trust_radius().unwrap_or(0.0) }
}

/// HiGHS feasible-set step. Returns 1 when this build has no `highs` feature.
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
        let _ = unsafe { (*solver).solver.set_highs_callback(cb, user) };
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
    /// Reserved 7-10 unused.
    RGMIN_MANIFOLD_CENTERED_MATRIX = 22,
}

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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_oblique(solver: *mut rgmin_solver_t, n: usize, m: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_oblique(n, m) };
}

/// Stiefel \(\mathrm{St}(n,p)\). `p = 1` is the sphere packing.
/// `p > 1` is packed column-major, length `n*p`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_stiefel(solver: *mut rgmin_solver_t, n: usize, p: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_stiefel(n, p) };
}

/// `n` unit-modulus complex numbers. Packed interleaved, length `2 n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_complex_circle(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_complex_circle(n) };
}

/// Complex Euclidean \(\mathbb{C}^n\). Packed interleaved, length `2 n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_euclidean_complex(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_euclidean_complex(n) };
}

/// Singleton of packed length `n`. manopt `constantfactory`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_constant(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_constant(n) };
}

/// Doubly-stochastic n-by-n, packed `n^2`. manopt
/// `multinomialdoublystochasticfactory`. Token 18 defaults to n = 2.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_multinomial_ds(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_multinomial_ds(n) };
}

/// Symmetric doubly-stochastic n-by-n, packed `n^2`. manopt
/// `multinomialsymmetricfactory`. Token 19 defaults to n = 2.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_multinomial_sym(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_multinomial_sym(n) };
}

/// Complex unit sphere in \(\mathbb{C}^n\). Packed interleaved, length `2 n`.
/// manopt `spherecomplexfactory`. Token 20 defaults to n = 1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_sphere_complex(solver: *mut rgmin_solver_t, n: usize) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_sphere_complex(n) };
}

/// Positive orthant of packed length `n`. manopt `positivefactory`.
/// Token 21 defaults to n = 1. Reserved tokens 7-10 unused.
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgmin_solver_set_periodic(solver: *mut rgmin_solver_t, enabled: i32) {
    if solver.is_null() {
        return;
    }
    unsafe { (*solver).solver.set_periodic(enabled != 0) };
}

/// One outer iteration. `x` is in/out. Callbacks live for this call only.
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

/// Two-loop `d = -H g`. No evaluation and no push.
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
) -> HessianOracle<
    impl Fn(ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    impl Fn(ndarray::ArrayView1<f64>) -> Array2<f64> + Send + Sync,
> {
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
) -> HvpOracle<
    impl Fn(ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    impl Fn(ndarray::ArrayView1<f64>, ndarray::ArrayView1<f64>) -> Array1<f64> + Send + Sync,
> {
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
) -> HessianOracle<
    impl Fn(ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    impl Fn(ndarray::ArrayView1<f64>) -> Array2<f64> + Send + Sync,
> {
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
        };
        (*solver).grad = rep.grad;
    }
    rgmin_status_t::RGMIN_SUCCESS
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
        assert_ne!(conjugacy_from_c(7).unwrap(), {
            Conjugacy::hybrid(Conjugacy::FletcherReeves, Conjugacy::PolakRibiere, true)
        });
    }

    #[test]
    fn conjugacy_from_c_rejects_unknown_and_method_t_codes() {
        for raw in [-1, 8, 99, 13] {
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
