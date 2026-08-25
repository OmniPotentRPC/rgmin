//! Narrow C waist for rgmin. Every vector is a DLPack tensor (dlpk),
//! the same contract as eindir_objective_t, so a later device kernel
//! does not change the ABI.

use std::cell::RefCell;
use std::ffi::CString;
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

use crate::{
    Accept, ApplyHessian, Conjugacy, Control, DirectionalCurvature, EigenParams, EigensolverKind,
    Error, HessianOracle, LineSearch, ManifoldKind, Method, NewtonKind, Oracle, QnStep, Restart,
    ScgParams, Solver, lowest_mode, minimize_method, minimize_method_hess, minimize_scg,
    minimize_scg_exact,
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
        Error::Oracle { .. } => rgmin_status_t::RGMIN_INTERNAL_ERROR,
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
pub const RGMIN_ABI_VERSION_MINOR: u16 = 17;
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

/// Iteration controls. `memory` is used only by L-BFGS (0 means 10).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rgmin_control_t {
    /// Maximum iterations.
    pub maxiter: usize,
    /// Stop when `||g||_2 < gtol`.
    pub gtol: f64,
    /// Initial line-search step.
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
    let solver = Solver::new(method_from_c(method, c.memory), control, dim).with_gtol(c.gtol);
    Box::into_raw(Box::new(rgmin_solver_t { solver }))
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

/// eOn `maxAtomMotionAppliedV`. Non-positive disables it.
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
    /// Symmetric doubly-stochastic n-by-n. Token defaults to n = 2.
    RGMIN_MANIFOLD_MULTINOMIAL_SYM = 19,
    /// Complex unit sphere C^n, packed 2n. Token defaults to n = 1.
    RGMIN_MANIFOLD_SPHERE_COMPLEX = 20,
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
            Ok(rep) => write_report(x, out, &rep),
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
            Ok(rep) => write_report(x, out, &rep),
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
            Ok(rep) => write_report(x, out, &rep),
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
            Ok(rep) => write_report(x, out, &rep),
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
    x: *mut DLManagedTensorVersioned,
    out: *mut rgmin_report_t,
    rep: &crate::Report,
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
