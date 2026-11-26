//! A caller compiled against layout revision 5 passes a 56-byte
//! `rgmin_control_t` and owns a report that ends at `stop`. The library
//! reads and writes no byte past either struct, and `require_convergence`
//! takes its default 0 for that caller. A caller compiled against layout
//! revision 6 passes 64 bytes, has `require_convergence` read and
//! `converged` written, and `noise_eps_f` takes its default 0.
//!
//! Each struct sits flush against a page mapped with no access, so a
//! read or write past it faults, and a canary tail shows a byte that
//! was read would have changed the result.

#![cfg(feature = "capi")]

mod consumer;

use std::mem::{offset_of, size_of};
use std::os::raw::c_void;

use dlpk::sys::DLManagedTensorVersioned;
use eindir_core::ffi::{eindir_core_abi_stamp, eindir_objective_t, eindir_status_t};
use rgmin::ffi::{
    RGMIN_CONTROL_REVISION5_SIZE, RGMIN_CONTROL_REVISION6_SIZE, RGMIN_REPORT_REVISION5_SIZE,
    rgmin_control_default, rgmin_control_init, rgmin_control_revision5_t, rgmin_control_t,
    rgmin_last_error, rgmin_manifold_t, rgmin_method_t, rgmin_minimize, rgmin_minimize_eindir,
    rgmin_minimize_eindir_sized, rgmin_minimize_hess, rgmin_minimize_hess_opts,
    rgmin_minimize_hess_opts_sized, rgmin_minimize_hvp, rgmin_minimize_hvp_opts,
    rgmin_minimize_hvp_opts_sized, rgmin_minimize_opts, rgmin_minimize_opts_sized,
    rgmin_minimize_rlbfgs, rgmin_minimize_rlbfgs_sized, rgmin_minimize_scg,
    rgmin_minimize_scg_with_options, rgmin_minimize_scg_with_options_sized, rgmin_report_t,
    rgmin_solver_create, rgmin_solver_create_sized, rgmin_solver_free,
    rgmin_solver_set_require_convergence, rgmin_solver_step, rgmin_status_t, rgmin_stop_t,
    rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
};

const FULL: usize = size_of::<rgmin_control_t>();
const SENTINEL: u8 = 0xA5;
const REPORT_BYTES: usize = size_of::<rgmin_report_t>();

/// Two spans of 64 KiB, the second mapped with no access. 64 KiB is a
/// multiple of every page size these targets use.
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod guard {
    use std::os::raw::{c_int, c_void};

    const SPAN: usize = 1 << 16;
    const PROT_NONE: c_int = 0;
    const PROT_RW: c_int = 1 | 2;
    #[cfg(target_os = "linux")]
    const MAP_PRIVATE_ANON: c_int = 0x02 | 0x20;
    #[cfg(target_os = "macos")]
    const MAP_PRIVATE_ANON: c_int = 0x0002 | 0x1000;

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            off: i64,
        ) -> *mut c_void;
        fn mprotect(addr: *mut c_void, len: usize, prot: c_int) -> c_int;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }

    pub struct Guarded {
        base: *mut u8,
    }

    impl Guarded {
        pub fn new() -> Self {
            let base = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    2 * SPAN,
                    PROT_RW,
                    MAP_PRIVATE_ANON,
                    -1,
                    0,
                )
            };
            assert!(!base.is_null() && base as isize != -1, "mmap failed");
            let base = base.cast::<u8>();
            let st = unsafe { mprotect(base.add(SPAN).cast(), SPAN, PROT_NONE) };
            assert_eq!(st, 0, "mprotect failed");
            Self { base }
        }

        /// `n` writable bytes that end at the inaccessible page.
        pub fn tail(&mut self, n: usize) -> *mut u8 {
            assert!(n.is_multiple_of(8) && n <= SPAN);
            unsafe { self.base.add(SPAN - n) }
        }
    }

    impl Drop for Guarded {
        fn drop(&mut self) {
            unsafe { munmap(self.base.cast(), 2 * SPAN) };
        }
    }
}

/// Without a guard page the canaries are the check.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod guard {
    pub struct Guarded {
        buf: Vec<u64>,
    }

    impl Guarded {
        pub fn new() -> Self {
            Self {
                buf: vec![0; 1 << 13],
            }
        }

        pub fn tail(&mut self, n: usize) -> *mut u8 {
            let len = self.buf.len() * 8;
            unsafe { self.buf.as_mut_ptr().cast::<u8>().add(len - n) }
        }
    }
}

use guard::Guarded;

unsafe fn cpu_f64(t: *const DLManagedTensorVersioned) -> *mut f64 {
    let dl = unsafe { &(*t).dl_tensor };
    unsafe { (dl.data as *mut u8).add(dl.byte_offset as usize) as *mut f64 }
}

unsafe extern "C" fn quad_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let x0 = unsafe { *cpu_f64(x) };
    unsafe { *value_out = 0.5 * x0 * x0 };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn quad_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    unsafe { *cpu_f64(g) = *cpu_f64(x) };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn quad_hess(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    h: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    unsafe { *cpu_f64(h) = 1.0 };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn quad_hvp(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    v: *const DLManagedTensorVersioned,
    hv: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    unsafe { *cpu_f64(hv) = *cpu_f64(v) };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn eindir_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> eindir_status_t {
    let x0 = unsafe { *cpu_f64(x) };
    unsafe { *value_out = 0.5 * x0 * x0 };
    eindir_status_t::EINDIR_SUCCESS
}

unsafe extern "C" fn eindir_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> eindir_status_t {
    unsafe { *cpu_f64(g) = *cpu_f64(x) };
    eindir_status_t::EINDIR_SUCCESS
}

fn objective() -> eindir_objective_t {
    eindir_objective_t {
        dim: 1,
        low: std::ptr::null_mut(),
        high: std::ptr::null_mut(),
        eval_fn: eindir_eval,
        grad_fn: Some(eindir_grad),
        user_data: std::ptr::null_mut(),
        free_fn: None,
    }
}

/// One iteration from x = 4 with gtol 0, so no run is converged and a
/// strict run returns `RGMIN_NOT_CONVERGED`.
fn revision5() -> rgmin_control_revision5_t {
    rgmin_control_revision5_t {
        maxiter: 1,
        gtol: 0.0,
        ..rgmin_control_default()
    }
}

/// `rgmin_control_t` as a layout revision 6 caller compiled it: the
/// fields through `require_convergence`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Revision6 {
    prefix: rgmin_control_revision5_t,
    require_convergence: i32,
}

fn revision6(strict: bool) -> Revision6 {
    Revision6 {
        prefix: revision5(),
        require_convergence: i32::from(strict),
    }
}

fn full(strict: bool) -> rgmin_control_t {
    rgmin_control_t {
        maxiter: 1,
        gtol: 0.0,
        require_convergence: i32::from(strict),
        noise_eps_f: 0.0,
        ..rgmin_control_t::default()
    }
}

/// Copy `c` into the last `size_of::<T>()` bytes before the guard.
fn place<T: Copy>(guard: &mut Guarded, c: T) -> *const rgmin_control_t {
    let n = size_of::<T>();
    let p = guard.tail(n);
    unsafe { p.cast::<T>().write(c) };
    p.cast()
}

/// The report a revision 5 caller owns, prefilled with the sentinel and
/// ending at the guard.
fn report(guard: &mut Guarded) -> *mut rgmin_report_t {
    let p = guard.tail(REPORT_BYTES);
    unsafe { p.write_bytes(SENTINEL, REPORT_BYTES) };
    p.cast()
}

fn tail_bytes(out: *const rgmin_report_t) -> Vec<u8> {
    let p = out.cast::<u8>();
    (RGMIN_REPORT_REVISION5_SIZE..REPORT_BYTES)
        .map(|i| unsafe { *p.add(i) })
        .collect()
}

fn last_error() -> String {
    let p = rgmin_last_error();
    assert!(!p.is_null());
    unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

#[derive(Clone, Copy, Debug)]
enum Entry {
    Minimize,
    Opts,
    Hess,
    HessOpts,
    Hvp,
    HvpOpts,
    Scg,
    ScgWithOptions,
    Rlbfgs,
    Eindir,
}

const ENTRIES: [Entry; 10] = [
    Entry::Minimize,
    Entry::Opts,
    Entry::Hess,
    Entry::HessOpts,
    Entry::Hvp,
    Entry::HvpOpts,
    Entry::Scg,
    Entry::ScgWithOptions,
    Entry::Rlbfgs,
    Entry::Eindir,
];

/// The exported symbol of `entry`, which reads layout revision 5.
fn call_plain(
    entry: Entry,
    ctrl: *const rgmin_control_t,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    let mut x = [4.0_f64];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 1) };
    let user = std::ptr::null_mut();
    let (e, g) = (Some(quad_eval as _), Some(quad_grad as _));
    let objective = objective();
    let stamp = eindir_core_abi_stamp();
    let sd = rgmin_method_t::RGMIN_STEEPEST;
    let newton = rgmin_method_t::RGMIN_NEWTON;
    let cg = rgmin_method_t::RGMIN_NEWTON_CG;
    let euclid = rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN;
    let status = unsafe {
        match entry {
            Entry::Minimize => rgmin_minimize(e, g, user, xt, ctrl, sd, out),
            Entry::Opts => rgmin_minimize_opts(e, g, user, xt, ctrl, sd, 0, 0, out),
            Entry::Hess => rgmin_minimize_hess(e, g, Some(quad_hess), user, xt, ctrl, newton, out),
            Entry::HessOpts => {
                rgmin_minimize_hess_opts(e, g, Some(quad_hess), user, xt, ctrl, newton, 0, 0, out)
            }
            Entry::Hvp => rgmin_minimize_hvp(e, g, Some(quad_hvp), user, xt, ctrl, cg, out),
            Entry::HvpOpts => {
                rgmin_minimize_hvp_opts(e, g, Some(quad_hvp), user, xt, ctrl, cg, 0, 0, out)
            }
            Entry::Scg => rgmin_minimize_scg(e, g, None, user, xt, ctrl, std::ptr::null(), out),
            Entry::ScgWithOptions => rgmin_minimize_scg_with_options(
                e,
                g,
                None,
                user,
                xt,
                ctrl,
                std::ptr::null(),
                std::ptr::null(),
                out,
            ),
            Entry::Rlbfgs => {
                rgmin_minimize_rlbfgs(e, g, user, xt, ctrl, std::ptr::null(), euclid, out)
            }
            Entry::Eindir => rgmin_minimize_eindir(&objective, &stamp, xt, ctrl, sd, out),
        }
    };
    unsafe { rgmin_tensor_free(xt) };
    status
}

/// The `_sized` symbol behind `entry` with the declared `ctrl_size`.
fn call_sized(
    entry: Entry,
    ctrl: *const rgmin_control_t,
    ctrl_size: usize,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    let mut x = [4.0_f64];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 1) };
    let user = std::ptr::null_mut();
    let (e, g) = (Some(quad_eval as _), Some(quad_grad as _));
    let objective = objective();
    let stamp = eindir_core_abi_stamp();
    let sd = rgmin_method_t::RGMIN_STEEPEST;
    let newton = rgmin_method_t::RGMIN_NEWTON;
    let cg = rgmin_method_t::RGMIN_NEWTON_CG;
    let euclid = rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN;
    let n = ctrl_size;
    let status = unsafe {
        match entry {
            Entry::Minimize | Entry::Opts => {
                rgmin_minimize_opts_sized(e, g, user, xt, ctrl, n, sd, 0, 0, out)
            }
            Entry::Hess | Entry::HessOpts => rgmin_minimize_hess_opts_sized(
                e,
                g,
                Some(quad_hess),
                user,
                xt,
                ctrl,
                n,
                newton,
                0,
                0,
                out,
            ),
            Entry::Hvp | Entry::HvpOpts => rgmin_minimize_hvp_opts_sized(
                e,
                g,
                Some(quad_hvp),
                user,
                xt,
                ctrl,
                n,
                cg,
                0,
                0,
                out,
            ),
            Entry::Scg | Entry::ScgWithOptions => rgmin_minimize_scg_with_options_sized(
                e,
                g,
                None,
                user,
                xt,
                ctrl,
                n,
                std::ptr::null(),
                std::ptr::null(),
                out,
            ),
            Entry::Rlbfgs => {
                rgmin_minimize_rlbfgs_sized(e, g, user, xt, ctrl, n, std::ptr::null(), euclid, out)
            }
            Entry::Eindir => rgmin_minimize_eindir_sized(&objective, &stamp, xt, ctrl, n, sd, out),
        }
    };
    unsafe { rgmin_tensor_free(xt) };
    status
}

#[test]
fn revision_5_sizes_are_the_released_layout() {
    assert_eq!(
        size_of::<rgmin_control_revision5_t>(),
        RGMIN_CONTROL_REVISION5_SIZE
    );
    assert_eq!(
        RGMIN_CONTROL_REVISION5_SIZE,
        offset_of!(rgmin_control_t, require_convergence)
    );
    assert_eq!(
        RGMIN_REPORT_REVISION5_SIZE,
        offset_of!(rgmin_report_t, converged)
    );
    assert_eq!(
        RGMIN_CONTROL_REVISION6_SIZE,
        offset_of!(rgmin_control_t, noise_eps_f)
    );
    assert_eq!(size_of::<Revision6>(), RGMIN_CONTROL_REVISION6_SIZE);
    assert_eq!(
        offset_of!(Revision6, require_convergence),
        offset_of!(rgmin_control_t, require_convergence)
    );
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(RGMIN_CONTROL_REVISION5_SIZE, 56);
        assert_eq!(RGMIN_CONTROL_REVISION6_SIZE, 64);
        assert_eq!(FULL, 72);
        assert_eq!(RGMIN_REPORT_REVISION5_SIZE, 28);
        assert_eq!(REPORT_BYTES, 32);
    }
}

#[test]
fn every_plain_entry_reads_only_the_revision_5_control() {
    for entry in ENTRIES {
        let mut guard = Guarded::new();
        let ctrl = place(&mut guard, revision5());
        let mut out_guard = Guarded::new();
        let out = report(&mut out_guard);
        let status = call_plain(entry, ctrl, out);
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS, "{entry:?}");
        assert_eq!(tail_bytes(out), [SENTINEL; 4], "{entry:?} wrote past stop");
        let written = unsafe { std::ptr::read(out) };
        assert!(written.grad_norm.is_finite(), "{entry:?}");
        assert_eq!(written.steps, 1, "{entry:?}");
    }
}

#[test]
fn bytes_past_a_revision_5_control_do_not_turn_strict_mode_on() {
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Canary {
        ctrl: rgmin_control_revision5_t,
        tail: [u8; 8],
    }
    for entry in ENTRIES {
        let mut guard = Guarded::new();
        let ctrl = place(
            &mut guard,
            Canary {
                ctrl: revision5(),
                tail: [0xFF; 8],
            },
        );
        let mut out = std::mem::MaybeUninit::<rgmin_report_t>::zeroed();
        let status = call_plain(entry, ctrl, out.as_mut_ptr());
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS, "{entry:?}");
        let status = call_sized(entry, ctrl, RGMIN_CONTROL_REVISION5_SIZE, out.as_mut_ptr());
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS, "{entry:?}");
    }
}

#[test]
fn a_declared_revision_5_size_reads_and_writes_the_revision_5_layout() {
    for entry in ENTRIES {
        let mut guard = Guarded::new();
        let ctrl = place(&mut guard, revision5());
        let mut out_guard = Guarded::new();
        let out = report(&mut out_guard);
        let status = call_sized(entry, ctrl, RGMIN_CONTROL_REVISION5_SIZE, out);
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS, "{entry:?}");
        assert_eq!(tail_bytes(out), [SENTINEL; 4], "{entry:?} wrote past stop");
    }
}

#[test]
fn the_current_size_reads_require_convergence_and_writes_converged() {
    for entry in ENTRIES {
        for strict in [false, true] {
            let mut guard = Guarded::new();
            let ctrl = place(&mut guard, full(strict));
            let mut out_guard = Guarded::new();
            let out = report(&mut out_guard);
            let status = call_sized(entry, ctrl, FULL, out);
            let want = if strict {
                rgmin_status_t::RGMIN_NOT_CONVERGED
            } else {
                rgmin_status_t::RGMIN_SUCCESS
            };
            assert_eq!(status, want, "{entry:?} strict {strict}");
            let written = unsafe { std::ptr::read(out) };
            assert_eq!(written.converged, 0, "{entry:?} strict {strict}");
            assert_eq!(written.steps, 1, "{entry:?} strict {strict}");
        }
    }
}

#[test]
fn a_declared_revision_6_size_reads_require_convergence_and_no_byte_past_it() {
    for entry in ENTRIES {
        for strict in [false, true] {
            let mut guard = Guarded::new();
            let ctrl = place(&mut guard, revision6(strict));
            let mut out_guard = Guarded::new();
            let out = report(&mut out_guard);
            let status = call_sized(entry, ctrl, RGMIN_CONTROL_REVISION6_SIZE, out);
            let want = if strict {
                rgmin_status_t::RGMIN_NOT_CONVERGED
            } else {
                rgmin_status_t::RGMIN_SUCCESS
            };
            assert_eq!(status, want, "{entry:?} strict {strict}");
            let written = unsafe { std::ptr::read(out) };
            assert_eq!(written.converged, 0, "{entry:?} strict {strict}");
            assert_eq!(written.steps, 1, "{entry:?} strict {strict}");
        }
    }
}

#[test]
fn a_larger_declared_size_reads_the_known_fields_and_ignores_the_rest() {
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Later {
        ctrl: rgmin_control_t,
        later: [u64; 2],
    }
    for entry in ENTRIES {
        let mut guard = Guarded::new();
        let later = Later {
            ctrl: full(true),
            later: [u64::MAX; 2],
        };
        let ctrl = place(&mut guard, later);
        let mut out = std::mem::MaybeUninit::<rgmin_report_t>::zeroed();
        let status = call_sized(entry, ctrl, size_of::<Later>(), out.as_mut_ptr());
        assert_eq!(status, rgmin_status_t::RGMIN_NOT_CONVERGED, "{entry:?}");
    }
}

#[test]
fn any_other_declared_size_is_refused_before_a_read() {
    for size in [
        0,
        8,
        RGMIN_CONTROL_REVISION5_SIZE - 1,
        RGMIN_CONTROL_REVISION5_SIZE + 1,
        RGMIN_CONTROL_REVISION6_SIZE - 1,
        RGMIN_CONTROL_REVISION6_SIZE + 1,
        FULL - 1,
    ] {
        for entry in ENTRIES {
            let mut guard = Guarded::new();
            let ctrl = guard.tail(8).cast::<rgmin_control_t>().cast_const();
            let mut out_guard = Guarded::new();
            let out = report(&mut out_guard);
            let status = call_sized(entry, ctrl, size, out);
            assert_eq!(
                status,
                rgmin_status_t::RGMIN_INVALID_PARAMETER,
                "{entry:?} size {size}"
            );
            assert!(
                last_error().contains("ctrl_size"),
                "{entry:?}: {}",
                last_error()
            );
            let untouched =
                (0..REPORT_BYTES).all(|i| unsafe { *out.cast::<u8>().add(i) } == SENTINEL);
            assert!(untouched, "{entry:?} size {size} wrote the report");
        }
        let mut guard = Guarded::new();
        let ctrl = guard.tail(8).cast::<rgmin_control_t>().cast_const();
        let solver =
            unsafe { rgmin_solver_create_sized(rgmin_method_t::RGMIN_STEEPEST, ctrl, size, 1) };
        assert!(solver.is_null(), "size {size}");
        assert!(last_error().contains("ctrl_size"));
    }
}

fn session_step(
    solver: *mut rgmin::ffi::rgmin_solver_t,
    x: &mut [f64; 1],
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 1) };
    let status = unsafe {
        rgmin_solver_step(
            solver,
            Some(quad_eval),
            Some(quad_grad),
            std::ptr::null_mut(),
            xt,
            out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    status
}

type Create = fn(*const rgmin_control_t) -> *mut rgmin::ffi::rgmin_solver_t;

#[test]
fn a_revision_5_session_reads_the_revision_5_control_and_writes_its_report() {
    let creates: [(&str, Create); 2] = [
        ("plain", |c| unsafe {
            rgmin_solver_create(rgmin_method_t::RGMIN_STEEPEST, c, 1)
        }),
        ("sized", |c| unsafe {
            rgmin_solver_create_sized(
                rgmin_method_t::RGMIN_STEEPEST,
                c,
                RGMIN_CONTROL_REVISION5_SIZE,
                1,
            )
        }),
    ];
    for (name, create) in creates {
        let mut guard = Guarded::new();
        let solver = create(place(&mut guard, revision5()));
        assert!(!solver.is_null(), "{name}");
        let mut x = [4.0_f64];
        for strict in [0, 1] {
            unsafe { rgmin_solver_set_require_convergence(solver, strict) };
            let mut out_guard = Guarded::new();
            let out = report(&mut out_guard);
            let status = session_step(solver, &mut x, out);
            assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS, "{name}");
            assert_eq!(tail_bytes(out), [SENTINEL; 4], "{name} wrote past stop");
            let at = offset_of!(rgmin_report_t, stop);
            let stop = unsafe { out.cast::<u8>().add(at).cast::<rgmin_stop_t>().read() };
            assert_ne!(stop, rgmin_stop_t::RGMIN_STOP_TOLERANCE, "{name}");
            let value = unsafe { out.cast::<f64>().read() };
            assert_eq!(value.to_bits(), (0.5 * x[0] * x[0]).to_bits(), "{name}");
        }
        unsafe { rgmin_solver_free(solver) };
    }
}

#[test]
fn a_current_session_writes_converged() {
    let mut guard = Guarded::new();
    let ctrl = place(
        &mut guard,
        rgmin_control_t {
            maxiter: 10,
            gtol: 1e-3,
            ..rgmin_control_t::default()
        },
    );
    let solver =
        unsafe { rgmin_solver_create_sized(rgmin_method_t::RGMIN_STEEPEST, ctrl, FULL, 1) };
    assert!(!solver.is_null());
    let mut out_guard = Guarded::new();
    let out = report(&mut out_guard);
    let mut converged = 0;
    let mut x = [4.0_f64];
    for _ in 0..20 {
        assert_eq!(
            session_step(solver, &mut x, out),
            rgmin_status_t::RGMIN_SUCCESS
        );
        converged = unsafe { std::ptr::read(out) }.converged;
        assert!(converged == 0 || converged == 1);
        if converged == 1 {
            break;
        }
    }
    assert_eq!(converged, 1);
    unsafe { rgmin_solver_free(solver) };
}

#[test]
fn control_init_writes_only_the_declared_layout() {
    let mut guard = Guarded::new();
    let p = guard.tail(RGMIN_CONTROL_REVISION5_SIZE);
    unsafe { p.write_bytes(SENTINEL, RGMIN_CONTROL_REVISION5_SIZE) };
    let st = unsafe { rgmin_control_init(p.cast(), RGMIN_CONTROL_REVISION5_SIZE) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    let got = unsafe { p.cast::<rgmin_control_revision5_t>().read() };
    let want = rgmin_control_default();
    assert_eq!(got.maxiter, want.maxiter);
    assert_eq!(got.gtol.to_bits(), want.gtol.to_bits());
    assert_eq!(got.noise_c.to_bits(), want.noise_c.to_bits());

    let mut guard = Guarded::new();
    let p = guard.tail(FULL).cast::<rgmin_control_t>();
    unsafe { p.cast::<u8>().write_bytes(SENTINEL, FULL) };
    let st = unsafe { rgmin_control_init(p, FULL) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(unsafe { p.read() }.require_convergence, 0);
    assert_eq!(unsafe { p.read() }.noise_eps_f.to_bits(), 0.0_f64.to_bits());

    let mut guard = Guarded::new();
    let p = guard.tail(RGMIN_CONTROL_REVISION6_SIZE);
    unsafe { p.write_bytes(SENTINEL, RGMIN_CONTROL_REVISION6_SIZE) };
    let st = unsafe { rgmin_control_init(p.cast(), RGMIN_CONTROL_REVISION6_SIZE) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    let got = unsafe { p.cast::<Revision6>().read() };
    assert_eq!(got.prefix.maxiter, want.maxiter);
    assert_eq!(got.prefix.noise_c.to_bits(), want.noise_c.to_bits());
    assert_eq!(got.require_convergence, 0);

    for size in [
        0,
        RGMIN_CONTROL_REVISION5_SIZE - 8,
        RGMIN_CONTROL_REVISION5_SIZE + 4,
    ] {
        let mut guard = Guarded::new();
        let p = guard.tail(FULL);
        unsafe { p.write_bytes(SENTINEL, FULL) };
        let st = unsafe { rgmin_control_init(p.cast(), size) };
        assert_eq!(st, rgmin_status_t::RGMIN_INVALID_PARAMETER, "size {size}");
        assert!(
            (0..FULL).all(|i| unsafe { *p.add(i) } == SENTINEL),
            "size {size}"
        );
    }
    assert_eq!(
        unsafe { rgmin_control_init(std::ptr::null_mut(), FULL) },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
}

#[test]
fn c_header_declares_its_size_and_reads_both_layouts() {
    consumer::compile_and_run("tests/c_control_layout.c", "rgmin_c_control_layout", &[]);
}
