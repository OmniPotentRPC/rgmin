//! Borrowed error string, const session queries, and the C quadratic.

#![cfg(feature = "capi")]

use std::ffi::CStr;
use std::path::PathBuf;
use std::process::Command;
use std::ptr;

use rgmin::ffi::{
    rgmin_control_t, rgmin_last_error, rgmin_last_error_message, rgmin_method_from_name,
    rgmin_method_t, rgmin_solver_create, rgmin_solver_dim, rgmin_solver_free, rgmin_solver_steps,
};

#[test]
fn the_error_string_is_borrowed_and_named_twice() {
    let mut out = rgmin_method_t::RGMIN_LBFGS;
    let st = unsafe { rgmin_method_from_name(c"not-a-method".as_ptr(), &mut out) };
    assert_ne!(st, rgmin::ffi::rgmin_status_t::RGMIN_SUCCESS);
    let a = rgmin_last_error();
    let b = rgmin_last_error_message();
    assert!(!a.is_null() && a == b);
    let text = unsafe { CStr::from_ptr(a) }.to_string_lossy();
    assert!(
        text.contains("not-a-method") || text.contains("unknown"),
        "{text}"
    );
}

#[test]
fn dim_and_steps_read_a_const_session() {
    let ctrl = rgmin_control_t {
        maxiter: 4,
        gtol: 1e-8,
        istep: 0.1,
        memory: 5,
        maxmove: 0.2,
    };
    let solver = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 3) };
    assert!(!solver.is_null());
    assert_eq!(unsafe { rgmin_solver_dim(solver) }, 3);
    assert_eq!(unsafe { rgmin_solver_steps(solver) }, 0);
    assert_eq!(unsafe { rgmin_solver_dim(ptr::null()) }, 0);
    assert_eq!(unsafe { rgmin_solver_steps(ptr::null()) }, 0);
    unsafe { rgmin_solver_free(solver) };
}

#[test]
fn c_quadratic_ends_below_1e_8() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let base = if let Ok(td) = std::env::var("CARGO_TARGET_DIR") {
        PathBuf::from(td).join(profile)
    } else {
        root.join("target").join(profile)
    };
    let so = [
        base.join("librgmin.so"),
        base.join("deps").join("librgmin.so"),
    ]
    .into_iter()
    .find(|path| path.is_file());
    let so = so.expect("librgmin.so");
    let libdir = so.parent().expect("libdir");
    let bin = base.join("quadratic_c");
    let status = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-I"])
        .arg(root.join("include"))
        .arg(root.join("docs/examples/quadratic.c"))
        .arg("-L")
        .arg(libdir)
        .args(["-lrgmin", "-lm"])
        .arg(format!("-Wl,-rpath,{}", libdir.display()))
        .arg("-o")
        .arg(&bin)
        .status()
        .expect("cc");
    assert!(status.success(), "cc {status}");
    let out = Command::new(&bin).output().expect("run");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("quadratic"), "{text}");
}
