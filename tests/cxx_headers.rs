//! Compile and run the C++ wrap headers against librgmin.
//!
//! The dest C ABI tests never included rgmin/optimize.hpp, so a
//! namespace/function clash in that header shipped green.

#![cfg(feature = "capi")]

mod consumer;

#[test]
fn cxx_headers_compile_link_and_set_box() {
    let ran = consumer::compile_and_run("tests/cxx_headers.cpp", "rgmin_cxx_headers", &[]);
    let stdout = String::from_utf8_lossy(&ran.stdout);
    assert!(
        stdout.contains("set_box="),
        "cxx smoke did not report set_box: {stdout}"
    );
    assert!(
        stdout.contains("constraints ok"),
        "cxx smoke did not run the distance constraints: {stdout}"
    );
    assert!(
        stdout.contains("pairs="),
        "cxx smoke did not report retained curvature: {stdout}"
    );
    assert!(
        stdout.contains("manifold ok"),
        "cxx smoke did not run the manifold fit: {stdout}"
    );
    assert!(
        stdout.contains("strict convergence ok"),
        "cxx smoke did not run strict convergence: {stdout}"
    );
}
