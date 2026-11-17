//! Compile the canonical compatibility header as C using the public include tree.

#![cfg(feature = "capi")]

mod consumer;

#[test]
fn public_c_header_compiles_and_checks_the_abi() {
    consumer::compile_and_run("tests/ffi_smoke.c", "rgmin_public_c_header");
}

#[test]
fn c_header_reads_the_noise_floor_stop() {
    consumer::compile_and_run("tests/c_noise_floor.c", "rgmin_c_noise_floor");
}

#[cfg(feature = "highs")]
#[test]
fn c_header_routes_and_clears_the_highs_callback() {
    consumer::compile_and_run("tests/c_highs_callback.c", "rgmin_c_highs_callback");
}
