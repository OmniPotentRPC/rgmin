//! Compiles the C++ shim against an installed rgpot.
//!
//! `RGPOT_PREFIX` names the meson install prefix holding
//! `include/rgpot` and `lib/librgpot.so`.

fn main() {
    let prefix = std::env::var("RGPOT_PREFIX")
        .expect("set RGPOT_PREFIX to an rgpot install prefix (meson install)");
    let include = format!("{prefix}/include");
    let lib = format!("{prefix}/lib");
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .opt_level(3)
        .include(&include)
        .define("RGPOT_HAS_POTLIB", "TRUE")
        .define("RGPOT_HAS_FORTRAN_POTS", "TRUE")
        .file("src/shim.cc")
        .compile("rgmin_bench_shim");
    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rustc-link-lib=dylib=rgpot");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed=src/shim.cc");
    println!("cargo:rerun-if-env-changed=RGPOT_PREFIX");
}
