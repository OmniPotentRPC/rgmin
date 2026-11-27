//! Links `rgpot::LJClusterPot` when `--features rgpot` and `RGPOT_PREFIX` are set.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=RGPOT_PREFIX");
    println!("cargo:rerun-if-changed=src/shim.cc");
    let rgpot = std::env::var("CARGO_FEATURE_RGPOT").is_ok();
    if !rgpot {
        return;
    }
    let prefix = std::env::var("RGPOT_PREFIX").unwrap_or_else(|_| {
        panic!("set RGPOT_PREFIX to an rgpot install prefix (meson install)");
    });
    let include = PathBuf::from(&prefix).join("include");
    let lib = lib_dir(&prefix);
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .opt_level(3)
        .include(&include)
        .file("src/shim.cc")
        .compile("rgpot_lj_shim");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    if let Some(stdcxx) = stdcxx_dir() {
        println!("cargo:rustc-link-search=native={}", stdcxx.display());
    }
    println!("cargo:rustc-link-lib=dylib=rgpot");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn stdcxx_dir() -> Option<PathBuf> {
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "g++".to_string());
    let out = std::process::Command::new(cxx)
        .arg("-print-file-name=libstdc++.so")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let path = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    path.parent().map(PathBuf::from)
}

fn lib_dir(prefix: &str) -> PathBuf {
    let multi = PathBuf::from(prefix).join("lib/x86_64-linux-gnu");
    if multi.join("librgpot.so").exists() {
        return multi;
    }
    PathBuf::from(prefix).join("lib")
}
