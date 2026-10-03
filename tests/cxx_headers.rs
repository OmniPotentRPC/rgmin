//! Compile and run the canonical and compatibility C++ headers.

#![cfg(feature = "capi")]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn lib_dir(root: &Path) -> PathBuf {
    if let Ok(td) = env::var("CARGO_TARGET_DIR") {
        return PathBuf::from(td).join(profile());
    }
    root.join("target").join(profile())
}

#[test]
fn cxx_headers_compile_link_and_reuse() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = root.join("tests/cxx_headers.cpp");
    let lib = lib_dir(&root);
    let so = lib.join("librgmin.so");
    let a = lib.join("librgmin.a");
    assert!(
        so.is_file() || a.is_file(),
        "librgmin missing under {} (so={} a={})",
        lib.display(),
        so.is_file(),
        a.is_file()
    );
    let out_dir = env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir);
    let bin = out_dir.join("rgmin_cxx_headers");
    let mut cmd = Command::new("c++");
    cmd.arg("-std=c++17")
        .arg("-O0")
        .arg(src)
        .arg("-I")
        .arg(root.join("include"))
        .arg("-I")
        .arg(root.join("tests/support"))
        .arg("-o")
        .arg(&bin);
    if so.is_file() {
        cmd.arg("-L")
            .arg(&lib)
            .arg("-lrgmin")
            .arg(format!("-Wl,-rpath,{}", lib.display()));
    } else {
        cmd.arg(&a).arg("-lpthread").arg("-ldl").arg("-lm");
    }
    let compiled = cmd.output().expect("spawn c++");
    assert!(
        compiled.status.success(),
        "c++ failed:\n{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let ran = Command::new(&bin).output().expect("run cxx smoke");
    assert!(
        ran.status.success(),
        "cxx smoke exit {}:\n{}\n{}",
        ran.status,
        String::from_utf8_lossy(&ran.stdout),
        String::from_utf8_lossy(&ran.stderr)
    );
    let stdout = String::from_utf8_lossy(&ran.stdout);
    assert!(
        stdout.contains("pairs=1"),
        "cxx smoke did not report retained curvature: {stdout}"
    );
}
