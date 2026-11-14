//! Compile and run the C++ wrap headers against librgmin.
//!
//! The dest C ABI tests never included rgmin/optimize.hpp, so a
//! namespace/function clash in that header shipped green.

#![cfg(feature = "capi")]

use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn cxx_compiler() -> &'static str {
    for bin in ["c++", "g++", "clang++"] {
        let mut child = match Command::new(bin)
            .args(["-std=c++17", "-x", "c++", "-", "-c", "-o", "/dev/null"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => child,
            Err(_) => continue,
        };
        let wrote = child.stdin.take().is_some_and(|mut stdin| {
            stdin
                .write_all(b"#include <cstdio>\nint main(){return 0;}\n")
                .is_ok()
        });
        if wrote && child.wait().map(|status| status.success()).unwrap_or(false) {
            return bin;
        }
    }
    "c++"
}

fn enabled_features() -> Vec<&'static str> {
    [
        ("capi", cfg!(feature = "capi")),
        ("par", cfg!(feature = "par")),
        ("simd", cfg!(feature = "simd")),
        ("highs", cfg!(feature = "highs")),
        ("slepc", cfg!(feature = "slepc")),
        ("primme", cfg!(feature = "primme")),
        ("chase", cfg!(feature = "chase")),
        ("libkrylov", cfg!(feature = "libkrylov")),
        ("readcon", cfg!(feature = "readcon")),
    ]
    .into_iter()
    .filter_map(|(name, on)| on.then_some(name))
    .collect()
}

/// The cdylib and staticlib that `cargo build` writes for this test's
/// features. `cargo test` refreshes only the rlib, so a library already
/// in the target directory can predate the sources and the header.
fn build_librgmin(root: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(root)
        .args(["build", "--lib", "--no-default-features", "--features"])
        .arg(enabled_features().join(","))
        .arg("--message-format=json-render-diagnostics")
        .stderr(Stdio::inherit());
    if !cfg!(debug_assertions) {
        cmd.arg("--release");
    }
    let built = cmd.output().expect("spawn cargo to build librgmin");
    assert!(
        built.status.success(),
        "cargo build of librgmin for the C++ smoke failed ({}); \
         run `cargo build --features capi` and read its errors",
        built.status
    );
    let (mut so, mut a) = (None, None);
    for line in String::from_utf8_lossy(&built.stdout).lines() {
        if !line.contains(r#""reason":"compiler-artifact""#) || !line.contains(r#""name":"rgmin""#)
        {
            continue;
        }
        let Some(start) = line.find(r#""filenames":["#) else {
            continue;
        };
        let rest = &line[start + r#""filenames":["#.len()..];
        let list = &rest[..rest.find(']').unwrap_or(rest.len())];
        for file in list.split(',').map(|f| f.trim().trim_matches('"')) {
            let path = PathBuf::from(file.replace(r"\\", r"\"));
            match path.file_name().and_then(|n| n.to_str()) {
                Some("librgmin.so") => so = Some(path),
                Some("librgmin.a") => a = Some(path),
                _ => {}
            }
        }
    }
    (so.filter(|p| p.is_file()), a.filter(|p| p.is_file()))
}

#[test]
fn cxx_headers_compile_link_and_set_box() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = root.join("tests/cxx_headers.cpp");
    let (so, a) = build_librgmin(&root);
    assert!(
        so.is_some() || a.is_some(),
        "cargo built no librgmin.so or librgmin.a"
    );
    let out_dir = env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir);
    let bin = out_dir.join("rgmin_cxx_headers");
    let mut cmd = Command::new(cxx_compiler());
    cmd.arg("-std=c++17")
        .arg("-O0")
        .arg(src)
        .arg("-I")
        .arg(root.join("include"))
        .arg("-o")
        .arg(&bin);
    if let Some(so) = so.as_ref() {
        let lib = so.parent().expect("librgmin.so parent");
        cmd.arg("-L")
            .arg(lib)
            .arg("-lrgmin")
            .arg(format!("-Wl,-rpath,{}", lib.display()));
    } else {
        cmd.arg(a.expect("static librgmin"))
            .arg("-lpthread")
            .arg("-ldl")
            .arg("-lm");
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
}
