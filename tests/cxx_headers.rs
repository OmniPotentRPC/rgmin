//! Compile and run the canonical and compatibility C++ headers.

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

fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn search_dirs(root: &Path) -> [PathBuf; 2] {
    let base = if let Ok(td) = env::var("CARGO_TARGET_DIR") {
        PathBuf::from(td).join(profile())
    } else {
        root.join("target").join(profile())
    };
    let deps = base.join("deps");
    [base, deps]
}

fn newest_file(paths: impl IntoIterator<Item = PathBuf>) -> Option<PathBuf> {
    paths
        .into_iter()
        .filter(|path| path.is_file())
        .max_by_key(|path| path.metadata().and_then(|meta| meta.modified()).ok())
}

#[test]
fn cxx_headers_compile_link_and_reuse() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = root.join("tests/cxx_headers.cpp");
    let dirs = search_dirs(&root);
    let so = newest_file(dirs.iter().map(|dir| dir.join("librgmin.so")));
    let a = newest_file(dirs.iter().map(|dir| dir.join("librgmin.a")));
    assert!(
        so.is_some() || a.is_some(),
        "librgmin missing under {} or {} (so={} a={})",
        dirs[0].display(),
        dirs[1].display(),
        so.is_some(),
        a.is_some()
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
        stdout.contains("pairs=1"),
        "cxx smoke did not report retained curvature: {stdout}"
    );
}
