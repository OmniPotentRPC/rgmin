//! Compile the canonical compatibility header as C using the public include tree.

#![cfg(feature = "capi")]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn library_dir(root: &Path) -> PathBuf {
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target"))
        .join(profile)
}

#[test]
fn public_c_header_compiles_and_checks_the_abi() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let lib = library_dir(&root);
    let so = lib.join("librgmin.so");
    let archive = lib.join("librgmin.a");
    assert!(
        so.is_file() || archive.is_file(),
        "missing library under {}",
        lib.display()
    );
    let output = env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir)
        .join("rgmin_public_c_header");
    let mut command = Command::new("cc");
    command
        .arg("-std=c11")
        .arg(root.join("tests/ffi_smoke.c"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-o")
        .arg(&output)
        .env_remove("CPATH")
        .env_remove("C_INCLUDE_PATH");
    if so.is_file() {
        command
            .arg("-L")
            .arg(&lib)
            .arg("-lrgmin")
            .arg(format!("-Wl,-rpath,{}", lib.display()));
    } else {
        command
            .arg(&archive)
            .arg("-lpthread")
            .arg("-ldl")
            .arg("-lm");
    }
    let compiled = command.output().expect("spawn C compiler");
    assert!(
        compiled.status.success(),
        "C compiler failed:\n{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let ran = Command::new(&output)
        .output()
        .expect("run public C consumer");
    assert!(
        ran.status.success(),
        "C consumer exit {}:\n{}\n{}",
        ran.status,
        String::from_utf8_lossy(&ran.stdout),
        String::from_utf8_lossy(&ran.stderr)
    );
}
