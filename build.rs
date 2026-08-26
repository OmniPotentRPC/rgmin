fn main() {
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=include/");
    println!("cargo:rustc-check-cfg=cfg(rgmin_has_slepc)");
    #[cfg(feature = "slepc")]
    probe_slepc();
}

#[cfg(feature = "slepc")]
struct SlepcProbe {
    includes: Vec<std::path::PathBuf>,
    link_paths: Vec<std::path::PathBuf>,
    link_libs: Vec<String>,
}

#[cfg(feature = "slepc")]
fn probe_slepc() {
    println!("cargo:rerun-if-changed=src/slepc_shim.c");
    println!("cargo:rerun-if-env-changed=PETSC_DIR");
    println!("cargo:rerun-if-env-changed=PETSC_ARCH");
    println!("cargo:rerun-if-env-changed=SLEPC_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    let Some(cfg) = discover_slepc() else {
        println!(
            "cargo:warning=slepc feature on, PETSc/SLEPc not found; EigensolverKind::Slepc stays EigenUnavailable"
        );
        return;
    };
    let mut build = cc::Build::new();
    build.file("src/slepc_shim.c");
    build.warnings(false);
    for inc in &cfg.includes {
        build.include(inc);
    }
    match build.try_compile("rgmin_slepc_shim") {
        Ok(()) => {
            for path in &cfg.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
            for lib in &cfg.link_libs {
                println!("cargo:rustc-link-lib={lib}");
            }
            println!("cargo:rustc-cfg=rgmin_has_slepc");
        }
        Err(err) => {
            println!(
                "cargo:warning=slepc shim did not compile ({err}); EigensolverKind::Slepc stays EigenUnavailable"
            );
        }
    }
}

#[cfg(feature = "slepc")]
fn discover_slepc() -> Option<SlepcProbe> {
    for name in ["slepc", "SLEPc"] {
        if let Ok(lib) = pkg_config::Config::new()
            .atleast_version("3.15")
            .cargo_metadata(false)
            .probe(name)
        {
            return Some(SlepcProbe {
                includes: lib.include_paths,
                link_paths: lib.link_paths,
                link_libs: lib.libs,
            });
        }
    }
    if let Some(probe) = probe_petsc_dirs() {
        return Some(probe);
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        if let Some(probe) = probe_prefix(std::path::Path::new(&prefix)) {
            return Some(probe);
        }
    }
    probe_prefix(std::path::Path::new("/usr"))
}

#[cfg(feature = "slepc")]
fn header_in(dir: &std::path::Path) -> bool {
    dir.join("slepceps.h").is_file() || dir.join("slepc").join("slepceps.h").is_file()
}

#[cfg(feature = "slepc")]
fn probe_prefix(prefix: &std::path::Path) -> Option<SlepcProbe> {
    let include = prefix.join("include");
    if !header_in(&include) {
        return None;
    }
    let lib = prefix.join("lib");
    Some(SlepcProbe {
        includes: vec![include],
        link_paths: vec![lib],
        link_libs: vec!["slepc".into(), "petsc".into()],
    })
}

#[cfg(feature = "slepc")]
fn probe_petsc_dirs() -> Option<SlepcProbe> {
    let petsc = std::env::var_os("PETSC_DIR")?;
    let petsc = std::path::PathBuf::from(petsc);
    let arch = std::env::var("PETSC_ARCH").unwrap_or_default();
    let slepc = std::env::var_os("SLEPC_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| petsc.clone());
    let mut includes = vec![petsc.join("include"), slepc.join("include")];
    let mut link_paths = vec![petsc.join("lib"), slepc.join("lib")];
    if !arch.is_empty() {
        includes.push(petsc.join(&arch).join("include"));
        includes.push(slepc.join(&arch).join("include"));
        link_paths.push(petsc.join(&arch).join("lib"));
        link_paths.push(slepc.join(&arch).join("lib"));
    }
    if !includes.iter().any(|d| header_in(d)) {
        return None;
    }
    Some(SlepcProbe {
        includes,
        link_paths,
        link_libs: vec!["slepc".into(), "petsc".into()],
    })
}
