fn main() {
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=include/");
    println!("cargo:rustc-check-cfg=cfg(rgmin_has_slepc)");
    println!("cargo:rustc-check-cfg=cfg(rgmin_has_primme)");
    println!("cargo:rustc-check-cfg=cfg(rgmin_has_chase)");
    println!("cargo:rustc-check-cfg=cfg(rgmin_has_libkrylov)");
    #[cfg(feature = "slepc")]
    probe_slepc();
    #[cfg(feature = "primme")]
    probe_primme();
    #[cfg(feature = "chase")]
    probe_chase();
    #[cfg(feature = "libkrylov")]
    probe_libkrylov();
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

#[cfg(feature = "primme")]
struct PrimmeProbe {
    includes: Vec<std::path::PathBuf>,
    link_paths: Vec<std::path::PathBuf>,
    link_libs: Vec<String>,
}

#[cfg(feature = "primme")]
fn probe_primme() {
    println!("cargo:rerun-if-changed=src/primme_shim.c");
    println!("cargo:rerun-if-env-changed=PRIMME_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    let Some(cfg) = discover_primme() else {
        println!(
            "cargo:warning=primme feature on, libprimme not found; EigensolverKind::Primme stays EigenUnavailable"
        );
        return;
    };
    let mut build = cc::Build::new();
    build.file("src/primme_shim.c");
    build.warnings(false);
    for inc in &cfg.includes {
        build.include(inc);
    }
    match build.try_compile("rgmin_primme_shim") {
        Ok(()) => {
            for path in &cfg.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
            for lib in &cfg.link_libs {
                println!("cargo:rustc-link-lib={lib}");
            }
            println!("cargo:rustc-cfg=rgmin_has_primme");
        }
        Err(err) => {
            println!(
                "cargo:warning=primme shim did not compile ({err}); EigensolverKind::Primme stays EigenUnavailable"
            );
        }
    }
}

#[cfg(feature = "primme")]
fn discover_primme() -> Option<PrimmeProbe> {
    if let Ok(lib) = pkg_config::Config::new()
        .atleast_version("3.0")
        .cargo_metadata(false)
        .probe("primme")
    {
        return Some(PrimmeProbe {
            includes: lib.include_paths,
            link_paths: lib.link_paths,
            link_libs: lib.libs,
        });
    }
    if let Ok(dir) = std::env::var("PRIMME_DIR") {
        if let Some(probe) = probe_primme_prefix(std::path::Path::new(&dir)) {
            return Some(probe);
        }
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        if let Some(probe) = probe_primme_prefix(std::path::Path::new(&prefix)) {
            return Some(probe);
        }
    }
    probe_primme_prefix(std::path::Path::new("/usr"))
}

#[cfg(feature = "primme")]
fn primme_header_in(dir: &std::path::Path) -> bool {
    dir.join("primme.h").is_file() || dir.join("primme").join("primme.h").is_file()
}

#[cfg(feature = "primme")]
fn probe_primme_prefix(prefix: &std::path::Path) -> Option<PrimmeProbe> {
    let include = prefix.join("include");
    if !primme_header_in(&include) {
        return None;
    }
    let lib = prefix.join("lib");
    Some(PrimmeProbe {
        includes: vec![include],
        link_paths: vec![lib],
        link_libs: vec!["primme".into()],
    })
}

#[cfg(feature = "chase")]
struct ChaseProbe {
    includes: Vec<std::path::PathBuf>,
    link_paths: Vec<std::path::PathBuf>,
    link_libs: Vec<String>,
}

#[cfg(feature = "chase")]
fn probe_chase() {
    println!("cargo:rerun-if-changed=src/chase_shim.c");
    println!("cargo:rerun-if-env-changed=CHASE_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    let Some(cfg) = discover_chase() else {
        println!(
            "cargo:warning=chase feature on, libchase not found; EigensolverKind::Chase stays EigenUnavailable"
        );
        return;
    };
    let mut build = cc::Build::new();
    build.file("src/chase_shim.c");
    build.warnings(false);
    for inc in &cfg.includes {
        build.include(inc);
    }
    match build.try_compile("rgmin_chase_shim") {
        Ok(()) => {
            for path in &cfg.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
            for lib in &cfg.link_libs {
                println!("cargo:rustc-link-lib={lib}");
            }
            println!("cargo:rustc-cfg=rgmin_has_chase");
        }
        Err(err) => {
            println!(
                "cargo:warning=chase shim did not compile ({err}); EigensolverKind::Chase stays EigenUnavailable"
            );
        }
    }
}

#[cfg(feature = "chase")]
fn discover_chase() -> Option<ChaseProbe> {
    for name in ["chase", "ChASE"] {
        if let Ok(lib) = pkg_config::Config::new()
            .cargo_metadata(false)
            .probe(name)
        {
            return Some(ChaseProbe {
                includes: lib.include_paths,
                link_paths: lib.link_paths,
                link_libs: lib.libs,
            });
        }
    }
    if let Ok(dir) = std::env::var("CHASE_DIR") {
        if let Some(probe) = probe_chase_prefix(std::path::Path::new(&dir)) {
            return Some(probe);
        }
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        if let Some(probe) = probe_chase_prefix(std::path::Path::new(&prefix)) {
            return Some(probe);
        }
    }
    probe_chase_prefix(std::path::Path::new("/usr"))
}

#[cfg(feature = "chase")]
fn chase_header_in(dir: &std::path::Path) -> bool {
    dir.join("chase_c_interface.h").is_file()
        || dir.join("interface").join("chase_c_interface.h").is_file()
}

#[cfg(feature = "chase")]
fn probe_chase_prefix(prefix: &std::path::Path) -> Option<ChaseProbe> {
    let include = prefix.join("include");
    let lib = prefix.join("lib");
    let has_lib = lib.join("libchase.a").is_file()
        || lib.join("libchase.so").is_file()
        || lib.join("libChASE.so").is_file();
    if !has_lib && !chase_header_in(&include) {
        return None;
    }
    if !has_lib {
        return None;
    }
    Some(ChaseProbe {
        includes: vec![include],
        link_paths: vec![lib],
        link_libs: vec!["chase".into()],
    })
}

#[cfg(feature = "libkrylov")]
struct LibkrylovProbe {
    includes: Vec<std::path::PathBuf>,
    link_paths: Vec<std::path::PathBuf>,
    link_libs: Vec<String>,
}

#[cfg(feature = "libkrylov")]
fn probe_libkrylov() {
    println!("cargo:rerun-if-changed=src/libkrylov_shim.c");
    println!("cargo:rerun-if-env-changed=LIBKRYLOV_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    let Some(cfg) = discover_libkrylov() else {
        println!(
            "cargo:warning=libkrylov feature on, libkrylov not found; EigensolverKind::Libkrylov stays EigenUnavailable"
        );
        return;
    };
    let mut build = cc::Build::new();
    build.file("src/libkrylov_shim.c");
    build.warnings(false);
    for inc in &cfg.includes {
        build.include(inc);
    }
    match build.try_compile("rgmin_libkrylov_shim") {
        Ok(()) => {
            for path in &cfg.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
            for lib in &cfg.link_libs {
                println!("cargo:rustc-link-lib={lib}");
            }
            println!("cargo:rustc-cfg=rgmin_has_libkrylov");
        }
        Err(err) => {
            println!(
                "cargo:warning=libkrylov shim did not compile ({err}); EigensolverKind::Libkrylov stays EigenUnavailable"
            );
        }
    }
}

#[cfg(feature = "libkrylov")]
fn discover_libkrylov() -> Option<LibkrylovProbe> {
    for name in ["krylov", "libkrylov"] {
        if let Ok(lib) = pkg_config::Config::new()
            .cargo_metadata(false)
            .probe(name)
        {
            let mut link_paths = lib.link_paths;
            let mut link_libs = lib.libs;
            if !link_libs.iter().any(|n| n == "openblas" || n == "blas") {
                if let Some((path, name)) = discover_openblas() {
                    if !link_paths.iter().any(|p| p == &path) {
                        link_paths.push(path);
                    }
                    link_libs.push(name);
                }
            }
            if !link_libs.iter().any(|n| n == "gfortran") {
                link_libs.push("gfortran".into());
            }
            return Some(LibkrylovProbe {
                includes: lib.include_paths,
                link_paths,
                link_libs,
            });
        }
    }
    if let Ok(dir) = std::env::var("LIBKRYLOV_DIR") {
        if let Some(probe) = probe_libkrylov_prefix(std::path::Path::new(&dir)) {
            return Some(probe);
        }
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        if let Some(probe) = probe_libkrylov_prefix(std::path::Path::new(&prefix)) {
            return Some(probe);
        }
    }
    probe_libkrylov_prefix(std::path::Path::new("/usr"))
}

#[cfg(feature = "libkrylov")]
fn libkrylov_header_in(dir: &std::path::Path) -> bool {
    dir.join("ckrylov.h").is_file() || dir.join("krylov").join("ckrylov.h").is_file()
}

#[cfg(feature = "libkrylov")]
fn probe_libkrylov_prefix(prefix: &std::path::Path) -> Option<LibkrylovProbe> {
    let include = prefix.join("include");
    let lib = prefix.join("lib");
    let has_lib = lib.join("libkrylov.a").is_file()
        || lib.join("libkrylov.so").is_file()
        || lib.join("libkrylov.dylib").is_file();
    if !has_lib || !libkrylov_header_in(&include) {
        return None;
    }
    let mut link_paths = vec![lib];
    let mut link_libs = vec!["krylov".into(), "gfortran".into()];
    if let Some((path, name)) = discover_openblas() {
        if !link_paths.iter().any(|p| p == &path) {
            link_paths.push(path);
        }
        link_libs.push(name);
    } else {
        link_libs.push("lapack".into());
        link_libs.push("blas".into());
    }
    Some(LibkrylovProbe {
        includes: vec![include],
        link_paths,
        link_libs,
    })
}

#[cfg(feature = "libkrylov")]
fn discover_openblas() -> Option<(std::path::PathBuf, String)> {
    for root in [
        std::env::var_os("OPENBLAS_DIR").map(std::path::PathBuf::from),
        std::env::var_os("CONDA_PREFIX").map(std::path::PathBuf::from),
    ]
    .into_iter()
    .flatten()
    {
        let lib = root.join("lib");
        if lib.join("libopenblas.so").is_file() || lib.join("libopenblas.a").is_file() {
            return Some((lib, "openblas".into()));
        }
    }
    let usr = std::path::Path::new("/usr/lib");
    if usr.join("libopenblas.so").is_file() {
        return Some((usr.to_path_buf(), "openblas".into()));
    }
    None
}
