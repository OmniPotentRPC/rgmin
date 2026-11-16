//! Compile the canonical compatibility header as C using the public include tree.

#![cfg(feature = "capi")]

use std::env;
use std::mem::{offset_of, size_of, size_of_val};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::null;

use rgmin::ffi::{
    RGMIN_PRECON_EXP, rgmin_conjugacy_t, rgmin_control_t, rgmin_eigen_options_t,
    rgmin_eigen_params_t, rgmin_exp_t, rgmin_precon_t, rgmin_scg_options_t, rgmin_scg_params_t,
    rgmin_scg_step_tolerance_t,
};
use rgmin::{Control, EigenParams, ExpParams, NoiseFloor, ScgOptions, ScgParams, ScgStepTolerance};

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

/// `bin`, behind the whitespace-separated `RGMIN_C_TEST_RUNNER` when that
/// is set, such as `valgrind --error-exitcode=99`.
fn consumer(bin: &Path) -> Command {
    let runner = env::var("RGMIN_C_TEST_RUNNER").unwrap_or_default();
    let mut words = runner.split_whitespace();
    let Some(program) = words.next() else {
        return Command::new(bin);
    };
    let mut command = Command::new(program);
    command.args(words).arg(bin);
    command
}

fn compile_and_run(source: &str, bin_name: &str, args: &[String]) {
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
        .join(bin_name);
    let mut command = Command::new("cc");
    command
        .arg("-std=c11")
        .arg(root.join(source))
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
    let ran = consumer(&output)
        .args(args)
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

#[test]
fn public_c_header_compiles_and_checks_the_abi() {
    compile_and_run("tests/ffi_smoke.c", "rgmin_public_c_header", &[]);
}

#[test]
fn c_header_reads_the_noise_floor_stop() {
    compile_and_run("tests/c_noise_floor.c", "rgmin_c_noise_floor", &[]);
}

#[cfg(feature = "highs")]
#[test]
fn c_header_routes_and_clears_the_highs_callback() {
    compile_and_run("tests/c_highs_callback.c", "rgmin_c_highs_callback", &[]);
}

/// Bit patterns of a field's elements, as the C side reads them.
trait Words {
    fn words(&self) -> Vec<u64>;
}

impl Words for f64 {
    fn words(&self) -> Vec<u64> {
        vec![self.to_bits()]
    }
}

impl Words for usize {
    fn words(&self) -> Vec<u64> {
        vec![*self as u64]
    }
}

impl Words for u32 {
    fn words(&self) -> Vec<u64> {
        vec![u64::from(*self)]
    }
}

impl Words for i32 {
    fn words(&self) -> Vec<u64> {
        vec![u64::from(*self as u32)]
    }
}

impl<T> Words for *const T {
    fn words(&self) -> Vec<u64> {
        vec![*self as usize as u64]
    }
}

impl<const N: usize> Words for [f64; N] {
    fn words(&self) -> Vec<u64> {
        self.iter().map(|v| v.to_bits()).collect()
    }
}

/// `NAME:SIZE` for the record, then `NAME:OFFSET:SIZE:BITS` for each field
/// element. The struct literal names every field, so a field added to a
/// record does not compile until it has a default here.
macro_rules! record {
    ($ty:ident { $($field:ident: $value:expr),* $(,)? }) => {{
        let want = $ty { $($field: $value),* };
        let mut args = vec![format!("{}:{}", stringify!($ty), size_of::<$ty>())];
        $(
            let words = Words::words(&want.$field);
            let size = size_of_val(&want.$field) / words.len();
            for (i, bits) in words.iter().enumerate() {
                let mut name = format!("{}.{}", stringify!($ty), stringify!($field));
                if words.len() > 1 {
                    name.push_str(&format!("[{i}]"));
                }
                let offset = offset_of!($ty, $field) + i * size;
                args.push(format!("{name}:{offset}:{size}:{bits:016x}"));
            }
        )*
        args
    }};
}

/// Each initializer, called over 0x40 stack bytes, gives the Rust default
/// in every field, as C reads it through the header.
#[test]
fn c_initializers_hold_the_rust_defaults() {
    let control = Control::default();
    let exp = ExpParams::default();
    let lengths = exp.lengths.unwrap_or([0.0; 3]);
    let scg = ScgParams::default();
    let eigen = EigenParams::default();
    let mut args = record!(rgmin_control_t {
        maxiter: control.maxiter,
        gtol: control.gtol,
        istep: control.istep,
        memory: 0,
        maxmove: control.maxmove.unwrap_or(0.0),
        noise_eps: control.noise_floor.map_or(0.0, |floor| floor.eps_g),
        noise_c: NoiseFloor::new(0.0).c,
    });
    args.extend(record!(rgmin_exp_t {
        a: exp.a,
        c_stab: exp.c_stab,
        mu: exp.mu.unwrap_or(f64::NAN),
        r_nn: exp.r_nn.unwrap_or(f64::NAN),
        r_cut: exp.r_cut.unwrap_or(f64::NAN),
        length_x: lengths[0],
        length_y: lengths[1],
        length_z: lengths[2],
        recalc_mu: i32::from(exp.recalc_mu),
    }));
    args.extend(record!(rgmin_scg_params_t {
        sigma0: scg.sigma0,
        lambda: scg.lambda,
        lambda_limit: scg.lambda_limit,
        tol_sol: scg.tol_sol,
        tol_func: scg.tol_func,
        conjugacy: rgmin_conjugacy_t::RGMIN_CONJUGACY_POLAK_RIBIERE as i32,
    }));
    let step_tolerance = match ScgOptions::default().step_tolerance {
        ScgStepTolerance::AbsoluteInfinity => {
            rgmin_scg_step_tolerance_t::RGMIN_SCG_ABSOLUTE_INFINITY
        }
        ScgStepTolerance::RelativeEuclidean => {
            rgmin_scg_step_tolerance_t::RGMIN_SCG_RELATIVE_EUCLIDEAN
        }
    };
    args.extend(record!(rgmin_scg_options_t {
        step_tolerance: step_tolerance as i32,
    }));
    args.extend(record!(rgmin_eigen_params_t {
        kind: eigen.kind as i32,
        nev: eigen.nev as u32,
        krylov: eigen.krylov as u32,
        max_iter: eigen.max_iter as u32,
        tol: eigen.tol,
    }));
    args.extend(record!(rgmin_eigen_options_t {
        degree: eigen.degree as u32,
        extra: eigen.extra as u32,
    }));
    args.extend(record!(rgmin_precon_t {
        struct_size: size_of::<rgmin_precon_t>() as u32,
        kind: RGMIN_PRECON_EXP,
        c_stab: f64::NAN,
        mu: f64::NAN,
        r_cut: f64::NAN,
        cell: [0.0; 3],
        numbers: null(),
        natoms: 0,
        ghost_z: null(),
        ghost_pos: null(),
        nghost: 0,
        nbr_i: null(),
        nbr_j: null(),
        nbr_r: null(),
        npairs: 0,
        anchor_i: null(),
        anchor_r: null(),
        nanchor: 0,
        exp_a: f64::NAN,
        bonds: null(),
        nbonds: 0,
        angles: null(),
        nangles: 0,
        dihedrals: 0,
        geometry_kind: 0,
    }));
    compile_and_run("tests/c_defaults.c", "rgmin_c_defaults", &args);
}
