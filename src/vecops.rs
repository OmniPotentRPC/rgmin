//! The vector seam: every O(n) operation a solver performs, in one
//! place.
#![allow(clippy::assign_op_pattern)]
//!
//! TAO writes each algorithm against PETSc `Vec` -- `VecDot`,
//! `VecAXPY`, `VecNorm` -- and that seam is its entire scaling story:
//! one `blmvm.c` runs serial, distributed, or on device because the
//! solver never touches storage. Here the objective side already has
//! its seam (`eindir` supplies `f` and `g`); this module is the
//! matching one for the solver's own algebra. Algorithms call these
//! functions; storage-specific execution lives behind them.
//!
//! Contiguous `dot` and `sum` follow ndarray 0.17
//! `numeric_util::unrolled_dot` and `unrolled_fold`: eight partials
//! `p0`..`p7`, then `(p0+p4)`, `(p1+p5)`, `(p2+p6)`, `(p3+p7)`, then
//! the tail in element order. Multiply and add stay separate.
//! The `simd` feature compiles that loop with `multiversion` 0.9 for
//! `x86_64+avx512f` (rustc >= 1.89), `x86_64+avx2`, `aarch64+sve`,
//! and `aarch64+neon`, plus the default clone. `RGMIN_SIMD` is
//! `scalar`, `avx2`, `avx512`, `neon`, or `sve`. An unknown name is
//! ignored. A name this CPU cannot run selects scalar. AVX2 is taken
//! only when FMA is also present; FMA is not enabled on the clone.
//! The `par` feature keeps that same tree: eight lane folds, fixed
//! chunks of 4096, combined in index order. Element updates use the
//! same chunk size. Thread count does not change a bit.

use std::cell::Cell;
use std::sync::OnceLock;

use ndarray::{Array1, ArrayView1};

/// Eight partials, the ndarray 0.17 fold width.
const W: usize = 8;
/// Fixed `par` chunk length. A multiple of [`W`], independent of the pool size.
#[cfg(feature = "par")]
const PAR_CHUNK: usize = 4096;

/// Below this length the par feature stays on the serial path: on the
/// reference host the rayon reductions measured 2.6x slower than
/// serial at n = 20000 (thread coordination outweighs O(n) arithmetic),
/// so parallel length-n algebra engages only where it has a chance.
#[cfg(feature = "par")]
pub(crate) const PAR_MIN_LEN: usize = 65_536;

/// Kernel clone selected for this thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    Scalar,
    Avx2,
    Avx512,
    Neon,
    Sve,
}

impl Path {
    fn name(self) -> &'static str {
        match self {
            Path::Scalar => "scalar",
            Path::Avx2 => "avx2",
            Path::Avx512 => "avx512",
            Path::Neon => "neon",
            Path::Sve => "sve",
        }
    }

    fn usable(self) -> bool {
        match self {
            Path::Scalar => true,
            Path::Avx2 => avx2_ready(),
            Path::Avx512 => avx512_ready(),
            Path::Neon => neon_ready(),
            Path::Sve => sve_ready(),
        }
    }
}

#[derive(Clone, Copy)]
enum Choice {
    Auto,
    Forced(Path),
}

thread_local! {
    static FORCE: Cell<Option<Path>> = const { Cell::new(None) };
}

fn parse(name: &str) -> Option<Path> {
    match name {
        "scalar" | "0" => Some(Path::Scalar),
        "avx2" | "avx2fma" => Some(Path::Avx2),
        "avx512" | "avx512f" => Some(Path::Avx512),
        "neon" => Some(Path::Neon),
        "sve" => Some(Path::Sve),
        _ => None,
    }
}

/// `RGMIN_SIMD`, read once. `None` means no override.
fn env_path() -> Option<Path> {
    static CACHED: OnceLock<Option<Path>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var("RGMIN_SIMD").ok()?;
        let path = parse(raw.trim())?;
        Some(if path.usable() { path } else { Path::Scalar })
    })
}

/// Override on this thread, else the cached environment, else multiversion.
fn choice() -> Choice {
    if let Some(path) = FORCE.with(|slot| slot.get()) {
        return Choice::Forced(if path.usable() { path } else { Path::Scalar });
    }
    if let Some(path) = env_path() {
        return Choice::Forced(path);
    }
    Choice::Auto
}

fn avx2_ready() -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        false
    }
}

fn avx512_ready() -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
    {
        false
    }
}

fn neon_ready() -> bool {
    cfg!(all(feature = "simd", target_feature = "neon"))
}

fn sve_ready() -> bool {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        std::arch::is_aarch64_feature_detected!("sve")
    }
    #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
    {
        false
    }
}

/// AVX2 without FMA would be multiversion's pick on that CPU.
/// The clone stays off; FMA is only a gate.
fn scalar_gate() -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if avx512_ready() {
            return false;
        }
        let avx2 = std::is_x86_feature_detected!("avx2");
        let fma = std::is_x86_feature_detected!("fma");
        avx2 && !fma
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        false
    }
}

/// Force one clone on this thread. Unknown names do nothing.
///
/// Returns whether that clone can run here. A known name this CPU
/// cannot run forces scalar and returns false: the path was not exercised.
#[doc(hidden)]
pub fn force_simd_path(name: &str) -> bool {
    let Some(path) = parse(name) else {
        return false;
    };
    let ok = path.usable();
    FORCE.with(|slot| slot.set(Some(if ok { path } else { Path::Scalar })));
    ok
}

/// Drop the per-thread override. The cached `RGMIN_SIMD` value stays.
#[doc(hidden)]
pub fn clear_simd_path() {
    FORCE.with(|slot| slot.set(None));
}

/// Whether this build and this CPU can execute `name`.
#[doc(hidden)]
pub fn simd_clone_usable(name: &str) -> bool {
    parse(name).is_some_and(|path| path.usable())
}

/// `auto` or the override currently in force on this thread.
#[doc(hidden)]
pub fn active_simd_path() -> &'static str {
    match choice() {
        Choice::Auto => "auto",
        Choice::Forced(path) => path.name(),
    }
}

macro_rules! dot_w8 {
    ($xs:expr, $ys:expr) => {{
        let xs: &[f64] = $xs;
        let ys: &[f64] = $ys;
        let n = xs.len();
        let mut i = 0;
        let mut sum = 0.0f64;
        let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        while i + W <= n {
            // SAFETY: `i + 7 < n` because `i + W <= n`.
            unsafe {
                p0 = p0 + *xs.get_unchecked(i) * *ys.get_unchecked(i);
                p1 = p1 + *xs.get_unchecked(i + 1) * *ys.get_unchecked(i + 1);
                p2 = p2 + *xs.get_unchecked(i + 2) * *ys.get_unchecked(i + 2);
                p3 = p3 + *xs.get_unchecked(i + 3) * *ys.get_unchecked(i + 3);
                p4 = p4 + *xs.get_unchecked(i + 4) * *ys.get_unchecked(i + 4);
                p5 = p5 + *xs.get_unchecked(i + 5) * *ys.get_unchecked(i + 5);
                p6 = p6 + *xs.get_unchecked(i + 6) * *ys.get_unchecked(i + 6);
                p7 = p7 + *xs.get_unchecked(i + 7) * *ys.get_unchecked(i + 7);
            }
            i += W;
        }
        sum = sum + (p0 + p4);
        sum = sum + (p1 + p5);
        sum = sum + (p2 + p6);
        sum = sum + (p3 + p7);
        while i < n {
            // SAFETY: `i < n`.
            unsafe {
                sum = sum + *xs.get_unchecked(i) * *ys.get_unchecked(i);
            }
            i += 1;
        }
        sum
    }};
}

macro_rules! sum_w8 {
    ($xs:expr) => {{
        let xs: &[f64] = $xs;
        let n = xs.len();
        let mut i = 0;
        let mut acc = 0.0f64;
        let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        while i + W <= n {
            // SAFETY: `i + 7 < n` because `i + W <= n`.
            unsafe {
                p0 = p0 + *xs.get_unchecked(i);
                p1 = p1 + *xs.get_unchecked(i + 1);
                p2 = p2 + *xs.get_unchecked(i + 2);
                p3 = p3 + *xs.get_unchecked(i + 3);
                p4 = p4 + *xs.get_unchecked(i + 4);
                p5 = p5 + *xs.get_unchecked(i + 5);
                p6 = p6 + *xs.get_unchecked(i + 6);
                p7 = p7 + *xs.get_unchecked(i + 7);
            }
            i += W;
        }
        acc = acc + (p0 + p4);
        acc = acc + (p1 + p5);
        acc = acc + (p2 + p6);
        acc = acc + (p3 + p7);
        while i < n {
            // SAFETY: `i < n`.
            unsafe {
                acc = acc + *xs.get_unchecked(i);
            }
            i += 1;
        }
        acc
    }};
}

macro_rules! axpy_body {
    ($a:expr, $xs:expr, $ys:expr) => {{
        let a: f64 = $a;
        let xs: &[f64] = $xs;
        let ys: &mut [f64] = $ys;
        for i in 0..xs.len() {
            ys[i] = ys[i] + (a * xs[i]);
        }
    }};
}

macro_rules! scale_body {
    ($a:expr, $ys:expr) => {{
        let a: f64 = $a;
        let ys: &mut [f64] = $ys;
        for p in ys.iter_mut() {
            *p = *p * a;
        }
    }};
}

macro_rules! mul_body {
    ($xs:expr, $ys:expr) => {{
        let xs: &[f64] = $xs;
        let ys: &mut [f64] = $ys;
        let n = xs.len().min(ys.len());
        for i in 0..n {
            ys[i] = ys[i] * xs[i];
        }
    }};
}

macro_rules! div_body {
    ($xs:expr, $ys:expr, $floor:expr) => {{
        let floor: f64 = $floor;
        let xs: &[f64] = $xs;
        let ys: &mut [f64] = $ys;
        let n = xs.len().min(ys.len());
        for i in 0..n {
            ys[i] = ys[i] / xs[i].max(floor);
        }
    }};
}

macro_rules! define_kernels {
    (
        fn $mv:ident($($arg:ident: $ty:ty),*) -> $ret:ty,
        avx2 $avx2:ident,
        avx512 $avx512:ident,
        neon $neon:ident,
        sve $sve:ident,
        $body:ident
    ) => {
        #[cfg(all(feature = "simd", rgmin_avx512))]
        #[inline(never)]
        #[multiversion::multiversion(targets(
            "x86_64+avx512f",
            "x86_64+avx2",
            "aarch64+sve",
            "aarch64+neon"
        ))]
        fn $mv($($arg: $ty),*) -> $ret {
            $body!($($arg),*)
        }

        #[cfg(all(feature = "simd", not(rgmin_avx512)))]
        #[inline(never)]
        #[multiversion::multiversion(targets("x86_64+avx2", "aarch64+sve", "aarch64+neon"))]
        fn $mv($($arg: $ty),*) -> $ret {
            $body!($($arg),*)
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        unsafe fn $avx2($($arg: $ty),*) -> $ret {
            $body!($($arg),*)
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
        #[target_feature(enable = "avx512f")]
        #[inline(never)]
        unsafe fn $avx512($($arg: $ty),*) -> $ret {
            $body!($($arg),*)
        }

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        #[target_feature(enable = "neon")]
        #[inline(never)]
        unsafe fn $neon($($arg: $ty),*) -> $ret {
            $body!($($arg),*)
        }

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        #[target_feature(enable = "sve")]
        #[inline(never)]
        unsafe fn $sve($($arg: $ty),*) -> $ret {
            $body!($($arg),*)
        }
    };
}

define_kernels!(
    fn dot_mv(xs: &[f64], ys: &[f64]) -> f64,
    avx2 dot_avx2_version,
    avx512 dot_avx512_version,
    neon dot_neon_version,
    sve dot_sve_version,
    dot_w8
);
define_kernels!(
    fn sum_mv(xs: &[f64]) -> f64,
    avx2 sum_avx2_version,
    avx512 sum_avx512_version,
    neon sum_neon_version,
    sve sum_sve_version,
    sum_w8
);
define_kernels!(
    fn axpy_mv(a: f64, xs: &[f64], ys: &mut [f64]) -> (),
    avx2 axpy_avx2_version,
    avx512 axpy_avx512_version,
    neon axpy_neon_version,
    sve axpy_sve_version,
    axpy_body
);
define_kernels!(
    fn scale_mv(a: f64, ys: &mut [f64]) -> (),
    avx2 scale_avx2_version,
    avx512 scale_avx512_version,
    neon scale_neon_version,
    sve scale_sve_version,
    scale_body
);
define_kernels!(
    fn mul_mv(xs: &[f64], ys: &mut [f64]) -> (),
    avx2 mul_avx2_version,
    avx512 mul_avx512_version,
    neon mul_neon_version,
    sve mul_sve_version,
    mul_body
);
define_kernels!(
    fn div_mv(xs: &[f64], ys: &mut [f64], floor: f64) -> (),
    avx2 div_avx2_version,
    avx512 div_avx512_version,
    neon div_neon_version,
    sve div_sve_version,
    div_body
);

macro_rules! dispatch_dot {
    ($choice:expr, $xs:expr, $ys:expr) => {{
        match $choice {
            Choice::Forced(Path::Scalar) => dot_w8!($xs, $ys),
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Choice::Forced(Path::Avx2) => unsafe { dot_avx2_version($xs, $ys) },
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            Choice::Forced(Path::Avx2) => dot_w8!($xs, $ys),
            #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
            Choice::Forced(Path::Avx512) => unsafe { dot_avx512_version($xs, $ys) },
            #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
            Choice::Forced(Path::Avx512) => dot_w8!($xs, $ys),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Choice::Forced(Path::Neon) => unsafe { dot_neon_version($xs, $ys) },
            #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
            Choice::Forced(Path::Neon) => dot_w8!($xs, $ys),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Choice::Forced(Path::Sve) => unsafe { dot_sve_version($xs, $ys) },
            #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
            Choice::Forced(Path::Sve) => dot_w8!($xs, $ys),
            #[cfg(feature = "simd")]
            Choice::Auto => {
                if scalar_gate() {
                    dot_w8!($xs, $ys)
                } else {
                    dot_mv($xs, $ys)
                }
            }
            #[cfg(not(feature = "simd"))]
            Choice::Auto => dot_w8!($xs, $ys),
        }
    }};
}

fn dot_contiguous(xs: &[f64], ys: &[f64]) -> f64 {
    let picked = choice();
    #[cfg(feature = "par")]
    if matches!(picked, Choice::Auto) && xs.len() >= PAR_MIN_LEN {
        return dot_parallel(xs, ys);
    }
    dispatch_dot!(picked, xs, ys)
}

macro_rules! dispatch_sum {
    ($choice:expr, $xs:expr) => {{
        match $choice {
            Choice::Forced(Path::Scalar) => sum_w8!($xs),
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Choice::Forced(Path::Avx2) => unsafe { sum_avx2_version($xs) },
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            Choice::Forced(Path::Avx2) => sum_w8!($xs),
            #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
            Choice::Forced(Path::Avx512) => unsafe { sum_avx512_version($xs) },
            #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
            Choice::Forced(Path::Avx512) => sum_w8!($xs),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Choice::Forced(Path::Neon) => unsafe { sum_neon_version($xs) },
            #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
            Choice::Forced(Path::Neon) => sum_w8!($xs),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Choice::Forced(Path::Sve) => unsafe { sum_sve_version($xs) },
            #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
            Choice::Forced(Path::Sve) => sum_w8!($xs),
            #[cfg(feature = "simd")]
            Choice::Auto => {
                if scalar_gate() {
                    sum_w8!($xs)
                } else {
                    sum_mv($xs)
                }
            }
            #[cfg(not(feature = "simd"))]
            Choice::Auto => sum_w8!($xs),
        }
    }};
}

fn sum_contiguous(xs: &[f64]) -> f64 {
    let picked = choice();
    #[cfg(feature = "par")]
    if matches!(picked, Choice::Auto) && xs.len() >= PAR_MIN_LEN {
        return sum_parallel(xs);
    }
    dispatch_sum!(picked, xs)
}

#[cfg(feature = "par")]
fn combine8(p: [f64; W]) -> f64 {
    let mut acc = 0.0;
    acc = acc + (p[0] + p[4]);
    acc = acc + (p[1] + p[5]);
    acc = acc + (p[2] + p[6]);
    acc = acc + (p[3] + p[7]);
    acc
}

#[cfg(feature = "par")]
fn dot_parallel(xs: &[f64], ys: &[f64]) -> f64 {
    use rayon::prelude::*;
    let n8 = xs.len() & !7;
    let partials: Vec<f64> = (0..W)
        .into_par_iter()
        .map(|lane| {
            let mut acc = 0.0;
            let mut base = 0;
            while base < n8 {
                let end = (base + PAR_CHUNK).min(n8);
                let mut i = base + lane;
                while i < end {
                    acc = acc + xs[i] * ys[i];
                    i += W;
                }
                base += PAR_CHUNK;
            }
            acc
        })
        .collect();
    let mut acc = combine8(partials.try_into().unwrap());
    let mut k = 0;
    while n8 + k < xs.len() && k < W - 1 {
        acc = acc + xs[n8 + k] * ys[n8 + k];
        k += 1;
    }
    acc
}

#[cfg(feature = "par")]
fn sum_parallel(xs: &[f64]) -> f64 {
    use rayon::prelude::*;
    let n8 = xs.len() & !7;
    let partials: Vec<f64> = (0..W)
        .into_par_iter()
        .map(|lane| {
            let mut acc = 0.0;
            let mut base = 0;
            while base < n8 {
                let end = (base + PAR_CHUNK).min(n8);
                let mut i = base + lane;
                while i < end {
                    acc = acc + xs[i];
                    i += W;
                }
                base += PAR_CHUNK;
            }
            acc
        })
        .collect();
    let mut acc = combine8(partials.try_into().unwrap());
    let mut k = 0;
    while n8 + k < xs.len() && k < W - 1 {
        acc = acc + xs[n8 + k];
        k += 1;
    }
    acc
}

#[cfg(feature = "par")]
fn par_chunks_zip(xs: &[f64], ys: &mut [f64], f: impl Fn(&[f64], &mut [f64]) + Sync + Send) {
    use rayon::prelude::*;
    if ys.len() < PAR_MIN_LEN {
        f(xs, ys);
        return;
    }
    ys.par_chunks_mut(PAR_CHUNK)
        .zip(xs.par_chunks(PAR_CHUNK))
        .for_each(|(cy, cx)| f(cx, cy));
}

#[cfg(feature = "par")]
fn par_chunks_mut(ys: &mut [f64], f: impl Fn(&mut [f64]) + Sync + Send) {
    use rayon::prelude::*;
    if ys.len() < PAR_MIN_LEN {
        f(ys);
        return;
    }
    ys.par_chunks_mut(PAR_CHUNK).for_each(f);
}

fn axpy_one(picked: Choice, a: f64, xs: &[f64], ys: &mut [f64]) {
    match picked {
        Choice::Forced(Path::Scalar) => axpy_body!(a, xs, ys),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        Choice::Forced(Path::Avx2) => unsafe { axpy_avx2_version(a, xs, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        Choice::Forced(Path::Avx2) => axpy_body!(a, xs, ys),
        #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
        Choice::Forced(Path::Avx512) => unsafe { axpy_avx512_version(a, xs, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
        Choice::Forced(Path::Avx512) => axpy_body!(a, xs, ys),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Neon) => unsafe { axpy_neon_version(a, xs, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Neon) => axpy_body!(a, xs, ys),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Sve) => unsafe { axpy_sve_version(a, xs, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Sve) => axpy_body!(a, xs, ys),
        #[cfg(feature = "simd")]
        Choice::Auto => {
            if scalar_gate() {
                axpy_body!(a, xs, ys);
            } else {
                axpy_mv(a, xs, ys);
            }
        }
        #[cfg(not(feature = "simd"))]
        Choice::Auto => axpy_body!(a, xs, ys),
    }
}

fn scale_one(picked: Choice, a: f64, ys: &mut [f64]) {
    match picked {
        Choice::Forced(Path::Scalar) => scale_body!(a, ys),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        Choice::Forced(Path::Avx2) => unsafe { scale_avx2_version(a, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        Choice::Forced(Path::Avx2) => scale_body!(a, ys),
        #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
        Choice::Forced(Path::Avx512) => unsafe { scale_avx512_version(a, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
        Choice::Forced(Path::Avx512) => scale_body!(a, ys),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Neon) => unsafe { scale_neon_version(a, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Neon) => scale_body!(a, ys),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Sve) => unsafe { scale_sve_version(a, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Sve) => scale_body!(a, ys),
        #[cfg(feature = "simd")]
        Choice::Auto => {
            if scalar_gate() {
                scale_body!(a, ys);
            } else {
                scale_mv(a, ys);
            }
        }
        #[cfg(not(feature = "simd"))]
        Choice::Auto => scale_body!(a, ys),
    }
}

fn mul_one(picked: Choice, xs: &[f64], ys: &mut [f64]) {
    match picked {
        Choice::Forced(Path::Scalar) => mul_body!(xs, ys),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        Choice::Forced(Path::Avx2) => unsafe { mul_avx2_version(xs, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        Choice::Forced(Path::Avx2) => mul_body!(xs, ys),
        #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
        Choice::Forced(Path::Avx512) => unsafe { mul_avx512_version(xs, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
        Choice::Forced(Path::Avx512) => mul_body!(xs, ys),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Neon) => unsafe { mul_neon_version(xs, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Neon) => mul_body!(xs, ys),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Sve) => unsafe { mul_sve_version(xs, ys) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Sve) => mul_body!(xs, ys),
        #[cfg(feature = "simd")]
        Choice::Auto => {
            if scalar_gate() {
                mul_body!(xs, ys);
            } else {
                mul_mv(xs, ys);
            }
        }
        #[cfg(not(feature = "simd"))]
        Choice::Auto => mul_body!(xs, ys),
    }
}

fn div_one(picked: Choice, xs: &[f64], ys: &mut [f64], floor: f64) {
    match picked {
        Choice::Forced(Path::Scalar) => div_body!(xs, ys, floor),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        Choice::Forced(Path::Avx2) => unsafe { div_avx2_version(xs, ys, floor) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        Choice::Forced(Path::Avx2) => div_body!(xs, ys, floor),
        #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
        Choice::Forced(Path::Avx512) => unsafe { div_avx512_version(xs, ys, floor) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
        Choice::Forced(Path::Avx512) => div_body!(xs, ys, floor),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Neon) => unsafe { div_neon_version(xs, ys, floor) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Neon) => div_body!(xs, ys, floor),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Sve) => unsafe { div_sve_version(xs, ys, floor) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Sve) => div_body!(xs, ys, floor),
        #[cfg(feature = "simd")]
        Choice::Auto => {
            if scalar_gate() {
                div_body!(xs, ys, floor);
            } else {
                div_mv(xs, ys, floor);
            }
        }
        #[cfg(not(feature = "simd"))]
        Choice::Auto => div_body!(xs, ys, floor),
    }
}

/// `x . y` summed from the first index upward.
///
/// [`dot`] uses the eight-partial tree. Noise projections and any
/// reduction that decides a step stay on this loop.
pub fn dot_ordered(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let mut acc = 0.0;
    for i in 0..x.len() {
        acc += x[i] * y[i];
    }
    acc
}

/// `x . y`.
pub fn dot(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    assert_eq!(x.len(), y.len());
    match (x.as_slice(), y.as_slice()) {
        (Some(xs), Some(ys)) => dot_contiguous(xs, ys),
        _ => x.dot(&y),
    }
}

/// `y += a x`.
pub fn axpy(a: f64, x: ArrayView1<f64>, y: &mut Array1<f64>) {
    assert_eq!(x.len(), y.len());
    let picked = choice();
    match (x.as_slice(), y.as_slice_mut()) {
        (Some(xs), Some(ys)) => {
            #[cfg(feature = "par")]
            {
                par_chunks_zip(xs, ys, move |cx, cy| axpy_one(picked, a, cx, cy));
            }
            #[cfg(not(feature = "par"))]
            {
                axpy_one(picked, a, xs, ys);
            }
        }
        _ => y.scaled_add(a, &x),
    }
}

/// `sum_i x_i`.
pub fn sum(x: ArrayView1<f64>) -> f64 {
    if let Some(xs) = x.as_slice() {
        sum_contiguous(xs)
    } else {
        x.sum()
    }
}

/// `||x||_2`.
pub fn nrm2(x: ArrayView1<f64>) -> f64 {
    dot(x, x).sqrt()
}

/// `y *= a`.
pub fn scale(a: f64, y: &mut Array1<f64>) {
    let picked = choice();
    match y.as_slice_mut() {
        Some(ys) => {
            #[cfg(feature = "par")]
            {
                par_chunks_mut(ys, move |cy| scale_one(picked, a, cy));
            }
            #[cfg(not(feature = "par"))]
            {
                scale_one(picked, a, ys);
            }
        }
        None => y.mapv_inplace(|v| v * a),
    }
}

/// `y[i] *= x[i]`.
pub fn mul_assign(x: ArrayView1<f64>, y: &mut Array1<f64>) {
    let n = x.len().min(y.len());
    let picked = choice();
    match (x.as_slice(), y.as_slice_mut()) {
        (Some(xs), Some(ys)) => {
            #[cfg(feature = "par")]
            {
                par_chunks_zip(&xs[..n], &mut ys[..n], move |cx, cy| {
                    mul_one(picked, cx, cy)
                });
            }
            #[cfg(not(feature = "par"))]
            {
                mul_one(picked, &xs[..n], &mut ys[..n]);
            }
        }
        _ => {
            for i in 0..n {
                y[i] *= x[i];
            }
        }
    }
}

/// `y[i] /= x[i].max(floor)`.
pub fn div_assign_floor(x: ArrayView1<f64>, y: &mut Array1<f64>, floor: f64) {
    let n = x.len().min(y.len());
    let picked = choice();
    match (x.as_slice(), y.as_slice_mut()) {
        (Some(xs), Some(ys)) => {
            #[cfg(feature = "par")]
            {
                par_chunks_zip(&xs[..n], &mut ys[..n], move |cx, cy| {
                    div_one(picked, cx, cy, floor);
                });
            }
            #[cfg(not(feature = "par"))]
            {
                div_one(picked, &xs[..n], &mut ys[..n], floor);
            }
        }
        _ => {
            for i in 0..n {
                y[i] /= x[i].max(floor);
            }
        }
    }
}

/// `||x||_inf`, infinity on any non-finite component: a broken vector
/// is never a small one.
pub fn nrminf(x: ArrayView1<f64>) -> f64 {
    fn fold(xs: &[f64]) -> f64 {
        let mut m = 0.0_f64;
        for v in xs {
            if !v.is_finite() {
                return f64::INFINITY;
            }
            m = m.max(v.abs());
        }
        m
    }
    let Some(xs) = x.as_slice() else {
        let mut m = 0.0_f64;
        for v in x.iter() {
            if !v.is_finite() {
                return f64::INFINITY;
            }
            m = m.max(v.abs());
        }
        return m;
    };
    #[cfg(feature = "par")]
    if xs.len() >= PAR_MIN_LEN {
        use rayon::prelude::*;
        let parts: Vec<f64> = xs.par_chunks(PAR_CHUNK).map(fold).collect();
        let mut m = 0.0_f64;
        for part in parts {
            if !part.is_finite() {
                return f64::INFINITY;
            }
            m = m.max(part);
        }
        return m;
    }
    fold(xs)
}

/// A solver-owned vector: storage plus the DLPack device it lives on.
///
/// This is the seam's storage handle (nbgu step of the TAO `Vec`
/// story): solvers hold `Vector`s and call the `v*` operations, which
/// dispatch on the device tag. The CPU arm wraps the crate's ndarray
/// storage at zero cost, so it is bit-identical to calling the slice
/// operations directly. A non-CPU tag needs a kernel backend behind
/// this seam; without one, construction fails loudly rather than
/// staging device data through the host, which the single-copy waist
/// contract forbids.
#[derive(Debug)]
pub struct Vector {
    data: Array1<f64>,
    device: dlpk::sys::DLDevice,
}

const CPU: dlpk::sys::DLDevice = dlpk::sys::DLDevice {
    device_type: dlpk::sys::DLDeviceType::kDLCPU,
    device_id: 0,
};

impl Vector {
    /// A zero CPU vector of length `n`.
    pub fn zeros_cpu(n: usize) -> Self {
        Self {
            data: Array1::zeros(n),
            device: CPU,
        }
    }

    /// Wrap host storage as a CPU vector. No copy.
    pub fn from_host(data: Array1<f64>) -> Self {
        Self { data, device: CPU }
    }

    /// Claim `data` for `device`. Only the CPU has a backend today;
    /// any other tag is refused so device data is never silently
    /// staged through the host.
    pub fn try_on(
        device: dlpk::sys::DLDevice,
        data: Array1<f64>,
    ) -> std::result::Result<Self, UnsupportedDevice> {
        if device.device_type == dlpk::sys::DLDeviceType::kDLCPU {
            Ok(Self { data, device })
        } else {
            Err(UnsupportedDevice {
                device_type: device.device_type as i32,
            })
        }
    }

    /// The device this vector lives on.
    pub fn device(&self) -> dlpk::sys::DLDevice {
        self.device
    }

    /// Host view of the CPU arm.
    pub fn host_view(&self) -> ArrayView1<'_, f64> {
        self.data.view()
    }

    /// Mutable host storage of the CPU arm.
    pub fn host_mut(&mut self) -> &mut Array1<f64> {
        &mut self.data
    }

    /// Unwrap into host storage.
    pub fn into_host(self) -> Array1<f64> {
        self.data
    }
}

/// The device tag names a backend this build does not carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsupportedDevice {
    /// The DLPack `DLDeviceType` discriminant that was refused.
    pub device_type: i32,
}

impl std::fmt::Display for UnsupportedDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DLPack device {} has no kernel backend in this build (CPU only)",
            self.device_type
        )
    }
}

impl std::error::Error for UnsupportedDevice {}

/// `x . y` on the vectors' device.
pub fn vdot(x: &Vector, y: &Vector) -> f64 {
    dot(x.data.view(), y.data.view())
}

/// `y += a x` on the vectors' device.
pub fn vaxpy(a: f64, x: &Vector, y: &mut Vector) {
    axpy(a, x.data.view(), &mut y.data);
}

/// `||x||_2` on the vector's device.
pub fn vnrm2(x: &Vector) -> f64 {
    nrm2(x.data.view())
}

/// `||x||_inf` on the vector's device, infinity on non-finite.
pub fn vnrminf(x: &Vector) -> f64 {
    nrminf(x.data.view())
}

/// `sum_i x_i` on the vector's device.
pub fn vsum(x: &Vector) -> f64 {
    sum(x.data.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn the_seam_matches_ndarray_on_the_serial_path() {
        let x = Array1::from((0..1000).map(|i| (i as f64).sin()).collect::<Vec<_>>());
        let y = Array1::from((0..1000).map(|i| (i as f64).cos()).collect::<Vec<_>>());
        let d = dot(x.view(), y.view());
        let mut z = y.clone();
        axpy(0.5, x.view(), &mut z);
        let mut reference = y.clone();
        reference.scaled_add(0.5, &x);
        assert!((d - x.dot(&y)).abs() < 1e-12 * d.abs().max(1.0));
        assert_eq!(z, reference);
        assert!((nrm2(x.view()) - x.dot(&x).sqrt()).abs() < 1e-12);
        assert!((sum(x.view()) - x.sum()).abs() < 1e-12);
    }

    #[test]
    fn the_infinity_norm_never_calls_a_broken_vector_small() {
        let x = Array1::from(vec![0.0, f64::NAN, 1.0]);
        assert!(nrminf(x.view()).is_infinite());
    }

    #[test]
    fn the_cpu_vector_is_the_slice_path_verbatim() {
        let a = Array1::from((0..64).map(|i| (i as f64).sin()).collect::<Vec<_>>());
        let b = Array1::from((0..64).map(|i| (i as f64).cos()).collect::<Vec<_>>());
        let vx = Vector::from_host(a.clone());
        let mut vy = Vector::from_host(b.clone());
        assert_eq!(vdot(&vx, &vy), dot(a.view(), b.view()));
        assert_eq!(vnrm2(&vx), nrm2(a.view()));
        vaxpy(0.25, &vx, &mut vy);
        let mut reference = b.clone();
        axpy(0.25, a.view(), &mut reference);
        assert_eq!(vy.into_host(), reference);
    }

    #[test]
    fn a_device_without_a_backend_is_refused_not_staged() {
        let cuda = dlpk::sys::DLDevice {
            device_type: dlpk::sys::DLDeviceType::kDLCUDA,
            device_id: 0,
        };
        let err = Vector::try_on(cuda, Array1::zeros(4)).unwrap_err();
        assert_eq!(err.device_type, dlpk::sys::DLDeviceType::kDLCUDA as i32);
    }
}
