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
//! Contiguous `dot` keeps eight partials `p0`..`p7`. Each product is
//! one `mul_add` into its partial. The partials then combine as
//! `(p0+p4)`, `(p1+p5)`, `(p2+p6)`, `(p3+p7)`, and the tail is one
//! `mul_add` per element. `sum` uses the same tree of plain adds.
//! The `simd` feature compiles that loop with `multiversion` 0.9 for
//! `x86_64+avx512f+fma` (rustc >= 1.89), `x86_64+avx2+fma`,
//! `aarch64+sve`, and `aarch64+neon`, plus the default clone.
//! `RGMIN_SIMD` is `scalar`, `avx2`, `avx512`, `neon`, or `sve`.
//! An unknown name is ignored. A name this CPU cannot run selects
//! the portable `mul_add` path. That path is the bit reference: a
//! hardware FMA and `f64::mul_add` round once, so every clone matches
//! it. The `par` feature folds each contiguous chunk of 4096 with
//! that same tree, then adds the chunk totals from the first index
//! upward. Element updates use the same chunk size. Thread count
//! does not change a bit.

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
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")
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

/// The AVX2 clone enables FMA. A CPU with AVX2 and no FMA stays on
/// the portable `mul_add` path, which has the same bits.
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
                p0 = (*xs.get_unchecked(i)).mul_add(*ys.get_unchecked(i), p0);
                p1 = (*xs.get_unchecked(i + 1)).mul_add(*ys.get_unchecked(i + 1), p1);
                p2 = (*xs.get_unchecked(i + 2)).mul_add(*ys.get_unchecked(i + 2), p2);
                p3 = (*xs.get_unchecked(i + 3)).mul_add(*ys.get_unchecked(i + 3), p3);
                p4 = (*xs.get_unchecked(i + 4)).mul_add(*ys.get_unchecked(i + 4), p4);
                p5 = (*xs.get_unchecked(i + 5)).mul_add(*ys.get_unchecked(i + 5), p5);
                p6 = (*xs.get_unchecked(i + 6)).mul_add(*ys.get_unchecked(i + 6), p6);
                p7 = (*xs.get_unchecked(i + 7)).mul_add(*ys.get_unchecked(i + 7), p7);
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
                sum = (*xs.get_unchecked(i)).mul_add(*ys.get_unchecked(i), sum);
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
            ys[i] = a.mul_add(xs[i], ys[i]);
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
            "x86_64+avx512f+fma",
            "x86_64+avx2+fma",
            "aarch64+sve",
            "aarch64+neon"
        ))]
        fn $mv($($arg: $ty),*) -> $ret {
            $body!($($arg),*)
        }

        #[cfg(all(feature = "simd", not(rgmin_avx512)))]
        #[inline(never)]
        #[multiversion::multiversion(targets(
            "x86_64+avx2+fma",
            "aarch64+sve",
            "aarch64+neon"
        ))]
        fn $mv($($arg: $ty),*) -> $ret {
            $body!($($arg),*)
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        unsafe fn $avx2($($arg: $ty),*) -> $ret {
            $body!($($arg),*)
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
        #[target_feature(enable = "avx512f,fma")]
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
        return dot_parallel(picked, xs, ys);
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
        return sum_parallel(picked, xs);
    }
    dispatch_sum!(picked, xs)
}

#[cfg(feature = "par")]
fn fold_in_index_order(parts: Vec<f64>) -> f64 {
    let mut acc = 0.0;
    for part in parts {
        acc = acc + part;
    }
    acc
}

#[cfg(feature = "par")]
fn dot_parallel(picked: Choice, xs: &[f64], ys: &[f64]) -> f64 {
    use rayon::prelude::*;
    let parts: Vec<f64> = xs
        .par_chunks(PAR_CHUNK)
        .zip(ys.par_chunks(PAR_CHUNK))
        .map(move |(cx, cy)| dispatch_dot!(picked, cx, cy))
        .collect();
    fold_in_index_order(parts)
}

#[cfg(feature = "par")]
fn sum_parallel(picked: Choice, xs: &[f64]) -> f64 {
    use rayon::prelude::*;
    let parts: Vec<f64> = xs
        .par_chunks(PAR_CHUNK)
        .map(move |cx| dispatch_sum!(picked, cx))
        .collect();
    fold_in_index_order(parts)
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
/// [`dot`] fuses each product into an eight-partial tree. This loop
/// keeps a separate multiply and add, in element order, for noise
/// projections.
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
        // A strided view has the same element order. Copying into the
        // contiguous kernel keeps the mul_add tree, including chunks.
        _ => {
            let xs: Vec<f64> = x.iter().copied().collect();
            let ys: Vec<f64> = y.iter().copied().collect();
            dot_contiguous(&xs, &ys)
        }
    }
}

/// `y += a x`, then `z · y` with the same eight-partial `mul_add` tree as [`dot`].
///
/// The axpy runs elementwise, so its visit order does not change a stored
/// value. The dot partials are updated in the same blocks as [`dot`], after
/// that element has been updated, which is the value a later [`dot`] reads.
pub(crate) fn axpy_dot(a: f64, x: ArrayView1<f64>, y: &mut Array1<f64>, z: ArrayView1<f64>) -> f64 {
    assert_eq!(x.len(), y.len());
    assert_eq!(z.len(), y.len());
    #[cfg(feature = "par")]
    if matches!(choice(), Choice::Auto) && y.len() >= PAR_MIN_LEN {
        axpy(a, x, y);
        return dot(z, y.view());
    }
    match (x.as_slice(), y.as_slice_mut(), z.as_slice()) {
        (Some(xs), Some(ys), Some(zs)) => dispatch_axpy_dot(a, xs, ys, zs),
        _ => {
            axpy(a, x, y);
            dot(z, y.view())
        }
    }
}

/// `y *= a`, then `z · y` with the eight-partial tree.
///
/// The scale is `y[i] = y[i] * a`, the operand order of an in-place scalar
/// multiply. Each dot partial then reads that updated element.
pub(crate) fn scale_dot(a: f64, y: &mut Array1<f64>, z: ArrayView1<f64>) -> f64 {
    assert_eq!(z.len(), y.len());
    #[cfg(feature = "par")]
    if matches!(choice(), Choice::Auto) && y.len() >= PAR_MIN_LEN {
        scale(a, y);
        return dot(z, y.view());
    }
    match (y.as_slice_mut(), z.as_slice()) {
        (Some(ys), Some(zs)) => dispatch_scale_dot(a, ys, zs),
        _ => {
            scale(a, y);
            dot(z, y.view())
        }
    }
}

/// Copy `src` into `dst`, the infinity norm of `src`, and `v · src`.
///
/// The norm uses the same eight running maxima as [`nrminf`]. The dot uses
/// the eight-partial tree on `src`, which is what `dst` holds afterwards.
pub(crate) fn copy_inf_dot(
    src: ArrayView1<f64>,
    dst: &mut Array1<f64>,
    v: ArrayView1<f64>,
) -> (f64, f64) {
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len(), v.len());
    #[cfg(feature = "par")]
    if matches!(choice(), Choice::Auto) && src.len() >= PAR_MIN_LEN {
        dst.assign(&src);
        return (nrminf(src), dot(v, src));
    }
    match (src.as_slice(), dst.as_slice_mut(), v.as_slice()) {
        (Some(ss), Some(ds), Some(vs)) => dispatch_copy_inf_dot(ss, ds, vs),
        _ => {
            dst.assign(&src);
            (nrminf(src), dot(v, src))
        }
    }
}

/// Copy `src` into `dst`, `src · src`, and `v · src`.
///
/// `all_finite` is false when any component is non-finite. Both dots use the
/// eight-partial `mul_add` tree.
pub(crate) fn copy_euclid_dot(
    src: ArrayView1<f64>,
    dst: &mut Array1<f64>,
    v: ArrayView1<f64>,
) -> (f64, f64, bool) {
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len(), v.len());
    #[cfg(feature = "par")]
    if matches!(choice(), Choice::Auto) && src.len() >= PAR_MIN_LEN {
        dst.assign(&src);
        let finite = src.iter().all(|t| t.is_finite());
        return (dot(src, src), dot(v, src), finite);
    }
    match (src.as_slice(), dst.as_slice_mut(), v.as_slice()) {
        (Some(ss), Some(ds), Some(vs)) => dispatch_copy_euclid_dot(ss, ds, vs),
        _ => {
            dst.assign(&src);
            let finite = src.iter().all(|t| t.is_finite());
            (dot(src, src), dot(v, src), finite)
        }
    }
}

/// ndarray `s · y`, ndarray `y · y`, and the left folds `Σ s²` and `Σ y²`.
///
/// The dots are eight partials of `p + a * b`, combined the way ndarray's
/// unrolled dot combines them. The square sums are one running add per
/// element, in index order, matching `iter().map(|v| v * v).sum()` for a
/// non-empty finite vector. An empty input returns `+0` for the dots and the
/// same empty-sum value as the iterator for the squares.
pub(crate) fn curvature_products(s: ArrayView1<f64>, y: ArrayView1<f64>) -> (f64, f64, f64, f64) {
    assert_eq!(s.len(), y.len());
    match (s.as_slice(), y.as_slice()) {
        (Some(ss), Some(ys)) => dispatch_curvature(ss, ys),
        _ => {
            let sy = s.dot(&y);
            let yy = y.dot(&y);
            let ss = s.iter().map(|v| v * v).sum();
            let yys = y.iter().map(|v| v * v).sum();
            (sy, yy, ss, yys)
        }
    }
}

#[inline(always)]
fn axpy_dot_lanes(a: f64, xs: &[f64], ys: &mut [f64], zs: &[f64]) -> f64 {
    let n = ys.len();
    let mut i = 0;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    while i + W <= n {
        unsafe {
            *ys.get_unchecked_mut(i) = a.mul_add(*xs.get_unchecked(i), *ys.get_unchecked(i));
            p0 = (*zs.get_unchecked(i)).mul_add(*ys.get_unchecked(i), p0);
            *ys.get_unchecked_mut(i + 1) =
                a.mul_add(*xs.get_unchecked(i + 1), *ys.get_unchecked(i + 1));
            p1 = (*zs.get_unchecked(i + 1)).mul_add(*ys.get_unchecked(i + 1), p1);
            *ys.get_unchecked_mut(i + 2) =
                a.mul_add(*xs.get_unchecked(i + 2), *ys.get_unchecked(i + 2));
            p2 = (*zs.get_unchecked(i + 2)).mul_add(*ys.get_unchecked(i + 2), p2);
            *ys.get_unchecked_mut(i + 3) =
                a.mul_add(*xs.get_unchecked(i + 3), *ys.get_unchecked(i + 3));
            p3 = (*zs.get_unchecked(i + 3)).mul_add(*ys.get_unchecked(i + 3), p3);
            *ys.get_unchecked_mut(i + 4) =
                a.mul_add(*xs.get_unchecked(i + 4), *ys.get_unchecked(i + 4));
            p4 = (*zs.get_unchecked(i + 4)).mul_add(*ys.get_unchecked(i + 4), p4);
            *ys.get_unchecked_mut(i + 5) =
                a.mul_add(*xs.get_unchecked(i + 5), *ys.get_unchecked(i + 5));
            p5 = (*zs.get_unchecked(i + 5)).mul_add(*ys.get_unchecked(i + 5), p5);
            *ys.get_unchecked_mut(i + 6) =
                a.mul_add(*xs.get_unchecked(i + 6), *ys.get_unchecked(i + 6));
            p6 = (*zs.get_unchecked(i + 6)).mul_add(*ys.get_unchecked(i + 6), p6);
            *ys.get_unchecked_mut(i + 7) =
                a.mul_add(*xs.get_unchecked(i + 7), *ys.get_unchecked(i + 7));
            p7 = (*zs.get_unchecked(i + 7)).mul_add(*ys.get_unchecked(i + 7), p7);
        }
        i += W;
    }
    let mut sum = 0.0;
    sum = sum + (p0 + p4);
    sum = sum + (p1 + p5);
    sum = sum + (p2 + p6);
    sum = sum + (p3 + p7);
    while i < n {
        unsafe {
            *ys.get_unchecked_mut(i) = a.mul_add(*xs.get_unchecked(i), *ys.get_unchecked(i));
            sum = (*zs.get_unchecked(i)).mul_add(*ys.get_unchecked(i), sum);
        }
        i += 1;
    }
    sum
}

#[inline(always)]
fn scale_dot_lanes(a: f64, ys: &mut [f64], zs: &[f64]) -> f64 {
    let n = ys.len();
    let mut i = 0;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    while i + W <= n {
        unsafe {
            *ys.get_unchecked_mut(i) = *ys.get_unchecked(i) * a;
            p0 = (*zs.get_unchecked(i)).mul_add(*ys.get_unchecked(i), p0);
            *ys.get_unchecked_mut(i + 1) = *ys.get_unchecked(i + 1) * a;
            p1 = (*zs.get_unchecked(i + 1)).mul_add(*ys.get_unchecked(i + 1), p1);
            *ys.get_unchecked_mut(i + 2) = *ys.get_unchecked(i + 2) * a;
            p2 = (*zs.get_unchecked(i + 2)).mul_add(*ys.get_unchecked(i + 2), p2);
            *ys.get_unchecked_mut(i + 3) = *ys.get_unchecked(i + 3) * a;
            p3 = (*zs.get_unchecked(i + 3)).mul_add(*ys.get_unchecked(i + 3), p3);
            *ys.get_unchecked_mut(i + 4) = *ys.get_unchecked(i + 4) * a;
            p4 = (*zs.get_unchecked(i + 4)).mul_add(*ys.get_unchecked(i + 4), p4);
            *ys.get_unchecked_mut(i + 5) = *ys.get_unchecked(i + 5) * a;
            p5 = (*zs.get_unchecked(i + 5)).mul_add(*ys.get_unchecked(i + 5), p5);
            *ys.get_unchecked_mut(i + 6) = *ys.get_unchecked(i + 6) * a;
            p6 = (*zs.get_unchecked(i + 6)).mul_add(*ys.get_unchecked(i + 6), p6);
            *ys.get_unchecked_mut(i + 7) = *ys.get_unchecked(i + 7) * a;
            p7 = (*zs.get_unchecked(i + 7)).mul_add(*ys.get_unchecked(i + 7), p7);
        }
        i += W;
    }
    let mut sum = 0.0;
    sum = sum + (p0 + p4);
    sum = sum + (p1 + p5);
    sum = sum + (p2 + p6);
    sum = sum + (p3 + p7);
    while i < n {
        unsafe {
            *ys.get_unchecked_mut(i) = *ys.get_unchecked(i) * a;
            sum = (*zs.get_unchecked(i)).mul_add(*ys.get_unchecked(i), sum);
        }
        i += 1;
    }
    sum
}

#[inline(always)]
fn copy_inf_dot_lanes(src: &[f64], dst: &mut [f64], v: &[f64]) -> (f64, f64) {
    let n = src.len();
    let mut i = 0;
    let (mut m0, mut m1, mut m2, mut m3, mut m4, mut m5, mut m6, mut m7) = (
        0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64,
    );
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    while i + W <= n {
        unsafe {
            let s0 = *src.get_unchecked(i);
            let s1 = *src.get_unchecked(i + 1);
            let s2 = *src.get_unchecked(i + 2);
            let s3 = *src.get_unchecked(i + 3);
            let s4 = *src.get_unchecked(i + 4);
            let s5 = *src.get_unchecked(i + 5);
            let s6 = *src.get_unchecked(i + 6);
            let s7 = *src.get_unchecked(i + 7);
            if !(s0.is_finite()
                && s1.is_finite()
                && s2.is_finite()
                && s3.is_finite()
                && s4.is_finite()
                && s5.is_finite()
                && s6.is_finite()
                && s7.is_finite())
            {
                dst[..n].copy_from_slice(src);
                return (f64::INFINITY, dot_slice(v, src));
            }
            *dst.get_unchecked_mut(i) = s0;
            *dst.get_unchecked_mut(i + 1) = s1;
            *dst.get_unchecked_mut(i + 2) = s2;
            *dst.get_unchecked_mut(i + 3) = s3;
            *dst.get_unchecked_mut(i + 4) = s4;
            *dst.get_unchecked_mut(i + 5) = s5;
            *dst.get_unchecked_mut(i + 6) = s6;
            *dst.get_unchecked_mut(i + 7) = s7;
            m0 = m0.max(s0.abs());
            m1 = m1.max(s1.abs());
            m2 = m2.max(s2.abs());
            m3 = m3.max(s3.abs());
            m4 = m4.max(s4.abs());
            m5 = m5.max(s5.abs());
            m6 = m6.max(s6.abs());
            m7 = m7.max(s7.abs());
            p0 = (*v.get_unchecked(i)).mul_add(s0, p0);
            p1 = (*v.get_unchecked(i + 1)).mul_add(s1, p1);
            p2 = (*v.get_unchecked(i + 2)).mul_add(s2, p2);
            p3 = (*v.get_unchecked(i + 3)).mul_add(s3, p3);
            p4 = (*v.get_unchecked(i + 4)).mul_add(s4, p4);
            p5 = (*v.get_unchecked(i + 5)).mul_add(s5, p5);
            p6 = (*v.get_unchecked(i + 6)).mul_add(s6, p6);
            p7 = (*v.get_unchecked(i + 7)).mul_add(s7, p7);
        }
        i += W;
    }
    let mut sum = 0.0;
    sum = sum + (p0 + p4);
    sum = sum + (p1 + p5);
    sum = sum + (p2 + p6);
    sum = sum + (p3 + p7);
    let mut m = m0.max(m1).max(m2).max(m3).max(m4).max(m5).max(m6).max(m7);
    while i < n {
        unsafe {
            let s = *src.get_unchecked(i);
            if !s.is_finite() {
                dst[..n].copy_from_slice(src);
                return (f64::INFINITY, dot_slice(v, src));
            }
            *dst.get_unchecked_mut(i) = s;
            m = m.max(s.abs());
            sum = (*v.get_unchecked(i)).mul_add(s, sum);
        }
        i += 1;
    }
    (m, sum)
}

#[inline(always)]
fn copy_euclid_dot_lanes(src: &[f64], dst: &mut [f64], v: &[f64]) -> (f64, f64, bool) {
    let n = src.len();
    let mut i = 0;
    let mut finite = true;
    let (mut g0, mut g1, mut g2, mut g3, mut g4, mut g5, mut g6, mut g7) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    while i + W <= n {
        unsafe {
            let s0 = *src.get_unchecked(i);
            let s1 = *src.get_unchecked(i + 1);
            let s2 = *src.get_unchecked(i + 2);
            let s3 = *src.get_unchecked(i + 3);
            let s4 = *src.get_unchecked(i + 4);
            let s5 = *src.get_unchecked(i + 5);
            let s6 = *src.get_unchecked(i + 6);
            let s7 = *src.get_unchecked(i + 7);
            finite &= s0.is_finite()
                && s1.is_finite()
                && s2.is_finite()
                && s3.is_finite()
                && s4.is_finite()
                && s5.is_finite()
                && s6.is_finite()
                && s7.is_finite();
            *dst.get_unchecked_mut(i) = s0;
            *dst.get_unchecked_mut(i + 1) = s1;
            *dst.get_unchecked_mut(i + 2) = s2;
            *dst.get_unchecked_mut(i + 3) = s3;
            *dst.get_unchecked_mut(i + 4) = s4;
            *dst.get_unchecked_mut(i + 5) = s5;
            *dst.get_unchecked_mut(i + 6) = s6;
            *dst.get_unchecked_mut(i + 7) = s7;
            g0 = s0.mul_add(s0, g0);
            g1 = s1.mul_add(s1, g1);
            g2 = s2.mul_add(s2, g2);
            g3 = s3.mul_add(s3, g3);
            g4 = s4.mul_add(s4, g4);
            g5 = s5.mul_add(s5, g5);
            g6 = s6.mul_add(s6, g6);
            g7 = s7.mul_add(s7, g7);
            p0 = (*v.get_unchecked(i)).mul_add(s0, p0);
            p1 = (*v.get_unchecked(i + 1)).mul_add(s1, p1);
            p2 = (*v.get_unchecked(i + 2)).mul_add(s2, p2);
            p3 = (*v.get_unchecked(i + 3)).mul_add(s3, p3);
            p4 = (*v.get_unchecked(i + 4)).mul_add(s4, p4);
            p5 = (*v.get_unchecked(i + 5)).mul_add(s5, p5);
            p6 = (*v.get_unchecked(i + 6)).mul_add(s6, p6);
            p7 = (*v.get_unchecked(i + 7)).mul_add(s7, p7);
        }
        i += W;
    }
    let mut gg = 0.0;
    gg = gg + (g0 + g4);
    gg = gg + (g1 + g5);
    gg = gg + (g2 + g6);
    gg = gg + (g3 + g7);
    let mut sg = 0.0;
    sg = sg + (p0 + p4);
    sg = sg + (p1 + p5);
    sg = sg + (p2 + p6);
    sg = sg + (p3 + p7);
    while i < n {
        unsafe {
            let s = *src.get_unchecked(i);
            finite &= s.is_finite();
            *dst.get_unchecked_mut(i) = s;
            gg = s.mul_add(s, gg);
            sg = (*v.get_unchecked(i)).mul_add(s, sg);
        }
        i += 1;
    }
    (gg, sg, finite)
}

#[inline(always)]
fn curvature_lanes(s: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    let n = s.len();
    if n == 0 {
        let empty: f64 = [].iter().map(|v: &f64| v * v).sum();
        return (0.0, 0.0, empty, empty);
    }
    let mut i = 0;
    let mut ss = 0.0;
    let mut yys = 0.0;
    let (mut sy0, mut sy1, mut sy2, mut sy3, mut sy4, mut sy5, mut sy6, mut sy7) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut yy0, mut yy1, mut yy2, mut yy3, mut yy4, mut yy5, mut yy6, mut yy7) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    while i + W <= n {
        unsafe {
            let s0 = *s.get_unchecked(i);
            let s1 = *s.get_unchecked(i + 1);
            let s2 = *s.get_unchecked(i + 2);
            let s3 = *s.get_unchecked(i + 3);
            let s4 = *s.get_unchecked(i + 4);
            let s5 = *s.get_unchecked(i + 5);
            let s6 = *s.get_unchecked(i + 6);
            let s7 = *s.get_unchecked(i + 7);
            let y0 = *y.get_unchecked(i);
            let y1 = *y.get_unchecked(i + 1);
            let y2 = *y.get_unchecked(i + 2);
            let y3 = *y.get_unchecked(i + 3);
            let y4 = *y.get_unchecked(i + 4);
            let y5 = *y.get_unchecked(i + 5);
            let y6 = *y.get_unchecked(i + 6);
            let y7 = *y.get_unchecked(i + 7);
            ss = ss + s0 * s0;
            ss = ss + s1 * s1;
            ss = ss + s2 * s2;
            ss = ss + s3 * s3;
            ss = ss + s4 * s4;
            ss = ss + s5 * s5;
            ss = ss + s6 * s6;
            ss = ss + s7 * s7;
            yys = yys + y0 * y0;
            yys = yys + y1 * y1;
            yys = yys + y2 * y2;
            yys = yys + y3 * y3;
            yys = yys + y4 * y4;
            yys = yys + y5 * y5;
            yys = yys + y6 * y6;
            yys = yys + y7 * y7;
            sy0 = sy0 + s0 * y0;
            sy1 = sy1 + s1 * y1;
            sy2 = sy2 + s2 * y2;
            sy3 = sy3 + s3 * y3;
            sy4 = sy4 + s4 * y4;
            sy5 = sy5 + s5 * y5;
            sy6 = sy6 + s6 * y6;
            sy7 = sy7 + s7 * y7;
            yy0 = yy0 + y0 * y0;
            yy1 = yy1 + y1 * y1;
            yy2 = yy2 + y2 * y2;
            yy3 = yy3 + y3 * y3;
            yy4 = yy4 + y4 * y4;
            yy5 = yy5 + y5 * y5;
            yy6 = yy6 + y6 * y6;
            yy7 = yy7 + y7 * y7;
        }
        i += W;
    }
    let mut sy = 0.0;
    sy = sy + (sy0 + sy4);
    sy = sy + (sy1 + sy5);
    sy = sy + (sy2 + sy6);
    sy = sy + (sy3 + sy7);
    let mut yy = 0.0;
    yy = yy + (yy0 + yy4);
    yy = yy + (yy1 + yy5);
    yy = yy + (yy2 + yy6);
    yy = yy + (yy3 + yy7);
    while i < n {
        unsafe {
            let sv = *s.get_unchecked(i);
            let yv = *y.get_unchecked(i);
            ss = ss + sv * sv;
            yys = yys + yv * yv;
            sy = sy + sv * yv;
            yy = yy + yv * yv;
        }
        i += 1;
    }
    (sy, yy, ss, yys)
}

#[inline(always)]
fn dot_slice(xs: &[f64], ys: &[f64]) -> f64 {
    dispatch_dot!(choice(), xs, ys)
}

fn dispatch_axpy_dot(a: f64, xs: &[f64], ys: &mut [f64], zs: &[f64]) -> f64 {
    match choice() {
        Choice::Forced(Path::Scalar) => axpy_dot_lanes(a, xs, ys, zs),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        Choice::Forced(Path::Avx2) => unsafe { axpy_dot_avx2(a, xs, ys, zs) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        Choice::Forced(Path::Avx2) => axpy_dot_lanes(a, xs, ys, zs),
        #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
        Choice::Forced(Path::Avx512) => unsafe { axpy_dot_avx512(a, xs, ys, zs) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
        Choice::Forced(Path::Avx512) => axpy_dot_lanes(a, xs, ys, zs),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Neon) => unsafe { axpy_dot_neon(a, xs, ys, zs) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Neon) => axpy_dot_lanes(a, xs, ys, zs),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Sve) => unsafe { axpy_dot_sve(a, xs, ys, zs) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Sve) => axpy_dot_lanes(a, xs, ys, zs),
        #[cfg(feature = "simd")]
        Choice::Auto => {
            if scalar_gate() {
                axpy_dot_lanes(a, xs, ys, zs)
            } else {
                axpy_dot_mv(a, xs, ys, zs)
            }
        }
        #[cfg(not(feature = "simd"))]
        Choice::Auto => axpy_dot_lanes(a, xs, ys, zs),
    }
}

fn dispatch_scale_dot(a: f64, ys: &mut [f64], zs: &[f64]) -> f64 {
    match choice() {
        Choice::Forced(Path::Scalar) => scale_dot_lanes(a, ys, zs),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        Choice::Forced(Path::Avx2) => unsafe { scale_dot_avx2(a, ys, zs) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        Choice::Forced(Path::Avx2) => scale_dot_lanes(a, ys, zs),
        #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
        Choice::Forced(Path::Avx512) => unsafe { scale_dot_avx512(a, ys, zs) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
        Choice::Forced(Path::Avx512) => scale_dot_lanes(a, ys, zs),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Neon) => unsafe { scale_dot_neon(a, ys, zs) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Neon) => scale_dot_lanes(a, ys, zs),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Sve) => unsafe { scale_dot_sve(a, ys, zs) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Sve) => scale_dot_lanes(a, ys, zs),
        #[cfg(feature = "simd")]
        Choice::Auto => {
            if scalar_gate() {
                scale_dot_lanes(a, ys, zs)
            } else {
                scale_dot_mv(a, ys, zs)
            }
        }
        #[cfg(not(feature = "simd"))]
        Choice::Auto => scale_dot_lanes(a, ys, zs),
    }
}

fn dispatch_copy_inf_dot(src: &[f64], dst: &mut [f64], v: &[f64]) -> (f64, f64) {
    match choice() {
        Choice::Forced(Path::Scalar) => copy_inf_dot_lanes(src, dst, v),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        Choice::Forced(Path::Avx2) => unsafe { copy_inf_dot_avx2(src, dst, v) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        Choice::Forced(Path::Avx2) => copy_inf_dot_lanes(src, dst, v),
        #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
        Choice::Forced(Path::Avx512) => unsafe { copy_inf_dot_avx512(src, dst, v) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
        Choice::Forced(Path::Avx512) => copy_inf_dot_lanes(src, dst, v),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Neon) => unsafe { copy_inf_dot_neon(src, dst, v) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Neon) => copy_inf_dot_lanes(src, dst, v),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Sve) => unsafe { copy_inf_dot_sve(src, dst, v) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Sve) => copy_inf_dot_lanes(src, dst, v),
        #[cfg(feature = "simd")]
        Choice::Auto => {
            if scalar_gate() {
                copy_inf_dot_lanes(src, dst, v)
            } else {
                copy_inf_dot_mv(src, dst, v)
            }
        }
        #[cfg(not(feature = "simd"))]
        Choice::Auto => copy_inf_dot_lanes(src, dst, v),
    }
}

fn dispatch_copy_euclid_dot(src: &[f64], dst: &mut [f64], v: &[f64]) -> (f64, f64, bool) {
    match choice() {
        Choice::Forced(Path::Scalar) => copy_euclid_dot_lanes(src, dst, v),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        Choice::Forced(Path::Avx2) => unsafe { copy_euclid_dot_avx2(src, dst, v) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        Choice::Forced(Path::Avx2) => copy_euclid_dot_lanes(src, dst, v),
        #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
        Choice::Forced(Path::Avx512) => unsafe { copy_euclid_dot_avx512(src, dst, v) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
        Choice::Forced(Path::Avx512) => copy_euclid_dot_lanes(src, dst, v),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Neon) => unsafe { copy_euclid_dot_neon(src, dst, v) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Neon) => copy_euclid_dot_lanes(src, dst, v),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Sve) => unsafe { copy_euclid_dot_sve(src, dst, v) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Sve) => copy_euclid_dot_lanes(src, dst, v),
        #[cfg(feature = "simd")]
        Choice::Auto => {
            if scalar_gate() {
                copy_euclid_dot_lanes(src, dst, v)
            } else {
                copy_euclid_dot_mv(src, dst, v)
            }
        }
        #[cfg(not(feature = "simd"))]
        Choice::Auto => copy_euclid_dot_lanes(src, dst, v),
    }
}

fn dispatch_curvature(s: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    match choice() {
        Choice::Forced(Path::Scalar) => curvature_lanes(s, y),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        Choice::Forced(Path::Avx2) => unsafe { curvature_avx2(s, y) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        Choice::Forced(Path::Avx2) => curvature_lanes(s, y),
        #[cfg(all(feature = "simd", target_arch = "x86_64", rgmin_avx512))]
        Choice::Forced(Path::Avx512) => unsafe { curvature_avx512(s, y) },
        #[cfg(not(all(feature = "simd", target_arch = "x86_64", rgmin_avx512)))]
        Choice::Forced(Path::Avx512) => curvature_lanes(s, y),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Neon) => unsafe { curvature_neon(s, y) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Neon) => curvature_lanes(s, y),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Choice::Forced(Path::Sve) => unsafe { curvature_sve(s, y) },
        #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
        Choice::Forced(Path::Sve) => curvature_lanes(s, y),
        #[cfg(feature = "simd")]
        Choice::Auto => {
            if scalar_gate() {
                curvature_lanes(s, y)
            } else {
                curvature_mv(s, y)
            }
        }
        #[cfg(not(feature = "simd"))]
        Choice::Auto => curvature_lanes(s, y),
    }
}

macro_rules! axpy_dot_body {
    ($a:expr, $xs:expr, $ys:expr, $zs:expr) => {
        axpy_dot_lanes($a, $xs, $ys, $zs)
    };
}
macro_rules! scale_dot_body {
    ($a:expr, $ys:expr, $zs:expr) => {
        scale_dot_lanes($a, $ys, $zs)
    };
}
macro_rules! copy_inf_dot_body {
    ($src:expr, $dst:expr, $v:expr) => {
        copy_inf_dot_lanes($src, $dst, $v)
    };
}
macro_rules! copy_euclid_dot_body {
    ($src:expr, $dst:expr, $v:expr) => {
        copy_euclid_dot_lanes($src, $dst, $v)
    };
}
macro_rules! curvature_body {
    ($s:expr, $y:expr) => {
        curvature_lanes($s, $y)
    };
}

define_kernels!(
    fn axpy_dot_mv(a: f64, xs: &[f64], ys: &mut [f64], zs: &[f64]) -> f64,
    avx2 axpy_dot_avx2,
    avx512 axpy_dot_avx512,
    neon axpy_dot_neon,
    sve axpy_dot_sve,
    axpy_dot_body
);
define_kernels!(
    fn scale_dot_mv(a: f64, ys: &mut [f64], zs: &[f64]) -> f64,
    avx2 scale_dot_avx2,
    avx512 scale_dot_avx512,
    neon scale_dot_neon,
    sve scale_dot_sve,
    scale_dot_body
);
define_kernels!(
    fn copy_inf_dot_mv(src: &[f64], dst: &mut [f64], v: &[f64]) -> (f64, f64),
    avx2 copy_inf_dot_avx2,
    avx512 copy_inf_dot_avx512,
    neon copy_inf_dot_neon,
    sve copy_inf_dot_sve,
    copy_inf_dot_body
);
define_kernels!(
    fn copy_euclid_dot_mv(src: &[f64], dst: &mut [f64], v: &[f64]) -> (f64, f64, bool),
    avx2 copy_euclid_dot_avx2,
    avx512 copy_euclid_dot_avx512,
    neon copy_euclid_dot_neon,
    sve copy_euclid_dot_sve,
    copy_euclid_dot_body
);
define_kernels!(
    fn curvature_mv(s: &[f64], y: &[f64]) -> (f64, f64, f64, f64),
    avx2 curvature_avx2,
    avx512 curvature_avx512,
    neon curvature_neon,
    sve curvature_sve,
    curvature_body
);

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
        _ => {
            for i in 0..x.len() {
                y[i] = a.mul_add(x[i], y[i]);
            }
        }
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
        // Eight running maxima. `max` of absolute values is exact, and for
        // finite inputs every association returns that same value. A
        // non-finite component is infinity, whichever lane sees it.
        let n = xs.len();
        let mut i = 0;
        let (mut m0, mut m1, mut m2, mut m3, mut m4, mut m5, mut m6, mut m7) = (
            0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64,
        );
        while i + W <= n {
            let s0 = xs[i];
            let s1 = xs[i + 1];
            let s2 = xs[i + 2];
            let s3 = xs[i + 3];
            let s4 = xs[i + 4];
            let s5 = xs[i + 5];
            let s6 = xs[i + 6];
            let s7 = xs[i + 7];
            if !(s0.is_finite()
                && s1.is_finite()
                && s2.is_finite()
                && s3.is_finite()
                && s4.is_finite()
                && s5.is_finite()
                && s6.is_finite()
                && s7.is_finite())
            {
                return f64::INFINITY;
            }
            m0 = m0.max(s0.abs());
            m1 = m1.max(s1.abs());
            m2 = m2.max(s2.abs());
            m3 = m3.max(s3.abs());
            m4 = m4.max(s4.abs());
            m5 = m5.max(s5.abs());
            m6 = m6.max(s6.abs());
            m7 = m7.max(s7.abs());
            i += W;
        }
        let mut m = m0.max(m1).max(m2).max(m3).max(m4).max(m5).max(m6).max(m7);
        while i < n {
            let s = xs[i];
            if !s.is_finite() {
                return f64::INFINITY;
            }
            m = m.max(s.abs());
            i += 1;
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

    fn seq_inf(xs: &[f64]) -> f64 {
        let mut m = 0.0_f64;
        for v in xs {
            if !v.is_finite() {
                return f64::INFINITY;
            }
            m = m.max(v.abs());
        }
        m
    }

    #[test]
    fn fused_passes_match_the_separate_kernels() {
        let lengths = [0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 128, 1000];
        for n in lengths {
            let x =
                Array1::from_iter((0..n).map(|i| ((i * 3) as f64).sin() * (1.0 + (i % 5) as f64)));
            let y0 = Array1::from_iter((0..n).map(|i| ((i * 5) as f64).cos() - 0.25 * i as f64));
            let z = Array1::from_iter((0..n).map(|i| ((i * 7) as f64).sin() * 0.5 - 1.0));
            let mut y = y0.clone();
            let mut y_ref = y0.clone();
            let a = -0.37;
            axpy(a, x.view(), &mut y_ref);
            let expect = dot(z.view(), y_ref.view());
            let got = axpy_dot(a, x.view(), &mut y, z.view());
            assert_eq!(y, y_ref, "axpy n={n}");
            assert_eq!(got.to_bits(), expect.to_bits(), "axpy_dot n={n}");

            let scale_a = 0.125;
            let mut ys = y.clone();
            let mut yr = y.clone();
            scale(scale_a, &mut yr);
            let expect = dot(z.view(), yr.view());
            let got = scale_dot(scale_a, &mut ys, z.view());
            assert_eq!(ys, yr, "scale n={n}");
            assert_eq!(got.to_bits(), expect.to_bits(), "scale_dot n={n}");

            let mut dst = Array1::zeros(n);
            let (inf, dinf) = copy_inf_dot(x.view(), &mut dst, z.view());
            assert_eq!(dst, x, "copy inf n={n}");
            assert_eq!(inf.to_bits(), nrminf(x.view()).to_bits(), "inf n={n}");
            assert_eq!(inf.to_bits(), seq_inf(x.as_slice().unwrap()).to_bits());
            assert_eq!(
                dinf.to_bits(),
                dot(z.view(), x.view()).to_bits(),
                "inf dot n={n}"
            );

            let (gg, sg, finite) = copy_euclid_dot(x.view(), &mut dst, z.view());
            assert!(finite || n == 0 || x.iter().any(|v| !v.is_finite()));
            assert_eq!(dst, x);
            assert_eq!(gg.to_bits(), dot(x.view(), x.view()).to_bits(), "gg n={n}");
            assert_eq!(sg.to_bits(), dot(z.view(), x.view()).to_bits(), "sg n={n}");

            let (sy, yy, ss, yys) = curvature_products(x.view(), y.view());
            assert_eq!(sy.to_bits(), x.dot(&y).to_bits(), "sy n={n}");
            assert_eq!(yy.to_bits(), y.dot(&y).to_bits(), "yy n={n}");
            let ss_ref: f64 = x.iter().map(|v| v * v).sum();
            let yy_ref: f64 = y.iter().map(|v| v * v).sum();
            assert_eq!(ss.to_bits(), ss_ref.to_bits(), "ss n={n}");
            assert_eq!(yys.to_bits(), yy_ref.to_bits(), "yys n={n}");
        }
        let mut broken = Array1::from_vec(vec![1.0, f64::NAN, -3.0, 4.0]);
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let mut dst = Array1::zeros(4);
        let (inf, _) = copy_inf_dot(broken.view(), &mut dst, v.view());
        assert!(inf.is_infinite());
        assert_eq!(nrminf(broken.view()).to_bits(), f64::INFINITY.to_bits());
        broken[1] = f64::NEG_INFINITY;
        assert!(nrminf(broken.view()).is_infinite());
        let zeros = Array1::from_vec(vec![-0.0, 0.0, -0.0, 0.0]);
        assert_eq!(
            nrminf(zeros.view()).to_bits(),
            seq_inf(zeros.as_slice().unwrap()).to_bits()
        );
    }
}
