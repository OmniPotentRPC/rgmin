//! Bit identity of the W = 8 vector kernels against ndarray.
//!
//! The golden file is the SHA-256 of the portable `mul_add` path.
//! Every clone this CPU can run has to reproduce it, including `par`
//! at 1, 2, 3, and 8 threads on a forced path. Auto with `par` keeps
//! axpy and scale on that hash. Long auto reductions are contiguous
//! chunks of 4096, identical at every thread count. A clone this CPU
//! cannot run is reported as not exercised. `dot` and `axpy` are fused,
//! so they are not the ndarray product.

use ndarray::Array1;
use rgmin::vecops::{
    active_simd_path, axpy, clear_simd_path, div_assign_floor, dot, force_simd_path, mul_assign,
    nrm2, scale, simd_clone_usable, sum,
};
use sha2::{Digest, Sha256};

const AXPY_ALPHA: f64 = -1.5;
const SCALE_ALPHA: f64 = 0.5;

fn lengths() -> Vec<usize> {
    let mut out: Vec<usize> = (0..=67).collect();
    out.push(1_000);
    out.push(1_000_000);
    out
}

fn mix(i: usize, salt: u64) -> f64 {
    let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let mag = ((z >> 12) as f64) / ((1u64 << 52) as f64);
    if z & 1 == 0 { mag } else { -mag }
}

/// Recipe version 1. Positions below `n` override the mixed value.
fn sample(n: usize, i: usize, salt: u64) -> f64 {
    if i == 0 {
        return -0.0;
    }
    if i == 1 && n > 1 {
        return f64::from_bits(0x8000_0000_0000_0001);
    }
    if i == 2 && n > 2 {
        return f64::from_bits(0x0000_0000_0000_0001);
    }
    if i == 3 && n > 3 {
        return 1.0 + f64::EPSILON;
    }
    if n > 4 && i + 1 == n {
        return -0.0;
    }
    if i == 8 && n > 8 {
        return 1.0e16;
    }
    if i == 16 && n > 16 {
        return -1.0e16;
    }
    if i == 24 && n > 24 {
        return 1.0;
    }
    if i == 4096 && n > 4096 {
        return 1.0 + f64::EPSILON;
    }
    if i == 65536 && n > 65536 {
        return f64::from_bits(0x8000_0000_0000_0001);
    }
    mix(i, salt)
}

fn pair(n: usize) -> (Array1<f64>, Array1<f64>) {
    let x = Array1::from_iter((0..n).map(|i| sample(n, i, 0xA11C_E5D0_u64)));
    let y = Array1::from_iter((0..n).map(|i| sample(n, i, 0xB0B0_F00D_u64)));
    (x, y)
}

fn feed_f64(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_le_bytes());
}

fn feed_vec(hasher: &mut Sha256, values: &[f64]) {
    for value in values {
        feed_f64(hasher, *value);
    }
}

struct Digests {
    dot: String,
    sum: String,
    nrm2: String,
    axpy: String,
    scale: String,
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

fn kernel_digests() -> Digests {
    let mut dot_h = Sha256::new();
    let mut sum_h = Sha256::new();
    let mut nrm_h = Sha256::new();
    let mut axpy_h = Sha256::new();
    let mut scale_h = Sha256::new();
    let mut cases: Vec<(usize, bool)> = lengths().into_iter().map(|n| (n, false)).collect();
    for n in [2usize, 8, 9, 17, 67] {
        cases.push((n, true));
    }
    for (n, nan) in cases {
        let (mut x, mut y) = pair(n);
        if nan && n > 1 {
            x[1] = f64::from_bits(0x7ff8_0000_0000_0001);
            y[1] = f64::from_bits(0x7ff8_0000_0000_0001);
        }
        let mut ax = y.clone();
        let mut sc = y.clone();
        let n_bytes = (n as u64).to_le_bytes();
        for hasher in [
            &mut dot_h,
            &mut sum_h,
            &mut nrm_h,
            &mut axpy_h,
            &mut scale_h,
        ] {
            hasher.update(n_bytes);
            hasher.update([u8::from(nan)]);
        }
        feed_f64(&mut dot_h, dot(x.view(), y.view()));
        feed_f64(&mut sum_h, sum(x.view()));
        feed_f64(&mut nrm_h, nrm2(x.view()));
        axpy(AXPY_ALPHA, x.view(), &mut ax);
        scale(SCALE_ALPHA, &mut sc);
        feed_vec(&mut axpy_h, ax.as_slice().unwrap());
        feed_vec(&mut scale_h, sc.as_slice().unwrap());
    }
    Digests {
        dot: hex(&dot_h.finalize()),
        sum: hex(&sum_h.finalize()),
        nrm2: hex(&nrm_h.finalize()),
        axpy: hex(&axpy_h.finalize()),
        scale: hex(&scale_h.finalize()),
    }
}

fn bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// Eight partials, each a `mul_add`, then the same pair tree as `dot`.
fn portable_dot(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    let mut i = 0;
    let mut sum = 0.0;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    while i + 8 <= n {
        p0 = xs[i].mul_add(ys[i], p0);
        p1 = xs[i + 1].mul_add(ys[i + 1], p1);
        p2 = xs[i + 2].mul_add(ys[i + 2], p2);
        p3 = xs[i + 3].mul_add(ys[i + 3], p3);
        p4 = xs[i + 4].mul_add(ys[i + 4], p4);
        p5 = xs[i + 5].mul_add(ys[i + 5], p5);
        p6 = xs[i + 6].mul_add(ys[i + 6], p6);
        p7 = xs[i + 7].mul_add(ys[i + 7], p7);
        i += 8;
    }
    sum = sum + (p0 + p4);
    sum = sum + (p1 + p5);
    sum = sum + (p2 + p6);
    sum = sum + (p3 + p7);
    while i < n {
        sum = xs[i].mul_add(ys[i], sum);
        i += 1;
    }
    sum
}

fn portable_axpy(a: f64, xs: &[f64], ys: &mut [f64]) {
    for i in 0..xs.len() {
        ys[i] = a.mul_add(xs[i], ys[i]);
    }
}

fn assert_matches_reference() {
    let mut cases: Vec<(usize, bool)> = lengths().into_iter().map(|n| (n, false)).collect();
    for n in [2usize, 8, 9, 17, 67] {
        cases.push((n, true));
    }
    for (n, nan) in cases {
        let (mut x, mut y) = pair(n);
        if nan && n > 1 {
            x[1] = f64::from_bits(0x7ff8_0000_0000_0001);
            y[1] = f64::from_bits(0x7ff8_0000_0000_0001);
        }
        let got = dot(x.view(), y.view());
        let expect = portable_dot(x.as_slice().unwrap(), y.as_slice().unwrap());
        assert!(
            bits(got, expect),
            "dot n={n} got {got:?} ({:#x}) portable {expect:?} ({:#x})",
            got.to_bits(),
            expect.to_bits()
        );
        let got = sum(x.view());
        let expect = x.sum();
        assert!(
            bits(got, expect),
            "sum n={n} got {got:?} ({:#x}) ndarray {expect:?} ({:#x})",
            got.to_bits(),
            expect.to_bits()
        );
        let got = nrm2(x.view());
        let expect = portable_dot(x.as_slice().unwrap(), x.as_slice().unwrap()).sqrt();
        assert!(
            bits(got, expect),
            "nrm2 n={n} got {got:?} ({:#x}) portable {expect:?} ({:#x})",
            got.to_bits(),
            expect.to_bits()
        );
        let mut z = y.clone();
        axpy(AXPY_ALPHA, x.view(), &mut z);
        let mut reference = y.clone();
        portable_axpy(
            AXPY_ALPHA,
            x.as_slice().unwrap(),
            reference.as_slice_mut().unwrap(),
        );
        for i in 0..n {
            assert_eq!(
                z[i].to_bits(),
                reference[i].to_bits(),
                "axpy n={n} i={i} got {:#x} ref {:#x}",
                z[i].to_bits(),
                reference[i].to_bits()
            );
        }
        let mut z = y.clone();
        scale(SCALE_ALPHA, &mut z);
        let mut reference = y.clone();
        reference.mapv_inplace(|v| v * SCALE_ALPHA);
        for i in 0..n {
            assert_eq!(
                z[i].to_bits(),
                reference[i].to_bits(),
                "scale n={n} i={i} got {:#x} ref {:#x}",
                z[i].to_bits(),
                reference[i].to_bits()
            );
        }
        let mut z = y.clone();
        mul_assign(x.view(), &mut z);
        for i in 0..n {
            assert_eq!(z[i].to_bits(), (y[i] * x[i]).to_bits(), "mul n={n} i={i}");
        }
        let mut z = y.clone();
        div_assign_floor(x.view(), &mut z, 0.25);
        for i in 0..n {
            let expect = y[i] / x[i].max(0.25);
            assert_eq!(z[i].to_bits(), expect.to_bits(), "div n={n} i={i}");
        }
    }
}

fn golden(key: &str) -> String {
    let text = include_str!("golden/isa-hashes.toml");
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim().trim_start_matches('=').trim();
            return rest.trim_matches('"').to_string();
        }
    }
    panic!("missing golden key {key}");
}

struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        clear_simd_path();
    }
}

#[test]
fn cpu_flags_and_clone_availability() {
    if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(line) = text.lines().find(|line| line.starts_with("flags")) {
            println!("{line}");
        }
    }
    for name in ["scalar", "avx2", "avx512", "neon", "sve"] {
        if name == "scalar" || simd_clone_usable(name) {
            println!("isa clone {name}: available");
        } else {
            println!("isa clone {name}: not exercised");
        }
    }
    assert!(simd_clone_usable("scalar"));
    assert!(!force_simd_path("not-a-real-isa"));
    clear_simd_path();
}

#[test]
fn scalar_matches_the_portable_path_and_golden() {
    let _guard = Guard;
    assert!(force_simd_path("scalar"));
    assert_matches_reference();
    let got = kernel_digests();
    let expect_dot = golden("dot");
    if std::env::var_os("RGMIN_PRINT_GOLDEN").is_some() {
        println!("dot = \"{}\"", got.dot);
        println!("sum = \"{}\"", got.sum);
        println!("nrm2 = \"{}\"", got.nrm2);
        println!("axpy = \"{}\"", got.axpy);
        println!("scale = \"{}\"", got.scale);
    }
    assert_eq!(got.dot, expect_dot, "dot");
    assert_eq!(got.sum, golden("sum"), "sum");
    assert_eq!(got.nrm2, golden("nrm2"), "nrm2");
    assert_eq!(got.axpy, golden("axpy"), "axpy");
    assert_eq!(got.scale, golden("scale"), "scale");
}

#[test]
fn every_runnable_clone_matches_the_golden_hash() {
    let scalar = {
        let _guard = Guard;
        assert!(force_simd_path("scalar"));
        kernel_digests()
    };
    for name in ["auto", "avx2", "avx512", "neon", "sve"] {
        let _guard = Guard;
        let ok = if name == "auto" {
            clear_simd_path();
            println!("isa path auto ({}): exercised", active_simd_path());
            true
        } else if force_simd_path(name) {
            println!("isa path {name}: exercised");
            true
        } else {
            println!("isa path {name}: not exercised");
            false
        };
        if !ok {
            continue;
        }
        if name == "auto" && cfg!(feature = "par") {
            let got = kernel_digests();
            assert_eq!(got.axpy, scalar.axpy, "auto axpy");
            assert_eq!(got.scale, scalar.scale, "auto scale");
            continue;
        }
        assert_matches_reference();
        let got = kernel_digests();
        assert_eq!(got.dot, scalar.dot, "{name} dot");
        assert_eq!(got.sum, scalar.sum, "{name} sum");
        assert_eq!(got.nrm2, scalar.nrm2, "{name} nrm2");
        assert_eq!(got.axpy, scalar.axpy, "{name} axpy");
        assert_eq!(got.scale, scalar.scale, "{name} scale");
    }
}

#[cfg(feature = "par")]
#[test]
fn par_matches_scalar_at_one_two_three_and_eight_threads() {
    let scalar = {
        let _guard = Guard;
        assert!(force_simd_path("scalar"));
        kernel_digests()
    };
    for threads in [1usize, 2, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            clear_simd_path();
            println!("isa par threads={threads} path=auto: exercised");
            let got = kernel_digests();
            assert_eq!(got.axpy, scalar.axpy, "threads {threads}");
            assert_eq!(got.scale, scalar.scale, "threads {threads}");
            for name in ["scalar", "avx2", "avx512", "neon", "sve"] {
                if !simd_clone_usable(name) && name != "scalar" {
                    println!("isa par threads={threads} path={name}: not exercised");
                    continue;
                }
                assert!(force_simd_path(name));
                println!("isa par threads={threads} path={name}: exercised");
                assert_matches_reference();
                let got = kernel_digests();
                assert_eq!(got.dot, scalar.dot, "{name} threads {threads}");
                assert_eq!(got.axpy, scalar.axpy, "{name} threads {threads}");
                clear_simd_path();
            }
        });
    }
}

#[cfg(feature = "par")]
fn chunk_dot(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    use ndarray::ArrayView1;
    let xs = x.as_slice().unwrap();
    let ys = y.as_slice().unwrap();
    let mut acc = 0.0;
    let mut i = 0;
    while i < xs.len() {
        let j = (i + 4096).min(xs.len());
        acc = acc + dot(ArrayView1::from(&xs[i..j]), ArrayView1::from(&ys[i..j]));
        i = j;
    }
    acc
}

#[cfg(feature = "par")]
fn chunk_sum(x: &Array1<f64>) -> f64 {
    use ndarray::ArrayView1;
    let xs = x.as_slice().unwrap();
    let mut acc = 0.0;
    let mut i = 0;
    while i < xs.len() {
        let j = (i + 4096).min(xs.len());
        acc = acc + sum(ArrayView1::from(&xs[i..j]));
        i = j;
    }
    acc
}

/// Long auto reductions are contiguous chunks of 4096, each folded by
/// the active kernel, then added from the first chunk upward.
#[cfg(feature = "par")]
#[test]
fn parallel_reductions_match_contiguous_chunks_at_every_thread_count() {
    let _guard = Guard;
    clear_simd_path();
    let n = 70_000;
    let (x, y) = pair(n);
    let expect_dot = chunk_dot(&x, &y);
    let expect_sum = chunk_sum(&x);
    let expect_nrm = chunk_dot(&x, &x).sqrt();
    for threads in [1usize, 2, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            clear_simd_path();
            assert_eq!(
                dot(x.view(), y.view()).to_bits(),
                expect_dot.to_bits(),
                "dot threads {threads}"
            );
            assert_eq!(
                sum(x.view()).to_bits(),
                expect_sum.to_bits(),
                "sum threads {threads}"
            );
            assert_eq!(
                nrm2(x.view()).to_bits(),
                expect_nrm.to_bits(),
                "nrm2 threads {threads}"
            );
        });
    }
}

fn fma_probe() -> (Array1<f64>, Array1<f64>, f64, f64) {
    // Index 0 fills partial p0. Index 8 is the tail, fused into that sum.
    // (1+eps)*(1-eps) rounds to 1, so a split product cancels the -1.
    // One mul_add keeps the -eps^2 term.
    let eps = f64::EPSILON;
    let mut x = Array1::zeros(9);
    let mut y = Array1::zeros(9);
    x[0] = -1.0;
    y[0] = 1.0;
    x[8] = 1.0 + eps;
    y[8] = 1.0 - eps;
    let split = (x[0] * y[0]) + (x[8] * y[8]);
    let fused = x[8].mul_add(y[8], x[0].mul_add(y[0], 0.0));
    (x, y, split, fused)
}

#[test]
fn a_strided_view_uses_the_same_mul_add_tree() {
    let _guard = Guard;
    clear_simd_path();
    let (x, y) = pair(32);
    let x2 = x.view().into_shape_with_order((16, 2)).unwrap();
    let y2 = y.view().into_shape_with_order((16, 2)).unwrap();
    let colx = x2.column(0);
    let coly = y2.column(0);
    assert!(colx.as_slice().is_none());
    let xs: Vec<f64> = colx.iter().copied().collect();
    let ys: Vec<f64> = coly.iter().copied().collect();
    assert_eq!(dot(colx, coly).to_bits(), portable_dot(&xs, &ys).to_bits());
}

#[test]
fn dot_fuses_the_product_that_a_split_multiply_would_round_away() {
    let _guard = Guard;
    let (x, y, split, fused) = fma_probe();
    assert_ne!(split.to_bits(), fused.to_bits());
    for name in ["scalar", "auto", "avx2", "avx512", "neon", "sve"] {
        let ran = if name == "auto" {
            clear_simd_path();
            true
        } else {
            force_simd_path(name)
        };
        if !ran {
            continue;
        }
        let got = dot(x.view(), y.view());
        assert_eq!(got.to_bits(), fused.to_bits(), "{name}");
        assert_ne!(got.to_bits(), split.to_bits(), "{name}");
    }
}

#[test]
fn an_unusable_override_falls_back_to_scalar_bits() {
    let _guard = Guard;
    let scalar = {
        assert!(force_simd_path("scalar"));
        let (x, y) = pair(32);
        dot(x.view(), y.view())
    };
    // `sve` is not runnable on x86_64. On aarch64 this still has to match.
    let _ran = force_simd_path("sve");
    let (x, y) = pair(32);
    let got = dot(x.view(), y.view());
    let expect = portable_dot(x.as_slice().unwrap(), y.as_slice().unwrap());
    assert!(bits(got, expect));
    if !simd_clone_usable("sve") {
        assert!(bits(got, scalar));
        assert_eq!(active_simd_path(), "scalar");
    }
}
