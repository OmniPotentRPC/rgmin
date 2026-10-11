//! Rank invariance of the `par` reductions.
//!
//! `dot`, `sum` and `nrm2` hash to one SHA-256 at 1, 2, 3 and 8 threads.
//! The serial fold stays bit-identical below `PAR_MIN_LEN`. Above it the
//! parallel result is the fixed chunk tree, and it stays within the
//! parenthesization bound of ndarray's fold.
//!
//! Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
//! ISBN 978-0-89871-802-7, Theorem 4.2, doi:10.1137/1.9780898718027.
//! Any parenthesization of m finite terms errs by at most
//! γ_{m-1} Σ |p_i|, γ_k = k u / (1 - k u), u = 2^{-53}.
//! Both the chunk tree and ndarray's dot are parenthesizations of the
//! rounded products, so
//! |dot_par - dot_serial| ≤ 2 γ_{n-1} Σ_i |fl(x_i y_i)|
//! when (n-1) u < 1. The same bound holds for `sum` with |x_i|.
//! For a positive `nrm2`, |sqrt(a) - sqrt(b)| = |a - b| / (sqrt(a) + sqrt(b)).

#![cfg(feature = "par")]

use std::collections::BTreeMap;

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use rgmin::vecops::{dot, nrm2, sum};
use sha2::{Digest, Sha256};

const CHUNK: usize = 4096;
const MIN_LEN: usize = 65_536;
const THREADS: [usize; 4] = [1, 2, 3, 8];

fn sha256_f64(values: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn tree(mut level: Vec<f64>) -> f64 {
    if level.is_empty() {
        return 0.0;
    }
    for _ in 0..64 {
        if level.len() <= 1 {
            break;
        }
        let n = level.len();
        let pairs = n / 2;
        let mut next = Vec::with_capacity(pairs + (n % 2));
        for i in 0..pairs {
            next.push(level[2 * i] + level[2 * i + 1]);
        }
        if n % 2 == 1 {
            next.push(level[n - 1]);
        }
        level = next;
    }
    level[0]
}

fn chunk_partials(n: usize, body: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let nchunks = n.div_ceil(CHUNK);
    let mut partials = Vec::with_capacity(nchunks);
    for chunk in 0..nchunks {
        let start = chunk * CHUNK;
        let end = (start + CHUNK).min(n);
        partials.push(body(start, end));
    }
    partials
}

fn oracle_dot(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < MIN_LEN {
        let x = Array1::from(xs.to_vec());
        let y = Array1::from(ys.to_vec());
        return x.dot(&y);
    }
    tree(chunk_partials(xs.len(), |start, end| {
        let mut acc = 0.0;
        for i in start..end {
            let prod = xs[i] * ys[i];
            acc += prod;
        }
        acc
    }))
}

fn oracle_sum(xs: &[f64]) -> f64 {
    if xs.len() < MIN_LEN {
        return xs.iter().sum();
    }
    tree(chunk_partials(xs.len(), |start, end| {
        let mut acc = 0.0;
        for value in &xs[start..end] {
            acc += *value;
        }
        acc
    }))
}

/// Values from 2^{-60} to 2^{60}, with signed zeros, a subnormal and a tie.
fn span(n: usize, phase: i32) -> Array1<f64> {
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let exp = (i as i32).wrapping_mul(17 + phase) % 121 - 60;
        let magnitude = (1.0 + (i % 5) as f64) * 2f64.powi(exp);
        let sign = if (i + phase.unsigned_abs() as usize) % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        values.push(sign * magnitude);
    }
    if n > 8 {
        values[0] = -0.0;
        values[1] = 0.0;
        values[2] = f64::from_bits(1);
        values[3] = -f64::from_bits(1);
        values[4] = 1.0;
        values[5] = 1.0;
    }
    Array1::from(values)
}

fn with_nan(mut values: Array1<f64>) -> Array1<f64> {
    let n = values.len();
    values[n / 3] = f64::NAN;
    values
}

fn on_pool<T: Send>(threads: usize, body: impl FnOnce() -> T + Send) -> T {
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    pool.install(body)
}

fn gamma(terms: usize) -> f64 {
    let k = terms.saturating_sub(1) as f64;
    let unit = f64::EPSILON / 2.0;
    let ku = k * unit;
    assert!(ku < 1.0, "the γ bound needs k u < 1");
    ku / (1.0 - ku)
}

fn sum_abs_products(xs: &[f64], ys: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..xs.len() {
        acc += (xs[i] * ys[i]).abs();
    }
    acc
}

/// Lift a computed sum of nonnegative terms above the exact sum.
fn parenthesization_gap(n: usize, sum_abs: f64) -> f64 {
    let g = gamma(n);
    // |s_hat - s| ≤ γ s for a sum of nonnegative terms, so s ≤ s_hat / (1-γ).
    // Each fold is within γ s of the exact sum, and the two folds differ by at most 2 γ s.
    2.0 * g / (1.0 - g) * sum_abs
}

fn assert_gap(par: f64, serial: f64, bound: f64) {
    let gap = (par - serial).abs();
    assert!(
        gap <= bound,
        "gap {gap:e} exceeds the fixed-tree bound {bound:e} (par {par:e}, serial {serial:e})"
    );
}

#[test]
fn short_vectors_match_the_serial_fold_bit_for_bit() {
    let x = Array1::from(vec![-0.0, 0.0, f64::from_bits(1), 1.0, -2.0, 2f64.powi(53)]);
    let y = Array1::from(vec![1.0, -0.0, 3.0, -1.0, 0.5, 1.0]);
    assert_eq!(dot(x.view(), y.view()).to_bits(), x.dot(&y).to_bits());
    assert_eq!(sum(x.view()).to_bits(), x.sum().to_bits());
    assert_eq!(nrm2(x.view()).to_bits(), x.dot(&x).sqrt().to_bits());
}

#[test]
fn the_chunk_tree_is_not_a_left_fold() {
    // Chunks [1, 2^53, -2^53, -1, 0, ...]. Adjacent pairing cancels
    // 2^53 with -2^53 after 1 has already been absorbed, so the -1 is
    // lost and the tree returns 0. A left fold returns -1.
    let n = MIN_LEN;
    let mut x = vec![0.0; n];
    x[0] = 1.0;
    x[CHUNK] = 2f64.powi(53);
    x[2 * CHUNK] = -2f64.powi(53);
    x[3 * CHUNK] = -1.0;
    let values = Array1::from(x);
    let parallel = on_pool(3, || sum(values.view()));
    let serial = values.sum();
    assert_eq!(parallel.to_bits(), 0.0f64.to_bits());
    assert_eq!(serial.to_bits(), (-1.0f64).to_bits());
    assert_eq!(
        parallel.to_bits(),
        oracle_sum(values.as_slice().unwrap()).to_bits()
    );
    let bound = parenthesization_gap(n, values.iter().map(|v| v.abs()).sum());
    assert_gap(parallel, serial, bound);
    assert_ne!(parallel.to_bits(), serial.to_bits());
}

#[test]
fn par_stays_inside_the_fixed_tree_bound() {
    for n in [MIN_LEN, 1_000_000] {
        let x = span(n, 1);
        let y = span(n, 3);
        let xs = x.as_slice().unwrap();
        let ys = y.as_slice().unwrap();
        let parallel_dot = on_pool(2, || dot(x.view(), y.view()));
        let parallel_sum = on_pool(2, || sum(x.view()));
        let serial_dot = x.dot(&y);
        let serial_sum = x.sum();
        assert_eq!(parallel_dot.to_bits(), oracle_dot(xs, ys).to_bits());
        assert_eq!(parallel_sum.to_bits(), oracle_sum(xs).to_bits());
        assert_gap(
            parallel_dot,
            serial_dot,
            parenthesization_gap(n, sum_abs_products(xs, ys)),
        );
        assert_gap(
            parallel_sum,
            serial_sum,
            parenthesization_gap(n, xs.iter().map(|v| v.abs()).sum()),
        );
        let parallel_nrm = on_pool(2, || nrm2(x.view()));
        let serial_nrm = x.dot(&x).sqrt();
        // |sqrt(a) - sqrt(b)| = |a - b| / (sqrt(a) + sqrt(b)) for a, b > 0.
        // |a - b| is at most the parenthesization gap on the squared entries.
        let dot_bound = parenthesization_gap(n, sum_abs_products(xs, xs));
        let denom = parallel_nrm + serial_nrm;
        let nrm_bound = if denom > 0.0 { dot_bound / denom } else { 0.0 };
        assert_gap(parallel_nrm, serial_nrm, nrm_bound);
    }
}

#[test]
fn hashes_match_at_one_two_three_and_eight_threads() {
    let mut rows = Vec::new();
    for n in [MIN_LEN, 1_000_000] {
        for (label, x, y) in [
            ("span", span(n, 1), span(n, 3)),
            ("nan", with_nan(span(n, 1)), with_nan(span(n, 3))),
        ] {
            let mut hashes = BTreeMap::<usize, [String; 3]>::new();
            for threads in THREADS {
                let (d, s, r) = on_pool(threads, || {
                    (dot(x.view(), y.view()), sum(x.view()), nrm2(x.view()))
                });
                hashes.insert(
                    threads,
                    [sha256_f64(&[d]), sha256_f64(&[s]), sha256_f64(&[r])],
                );
            }
            let first = hashes[&1].clone();
            for threads in THREADS {
                assert_eq!(
                    hashes[&threads], first,
                    "{label} n={n} threads={threads} hash diverged"
                );
            }
            rows.push((n, label, first));
        }
    }
    for (n, label, [d, s, r]) in rows {
        println!("hash n={n} input={label} dot={d} sum={s} nrm2={r}");
    }
}

#[test]
#[ignore = "release timing table"]
fn strong_scaling_of_dot() {
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let thread_counts = [1usize, 2, 4, max_threads];
    let pools: Vec<_> = thread_counts
        .iter()
        .map(|&threads| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
        })
        .collect();
    println!("threads,n,variant,median_s,min_s,max_s");
    for n in [1_000_000usize, 10_000_000] {
        let x = span(n, 1);
        let y = span(n, 9);
        let xs = x.as_slice().unwrap();
        let ys = y.as_slice().unwrap();
        let time_tree = |pool: &rayon::ThreadPool| {
            pool.install(|| {
                let start = std::time::Instant::now();
                let value = std::hint::black_box(dot(x.view(), y.view()));
                (start.elapsed().as_secs_f64(), value)
            })
        };
        let time_baseline = |pool: &rayon::ThreadPool| {
            pool.install(|| {
                use rayon::prelude::*;
                let start = std::time::Instant::now();
                let value: f64 = xs
                    .par_chunks(CHUNK)
                    .zip(ys.par_chunks(CHUNK))
                    .map(|(ca, cb)| ca.iter().zip(cb).map(|(p, q)| p * q).sum::<f64>())
                    .sum();
                (start.elapsed().as_secs_f64(), std::hint::black_box(value))
            })
        };
        let time_serial = || {
            let start = std::time::Instant::now();
            let value = std::hint::black_box(x.dot(&y));
            (start.elapsed().as_secs_f64(), value)
        };
        for (pool, _) in pools.iter().zip(thread_counts) {
            let _ = time_tree(pool);
            let _ = time_baseline(pool);
            let _ = time_serial();
        }
        let mut samples: BTreeMap<(usize, &str), Vec<f64>> = BTreeMap::new();
        for _ in 0..5 {
            for (pool, threads) in pools.iter().zip(thread_counts) {
                let (secs, _) = time_tree(pool);
                samples.entry((threads, "tree")).or_default().push(secs);
                let (secs, _) = time_baseline(pool);
                samples
                    .entry((threads, "rayon_sum"))
                    .or_default()
                    .push(secs);
                let (secs, _) = time_serial();
                samples.entry((threads, "serial")).or_default().push(secs);
            }
        }
        let mut medians = BTreeMap::new();
        for ((threads, variant), mut times) in samples {
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = times[times.len() / 2];
            let min = times[0];
            let max = times[times.len() - 1];
            medians.insert((threads, variant), median);
            println!("{threads},{n},{variant},{median:.6e},{min:.6e},{max:.6e}");
        }
        let t1 = medians[&(1, "tree")];
        for threads in thread_counts {
            let tp = medians[&(threads, "tree")];
            let efficiency = t1 / (threads as f64 * tp);
            println!("efficiency n={n} threads={threads} {efficiency:.3}");
        }
    }
}
