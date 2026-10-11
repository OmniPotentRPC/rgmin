//! Strong scaling of the geometry-metric parallel path.
//!
//! Ignored in the default suite: it is a measurement, not a correctness
//! check. Run it in release:
//!
//! ```text
//! cargo test --release --features par --test precon_scaling -- --ignored --nocapture
//! ```
//!
//! The parallel path is the `par` feature. Pair ranges, matrix rows, and
//! dot-product chunks are a fixed partition reduced in index order.
//! Without `par` the same loops are serial, so a thread pool does not
//! change the work.

#![cfg(feature = "par")]

use std::time::Instant;

use ndarray::Array1;
use rgmin::{Lindh, LindhParams};

fn cluster(n: usize) -> (Array1<f64>, Vec<i32>) {
    let mut pos = Array1::zeros(3 * n);
    let z = vec![14i32; n];
    for i in 0..n {
        pos[3 * i] = (i % 10) as f64 * 2.35;
        pos[3 * i + 1] = ((i / 10) % 10) as f64 * 2.35;
        pos[3 * i + 2] = (i / 100) as f64 * 2.35;
    }
    (pos, z)
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn once(threads: usize, pos: &Array1<f64>, z: &[i32], rhs: &Array1<f64>) -> (f64, f64) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let mut params = LindhParams::default();
        params.mu = Some(1.0);
        params.r_cut = Some(4.0);
        let t0 = Instant::now();
        let metric = Lindh::assemble(pos.view(), z, &[], &params).unwrap();
        let assemble_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for _ in 0..8 {
            let solved = metric.solve(rhs.view()).expect("pcg");
            std::hint::black_box(&solved);
        }
        let solve_s = t1.elapsed().as_secs_f64();
        (assemble_s, solve_s)
    })
}

#[test]
#[ignore]
fn strong_scaling_table() {
    let threads: Vec<usize> = {
        let max = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let mut t = vec![1usize, 2, 4];
        if max > 4 && !t.contains(&max) {
            t.push(max);
        } else if !t.contains(&max) {
            t.push(max);
        }
        t.retain(|&n| n <= max);
        if t.is_empty() {
            t.push(1);
        }
        t
    };
    let (pos, z) = cluster(160);
    let rhs = Array1::from_iter((0..pos.len()).map(|i| ((i % 7) as f64 - 3.0) * 0.05));
    // One throwaway pass so the first recorded sample is not a cold start.
    let _ = once(1, &pos, &z, &rhs);

    const REPS: usize = 5;
    let mut assemble: Vec<Vec<f64>> = vec![Vec::new(); threads.len()];
    let mut solve: Vec<Vec<f64>> = vec![Vec::new(); threads.len()];
    for _rep in 0..REPS {
        for (k, &n) in threads.iter().enumerate() {
            let (a, s) = once(n, &pos, &z, &rhs);
            assemble[k].push(a);
            solve[k].push(s);
        }
    }

    println!(
        "threads,assemble_median_s,assemble_min_s,assemble_max_s,assemble_eff,solve_median_s,solve_min_s,solve_max_s,solve_eff"
    );
    let mut assemble_1 = assemble[0].clone();
    let mut solve_1 = solve[0].clone();
    let a1 = median(&mut assemble_1);
    let s1 = median(&mut solve_1);
    for (k, &n) in threads.iter().enumerate() {
        let mut a = assemble[k].clone();
        let mut s = solve[k].clone();
        let am = median(&mut a);
        let sm = median(&mut s);
        let amin = a.iter().copied().fold(f64::INFINITY, f64::min);
        let amax = a.iter().copied().fold(0.0, f64::max);
        let smin = s.iter().copied().fold(f64::INFINITY, f64::min);
        let smax = s.iter().copied().fold(0.0, f64::max);
        let aeff = a1 / (n as f64 * am);
        let seff = s1 / (n as f64 * sm);
        println!("{n},{am:.6},{amin:.6},{amax:.6},{aeff:.3},{sm:.6},{smin:.6},{smax:.6},{seff:.3}");
    }
}
