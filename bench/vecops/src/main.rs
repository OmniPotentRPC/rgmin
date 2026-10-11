//! One process measurement of `dot` or `axpy`.
//!
//! Prints the elapsed seconds of a fixed iteration count. The driver
//! interleaves this binary with the same binary built from `main`.

use std::hint::black_box;
use std::time::Instant;

use ndarray::Array1;
use rgmin::vecops::{axpy, dot};

fn main() {
    let mut args = std::env::args().skip(1);
    let kernel = args.next().expect("kernel dot|axpy");
    let n: usize = args.next().expect("n").parse().expect("n");
    let iters: u64 = args.next().expect("iters").parse().expect("iters");
    let mut x = Array1::<f64>::zeros(n);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = i as f64;
        x[i] = (t * 0.001).sin();
        y[i] = (t * 0.001).cos();
    }
    if n > 0 {
        x[0] = -0.0;
    }
    if n > 1 {
        x[1] = f64::from_bits(0x0000_0000_0000_0001);
    }
    let started = Instant::now();
    match kernel.as_str() {
        "dot" => {
            let mut acc = 0.0;
            for _ in 0..iters {
                acc += dot(x.view(), y.view());
            }
            black_box(acc);
        }
        "axpy" => {
            for _ in 0..iters {
                axpy(1.0000001, x.view(), &mut y);
                axpy(-1.0000001, x.view(), &mut y);
            }
            black_box(y[0]);
        }
        other => panic!("unknown kernel {other}"),
    }
    println!("{:.9}", started.elapsed().as_secs_f64());
}
