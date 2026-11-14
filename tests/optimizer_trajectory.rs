//! Bit-identical iterates on a nearest-neighbour Morse chain.
//!
//! The hash is FNV-1a over the little-endian bits of every accepted
//! coordinate, then the value and the gradient norm. A change that only
//! drops a copy leaves the hash alone.

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, FireKind, LineSearch, Method, Solver, minimize_method};

const D0: f64 = 0.7102;
const ALPHA: f64 = 1.6047;
const R0: f64 = 2.897;

fn morse_chain(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut energy = 0.0;
    let mut grad = Array1::zeros(x.len());
    for i in 0..n.saturating_sub(1) {
        let mut diff = [0.0; 3];
        let mut r2 = 0.0;
        for a in 0..3 {
            diff[a] = x[3 * i + a] - x[3 * (i + 1) + a];
            r2 += diff[a] * diff[a];
        }
        let r = r2.sqrt();
        // `!(r > 0.0)` is true for a NaN length as well as a non-positive
        // one. A plain `r <= 0.0` would let a NaN through.
        if !r.is_finite() || r <= 0.0 {
            continue;
        }
        let ex = (-ALPHA * (r - R0)).exp();
        energy += D0 * (ex * ex - 2.0 * ex);
        let slope = (2.0 * D0 * ALPHA * ex * (1.0 - ex)) / r;
        for a in 0..3 {
            let piece = slope * diff[a];
            grad[3 * i + a] += piece;
            grad[3 * (i + 1) + a] -= piece;
        }
    }
    (energy, grad)
}

fn start_chain(natoms: usize) -> Array1<f64> {
    let mut x = Array1::zeros(natoms * 3);
    for i in 0..natoms {
        let t = i as f64;
        x[3 * i] = t * 3.15;
        x[3 * i + 1] = 0.20 * (t * 0.37).sin();
        x[3 * i + 2] = 0.12 * (t * 0.21).cos();
    }
    x
}

fn mix(hash: u64, bits: u64) -> u64 {
    hash ^ bits.wrapping_mul(0x100000001b3)
}

fn hash_report(mut hash: u64, value: f64, coords: &Array1<f64>, grad_norm: f64) -> u64 {
    hash = mix(hash, value.to_bits());
    hash = mix(hash, grad_norm.to_bits());
    for c in coords.iter() {
        hash = mix(hash, c.to_bits());
    }
    hash
}

fn control(steps: usize) -> Control {
    Control {
        maxiter: steps,
        gtol: 0.0,
        istep: 0.1,
        maxmove: None,
        stop_norm: rgmin::StopNorm::Two,
        ftol_rel: None,
        noise_floor: None,
    }
}

fn wolfe() -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 20,
    }
}

fn hash_oneshot(method: Method, linesearch: LineSearch, natoms: usize, steps: usize) -> u64 {
    let obj = rgmin::Oracle::unbounded(natoms * 3, morse_chain);
    let report = minimize_method(
        &obj,
        start_chain(natoms),
        &control(steps),
        method,
        linesearch,
    )
    .unwrap();
    hash_report(
        0xcbf29ce484222325,
        report.value,
        &report.coords,
        report.grad_norm,
    )
}

fn hash_session(method: Method, natoms: usize, steps: usize) -> u64 {
    let obj = rgmin::Oracle::unbounded(natoms * 3, morse_chain);
    let mut x = start_chain(natoms);
    let mut solver = Solver::new(method, control(steps), x.len());
    solver.set_linesearch(wolfe());
    let mut hash = 0xcbf29ce484222325;
    for _ in 0..steps {
        let report = solver.step(&obj, &mut x).unwrap();
        hash = hash_report(hash, report.value, &report.coords, report.grad_norm);
    }
    hash
}

#[test]
fn morse_chain_iterates_stay_bit_identical() {
    const N: usize = 24;
    const STEPS: usize = 8;
    let got = [
        (
            "lbfgs",
            hash_oneshot(Method::lbfgs(), LineSearch::default(), N, STEPS),
        ),
        (
            "lbfgs-wolfe",
            hash_oneshot(Method::lbfgs(), wolfe(), N, STEPS),
        ),
        (
            "nlcg",
            hash_oneshot(Method::polak_ribiere(), wolfe(), N, STEPS),
        ),
        ("sd", hash_oneshot(Method::Steepest, wolfe(), N, STEPS)),
        (
            "adam",
            hash_oneshot(Method::adam(), LineSearch::default(), N, STEPS),
        ),
        ("lbfgs-session", hash_session(Method::lbfgs(), N, STEPS)),
        (
            "fire1",
            hash_session(Method::Fire { kind: FireKind::V1 }, N, STEPS),
        ),
        (
            "fire2",
            hash_session(Method::Fire { kind: FireKind::V2 }, N, STEPS),
        ),
    ];
    for (name, hash) in got {
        let want = match name {
            "lbfgs" => 0x20cce606614e993a,
            "lbfgs-wolfe" => 0xc2156ead0b735471,
            "nlcg" => 0xf79e4ea9957c7702,
            "sd" => 0x2252179ee1695692,
            "adam" => 0xbf23ac7de4adede3,
            "lbfgs-session" => 0x2fcc34118a8f8315,
            "fire1" => 0x91356a7e96d8c8d4,
            "fire2" => 0xdbdb7fbf910adad6,
            _ => panic!("unknown {name}"),
        };
        assert_eq!(hash, want, "{name} trajectory hash {hash:#x} != {want:#x}");
    }
}
