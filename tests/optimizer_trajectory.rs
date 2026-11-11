//! Bit-identical iterates of the minimizers on a nearest-neighbour Morse chain.
//!
//! The hash is FNV-1a over the little-endian bits of every accepted
//! coordinate, then the final value and gradient. A change to clipping,
//! copying, or a dot product that only reorders work leaves this hash
//! alone; a change to the step does not.

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, FireKind, FireVariant, LineSearch, Method, Solver, minimize_method};

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
        if !(r > 0.0) {
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

fn wolfe() -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 20,
    }
}

fn control(steps: usize) -> Control {
    Control {
        maxiter: steps,
        gtol: 0.0,
        istep: 0.1,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    }
}

fn hash_oneshot(method: Method, natoms: usize, steps: usize) -> u64 {
    let obj = rgmin::Oracle::unbounded(natoms * 3, morse_chain);
    let report =
        minimize_method(&obj, start_chain(natoms), &control(steps), method, wolfe()).unwrap();
    hash_report(
        0xcbf29ce484222325,
        report.value,
        &report.coords,
        report.grad_norm,
    )
}

fn hash_session(method: Method, natoms: usize, steps: usize, fire2: bool) -> u64 {
    let obj = rgmin::Oracle::unbounded(natoms * 3, morse_chain);
    let mut x = start_chain(natoms);
    let mut solver = Solver::new(method, control(steps), x.len());
    solver.set_linesearch(wolfe());
    if fire2 {
        solver.set_fire_variant(FireVariant::Guenole2020);
    }
    let mut hash = 0xcbf29ce484222325;
    for _ in 0..steps {
        let report = solver.step(&obj, &mut x).unwrap();
        hash = hash_report(hash, report.value, &report.coords, report.grad_norm);
    }
    hash
}

fn expect(name: &str, got: u64) {
    let want = match name {
        "lbfgs" => 0x8fa602d6fc6cce2d,
        "nlcg" => 0x0e21edbc2819e7a2,
        "sd" => 0xe7a20697ff9ba6cc,
        "adam" => 0xa85a9ca41a68cea4,
        "lbfgs-session" => 0xd76cccacd77e3562,
        "fire1" => 0x9125a47e9685b768,
        "fire2" => 0xee039bc89c6c44cb,
        "fire2g" => 0xc7d4e163bffb635d,
        "quickmin" => 0x1e6ba25e9762d1e3,
        "bb" => 0xb0f76dc7e391359c,
        "ode12r" => 0x630962b5e1cd9d47,
        "diis" => 0x7ba4c45fabdac0a1,
        _ => panic!("unknown {name}"),
    };
    assert_eq!(got, want, "{name} trajectory hash {got:#x} != {want:#x}");
}

#[test]
fn morse_chain_iterates_stay_bit_identical() {
    const N: usize = 24;
    const STEPS: usize = 8;
    expect("lbfgs", hash_oneshot(Method::lbfgs(), N, STEPS));
    expect("nlcg", hash_oneshot(Method::polak_ribiere(), N, STEPS));
    expect("sd", hash_oneshot(Method::Steepest, N, STEPS));
    expect("adam", hash_oneshot(Method::adam(), N, STEPS));
    expect(
        "lbfgs-session",
        hash_session(Method::lbfgs(), N, STEPS, false),
    );
    expect(
        "fire1",
        hash_session(Method::Fire { kind: FireKind::V1 }, N, STEPS, false),
    );
    expect(
        "fire2",
        hash_session(Method::Fire { kind: FireKind::V2 }, N, STEPS, false),
    );
    expect(
        "fire2g",
        hash_session(Method::Fire { kind: FireKind::V2 }, N, STEPS, true),
    );
    expect("quickmin", hash_session(Method::QuickMin, N, STEPS, false));
    expect("bb", hash_session(Method::Bb, N, STEPS, false));
    expect("ode12r", hash_session(Method::Ode12r, N, STEPS, false));
    expect("diis", hash_session(Method::diis(), N, STEPS, false));
}
