//! `Lbfgs` with and without the relative-decrease stop `ftol`.
//!
//! Default `Lbfgs` (memory 8, strong Wolfe, `norminf(g) < 1e-6`) on the
//! extended Rosenbrock function from `(-1.2, 1, -1.2, 1, ...)` and on
//! Lennard-Jones clusters from a jittered cubic grid, once with
//! `ftol = 0` and once with `ftol = LBFGSB_FTOL`. Each row gives the
//! evaluations, the final value and the largest gradient component.
//!
//! `cargo run --release --example lbfgs_ftol`

use ndarray::{Array1, ArrayView1};
use rgmin::{LBFGSB_FTOL, Lbfgs};

/// Iterations; only a stop test ends these runs.
const BUDGET: usize = 100_000;
/// Grid spacing and jitter of the seven-atom start in tests/stop_norm.rs.
const SPACING: f64 = 1.1;
const JITTER: f64 = 0.03;
const ROSENBROCK_DIMS: [usize; 3] = [2, 10, 100];
const CLUSTER_SIZES: [usize; 3] = [7, 13, 38];

fn rosenbrock(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mut f = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..x.len() - 1 {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        f += a * a + 100.0 * b * b;
        g[i] += -2.0 * a - 400.0 * x[i] * b;
        g[i + 1] += 200.0 * b;
    }
    (f, g)
}

fn lennard_jones(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let d: [f64; 3] = std::array::from_fn(|a| x[3 * i + a] - x[3 * j + a]);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let s6 = r2.powi(-3);
            e += 4.0 * (s6 * s6 - s6);
            let du_over_r = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
            for a in 0..3 {
                g[3 * i + a] += du_over_r * d[a];
                g[3 * j + a] -= du_over_r * d[a];
            }
        }
    }
    (e, g)
}

fn rosenbrock_start(n: usize) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }))
}

fn grid_start(atoms: usize) -> Array1<f64> {
    let side = (atoms as f64).cbrt().ceil() as usize;
    let mut x = Vec::with_capacity(3 * atoms);
    for k in 0..atoms {
        let cell = [k % side, (k / side) % side, k / (side * side)];
        for (a, c) in cell.iter().enumerate() {
            x.push(SPACING * *c as f64 + JITTER * ((3 * k + a) as f64).sin());
        }
    }
    Array1::from(x)
}

fn run(label: &str, x0: &Array1<f64>, fg: fn(ArrayView1<f64>) -> (f64, Array1<f64>)) {
    for ftol in [0.0, LBFGSB_FTOL] {
        let mut opt = Lbfgs::default();
        opt.ftol = ftol;
        let (f, x, evals) = opt.minimize(x0.view(), BUDGET, |v| Some(fg(v)));
        let gmax = fg(x.view()).1.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        println!("{label:>14} {ftol:>9.2e} {evals:>6} {f:>22.15e} {gmax:>9.2e}");
    }
}

fn main() {
    println!(
        "{:>14} {:>9} {:>6} {:>22} {:>9}",
        "problem", "ftol", "evals", "f", "gmax"
    );
    for n in ROSENBROCK_DIMS {
        run(&format!("rosenbrock {n}"), &rosenbrock_start(n), rosenbrock);
    }
    for atoms in CLUSTER_SIZES {
        run(&format!("LJ{atoms}"), &grid_start(atoms), lennard_jones);
    }
}
