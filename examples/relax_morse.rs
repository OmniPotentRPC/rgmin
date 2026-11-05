//! Relax the 256-atom Pt Morse fixture with L-BFGS, FIRE and FIRE 2.0.
//!
//! Each method holds one `Solver` session and calls `step` until the
//! largest per-atom force falls below 0.01 eV/Å. Every energy and force
//! evaluation the session asks for is counted. The trace goes to stdout
//! as CSV (`method,step,force_calls,energy,fmax`); row 0 is the start,
//! evaluated outside the count. A one-line summary per method goes to
//! stderr.
//!
//! `cargo run --release --example relax_morse > trace.csv`

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, FireKind, Method, Oracle, Solver, StopNorm};

/// Pt Morse parameters of tests/newton_cg_morse.rs: eV, 1/Å, Å.
const D0: f64 = 0.7102;
const ALPHA: f64 = 1.6047;
const R0: f64 = 2.897;
const CUTOFF: f64 = 7.0;

const FMAX: f64 = 0.01;
const BUDGET: usize = 2000;

/// Positions and the orthorhombic cell of an eOn `.con` file.
fn read_con(path: &str) -> (Array1<f64>, [f64; 3]) {
    let text = std::fs::read_to_string(path).expect("read the fixture");
    let lines: Vec<&str> = text.lines().collect();
    let nums = |l: &str| -> Vec<f64> {
        l.split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect()
    };
    let cell = nums(lines[2]);
    let first = lines
        .iter()
        .position(|l| l.starts_with("Coordinates"))
        .unwrap()
        + 1;
    let mut x = Vec::new();
    for l in &lines[first..] {
        let v = nums(l);
        if v.len() < 3 {
            break;
        }
        x.extend(&v[..3]);
    }
    (Array1::from(x), [cell[0], cell[1], cell[2]])
}

/// Morse energy and gradient under the minimum image, shifted to zero at
/// the cutoff.
fn morse(x: ArrayView1<f64>, cell: [f64; 3]) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let xc = (-ALPHA * (CUTOFF - R0)).exp();
    let shift = D0 * (xc * xc - 2.0 * xc);
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            for a in 0..3 {
                let dx = x[3 * i + a] - x[3 * j + a];
                d[a] = dx - cell[a] * (dx / cell[a]).round();
            }
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r >= CUTOFF {
                continue;
            }
            let ex = (-ALPHA * (r - R0)).exp();
            e += D0 * (ex * ex - 2.0 * ex) - shift;
            let du = 2.0 * D0 * ALPHA * ex * (1.0 - ex);
            for a in 0..3 {
                g[3 * i + a] += du * d[a] / r;
                g[3 * j + a] -= du * d[a] / r;
            }
        }
    }
    (e, g)
}

fn main() {
    let path = format!(
        "{}/tests/data/morse_pos_0000.con",
        env!("CARGO_MANIFEST_DIR")
    );
    let (x0, cell) = read_con(&path);
    let (e0, g0) = morse(x0.view(), cell);
    let f0 = StopNorm::MaxAtom.of(g0.view());

    let methods = [
        ("lbfgs", Method::Lbfgs { memory: 20 }, 0.2),
        ("fire", Method::Fire { kind: FireKind::V1 }, 0.1),
        ("fire2", Method::Fire { kind: FireKind::V2 }, 0.1),
    ];
    println!("method,step,force_calls,energy,fmax");
    for (name, method, istep) in methods {
        let calls = AtomicUsize::new(0);
        let obj = Oracle::unbounded(x0.len(), |x: ArrayView1<f64>| {
            calls.fetch_add(1, Ordering::Relaxed);
            morse(x, cell)
        });
        let control = Control {
            maxiter: BUDGET,
            gtol: FMAX,
            istep,
            maxmove: None,
            ..Default::default()
        };
        let mut solver = Solver::new(method, control, x0.len());
        solver.set_atom_maxmove(0.2);
        solver.set_stop_norm(StopNorm::MaxAtom);

        let mut x = x0.clone();
        println!("{name},0,0,{e0:.6},{f0:.6}");
        let mut steps = 0;
        let mut last = f0;
        while steps < BUDGET && last >= FMAX {
            let rep = solver.step(&obj, &mut x).expect("step");
            steps += 1;
            last = rep.grad_norm;
            let n = calls.load(Ordering::Relaxed);
            println!("{name},{steps},{n},{:.6},{last:.6}", rep.value);
        }
        eprintln!(
            "{name}: {steps} steps, {} force calls, E = {:.6} eV, fmax = {last:.6} eV/Å",
            calls.load(Ordering::Relaxed),
            morse(x.view(), cell).0,
        );
    }
}
