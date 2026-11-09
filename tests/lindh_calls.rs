//! Three-bond start: plain L-BFGS against an opt-in Lindh model.
//!
//! The capped run sets a 0.2 per-atom cap, which selects the geometry
//! step. Accept::None stays the session default.

use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rgmin::{BondTerm, Control, FfParams, FfTerms, LindhParams, Method, Solver};

struct Bonds {
    bounds: Bounds<f64>,
    calls: AtomicUsize,
}

fn dist(x: ArrayView1<f64>, i: usize, j: usize) -> (f64, [f64; 3]) {
    let mut d = [0.0; 3];
    for c in 0..3 {
        d[c] = x[3 * i + c] - x[3 * j + c];
    }
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    (r, d)
}

impl Objective<f64> for Bonds {
    fn dim(&self) -> usize {
        9
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Bonds {
    fn dim(&self) -> usize {
        9
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for Bonds {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let targets = [(0usize, 1usize, 0.96), (0, 2, 0.96), (1, 2, 1.52)];
        let k = 25.0;
        let mut g = Array1::zeros(9);
        let mut f = 0.0;
        for (i, j, r0) in targets {
            let (r, d) = dist(x, i, j);
            let gap = r - r0;
            f += 0.5 * k * gap * gap;
            // Exact coincidence has no unit vector. Every positive
            // distance, including one below 1e-12, keeps its force.
            if r > 0.0 {
                let s = k * gap / r;
                for c in 0..3 {
                    g[3 * i + c] += s * d[c];
                    g[3 * j + c] -= s * d[c];
                }
            }
        }
        (f, g)
    }
}

fn start() -> Array1<f64> {
    Array1::from_vec(vec![0.0, 0.05, -0.02, 1.15, 0.08, 0.0, 0.15, 1.05, 0.12])
}

fn control() -> Control {
    Control {
        maxiter: 80,
        gtol: 1e-6,
        istep: 0.2,
        maxmove: None,
        ..Control::default()
    }
}

struct Out {
    calls: usize,
    steps: usize,
    value: f64,
    gnorm: f64,
    shortest: f64,
}

fn shortest(x: ArrayView1<f64>) -> f64 {
    let (r01, _) = dist(x, 0, 1);
    let (r02, _) = dist(x, 0, 2);
    let (r12, _) = dist(x, 1, 2);
    r01.min(r02).min(r12)
}

fn run(lindh: bool, cap: Option<f64>, limit: usize) -> Out {
    let obj = Bonds {
        bounds: Bounds::new(Array1::from_elem(9, -1e3), Array1::from_elem(9, 1e3), 0.0),
        calls: AtomicUsize::new(0),
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 8 }, control(), 9);
    if let Some(c) = cap {
        solver.set_atom_maxmove(c);
    }
    let mut x = start();
    if lindh {
        solver
            .set_lindh(x.view(), &[8, 1, 1], &[], &LindhParams::default())
            .unwrap();
    }
    let mut steps = 0;
    let mut value = 0.0;
    let mut gnorm = f64::INFINITY;
    for _ in 0..limit {
        let report = solver.step(&obj, &mut x).unwrap();
        steps = report.steps;
        value = report.value;
        gnorm = report.grad_norm;
        if gnorm < 1e-6 {
            break;
        }
    }
    Out {
        calls: obj.calls.load(Ordering::Relaxed),
        steps,
        value,
        gnorm,
        shortest: shortest(x.view()),
    }
}

fn run_ff(cap: Option<f64>, limit: usize) -> Out {
    let obj = Bonds {
        bounds: Bounds::new(Array1::from_elem(9, -1e3), Array1::from_elem(9, 1e3), 0.0),
        calls: AtomicUsize::new(0),
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 8 }, control(), 9);
    if let Some(c) = cap {
        solver.set_atom_maxmove(c);
    }
    let mut x = start();
    let bonds = [
        BondTerm {
            i: 0,
            j: 1,
            k: 25.0,
        },
        BondTerm {
            i: 0,
            j: 2,
            k: 25.0,
        },
        BondTerm {
            i: 1,
            j: 2,
            k: 25.0,
        },
    ];
    let terms = FfTerms {
        bonds: &bonds,
        ..FfTerms::default()
    };
    solver
        .set_ff(x.view(), terms, &FfParams::default())
        .unwrap();
    let mut steps = 0;
    let mut value = 0.0;
    let mut gnorm = f64::INFINITY;
    for _ in 0..limit {
        let report = solver.step(&obj, &mut x).unwrap();
        steps = report.steps;
        value = report.value;
        gnorm = report.grad_norm;
        if gnorm < 1e-6 {
            break;
        }
    }
    Out {
        calls: obj.calls.load(Ordering::Relaxed),
        steps,
        value,
        gnorm,
        shortest: shortest(x.view()),
    }
}

fn show(label: &str, out: &Out) {
    println!(
        "{label} calls={} steps={} value={:.6e} gnorm={:.6e} shortest={:.6e}",
        out.calls, out.steps, out.value, out.gnorm, out.shortest
    );
}

fn holds_a_bond(out: &Out) {
    assert!(
        out.shortest > 0.2,
        "collapsed, shortest bond {}",
        out.shortest
    );
}

#[test]
fn three_bond_capped_and_plain() {
    let plain_cap = run(false, Some(0.2), 80);
    let lindh_cap = run(true, Some(0.2), 80);
    let ff_cap = run_ff(Some(0.2), 80);
    let plain = run(false, None, 40);
    show("capped-plain", &plain_cap);
    show("capped-lindh", &lindh_cap);
    show("capped-ff", &ff_cap);
    show("default-plain", &plain);
    assert!(plain_cap.gnorm < 1e-6, "capped plain {}", plain_cap.gnorm);
    assert!(lindh_cap.gnorm < 1e-6, "capped lindh {}", lindh_cap.gnorm);
    assert!(ff_cap.gnorm < 1e-6, "capped ff {}", ff_cap.gnorm);
    assert!(plain.gnorm < 1e-6, "plain {}", plain.gnorm);
    holds_a_bond(&plain_cap);
    holds_a_bond(&lindh_cap);
    holds_a_bond(&ff_cap);
    holds_a_bond(&plain);
    assert_eq!((plain_cap.calls, plain_cap.steps), (10, 9));
    assert_eq!((lindh_cap.calls, lindh_cap.steps), (19, 14));
    assert_eq!((ff_cap.calls, ff_cap.steps), (12, 8));
    assert_eq!((plain.calls, plain.steps), (11, 10));
}

#[test]
#[ignore = "pending: a step cap or trust region for model-Hessian or FF H0, versus a line search"]
fn three_bond_uncapped_lindh() {
    let lindh = run(true, None, 40);
    show("default-lindh", &lindh);
    holds_a_bond(&lindh);
    assert!(lindh.gnorm < 1e-6, "lindh gnorm {}", lindh.gnorm);
}

#[test]
#[ignore = "pending: a step cap or trust region for model-Hessian or FF H0, versus a line search"]
fn three_bond_uncapped_ff() {
    let ff = run_ff(None, 40);
    show("default-ff", &ff);
    holds_a_bond(&ff);
    assert!(ff.gnorm < 1e-6, "ff gnorm {}", ff.gnorm);
}
