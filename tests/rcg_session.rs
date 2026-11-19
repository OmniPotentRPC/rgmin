//! Riemannian CG as a session method: it obeys the caps, the constraints
//! and the stops that every other method in the session obeys.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Oracle, Solver};

/// `f = |x - c|^2 / 2` with `c` ten units from the start.
fn far_bowl(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let target = [10.0, 0.0, 0.0, 10.0, 0.0, 0.0];
    let g = Array1::from_iter(x.iter().zip(target).map(|(a, c)| a - c));
    (0.5 * g.dot(&g), g)
}

fn atom_motion(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (0..a.len() / 3)
        .map(|i| {
            (0..3)
                .map(|k| (a[3 * i + k] - b[3 * i + k]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max)
}

#[test]
fn rcg_obeys_the_per_atom_cap() {
    let obj = Oracle::unbounded(6, far_bowl);
    let mut solver = Solver::new(Method::rcg(), Control::default(), 6);
    solver.set_atom_maxmove(0.01);
    let start = Array1::zeros(6);
    let mut x = start.clone();
    solver.step(&obj, &mut x).unwrap();
    let moved = atom_motion(&start, &x);
    assert!(moved > 0.0, "no move");
    assert!(moved <= 0.01 * (1.0 + 1e-9), "moved {moved} past 0.01");
}

#[test]
fn rcg_obeys_the_euclidean_cap() {
    let obj = Oracle::unbounded(6, far_bowl);
    let mut solver = Solver::new(Method::rcg(), Control::default().with_maxmove(0.05), 6);
    let start = Array1::zeros(6);
    let mut x = start.clone();
    solver.step(&obj, &mut x).unwrap();
    let moved = (&x - &start).dot(&(&x - &start)).sqrt();
    assert!(moved > 0.0, "no move");
    assert!(moved <= 0.05 * (1.0 + 1e-9), "moved {moved} past 0.05");
}

#[test]
fn a_capped_rcg_still_reaches_the_minimum() {
    let obj = Oracle::unbounded(6, far_bowl);
    let mut solver = Solver::new(
        Method::rcg(),
        Control::default().with_maxiter(400).with_gtol(1e-8),
        6,
    );
    solver.set_atom_maxmove(0.5);
    let mut x = Array1::zeros(6);
    let report = solver.run(&obj, &mut x).unwrap();
    assert!(report.grad_norm <= 1e-8, "grad {}", report.grad_norm);
    let expect = array![10.0, 0.0, 0.0, 10.0, 0.0, 0.0];
    assert!((&x - &expect).iter().all(|d| d.abs() < 1e-6), "x = {x}");
}
