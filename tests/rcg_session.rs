//! Riemannian CG as a session method: it obeys the caps, the constraints
//! and the stops that every other method in the session obeys.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    ConstraintError, Control, DistanceConstraints, Error, LineSearch, ManifoldKind, Method, Oracle,
    Solver, Status,
};

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
    // The curvature point of this bowl sits far outside the cap. Armijo
    // backtracking takes the opening trial, which the cap still bounds.
    solver.set_linesearch(LineSearch::backtracking(1e-4, 0.5, 8));
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
    solver.set_linesearch(LineSearch::backtracking(1e-4, 0.5, 8));
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
    solver.set_linesearch(LineSearch::backtracking(1e-4, 0.5, 8));
    solver.set_atom_maxmove(0.5);
    let mut x = Array1::zeros(6);
    let report = solver.run(&obj, &mut x).unwrap();
    assert!(report.grad_norm <= 1e-8, "grad {}", report.grad_norm);
    let expect = array![10.0, 0.0, 0.0, 10.0, 0.0, 0.0];
    assert!((&x - &expect).iter().all(|d| d.abs() < 1e-6), "x = {x}");
}

fn pull_on_the_last_atom(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mut g = Array1::zeros(x.len());
    g[x.len() - 2] = 1.0;
    (x[x.len() - 2], g)
}

#[test]
fn rcg_reports_a_projection_that_fails_on_a_dependent_set() {
    // Four coplanar atoms with all six distances fixed: the rows of the
    // constraint Jacobian are dependent, so no tangent space exists.
    let start = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let pairs = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let set = DistanceConstraints::at(pairs, start.view()).unwrap();
    let mut solver = Solver::new(Method::rcg(), Control::default(), 12);
    solver.set_distance_constraints(set).unwrap();
    let obj = Oracle::unbounded(12, pull_on_the_last_atom);
    let mut x = start.clone();
    match solver.step(&obj, &mut x) {
        Err(Error::Constraint(ConstraintError::Dependent { .. })) => {}
        other => panic!("expected a dependent-set error, got {other:?}"),
    }
    assert_eq!(x, start, "a refused step leaves x");
}

fn tilt(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let slope = [0.0, 0.0, 1.0, 0.0, 0.0, 2.0];
    let value = x.iter().zip(slope).map(|(a, c)| a * c).sum();
    (value, Array1::from_iter(slope))
}

fn bond_length(x: &Array1<f64>) -> f64 {
    (0..3)
        .map(|k| (x[k] - x[3 + k]).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn rcg_keeps_the_manifold_and_the_bonds_together() {
    // Two atoms on the unit sphere of R^6, a distance of 1 apart.
    let start = array![0.6, 0.0, 0.0, 0.0, 0.8, 0.0];
    assert!((bond_length(&start) - 1.0).abs() < 1e-15);
    let set = DistanceConstraints::at(vec![(0, 1)], start.view()).unwrap();
    let mut solver = Solver::new(Method::rcg(), Control::default().with_gtol(1e-10), 6);
    solver.set_distance_constraints(set).unwrap();
    solver.set_manifold(ManifoldKind::Sphere);
    let obj = Oracle::unbounded(6, tilt);
    let mut x = start.clone();
    let first = tilt(x.view()).0;
    for _ in 0..40 {
        solver.step(&obj, &mut x).unwrap();
        let radius = x.dot(&x).sqrt();
        assert!(
            (radius - 1.0).abs() < 1e-8,
            "left the sphere: |x| = {radius}"
        );
        assert!(
            (bond_length(&x) - 1.0).abs() < 1e-8,
            "left the bonds: {}",
            bond_length(&x)
        );
    }
    assert!(
        tilt(x.view()).0 < first - 0.1,
        "no descent: {}",
        tilt(x.view()).0
    );
}

/// The two atoms of a dimer pulled apart along the bond: the force on the
/// dimer is all constraint force.
fn pull_along_the_bond(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mut g = Array1::zeros(x.len());
    g[0] = -1.0;
    g[3] = 1.0;
    (x[3] - x[0], g)
}

fn dimer() -> (Array1<f64>, Solver) {
    let start = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let set = DistanceConstraints::at(vec![(0, 1)], start.view()).unwrap();
    let mut solver = Solver::new(Method::rcg(), Control::default().with_gtol(1e-8), 6);
    solver.set_distance_constraints(set).unwrap();
    (start, solver)
}

#[test]
fn rcg_stops_on_the_projected_gradient_of_a_constrained_start() {
    let (start, mut solver) = dimer();
    let obj = Oracle::unbounded(6, pull_along_the_bond);
    let mut x = start.clone();
    let report = solver.step(&obj, &mut x).unwrap();
    assert!(report.grad_norm < 1e-12, "grad_norm {}", report.grad_norm);
    assert_eq!(report.status, Status::Tolerance);
    let grad = report.grad.expect("the gradient is reported");
    assert!(grad.iter().all(|g| g.abs() < 1e-12), "grad {grad}");
    assert_eq!(x, start);
}

#[test]
fn rcg_assesses_a_constrained_start_by_the_projected_gradient() {
    let (start, _) = dimer();
    let set = DistanceConstraints::at(vec![(0, 1)], start.view()).unwrap();
    let control = Control::default().with_gtol(1e-8).with_maxiter(0);
    let mut solver = Solver::new(Method::rcg(), control, 6);
    solver.set_distance_constraints(set).unwrap();
    let obj = Oracle::unbounded(6, pull_along_the_bond);
    let mut x = start;
    let report = solver.run(&obj, &mut x).unwrap();
    assert_eq!(report.status, Status::Tolerance, "{report:?}");
}

#[test]
fn rcg_reports_the_projected_gradient_after_a_constrained_move() {
    let start = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let set = DistanceConstraints::at(vec![(0, 1)], start.view()).unwrap();
    let mut solver = Solver::new(Method::rcg(), Control::default().with_gtol(1e-6), 6);
    solver.set_distance_constraints(set).unwrap();
    // Hold atom 0 at the origin and pull atom 1 toward (3, 3, 0). On the
    // unit bond sphere it settles at the point nearest that target, where
    // the whole force is normal.
    let obj = Oracle::unbounded(6, |x: ArrayView1<f64>| {
        let g = array![x[0], x[1], x[2], x[3] - 3.0, x[4] - 3.0, x[5]];
        (0.5 * g.dot(&g), g)
    });
    let mut x = start.clone();
    let mut report = None;
    for _ in 0..200 {
        let r = solver.step(&obj, &mut x).unwrap();
        let done = r.status != Status::Iterating;
        report = Some(r);
        if done {
            break;
        }
    }
    let report = report.unwrap();
    assert_eq!(report.status, Status::Tolerance, "{report:?}");
    assert!(report.grad_norm <= 1e-6, "grad_norm {}", report.grad_norm);
    let grad = report.grad.unwrap();
    let bond = [x[3] - x[0], x[4] - x[1], x[5] - x[2]];
    let along: f64 = (0..3).map(|k| bond[k] * (grad[3 + k] - grad[k])).sum();
    assert!(
        along.abs() < 1e-8,
        "reported gradient is not tangent: {along}"
    );
}
