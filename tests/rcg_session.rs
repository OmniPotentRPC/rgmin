//! Riemannian CG as a session method: it obeys the caps, the constraints
//! and the stops that every other method in the session obeys.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    ConstraintError, Control, DistanceConstraints, Error, LineSearch, ManifoldKind, Method,
    NoiseFloor, Oracle, Solver, Status,
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

/// `f = |x|^2 / 2` on two atoms of the x axis pulls along the bond, so
/// its gradient cut by the bond tangent is zero wherever they sit. A
/// start twice the bond length apart is stationary only once it is put
/// on the bond, by a zero-budget run and by a step alike.
#[test]
fn rcg_puts_a_start_on_its_bonds_before_it_reports_tolerance() {
    let set = DistanceConstraints::new(vec![(0, 1)], vec![1.0]).unwrap();
    let start = array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let obj = Oracle::unbounded(6, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
    for maxiter in [0, 50] {
        let control = Control::default().with_gtol(1e-8).with_maxiter(maxiter);
        let mut solver = Solver::new(Method::rcg(), control, 6);
        solver.set_distance_constraints(set.clone()).unwrap();
        let mut x = start.clone();
        let report = solver.run(&obj, &mut x).unwrap();
        assert_eq!(report.status, Status::Tolerance, "maxiter {maxiter}");
        assert!(
            set.excess(x.view()) <= 1.0,
            "maxiter {maxiter}: bond length {}",
            bond_length(&x)
        );
        assert_eq!(report.value, 0.5 * x.dot(&x), "maxiter {maxiter}");
    }
}
#[test]
fn session_projects_the_euclidean_gradient_once_on_the_positive_orthant() {
    // f = 0.5 * ||x - (1, 1)||^2. At x = (2, 0.5) the Euclidean gradient
    // is (1, -0.5). On the positive orthant the Riemannian gradient scales
    // each entry by x_i^2, which is not a projection: a second scaling is
    // a different direction and a different retraction.
    let x0 = array![2.0, 0.5];
    let g = array![1.0, -0.5];
    let istep = 1e-3;
    let once = retracted(&x0, &scaled(&g, &x0), istep);
    let twice = retracted(&x0, &scaled(&scaled(&g, &x0), &x0), istep);
    let gap: f64 = once
        .iter()
        .zip(twice.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        gap > 1e-6,
        "the two projections retract to the same point: {once} and {twice}"
    );

    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let d0 = x[0] - 1.0;
        let d1 = x[1] - 1.0;
        (0.5 * (d0 * d0 + d1 * d1), array![d0, d1])
    });
    let mut solver = Solver::new(
        Method::rcg(),
        Control::default().with_istep(istep).with_gtol(1e-12),
        2,
    );
    solver.set_positive(2);
    // Armijo accepts the opening trial. Strong Wolfe would not, because a
    // short step leaves the slope almost unchanged.
    solver.set_linesearch(LineSearch::backtracking(1e-4, 0.5, 8));
    let mut x = x0.clone();
    solver.step(&obj, &mut x).unwrap();
    for i in 0..2 {
        assert!(
            (x[i] - once[i]).abs() < 1e-10,
            "x[{i}] = {} from one projection {}, and twice would be {}",
            x[i],
            once[i],
            twice[i]
        );
    }
}

/// Componentwise `g_i * x_i^2`, the positive-orthant `egrad2rgrad`.
fn scaled(g: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    Array1::from_iter(g.iter().zip(x.iter()).map(|(gi, xi)| gi * xi * xi))
}

/// Retraction of the steepest step of length `istep` in the orthant metric.
fn retracted(x: &Array1<f64>, rgrad: &Array1<f64>, istep: f64) -> Array1<f64> {
    let speed = x
        .iter()
        .zip(rgrad.iter())
        .map(|(xi, ri)| (ri / xi).powi(2))
        .sum::<f64>()
        .sqrt();
    let alpha = istep / speed;
    Array1::from_iter(x.iter().zip(rgrad.iter()).map(|(xi, ri)| {
        let dir = -ri;
        xi * (alpha * dir / xi).exp()
    }))
}

/// A start off the manifold is retracted onto it by a zero step before the
/// first evaluation, as `minimize_rcg` does, and the caller's point holds
/// the projected start. The run then reports at that point.
#[test]
fn an_off_manifold_start_is_projected_onto_the_sphere() {
    let start = array![3.0, 0.0, 4.0, 0.0, 0.0, 0.0];
    let obj = Oracle::unbounded(6, tilt);
    let mut solver = Solver::new(Method::rcg(), Control::default().with_gtol(1e6), 6);
    solver.set_manifold(ManifoldKind::Sphere);
    let mut x = start.clone();
    let report = solver.step(&obj, &mut x).unwrap();
    assert_eq!(report.status, Status::Tolerance, "{report:?}");
    let expected = &start / 5.0;
    for i in 0..6 {
        assert!(
            (x[i] - expected[i]).abs() < 1e-14,
            "x[{i}] = {} is not the closest point {}",
            x[i],
            expected[i]
        );
    }
    assert_eq!(report.value, tilt(x.view()).0, "the value is read at x");

    let mut solver = Solver::new(Method::rcg(), Control::default().with_gtol(1e-6), 6);
    solver.set_manifold(ManifoldKind::Sphere);
    let mut x = start;
    let report = solver.run(&obj, &mut x).unwrap();
    assert_eq!(report.status, Status::Tolerance, "{report:?}");
    assert!(
        (x.dot(&x).sqrt() - 1.0).abs() < 1e-12,
        "|x| = {}",
        x.dot(&x).sqrt()
    );
}

/// `f = (x0 - 2)^2 / 2 + 4 (x1 - 2)^2` is stationary at the off-sphere
/// start `(2, 2)`. Its projection `(1, 1) / sqrt(2)` carries a gradient
/// with a tangent part, so a zero-budget run that reads the start where a
/// step would read it reports the budget at a point on the sphere.
#[test]
fn a_zero_budget_run_projects_an_off_manifold_start() {
    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let (a, b) = (x[0] - 2.0, x[1] - 2.0);
        (0.5 * a * a + 4.0 * b * b, array![a, 8.0 * b])
    });
    let control = Control::default().with_gtol(1e-8).with_maxiter(0);
    let mut solver = Solver::new(Method::rcg(), control, 2);
    solver.set_manifold(ManifoldKind::Sphere);
    let mut x = array![2.0, 2.0];
    let report = solver.run(&obj, &mut x).unwrap();
    assert_eq!(report.status, Status::Budget, "{report:?}");
    let side = std::f64::consts::FRAC_1_SQRT_2;
    assert!(
        (x[0] - side).abs() < 1e-14 && (x[1] - side).abs() < 1e-14,
        "x = {x} is not the closest point on the sphere"
    );
    assert!(report.grad_norm > 1.0, "grad_norm {}", report.grad_norm);
    assert_eq!(report.value, obj.value_and_gradient(x.view()).0);
}

/// The box clips every trial of the step, so the objective is never read
/// outside it and the line search measures the curve that is walked.
struct Boxed {
    bounds: Bounds<f64>,
    lo: Array1<f64>,
    hi: Array1<f64>,
    outside: std::sync::atomic::AtomicBool,
}

impl Boxed {
    fn new(lo: Array1<f64>, hi: Array1<f64>) -> Self {
        Self {
            bounds: Bounds::new(lo.clone(), hi.clone(), 0.0),
            lo,
            hi,
            outside: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl Objective<f64> for Boxed {
    fn dim(&self) -> usize {
        self.bounds.dims
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Boxed {
    fn dim(&self) -> usize {
        self.bounds.dims
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for Boxed {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let out = x
            .iter()
            .zip(self.lo.iter().zip(self.hi.iter()))
            .any(|(v, (lo, hi))| v < lo || v > hi);
        if out {
            self.outside
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let target = [4.0, -4.0];
        let g = Array1::from_iter(x.iter().zip(target).map(|(a, c)| a - c));
        (0.5 * g.dot(&g), g)
    }
}

#[test]
fn rcg_trials_stay_inside_the_box() {
    let obj = Boxed::new(array![-1.0, -1.0], array![1.0, 1.0]);
    let mut solver = Solver::new(Method::rcg(), Control::default().with_maxiter(6), 2);
    let mut x = array![0.0, 0.0];
    solver.run(&obj, &mut x).unwrap();
    assert!(
        !obj.outside.load(std::sync::atomic::Ordering::Relaxed),
        "the objective was read outside its box"
    );
    assert!(
        (x[0] - 1.0).abs() < 1e-12 && (x[1] + 1.0).abs() < 1e-12,
        "{x}"
    );
}

/// A step whose best decrease misses the energy noise is kept at its
/// start and reported as the noise floor, as every other method does.
#[test]
fn rcg_reports_the_energy_noise_floor() {
    let quad = Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]));
    let control = Control::default()
        .with_maxiter(20)
        .with_gtol(1e-8)
        .with_noise_floor(NoiseFloor::new(0.0).with_eps_f(1.0));
    let mut probe = Solver::new(Method::rcg(), control.clone(), 1);
    let mut x = array![0.5];
    assert_eq!(
        probe.step(&quad, &mut x).unwrap().status,
        Status::NoiseFloor
    );
    assert_eq!(x, array![0.5], "a noise-floor step keeps its start");
    let mut solver = Solver::new(Method::rcg(), control, 1);
    let mut x = array![0.5];
    let report = solver.run(&quad, &mut x).unwrap();
    assert_eq!(report.status, Status::NoiseFloor, "{report:?}");
    assert_eq!(solver.steps(), 1);
    assert_eq!(report.value, 0.125);
}

/// The Poincare ball divides a Euclidean gradient by `lambda^2`, with
/// `lambda = 2 / (1 - |x|^2)`. At `x = (0.5, 0)` one conversion of
/// `g = x` has norm `0.5 / lambda^2 = 0.0703` and two have `0.0099`. A
/// kept start converts once, so it stays above `gtol = 0.03` and reports
/// the noise floor, not the gradient tolerance.
#[test]
fn rcg_noise_floor_on_the_poincare_ball_converts_the_gradient_once() {
    let quad = Oracle::unbounded(2, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
    let control = Control::default()
        .with_maxiter(20)
        .with_gtol(0.03)
        .with_noise_floor(NoiseFloor::new(0.0).with_eps_f(1.0));
    let mut solver = Solver::new(Method::rcg(), control, 2);
    solver.set_manifold(ManifoldKind::PoincareBall);
    let mut x = array![0.5, 0.0];
    let report = solver.step(&quad, &mut x).unwrap();
    assert_eq!(x, array![0.5, 0.0], "a noise-floor step keeps its start");
    assert_eq!(report.status, Status::NoiseFloor, "{report:?}");
    let lambda = 2.0 / (1.0 - 0.25);
    let once = 0.5 / (lambda * lambda);
    assert!(
        (report.grad_norm - once).abs() <= 1e-12 * once,
        "grad_norm {} want {once}",
        report.grad_norm
    );
}

/// Records every point the objective is asked to read.
struct Seen {
    bounds: Bounds<f64>,
    target: Array1<f64>,
    seen: std::sync::Mutex<Vec<Array1<f64>>>,
}

impl Seen {
    fn new(lo: Array1<f64>, hi: Array1<f64>, target: Array1<f64>) -> Self {
        Self {
            bounds: Bounds::new(lo, hi, 0.0),
            target,
            seen: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn points(&self) -> Vec<Array1<f64>> {
        self.seen.lock().expect("seen").clone()
    }
}

impl Objective<f64> for Seen {
    fn dim(&self) -> usize {
        self.bounds.dims
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Seen {
    fn dim(&self) -> usize {
        self.bounds.dims
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for Seen {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.seen.lock().expect("seen").push(x.to_owned());
        let g = &x.to_owned() - &self.target;
        (0.5 * g.dot(&g), g)
    }
}

fn in_closed_box(lo: &Array1<f64>, hi: &Array1<f64>, x: &Array1<f64>) -> bool {
    x.iter().zip(lo.iter().zip(hi.iter())).all(|(v, (a, b))| {
        let slack = 1e-10 * (1.0 + v.abs());
        *v + slack >= *a && *v - slack <= *b
    })
}

/// A box that cuts the sphere. Every accepted trial is a retraction, so it
/// keeps unit length, and it lies in the box.
#[test]
fn bounded_sphere_iterates_stay_on_the_sphere() {
    let lo = array![-0.2, -2.0];
    let hi = array![0.2, 2.0];
    let obj = Seen::new(lo.clone(), hi.clone(), array![1.0, 0.0]);
    let mut solver = Solver::new(Method::rcg(), Control::default().with_maxiter(6), 2);
    solver.set_manifold(ManifoldKind::Sphere);
    let mut x = array![0.0, 1.0];
    solver.run(&obj, &mut x).unwrap();
    let points = obj.points();
    assert!(points.len() > 1, "the run never tried a step");
    for (i, p) in points.iter().chain(std::iter::once(&x)).enumerate() {
        let norm = p.dot(p).sqrt();
        assert!(
            (norm - 1.0).abs() <= 1e-10,
            "iterate {i} left the sphere: |x| = {norm}, x = {p}"
        );
        assert!(in_closed_box(&lo, &hi, p), "iterate {i} left the box: {p}");
    }
}

/// A box beside the unit circle holds no point of it. A step and a
/// zero-budget run both refuse the start before the objective reads a
/// point, and leave the caller's `x` as it was.
#[test]
fn a_start_with_no_retraction_in_the_box_is_refused() {
    for maxiter in [0, 6] {
        let obj = Seen::new(array![2.0, -1.0], array![3.0, 1.0], array![2.0, 0.0]);
        let control = Control::default().with_maxiter(maxiter);
        let mut solver = Solver::new(Method::rcg(), control, 2);
        solver.set_manifold(ManifoldKind::Sphere);
        let start = array![1.0, 0.0];
        let mut x = start.clone();
        let error = solver.run(&obj, &mut x).unwrap_err();
        assert!(
            matches!(error, Error::OffManifold { kind: "sphere" }),
            "maxiter {maxiter}: {error:?}"
        );
        assert_eq!(x, start, "maxiter {maxiter}");
        assert!(
            obj.points().is_empty(),
            "maxiter {maxiter}: {:?}",
            obj.points()
        );
    }
}

/// A box that cuts the Stiefel manifold. Every accepted trial stays
/// orthonormal: the columns of `X` obey `X^T X = I`.
#[test]
fn bounded_stiefel_iterates_stay_orthonormal() {
    let n = 4;
    let p = 2;
    let start = array![0.8, 0.6, 0.0, 0.0, -0.6, 0.8, 0.0, 0.0];
    let target = array![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let lo = Array1::from_elem(n * p, -0.95);
    let hi = Array1::from_elem(n * p, 0.95);
    let obj = Seen::new(lo.clone(), hi.clone(), target);
    let mut solver = Solver::new(Method::rcg(), Control::default().with_maxiter(6), n * p);
    solver.set_manifold(ManifoldKind::stiefel(n, p));
    let mut x = start;
    solver.run(&obj, &mut x).unwrap();
    let points = obj.points();
    assert!(points.len() > 1, "the run never tried a step");
    for (i, point) in points.iter().chain(std::iter::once(&x)).enumerate() {
        let mut defect: f64 = 0.0;
        for a in 0..p {
            for b in 0..p {
                let mut gram = 0.0;
                for row in 0..n {
                    gram += point[row + n * a] * point[row + n * b];
                }
                let want = if a == b { 1.0 } else { 0.0 };
                defect = defect.max((gram - want).abs());
            }
        }
        assert!(
            defect <= 1e-10,
            "iterate {i} left Stiefel: defect {defect}, x = {point}"
        );
        assert!(
            in_closed_box(&lo, &hi, point),
            "iterate {i} left the box: {point}"
        );
    }
}
