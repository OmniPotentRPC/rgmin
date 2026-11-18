//! An accepted L-BFGS step of about 1e-10 that leaves the value and the
//! gradient norm bitwise unchanged, and then keeps doing it.
//!
//! A caller with `maxiter == 0` keeps taking steps while the report is
//! still iterating. On e7e66b60 that report stays `Iterating`, so these
//! tests do not return. The stop waits `max(n, 2)` accepted steps: one
//! flat step is the whole story of a 1-D overshoot, and a wider problem
//! waits out its own dimension.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, EON_MAX_MOVE, Lbfgs, LbfgsStep, LineSearch, Method, Solver, Status};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const VALUE: f64 = 759_036.0;
const GRAD: f64 = 2.14983;
/// A trial farther than this from the current point is uphill, so a unit
/// Wolfe trial is refused and the search accepts inside the ball.
const FLAT: f64 = 2e-10;

/// The plateau a real fit sits on: each accepted step moves the value by
/// a few ulps and the gradient norm by about 1e-12 relative, so neither
/// repeats its bits while both stay inside the objective's resolution.
struct Drift {
    on: bool,
    step: AtomicUsize,
}

impl Drift {
    fn off() -> Self {
        Self {
            on: false,
            step: AtomicUsize::new(0),
        }
    }

    fn on() -> Self {
        Self {
            on: true,
            ..Self::off()
        }
    }

    fn advance(&self) {
        self.step.fetch_add(1, Ordering::Relaxed);
    }

    fn phase(&self) -> f64 {
        if self.on {
            (self.step.load(Ordering::Relaxed) % 3) as f64
        } else {
            0.0
        }
    }

    fn value(&self) -> f64 {
        let ulp = f64::from_bits(VALUE.to_bits() + 1) - VALUE;
        VALUE + self.phase() * ulp
    }

    fn grad(&self) -> f64 {
        GRAD * (1.0 + self.phase() * 1e-12)
    }
}

struct ConstantForce {
    dim: usize,
    anchor: Mutex<Array1<f64>>,
    bounds: Bounds<f64>,
    drift: Drift,
}

impl ConstantForce {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            anchor: Mutex::new(Array1::zeros(dim)),
            bounds: Bounds::new(
                Array1::from_elem(dim, -1e6),
                Array1::from_elem(dim, 1e6),
                0.0,
            ),
            drift: Drift::off(),
        }
    }

    fn drifting(dim: usize) -> Self {
        Self {
            drift: Drift::on(),
            ..Self::new(dim)
        }
    }

    fn set_anchor(&self, x: &Array1<f64>) {
        *self.anchor.lock().unwrap() = x.clone();
        self.drift.advance();
    }
}

impl Objective<f64> for ConstantForce {
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let anchor = self.anchor.lock().unwrap();
        let r = (&x - &*anchor).mapv(|v| v * v).sum().sqrt();
        if r > FLAT {
            VALUE + (r - FLAT)
        } else {
            self.drift.value()
        }
    }
}

impl Gradient<f64> for ConstantForce {
    fn dim(&self) -> usize {
        self.dim
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let anchor = self.anchor.lock().unwrap();
        let d = &x - &*anchor;
        let r = d.mapv(|v| v * v).sum().sqrt();
        if r > FLAT {
            return &d * (100.0 / r);
        }
        let mut g = Array1::zeros(self.dim);
        g[0] = -self.drift.grad();
        g
    }
}

impl DifferentiableObjective<f64> for ConstantForce {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

/// Two-dimensional Wolfe objective. Off the anchor the gradient is
/// perpendicular to the step, so strong curvature holds and the norm
/// does not change. That is the H2CO retry: the line search, not the cap.
struct WolfePlane {
    anchor: Mutex<Array1<f64>>,
    bounds: Bounds<f64>,
    calls: AtomicUsize,
    drift: Drift,
}

impl WolfePlane {
    fn new() -> Self {
        Self {
            anchor: Mutex::new(array![0.0, 0.0]),
            bounds: Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0),
            calls: AtomicUsize::new(0),
            drift: Drift::off(),
        }
    }

    fn drifting() -> Self {
        Self {
            drift: Drift::on(),
            ..Self::new()
        }
    }

    fn set_anchor(&self, x: &Array1<f64>) {
        *self.anchor.lock().unwrap() = x.clone();
        self.drift.advance();
    }
}

impl Objective<f64> for WolfePlane {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let anchor = self.anchor.lock().unwrap();
        let d = &x - &*anchor;
        let r = d.dot(&d).sqrt();
        if r > FLAT {
            VALUE + (r - FLAT)
        } else {
            self.drift.value()
        }
    }
}

impl Gradient<f64> for WolfePlane {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let anchor = self.anchor.lock().unwrap();
        let dx = x[0] - anchor[0];
        let dy = x[1] - anchor[1];
        let r = (dx * dx + dy * dy).sqrt();
        if r > FLAT {
            return array![100.0 * dx / r, 100.0 * dy / r];
        }
        let grad = self.drift.grad();
        if r == 0.0 {
            return array![-grad, 0.0];
        }
        array![-grad * dy / r, grad * dx / r]
    }
}

impl DifferentiableObjective<f64> for WolfePlane {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        (self.eval(x), self.grad(x))
    }
}

fn session(dim: usize, maxiter: usize, wolfe: bool) -> Solver {
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control::default()
            .with_maxiter(maxiter)
            .with_gtol(1e-2)
            .with_istep(1.0)
            // A negative cap is the hyperparameter session: no Euclidean
            // limit. The default step stays the capped eOn move unless
            // this test opts into Wolfe.
            .with_maxmove(-1.0),
        dim,
    );
    if wolfe {
        solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
        solver.set_atom_maxmove(0.0);
        solver.set_linesearch(LineSearch::wolfe(1e-4, 0.9, 80));
    } else {
        solver.set_lbfgs_capped(1e-10).unwrap();
        solver.set_atom_maxmove(1e-10);
    }
    solver
}

fn assert_about_1e_10(step: f64) {
    assert!(
        (step - 1e-10).abs() < 5e-10,
        "accepted step {step} is not about 1e-10"
    );
}

fn assert_frozen(value: f64, grad_norm: f64) {
    assert!(
        (value - VALUE).abs() <= 4.0 * f64::EPSILON * VALUE,
        "value {value}"
    );
    assert!(
        (grad_norm - GRAD).abs() <= GRAD * f64::EPSILON.sqrt(),
        "grad {grad_norm}"
    );
    assert!(grad_norm >= 1e-2, "grad {grad_norm} is above the tolerance");
}

fn step_length(before: &Array1<f64>, after: &Array1<f64>) -> f64 {
    (after - before).mapv(|v| v * v).sum().sqrt()
}

/// A run that has not stopped by now never will: the stop waits
/// `max(n, 2)` steps.
const GUARD: usize = 200;

fn drive_capped(dim: usize, maxiter: usize) -> (Status, usize, f64, f64) {
    drive_capped_on(ConstantForce::new(dim), maxiter)
}

fn drive_capped_on(objective: ConstantForce, maxiter: usize) -> (Status, usize, f64, f64) {
    let dim = objective.dim;
    let mut solver = session(dim, maxiter, false);
    let mut x = Array1::zeros(dim);
    let mut accepted = 0usize;
    loop {
        let before = x.clone();
        let report = solver.step(&objective, &mut x).unwrap();
        accepted += 1;
        assert_about_1e_10(step_length(&before, &x));
        assert_frozen(report.value, report.grad_norm);
        objective.set_anchor(&x);
        if maxiter > 0 {
            assert_eq!(report.status, Status::Iterating, "{report:?}");
            if accepted == maxiter {
                return (report.status, accepted, report.value, report.grad_norm);
            }
        } else if report.status != Status::Iterating || accepted == GUARD {
            return (report.status, accepted, report.value, report.grad_norm);
        }
    }
}

fn drive_wolfe() -> (Status, usize) {
    drive_wolfe_on(WolfePlane::new())
}

fn drive_wolfe_on(objective: WolfePlane) -> (Status, usize) {
    let mut solver = session(2, 0, true);
    let mut x = array![0.0, 0.0];
    let mut accepted = 0usize;
    loop {
        let before = x.clone();
        let report = solver.step(&objective, &mut x).unwrap();
        accepted += 1;
        assert_about_1e_10(step_length(&before, &x));
        assert_frozen(report.value, report.grad_norm);
        objective.set_anchor(&x);
        if accepted == GUARD {
            return (report.status, accepted);
        }
        if report.status != Status::Iterating {
            assert!(
                objective.calls.load(Ordering::Relaxed) < 2_000,
                "evaluations {}",
                objective.calls.load(Ordering::Relaxed)
            );
            return (report.status, accepted);
        }
    }
}

#[test]
fn the_default_step_stays_the_capped_eon_move() {
    assert_eq!(Lbfgs::default().step, LbfgsStep::capped(EON_MAX_MOVE));
    assert_eq!(
        Solver::new(Method::lbfgs(), Control::default(), 2).lbfgs_step(),
        LbfgsStep::capped(EON_MAX_MOVE)
    );
}

#[test]
fn one_dimensional_unlimited_fit_stops_after_two_repeated_steps() {
    let (status, accepted, _, grad) = drive_capped(1, 0);
    assert_eq!(status, Status::StepTolerance);
    assert_eq!(accepted, 2);
    assert!(grad >= 1e-2);
}

#[test]
fn four_dimensional_unlimited_fit_stops_after_its_period() {
    let (status, accepted, _, grad) = drive_capped(4, 0);
    assert_eq!(status, Status::StepTolerance);
    assert_eq!(accepted, 4);
    assert!(grad >= 1e-2);
}

#[test]
fn wolfe_plane_stops_on_the_same_period() {
    let (status, accepted) = drive_wolfe();
    assert_eq!(status, Status::StepTolerance);
    assert_eq!(accepted, 2);
}

#[test]
fn a_positive_budget_walks_past_the_one_dimensional_period() {
    let (status, accepted, _, _) = drive_capped(1, 4);
    assert_eq!(status, Status::Iterating);
    assert_eq!(accepted, 4);
}

/// `f = 759036 + 1/2 sum (i + 1) x_i^2`, an ordinary bowl that converges.
struct Bowl {
    dim: usize,
    bounds: Bounds<f64>,
}

impl Bowl {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            bounds: Bounds::new(
                Array1::from_elem(dim, -1e6),
                Array1::from_elem(dim, 1e6),
                0.0,
            ),
        }
    }
}

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        VALUE
            + 0.5
                * x.iter()
                    .enumerate()
                    .map(|(i, v)| (i + 1) as f64 * v * v)
                    .sum::<f64>()
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        self.dim
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter(x.iter().enumerate().map(|(i, v)| (i + 1) as f64 * v))
    }
}

impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

/// The one-shot drivers keep reading a zero budget as zero iterations:
/// the plateau stop is for a caller's own step loop, not a new budget.
#[test]
fn a_zero_budget_one_shot_still_takes_no_step() {
    let control = Control::default().with_maxiter(0).with_gtol(1e-8);
    let x0 = array![1.0, -1.0, 0.5];
    let report = rgmin::minimize_lbfgs(
        &Bowl::new(3),
        x0.clone(),
        &control,
        LineSearch::default(),
        5,
    )
    .unwrap();
    assert_eq!(report.steps, 0);
    assert_eq!(report.status, Status::Budget);
    assert_eq!(report.coords, x0);

    let mut solver = Solver::new(Method::lbfgs(), control, 3);
    let mut x = x0.clone();
    solver.run(&Bowl::new(3), &mut x).unwrap();
    assert_eq!(x, x0);
    assert_eq!(solver.steps(), 0);
}

/// The real plateau does not repeat bits. A value inside the roundoff
/// window and a gradient norm inside `sqrt(eps)` of the step that opened
/// the run are the same plateau.
#[test]
fn a_drifting_plateau_stops_on_the_same_period() {
    let (status, accepted, _, grad) = drive_capped_on(ConstantForce::drifting(3), 0);
    assert_eq!(status, Status::StepTolerance);
    assert_eq!(accepted, 3);
    assert!(grad >= 1e-2);
}

#[test]
fn a_drifting_wolfe_plane_stops_on_the_same_period() {
    let (status, accepted) = drive_wolfe_on(WolfePlane::drifting());
    assert_eq!(status, Status::StepTolerance);
    assert_eq!(accepted, 2);
}

type ValueAndGradient = fn(ArrayView1<f64>) -> (f64, Array1<f64>);

/// An objective from one `(f, g)` function, for the convergence guards.
struct Smooth {
    dim: usize,
    bounds: Bounds<f64>,
    f: ValueAndGradient,
}

impl Smooth {
    fn new(dim: usize, f: ValueAndGradient) -> Self {
        Self {
            dim,
            bounds: Bounds::new(
                Array1::from_elem(dim, -1e6),
                Array1::from_elem(dim, 1e6),
                0.0,
            ),
            f,
        }
    }
}

impl Objective<f64> for Smooth {
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        (self.f)(x).0
    }
}

impl Gradient<f64> for Smooth {
    fn dim(&self) -> usize {
        self.dim
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        (self.f)(x).1
    }
}

impl DifferentiableObjective<f64> for Smooth {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.f)(x)
    }
}

/// Curvatures `1 .. 1e5` on six coordinates, offset to the plateau value.
fn ill_conditioned(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let h = |i: usize| 10f64.powf(i as f64);
    let g = Array1::from_iter(x.iter().enumerate().map(|(i, v)| h(i) * v));
    (VALUE + 0.5 * x.dot(&g), g)
}

/// Rosenbrock's curved valley, offset to the plateau value.
fn valley(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (a, b) = (x[0], x[1]);
    (
        VALUE + (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2),
        array![
            -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
            200.0 * (b - a * a)
        ],
    )
}

/// Late steps on these change the value by less than its resolution
/// while the gradient norm still falls, so neither is the plateau.
#[test]
fn unlimited_fits_on_a_flat_valley_or_an_ill_conditioned_bowl_still_converge() {
    let cases: [(ValueAndGradient, Array1<f64>, f64); 2] = [
        (ill_conditioned, Array1::from_elem(6, 1.0), 1e-4),
        (valley, array![-1.2, 1.0], 1e-5),
    ];
    for (f, x0, gtol) in cases {
        for wolfe in [false, true] {
            for window in [None, Some(1e-9)] {
                let objective = Smooth::new(x0.len(), f);
                let mut solver = Solver::new(
                    Method::lbfgs(),
                    Control::default().with_maxiter(0).with_gtol(gtol),
                    x0.len(),
                );
                if wolfe {
                    solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
                    solver.set_linesearch(LineSearch::wolfe(1e-4, 0.9, 80));
                }
                if let Some(relative) = window {
                    assert!(solver.set_objective_roundoff(relative));
                }
                let mut x = x0.clone();
                let mut report = solver.step(&objective, &mut x).unwrap();
                for _ in 0..1_000 {
                    if report.status != Status::Iterating || report.grad_norm < gtol {
                        break;
                    }
                    report = solver.step(&objective, &mut x).unwrap();
                }
                assert!(
                    report.grad_norm < gtol && report.status != Status::StepTolerance,
                    "wolfe {wolfe} window {window:?}: {report:?}"
                );
            }
        }
    }
}

#[cfg(feature = "capi")]
mod capi {
    use super::{GRAD, VALUE};
    use dlpk::sys::DLManagedTensorVersioned;
    use rgmin::ffi::*;
    use std::os::raw::c_void;

    const DIM: usize = 3;

    unsafe fn first(t: *const DLManagedTensorVersioned) -> *mut f64 {
        let dl = unsafe { &(*t).dl_tensor };
        unsafe { (dl.data as *mut u8).add(dl.byte_offset as usize) as *mut f64 }
    }

    /// A constant value and force: every capped step moves `x` and
    /// changes nothing the objective reports.
    unsafe extern "C" fn plateau(
        _user: *mut c_void,
        _x: *const DLManagedTensorVersioned,
        value: *mut f64,
        g: *mut DLManagedTensorVersioned,
    ) -> rgmin_status_t {
        unsafe {
            *value = VALUE;
            let g = first(g);
            *g = -GRAD;
            for i in 1..DIM {
                *g.add(i) = 0.0;
            }
        }
        rgmin_status_t::RGMIN_SUCCESS
    }

    /// The stop codes a fused C step loop reads, under the energy
    /// acceptance `rgmin_solver_create` opens with.
    fn stops(maxiter: usize, steps: usize) -> Vec<rgmin_stop_t> {
        let control = rgmin_control_t {
            maxiter,
            gtol: 1e-2,
            istep: 1.0,
            memory: 8,
            maxmove: 0.0,
            noise_eps: 0.0,
            noise_c: 0.0,
            require_convergence: 0,
        };
        let mut x = [0.0; DIM];
        let mut seen = Vec::new();
        unsafe {
            let session = rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &control, DIM);
            assert!(!session.is_null());
            for _ in 0..steps {
                let mut out = rgmin_report_t {
                    value: 0.0,
                    steps: 0,
                    grad_norm: 0.0,
                    converged: 0,
                    stop: rgmin_stop_t::RGMIN_STOP_TOLERANCE,
                };
                let before = x;
                let xt = rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), DIM);
                let st = rgmin_solver_step_fg(
                    session,
                    Some(plateau),
                    std::ptr::null_mut(),
                    xt,
                    &mut out,
                );
                rgmin_tensor_free(xt);
                assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
                assert_ne!(x, before);
                assert_eq!(out.value, VALUE);
                assert_eq!(out.grad_norm, GRAD);
                seen.push(out.stop);
                if out.stop != rgmin_stop_t::RGMIN_STOP_BUDGET {
                    break;
                }
            }
            rgmin_solver_free(session);
        }
        seen
    }

    #[test]
    fn a_c_step_loop_with_no_budget_reads_step_tolerance_after_n_steps() {
        use rgmin_stop_t::{RGMIN_STOP_BUDGET as GO, RGMIN_STOP_STEP_TOLERANCE as STALL};
        assert_eq!(stops(0, 100), [GO, GO, STALL]);
        assert_eq!(stops(5, 5), [GO; 5]);
    }
}
