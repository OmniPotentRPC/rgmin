//! Default per-atom step cap.
//!
//! The start is the seed-1 packing from `bench/atomistic` `lj38`, scaled
//! by one half. Without a cap the first quasi-Newton step leaves the
//! cluster. The default cap keeps it bound. An explicit cap replaces
//! the default.

use std::sync::OnceLock;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, FireKind, HessianObjective, LineSearch, Method, NewtonKind, Solver, Status,
    minimize_method, minimize_method_hess,
};

const N: usize = 38;
/// Final pairs of the capped minima measured in this file sit under 4.2.
/// The dissociated step leaves a pair at 454.
const BOUND_PAIR: f64 = 6.0;
const CAP: f64 = 0.2;

struct Lj38;

impl Objective<f64> for Lj38 {
    fn dim(&self) -> usize {
        3 * N
    }
    fn bounds(&self) -> &Bounds<f64> {
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| {
            Bounds::new(
                Array1::from_elem(3 * N, -1.0e3),
                Array1::from_elem(3 * N, 1.0e3),
                0.0,
            )
        })
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        lj(x).0
    }
}

impl Gradient<f64> for Lj38 {
    fn dim(&self) -> usize {
        3 * N
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        lj(x).1
    }
}

impl DifferentiableObjective<f64> for Lj38 {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        lj(x)
    }
}

fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mut energy = 0.0;
    let mut grad = Array1::zeros(x.len());
    for i in 0..N {
        for j in (i + 1)..N {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            let inv = 1.0 / r;
            let sr6 = inv.powi(6);
            let sr12 = sr6 * sr6;
            energy += 4.0 * (sr12 - sr6);
            let scale = (24.0 * inv) * (sr6 - 2.0 * sr12) * inv;
            grad[3 * i] += scale * dx;
            grad[3 * i + 1] += scale * dy;
            grad[3 * i + 2] += scale * dz;
            grad[3 * j] -= scale * dx;
            grad[3 * j + 1] -= scale * dy;
            grad[3 * j + 2] -= scale * dz;
        }
    }
    (energy, grad)
}

/// Sphere packing used by `bench/atomistic` `lj38`, then scaled by 1/2.
fn compressed_lj38() -> Array1<f64> {
    let radius = 2.4_f64;
    let mut state = 1_u64.wrapping_mul(7919).wrapping_add(17);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut uniform = || (next() >> 11) as f64 / (1u64 << 53) as f64;
    let mut pos = Vec::with_capacity(3 * N);
    while pos.len() < 3 * N {
        let p = [
            (2.0 * uniform() - 1.0) * radius,
            (2.0 * uniform() - 1.0) * radius,
            (2.0 * uniform() - 1.0) * radius,
        ];
        if p.iter().map(|v| v * v).sum::<f64>() > radius * radius {
            continue;
        }
        let close = pos.chunks(3).any(|q| {
            let d = (0..3)
                .map(|k| {
                    let delta: f64 = q[k] - p[k];
                    delta * delta
                })
                .sum::<f64>();
            d < 0.81
        });
        if !close {
            pos.extend_from_slice(&p);
        }
    }
    Array1::from_vec(pos).mapv(|v| v * 0.5)
}

fn max_pair(x: &Array1<f64>) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..N {
        for j in (i + 1)..N {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            m = m.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    m
}

fn max_atom_step(before: &Array1<f64>, after: &Array1<f64>) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..N {
        let dx = after[3 * i] - before[3 * i];
        let dy = after[3 * i + 1] - before[3 * i + 1];
        let dz = after[3 * i + 2] - before[3 * i + 2];
        m = m.max((dx * dx + dy * dy + dz * dz).sqrt());
    }
    m
}

fn session(method: Method, cap: Option<f64>) -> Solver {
    let control = Control {
        maxiter: 800,
        gtol: 1e-2,
        ..Control::default()
    };
    let mut solver = Solver::new(method, control, 3 * N);
    if let Some(value) = cap {
        solver.set_atom_maxmove(value);
    }
    solver
}

fn first_move(method: Method, cap: Option<f64>) -> f64 {
    let mut x = compressed_lj38();
    let before = x.clone();
    session(method, cap).step(&Lj38, &mut x).unwrap();
    max_atom_step(&before, &x)
}

fn relax(method: Method) -> (Status, f64, f64, f64) {
    let mut solver = session(method, None);
    let mut x = compressed_lj38();
    let before = x.clone();
    let mut last = solver.step(&Lj38, &mut x).unwrap();
    let mut peak_step = max_atom_step(&before, &x);
    for _ in 1..800 {
        if last.grad_norm < 1e-2 {
            break;
        }
        let before = x.clone();
        last = solver.step(&Lj38, &mut x).unwrap();
        peak_step = peak_step.max(max_atom_step(&before, &x));
    }
    (last.status, last.value, max_pair(&x), peak_step)
}

#[test]
fn default_cap_keeps_lj38_bound_for_lbfgs_and_bfgs() {
    let start = compressed_lj38();
    let start_pair = max_pair(&start);
    assert!(
        start_pair < BOUND_PAIR,
        "the compressed start is still one cluster, pair {start_pair}"
    );
    for method in [Method::lbfgs(), Method::Bfgs] {
        let (status, energy, pair, peak_step) = relax(method.clone());
        assert_eq!(status, Status::Tolerance, "{method:?}");
        assert!(energy < -160.0, "{method:?} energy {energy}");
        assert!(pair <= BOUND_PAIR, "{method:?} pair {pair}");
        assert!(
            peak_step <= CAP + 1e-9,
            "{method:?} peak atom step {peak_step}"
        );
    }
}

#[test]
fn explicit_clear_dissociates_and_a_larger_cap_overrides() {
    for method in [Method::lbfgs(), Method::Bfgs] {
        let cleared = first_move(method.clone(), Some(0.0));
        assert!(
            cleared > 1.0,
            "{method:?} cleared cap still moved only {cleared}"
        );
        let wider = first_move(method.clone(), Some(0.5));
        assert!(wider > CAP, "{method:?} wider cap stayed at {wider}");
        assert!(wider <= 0.5 + 1e-9, "{method:?} wider cap moved {wider}");
    }
}

#[test]
fn one_shot_minimizers_use_the_same_default() {
    let start = compressed_lj38();
    let control = Control {
        maxiter: 800,
        gtol: 1e-2,
        ..Control::default()
    };
    let linesearch = LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 20,
    };
    for method in [Method::lbfgs(), Method::Bfgs] {
        let report =
            minimize_method(&Lj38, start.clone(), &control, method.clone(), linesearch).unwrap();
        assert_eq!(report.status, Status::Tolerance, "{method:?}");
        assert!(report.value < -160.0, "{method:?} {}", report.value);
        assert!(max_pair(&report.coords) <= BOUND_PAIR, "{method:?}");
    }
    let open = Control {
        maxmove: Some(1000.0),
        ..control
    };
    let report = minimize_method(&Lj38, start, &open, Method::lbfgs(), linesearch).unwrap();
    assert!(
        max_pair(&report.coords) > BOUND_PAIR,
        "Euclidean maxmove 1000 left pair {}",
        max_pair(&report.coords)
    );
}

#[test]
fn explicit_euclidean_maxmove_still_limits_the_step() {
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 2,
            gtol: 1e-12,
            istep: 1.0,
            maxmove: Some(0.05),
            ..Control::default()
        },
        3,
    );
    solver.set_accept(rgmin::Accept::Step);
    let mut x = array![0.0, 0.0, 0.0];
    let obj = ConstantForce;
    solver.step(&obj, &mut x).unwrap();
    let step = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    assert!(step > 0.0, "the step vanished");
    assert!(step <= 0.05 + 1e-12, "euclidean cap ignored, step {step}");
}

struct ConstantForce;

impl Objective<f64> for ConstantForce {
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-10.0, -10.0, -10.0], array![10.0, 10.0, 10.0], 0.0))
    }
    fn eval(&self, _x: ArrayView1<f64>) -> f64 {
        1.0
    }
}

impl Gradient<f64> for ConstantForce {
    fn dim(&self) -> usize {
        3
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        array![0.4, 0.0, 0.0]
    }
}

impl DifferentiableObjective<f64> for ConstantForce {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn default_cap_bounds_the_first_step_of_every_cartesian_arm() {
    let methods = [
        Method::lbfgs(),
        Method::Bfgs,
        Method::Sr1,
        Method::Sr2,
        Method::diis(),
        Method::polak_ribiere(),
        Method::Fire { kind: FireKind::V1 },
        Method::Fire { kind: FireKind::V2 },
        Method::QuickMin,
        Method::Steepest,
        Method::adam(),
    ];
    for method in methods {
        let step = first_move(method.clone(), None);
        assert!(step <= CAP + 1e-9, "{method:?} default atom step {step}");
        assert!(step > 0.05, "{method:?} default atom step {step}");
    }
}

/// One atom, `f = ½‖x‖²`. The Newton step has length √3.
struct Bowl;

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-10.0, -10.0, -10.0], array![10.0, 10.0, 10.0], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * x.dot(&x)
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        3
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}

impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

impl HessianObjective for Bowl {
    fn hessian(&self, _x: ArrayView1<f64>) -> ndarray::Array2<f64> {
        ndarray::Array2::eye(3)
    }
}

fn bowl_move(method: Method, clear: bool) -> f64 {
    let mut solver = Solver::new(
        method,
        Control {
            maxiter: 2,
            gtol: 1e-14,
            istep: 1.0,
            ..Control::default()
        },
        3,
    );
    if clear {
        solver.set_atom_maxmove(0.0);
    }
    let mut x = array![1.0, 1.0, 1.0];
    let before = x.clone();
    solver.step_hess(&Bowl, &mut x).unwrap();
    ((&x - &before).dot(&(&x - &before))).sqrt()
}

#[test]
fn default_cap_clips_newton_and_dogleg() {
    for method in [
        Method::Newton {
            kind: NewtonKind::Shifted,
        },
        Method::Dogleg,
    ] {
        let capped = bowl_move(method.clone(), false);
        assert!(capped <= CAP + 1e-9, "{method:?} moved {capped}");
        assert!(capped > 0.05, "{method:?} moved {capped}");
        let open = bowl_move(method.clone(), true);
        assert!(open > 0.5, "{method:?} cleared cap moved {open}");
    }
    let control = Control {
        maxiter: 1,
        gtol: 1e-14,
        ..Control::default()
    };
    let capped = minimize_method_hess(
        &Bowl,
        array![1.0, 1.0, 1.0],
        &control,
        Method::Newton {
            kind: NewtonKind::Shifted,
        },
        LineSearch::default(),
    )
    .unwrap();
    let moved = capped.coords.dot(&capped.coords).sqrt();
    let start = 3.0_f64.sqrt();
    assert!(
        (start - moved) <= CAP + 1e-9,
        "one-shot Newton moved {}",
        start - moved
    );
    let open_control = Control {
        maxmove: Some(10.0),
        ..control
    };
    let open = minimize_method_hess(
        &Bowl,
        array![1.0, 1.0, 1.0],
        &open_control,
        Method::Newton {
            kind: NewtonKind::Shifted,
        },
        LineSearch::default(),
    )
    .unwrap();
    assert!(
        open.coords.dot(&open.coords).sqrt() < 1e-8,
        "Euclidean maxmove left {:?}",
        open.coords
    );
}

/// Two atoms, `f = ½‖x‖²`, started at `(1,1,1, 0,0,0)`.
/// The gradient sits on the first atom, so the Newton step has length √3
/// and a Euclidean ball of `0.2 * √2` still moves that atom past 0.2.
struct TwoAtomBowl;

impl Objective<f64> for TwoAtomBowl {
    fn dim(&self) -> usize {
        6
    }
    fn bounds(&self) -> &Bounds<f64> {
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(Array1::from_elem(6, -10.0), Array1::from_elem(6, 10.0), 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * x.dot(&x)
    }
}

impl Gradient<f64> for TwoAtomBowl {
    fn dim(&self) -> usize {
        6
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}

impl DifferentiableObjective<f64> for TwoAtomBowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn largest_atom_move(before: &Array1<f64>, after: &Array1<f64>) -> f64 {
    let mut peak = 0.0_f64;
    for i in 0..before.len() / 3 {
        let dx = after[3 * i] - before[3 * i];
        let dy = after[3 * i + 1] - before[3 * i + 1];
        let dz = after[3 * i + 2] - before[3 * i + 2];
        peak = peak.max((dx * dx + dy * dy + dz * dz).sqrt());
    }
    peak
}

fn one_shot_sr_atom_step(method: Method, maxmove: Option<f64>) -> f64 {
    let start = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let control = Control {
        maxiter: 1,
        gtol: 1e-14,
        istep: 1.0,
        maxmove,
        ..Control::default()
    };
    let report = minimize_method(
        &TwoAtomBowl,
        start.clone(),
        &control,
        method,
        LineSearch::default(),
    )
    .unwrap();
    largest_atom_move(&start, &report.coords)
}

#[test]
fn one_shot_sr_honours_the_default_atom_cap() {
    for method in [Method::Sr1, Method::Sr2] {
        let capped = one_shot_sr_atom_step(method.clone(), None);
        assert!(
            capped <= CAP + 1e-9,
            "{method:?} default atom step {capped}"
        );
        assert!(capped > 0.05, "{method:?} default atom step {capped}");
        let open = one_shot_sr_atom_step(method.clone(), Some(1000.0));
        assert!(
            open > CAP,
            "{method:?} Euclidean maxmove 1000 stayed at {open}"
        );
        let tight = one_shot_sr_atom_step(method.clone(), Some(0.05));
        assert!(
            tight <= 0.05 + 1e-9,
            "{method:?} Euclidean maxmove 0.05 moved {tight}"
        );
        assert!(tight > 0.0, "{method:?} tight cap vanished");
    }
}

#[cfg(feature = "capi")]
mod c_abi {
    use std::os::raw::c_void;

    use dlpk::sys::DLManagedTensorVersioned;
    use ndarray::Array1;
    use rgmin::ffi::{
        rgmin_control_t, rgmin_method_t, rgmin_minimize, rgmin_report_t, rgmin_solver_create,
        rgmin_solver_free, rgmin_solver_set_atom_maxmove, rgmin_solver_step, rgmin_status_t,
        rgmin_stop_t, rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
    };

    use super::{CAP, N, compressed_lj38, lj, max_atom_step};

    unsafe fn cpu_f64(t: *const DLManagedTensorVersioned) -> (*const f64, usize) {
        let dl = unsafe { &(*t).dl_tensor };
        let n = unsafe { *dl.shape as usize };
        let p = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
        (p, n)
    }

    fn copy_in(p: *const f64, n: usize) -> Array1<f64> {
        let mut buf = vec![0.0; n];
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = unsafe { *p.add(i) };
        }
        Array1::from_vec(buf)
    }

    unsafe extern "C" fn lj_eval(
        _user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        value_out: *mut f64,
    ) -> rgmin_status_t {
        let (p, n) = unsafe { cpu_f64(x) };
        assert_eq!(n, 3 * N);
        unsafe { *value_out = lj(copy_in(p, n).view()).0 };
        rgmin_status_t::RGMIN_SUCCESS
    }

    unsafe extern "C" fn lj_grad(
        _user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        g: *mut DLManagedTensorVersioned,
    ) -> rgmin_status_t {
        let (p, n) = unsafe { cpu_f64(x) };
        let (gp, gn) = unsafe { cpu_f64(g as *const _) };
        assert_eq!(n, gn);
        let grad = lj(copy_in(p, n).view()).1;
        for i in 0..n {
            unsafe { *(gp as *mut f64).add(i) = grad[i] };
        }
        rgmin_status_t::RGMIN_SUCCESS
    }

    fn control(maxmove: f64, maxiter: usize) -> rgmin_control_t {
        rgmin_control_t {
            maxiter,
            gtol: 1e-12,
            istep: 1.0,
            memory: 10,
            maxmove,
            noise_eps: 0.0,
            noise_c: 0.0,
        }
    }

    fn report() -> rgmin_report_t {
        rgmin_report_t {
            value: 0.0,
            steps: 0,
            grad_norm: 0.0,
            stop: rgmin_stop_t::RGMIN_STOP_TOLERANCE,
        }
    }

    fn session_step(method: rgmin_method_t, cap: Option<f64>) -> f64 {
        let mut x = compressed_lj38().to_vec();
        let before = Array1::from_vec(x.clone());
        let ctrl = control(0.0, 1);
        let session = unsafe { rgmin_solver_create(method, &ctrl, 3 * N) };
        assert!(!session.is_null());
        if let Some(value) = cap {
            unsafe { rgmin_solver_set_atom_maxmove(session, value) };
        }
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), x.len()) };
        let mut out = report();
        let status = unsafe {
            rgmin_solver_step(
                session,
                Some(lj_eval),
                Some(lj_grad),
                std::ptr::null_mut(),
                xt,
                &mut out,
            )
        };
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
        unsafe {
            rgmin_tensor_free(xt);
            rgmin_solver_free(session);
        }
        max_atom_step(&before, &Array1::from_vec(x))
    }

    fn one_shot(maxmove: f64) -> f64 {
        let mut x = compressed_lj38().to_vec();
        let before = Array1::from_vec(x.clone());
        let ctrl = control(maxmove, 1);
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), x.len()) };
        let mut out = report();
        let status = unsafe {
            rgmin_minimize(
                Some(lj_eval),
                Some(lj_grad),
                std::ptr::null_mut(),
                xt,
                &ctrl,
                rgmin_method_t::RGMIN_LBFGS,
                &mut out,
            )
        };
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
        unsafe { rgmin_tensor_free(xt) };
        max_atom_step(&before, &Array1::from_vec(x))
    }

    #[test]
    fn c_abi_default_caps_and_an_explicit_cap_overrides() {
        for method in [rgmin_method_t::RGMIN_LBFGS, rgmin_method_t::RGMIN_BFGS] {
            let capped = session_step(method, None);
            assert!(capped <= CAP + 1e-9, "{method:?} default moved {capped}");
            let cleared = session_step(method, Some(0.0));
            assert!(cleared > 1.0, "{method:?} cleared cap moved {cleared}");
            let wider = session_step(method, Some(0.5));
            assert!(wider > CAP, "{method:?} wider cap stayed at {wider}");
            assert!(wider <= 0.5 + 1e-9, "{method:?} wider cap moved {wider}");
        }
        let capped = one_shot(0.0);
        assert!(capped <= CAP + 1e-9, "one-shot default moved {capped}");
        let open = one_shot(1000.0);
        assert!(open > 1.0, "one-shot Euclidean maxmove moved {open}");
    }
}
