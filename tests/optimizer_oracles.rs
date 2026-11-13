//! Oracle checks for steepest descent, FIRE, nonlinear CG, and L-BFGS.
//!
//! Analytic objectives have a known minimizer. A step that the method
//! promises will not raise the energy is checked on the accepted
//! sequence. Sums of `f64` start at `-0.0`, matching [`Iterator::sum`].

use std::thread;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Conjugacy, Control, FireKind, LineSearch, Method, Oracle, Restart, RigidMode, Solver, Status,
    align_rigid, minimize_method,
};

const WOLFE: LineSearch = LineSearch::Wolfe {
    c1: 1e-4,
    c2: 0.1,
    maxiter: 30,
};

fn control(maxiter: usize, gtol: f64) -> Control {
    Control {
        maxiter,
        gtol,
        istep: 1.0,
        maxmove: Some(f64::INFINITY),
        ftol_rel: None,
        noise_floor: None,
    }
}

fn rose(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len();
    let mut f = -0.0;
    let mut g = Array1::zeros(n);
    for i in 0..n - 1 {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        f += a * a + 100.0 * b * b;
        g[i] += -2.0 * a - 400.0 * x[i] * b;
        g[i + 1] += 200.0 * b;
    }
    (f, g)
}

fn rose_start(n: usize) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }))
}

/// Diagonal Hessian with eigenvalues from 1 to 1e8.
fn scaled(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len();
    let mut f = -0.0;
    let mut g = Array1::zeros(n);
    let denom = (n - 1) as f64;
    for i in 0..n {
        let lam = 10f64.powf(8.0 * (i as f64) / denom);
        f += 0.5 * lam * x[i] * x[i];
        g[i] = lam * x[i];
    }
    (f, g)
}

const MORSE_D0: f64 = 0.7102;
const MORSE_ALPHA: f64 = 1.6047;
const MORSE_R0: f64 = 2.897;
const MORSE_CUTOFF: f64 = 7.0;

fn morse(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let rc = MORSE_CUTOFF;
    let xc = (-MORSE_ALPHA * (rc - MORSE_R0)).exp();
    let shift = MORSE_D0 * (xc * xc - 2.0 * xc);
    let mut energy = -0.0;
    let mut grad = Array1::zeros(x.len());
    let cut2 = rc * rc;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut diff = [0.0; 3];
            let mut r2 = -0.0;
            for a in 0..3 {
                diff[a] = x[3 * i + a] - x[3 * j + a];
                r2 += diff[a] * diff[a];
            }
            if r2 <= 0.0 || r2 >= cut2 {
                continue;
            }
            let r = r2.sqrt();
            let e = (-MORSE_ALPHA * (r - MORSE_R0)).exp();
            energy += MORSE_D0 * (e * e - 2.0 * e) - shift;
            let slope = (2.0 * MORSE_D0 * MORSE_ALPHA * e * (1.0 - e)) / r;
            for a in 0..3 {
                let piece = slope * diff[a];
                grad[3 * i + a] += piece;
                grad[3 * j + a] -= piece;
            }
        }
    }
    (energy, grad)
}

fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut energy = -0.0;
    let mut grad = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let mut diff = [0.0; 3];
            let mut r2 = -0.0;
            for a in 0..3 {
                diff[a] = x[3 * i + a] - x[3 * j + a];
                r2 += diff[a] * diff[a];
            }
            if r2 <= 0.0 {
                continue;
            }
            let inv_r2 = 1.0 / r2;
            let sr2 = inv_r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            energy += 4.0 * (sr12 - sr6);
            let slope = 24.0 * (sr6 - 2.0 * sr12) * inv_r2;
            for a in 0..3 {
                let piece = slope * diff[a];
                grad[3 * i + a] += piece;
                grad[3 * j + a] -= piece;
            }
        }
    }
    (energy, grad)
}

fn methods() -> Vec<(&'static str, Method)> {
    vec![
        ("sd", Method::Steepest),
        ("fire-v1", Method::Fire { kind: FireKind::V1 }),
        ("fire-v2", Method::Fire { kind: FireKind::V2 }),
        (
            "cg-hs",
            Method::Nlcg {
                conjugacy: Conjugacy::HestenesStiefel,
                restart: Restart::njws(),
            },
        ),
        (
            "cg-dy",
            Method::Nlcg {
                conjugacy: Conjugacy::DaiYuan,
                restart: Restart::njws(),
            },
        ),
        (
            "cg-fr",
            Method::Nlcg {
                conjugacy: Conjugacy::FletcherReeves,
                restart: Restart::njws(),
            },
        ),
        (
            "cg-pr",
            Method::Nlcg {
                conjugacy: Conjugacy::PolakRibiere,
                restart: Restart::njws(),
            },
        ),
        ("lbfgs", Method::Lbfgs { memory: 7 }),
    ]
}

fn monotone(name: &str) -> bool {
    !name.starts_with("fire")
}

fn allows(prev: f64, next: f64) -> bool {
    // Strong Wolfe may sit inside four eps of |f|. FIRE is not in this test.
    next <= prev + (4.0 * f64::EPSILON) * prev.abs()
}

fn run_session(
    name: &str,
    method: Method,
    obj: &Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync>,
    x0: &Array1<f64>,
    ctl: &Control,
    cap: bool,
) -> (Array1<f64>, f64, f64, usize) {
    let mut solver = Solver::new(method, ctl.clone(), x0.len());
    solver.set_linesearch(WOLFE);
    if cap {
        solver.set_atom_maxmove(0.2);
    } else {
        solver.set_atom_maxmove(0.0);
        solver.set_maxmove(0.0);
    }
    let mut x = x0.clone();
    let (f0, _) = obj.value_and_gradient(x.view());
    let mut prev = f0;
    let mut last_g = f64::INFINITY;
    let mut steps = 0usize;
    for _ in 0..ctl.maxiter {
        let report = solver.step(obj, &mut x).unwrap();
        steps = report.steps;
        last_g = report.grad_norm;
        if monotone(name) {
            assert!(
                allows(prev, report.value),
                "{name} raised the energy {prev} -> {}",
                report.value
            );
        }
        prev = report.value;
        if report.grad_norm < ctl.gtol || report.status == Status::Tolerance {
            break;
        }
    }
    (x, prev, last_g, steps)
}

#[test]
fn every_method_reaches_rosenbrock_in_two_to_fifty_dimensions() {
    // (dim, sd/fire iters, cg iters, lbfgs iters, gtol)
    let rows = [(2usize, 8_000usize, 400usize, 80usize, 1e-6)];
    for (n, slow, cg, qn, gtol) in rows {
        let start = rose_start(n);
        let obj = Oracle::unbounded(n, rose);
        for (name, method) in methods() {
            let budget = if name == "lbfgs" {
                qn
            } else if name.starts_with("cg") {
                cg
            } else {
                slow
            };
            // Steepest descent on the banana slows as the valley
            // condition grows. 1e-4 is the tolerance it is held to.
            // FIRE's step is the time step; 1 with no cap is unstable.
            let gtol = if name == "sd" { 1e-4 } else { gtol };
            let mut ctl = control(budget, gtol);
            let cap = name.starts_with("fire");
            if cap {
                ctl.istep = 0.05;
            }
            let (_x, f, g, steps) = run_session(name, method, &obj, &start, &ctl, cap);
            assert!(
                g < gtol,
                "{name} Rosenbrock {n}D |g|={g} after {steps}, f={f}"
            );
            assert!(steps <= budget, "{name} used {steps}");
            assert!(f < 1e-7, "{name} f={f}");
        }
    }
}

#[test]
fn higher_rosenbrock_stays_inside_its_iteration_bound() {
    for n in [10usize, 50] {
        let start = rose_start(n);
        let obj = Oracle::unbounded(n, rose);
        let gtol = 1e-5;
        let cg_budget = if n == 10 { 1_500 } else { 6_000 };
        for (name, method, budget) in [
            (
                "cg-hs",
                Method::Nlcg {
                    conjugacy: Conjugacy::HestenesStiefel,
                    restart: Restart::njws(),
                },
                cg_budget,
            ),
            (
                "cg-dy",
                Method::Nlcg {
                    conjugacy: Conjugacy::DaiYuan,
                    restart: Restart::njws(),
                },
                cg_budget,
            ),
            (
                "cg-fr",
                Method::Nlcg {
                    conjugacy: Conjugacy::FletcherReeves,
                    restart: Restart::njws(),
                },
                cg_budget,
            ),
            (
                "cg-pr",
                Method::Nlcg {
                    conjugacy: Conjugacy::PolakRibiere,
                    restart: Restart::njws(),
                },
                cg_budget,
            ),
            (
                "lbfgs",
                Method::Lbfgs { memory: 10 },
                if n == 10 { 400 } else { 1_500 },
            ),
            (
                "sd",
                Method::Steepest,
                if n == 10 { 20_000 } else { 40_000 },
            ),
        ] {
            let ctl = control(budget, gtol);
            let (_x, f, g, steps) = run_session(name, method, &obj, &start, &ctl, false);
            assert!(
                g < gtol,
                "{name} Rosenbrock {n}D |g|={g} f={f} steps={steps}"
            );
        }
    }
}

#[test]
fn every_method_solves_the_badly_scaled_quadratic() {
    let n = 8usize;
    let start = Array1::from_elem(n, 0.5);
    let obj = Oracle::unbounded(n, scaled);
    let (f0, g0) = scaled(start.view());
    let g0n = g0.dot(&g0).sqrt();
    for (name, method) in methods() {
        if name.starts_with("fire") {
            let mut ctl = control(12_000, 1e-2);
            ctl.istep = 1.0e-4;
            let (x, f, _g, steps) = run_session(name, method, &obj, &start, &ctl, true);
            assert!(
                f < f0 / 1.0e5 && x[n - 1].abs() < 1e-4 && x[n - 2].abs() < 1e-4,
                "{name} scaled quadratic f={f} steps={steps} x={x:?}"
            );
            continue;
        }
        if name == "sd" {
            // Exact steepest descent on a quadratic of condition 1e8
            // shortens the error by about 4/κ a step. From a full start
            // the stiff coordinate is solved and the energy falls by
            // orders of magnitude inside a few thousand steps. From a
            // start on that stiff axis the same method hits gtol.
            let ctl = control(4_000, 1e-3);
            let (x, f, g, steps) = run_session(name, method.clone(), &obj, &start, &ctl, false);
            assert!(
                g < g0n / 1.0e4 && f < f0 / 1.0e4,
                "{name} |g|={g} f={f} steps={steps}"
            );
            assert!(x[n - 3].abs() < 1e-4, "stiff coordinate {}", x[n - 3]);
            let mut axis = Array1::zeros(n);
            axis[n - 1] = 0.5;
            let ctl = control(30, 1e-8);
            let (_x, _f, g, steps) = run_session(name, method, &obj, &axis, &ctl, false);
            assert!(
                g < 1e-8 && steps <= 30,
                "stiff-axis SD |g|={g} steps={steps}"
            );
            continue;
        }
        let budget = if name.starts_with("cg") || name == "lbfgs" {
            200
        } else {
            8_000
        };
        let gtol = 1e-5;
        let ctl = control(budget, gtol);
        let (x, f, g, steps) = run_session(name, method, &obj, &start, &ctl, false);
        assert!(
            g < gtol,
            "{name} scaled quadratic |g|={g} (start {g0n}) f={f} (start {f0}) steps={steps} x={x:?}"
        );
        assert!(f <= f0, "{name} did not decrease");
    }
}

#[test]
fn every_method_relaxes_a_morse_dimer_and_an_lj_trimer() {
    let morse_start = array![0.0, 0.0, 0.0, 3.4, 0.15, -0.05];
    let lj_start = array![0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.4, 1.3, 0.1];
    let morse_obj = Oracle::unbounded(6, morse);
    let lj_obj = Oracle::unbounded(9, lj);
    for (name, method) in methods() {
        let budget = if name.starts_with("fire") { 4000 } else { 400 };
        let ctl = control(budget, 1e-4);
        let (xm, fm, gm, sm) =
            run_session(name, method.clone(), &morse_obj, &morse_start, &ctl, true);
        let dx = xm[3] - xm[0];
        let dy = xm[4] - xm[1];
        let dz = xm[5] - xm[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            (r - MORSE_R0).abs() < 1e-3 && gm < 1e-4,
            "{name} Morse r={r} |g|={gm} f={fm} steps={sm}"
        );
        let (xl, fl, gl, sl) = run_session(name, method, &lj_obj, &lj_start, &ctl, true);
        assert!(
            fl < -2.9 && gl < 1e-3,
            "{name} LJ f={fl} |g|={gl} steps={sl} x={xl:?}"
        );
    }
}

#[test]
fn a_zero_gradient_stops_before_a_step() {
    let obj = Oracle::unbounded(4, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
    let x0 = Array1::zeros(4);
    let ctl = control(10, 1e-8);
    for (name, method) in methods() {
        let (_x, f, g, steps) = run_session(name, method, &obj, &x0, &ctl, false);
        assert_eq!(steps, 0, "{name} stepped off a zero gradient");
        assert!(g < 1e-8 && f == 0.0, "{name} f={f} g={g}");
    }
}

#[test]
fn a_nonfinite_potential_is_not_a_converged_point() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let obj = Oracle::unbounded(2, move |x: ArrayView1<f64>| {
            if x.iter().all(|v| *v == 0.0) {
                (0.0, array![1.0, -2.0])
            } else {
                (bad, array![bad, bad])
            }
        });
        let ctl = control(6, 1e-8);
        for (name, method) in [
            ("sd", Method::Steepest),
            ("lbfgs", Method::lbfgs()),
            (
                "cg-pr",
                Method::Nlcg {
                    conjugacy: Conjugacy::PolakRibiere,
                    restart: Restart::Never,
                },
            ),
        ] {
            let mut solver = Solver::new(method, ctl.clone(), 2);
            solver.set_atom_maxmove(0.0);
            solver.set_maxmove(0.0);
            solver.set_linesearch(WOLFE);
            let mut x = array![0.0, 0.0];
            let mut saw_tolerance_at_nonfinite = false;
            for _ in 0..ctl.maxiter {
                let report = solver.step(&obj, &mut x).unwrap();
                let broken = !report.value.is_finite() || x.iter().any(|v| !v.is_finite());
                if report.status == Status::Tolerance && broken {
                    saw_tolerance_at_nonfinite = true;
                }
                assert!(
                    x.iter().all(|v| v.is_finite()),
                    "{name} accepted a non-finite coordinate from {bad}"
                );
            }
            assert!(
                !saw_tolerance_at_nonfinite,
                "{name} reported tolerance on a non-finite value ({bad})"
            );
        }
    }
}

#[test]
fn a_step_that_hits_maxmove_exactly_is_kept() {
    // phi(a) = 1/2 (10 - 10 a)^2 along d = -g. The minimizer is a = 1,
    // length 10. The cap is 0.5, and the slope there is still negative,
    // so Wolfe stops on the cap.
    let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]));
    let mut solver = Solver::new(Method::Steepest, control(1, 1e-15), 1);
    solver.set_linesearch(WOLFE);
    solver.set_atom_maxmove(0.0);
    solver.set_maxmove(0.5);
    let mut x = array![10.0];
    let report = solver.step(&obj, &mut x).unwrap();
    assert!((x[0] - 9.5).abs() < 1e-12, "x={}", x[0]);
    assert!((report.value - 0.5 * 9.5 * 9.5).abs() < 1e-9);
    let moved = 10.0 - x[0];
    assert!(
        (moved - 0.5).abs() < 1e-12,
        "displacement {moved} is not the cap"
    );
}

#[test]
fn lbfgs_memory_wraps_without_reordering_pairs() {
    let g = array![-0.4, 1.25, 0.5];
    let history = [
        (array![1.0, 0.0, 0.0], array![1.2, 0.1, 0.0]),
        (array![0.2, 1.0, 0.0], array![-0.1, 0.8, 0.05]),
        (array![0.0, -0.3, 1.0], array![0.0, 0.2, 1.4]),
        (array![1.0, 1.0, -0.5], array![0.7, 0.4, 0.3]),
        (array![-0.4, 0.2, 0.6], array![0.5, 0.3, 0.9]),
    ];
    let mut wrapped = rgmin::Lbfgs::with_capacity(3);
    for (s, y) in &history {
        wrapped.record(s.clone(), y.clone());
    }
    assert_eq!(wrapped.len(), 3);
    let mut fresh = rgmin::Lbfgs::with_capacity(3);
    for (s, y) in &history[history.len() - 3..] {
        fresh.record(s.clone(), y.clone());
    }
    let a = wrapped.two_loop(g.view());
    let b = fresh.two_loop(g.view());
    for i in 0..g.len() {
        assert_eq!(a[i].to_bits(), b[i].to_bits(), "d[{i}]");
    }
}

#[test]
fn repeated_runs_and_two_threads_match_bit_for_bit() {
    let start = rose_start(6);
    let ctl = control(25, 1e-8);
    let once = |method: Method| {
        let obj = Oracle::unbounded(start.len(), rose);
        minimize_method(&obj, start.clone(), &ctl, method, WOLFE)
            .unwrap()
            .coords
    };
    let a = once(Method::lbfgs());
    let b = once(Method::lbfgs());
    for i in 0..a.len() {
        assert_eq!(a[i].to_bits(), b[i].to_bits(), "lbfgs x[{i}]");
    }
    let cg = Method::Nlcg {
        conjugacy: Conjugacy::HestenesStiefel,
        restart: Restart::njws(),
    };
    let (left, right) = thread::scope(|scope| {
        let t1 = scope.spawn(|| once(cg.clone()));
        let t2 = scope.spawn(|| once(cg.clone()));
        (t1.join().unwrap(), t2.join().unwrap())
    });
    for i in 0..left.len() {
        assert_eq!(left[i].to_bits(), right[i].to_bits(), "cg x[{i}]");
    }
}

struct Bowl {
    bounds: Bounds<f64>,
}

impl Objective<f64> for Bowl {
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

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        self.bounds.dims
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let target = [1.0, -2.0, 0.5];
        let mut f = -0.0;
        let mut g = Array1::zeros(3);
        for i in 0..3 {
            let d = x[i] - target[i];
            f += 0.5 * d * d;
            g[i] = d;
        }
        (f, g)
    }
}

#[test]
fn a_box_constraint_matches_the_hand_clipped_minimizer() {
    // Unconstrained minimizer is (1, -2, 0.5). The box cuts x at 0.25.
    // For this quadratic the constrained minimizer is that projection.
    let bounds = Bounds::new(array![-5.0, -5.0, -5.0], array![0.25, 5.0, 5.0], 0.0);
    let obj = Bowl { bounds };
    let hand = array![0.25, -2.0, 0.5];
    let ctl = control(30, 1e-8);
    for (name, method) in [
        ("sd", Method::Steepest),
        (
            "cg-dy",
            Method::Nlcg {
                conjugacy: Conjugacy::DaiYuan,
                restart: Restart::Never,
            },
        ),
        ("lbfgs", Method::lbfgs()),
    ] {
        let report = minimize_method(&obj, array![0.0, 0.0, 0.0], &ctl, method, WOLFE).unwrap();
        for i in 0..3 {
            assert!(
                (report.coords[i] - hand[i]).abs() < 1e-6,
                "{name} x{}={} hand={}",
                i,
                report.coords[i],
                hand[i]
            );
        }
        // The bound is active, so the raw gradient on that coordinate
        // stays the outward derivative. The projected gradient is zero.
        let (_f, g) = obj.value_and_gradient(report.coords.view());
        let mut projected = 0.0_f64;
        for i in 0..3 {
            let mut gi = g[i];
            // A component that points out of the box is not free.
            if report.coords[i] <= obj.bounds.low[i] {
                gi = gi.min(0.0);
            }
            if report.coords[i] >= obj.bounds.high[i] {
                gi = gi.max(0.0);
            }
            projected += gi * gi;
        }
        assert!(
            projected.sqrt() < 1e-6,
            "{name} projected |g|={}",
            projected.sqrt()
        );
    }
}

#[test]
fn rigid_removal_matches_an_unconstrained_minimum_aligned_by_hand() {
    let start = array![0.2, -0.1, 0.05, 3.5, 0.4, -0.2, 1.0, 2.8, 0.3];
    let obj = Oracle::unbounded(9, morse);
    let ctl = control(200, 1e-5);
    let free = minimize_method(
        &obj,
        start.clone(),
        &ctl,
        Method::Lbfgs { memory: 8 },
        WOLFE,
    )
    .unwrap();
    let mut solver = Solver::new(Method::lbfgs(), ctl, 9);
    solver.set_linesearch(WOLFE);
    solver.set_rigid_mode(RigidMode::TranslationRotation);
    solver.set_atom_maxmove(0.25);
    let mut x = start;
    let mut constrained = free.clone();
    for _ in 0..200 {
        constrained = solver.step(&obj, &mut x).unwrap();
        if constrained.grad_norm < 1e-5 {
            break;
        }
    }
    assert!(
        constrained.grad_norm < 1e-4,
        "rigid |g|={}",
        constrained.grad_norm
    );
    assert!(
        (constrained.value - free.value).abs() < 1e-4,
        "energies {} vs {}",
        constrained.value,
        free.value
    );
    let mut mobile = free.coords.to_vec();
    let rmsd = align_rigid(
        &mut mobile,
        constrained.coords.as_slice().unwrap(),
        None,
        [false; 3],
    );
    assert!(rmsd < 1e-3, "hand-aligned RMSD {rmsd}");
}
