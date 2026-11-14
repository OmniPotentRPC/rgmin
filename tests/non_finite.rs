//! A non-finite value or gradient refuses a trial step; it never passes one.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::SeqCst};

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, Error, Lbfgs, LineSearch, Method, Oracle, Report, Solver};

/// `(calls, largest |x_i| probed)`, NaN counting as infinite.
#[derive(Default)]
struct Probes(Mutex<(usize, f64)>);

impl Probes {
    fn see(&self, x: ArrayView1<f64>) {
        let mut p = self.0.lock().unwrap();
        p.0 += 1;
        for v in x {
            p.1 = if v.is_finite() {
                p.1.max(v.abs())
            } else {
                f64::INFINITY
            };
        }
    }

    fn take(&self) -> (usize, f64) {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

/// `Σ (x_i - 3)²` answered only inside `|x_i| <= 2`, as an engine that
/// cannot converge past some geometry. Outside, the value is `value` and
/// each gradient component is `grad`, or the analytic one when `None`.
fn bowl(x: ArrayView1<f64>, value: f64, grad: Option<f64>) -> (f64, Array1<f64>) {
    let g = x.mapv(|v| 2.0 * (v - 3.0));
    if x.iter().all(|v| v.abs() <= 2.0) {
        (x.iter().map(|v| (v - 3.0) * (v - 3.0)).sum(), g)
    } else {
        (value, grad.map_or(g, |c| Array1::from_elem(x.len(), c)))
    }
}

const CONTROL: Control = Control {
    maxiter: 40,
    gtol: 1e-8,
    istep: 1.0,
    maxmove: None,
    ftol_rel: None,
    noise_floor: None,
};

/// Every method under every line search, then as a session under every
/// acceptance, each outcome handed to `check` with its probes.
fn every_path<F, C>(x0: &Array1<f64>, fg: F, check: C)
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Sync,
    C: Fn(String, rgmin::Result<Report>, (usize, f64)),
{
    let (c, beta, maxiter, c1, c2) = (0.25, 0.5, 30, 1e-4, 0.9);
    let searches = [
        LineSearch::default(),
        LineSearch::Backtracking { c, beta, maxiter },
        LineSearch::Goldstein { c, beta, maxiter },
        LineSearch::Wolfe { c1, c2, maxiter },
    ];
    let probes = Probes::default();
    let seen = |v: ArrayView1<f64>| {
        probes.see(v);
        fg(v)
    };
    let obj = Oracle::unbounded(2, &seen);
    for token in [
        "sd", "cg", "bb", "bfgs", "lbfgs", "sr1", "sr2", "fire", "fire2",
    ] {
        let method = Method::from_name(token).unwrap();
        for ls in searches {
            let out = rgmin::minimize_method(&obj, x0.clone(), &CONTROL, method.clone(), ls);
            check(format!("{token} {ls:?}"), out, probes.take());
        }
        for accept in [
            Accept::None,
            Accept::Step,
            Accept::Energy,
            Accept::Nonmonotone,
        ] {
            let mut solver = Solver::new(method.clone(), CONTROL.clone(), 2);
            solver.set_accept(accept);
            let mut x = x0.clone();
            let out =
                (0..CONTROL.maxiter).try_fold(None, |_, _| solver.step(&obj, &mut x).map(Some));
            let out = out.map(Option::unwrap);
            check(format!("{token} session {accept:?}"), out, probes.take());
        }
    }
}

/// From inside the answered box every path must stay inside it, probe no
/// runaway or NaN coordinate, and report a finite value. FIRE may refuse
/// with [`Error::Oracle`]. A refused Accept::Step trial returns the finite
/// measured start and caches its gradient.
fn refused_outside(value: f64, grad: Option<f64>) {
    let x0 = array![0.0, 0.0];
    let inside = |x: &Array1<f64>| x.iter().all(|v| v.abs() <= 2.0);
    let probes = Probes::default();
    let (f, x, _) = Lbfgs::default().minimize(x0.view(), 50, |v| {
        probes.see(v);
        Some(bowl(v, value, grad))
    });
    let far = probes.take().1;
    assert!(
        far < 8.0 && f.is_finite() && f < 18.0 && inside(&x),
        "L-BFGS {far} {f} {x}"
    );
    every_path(
        &x0,
        |v| bowl(v, value, grad),
        |what, out, (_, far)| {
            assert!(far < 100.0, "{what} probed {far}");
            match out {
                Ok(r) => assert!(r.value.is_finite() && inside(&r.coords), "{what}: {r:?}"),
                Err(e) => assert!(
                    matches!(e, Error::Oracle { .. }) && what.contains("fire"),
                    "{what}: {e}"
                ),
            }
        },
    );
}

/// One evaluation at `x0`, then [`Error::Oracle`] on every path.
fn refused_start(x0: &Array1<f64>, fg: impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Sync) {
    let probes = Probes::default();
    let (f, x, _, hit) = Lbfgs::default().minimize_recognized(
        x0.view(),
        50,
        |v| {
            probes.see(v);
            Some(fg(v))
        },
        |_, _, _| None,
    );
    assert_eq!((f, &x, hit, probes.take().0), (f64::INFINITY, x0, false, 1));
    let start = x0[0].abs();
    every_path(x0, fg, |what, out, probes| {
        assert!(matches!(out, Err(Error::Oracle { .. })), "{what}: {out:?}");
        assert_eq!(probes, (1, start), "{what}");
    });
}

#[test]
fn a_nan_trial_energy_is_refused() {
    refused_outside(f64::NAN, None);
    refused_outside(f64::NAN, Some(0.0));
    refused_outside(0.0, Some(f64::NAN));
}

#[test]
fn an_infinite_energy_is_refused() {
    refused_outside(f64::NEG_INFINITY, None);
    refused_outside(f64::INFINITY, None);
    refused_start(&array![2.5, 2.5], |v| bowl(v, f64::INFINITY, None));
}

#[test]
fn a_nan_start_gradient_sends_no_nan_coordinate() {
    refused_start(&array![1.0, 1.0], |v| {
        (1.0, Array1::from_elem(v.len(), f64::NAN))
    });
}

/// An engine whose answers fail partway through a relaxation, after two
/// good steps. Each refused step must leave the session as it was: a retry
/// probes the same trials, and once the engine answers again the next step
/// matches a session that never saw the refusals.
#[test]
fn a_refused_step_leaves_the_session_as_it_was() {
    let open = AtomicBool::new(true);
    let probed = Mutex::new(Vec::new());
    let fg = |x: ArrayView1<f64>| {
        probed.lock().unwrap().push(x.to_owned());
        let w = array![0.1, 0.2, 0.3];
        let f = (&w * &x.mapv(|v| (v - 1.0) * (v - 1.0))).sum();
        let f = if open.load(SeqCst) { f } else { f64::NAN };
        (f, 2.0 * &w * &x.mapv(|v| v - 1.0))
    };
    let obj = Oracle::unbounded(3, &fg);
    for (token, cap, geometry) in [
        ("fire", None, false),
        ("fire2", None, false),
        ("lbfgs", Some(0.1), true),
    ] {
        let run = |refuse: bool| {
            let mut solver = Solver::new(Method::from_name(token).unwrap(), CONTROL.clone(), 3);
            solver.set_geometry_step(geometry);
            if let Some(cap) = cap {
                solver.set_atom_maxmove(cap);
            }
            let mut x = Array1::zeros(3);
            for _ in 0..2 {
                solver.step(&obj, &mut x).unwrap();
            }
            open.store(!refuse, SeqCst);
            let start = x.clone();
            let mut tries = Vec::new();
            for _ in 0..2 * usize::from(refuse) {
                probed.lock().unwrap().clear();
                let out = solver.step(&obj, &mut x);
                assert!(matches!(out, Err(Error::Oracle { .. })), "{token}: {out:?}");
                assert_eq!(x, start, "{token}");
                tries.push(std::mem::take(&mut *probed.lock().unwrap()));
            }
            if refuse {
                assert!(
                    !tries[0].is_empty() && tries[0] == tries[1],
                    "{token}: {tries:?}"
                );
            }
            open.store(true, SeqCst);
            let r = solver.step(&obj, &mut x).unwrap();
            (r.value, r.coords)
        };
        assert_eq!(run(true), run(false), "{token}");
    }
}

/// A refusal on the very first step leaves nothing behind either. The
/// steep start makes the geometry L-BFGS session choose its stiff seed.
/// After the refusal the caller moves to a shallow point, where the next
/// step must match a new session's, stiffness choice included.
#[test]
fn a_refused_first_step_leaves_the_session_new() {
    let open = AtomicBool::new(true);
    let probed = Mutex::new(Vec::new());
    let x0 = Array1::<f64>::zeros(3);
    let fg = |x: ArrayView1<f64>| {
        probed.lock().unwrap().push(x.to_owned());
        let f = 50.0 * x.mapv(|v| (v - 1.0) * (v - 1.0)).sum();
        let answered = open.load(SeqCst) || x == x0;
        let f = if answered { f } else { f64::NAN };
        (f, x.mapv(|v| 100.0 * (v - 1.0)))
    };
    let obj = Oracle::unbounded(3, &fg);
    for (token, cap, geometry) in [
        ("fire", None, false),
        ("fire2", None, false),
        ("lbfgs", Some(0.1), true),
    ] {
        let run = |refuse: bool| {
            let mut solver = Solver::new(Method::from_name(token).unwrap(), CONTROL.clone(), 3);
            solver.set_geometry_step(geometry);
            if let Some(cap) = cap {
                solver.set_atom_maxmove(cap);
            }
            open.store(!refuse, SeqCst);
            let mut tries = Vec::new();
            for _ in 0..2 * usize::from(refuse) {
                let mut x = x0.clone();
                probed.lock().unwrap().clear();
                let out = solver.step(&obj, &mut x);
                assert!(matches!(out, Err(Error::Oracle { .. })), "{token}: {out:?}");
                assert_eq!(x, x0, "{token}");
                tries.push(std::mem::take(&mut *probed.lock().unwrap()));
            }
            if refuse {
                assert!(
                    tries[0].len() > 1 && tries[0] == tries[1],
                    "{token}: {tries:?}"
                );
            }
            open.store(true, SeqCst);
            let mut x = Array1::from_elem(3, 0.99);
            probed.lock().unwrap().clear();
            let r = solver.step(&obj, &mut x).unwrap();
            (r.value, r.coords, r.steps, probed.lock().unwrap().clone())
        };
        assert_eq!(run(true), run(false), "{token}");
    }
}

fn weighted_quadratic(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let weights = array![0.1, 0.2, 0.3];
    let displacement = x.mapv(|v| v - 1.0);
    (
        (&weights * &displacement.mapv(|v| v * v)).sum(),
        2.0 * &weights * &displacement,
    )
}

/// A refused direct Step reports the cached measured start. Its curvature
/// and next trial match a run without refusals, including proposal guards.
#[test]
fn a_refused_direct_step_keeps_its_measured_start_and_history() {
    let open = AtomicBool::new(true);
    let probed = Mutex::new(Vec::new());
    let fg = |x: ArrayView1<f64>| {
        probed.lock().unwrap().push(x.to_owned());
        let (value, gradient) = weighted_quadratic(x);
        (if open.load(SeqCst) { value } else { f64::NAN }, gradient)
    };
    let obj = Oracle::unbounded(3, &fg);
    for guarded in [false, true] {
        let run = |refuse: bool| {
            open.store(true, SeqCst);
            let mut solver = Solver::new(Method::lbfgs(), CONTROL.clone(), 3);
            solver.set_accept(Accept::Step);
            solver.set_lbfgs_neb_guards(guarded);
            let mut x = Array1::zeros(3);
            solver.step(&obj, &mut x).unwrap();
            let start_report = solver.step(&obj, &mut x).unwrap();
            let start = x.clone();
            let pairs = solver.pair_count();
            assert!(pairs > 0);
            let gradient = weighted_quadratic(start.view()).1;
            let direction = solver.search_direction(gradient.view()).unwrap();
            let mut trials = Vec::new();
            open.store(!refuse, SeqCst);
            for iteration in 0..2 * usize::from(refuse) {
                probed.lock().unwrap().clear();
                let report = solver.step(&obj, &mut x).unwrap();
                assert_eq!(x, start);
                assert_eq!(report.coords, start);
                assert_eq!(report.value, start_report.value);
                assert_eq!(report.grad_norm, start_report.grad_norm);
                assert_eq!(report.steps, start_report.steps + iteration + 1);
                assert_eq!(solver.pair_count(), pairs);
                assert_eq!(solver.search_direction(gradient.view()).unwrap(), direction);
                let seen = std::mem::take(&mut *probed.lock().unwrap());
                assert_eq!(seen.len(), 1, "cached start must not be evaluated again");
                assert!(seen[0].iter().all(|v| v.is_finite()));
                trials.push(seen[0].clone());
            }
            if refuse {
                assert_eq!(trials[0], trials[1]);
            }
            open.store(true, SeqCst);
            probed.lock().unwrap().clear();
            let report = solver.step(&obj, &mut x).unwrap();
            let seen = std::mem::take(&mut *probed.lock().unwrap());
            assert_eq!(seen.len(), 1);
            if refuse {
                assert_eq!(seen[0], trials[0]);
            }
            (
                report.value,
                report.coords,
                report.grad_norm,
                solver.pair_count(),
                seen,
            )
        };
        assert_eq!(run(true), run(false), "guards {guarded}");
    }
}

/// The finite start of a refused first Step is adopted and cached. A
/// caller move then records its measured secant, matching explicit history.
#[test]
fn a_refused_first_direct_step_adopts_only_its_finite_start() {
    let open = AtomicBool::new(false);
    let probed = Mutex::new(Vec::new());
    let start = Array1::<f64>::zeros(3);
    let fg = |x: ArrayView1<f64>| {
        probed.lock().unwrap().push(x.to_owned());
        let value = 50.0 * x.mapv(|v| (v - 1.0) * (v - 1.0)).sum();
        let answered = open.load(SeqCst) || x == start;
        (
            if answered { value } else { f64::NAN },
            x.mapv(|v| 100.0 * (v - 1.0)),
        )
    };
    let obj = Oracle::unbounded(3, &fg);
    let mut solver = Solver::new(Method::lbfgs(), CONTROL.clone(), 3);
    solver.set_accept(Accept::Step);
    let mut x = start.clone();
    let mut trials = Vec::new();
    for iteration in 0..2 {
        probed.lock().unwrap().clear();
        let report = solver.step(&obj, &mut x).unwrap();
        assert_eq!(x, start);
        assert_eq!(report.coords, start);
        assert_eq!(report.value, 150.0);
        assert!((report.grad_norm - 100.0 * 3.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!(report.steps, iteration + 1);
        assert_eq!(solver.pair_count(), 0);
        let seen = std::mem::take(&mut *probed.lock().unwrap());
        assert_eq!(seen.len(), if iteration == 0 { 2 } else { 1 });
        if iteration == 0 {
            assert_eq!(seen[0], start);
        }
        assert!(seen.last().unwrap().iter().all(|v| v.is_finite()));
        trials.push(seen.last().unwrap().clone());
    }
    assert_eq!(trials[0], trials[1]);
    open.store(true, SeqCst);
    x.fill(0.99);
    let moved = x.clone();
    let secant = &moved - &start;
    let gradient_change = moved.mapv(|v| 100.0 * (v - 1.0)) - start.mapv(|v| 100.0 * (v - 1.0));
    probed.lock().unwrap().clear();
    let report = solver.step(&obj, &mut x).unwrap();
    assert_eq!(probed.lock().unwrap().len(), 2);
    assert_eq!(report.steps, 3);
    assert_eq!(solver.pair_count(), 2);
    let mut reference = Solver::new(Method::lbfgs(), CONTROL.clone(), 3);
    reference.set_accept(Accept::Step);
    assert!(reference.push_pair(secant.view(), gradient_change.view()));
    let mut reference_x = moved;
    let expected = reference.step(&obj, &mut reference_x).unwrap();
    assert_eq!(x, reference_x);
    assert_eq!(report.value, expected.value);
    assert_eq!(report.grad_norm, expected.grad_norm);
    assert_eq!(solver.pair_count(), reference.pair_count());
}

/// A caller move has a measured secant even when the following Step is
/// refused. Adopting that start records the pair once and caches its answer.
#[test]
fn a_refused_trial_after_a_move_records_the_move_once() {
    let open = AtomicBool::new(true);
    let probed = Mutex::new(Vec::new());
    let moved = array![0.25, 0.38, 0.63];
    let fg = |x: ArrayView1<f64>| {
        probed.lock().unwrap().push(x.to_owned());
        let (value, gradient) = weighted_quadratic(x);
        let answered = open.load(SeqCst) || x == moved;
        (if answered { value } else { f64::NAN }, gradient)
    };
    let obj = Oracle::unbounded(3, &fg);
    let run = |refuse: bool| {
        open.store(true, SeqCst);
        let mut solver = Solver::new(Method::lbfgs(), CONTROL.clone(), 3);
        solver.set_accept(Accept::Step);
        let mut x = Array1::zeros(3);
        let previous = solver.step(&obj, &mut x).unwrap();
        let pairs = solver.pair_count();
        x.assign(&moved);
        let (value, gradient) = weighted_quadratic(moved.view());
        let mut trials = Vec::new();
        let mut retained_direction = None;
        open.store(!refuse, SeqCst);
        for iteration in 0..2 * usize::from(refuse) {
            probed.lock().unwrap().clear();
            let report = solver.step(&obj, &mut x).unwrap();
            assert_eq!(x, moved);
            assert_eq!(report.coords, moved);
            assert_eq!(report.value, value);
            assert!((report.grad_norm - gradient.dot(&gradient).sqrt()).abs() < 1e-14);
            assert_eq!(report.steps, previous.steps + iteration + 1);
            assert_eq!(solver.pair_count(), pairs + 1);
            let direction = solver.search_direction(gradient.view()).unwrap();
            if let Some(expected) = &retained_direction {
                assert_eq!(&direction, expected);
            }
            retained_direction = Some(direction);
            let seen = std::mem::take(&mut *probed.lock().unwrap());
            assert_eq!(seen.len(), if iteration == 0 { 2 } else { 1 });
            if iteration == 0 {
                assert_eq!(seen[0], moved);
            }
            assert!(seen.last().unwrap().iter().all(|v| v.is_finite()));
            trials.push(seen.last().unwrap().clone());
        }
        if refuse {
            assert_eq!(trials[0], trials[1]);
        }
        open.store(true, SeqCst);
        probed.lock().unwrap().clear();
        let report = solver.step(&obj, &mut x).unwrap();
        let seen = std::mem::take(&mut *probed.lock().unwrap());
        assert_eq!(seen.len(), if refuse { 1 } else { 2 });
        if refuse {
            assert_eq!(seen[0], trials[0]);
        } else {
            assert_eq!(seen[0], moved);
        }
        assert_eq!(report.steps, previous.steps + 2 * usize::from(refuse) + 1);
        (
            report.value,
            report.coords,
            report.grad_norm,
            solver.pair_count(),
            seen.last().unwrap().clone(),
        )
    };
    assert_eq!(run(true), run(false));
}
