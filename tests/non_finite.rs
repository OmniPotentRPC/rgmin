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
        for accept in [Accept::None, Accept::Energy, Accept::Nonmonotone] {
            let mut solver = Solver::new(method.clone(), CONTROL, 2);
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
/// runaway or NaN coordinate, and report a finite value. FIRE and
/// `Accept::None` take one step with no line search to shorten it, so they
/// may refuse with [`Error::Oracle`] instead. A session under the energy
/// test may report [`Error::AcceptStalled`] once it refuses the steepest
/// restart too.
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
                    match e {
                        Error::Oracle { .. } => what.contains("fire") || what.ends_with("None"),
                        Error::AcceptStalled { .. } => {
                            what.ends_with("Energy") || what.ends_with("Nonmonotone")
                        }
                        _ => false,
                    },
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
    for (token, cap) in [("fire", None), ("fire2", None), ("lbfgs", Some(0.1))] {
        let run = |refuse: bool| {
            let mut solver = Solver::new(Method::from_name(token).unwrap(), CONTROL, 3);
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
