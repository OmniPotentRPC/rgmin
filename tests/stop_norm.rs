//! Session stop norms: ASE `fmax` / eOn `max_atom` and the infinity norm.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, FireKind, LineSearch, Method, Oracle, Solver, StopNorm};

fn lj_cluster(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let d: [f64; 3] = std::array::from_fn(|a| x[3 * i + a] - x[3 * j + a]);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let s6 = r2.powi(-3);
            e += 4.0 * (s6 * s6 - s6);
            let du_over_r = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
            for a in 0..3 {
                g[3 * i + a] += du_over_r * d[a];
                g[3 * j + a] -= du_over_r * d[a];
            }
        }
    }
    (e, g)
}

/// Seven atoms on a jittered 1.1 grid: far from a minimum, no close pairs.
fn lj7_start() -> Array1<f64> {
    let mut x = Vec::new();
    for k in 0..7 {
        let p = [(k % 2) as f64, ((k / 2) % 2) as f64, (k / 4) as f64];
        for (a, v) in p.iter().enumerate() {
            x.push(1.1 * v + 0.03 * ((3 * k + a) as f64).sin());
        }
    }
    Array1::from(x)
}

fn lbfgs() -> Method {
    Method::Lbfgs { memory: 20 }
}

fn fire() -> Method {
    Method::Fire { kind: FireKind::V1 }
}

fn session(method: Method, gtol: f64, norm: Option<StopNorm>) -> Solver {
    let control = Control {
        maxiter: 2000,
        gtol,
        istep: 0.1,
        maxmove: None,
        ..Default::default()
    };
    let mut s = Solver::new(method, control, 21);
    s.set_atom_maxmove(0.2);
    if let Some(norm) = norm {
        s.set_stop_norm(norm);
    }
    s
}

/// Steps until the report clears `gtol`. Returns the coordinates after
/// every report and the oracle calls spent.
fn relax(mut s: Solver, gtol: f64) -> (Vec<Array1<f64>>, usize) {
    let calls = AtomicUsize::new(0);
    let obj = Oracle::unbounded(21, |x: ArrayView1<f64>| {
        calls.fetch_add(1, Ordering::Relaxed);
        lj_cluster(x)
    });
    let mut x = lj7_start();
    let mut path = Vec::new();
    for _ in 0..2000 {
        let rep = s.step(&obj, &mut x).unwrap();
        path.push(rep.coords.clone());
        if rep.grad_norm < gtol {
            return (path, calls.load(Ordering::Relaxed));
        }
    }
    panic!("no convergence in 2000 steps");
}

#[test]
fn stop_norms_match_the_reference_definitions() {
    let g = array![3.0, 4.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, 2.0];
    assert_eq!(StopNorm::Two.of(g.view()), 33.0_f64.sqrt());
    assert_eq!(StopNorm::MaxAtom.of(g.view()), 5.0);
    assert_eq!(StopNorm::Inf.of(g.view()), 4.0);
    assert_eq!(
        StopNorm::Rms.of(g.view()),
        33.0_f64.sqrt() / 10.0_f64.sqrt()
    );
    assert_eq!(StopNorm::Rms.of(Array1::zeros(0).view()), 0.0);
    assert_eq!(StopNorm::default(), StopNorm::Two);
    let bad = array![0.0, f64::NAN, 0.0, 9.0, 0.0, 0.0];
    assert_eq!(StopNorm::MaxAtom.of(bad.view()), f64::INFINITY);
    assert_eq!(StopNorm::Inf.of(bad.view()), f64::INFINITY);
    assert_eq!(StopNorm::Rms.of(bad.view()), f64::INFINITY);
}

/// Four atoms each at force 0.9: max-atom 0.9, two-norm 1.8. At
/// tolerance 1 the max-atom session returns the start untouched after
/// one evaluation, and the two-norm session moves.
#[test]
fn max_atom_session_stops_where_the_two_norm_session_moves() {
    let start = Array1::from_iter((0..12).map(|k| if k % 3 == 0 { 0.9 } else { 0.0 }));
    for method in [lbfgs, fire] {
        for (norm, moves) in [(StopNorm::MaxAtom, false), (StopNorm::Two, true)] {
            let calls = AtomicUsize::new(0);
            let obj = Oracle::unbounded(12, |x: ArrayView1<f64>| {
                calls.fetch_add(1, Ordering::Relaxed);
                (0.5 * x.dot(&x), x.to_owned())
            });
            let control = Control {
                maxiter: 10,
                gtol: 1.0,
                istep: 0.1,
                maxmove: None,
                ..Default::default()
            };
            let mut s = Solver::new(method(), control, 12);
            s.set_stop_norm(norm);
            let mut x = start.clone();
            let rep = s.step(&obj, &mut x).unwrap();
            assert_eq!(x != start, moves, "{norm:?}");
            assert_eq!(rep.steps, usize::from(moves), "{norm:?}");
            if !moves {
                assert_eq!(calls.load(Ordering::Relaxed), 1);
                assert_eq!(rep.grad_norm, 0.9);
            }
        }
    }
}

/// Every report carries the gradient at its coordinates, and its norm is
/// the selected norm of that gradient.
#[test]
fn report_carries_the_gradient_its_norm_measures() {
    for method in [lbfgs, fire] {
        for norm in [
            None,
            Some(StopNorm::MaxAtom),
            Some(StopNorm::Inf),
            Some(StopNorm::Rms),
        ] {
            let mut s = session(method(), 1e-8, norm);
            let obj = Oracle::unbounded(21, lj_cluster);
            let mut x = lj7_start();
            for _ in 0..25 {
                let rep = s.step(&obj, &mut x).unwrap();
                let grad = rep.grad.expect("session report has a gradient");
                assert_eq!(grad, lj_cluster(rep.coords.view()).1);
                assert_eq!(rep.grad_norm, norm.unwrap_or_default().of(grad.view()));
                assert_eq!(rep.stop_norm, norm.unwrap_or_default());
            }
        }
    }
    let one_shot = rgmin::minimize_lbfgs(
        &Oracle::unbounded(21, lj_cluster),
        lj7_start(),
        &Control::default(),
        rgmin::LineSearch::default(),
        10,
    )
    .unwrap();
    assert!(one_shot.grad.is_none());
}

/// The norm moves only the stop point. A max-atom run follows the
/// two-norm path, stops sooner, and ends with every atom force below
/// the tolerance, as ASE `fmax` requires.
#[test]
fn max_atom_run_is_a_prefix_of_the_two_norm_run() {
    let fmax = 0.01;
    for method in [lbfgs, fire] {
        let (two, two_calls) = relax(session(method(), fmax, None), fmax);
        let (atom, atom_calls) = relax(session(method(), fmax, Some(StopNorm::MaxAtom)), fmax);
        // The eOn cap can meet the max-atom test and the two-norm on the
        // same accepted step. The max-atom run is still a prefix.
        assert!(atom.len() <= two.len());
        assert!(atom_calls <= two_calls);
        assert_eq!(atom[..], two[..atom.len()]);
        let last = atom.last().unwrap();
        let force = lj_cluster(last.view()).1;
        for f in force.exact_chunks(3) {
            assert!(f.dot(&f).sqrt() < fmax);
        }
        assert!(StopNorm::MaxAtom.of(lj_cluster(two.last().unwrap().view()).1.view()) < fmax);
    }
}

fn from_control(method: Method, gtol: f64, norm: StopNorm) -> Solver {
    let control = Control {
        maxiter: 2000,
        gtol,
        istep: 0.1,
        stop_norm: norm,
        ..Default::default()
    };
    let mut s = Solver::new(method, control, 21);
    s.set_atom_maxmove(0.2);
    s
}

/// A session takes its norm from `Control`, and the setter replaces it.
#[test]
fn control_selects_the_session_norm() {
    assert_eq!(Control::default().stop_norm, StopNorm::Two);
    for norm in [
        StopNorm::Two,
        StopNorm::MaxAtom,
        StopNorm::Inf,
        StopNorm::Rms,
    ] {
        let mut s = from_control(lbfgs(), 1.0, norm);
        assert_eq!(s.stop_norm(), norm);
        s.set_stop_norm(StopNorm::Two);
        assert_eq!(s.stop_norm(), StopNorm::Two);
    }
}

/// Each selector set in `Control` stops LJ7 at a point the Euclidean
/// session passes through without stopping: the run is a strict prefix
/// of the two-norm run, its last report clears `gtol` in the selected
/// norm, and the two-norm there is still at or above `gtol`.
#[test]
fn every_control_norm_stops_before_the_two_norm() {
    let gtol = 0.01;
    for method in [lbfgs, fire] {
        let (two, _) = relax(from_control(method(), gtol, StopNorm::Two), gtol);
        assert_eq!(two, relax(session(method(), gtol, None), gtol).0);
        for norm in [StopNorm::MaxAtom, StopNorm::Inf, StopNorm::Rms] {
            let (path, _) = relax(from_control(method(), gtol, norm), gtol);
            assert!(path.len() <= two.len(), "{norm:?}");
            assert_eq!(path[..], two[..path.len()], "{norm:?}");
            let g = lj_cluster(path.last().unwrap().view()).1;
            assert!(norm.of(g.view()) < gtol, "{norm:?}");
            if path.len() < two.len() {
                assert!(StopNorm::Two.of(g.view()) >= gtol, "{norm:?}");
            }
            println!(
                "{method:?} {norm:?}: {} of {} reports",
                path.len(),
                two.len(),
                method = method()
            );
        }
    }
}

/// `minimize_method` runs FIRE as a session, which takes the norm from
/// `Control` and names it in the report. One-shot L-BFGS runs no session
/// and reports the two-norm under the same `Control`.
#[test]
fn report_names_the_norm_of_grad_norm() {
    let control = Control {
        maxiter: 2000,
        gtol: 0.01,
        istep: 0.1,
        maxmove: Some(0.2),
        stop_norm: StopNorm::MaxAtom,
        ftol_rel: None,
        noise_floor: None,
    };
    let obj = Oracle::unbounded(21, lj_cluster);
    let rep =
        rgmin::minimize_method(&obj, lj7_start(), &control, fire(), LineSearch::default()).unwrap();
    let g = lj_cluster(rep.coords.view()).1;
    assert_eq!(rep.stop_norm, StopNorm::MaxAtom);
    assert_eq!(rep.grad_norm, StopNorm::MaxAtom.of(g.view()));
    assert!(rep.grad_norm < control.gtol);
    let rep =
        rgmin::minimize_lbfgs(&obj, lj7_start(), &control, LineSearch::default(), 20).unwrap();
    let g = lj_cluster(rep.coords.view()).1;
    assert_eq!(rep.stop_norm, StopNorm::Two);
    assert_eq!(rep.grad_norm, StopNorm::Two.of(g.view()));
    assert!(rep.grad_norm < control.gtol);
}
