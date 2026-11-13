//! The free nonlinear-CG loop opens at half the step last accepted and
//! does not search a direction that fails to descend.

use std::sync::Mutex;

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Conjugacy, Control, LineSearch, Method, Oracle, Restart, Solver, minimize};

fn control() -> Control {
    Control {
        maxiter: 2,
        gtol: 1e-18,
        istep: 1.0,
        maxmove: Some(f64::INFINITY),
        ftol_rel: None,
        noise_floor: None,
    }
}

fn backtrack() -> LineSearch {
    LineSearch::Backtracking {
        c: 1e-4,
        beta: 0.5,
        maxiter: 20,
    }
}

#[test]
fn the_second_opening_step_is_half_the_step_last_accepted() {
    // f = 1/2 (x^2 + 4 y^2). Backtracking from istep = 0.2 accepts 0.2.
    // The next trial is 0.1, not the slope ratio (~5.3) and not
    // min(1, 1.01 * 2 Δf / φ'(0)) = 1.
    let seen = Mutex::new(Vec::<Array1<f64>>::new());
    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        seen.lock().unwrap().push(x.to_owned());
        let f = 0.5 * (x[0] * x[0] + 4.0 * x[1] * x[1]);
        let g = array![x[0], 4.0 * x[1]];
        (f, g)
    });
    let mut ctl = control();
    ctl.istep = 0.2;
    minimize(
        &obj,
        array![1.0, 1.0],
        &ctl,
        Conjugacy::PolakRibiere,
        Restart::Never,
        backtrack(),
    )
    .unwrap();
    let pts = seen.lock().unwrap().clone();
    let x0 = array![1.0, 1.0];
    let g0 = array![1.0, 4.0];
    let d0 = array![-1.0, -4.0];
    let x1 = array![0.8, 0.2];
    let g1 = array![0.8, 0.8];
    let y = &g1 - &g0;
    let beta = g1.dot(&y) / g0.dot(&g0);
    let d1 = &(-&g1) + &(beta * &d0);
    let slope = 0.2 * (g0.dot(&d0) / g1.dot(&d1));
    let quad: f64 = 1.01 * 2.0 * (0.4 - 2.5) / g1.dot(&d1);
    let expect = &x1 + &(0.1 * &d1);
    let slope_pt = &x1 + &(slope * &d1);
    let quad_pt = &x1 + &(quad.min(1.0) * &d1);
    // The first point off the opening ray is the second iteration's trial.
    let second = pts
        .iter()
        .find(|p| {
            let step = *p - &x0;
            (step[0] * d0[1] - step[1] * d0[0]).abs() > 1e-9
        })
        .expect("second iteration did not probe");
    for i in 0..2 {
        assert!(
            (second[i] - expect[i]).abs() < 1e-9,
            "probe {second:?} expected half-step {expect:?}"
        );
        assert!((second[i] - slope_pt[i]).abs() > 1e-6);
        assert!((second[i] - quad_pt[i]).abs() > 1e-6);
    }
}

#[test]
fn an_uphill_fletcher_reeves_direction_is_replaced_by_steepest_descent() {
    // The value is a bowl. The gradient jumps after the first accepted
    // step so Fletcher-Reeves builds an uphill direction. The next probe
    // has to follow -g, whose y component is negative. Searching that
    // uphill vector backwards moves y the other way.
    let seen = Mutex::new(Vec::<Array1<f64>>::new());
    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        seen.lock().unwrap().push(x.to_owned());
        let f = 0.5 * (x[0] * x[0] + x[1] * x[1]);
        let g = if x[0] < 1.5 {
            array![-5.0, 4.0]
        } else {
            array![1.0, 0.0]
        };
        (f, g)
    });
    minimize(
        &obj,
        array![2.0, 0.0],
        &control(),
        Conjugacy::FletcherReeves,
        Restart::Never,
        backtrack(),
    )
    .unwrap();
    let pts = seen.lock().unwrap().clone();
    let second = pts
        .iter()
        .find(|p| (p[1]).abs() > 1e-12)
        .expect("no second probe");
    assert!(
        second[1] < 0.0,
        "probe {second:?} did not follow -g = (5, -4)"
    );
    let restart = array![5.0, -4.0];
    let step = second - &array![1.0, 0.0];
    let cross = step[0] * restart[1] - step[1] * restart[0];
    assert!(
        cross.abs() < 1e-8 * step.dot(&step).sqrt(),
        "probe {second:?} is not parallel to -g"
    );
}

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

#[test]
fn a_fresh_session_uses_the_cg_curvature_and_quasi_newton_keeps_its_own() {
    let ctl = control();
    let cg = Solver::new(Method::nlcg(Conjugacy::FletcherReeves), ctl.clone(), 2);
    match cg.linesearch() {
        LineSearch::Wolfe { c1, c2, .. } => {
            assert!((c1 - 1e-4).abs() < 1e-15);
            assert!((c2 - 0.1).abs() < 1e-15);
        }
        other => panic!("cg search {other:?}"),
    }
    let qn = Solver::new(Method::lbfgs(), ctl, 2);
    match qn.linesearch() {
        LineSearch::Wolfe { c2, .. } => assert!((c2 - 0.9).abs() < 1e-15),
        other => panic!("qn search {other:?}"),
    }
}

#[test]
fn session_cg_reaches_the_stiff_quadratic_without_a_caller_linesearch() {
    // c2 = 0.9 makes the NJWS angle test restart every step. The same
    // quadratic then stays above 1e5 after forty steepest-descent steps.
    let obj = Oracle::unbounded(8, scaled);
    let start = Array1::from_elem(8, 0.5);
    let ctl = Control {
        maxiter: 80,
        gtol: 1e-5,
        istep: 1.0,
        maxmove: Some(f64::INFINITY),
        ftol_rel: None,
        noise_floor: None,
    };
    for beta in [
        Conjugacy::FletcherReeves,
        Conjugacy::DaiYuan,
        Conjugacy::HestenesStiefel,
        Conjugacy::PolakRibiere,
    ] {
        let mut solver = Solver::new(
            Method::Nlcg {
                conjugacy: beta.clone(),
                restart: Restart::njws(),
            },
            ctl.clone(),
            8,
        );
        solver.set_atom_maxmove(0.0);
        solver.set_maxmove(0.0);
        let mut x = start.clone();
        let mut steps = 0usize;
        let mut grad = f64::INFINITY;
        for _ in 0..ctl.maxiter {
            let report = solver.step(&obj, &mut x).unwrap();
            steps = report.steps;
            grad = report.grad_norm;
            if grad < ctl.gtol {
                break;
            }
        }
        assert!(
            grad < ctl.gtol && steps < ctl.maxiter,
            "{beta:?} steps={steps} |g|={grad:.3e}"
        );
    }
}
