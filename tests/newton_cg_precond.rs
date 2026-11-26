//! Trust-region Newton-CG under a caller preconditioner.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, FdHvp, HvpOracle, IdentityPrecond, Oracle, Preconditioner, minimize_newton_cg,
    minimize_newton_cg_precond,
};

fn ctrl(maxiter: usize, gtol: f64) -> Control {
    let mut control = Control::default();
    control.maxiter = maxiter;
    control.gtol = gtol;
    control.istep = 1.0;
    control.maxmove = None;
    control
}

fn same_report(a: &rgmin::Report, b: &rgmin::Report) {
    assert_eq!(a.steps, b.steps);
    assert_eq!(a.value.to_bits(), b.value.to_bits());
    assert_eq!(a.grad_norm.to_bits(), b.grad_norm.to_bits());
    assert_eq!(a.coords.len(), b.coords.len());
    for (u, v) in a.coords.iter().zip(b.coords.iter()) {
        assert_eq!(u.to_bits(), v.to_bits());
    }
}

#[test]
fn identity_reproduces_the_stiff_bowl_solve() {
    let n = 50;
    let build = || {
        HvpOracle::unbounded(
            n,
            |x| {
                let f = x
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i + 1) as f64 * v * v / 2.0)
                    .sum();
                let g = Array1::from_iter(x.iter().enumerate().map(|(i, v)| (i + 1) as f64 * v));
                (f, g)
            },
            |_, v: ArrayView1<f64>| {
                Array1::from_iter(v.iter().enumerate().map(|(i, w)| (i + 1) as f64 * w))
            },
        )
    };
    let init = Array1::from_elem(n, 1.0);
    let plain = minimize_newton_cg(&build(), init.clone(), &ctrl(200, 1e-8)).unwrap();
    let pre =
        minimize_newton_cg_precond(&build(), init, &ctrl(200, 1e-8), &IdentityPrecond).unwrap();
    same_report(&plain, &pre);
}

fn rosenbrock_2d() -> impl rgmin::HessianVector {
    HvpOracle::unbounded(
        2,
        |x| {
            let (a, b) = (1.0, 100.0);
            let f = (a - x[0]).powi(2) + b * (x[1] - x[0] * x[0]).powi(2);
            let g = array![
                -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0] * x[0]),
                2.0 * b * (x[1] - x[0] * x[0])
            ];
            (f, g)
        },
        |x, v| {
            let b = 100.0;
            let h11 = 2.0 - 4.0 * b * (x[1] - 3.0 * x[0] * x[0]);
            let h12 = -4.0 * b * x[0];
            let h22 = 2.0 * b;
            array![h11 * v[0] + h12 * v[1], h12 * v[0] + h22 * v[1]]
        },
    )
}

#[test]
fn identity_reproduces_the_rosenbrock_solve() {
    let plain = minimize_newton_cg(&rosenbrock_2d(), array![-1.2, 1.0], &ctrl(500, 1e-8)).unwrap();
    let pre = minimize_newton_cg_precond(
        &rosenbrock_2d(),
        array![-1.2, 1.0],
        &ctrl(500, 1e-8),
        &IdentityPrecond,
    )
    .unwrap();
    same_report(&plain, &pre);
}

#[test]
fn identity_reproduces_the_finite_difference_solve() {
    let base = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let f = (x[0] - 3.0).powi(2) + 10.0 * (x[1] + 1.0).powi(2);
        (f, array![2.0 * (x[0] - 3.0), 20.0 * (x[1] + 1.0)])
    });
    let fd = FdHvp::new(&base, 1e-6);
    let plain = minimize_newton_cg(&fd, array![0.0, 0.0], &ctrl(100, 1e-7)).unwrap();
    let pre = minimize_newton_cg_precond(&fd, array![0.0, 0.0], &ctrl(100, 1e-7), &IdentityPrecond)
        .unwrap();
    same_report(&plain, &pre);
}

/// `P^{-1} r = r / d`: the exact inverse of a diagonal Hessian.
struct DiagInverse(Array1<f64>);

impl Preconditioner for DiagInverse {
    fn solve(&self, r: ArrayView1<f64>) -> Array1<f64> {
        &r / &self.0
    }
}

#[test]
fn an_exact_diagonal_preconditioner_collapses_the_inner_iterations() {
    // d_i spans 1 .. 1e6 geometrically.
    let n = 50;
    let d = Array1::from_iter((0..n).map(|i| 10f64.powf(6.0 * i as f64 / (n - 1) as f64)));
    let actions = AtomicUsize::new(0);
    let obj = HvpOracle::unbounded(
        n,
        {
            let d = d.clone();
            move |x| {
                let f = x.iter().zip(d.iter()).map(|(v, l)| l * v * v / 2.0).sum();
                let g = Array1::from_iter(x.iter().zip(d.iter()).map(|(v, l)| l * v));
                (f, g)
            }
        },
        {
            let d = d.clone();
            let actions = &actions;
            move |_, v: ArrayView1<f64>| {
                actions.fetch_add(1, Ordering::Relaxed);
                Array1::from_iter(v.iter().zip(d.iter()).map(|(w, l)| l * w))
            }
        },
    );
    let init = Array1::from_elem(n, 1.0);
    // A radius far above the minimiser's norm keeps the region inactive,
    // so only the inner solve differs between the two runs.
    let mut control = ctrl(50, 1e-8);
    control.istep = 1.0e8;

    let plain = minimize_newton_cg(&obj, init.clone(), &control).unwrap();
    let plain_actions = actions.swap(0, Ordering::Relaxed);

    let pre = minimize_newton_cg_precond(&obj, init, &control, &DiagInverse(d.clone())).unwrap();
    let pre_actions = actions.load(Ordering::Relaxed);

    for rep in [&plain, &pre] {
        assert!(rep.grad_norm < 1e-8, "grad_norm {}", rep.grad_norm);
        assert!(
            rep.coords.iter().all(|v| v.abs() < 1e-8),
            "not at the minimiser"
        );
    }
    assert!(
        pre_actions <= 2 * pre.steps.max(1),
        "{pre_actions} actions over {} outer steps",
        pre.steps
    );
    assert!(
        plain_actions >= 10 * pre_actions,
        "identity {plain_actions} actions vs preconditioned {pre_actions}"
    );
}
