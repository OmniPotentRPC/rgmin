//! Matrix-free Newton over Hessian actions.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, array};
use rgmin::nlcg::{Conjugacy, Restart};
use rgmin::{
    Control, FdHvp, HvpOracle, Oracle, ScgParams, minimize_newton_cg, minimize_scg_exact,
    steihaug_cg,
};

fn ctrl(maxiter: usize, gtol: f64) -> Control {
    Control {
        maxiter,
        gtol,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
    }
}

#[test]
fn the_cg_path_solves_a_stiff_bowl_in_a_few_actions() {
    // f = sum_i i x_i^2 / 2, condition number n.
    let n = 50;
    let actions = AtomicUsize::new(0);
    let counted = HvpOracle::unbounded(
        n,
        move |x| {
            let f = x
                .iter()
                .enumerate()
                .map(|(i, v)| (i + 1) as f64 * v * v / 2.0)
                .sum();
            let g = Array1::from_iter(x.iter().enumerate().map(|(i, v)| (i + 1) as f64 * v));
            (f, g)
        },
        |_, v: ndarray::ArrayView1<f64>| {
            actions.fetch_add(1, Ordering::Relaxed);
            Array1::from_iter(v.iter().enumerate().map(|(i, w)| (i + 1) as f64 * w))
        },
    );
    let init = Array1::from_elem(n, 1.0);
    let rep = minimize_newton_cg(&counted, init, &ctrl(200, 1e-8)).unwrap();
    assert!(rep.grad_norm < 1e-8, "grad_norm {}", rep.grad_norm);
    assert!(rep.value.abs() < 1e-12, "value {}", rep.value);
    let used = actions.load(Ordering::Relaxed);
    assert!(used < 8 * n, "{used} actions for n={n}");
}

#[test]
fn rosenbrock_reaches_the_valley_floor() {
    let obj = HvpOracle::unbounded(
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
    );
    let rep = minimize_newton_cg(&obj, array![-1.2, 1.0], &ctrl(500, 1e-8)).unwrap();
    assert!(rep.grad_norm < 1e-8, "grad_norm {}", rep.grad_norm);
    assert!((rep.coords[0] - 1.0).abs() < 1e-6);
    assert!((rep.coords[1] - 1.0).abs() < 1e-6);
}

#[test]
fn negative_curvature_walks_to_the_trust_boundary() {
    // f = (x^2 - y^2)/2: indefinite everywhere.
    let obj = HvpOracle::unbounded(
        2,
        |x| ((x[0] * x[0] - x[1] * x[1]) / 2.0, array![x[0], -x[1]]),
        |_, v| array![v[0], -v[1]],
    );
    let x = array![0.3, 0.01];
    let g = array![0.3, -0.01];
    let (p, drop) = steihaug_cg(&obj, x.view(), &g, 0.5, 1e-10, 20);
    let pn = p.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((pn - 0.5).abs() < 1e-10, "step lands on the boundary, {pn}");
    assert!(drop > 0.0, "the model must predict a drop, got {drop}");
}

#[test]
fn finite_difference_actions_match_the_analytic_hessian() {
    // f = sum x_i^4: H v = 12 x_i^2 v_i.
    let base = Oracle::unbounded(3, |x: ndarray::ArrayView1<f64>| {
        let f = x.iter().map(|v| v.powi(4)).sum();
        let g = Array1::from_iter(x.iter().map(|v| 4.0 * v.powi(3)));
        (f, g)
    });
    let fd = FdHvp::new(&base, 1e-6);
    let x = array![0.5, -1.0, 2.0];
    let v = array![1.0, 2.0, -1.0];
    let hv = rgmin::HessianVector::hessian_vector(&fd, x.view(), v.view());
    for (i, xi) in x.iter().enumerate() {
        let want = 12.0 * xi * xi * v[i];
        assert!(
            (hv[i] - want).abs() < 1e-4 * (1.0 + want.abs()),
            "component {i}: {} vs {want}",
            hv[i]
        );
    }
}

#[test]
fn a_hessian_action_drives_scg_exact_through_the_blanket() {
    let obj = HvpOracle::unbounded(
        4,
        |x| {
            let f = x.iter().map(|v| v * v).sum::<f64>();
            (f, x.mapv(|v| 2.0 * v))
        },
        |_, v| v.mapv(|w| 2.0 * w),
    );
    let rep = minimize_scg_exact(
        &obj,
        Array1::from_elem(4, 1.5),
        &ctrl(200, 1e-8),
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::Njws { threshold: 0.1 },
    )
    .unwrap();
    assert!(rep.value.abs() < 1e-10, "value {}", rep.value);
}

#[test]
fn the_fd_wrapper_minimizes_without_an_analytic_hessian() {
    let base = Oracle::unbounded(2, |x: ndarray::ArrayView1<f64>| {
        let f = (x[0] - 3.0).powi(2) + 10.0 * (x[1] + 1.0).powi(2);
        (f, array![2.0 * (x[0] - 3.0), 20.0 * (x[1] + 1.0)])
    });
    let fd = FdHvp::new(&base, 1e-6);
    let rep = minimize_newton_cg(&fd, array![0.0, 0.0], &ctrl(100, 1e-7)).unwrap();
    assert!((rep.coords[0] - 3.0).abs() < 1e-5);
    assert!((rep.coords[1] + 1.0).abs() < 1e-5);
}

#[test]
fn nystrom_flattens_a_decaying_spectrum() {
    use rgmin::{NystromPrecond, steihaug_pcg};
    // lambda_i = 1e4 / i^2: a few stiff modes over a soft bulk.
    let n = 300;
    let lam: Vec<f64> = (1..=n).map(|i| 1.0e4 / ((i * i) as f64)).collect();
    let actions = AtomicUsize::new(0);
    let obj = HvpOracle::unbounded(
        n,
        {
            let lam = lam.clone();
            move |x| {
                let f = x.iter().zip(&lam).map(|(v, l)| l * v * v / 2.0).sum();
                let g = Array1::from_iter(x.iter().zip(&lam).map(|(v, l)| l * v));
                (f, g)
            }
        },
        {
            let lam = lam.clone();
            let actions = &actions;
            move |_, v: ndarray::ArrayView1<f64>| {
                actions.fetch_add(1, Ordering::Relaxed);
                Array1::from_iter(v.iter().zip(&lam).map(|(w, l)| l * w))
            }
        },
    );
    let x0 = Array1::zeros(n);
    let g = Array1::from_iter(lam.iter().map(|l| l.sqrt()));
    let radius = 1.0e9;
    let rtol = 1.0e-8;

    actions.store(0, Ordering::Relaxed);
    let (p_plain, _) = steihaug_cg(&obj, x0.view(), &g, radius, rtol, 4 * n);
    let plain_actions = actions.load(Ordering::Relaxed);

    actions.store(0, Ordering::Relaxed);
    let rank = 16;
    let pre = NystromPrecond::build(&obj, x0.view(), rank, 42);
    let (p_pcg, _) = steihaug_pcg(&obj, x0.view(), &g, radius, rtol, 4 * n, &pre);
    let total_pcg_actions = actions.load(Ordering::Relaxed);

    // Same subproblem, same answer: CG is exact under any SPD
    // preconditioner.
    let diff = (&p_plain - &p_pcg)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let scale = p_plain.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(diff < 1e-5 * scale.max(1.0), "steps differ by {diff}");
    // The sketch (rank actions) plus the preconditioned solve beats
    // the plain solve on this spectrum.
    assert!(
        total_pcg_actions < plain_actions,
        "pcg {total_pcg_actions} (incl. {rank} sketch) vs plain {plain_actions}"
    );
}

#[test]
fn the_preconditioned_boundary_lives_in_the_sketch_metric() {
    use rgmin::{IdentityPrecond, steihaug_pcg};
    // Identity preconditioner must reproduce steihaug_cg exactly.
    let obj = HvpOracle::unbounded(
        2,
        |x| ((x[0] * x[0] - x[1] * x[1]) / 2.0, array![x[0], -x[1]]),
        |_, v| array![v[0], -v[1]],
    );
    let x = array![0.3, 0.01];
    let g = array![0.3, -0.01];
    let (p1, d1) = steihaug_cg(&obj, x.view(), &g, 0.5, 1e-10, 20);
    let (p2, d2) = steihaug_pcg(&obj, x.view(), &g, 0.5, 1e-10, 20, &IdentityPrecond);
    assert_eq!(p1, p2);
    assert_eq!(d1, d2);
}
