use ndarray::array;
use rgmin::{
    Control, EonCg, HessianOracle, HvpOracle, IterationView, LineSearch, ManifoldKind, Method,
    NewtonKind, ObserverSignal, Oracle, Rlbfgs, StopReason, minimize_eon_cg_observed,
    minimize_method, minimize_method_hess_observed, minimize_method_hvp_observed,
    minimize_method_observed, minimize_newton_observed, minimize_rlbfgs_observed,
};

fn quadratic()
-> Oracle<impl Fn(ndarray::ArrayView1<f64>) -> (f64, ndarray::Array1<f64>) + Send + Sync> {
    Oracle::unbounded(2, |x| {
        let grad = x.to_owned();
        let value = 0.5 * x.dot(&x);
        (value, grad)
    })
}

#[test]
fn observer_can_stop_a_run_with_a_distinct_reason() {
    let obj = quadratic();
    let mut seen = 0usize;
    let control = Control::default()
        .with_maxiter(100)
        .with_gtol(1e-16)
        .with_istep(0.25);
    let report = minimize_method_observed(
        &obj,
        array![1.0, -0.5],
        &control,
        Method::Steepest,
        LineSearch::default(),
        Some(&mut |state: IterationView<'_>| {
            seen += 1;
            if state.iteration >= 2 {
                ObserverSignal::Stop
            } else {
                ObserverSignal::Continue
            }
        }),
    )
    .expect("observed run");
    assert_eq!(seen, 3);
    assert_eq!(report.stop_reason(), StopReason::ObserverRequested);
}

#[test]
fn observer_can_record_a_two_coordinate_projection() {
    let obj = quadratic();
    let control = Control::default()
        .with_maxiter(8)
        .with_gtol(1e-12)
        .with_istep(0.2);
    let mut projection = Vec::new();
    let report = minimize_method_observed(
        &obj,
        array![2.0, -1.0],
        &control,
        Method::Steepest,
        LineSearch::default(),
        Some(&mut |state: IterationView<'_>| {
            projection.push((state.x[0], state.x[1] - state.x[0]));
            ObserverSignal::Continue
        }),
    )
    .expect("projection run");
    assert!(projection.len() >= report.steps);
    assert!(projection.len() >= 2);
}

fn stop_at_third(seen: &mut usize) -> impl FnMut(IterationView<'_>) -> ObserverSignal + '_ {
    move |state: IterationView<'_>| {
        assert_eq!(state.iteration, *seen);
        *seen += 1;
        if *seen == 3 {
            ObserverSignal::Stop
        } else {
            ObserverSignal::Continue
        }
    }
}

#[test]
fn adam_and_pso_call_the_observer_and_honour_its_stop() {
    let obj = quadratic();
    let control = Control::default()
        .with_maxiter(50)
        .with_gtol(1e-16)
        .with_istep(0.25);
    for method in [Method::adam(), Method::pso()] {
        let mut seen = 0usize;
        let report = minimize_method_observed(
            &obj,
            array![1.0, -0.5],
            &control,
            method.clone(),
            LineSearch::default(),
            Some(&mut stop_at_third(&mut seen)),
        )
        .expect("observed run");
        assert_eq!(seen, 3, "{method:?}");
        assert_eq!(report.stop_reason(), StopReason::ObserverRequested);
        assert_eq!(report.steps, 3, "{method:?}");
    }
}

#[test]
fn adam_and_pso_observe_every_iteration_of_a_full_run() {
    let obj = quadratic();
    let control = Control::default()
        .with_maxiter(6)
        .with_gtol(1e-16)
        .with_istep(0.25);
    for method in [Method::adam(), Method::pso()] {
        let plain = minimize_method(
            &obj,
            array![1.0, -0.5],
            &control,
            method.clone(),
            LineSearch::default(),
        )
        .expect("plain run");
        let mut last = None;
        let mut seen = 0usize;
        let observed = minimize_method_observed(
            &obj,
            array![1.0, -0.5],
            &control,
            method.clone(),
            LineSearch::default(),
            Some(&mut |state: IterationView<'_>| {
                seen += 1;
                last = Some((state.x.to_owned(), state.value));
                ObserverSignal::Continue
            }),
        )
        .expect("observed run");
        assert_eq!(seen, 6, "{method:?}");
        assert_eq!(observed.steps, plain.steps, "{method:?}");
        assert_eq!(observed.coords, plain.coords, "{method:?}");
        assert_eq!(observed.value, plain.value, "{method:?}");
        let (x, value) = last.expect("observed iterate");
        assert_eq!(x, observed.coords, "{method:?}");
        assert_eq!(value, observed.value, "{method:?}");
    }
}

#[test]
fn a_session_method_observes_the_iterate_that_ends_the_run() {
    let obj = quadratic();
    let control = Control::default().with_maxiter(200).with_gtol(1e-6);
    let mut seen = 0usize;
    let mut last = None;
    let report = minimize_method_observed(
        &obj,
        array![1.0, -0.5],
        &control,
        Method::Bb,
        LineSearch::default(),
        Some(&mut |state: IterationView<'_>| {
            seen += 1;
            last = Some((state.iteration, state.x.to_owned(), state.grad_norm));
            ObserverSignal::Continue
        }),
    )
    .expect("session run");
    assert_eq!(report.stop_reason(), StopReason::Converged);
    let (iteration, x, grad_norm) = last.expect("observed iterate");
    assert_eq!(iteration + 1, seen);
    assert_eq!(x, report.coords);
    assert_eq!(grad_norm, report.grad_norm);
    assert!(grad_norm < 1e-6);
}

#[test]
fn a_session_observer_stop_on_the_final_iterate_wins() {
    let obj = quadratic();
    let control = Control::default().with_maxiter(200).with_gtol(1e-6);
    let report = minimize_method_observed(
        &obj,
        array![1.0, -0.5],
        &control,
        Method::Bb,
        LineSearch::default(),
        Some(&mut |state: IterationView<'_>| {
            if state.grad_norm < 1e-6 {
                ObserverSignal::Stop
            } else {
                ObserverSignal::Continue
            }
        }),
    )
    .expect("session run");
    assert_eq!(report.stop_reason(), StopReason::ObserverRequested);
    assert!(report.grad_norm < 1e-6);
}

#[test]
fn lbfgs_reports_the_length_of_each_accepted_step() {
    let obj = quadratic();
    let control = Control::default().with_maxiter(20).with_gtol(1e-10);
    let mut lengths = Vec::new();
    let mut previous = array![2.0, -1.0];
    let report = minimize_method_observed(
        &obj,
        previous.clone(),
        &control,
        Method::lbfgs_with_memory(10),
        LineSearch::default(),
        Some(&mut |state: IterationView<'_>| {
            let moved = (&state.x - &previous).dot(&(&state.x - &previous)).sqrt();
            lengths.push((state.step_length, moved));
            previous = state.x.to_owned();
            ObserverSignal::Continue
        }),
    )
    .expect("l-bfgs run");
    assert_eq!(report.stop_reason(), StopReason::Converged);
    assert!(!lengths.is_empty());
    for (reported, moved) in lengths {
        assert!(reported > 0.0);
        assert!((reported - moved).abs() <= 1e-12 * moved.max(1.0));
    }
}

#[test]
fn sr1_and_sr2_observe_only_accepted_iterates() {
    let obj = eindir_core::objectives::Rosenbrock::<2>::new();
    let control = Control::default()
        .with_maxiter(200)
        .with_gtol(1e-8)
        .with_istep(10.0);
    for method in [Method::Sr1, Method::Sr2] {
        let mut seen: Vec<(ndarray::Array1<f64>, f64)> = Vec::new();
        let report = minimize_method_observed(
            &obj,
            array![0.0, 3.0],
            &control,
            method.clone(),
            LineSearch::default(),
            Some(&mut |state: IterationView<'_>| {
                seen.push((state.x.to_owned(), state.step_length));
                ObserverSignal::Continue
            }),
        )
        .expect("trust-region run");
        assert!(seen.len() >= 2, "{method:?}");
        assert!(seen.len() < report.steps, "{method:?} rejected no step");
        for (_, step_length) in &seen {
            assert!(*step_length > 0.0, "{method:?}");
        }
        for pair in seen.windows(2) {
            assert_ne!(pair[0].0, pair[1].0, "{method:?} observed a rejected step");
        }
    }
}

#[test]
fn an_observer_stop_is_not_a_convergence_failure() {
    let obj = quadratic();
    let control = Control::default()
        .with_maxiter(50)
        .with_gtol(1e-12)
        .with_require_convergence(true);
    for method in [Method::Steepest, Method::adam(), Method::Bb] {
        let mut seen = 0usize;
        let report = minimize_method_observed(
            &obj,
            array![1.0, -0.5],
            &control,
            method.clone(),
            LineSearch::default(),
            Some(&mut stop_at_third(&mut seen)),
        )
        .unwrap_or_else(|err| panic!("{method:?}: {err}"));
        assert_eq!(report.stop_reason(), StopReason::ObserverRequested);
        assert!(!report.converged, "{method:?}");
    }
}

#[test]
fn one_shot_and_session_methods_observe_the_converged_iterate_alike() {
    let obj = quadratic();
    let gtol = 1e-6;
    let control = Control::default().with_maxiter(2000).with_gtol(gtol);
    let methods = [
        Method::Steepest,
        Method::Bfgs,
        Method::lbfgs_with_memory(10),
        Method::Sr1,
        Method::adam(),
        Method::Bb,
        Method::fire(rgmin::FireKind::V1),
    ];
    for method in methods {
        let mut calls = 0usize;
        let mut last = None;
        let report = minimize_method_observed(
            &obj,
            array![1.0, -0.5],
            &control,
            method.clone(),
            LineSearch::default(),
            Some(&mut |state: IterationView<'_>| {
                calls += 1;
                last = Some((state.x.to_owned(), state.grad_norm));
                ObserverSignal::Continue
            }),
        )
        .expect("converging run");
        assert_eq!(report.stop_reason(), StopReason::Converged, "{method:?}");
        let (x, grad_norm) = last.expect("observed iterate");
        assert_eq!(x, report.coords, "{method:?}");
        assert_eq!(grad_norm, report.grad_norm, "{method:?}");
        assert!(grad_norm < gtol, "{method:?}");

        let mut stopped_calls = 0usize;
        let stopped = minimize_method_observed(
            &obj,
            array![1.0, -0.5],
            &control,
            method.clone(),
            LineSearch::default(),
            Some(&mut |state: IterationView<'_>| {
                stopped_calls += 1;
                if state.grad_norm < gtol {
                    ObserverSignal::Stop
                } else {
                    ObserverSignal::Continue
                }
            }),
        )
        .expect("stopped run");
        assert_eq!(
            stopped.stop_reason(),
            StopReason::ObserverRequested,
            "{method:?}"
        );
        assert_eq!(stopped_calls, calls, "{method:?}");
        assert_eq!(stopped.coords, report.coords, "{method:?}");
    }
}

#[test]
fn every_session_method_observes_the_iterate_it_returns() {
    let obj = quadratic();
    let control = Control::default().with_maxiter(500).with_gtol(1e-6);
    let names = [
        "fire",
        "fire2",
        "bb",
        "quickmin",
        "ode12r",
        "diis",
        "gromacs-steep",
        "gpmin",
        "newton-cg",
        "newton-pcg",
        "gromacs-cg",
    ];
    for name in names {
        let method = Method::from_name(name).expect("known method");
        let mut last = None;
        let report = minimize_method_observed(
            &obj,
            array![1.0, -0.5],
            &control,
            method,
            LineSearch::default(),
            Some(&mut |state: IterationView<'_>| {
                last = Some((state.x.to_owned(), state.value, state.grad_norm));
                ObserverSignal::Continue
            }),
        )
        .unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!(report.stop_reason(), StopReason::Converged, "{name}");
        let (x, value, grad_norm) = last.unwrap_or_else(|| panic!("{name}: no iterate"));
        assert_eq!(x, report.coords, "{name}");
        assert_eq!(value, report.value, "{name}");
        assert_eq!(grad_norm, report.grad_norm, "{name}");
    }
}

fn bowl_hess() -> impl rgmin::HessianObjective {
    HessianOracle::unbounded(
        2,
        |x| (0.5 * x.dot(&x), x.to_owned()),
        |_| ndarray::Array2::eye(2),
    )
}

fn bowl_hvp() -> impl rgmin::HessianVector {
    HvpOracle::unbounded(2, |x| (0.5 * x.dot(&x), x.to_owned()), |_x, v| v.to_owned())
}

fn near_origin(x: &[f64]) {
    for &coord in x {
        assert!(coord.abs() < 1e-6, "{coord}");
    }
}

#[test]
fn hessian_hvp_rlbfgs_rtr_and_eon_observe_accepted_steps() {
    let control = Control::default().with_maxiter(40).with_gtol(1e-12);
    let start = array![1.0, -0.5];

    let mut newton_seen = 0usize;
    let newton = minimize_newton_observed(
        &bowl_hess(),
        start.clone(),
        &control,
        NewtonKind::Shifted,
        Some(&mut |state: IterationView<'_>| {
            newton_seen += 1;
            assert!(state.x.len() == 2);
            ObserverSignal::Continue
        }),
    )
    .expect("newton");
    assert!(newton_seen >= 1);
    assert_eq!(newton.stop_reason(), StopReason::Converged);
    near_origin(newton.coords.as_slice().unwrap());

    let mut dogleg_seen = 0usize;
    let dogleg = minimize_method_hess_observed(
        &bowl_hess(),
        start.clone(),
        &control,
        Method::Dogleg,
        LineSearch::default(),
        Some(&mut |_state: IterationView<'_>| {
            dogleg_seen += 1;
            ObserverSignal::Continue
        }),
    )
    .expect("dogleg");
    assert!(dogleg_seen >= 1);
    assert_eq!(dogleg.stop_reason(), StopReason::Converged);

    let mut hvp_seen = 0usize;
    let hvp = minimize_method_hvp_observed(
        &bowl_hvp(),
        start.clone(),
        &control,
        Method::NewtonCg,
        LineSearch::default(),
        Some(&mut |state: IterationView<'_>| {
            hvp_seen += 1;
            assert!(state.step_length >= 0.0);
            ObserverSignal::Continue
        }),
    )
    .expect("newton-cg");
    assert!(hvp_seen >= 1);
    assert_eq!(hvp.stop_reason(), StopReason::Converged);
    near_origin(hvp.coords.as_slice().unwrap());

    let mut rl_seen = 0usize;
    let rl = minimize_rlbfgs_observed(
        &ManifoldKind::Euclidean,
        &quadratic(),
        start.clone(),
        &control,
        &Rlbfgs::default(),
        Some(&mut |_state: IterationView<'_>| {
            rl_seen += 1;
            ObserverSignal::Continue
        }),
    )
    .expect("rlbfgs");
    assert!(rl_seen >= 1);
    assert_eq!(rl.stop_reason(), StopReason::Converged);
    near_origin(rl.coords.as_slice().unwrap());

    let mut opts = rgmin::rtr::RtrOptions::default();
    opts.max_iter = 20;
    opts.grad_tol = 1e-12;
    let mut rtr_seen = 0usize;
    let rtr = rgmin::rtr::minimize_rtr_observed(
        &rgmin::manifold::Euclidean,
        start.clone(),
        |x| 0.5 * x.dot(x),
        |x| x.to_owned(),
        |_x, v| v.to_owned(),
        opts,
        Some(&mut |state: IterationView<'_>| {
            rtr_seen += 1;
            assert!(state.grad_norm.is_finite());
            ObserverSignal::Continue
        }),
    );
    assert!(rtr_seen >= 1);
    assert!(!rtr.observer_stop);
    near_origin(rtr.x.as_slice().unwrap());

    let mut x = [1.0, -0.5];
    let mut eon_seen = 0usize;
    let eon = minimize_eon_cg_observed(
        &mut x,
        |point, grad| {
            let mut value = 0.0;
            for (coord, slot) in point.iter().zip(grad.iter_mut()) {
                value += 0.5 * coord * coord;
                *slot = *coord;
            }
            value
        },
        &EonCg::default().with_max_iter(40).with_gtol(1e-12),
        Some(&mut |state: IterationView<'_>| {
            eon_seen += 1;
            assert!(state.step_length >= 0.0);
            ObserverSignal::Continue
        }),
    );
    assert!(eon_seen >= 1);
    assert!(eon.converged);
    assert!(!eon.observer_stop);
    near_origin(&x);
}

#[test]
fn observer_stop_covers_the_second_order_and_manifold_solvers() {
    let control = Control::default().with_maxiter(40).with_gtol(1e-16);
    let start = array![1.0, -0.5];
    let mut stop = |_state: IterationView<'_>| ObserverSignal::Stop;

    let newton = minimize_newton_observed(
        &bowl_hess(),
        start.clone(),
        &control,
        NewtonKind::Shifted,
        Some(&mut stop),
    )
    .expect("newton stop");
    assert_eq!(newton.stop_reason(), StopReason::ObserverRequested);

    let hvp = minimize_method_hvp_observed(
        &bowl_hvp(),
        start.clone(),
        &control,
        Method::NewtonPcg,
        LineSearch::default(),
        Some(&mut stop),
    )
    .expect("hvp stop");
    assert_eq!(hvp.stop_reason(), StopReason::ObserverRequested);

    let rl = minimize_rlbfgs_observed(
        &ManifoldKind::Euclidean,
        &quadratic(),
        start.clone(),
        &control,
        &Rlbfgs::default(),
        Some(&mut stop),
    )
    .expect("rlbfgs stop");
    assert_eq!(rl.stop_reason(), StopReason::ObserverRequested);

    let mut opts = rgmin::rtr::RtrOptions::default();
    opts.max_iter = 20;
    opts.grad_tol = 1e-16;
    let rtr = rgmin::rtr::minimize_rtr_observed(
        &rgmin::manifold::Euclidean,
        start,
        |x| 0.5 * x.dot(x),
        |x| x.to_owned(),
        |_x, v| v.to_owned(),
        opts,
        Some(&mut stop),
    );
    assert!(rtr.observer_stop);
    assert!(rtr.iterations >= 1);

    let mut x = [1.0, -0.5];
    let eon = minimize_eon_cg_observed(
        &mut x,
        |point, grad| {
            let mut value = 0.0;
            for (coord, slot) in point.iter().zip(grad.iter_mut()) {
                value += 0.5 * coord * coord;
                *slot = *coord;
            }
            value
        },
        &EonCg::default().with_max_iter(20).with_gtol(1e-16),
        Some(&mut stop),
    );
    assert!(eon.observer_stop);
    assert!(!eon.converged);
}
