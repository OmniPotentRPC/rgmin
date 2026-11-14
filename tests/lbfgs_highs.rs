//! L-BFGS quadratic model through HiGHS.

#![cfg(feature = "highs")]

use approx::assert_relative_eq;
use ndarray::Array2;
use ndarray::{Array1, ArrayView1};
use rgmin::{HighsStep, Lbfgs};

#[test]
fn session_box_recognizes_boundary_stationarity() {
    use ndarray::array;
    use rgmin::{Control, Method, Oracle, Solver};

    let obj = Oracle::unbounded(3, |x| {
        let g = &x - &array![0.0, 2.0, -4.0];
        (0.5 * g.dot(&g), g)
    });
    let mut x = array![1.0, 1.0, 3.0];
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 3).with_gtol(1e-10);
    solver.set_highs(true);
    assert!(solver.set_box(Some(vec![1.0, -10.0, 3.0]), Some(vec![10.0, 1.0, 3.0])));
    let report = solver.step(&obj, &mut x).unwrap();
    assert_eq!(x, array![1.0, 1.0, 3.0]);
    assert_eq!(report.steps, 0);
    assert_eq!(report.grad_norm, 0.0);
    assert_relative_eq!(report.value, 25.5, epsilon = 1e-14);
}

#[test]
fn box_projection_keeps_free_components_near_a_wall() {
    use ndarray::array;

    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep::default());
    let lo = [f64::NEG_INFINITY, 0.0];
    for distance in [1e-8, 1e-12] {
        let x = array![3.0, distance];
        let gradient = array![3.0, 1.0 + distance];
        let step = opt
            .highs_step_boxed(x.view(), gradient.view(), Some(&lo), None)
            .unwrap();
        assert_relative_eq!(step[0], -3.0, epsilon = 1e-14);
        assert_relative_eq!(step[1], -distance, epsilon = 1e-16);
        assert!(gradient.dot(&step) < 0.0);
    }
}

#[test]
fn session_box_reaches_the_constrained_quadratic_minimum() {
    use ndarray::array;
    use rgmin::{Control, Method, Oracle, Solver};

    let obj = Oracle::unbounded(2, |x| {
        assert!(x[1] >= 0.0, "line search left the box: {x:?}");
        let gradient = array![x[0], x[1] + 1.0];
        (0.5 * gradient.dot(&gradient), gradient)
    });
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2).with_gtol(1e-10);
    solver.set_highs(true);
    // The default per-atom cap would shorten the Newton step. This test
    // checks the box, so the cap is off.
    solver.set_atom_maxmove(0.0);
    assert!(solver.set_box(Some(vec![f64::NEG_INFINITY, 0.0]), None));
    let mut x = array![3.0, 1e-8];
    let report = solver.step(&obj, &mut x).unwrap();
    assert_relative_eq!(x[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(x[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(report.value, 0.5, epsilon = 1e-12);
    assert!(report.grad_norm <= 1e-10, "{report:?}");
}

#[test]
fn box_projection_restarts_a_non_descent_curvature_direction() {
    use ndarray::array;

    let mut opt = Lbfgs::default();
    opt.record(array![1.0, 2.0], array![1.0, 0.0]);
    opt.highs = Some(HighsStep::default());
    let hi = [f64::INFINITY, 0.0];
    let x = array![0.0, 0.0];
    let gradient = array![1.0, -1.0];
    let step = opt
        .highs_step_boxed(x.view(), gradient.view(), None, Some(&hi))
        .unwrap();
    assert!(gradient.dot(&step) < 0.0, "non-descent step: {step:?}");
    assert_relative_eq!(step[0], -1.0, epsilon = 1e-14);
    assert_eq!(step[1], 0.0);
}

#[test]
fn session_box_keeps_every_line_search_evaluation_feasible() {
    use ndarray::array;
    use rgmin::{Control, Method, Oracle, Solver};

    let obj = Oracle::unbounded(2, |x| {
        assert!(x[0] >= 1.0 && x[0] <= 10.0, "lower wall: {x:?}");
        assert!(x[1] >= -10.0 && x[1] <= 1.0, "upper wall: {x:?}");
        let g = &x - &array![0.0, 2.0];
        (0.5 * g.dot(&g), g)
    });
    let mut x = array![2.0, -1.0];
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2).with_gtol(1e-10);
    solver.set_highs(true);
    assert!(solver.set_box(Some(vec![1.0, -10.0]), Some(vec![10.0, 1.0])));
    let mut report = solver.step(&obj, &mut x).unwrap();
    for _ in 0..100 {
        if report.grad_norm <= 1e-10 {
            break;
        }
        report = solver.step(&obj, &mut x).unwrap();
    }
    assert!(report.grad_norm <= 1e-10, "{report:?}");
    assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(report.value, 1.0, epsilon = 1e-10);
}

#[test]
fn session_box_normal_does_not_hide_a_free_gradient() {
    use ndarray::array;
    use rgmin::{Control, Method, Oracle, Solver};

    let obj = Oracle::unbounded(2, |x| (0.5 * x.dot(&x), x.to_owned()));
    let mut x = array![1.0, 2.0];
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2).with_gtol(1e-10);
    solver.set_highs(true);
    // Free coordinate 2 sits past the default per-atom cap.
    solver.set_atom_maxmove(0.0);
    assert!(solver.set_box(Some(vec![1.0, -10.0]), Some(vec![10.0, 10.0])));
    let report = solver.step(&obj, &mut x).unwrap();
    assert!(report.steps > 0);
    assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(x[1], 0.0, epsilon = 1e-10);
    assert!(report.grad_norm <= 1e-10, "{report:?}");
}

#[test]
fn session_box_wolfe_evaluations_respect_the_domain() {
    use ndarray::array;
    use rgmin::{Accept, Control, Method, Oracle, Solver};

    let obj = Oracle::unbounded(2, |x| {
        assert!(x[0] >= 1.0 && x[0] <= 10.0, "lower wall: {x:?}");
        assert!(x[1] >= -10.0 && x[1] <= 1.0, "upper wall: {x:?}");
        let g = &x - &array![0.0, 2.0];
        (0.5 * g.dot(&g), g)
    });
    let mut x = array![2.0, -1.0];
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2).with_gtol(1e-10);
    solver.set_accept(Accept::Energy);
    solver.set_highs(true);
    assert!(solver.set_box(Some(vec![1.0, -10.0]), Some(vec![10.0, 1.0])));
    let mut report = solver.step(&obj, &mut x).unwrap();
    for _ in 0..100 {
        if report.grad_norm <= 1e-10 {
            break;
        }
        report = solver.step(&obj, &mut x).unwrap();
    }
    assert!(report.grad_norm <= 1e-10, "{report:?}");
    assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(report.value, 1.0, epsilon = 1e-10);
}

fn check_fire_session_box(kind: rgmin::FireKind) {
    use ndarray::array;
    use rgmin::{Control, Method, Oracle, Solver};

    let objective = Oracle::unbounded(2, |x| {
        assert!(x[1] >= 0.0, "callback outside the coordinate box: {x:?}");
        let gradient = array![x[0], x[1] + 1.0];
        (0.5 * gradient.dot(&gradient), gradient)
    });
    let control = Control {
        istep: 0.1,
        gtol: 1e-8,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Fire { kind }, control, 2);
    solver.set_highs(true);
    assert!(solver.set_box(Some(vec![f64::NEG_INFINITY, 0.0]), None));
    let mut x = array![3.0, 1e-8];
    let mut report = solver.step(&objective, &mut x).unwrap();
    for _ in 0..1000 {
        assert!(x[1] >= 0.0, "FIRE iterate outside the box: {x:?}");
        assert!(report.value.is_finite() && report.grad_norm.is_finite());
        if report.grad_norm <= 1e-8 {
            break;
        }
        report = solver.step(&objective, &mut x).unwrap();
    }
    assert!(report.grad_norm <= 1e-8, "{report:?}");
    assert_relative_eq!(x[0], 0.0, epsilon = 1e-8);
    assert_relative_eq!(x[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(report.value, 0.5, epsilon = 1e-12);
}

#[test]
fn fire_session_box_keeps_a_free_coordinate_convergent() {
    check_fire_session_box(rgmin::FireKind::V1);
}

#[test]
fn fire2_session_box_keeps_a_free_coordinate_convergent() {
    check_fire_session_box(rgmin::FireKind::V2);
}

fn quad(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let scales = [1.0, 10.0, 100.0, 1000.0];
    let mut f = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..x.len() {
        let c = scales[i % scales.len()];
        f += 0.5 * c * x[i] * x[i];
        g[i] = c * x[i];
    }
    (f, g)
}

#[test]
fn empty_memory_unbounded_step_is_steepest() {
    let opt = {
        let mut o = Lbfgs::default();
        o.highs = Some(HighsStep::default());
        o
    };
    let x = Array1::from(vec![1.0, 2.0, 3.0]);
    let g = Array1::from(vec![4.0, -5.0, 6.0]);
    let d = opt.highs_step(x.view(), g.view()).unwrap();
    assert_relative_eq!(d[0], -4.0, epsilon = 1e-6);
    assert_relative_eq!(d[1], 5.0, epsilon = 1e-6);
    assert_relative_eq!(d[2], -6.0, epsilon = 1e-6);
}

#[test]
fn unconstrained_qp_matches_two_loop() {
    let x = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
    let scales = [1.0, 10.0, 100.0, 1000.0];
    let s = Array1::from(vec![-0.1, -0.1, -0.1, -0.1]);
    let y = Array1::from_iter((0..4).map(|i| scales[i] * s[i]));
    let mut warm = Lbfgs::default();
    warm.record(s, y);
    let g = quad(x.view()).1;
    let two = warm.two_loop(g.view());
    warm.highs = Some(HighsStep::default());
    let qp = warm.highs_step(x.view(), g.view()).unwrap();
    for i in 0..two.len() {
        assert_relative_eq!(qp[i], two[i], epsilon = 1e-5, max_relative = 1e-4);
    }
}

#[test]
fn box_keeps_the_trial_inside() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep {
        trust: Some(0.25),
        lo: Some(-0.3),
        hi: Some(0.3),
        equalities: Vec::new(),
        center_axes: None,
    });
    let x = Array1::from(vec![0.2, -0.2]);
    let g = Array1::from(vec![10.0, -10.0]);
    let d = opt.highs_step(x.view(), g.view()).unwrap();
    for i in 0..x.len() {
        let t = x[i] + d[i];
        assert!((-0.3 - 1e-9..=0.3 + 1e-9).contains(&t), "left the box: {t}");
        assert!(d[i].abs() <= 0.25 + 1e-9, "left the trust: {}", d[i]);
    }
}

#[test]
fn highs_lbfgs_converges_on_the_quadratic() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep::default());
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let (f, x, evals) = opt.minimize(x0.view(), 80, |v| Some(quad(v)));
    assert!(f < 1e-8, "did not converge, f = {f}, evals = {evals}");
    assert!(x.iter().all(|v| v.abs() < 1e-3));
}

#[test]
fn equality_projection_scales_to_thirty_two_variables() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep {
        equalities: vec![(vec![(0, 1.0), (1, 1.0)], 0.0)],
        ..HighsStep::default()
    });
    let x0 = Array1::from_elem(32, 1.0);
    let g0 = quad(x0.view()).1;
    let p = opt.highs_step(x0.view(), g0.view()).unwrap();
    assert!(p.iter().all(|value| value.is_finite()));
    assert!(
        (p[0] + p[1]).abs() < 1e-8,
        "equality on p: p0+p1={}",
        p[0] + p[1]
    );
    let (_, x, _) = opt.minimize(x0.view(), 80, |v| Some(quad(v)));
    assert!(x.iter().all(|value| value.is_finite()));
}

#[test]
fn trust_scale_keeps_the_two_loop_direction() {
    let x = Array1::from(vec![0.0, 0.0]);
    let g = Array1::from(vec![10.0, 1.0]);
    let mut opt = Lbfgs::default();
    let raw = opt.two_loop(g.view());
    opt.highs = Some(HighsStep {
        trust: Some(0.25),
        lo: None,
        hi: None,
        equalities: Vec::new(),
        center_axes: None,
    });
    let p = opt.highs_step(x.view(), g.view()).unwrap();
    let n2 = raw.dot(&raw).sqrt();
    let np = p.dot(&p).sqrt();
    let pinf = p.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(pinf <= 0.25 + 1e-12);
    for i in 0..2 {
        assert_relative_eq!(p[i] / np, raw[i] / n2, epsilon = 1e-12);
    }
}

#[test]
fn center_axes_kills_the_mean() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep {
        trust: Some(0.5),
        lo: None,
        hi: None,
        equalities: Vec::new(),
        center_axes: Some((4, 2)),
    });
    let x = Array1::zeros(8);
    let g = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    let p = opt.highs_step(x.view(), g.view()).unwrap();
    let mut mx = 0.0;
    let mut my = 0.0;
    for i in 0..4 {
        mx += p[i * 2];
        my += p[i * 2 + 1];
    }
    assert!(mx.abs() < 1e-12, "axis 0 mean {mx}");
    assert!(my.abs() < 1e-12, "axis 1 mean {my}");
    for v in p.iter() {
        assert!(v.abs() <= 0.5 + 1e-12);
    }
}

#[test]
fn highs_newton_qp_on_a_quadratic_respects_a_box() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::{ArrayView1, array};
    use rgmin::{Control, HessianObjective, Method, QnStep, Solver};

    struct Quad;
    impl Objective<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            5.0 * x[0] * x[0] + 0.5 * x[1] * x[1]
        }
    }
    impl Gradient<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            array![10.0 * x[0], x[1]]
        }
    }
    impl DifferentiableObjective<f64> for Quad {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }
    impl HessianObjective for Quad {
        fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![10.0, 0.0, 0.0, 1.0]).unwrap()
        }
    }

    let obj = Quad;
    let mut x = array![2.0, -3.0];
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            maxiter: 8,
            gtol: 1e-10,
            istep: 1.0,
            maxmove: None,
            ..Default::default()
        },
        2,
    );
    solver.set_qn_step(QnStep::Newton);
    solver.set_highs(true);
    solver.set_atom_maxmove(0.5);
    let first = solver.step_hess(&obj, &mut x).unwrap();
    assert!(first.grad_norm.is_finite());
    assert!((x[0] - 2.0).abs() <= 0.5 + 1e-9);
    assert!((x[1] + 3.0).abs() <= 0.5 + 1e-9);
    for _ in 0..12 {
        let rep = solver.step_hess(&obj, &mut x).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }
    assert!(x.iter().all(|v| v.abs() < 1e-5), "end {x:?}");
}

#[test]
fn per_coordinate_box_is_independent() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep::default());
    let lo = [-0.1, -10.0];
    let hi = [0.1, 10.0];
    let x = Array1::from(vec![0.0, 0.0]);
    let g = Array1::from(vec![10.0, 10.0]);
    let d = opt
        .highs_step_boxed(x.view(), g.view(), Some(&lo), Some(&hi))
        .unwrap();
    for i in 0..2 {
        let t = x[i] + d[i];
        let (lo, hi) = if i == 0 { (-0.1, 0.1) } else { (-10.0, 10.0) };
        assert!(t >= lo - 1e-9 && t <= hi + 1e-9, "left the box: {t}");
    }
}

#[test]
fn per_coord_box_is_not_uniform() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep::default());
    let lo = [-0.05, -10.0];
    let hi = [0.05, 10.0];
    let x = Array1::from(vec![0.0, 0.0]);
    let g = Array1::from(vec![1.0, 100.0]);
    let d = opt
        .highs_step_boxed(x.view(), g.view(), Some(&lo), Some(&hi))
        .unwrap();
    assert!(
        (x[0] + d[0]).abs() <= 0.05 + 1e-9,
        "tight axis left the box: {}",
        x[0] + d[0]
    );
    assert!(
        (x[1] + d[1]).abs() > 0.5,
        "wide axis was clipped as if uniform: {}",
        x[1] + d[1]
    );
    assert!(
        (x[1] + d[1]).abs() <= 10.0 + 1e-9,
        "wide axis left its own box: {}",
        x[1] + d[1]
    );
}

#[test]
fn session_set_box_holds_a_per_coordinate_wall() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::{ArrayView1, array};
    use rgmin::{Control, HessianObjective, Method, QnStep, Solver};

    struct Quad;
    impl Objective<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            5.0 * x[0] * x[0] + 0.5 * x[1] * x[1]
        }
    }
    impl Gradient<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            array![10.0 * x[0], x[1]]
        }
    }
    impl DifferentiableObjective<f64> for Quad {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }
    impl HessianObjective for Quad {
        fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![10.0, 0.0, 0.0, 1.0]).unwrap()
        }
    }

    let obj = Quad;
    let mut x = array![2.0, -3.0];
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            maxiter: 8,
            gtol: 1e-10,
            istep: 1.0,
            maxmove: None,
            ..Default::default()
        },
        2,
    );
    solver.set_qn_step(QnStep::Newton);
    solver.set_highs(true);
    assert!(solver.set_box(Some(vec![1.5, -10.0]), Some(vec![10.0, 10.0])));
    let first = solver.step_hess(&obj, &mut x).unwrap();
    assert!(first.grad_norm.is_finite());
    assert!(x[0] >= 1.5 - 1e-9, "left the wall {}", x[0]);
    assert!(x[1] >= -10.0 - 1e-9);
}

#[cfg(feature = "capi")]
#[test]
fn c_abi_set_box_keeps_the_trial_inside() {
    use rgmin::ffi::{
        rgmin_accept_t, rgmin_control_t, rgmin_method_t, rgmin_report_t, rgmin_solver_create,
        rgmin_solver_free, rgmin_solver_set_accept, rgmin_solver_set_box, rgmin_solver_set_highs,
        rgmin_solver_step, rgmin_status_t, rgmin_stop_t, rgmin_tensor_borrow_cpu_f64,
        rgmin_tensor_free,
    };
    use std::os::raw::c_void;

    unsafe extern "C" fn ev(
        _user: *mut c_void,
        x: *const dlpk::sys::DLManagedTensorVersioned,
        value_out: *mut f64,
    ) -> rgmin_status_t {
        let dl = unsafe { &(*x).dl_tensor };
        let p = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
        let x0 = unsafe { *p };
        let x1 = unsafe { *p.add(1) };
        unsafe {
            *value_out = 5.0 * x0 * x0 + 0.5 * x1 * x1;
        }
        rgmin_status_t::RGMIN_SUCCESS
    }
    unsafe extern "C" fn gd(
        _user: *mut c_void,
        x: *const dlpk::sys::DLManagedTensorVersioned,
        g: *mut dlpk::sys::DLManagedTensorVersioned,
    ) -> rgmin_status_t {
        let dl = unsafe { &(*x).dl_tensor };
        let p = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
        let gl = unsafe { &(*g).dl_tensor };
        let gp = unsafe { (gl.data as *mut u8).add(gl.byte_offset as usize) as *mut f64 };
        unsafe {
            *gp = 10.0 * *p;
            *gp.add(1) = *p.add(1);
        }
        rgmin_status_t::RGMIN_SUCCESS
    }

    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 1e-12,
        istep: 1.0,
        memory: 4,
        maxmove: 0.0,
        noise_eps: 0.0,
        noise_c: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    let lo = [-0.3_f64, -0.3];
    let hi = [0.3_f64, 0.3];
    assert_eq!(unsafe { rgmin_solver_set_highs(session, 1) }, 0);
    assert_eq!(
        unsafe { rgmin_solver_set_box(session, lo.as_ptr(), hi.as_ptr(), 2) },
        0
    );
    unsafe { rgmin_solver_set_accept(session, rgmin_accept_t::RGMIN_ACCEPT_NONE) };
    let mut x = [0.2_f64, -0.2];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
        stop: rgmin_stop_t::RGMIN_STOP_TOLERANCE,
    };
    let st = unsafe {
        rgmin_solver_step(
            session,
            Some(ev),
            Some(gd),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    unsafe { rgmin_solver_free(session) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(x[0] >= -0.3 - 1e-9 && x[0] <= 0.3 + 1e-9, "x0 {}", x[0]);
    assert!(x[1] >= -0.3 - 1e-9 && x[1] <= 0.3 + 1e-9, "x1 {}", x[1]);
}

#[test]
fn null_side_is_unbounded_on_that_side() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep {
        trust: None,
        lo: Some(0.0),
        hi: None,
        equalities: Vec::new(),
        center_axes: None,
    });
    let x = Array1::from(vec![0.5, 0.5]);
    let g = Array1::from(vec![-10.0, -1.0]);
    let d = opt.highs_step(x.view(), g.view()).unwrap();
    assert!(
        x[0] + d[0] >= -1e-12 && x[1] + d[1] >= -1e-12,
        "lower side must hold: {:?}",
        (x[0] + d[0], x[1] + d[1])
    );
    assert!(
        d[0] > 5.0 && d[1] > 0.5,
        "unbounded upper must not clip ascent: {d:?}"
    );
}

#[test]
fn session_set_box_survives_set_highs_and_clips_newton() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::{ArrayView1, array};
    use rgmin::{Control, HessianObjective, Method, QnStep, Solver};

    struct Quad;
    impl Objective<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            5.0 * x[0] * x[0] + 0.5 * x[1] * x[1]
        }
    }
    impl Gradient<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            array![10.0 * x[0], x[1]]
        }
    }
    impl DifferentiableObjective<f64> for Quad {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }
    impl HessianObjective for Quad {
        fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![10.0, 0.0, 0.0, 1.0]).unwrap()
        }
    }

    let obj = Quad;
    let mut x = array![2.0, -3.0];
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            maxiter: 8,
            gtol: 1e-10,
            istep: 1.0,
            maxmove: None,
            ..Default::default()
        },
        2,
    );
    solver.set_qn_step(QnStep::Newton);
    assert!(solver.set_box(Some(vec![0.0, -0.4]), Some(vec![0.4, 0.0])));
    solver.set_highs(true);
    let first = solver.step_hess(&obj, &mut x).unwrap();
    assert!(first.grad_norm.is_finite());
    assert!(x[0] >= -1e-12 && x[0] <= 0.4 + 1e-9, "x0={x:?}");
    assert!(x[1] >= -0.4 - 1e-9 && x[1] <= 1e-12, "x1={x:?}");
}

#[test]
fn newton_qp_applies_session_equalities() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::{ArrayView1, array};
    use rgmin::{Control, HessianObjective, Method, QnStep, Solver};

    struct Quad;
    impl Objective<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            5.0 * x[0] * x[0] + 0.5 * x[1] * x[1]
        }
    }
    impl Gradient<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            array![10.0 * x[0], x[1]]
        }
    }
    impl DifferentiableObjective<f64> for Quad {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }
    impl HessianObjective for Quad {
        fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![10.0, 0.0, 0.0, 1.0]).unwrap()
        }
    }

    let obj = Quad;
    let x0 = array![1.0, -0.4];
    let mut x = x0.clone();
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            maxiter: 1,
            gtol: 1e-12,
            istep: 1.0,
            maxmove: None,
            ..Default::default()
        },
        2,
    );
    solver.set_qn_step(QnStep::Newton);
    solver.set_highs(true);
    assert!(solver.add_equality(vec![(0, 1.0), (1, 1.0)], 0.0));
    let first = solver.step_hess(&obj, &mut x).unwrap();
    assert!(first.grad_norm.is_finite());
    let dp0 = x[0] - x0[0];
    let dp1 = x[1] - x0[1];
    assert!(
        (dp0 + dp1).abs() < 1e-8,
        "Newton equality a·p=0 missed: p=({dp0}, {dp1})"
    );
}

#[test]
fn default_highs_options_choose_the_engine() {
    // Released default picks the engine. IPM with crossover off is opt-in.
    let options = rgmin::HighsOptions::default();
    assert_eq!(options.solver, rgmin::HighsSolverKind::Choose);
    assert_eq!(options.crossover, rgmin::HighsCrossover::Choose);
    let mut opt = Lbfgs::default();
    opt.highs_options.solver = rgmin::HighsSolverKind::Ipm;
    opt.highs_options.crossover = rgmin::HighsCrossover::Off;
    assert_eq!(opt.highs_options.solver, rgmin::HighsSolverKind::Ipm);
    assert_eq!(opt.highs_options.crossover, rgmin::HighsCrossover::Off);
}

#[test]
fn ipm_equality_projection_holds() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep {
        equalities: vec![(vec![(0, 1.0), (1, 1.0)], 0.0)],
        ..HighsStep::default()
    });
    opt.highs_options.solver = rgmin::HighsSolverKind::Ipm;
    opt.highs_options.crossover = rgmin::HighsCrossover::Off;
    let x0 = Array1::from(vec![1.0, -0.5, 0.25, 0.0]);
    let g0 = quad(x0.view()).1;
    let p = opt.highs_step(x0.view(), g0.view()).unwrap();
    assert!(p.iter().all(|v| v.is_finite()));
    assert!((p[0] + p[1]).abs() < 1e-8, "ipm a·p {}", p[0] + p[1]);
}

#[cfg(feature = "capi")]
#[test]
fn c_abi_set_highs_solver_is_closed() {
    use rgmin::ffi::{
        rgmin_control_t, rgmin_highs_crossover_t, rgmin_highs_solver_t, rgmin_method_t,
        rgmin_solver_create, rgmin_solver_free, rgmin_solver_set_highs,
        rgmin_solver_set_highs_crossover, rgmin_solver_set_highs_solver,
    };
    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 1e-8,
        istep: 0.1,
        memory: 4,
        maxmove: 0.0,
        noise_eps: 0.0,
        noise_c: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    assert_eq!(unsafe { rgmin_solver_set_highs(session, 1) }, 0);
    assert_eq!(
        unsafe {
            rgmin_solver_set_highs_solver(session, rgmin_highs_solver_t::RGMIN_HIGHS_IPM as i32)
        },
        0
    );
    assert_eq!(
        unsafe {
            rgmin_solver_set_highs_solver(session, rgmin_highs_solver_t::RGMIN_HIGHS_IPX as i32)
        },
        0
    );
    assert_eq!(
        unsafe {
            rgmin_solver_set_highs_crossover(
                session,
                rgmin_highs_crossover_t::RGMIN_HIGHS_CROSSOVER_OFF as i32,
            )
        },
        0
    );
    assert_eq!(unsafe { rgmin_solver_set_highs_solver(session, 99) }, 1);
    assert_eq!(unsafe { rgmin_solver_set_highs_crossover(session, 9) }, 1);
    unsafe { rgmin_solver_free(session) };
    let null = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            rgmin_solver_set_highs_solver(null, rgmin_highs_solver_t::RGMIN_HIGHS_IPM as i32)
        },
        1
    );
}

fn scipy_qp_gold(payload: &str) -> Vec<f64> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/highs_scipy_gold.py");
    let py = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut child = Command::new(&py)
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("{py} spawn: {e}"));
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(payload.as_bytes())
        .expect("write gold stdin");
    let out = child.wait_with_output().expect("gold wait");
    assert!(
        out.status.success(),
        "scipy gold failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("\"success\": true") || text.contains("\"success\":true"),
        "scipy gold not success: {text}"
    );
    let start = text.find("\"p\"").expect("p key");
    let lb = text[start..].find('[').expect("[") + start;
    let rb = text[lb..].find(']').expect("]") + lb;
    text[lb + 1..rb]
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("p i"))
        .collect()
}

#[test]
fn dest_ipm_matches_scipy_trust_constr() {
    let x0 = Array1::from(vec![1.0, -0.5, 0.25, 0.0]);
    let g0 = quad(x0.view()).1;
    let mut dest = Lbfgs::default();
    dest.highs = Some(HighsStep {
        equalities: vec![(vec![(0, 1.0), (1, 1.0)], 0.0)],
        ..HighsStep::default()
    });
    dest.highs_options.solver = rgmin::HighsSolverKind::Ipm;
    dest.highs_options.crossover = rgmin::HighsCrossover::Off;
    let p_dest = dest.highs_step(x0.view(), g0.view()).unwrap();
    let d = dest.two_loop(g0.view());
    let payload = format!(
        r#"{{"d":[{},{},{},{}],"A":[[1,1,0,0]],"b":[0]}}"#,
        d[0], d[1], d[2], d[3]
    );
    let p_scipy = scipy_qp_gold(&payload);
    assert_eq!(p_dest.len(), p_scipy.len());
    for i in 0..p_dest.len() {
        assert!(
            (p_dest[i] - p_scipy[i]).abs() < 1e-5,
            "coord {i}: dest {} scipy {}",
            p_dest[i],
            p_scipy[i]
        );
    }
}

#[test]
fn dest_ipm_box_equality_stays_inside_after_scaling() {
    let x = Array1::from(vec![0.0, 0.0, 0.0]);
    let g = Array1::from(vec![2.0, -1.0, 0.5]);
    let mut dest = Lbfgs::default();
    dest.highs = Some(HighsStep {
        lo: Some(-0.2),
        hi: Some(0.2),
        equalities: vec![(vec![(0, 1.0), (1, 1.0), (2, 1.0)], 0.0)],
        ..HighsStep::default()
    });
    dest.highs_options.solver = rgmin::HighsSolverKind::Ipm;
    dest.highs_options.crossover = rgmin::HighsCrossover::Off;
    // Uniform bounds scale the two-loop step before the equality projection.
    // That is not the unscaled scipy QP.
    let scaled = dest.highs_step(x.view(), g.view()).unwrap();
    assert!(scaled.iter().all(|v| v.is_finite()));
    assert!((scaled[0] + scaled[1] + scaled[2]).abs() < 1e-7);
    for i in 0..3 {
        assert!((x[i] + scaled[i]).abs() <= 0.2 + 1e-8);
    }
}

struct CbHits {
    n: std::sync::atomic::AtomicU32,
}

unsafe extern "C" fn count_highs_cb(
    _kind: i32,
    _message: *const std::os::raw::c_char,
    _interrupt: *mut i32,
    user: *mut std::os::raw::c_void,
) {
    if user.is_null() {
        return;
    }
    unsafe {
        (*(user as *const CbHits))
            .n
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[test]
fn highs_callback_fires_on_ipm_equality() {
    let hits = CbHits {
        n: std::sync::atomic::AtomicU32::new(0),
    };
    let mut dest = Lbfgs::default();
    dest.highs = Some(HighsStep {
        equalities: vec![(vec![(0, 1.0), (1, 1.0)], 0.0)],
        ..HighsStep::default()
    });
    dest.highs_options.solver = rgmin::HighsSolverKind::Ipm;
    dest.highs_options.crossover = rgmin::HighsCrossover::Off;
    dest.highs_options.callback = Some(unsafe {
        rgmin::HighsCallback::new(count_highs_cb, (&hits as *const CbHits).cast_mut().cast())
    });
    let x0 = Array1::from(vec![1.0, -0.5, 0.25, 0.0]);
    let g0 = quad(x0.view()).1;
    let p = dest.highs_step(x0.view(), g0.view()).unwrap();
    assert!(p.iter().all(|v| v.is_finite()));
    assert!(
        hits.n.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "HiGHS callback never ran"
    );
}
