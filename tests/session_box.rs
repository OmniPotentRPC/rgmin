//! Session coordinate boxes preserve the callback domain and free directions.

#![cfg(feature = "highs")]

use approx::assert_relative_eq;
use ndarray::{Array1, Array2};

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
    assert!(solver.set_box(Some(vec![f64::NEG_INFINITY, 0.0]), None));
    let mut x = array![3.0, 1e-8];
    let report = solver.step(&obj, &mut x).unwrap();
    assert_relative_eq!(x[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(x[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(report.value, 0.5, epsilon = 1e-12);
    assert!(report.grad_norm <= 1e-10, "{report:?}");
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
            ftol_rel: None,
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
        rgmin_solver_step, rgmin_status_t, rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
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
            ftol_rel: None,
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
            ftol_rel: None,
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
fn session_equality_and_trust_reach_the_constrained_model_solution() {
    use ndarray::array;
    use rgmin::{Control, Method, Oracle, Solver};
    let objective = Oracle::unbounded(2, |x| {
        let g = &x - &array![2.0, -1.0];
        (0.5 * g.dot(&g), g)
    });
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2);
    solver.set_highs(true);
    assert!(solver.set_trust(0.2));
    assert!(solver.add_equality(vec![(0, 1.0), (1, 1.0)], 0.0));
    let mut x = array![0.0, 0.0];
    let report = solver.step(&objective, &mut x).unwrap();
    assert_relative_eq!(x[0], 0.2, epsilon = 1e-7);
    assert_relative_eq!(x[1], -0.2, epsilon = 1e-7);
    assert_relative_eq!(report.value, 1.94, epsilon = 1e-7);
    assert!(solver.clear_equalities());
    assert!(solver.set_trust(0.0));
    solver.forget();
    let report = solver.step(&objective, &mut x).unwrap();
    assert_relative_eq!(x[0], 2.0, epsilon = 1e-10);
    assert_relative_eq!(x[1], -1.0, epsilon = 1e-10);
    assert!(report.grad_norm < 1e-10);
}

#[test]
fn a_rounded_wall_does_not_halve_the_free_step() {
    use ndarray::array;
    use rgmin::{Control, Method, Oracle, Solver};

    let wall = 0.02_f64.ln().next_up();
    for sign in [-1.0, 1.0] {
        for start in [0.32673629493102274, 0.32673630012041155] {
            let target = array![sign * wall, 0.0];
            let objective = Oracle::unbounded(2, |x| {
                let gradient = &x - &target;
                (0.5 * gradient.dot(&gradient), gradient)
            });
            let mut solver = Solver::new(
                Method::lbfgs(),
                Control {
                    istep: 1.0,
                    gtol: 1e-12,
                    ..Control::default()
                },
                2,
            );
            solver.set_highs(true);
            let lo = if sign > 0.0 { wall } else { f64::NEG_INFINITY };
            let hi = if sign < 0.0 { -wall } else { f64::INFINITY };
            assert!(solver.set_box(Some(vec![lo, -10.0]), Some(vec![hi, 10.0])));
            let mut x = array![sign * start, 1.0];
            let report = solver.step(&objective, &mut x).unwrap();
            assert!(x[0] >= lo && x[0] <= hi, "outside wall: {x:?}");
            assert_relative_eq!(x[0], target[0], epsilon = 1e-14);
            assert_eq!(x[1], 0.0, "free step was shortened at start {start}");
            assert!(report.grad_norm <= 1e-12, "{report:?}");
        }
    }
}
