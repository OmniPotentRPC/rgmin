#![cfg(feature = "highs")]

use ndarray::{Array1, ArrayView1};
use rgmin::{HighsStep, Lbfgs};

#[test]
fn automatic_engine_and_crossover_preserve_main_defaults() {
    let options = rgmin::HighsOptions::default();
    assert_eq!(options.solver, rgmin::HighsSolverKind::Choose);
    assert_eq!(options.crossover, rgmin::HighsCrossover::Choose);
    assert!(options.callback.is_none());
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
fn ipm_equality_projection_holds() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep {
        equalities: vec![(vec![(0, 1.0), (1, 1.0)], 0.0)],
        ..HighsStep::default()
    });
    opt.highs_options = rgmin::HighsOptions { solver: rgmin::HighsSolverKind::Ipm,
        crossover: rgmin::HighsCrossover::Off, ..rgmin::HighsOptions::default() };
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
    dest.highs_options = rgmin::HighsOptions { solver: rgmin::HighsSolverKind::Ipm,
        crossover: rgmin::HighsCrossover::Off, ..rgmin::HighsOptions::default() };
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
fn dest_ipm_box_equality_matches_scipy() {
    let x = Array1::from(vec![0.0, 0.0, 0.0]);
    let g = Array1::from(vec![2.0, -1.0, 0.5]);
    let mut dest = Lbfgs::default();
    dest.highs = Some(HighsStep {
        lo: Some(-0.2),
        hi: Some(0.2),
        equalities: vec![(vec![(0, 1.0), (1, 1.0), (2, 1.0)], 0.0)],
        ..HighsStep::default()
    });
    dest.highs_options = rgmin::HighsOptions { solver: rgmin::HighsSolverKind::Ipm,
        crossover: rgmin::HighsCrossover::Off, ..rgmin::HighsOptions::default() };
    let p_dest = rgmin::lbfgs_qp::highs_projected_step(
        &dest.two_loop(g.view()), x.view(), dest.highs.as_ref().unwrap(), &dest.highs_options).unwrap();
    let d = dest.two_loop(g.view());
    let payload = format!(
        r#"{{"d":[{},{},{}],"lo":[-0.2,-0.2,-0.2],"hi":[0.2,0.2,0.2],"A":[[1,1,1]],"b":[0]}}"#,
        d[0], d[1], d[2]
    );
    let p_scipy = scipy_qp_gold(&payload);
    for i in 0..3 {
        assert!(
            (p_dest[i] - p_scipy[i]).abs() < 2e-5,
            "coord {i}: dest {} scipy {}",
            p_dest[i],
            p_scipy[i]
        );
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
    let x0 = Array1::from(vec![1.0, -0.5, 0.25, 0.0]);
    let g0 = quad(x0.view()).1;
    dest.highs_options = rgmin::HighsOptions { solver: rgmin::HighsSolverKind::Ipm,
        crossover: rgmin::HighsCrossover::Off,
        callback: Some(unsafe { rgmin::HighsCallback::new(count_highs_cb,
            (&hits as *const CbHits).cast_mut().cast()) }) };
    let p = dest.highs_step(x0.view(), g0.view()).unwrap();
    assert!(p.iter().all(|v| v.is_finite()));
    assert!(
        hits.n.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "HiGHS callback never ran"
    );
}

#[test]
fn session_routes_and_clears_callback_for_first_and_second_order_steps() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::{Array2, array};
    use rgmin::{Accept, Control, HessianObjective, Method, QnStep, Solver};
    use std::sync::atomic::Ordering;

    struct Quad;
    impl Objective<f64> for Quad {
        fn dim(&self) -> usize { 2 }
        fn bounds(&self) -> &Bounds<f64> {
            static BOUNDS: std::sync::OnceLock<Bounds<f64>> = std::sync::OnceLock::new();
            BOUNDS.get_or_init(|| Bounds::new(array![-10.0, -10.0], array![10.0, 10.0], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 { 0.5 * x.dot(&x) }
    }
    impl Gradient<f64> for Quad {
        fn dim(&self) -> usize { 2 }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> { x.to_owned() }
    }
    impl DifferentiableObjective<f64> for Quad {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }
    impl HessianObjective for Quad {
        fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> { Array2::eye(2) }
    }
    for second_order in [false, true] {
        let hits = CbHits { n: std::sync::atomic::AtomicU32::new(0) };
        let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2);
        solver.set_accept(Accept::Step);
        assert!(solver.set_highs_solver(rgmin::HighsSolverKind::Ipm));
        assert!(solver.set_highs_crossover(rgmin::HighsCrossover::Off));
        assert!(solver.set_highs_callback(Some(unsafe {
            rgmin::HighsCallback::new(count_highs_cb, (&hits as *const CbHits).cast_mut().cast())
        })));
        solver.set_highs(true);
        if second_order { solver.set_qn_step(QnStep::Newton); }
        assert!(solver.add_equality(vec![(0, 1.0), (1, 1.0)], 0.0));
        let mut x = array![2.0, -1.0];
        let start = x.clone();
        let report = if second_order { solver.step_hess(&Quad, &mut x) }
            else { solver.step(&Quad, &mut x) }.unwrap();
        let called = hits.n.load(Ordering::Relaxed);
        assert!(called > 0, "callback did not reach the session QP");
        assert_eq!(report.steps, 1);
        assert!(((x[0]-start[0]) + (x[1]-start[1])).abs() < 1e-7);
        assert!(report.value < 0.5 * start.dot(&start));

        assert!(solver.set_highs_callback(None));
        x = start;
        let report = if second_order { solver.step_hess(&Quad, &mut x) }
            else { solver.step(&Quad, &mut x) }.unwrap();
        assert_eq!(report.steps, 2);
        assert_eq!(hits.n.load(Ordering::Relaxed), called, "cleared context was called");
        assert!((x.sum() - 1.0).abs() < 1e-7);
    }
}

#[test]
fn exact_projection_respects_packed_center_and_zero_radius() {
    use ndarray::array;
    let d = array![1.0, 3.0];
    let x = array![0.0, 0.0];
    let options = rgmin::HighsOptions::default();
    let mut constraints = HighsStep { center_axes: Some((2, 1)), ..HighsStep::default() };
    let p = rgmin::lbfgs_qp::highs_projected_step(&d, x.view(), &constraints, &options).unwrap();
    assert!((p[0] + 1.0).abs() < 1e-7 && (p[1] - 1.0).abs() < 1e-7, "{p:?}");
    constraints.trust = Some(0.0);
    let p = rgmin::lbfgs_qp::highs_projected_step(&d, x.view(), &constraints, &options).unwrap();
    assert_eq!(p, array![0.0, 0.0]);
    constraints.center_axes = Some((3, 1));
    assert!(rgmin::lbfgs_qp::highs_projected_step(&d, x.view(), &constraints, &options).is_err());
}

#[test]
fn interrupting_the_linear_model_keeps_the_accepted_point() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::Array2;
    use rgmin::{Control, Error, HessianObjective, Method, QnStep, Solver};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Linear {
        bounds: Bounds<f64>, gradient: Array1<f64>, calls: AtomicUsize,
    }
    impl Objective<f64> for Linear {
        fn dim(&self) -> usize { self.gradient.len() }
        fn bounds(&self) -> &Bounds<f64> { &self.bounds }
        fn eval(&self, x: ArrayView1<f64>) -> f64 { self.gradient.dot(&x) }
    }
    impl Gradient<f64> for Linear {
        fn dim(&self) -> usize { self.gradient.len() }
        fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> { self.gradient.clone() }
    }
    impl DifferentiableObjective<f64> for Linear {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            self.calls.fetch_add(1, Ordering::Relaxed);
            (self.eval(x), self.grad(x))
        }
    }
    impl HessianObjective for Linear {
        fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
            Array2::zeros((self.gradient.len(), self.gradient.len()))
        }
    }
    unsafe extern "C" fn interrupt_solver(
        kind: i32, _message: *const std::os::raw::c_char,
        interrupt: *mut i32, user: *mut std::os::raw::c_void,
    ) {
        if kind == 1 || kind == 2 {
            unsafe {
                (*(user as *const AtomicUsize)).fetch_add(1, Ordering::Relaxed);
                *interrupt = 1;
            }
        }
    }
    let n = 48;
    let objective = Linear {
        bounds: Bounds::new(Array1::from_elem(n, -1.0), Array1::from_elem(n, 1.0), 0.0),
        gradient: Array1::from_iter((0..n).map(|j| ((j*13+1)%19) as f64 - 9.0)),
        calls: AtomicUsize::new(0),
    };
    let hits = AtomicUsize::new(0);
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), n);
    solver.set_qn_step(QnStep::Newton);
    solver.set_highs(true);
    assert!(solver.set_highs_solver(rgmin::HighsSolverKind::Ipm));
    assert!(solver.set_highs_crossover(rgmin::HighsCrossover::Off));
    assert!(solver.set_trust(0.1));
    for row in 0..12 {
        let coefficients = (0..n).map(|j| (j, (((row+1)*(j+3))%37) as f64 - 18.0)).collect();
        assert!(solver.add_equality(coefficients, 0.0));
    }
    assert!(solver.set_highs_callback(Some(unsafe {
        rgmin::HighsCallback::new(interrupt_solver, (&hits as *const AtomicUsize).cast_mut().cast())
    })));
    let mut point = Array1::zeros(n);
    let start = point.clone();
    let result = solver.step_hess(&objective, &mut point);
    assert!(matches!(result, Err(Error::Highs(_))), "{result:?}");
    assert!(hits.load(Ordering::Relaxed) > 0, "interrupt callback was not invoked");
    assert_eq!(point, start);
    assert_eq!(solver.pair_count(), 0);
    assert_eq!(objective.calls.load(Ordering::Relaxed), 1);
    assert!(solver.set_highs_callback(None));
    let report = solver.step_hess(&objective, &mut point).unwrap();
    assert!(report.value < 0.0, "{report:?}");
    assert!(point.iter().all(|v| v.abs() <= 0.1+1e-7));
}
