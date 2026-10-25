//! Geometry L-BFGS and the Newton / dogleg saddle kick.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::{Control, HessianOracle, Method, NewtonKind, Solver};

fn bowl(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let f = 0.5 * x.dot(&x);
    (f, x.to_owned())
}

fn well(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let y = x[1];
    let f = (y * y - 1.0).powi(2) + 0.5 * x[0] * x[0];
    let g = array![x[0], 4.0 * y * (y * y - 1.0)];
    (f, g)
}

fn well_h(x: ArrayView1<f64>) -> Array2<f64> {
    let y = x[1];
    Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 12.0 * y * y - 4.0]).unwrap()
}

fn control(gtol: f64) -> Control {
    Control {
        maxiter: 80,
        gtol,
        istep: 0.2,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    }
}

#[test]
fn geometry_lbfgs_reaches_the_bowl_with_a_per_atom_cap() {
    let obj = rgmin::Oracle::unbounded(3, bowl);
    let mut solver = Solver::new(Method::lbfgs(), control(1e-8), 3);
    solver.set_atom_maxmove(0.2);
    solver.set_geometry_step(true);
    let mut x = array![1.2, -0.4, 0.7];
    let mut last = f64::MAX;
    for _ in 0..80 {
        let rep = solver.step(&obj, &mut x).expect("step");
        last = rep.grad_norm;
        if last <= 1e-8 {
            break;
        }
    }
    assert!(last <= 1e-8, "geometry L-BFGS |g|={last} x={x:?}");
    assert!(x.iter().all(|v| v.abs() < 1e-6), "x={x:?}");
}

#[test]
fn geometry_lbfgs_probes_only_severely_clipped_starts() {
    for (start, expected_calls) in [(1.0, 2), (10_000.0, 3)] {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let obj = rgmin::Oracle::unbounded(3, move |x: ArrayView1<f64>| {
            seen.fetch_add(1, Ordering::Relaxed);
            bowl(x)
        });
        let mut solver = Solver::new(Method::lbfgs(), control(1e-12), 3);
        solver.set_atom_maxmove(0.2);
        solver.set_geometry_step(true);
        let mut x = array![start, 0.0, 0.0];
        solver.step(&obj, &mut x).expect("step");
        assert_eq!(calls.load(Ordering::Relaxed), expected_calls);
    }
}

#[test]
fn dogleg_and_newton_leave_a_saddle() {
    let obj = HessianOracle::unbounded(2, well, well_h);
    let gtol = 1e-8;
    let methods = [
        ("dogleg", Method::Dogleg),
        (
            "newton",
            Method::Newton {
                kind: NewtonKind::Shifted,
            },
        ),
    ];
    for (name, method) in methods {
        let mut solver = Solver::new(method, control(gtol), 2);
        solver.set_atom_maxmove(0.25);
        solver.set_geometry_step(true);
        let mut x = array![0.0, 0.0];
        let first = solver.step_hess(&obj, &mut x).expect("kick");
        assert!(
            first.grad_norm > gtol,
            "{name} stayed on the saddle |g|={} x={x:?}",
            first.grad_norm
        );
        assert!(x[1].abs() > 0.2, "{name} kick was {x:?}");
        let mut last = first.grad_norm;
        for _ in 0..60 {
            if last <= gtol {
                break;
            }
            let rep = solver.step_hess(&obj, &mut x).expect("step");
            last = rep.grad_norm;
        }
        assert!(last <= gtol, "{name} |g|={last} x={x:?}");
        assert!(
            (x[1].abs() - 1.0).abs() < 1e-4 && x[0].abs() < 1e-4,
            "{name} ended at {x:?}"
        );
    }
}

#[test]
fn geometry_lbfgs_clips_the_finite_difference_probe() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use std::sync::{Mutex, OnceLock};

    struct Tight {
        seen: Mutex<Vec<f64>>,
    }
    impl Objective<f64> for Tight {
        fn dim(&self) -> usize {
            1
        }
        fn bounds(&self) -> &Bounds<f64> {
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![0.0], array![0.004], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.value_and_gradient(x).0
        }
    }
    impl Gradient<f64> for Tight {
        fn dim(&self) -> usize {
            1
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            self.value_and_gradient(x).1
        }
    }
    impl DifferentiableObjective<f64> for Tight {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            let x0 = x[0];
            assert!(
                (0.0..=0.004).contains(&x0),
                "finite-difference probe left the box at {x0}"
            );
            self.seen.lock().expect("seen").push(x0);
            (0.5 * (x0 - 1.0) * (x0 - 1.0), array![x0 - 1.0])
        }
    }

    let obj = Tight {
        seen: Mutex::new(Vec::new()),
    };
    let mut solver = Solver::new(Method::lbfgs(), control(1e-8), 1);
    solver.set_atom_maxmove(0.2);
    solver.set_geometry_step(true);
    let mut x = array![0.0];
    solver.step(&obj, &mut x).expect("step");
    let seen = obj.seen.lock().expect("seen").clone();
    assert!(
        seen.iter().any(|v| (*v - 0.004).abs() < 1e-15),
        "clipped probe 0.004 was not evaluated: {seen:?}"
    );
    assert!(
        seen.iter().all(|v| *v <= 0.004 + 1e-15),
        "probe outside the box: {seen:?}"
    );
}

#[test]
fn geometry_lbfgs_resets_on_negative_curvature() {
    let obj = rgmin::Oracle::unbounded(1, |x| {
        let y = x[0];
        (-0.5 * y * y, array![-y])
    });
    let mut solver = Solver::new(Method::lbfgs(), control(1e-8), 1);
    assert!(solver.push_pair(array![0.1].view(), array![0.1].view()));
    assert_eq!(solver.pair_count(), 1);
    solver.set_atom_maxmove(0.2);
    solver.set_geometry_step(true);
    let mut x = array![1.0];
    solver.step(&obj, &mut x).expect("step");
    assert_eq!(
        solver.pair_count(),
        0,
        "negative curvature left a pair in memory"
    );
    assert!(
        (x[0] - 1.1).abs() < 1e-12,
        "reset step did not use the ordinary-start scale, x={x:?}"
    );
}

#[test]
fn geometry_lbfgs_falls_back_after_ten_halvings() {
    use std::sync::atomic::{AtomicU32, Ordering};

    let calls = AtomicU32::new(0);
    let obj = rgmin::Oracle::unbounded(1, |x| {
        calls.fetch_add(1, Ordering::Relaxed);
        let y = x[0];
        let value = if (y - 1.0).abs() < 1e-12 { 0.0 } else { 1.0 };
        (value, array![y])
    });
    let mut solver = Solver::new(Method::lbfgs(), control(1e-8), 1);
    solver.set_atom_maxmove(0.5);
    solver.set_geometry_step(true);
    let mut x = array![1.0];
    solver.step(&obj, &mut x).expect("step");
    assert!(
        (x[0] - 0.9).abs() < 1e-12,
        "fallback was not the short steepest step, x={x:?}"
    );
    assert!(
        calls.load(Ordering::Relaxed) > 10,
        "halving did not run, calls={}",
        calls.load(Ordering::Relaxed)
    );
}
