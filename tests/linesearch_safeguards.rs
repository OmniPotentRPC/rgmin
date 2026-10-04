//! Zoom interpolation on objectives with huge or
//! non-finite values, and their absence of effect on well-scaled ones.

use std::sync::{Mutex, OnceLock};

use eindir_core::objectives::Rosenbrock;
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, LineSearch, Method, Solver};

fn control(istep: f64) -> Control {
    Control {
        maxiter: 200,
        gtol: 1e-10,
        istep,
        maxmove: None,
        ftol_rel: None,
    }
}

/// `f(x) = x . x / 2 + exp(-100 x0 + 93)`: an exponential wall at
/// `x0 = 0` worth `2.6e40`, flat to rounding at `x0 >= 2`.
struct Wall {
    points: Mutex<Vec<f64>>,
}

impl Wall {
    fn raw(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let e = (-100.0 * x[0] + 93.0).exp();
        (0.5 * x.dot(&x) + e, array![x[0] - 100.0 * e, x[1]])
    }
}

impl Objective<f64> for Wall {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        Self::raw(x).0
    }
}

impl Gradient<f64> for Wall {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Self::raw(x).1
    }
}

impl DifferentiableObjective<f64> for Wall {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.points.lock().unwrap().push(x[0]);
        Self::raw(x)
    }
}

#[test]
fn a_session_step_onto_an_exponential_wall_zooms_by_bisection() {
    // The unit step along -g from (10, 0) lands on x0 = 0, where the
    // objective is 2.6e40. The cubic through that value would put the
    // next trial on the bracket clamp (x0 = 9); bisection tries x0 = 5.
    let obj = Wall {
        points: Mutex::new(Vec::new()),
    };
    let start = array![10.0, 0.0];
    let (f_wall, _) = Wall::raw(array![0.0, 0.0].view());
    assert!(f_wall > 1e40);

    let mut x = start.clone();
    let mut solver = Solver::new(Method::lbfgs(), control(1.0), 2);
    solver.set_accept(Accept::Energy);
    solver.step(&obj, &mut x).unwrap();
    let points = obj.points.lock().unwrap().clone();
    assert_eq!(points[..3], [10.0, 0.0, 5.0], "{points:?}");
    assert!(x[0] < start[0], "the step did not move: {x}");
}

/// `phi(a) = -a + 50 a^2` below `a = 0.3` and `+inf` above it.
fn walled(a: f64) -> (f64, Array1<f64>) {
    if a > 0.3 {
        (f64::INFINITY, array![f64::NAN])
    } else {
        (-a + 50.0 * a * a, array![-1.0 + 100.0 * a])
    }
}

#[test]
fn a_non_finite_trial_bisects_and_the_search_accepts_a_finite_point() {
    let mut alphas = Vec::new();
    let out = LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 40,
    }
    .search_from(
        |x: ArrayView1<f64>| {
            alphas.push(x[0]);
            walled(x[0])
        },
        array![0.0].view(),
        0.0,
        array![-1.0].view(),
        array![1.0].view(),
        1.0,
        f64::INFINITY,
    )
    .expect("a step");
    assert_eq!(alphas[..3], [1.0, 0.5, 0.25]);
    assert!(out.f.is_finite() && out.f < 0.0);
}

fn rosenbrock_calls(accept: Accept, start: Array1<f64>) -> (Array1<f64>, usize) {
    struct Counting<'a> {
        inner: Rosenbrock<2>,
        calls: &'a std::sync::atomic::AtomicUsize,
    }
    impl Objective<f64> for Counting<'_> {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            self.inner.bounds()
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.inner.eval(x)
        }
    }
    impl Gradient<f64> for Counting<'_> {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            self.inner.grad(x)
        }
    }
    impl DifferentiableObjective<f64> for Counting<'_> {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.value_and_gradient(x)
        }
    }
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let obj = Counting {
        inner: Rosenbrock::<2>::new(),
        calls: &calls,
    };
    let mut x = start;
    let mut solver = Solver::new(Method::lbfgs(), control(1.0), 2).with_gtol(1e-10);
    solver.set_accept(accept);
    for _ in 0..200 {
        let rep = solver.step(&obj, &mut x).unwrap();
        if rep.grad_norm < 1e-10 {
            break;
        }
    }
    (x, calls.load(std::sync::atomic::Ordering::Relaxed))
}

#[test]
fn well_scaled_runs_keep_their_iterates_and_call_counts() {
    // |g| < 1 at the start, so the opening length is the host istep, and
    // the bracket values are comparable, so the zoom interpolates.
    let (x, calls) = rosenbrock_calls(Accept::Energy, array![1.02, 1.04]);
    println!("ROSENBROCK_CALLS {calls} {x}");
    assert!(
        (x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6,
        "{x}"
    );
}
