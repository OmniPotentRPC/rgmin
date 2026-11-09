//! Relative value window for an explicit Wolfe search.
//!
//! Accept::None stays one oracle. The window does not change that step.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::linesearch::conditions::approximate_strong_wolfe;
use rgmin::{Control, LineSearch, LineSearchOptions, Method, Solver};

const START: f64 = 1.0 / 4096.0;
const NOISE: f64 = 1.0 / 16777216.0;

struct NoisyBowl {
    bounds: Bounds<f64>,
}

impl Objective<f64> for NoisyBowl {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if x[0] == START {
            100.0
        } else {
            100.0 + NOISE + 0.5 * x[0] * x[0]
        }
    }
}

impl Gradient<f64> for NoisyBowl {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![x[0]]
    }
}

impl DifferentiableObjective<f64> for NoisyBowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn wolfe() -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 20,
    }
}

#[test]
fn the_window_accepts_a_small_rise_and_refuses_a_large_one() {
    let accept = approximate_strong_wolfe;
    assert!(accept(100.0 + NOISE, 100.0, 0.0, -1.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(100.0 + 2e-6, 100.0, 0.0, -1.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(100.0 + NOISE, 100.0, -1.0, -1.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(100.0 + NOISE, 100.0, 1.0, -1.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(100.0, 100.0, 0.0, 0.0, 1e-4, 0.9, 1e-8));
    assert!(!accept(100.0 + NOISE, 100.0, 0.0, -1.0, 1e-4, 0.9, 0.0));
}

#[test]
fn invalid_values_leave_the_stored_window() {
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 1);
    assert!(solver.set_objective_roundoff(1e-8));
    for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!solver.set_objective_roundoff(value));
        assert!(LineSearchOptions::with_objective_roundoff(value).is_none());
    }
}

#[test]
fn accept_none_ignores_the_window() {
    let obj = NoisyBowl {
        bounds: Bounds::new(array![-1.0], array![1.0], 0.0),
    };
    let mut plain = Solver::new(
        Method::lbfgs(),
        Control {
            gtol: 1e-12,
            istep: 1.0,
            ..Control::default()
        },
        1,
    );
    let mut windowed = Solver::new(
        Method::lbfgs(),
        Control {
            gtol: 1e-12,
            istep: 1.0,
            ..Control::default()
        },
        1,
    );
    assert!(windowed.set_objective_roundoff(1e-8));
    let mut a = array![START];
    let mut b = array![START];
    let ra = plain.step(&obj, &mut a).unwrap();
    let rb = windowed.step(&obj, &mut b).unwrap();
    assert_eq!(a[0].to_bits(), b[0].to_bits());
    assert_eq!(ra.grad_norm.to_bits(), rb.grad_norm.to_bits());
}

#[test]
fn an_explicit_wolfe_search_uses_the_window() {
    let obj = NoisyBowl {
        bounds: Bounds::new(array![-1.0], array![1.0], 0.0),
    };
    let options = LineSearchOptions::with_objective_roundoff(1e-8).unwrap();
    for (direction, initial) in [(START, 1.0), (-START, 2.0)] {
        let result = wolfe()
            .search_from_with_options(
                |x| obj.value_and_gradient(x),
                array![START].view(),
                100.0,
                array![START].view(),
                array![direction].view(),
                initial,
                2.0,
                options,
            )
            .expect("windowed wolfe");
        assert!(result.g[0].abs() <= 0.9 * START, "g {}", result.g[0]);
        assert!(result.f - 100.0 <= 1e-6, "f {}", result.f);
        assert!(result.alpha > 0.0 && result.alpha <= 2.0);
        if initial == 2.0 {
            assert!(result.evals > 1, "evals {}", result.evals);
        }
    }
}
