//! `forget_evaluation` drops the cached point and keeps method memory.
//!
//! Accept::None is one oracle at the trial. A zero gradient stops when
//! its norm is under the tolerance.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, FireKind, Method, Solver};

struct Flipping {
    flip: AtomicBool,
    calls: AtomicUsize,
    bounds: Bounds<f64>,
}

impl Objective<f64> for Flipping {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Flipping {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for Flipping {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut g = x.to_owned();
        if self.flip.load(Ordering::Relaxed) {
            g[0] = -g[0];
        }
        (0.5 * x.dot(&x), g)
    }
}

#[test]
fn forget_evaluation_re_evaluates_and_keeps_memory() {
    let obj = Flipping {
        flip: AtomicBool::new(false),
        calls: AtomicUsize::new(0),
        bounds: Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0),
    };
    for method in [
        Method::Lbfgs { memory: 5 },
        Method::Fire { kind: FireKind::V2 },
    ] {
        let mut solver = Solver::new(
            method.clone(),
            Control {
                maxiter: 10,
                gtol: 1e-12,
                istep: 0.1,
                maxmove: Some(0.05),
                ..Control::default()
            },
            2,
        );
        obj.flip.store(false, Ordering::Relaxed);
        let mut x = array![1.0, 0.5];
        solver.step(&obj, &mut x).unwrap();
        solver.step(&obj, &mut x).unwrap();
        let kept = if matches!(method, Method::Lbfgs { .. }) {
            solver.pair_count() as f64
        } else {
            1.0
        };
        assert!(kept > 0.0, "{method:?} built no memory");
        obj.flip.store(true, Ordering::Relaxed);
        solver.forget_evaluation();
        if matches!(method, Method::Lbfgs { .. }) {
            assert_eq!(solver.pair_count() as f64, kept);
        }
        let before = obj.calls.load(Ordering::Relaxed);
        let start = x.clone();
        solver.step(&obj, &mut x).unwrap();
        let refreshed = obj.calls.load(Ordering::Relaxed) - before;
        // L-BFGS takes one call at the current x and one at the trial.
        // FIRE 2.0 may also refresh the force after a rollback.
        if matches!(method, Method::Lbfgs { .. }) {
            assert_eq!(refreshed, 2, "{method:?}");
        } else {
            assert!(refreshed >= 2, "{method:?} calls {refreshed}");
        }
        let before = obj.calls.load(Ordering::Relaxed);
        solver.step(&obj, &mut x).unwrap();
        let cached = obj.calls.load(Ordering::Relaxed) - before;
        assert!(
            cached < refreshed,
            "{method:?} cached {cached} vs {refreshed}"
        );
        assert!(x != start);
    }
}
