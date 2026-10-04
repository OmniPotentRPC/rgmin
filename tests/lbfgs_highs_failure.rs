#![cfg(feature = "highs")]

use ndarray::{ArrayView1, array};
use rgmin::{HighsStep, Lbfgs};
use std::cell::Cell;

fn contradictory() -> Lbfgs {
    let mut opt = Lbfgs::default();
    opt.record(array![1.0, 0.0], array![2.0, 0.0]);
    opt.highs = Some(HighsStep {
        equalities: vec![(vec![(0, 1.0)], 0.0), (vec![(0, 1.0)], 1.0)],
        ..HighsStep::default()
    });
    opt
}

#[test]
fn infeasible_equalities_return_the_qp_error() {
    let opt = contradictory();
    let x = array![1.0, 2.0];
    let result = opt.highs_step(x.view(), x.view());
    assert!(result.is_err(), "infeasible model returned {result:?}");
    assert_eq!(opt.len(), 1);
}

#[test]
fn callback_minimizers_do_not_take_an_unconstrained_fallback_step() {
    for recognized in [false, true] {
        let mut opt = contradictory();
        let initial = array![1.0, 2.0];
        let calls = Cell::new(0usize);
        let objective = |x: ArrayView1<f64>| {
            calls.set(calls.get() + 1);
            Some((0.5 * x.dot(&x), x.to_owned()))
        };
        let (value, x, count) = if recognized {
            let (value, x, count, reused) = opt.minimize_recognized(initial.view(), 4, objective, |_, _, _| None);
            assert!(!reused);
            (value, x, count)
        } else {
            opt.minimize_watched(initial.view(), 4, objective, |_, _| true)
        };
        assert_eq!(x, initial);
        assert_eq!(value, 2.5);
        assert_eq!(count, 1);
        assert_eq!(calls.get(), 1);
        assert_eq!(opt.len(), 1);
    }
}
