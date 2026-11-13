//! The opt-in relative-decrease stop of `Lbfgs` (`ftol`, L-BFGS-B `factr`).

use ndarray::{Array1, ArrayView1, array};
use rgmin::{LBFGSB_FACTR, LBFGSB_FTOL, Lbfgs, LbfgsStep};

fn wolfe() -> Lbfgs {
    let mut opt = Lbfgs::default();
    opt.step = LbfgsStep::Wolfe;
    opt
}

/// SciPy 1.18.1 `scipy/optimize/_lbfgsb_py.py`, the `ftol` default of
/// `_minimize_lbfgsb`, written there as `2.2204460492503131e-09`.
const SCIPY_FTOL: f64 = 2.220_446_049_250_313e-9;

/// Enough iterations that only a stop test ends the Rosenbrock runs.
const MAX_ITER: usize = 1000;

fn rosen(x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
    let a = 1.0 - x[0];
    let b = x[1] - x[0] * x[0];
    Some((
        a * a + 100.0 * b * b,
        array![-2.0 * a - 400.0 * x[0] * b, 200.0 * b],
    ))
}

fn start() -> Array1<f64> {
    array![-1.2, 1.0]
}

/// Values at each iterate the hook sees, and the run's result.
fn watched(opt: &mut Lbfgs) -> (Vec<f64>, (f64, Array1<f64>, usize)) {
    let mut seen = Vec::new();
    let out = opt.minimize_watched(start().view(), MAX_ITER, rosen, |_, f| {
        seen.push(f);
        true
    });
    (seen, out)
}

fn passes(ftol: f64, f_old: f64, f_new: f64) -> bool {
    f_old - f_new <= ftol * f_old.abs().max(f_new.abs()).max(1.0)
}

#[test]
fn the_constant_is_the_scipy_default() {
    assert_eq!(LBFGSB_FTOL, SCIPY_FTOL);
    assert_eq!(LBFGSB_FTOL / f64::EPSILON, LBFGSB_FACTR);
}

#[test]
fn the_test_is_off_by_default() {
    assert_eq!(Lbfgs::default().ftol, 0.0);
    let mut base = Lbfgs::default();
    let expected = base.minimize(start().view(), MAX_ITER, rosen);
    for ftol in [0.0, -LBFGSB_FTOL, f64::NAN] {
        let mut opt = Lbfgs::default();
        opt.ftol = ftol;
        assert_eq!(
            opt.minimize(start().view(), MAX_ITER, rosen),
            expected,
            "ftol {ftol}"
        );
    }
}

/// With the gradient test off, the run ends at the first iterate whose
/// step passes the relative-decrease test, and the hook sees that iterate.
#[test]
fn the_first_small_decrease_ends_the_relaxation() {
    let mut opt = wolfe();
    opt.gtol = 0.0;
    opt.ftol = LBFGSB_FTOL;
    let (seen, (f, _, evals)) = watched(&mut opt);
    eprintln!(
        "ftol run: {} iterates, {evals} evaluations, f = {f:e}",
        seen.len()
    );
    assert!(seen.len() < MAX_ITER);
    assert_eq!(*seen.last().unwrap(), f);
    let steps: Vec<(f64, f64)> = seen.windows(2).map(|w| (w[0], w[1])).collect();
    let (last, earlier) = steps.split_last().unwrap();
    assert!(passes(LBFGSB_FTOL, last.0, last.1));
    for &(f_old, f_new) in earlier {
        assert!(
            f_old == f_new || !passes(LBFGSB_FTOL, f_old, f_new),
            "an earlier step {f_old:e} -> {f_new:e} passed the test"
        );
    }

    let mut plain = wolfe();
    plain.gtol = 0.0;
    let (_, (f_plain, _, evals_plain)) = watched(&mut plain);
    eprintln!("gtol 0 run: {evals_plain} evaluations, f = {f_plain:e}");
    assert!(evals < evals_plain, "{evals} >= {evals_plain}");
    assert!(passes(LBFGSB_FTOL, f, f_plain));
}

#[test]
fn the_recognising_driver_stops_at_the_same_point() {
    let mut opt = wolfe();
    opt.gtol = 0.0;
    opt.ftol = LBFGSB_FTOL;
    let (f, x, evals) = opt.minimize(start().view(), MAX_ITER, rosen);
    let mut other = wolfe();
    other.gtol = 0.0;
    other.ftol = LBFGSB_FTOL;
    let (f_r, x_r, evals_r, stand_in) =
        other.minimize_recognized(start().view(), MAX_ITER, rosen, |_, _, _| None);
    assert!(!stand_in);
    assert_eq!((f, x, evals), (f_r, x_r, evals_r));
}

/// The box driver uses the same relative-decrease stop as `minimize`.
#[test]
fn the_box_driver_stops_on_the_relative_decrease() {
    let inf = Array1::from_elem(2, f64::INFINITY);
    let ninf = -inf.clone();
    let mut plain = Lbfgs::default();
    plain.gtol = 0.0;
    let (_, _, evals_plain) =
        plain.minimize_box(start().view(), ninf.view(), inf.view(), MAX_ITER, rosen);
    let mut opt = Lbfgs::default();
    opt.gtol = 0.0;
    opt.ftol = LBFGSB_FTOL;
    let mut seen = Vec::new();
    let (f, _, evals) = opt.minimize_box_watched(
        start().view(),
        ninf.view(),
        inf.view(),
        MAX_ITER,
        rosen,
        |_, value| {
            seen.push(value);
            true
        },
    );
    assert!(evals < evals_plain, "{evals} >= {evals_plain}");
    assert!(seen.len() >= 2, "iterates {seen:?}");
    let f_old = seen[seen.len() - 2];
    let f_new = *seen.last().unwrap();
    assert!(passes(LBFGSB_FTOL, f_old, f_new), "{f_old:e} -> {f_new:e}");
    assert_eq!(f_new, f);
}

/// Under the default gradient stop the test can only end a run earlier.
#[test]
fn with_the_gradient_test_it_never_costs_evaluations() {
    let mut plain = wolfe();
    let (_, _, evals_plain) = plain.minimize(start().view(), MAX_ITER, rosen);
    let mut opt = wolfe();
    opt.ftol = LBFGSB_FTOL;
    let (_, _, evals) = opt.minimize(start().view(), MAX_ITER, rosen);
    eprintln!("gtol 1e-6: {evals_plain} evaluations alone, {evals} with ftol");
    assert!(evals <= evals_plain);
}
