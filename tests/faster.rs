//! L-BFGS should take fewer Rosenbrock steps than steepest descent.

use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use rgmin::{Control, LineSearch, Method, minimize_method};

#[test]
fn lbfgs_fewer_steps_than_steepest_on_rosenbrock() {
    let obj = Rosenbrock::<2>::new();
    let ctrl = Control {
        maxiter: 200,
        gtol: 1e-8,
        istep: 0.1,
        maxmove: None,
        ftol_rel: None,
    };
    let ls = LineSearch::Brent {
        maxiter: 40,
        tol: 1e-12,
    };
    let lbfgs = minimize_method(&obj, array![-1.2, 1.0], &ctrl, Method::lbfgs(), ls).unwrap();
    let sd = minimize_method(&obj, array![-1.2, 1.0], &ctrl, Method::Steepest, ls).unwrap();
    assert!(lbfgs.value < 1e-8);
    assert!(
        lbfgs.steps < sd.steps,
        "L-BFGS steps {} vs steepest {}",
        lbfgs.steps,
        sd.steps
    );
}
