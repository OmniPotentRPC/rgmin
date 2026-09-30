//! L-BFGS, Polak-Ribiere NLCG, and steepest descent on Rosenbrock 2D.

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
    let start = array![-1.2, 1.0];
    let lbfgs = minimize_method(&obj, start.clone(), &ctrl, Method::lbfgs(), ls).unwrap();
    let nlcg = minimize_method(&obj, start.clone(), &ctrl, Method::polak_ribiere(), ls).unwrap();
    let sd = minimize_method(&obj, start, &ctrl, Method::Steepest, ls).unwrap();
    eprintln!("L-BFGS steps {} value {}", lbfgs.steps, lbfgs.value);
    eprintln!(
        "Polak-Ribiere NLCG steps {} value {}",
        nlcg.steps, nlcg.value
    );
    eprintln!("steepest descent steps {} value {}", sd.steps, sd.value);
    assert!(
        lbfgs.steps < sd.steps,
        "L-BFGS steps {} vs steepest {}",
        lbfgs.steps,
        sd.steps
    );
}
