//! Scaled L-BFGS on the 10-D Rosenbrock start used by the bench suite.
//!
//! A recorded `rgmin-scaled` row stopped at energy about 4 with an
//! infinity-norm under 1e-5. That point is not the minimum. This test
//! is the session path the bench runs.

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, LbfgsStep, Method, Oracle, Solver, StopNorm};

fn rosenbrock(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mut f = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..x.len() - 1 {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        f += a * a + 100.0 * b * b;
        g[i] += -2.0 * a - 400.0 * x[i] * b;
        g[i + 1] += 200.0 * b;
    }
    (f, g)
}

fn relax(step: LbfgsStep) -> (f64, f64, usize) {
    let n = 10;
    let x0 = Array1::from_iter((0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }));
    let obj = Oracle::unbounded(n, rosenbrock);
    let control = Control {
        maxiter: 20_000,
        gtol: 1e-5,
        istep: 1.0,
        stop_norm: StopNorm::Inf,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 8 }, control, n);
    solver.set_lbfgs_step(step).unwrap();
    if matches!(step, LbfgsStep::Wolfe) {
        solver.set_atom_maxmove(0.0);
    }
    let mut x = x0;
    let report = solver.run(&obj, &mut x).unwrap();
    let (f, g) = rosenbrock(x.view());
    let ginf = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    (f, ginf, report.steps)
}

/// ASE's length control (shorten the step, keep the pairs) stops in the
/// well at `x0 ≈ -1`, where the infinity norm is below `1e-5` and the
/// value is about 4. The eOn cap clears that pair and reaches the
/// minimum at 0. SciPy's L-BFGS-B also reaches 0; the difference is the
/// step rule, not the tolerance.
#[test]
fn the_eon_cap_leaves_the_rosenbrock10_well_that_traps_the_scaled_step() {
    let (scaled, g_scaled, _) = relax(LbfgsStep::Scaled { max_atom: 0.2 });
    assert!(
        scaled > 1.0 && g_scaled < 1e-4,
        "scaled should stop in the well, f {scaled} g {g_scaled}"
    );
    let (capped, g_capped, steps) = relax(LbfgsStep::Capped { max_atom: 0.2 });
    assert!(
        capped < 1e-8 && g_capped < 1e-4,
        "eOn cap f {capped} g {g_capped} steps {steps}"
    );
    let (wolfe, g_wolfe, _) = relax(LbfgsStep::Wolfe);
    assert!(
        wolfe < 1e-8 && g_wolfe < 1e-4,
        "Wolfe f {wolfe} g {g_wolfe}"
    );
}
