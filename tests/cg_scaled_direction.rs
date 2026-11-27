//! Accept::Step multiplies the whole conjugate direction by istep.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Conjugacy, Control, Method, Oracle, Solver};

fn grad(x: ArrayView1<f64>) -> Array1<f64> {
    array![x[0], 4.0 * x[1]]
}

#[test]
fn hestenes_stiefel_keeps_one_scale_when_the_step_is_fixed() {
    // g = (x, 4 y). istep = 1.2 stores d = -1.2 g. Hestenes-Stiefel
    // divides by d, so beta has to see the unscaled direction. The next
    // Accept::Step move is 1.2 * (-g + beta d_unscaled).
    let istep = 1.2;
    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let g = grad(x);
        (0.5 * x[0] * x[0] + 2.0 * x[1] * x[1], g)
    });
    let control = Control::default()
        .with_istep(istep)
        .with_gtol(1e-14)
        .with_maxmove(Some(-1.0));
    let mut solver = Solver::new(Method::nlcg(Conjugacy::HestenesStiefel), control, 2);
    solver.set_accept(Accept::Step);
    let mut x = array![1.0, 1.0];
    solver.step(&obj, &mut x).expect("first step");
    let g0 = array![1.0, 4.0];
    let d_unscaled = array![-g0[0], -g0[1]];
    let x1 = array![1.0 + istep * d_unscaled[0], 1.0 + istep * d_unscaled[1]];
    assert!(
        (x[0] - x1[0]).abs() <= 1e-12 && (x[1] - x1[1]).abs() <= 1e-12,
        "first step {x:?} want {x1:?}"
    );
    let g1 = grad(x.view());
    let y = &g1 - &g0;
    let beta = g1.dot(&y) / y.dot(&d_unscaled);
    let dir = array![
        istep * (-g1[0] + beta * d_unscaled[0]),
        istep * (-g1[1] + beta * d_unscaled[1]),
    ];
    solver.step(&obj, &mut x).expect("second step");
    assert!(
        (x[0] - (x1[0] + dir[0])).abs() <= 1e-9 && (x[1] - (x1[1] + dir[1])).abs() <= 1e-9,
        "second step {x:?} want {:?}",
        &x1 + &dir
    );
}
