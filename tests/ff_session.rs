//! Bonded metric on a quadratic whose Hessian is that metric.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rgmin::{BondTerm, Control, Ff, FfParams, FfTerms, Method, Solver};

struct ModelQuad<P> {
    apply: P,
    dim: usize,
    bounds: Bounds<f64>,
}

impl<P> Objective<f64> for ModelQuad<P>
where
    P: Fn(ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * x.dot(&(self.apply)(x))
    }
}

impl<P> Gradient<f64> for ModelQuad<P>
where
    P: Fn(ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.dim
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        (self.apply)(x)
    }
}

impl<P> DifferentiableObjective<f64> for ModelQuad<P>
where
    P: Fn(ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let g = (self.apply)(x);
        (0.5 * x.dot(&g), g)
    }
}

fn water() -> Array1<f64> {
    let ang = 104.5_f64.to_radians();
    ndarray::array![
        0.0,
        0.0,
        0.0,
        0.96,
        0.0,
        0.0,
        0.96 * ang.cos(),
        0.96 * ang.sin(),
        0.0
    ]
}

fn steps_on(
    mut solver: Solver,
    obj: &impl DifferentiableObjective<f64>,
    x0: &Array1<f64>,
) -> usize {
    solver.set_atom_maxmove(0.0);
    let mut x = x0.clone();
    let mut taken = 0;
    for _ in 0..40 {
        let g = obj.grad(x.view());
        if g.dot(&g).sqrt() < 1e-8 {
            break;
        }
        solver.step(obj, &mut x).unwrap();
        taken += 1;
    }
    taken
}

#[test]
fn ff_shortens_lbfgs_on_a_quadratic() {
    let geom = water();
    let bonds = [
        BondTerm { i: 0, j: 1, k: 1.0 },
        BondTerm { i: 0, j: 2, k: 1.0 },
    ];
    let params = FfParams {
        mu: Some(1.0),
        ..FfParams::default()
    };
    let terms = FfTerms {
        bonds: &bonds,
        ..FfTerms::default()
    };
    let ff = Ff::assemble(geom.view(), terms, &params).unwrap();
    let obj = ModelQuad {
        apply: |x: ArrayView1<f64>| ff.apply(x),
        dim: geom.len(),
        bounds: Bounds::new(Array1::from_elem(9, -1e6), Array1::from_elem(9, 1e6), 0.0),
    };
    let control = Control {
        maxiter: 1,
        gtol: 1e-8,
        istep: 1.0,
        maxmove: None,
        ..Control::default()
    };
    let plain = Solver::new(Method::Lbfgs { memory: 5 }, control.clone(), geom.len());
    let mut with = Solver::new(Method::Lbfgs { memory: 5 }, control, geom.len());
    with.set_ff(geom.view(), terms, &params).unwrap();
    let without = steps_on(plain, &obj, &geom);
    let taken = steps_on(with, &obj, &geom);
    assert!(taken < without, "ff {taken}, plain {without}");
    assert!(taken <= 3, "ff took {taken}");
}
