//! Thin Grassmann, unitary, and factor-shape setters.
//!
//! They call `set_manifold`. A Euclidean session that never calls
//! them keeps its steps. A zero gradient stops under the tolerance.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Grassmann, ManifoldKind, Method, Solver};

struct Sphere2;

impl Objective<f64> for Sphere2 {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        use std::sync::OnceLock;
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-4.0, -4.0], array![4.0, 4.0], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }
}

impl Gradient<f64> for Sphere2 {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![2.0 * x[0], 2.0 * x[1]]
    }
}

impl DifferentiableObjective<f64> for Sphere2 {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn grassmann_rejects_a_3n_cluster() {
    struct Wide;
    impl Objective<f64> for Wide {
        fn dim(&self) -> usize {
            114
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(
                    Array1::from_elem(114, -8.0),
                    Array1::from_elem(114, 8.0),
                    0.0,
                )
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x[0]
        }
    }
    impl Gradient<f64> for Wide {
        fn dim(&self) -> usize {
            114
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            Array1::zeros(x.len())
        }
    }
    impl DifferentiableObjective<f64> for Wide {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }
    let mut solver = Solver::new(Method::Steepest, Control::default(), 114);
    solver.set_grassmann(5, 2);
    let mut x = Array1::from_elem(114, 0.1);
    let err = solver.step(&Wide, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "grassmann");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn set_factor_shape_drops_stale_lbfgs_pairs() {
    let gman = Grassmann { n: 4, p: 2 };
    let s2 = 0.5_f64.sqrt();
    let mut x = array![s2, s2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            gtol: 1e-14,
            istep: 0.1,
            ..Control::default()
        },
        8,
    );
    solver.set_manifold(ManifoldKind::Grassmann);
    solver.set_factor_shape(4, 2);
    let _ = solver.step(&quad8(), &mut x).unwrap();
    assert!(solver.pair_count() >= 1);
    solver.set_factor_shape(8, 1);
    solver.set_factor_shape(4, 2);
    assert_eq!(solver.pair_count(), 0);
    let _ = solver.step(&quad8(), &mut x).unwrap();
    let cols = gman.unpack(&x).unwrap();
    assert!((cols[0].dot(&cols[0]) - 1.0).abs() < 1e-8, "{x:?}");
    assert!(cols[0].dot(&cols[1]).abs() < 1e-8, "{x:?}");
    assert!((cols[1].dot(&cols[1]) - 1.0).abs() < 1e-8, "{x:?}");
}

#[test]
fn unitary_of_order_one_stays_on_the_circle() {
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            gtol: 1e-12,
            istep: 0.05,
            ..Control::default()
        },
        2,
    );
    solver.set_unitary(1);
    let mut x = array![0.6, 0.8];
    for _ in 0..8 {
        let rep = solver.step(&Sphere2, &mut x).unwrap();
        let n = (x[0] * x[0] + x[1] * x[1]).sqrt();
        assert!((n - 1.0).abs() < 1e-8, "left U(1) {x:?}");
        if rep.grad_norm < 1e-6 {
            break;
        }
    }
}

fn quad8() -> impl DifferentiableObjective<f64> {
    Quad8
}

struct Quad8;

impl Objective<f64> for Quad8 {
    fn dim(&self) -> usize {
        8
    }
    fn bounds(&self) -> &Bounds<f64> {
        use std::sync::OnceLock;
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(Array1::from_elem(8, -8.0), Array1::from_elem(8, 8.0), 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let a = [1.0, 2.0, 3.0, 4.0];
        let mut f = 0.0;
        for j in 0..2 {
            for i in 0..4 {
                let xi = x[j * 4 + i];
                f += 0.5 * (j as f64 + 1.0) * a[i] * xi * xi;
            }
        }
        f
    }
}

impl Gradient<f64> for Quad8 {
    fn dim(&self) -> usize {
        8
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let a = [1.0, 2.0, 3.0, 4.0];
        let mut g = Array1::zeros(8);
        for j in 0..2 {
            for i in 0..4 {
                g[j * 4 + i] = (j as f64 + 1.0) * a[i] * x[j * 4 + i];
            }
        }
        g
    }
}

impl DifferentiableObjective<f64> for Quad8 {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}
