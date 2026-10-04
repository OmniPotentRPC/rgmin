use eindir_core::objectives::Rosenbrock;
use ndarray::{Array1, array};
use rgmin::{Control, Method, Solver};
fn control() -> Control {
    Control { maxiter: 80, gtol: 1e-8, istep: 0.1, maxmove: None, ftol_rel: None }
}

#[test]
fn spd_session_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::manifold::is_spd;
    use rgmin::ManifoldKind;

    struct FrobeniusI;
    impl Objective<f64> for FrobeniusI {
        fn dim(&self) -> usize {
            4
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(4, -4.0), Array1::from_elem(4, 4.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            let i = [1.0, 0.0, 0.0, 1.0];
            0.5 * x.iter().zip(i).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
        }
    }
    impl Gradient<f64> for FrobeniusI {
        fn dim(&self) -> usize {
            4
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let i = [1.0, 0.0, 0.0, 1.0];
            Array1::from_iter(x.iter().zip(i).map(|(a, b)| a - b))
        }
    }
    impl DifferentiableObjective<f64> for FrobeniusI {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = FrobeniusI;
    let mut x = array![2.0, 0.3, 0.3, 1.5];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-8,
            istep: 0.1,
            maxmove: None,
            ftol_rel: None,
        },
        4,
    );
    solver.set_manifold(ManifoldKind::Spd);
    solver.set_accept(rgmin::Accept::Step);
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(is_spd(&x), "left the SPD set {x:?}");
        assert!((x[1] - x[2]).abs() < 1e-12, "not symmetric {x:?}");
    }
}

#[test]
fn spd_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::Spd);
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "spd");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn grassmann_session_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;

    /// Brockett cost on Gr(4, 2): 1/2 trace(X^T A X), A = diag(1,2,3,4).
    struct Brockett;
    impl Objective<f64> for Brockett {
        fn dim(&self) -> usize {
            8
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(8, -4.0), Array1::from_elem(8, 4.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            let a = [1.0, 2.0, 3.0, 4.0];
            let mut s = 0.0;
            for j in 0..2 {
                for i in 0..4 {
                    let v = x[j * 4 + i];
                    s += a[i] * v * v;
                }
            }
            0.5 * s
        }
    }
    impl Gradient<f64> for Brockett {
        fn dim(&self) -> usize {
            8
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let a = [1.0, 2.0, 3.0, 4.0];
            let mut g = Array1::zeros(8);
            for j in 0..2 {
                for i in 0..4 {
                    g[j * 4 + i] = a[i] * x[j * 4 + i];
                }
            }
            g
        }
    }
    impl DifferentiableObjective<f64> for Brockett {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = Brockett;
    let mut x = array![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-8,
            istep: 0.2,
            maxmove: None,
            ftol_rel: None,
        },
        8,
    );
    solver.set_manifold(ManifoldKind::grassmann(4, 2));
    solver.set_accept(rgmin::Accept::Step);
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        let c0 = &x.as_slice().unwrap()[0..4];
        let c1 = &x.as_slice().unwrap()[4..8];
        let n0: f64 = c0.iter().map(|a| a * a).sum();
        let n1: f64 = c1.iter().map(|a| a * a).sum();
        let d: f64 = c0.iter().zip(c1).map(|(a, b)| a * b).sum();
        assert!((n0 - 1.0).abs() < 1e-10, "col0 left Stiefel {x:?}");
        assert!((n1 - 1.0).abs() < 1e-10, "col1 left Stiefel {x:?}");
        assert!(d.abs() < 1e-10, "columns not orthonormal {x:?}");
    }
}

#[test]
fn grassmann_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::grassmann(5, 2));
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "grassmann");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

