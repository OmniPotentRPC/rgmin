use ndarray::{Array1, array};
use rgmin::{Control, Method, Solver};

#[test]
fn spd_session_step_is_riemannian_and_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::manifold::{Manifold, ManifoldKind, Spd, is_spd};

    struct FrobeniusI;
    impl Objective<f64> for FrobeniusI {
        fn dim(&self) -> usize {
            4
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(4, -1e12), Array1::from_elem(4, 1e12), 0.0)
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
    let x0 = array![2.0, 0.0, 0.0, 2.0];
    let egrad = array![1.0, 0.0, 0.0, 1.0];
    let rgrad = Spd.egrad2rgrad(&x0, &egrad);
    let dir = rgrad.mapv(|v| -0.1 * v);
    let want = Spd.retract(&x0, &dir);

    let mut x = x0.clone();
    let mut solver = Solver::new(
        Method::Bb,
        Control {
            maxiter: 20,
            gtol: 1e-12,
            istep: 0.1,
            maxmove: None,
            ftol_rel: None,
        },
        4,
    );
    solver.set_manifold(ManifoldKind::Spd);
    solver.set_accept(rgmin::Accept::Step);
    let _ = solver.step(&obj, &mut x).unwrap();
    assert!(is_spd(&x), "left the SPD set {x:?}");
    for i in 0..4 {
        assert!(
            (x[i] - want[i]).abs() < 1e-12,
            "step is not the affine-invariant retract {x:?} want {want:?}"
        );
    }
    for _ in 0..19 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(is_spd(&x), "left the SPD set {x:?}");
        assert!((x[1] - x[2]).abs() < 1e-12, "not symmetric {x:?}");
    }
}

#[test]
fn lbfgs_first_step_is_the_riemannian_retract() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::{Accept, Grassmann, Manifold, ManifoldKind};

    struct Brockett;
    impl Objective<f64> for Brockett {
        fn dim(&self) -> usize {
            8
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(8, -8.0), Array1::from_elem(8, 8.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            // 1/2 tr(X^T A X N), A = diag(1,2,3,4), N = diag(1,2).
            let a = [1.0, 2.0, 3.0, 4.0];
            let nmu = [1.0, 2.0];
            let mut f = 0.0;
            for j in 0..2 {
                let mut q = 0.0;
                for i in 0..4 {
                    let xi = x[j * 4 + i];
                    q += a[i] * xi * xi;
                }
                f += 0.5 * nmu[j] * q;
            }
            f
        }
    }
    impl Gradient<f64> for Brockett {
        fn dim(&self) -> usize {
            8
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let a = [1.0, 2.0, 3.0, 4.0];
            let nmu = [1.0, 2.0];
            let mut g = Array1::zeros(8);
            for j in 0..2 {
                for i in 0..4 {
                    g[j * 4 + i] = a[i] * nmu[j] * x[j * 4 + i];
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

    let gman = Grassmann { n: 4, p: 2 };
    let s2 = 0.5_f64.sqrt();
    let mut x = array![s2, s2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    {
        let cols = gman.unpack(&x).unwrap();
        assert!((cols[0].dot(&cols[0]) - 1.0).abs() < 1e-14);
        assert!(cols[0].dot(&cols[1]).abs() < 1e-14);
        assert!((cols[1].dot(&cols[1]) - 1.0).abs() < 1e-14);
    }
    let egrad = Brockett.grad(x.view());
    let rgrad = gman.project(&x, &egrad);
    let xtg = {
        let cols = gman.unpack(&x).unwrap();
        let gs = gman.unpack(&rgrad).unwrap();
        cols[0].dot(&gs[0]).abs()
            + cols[0].dot(&gs[1]).abs()
            + cols[1].dot(&gs[0]).abs()
            + cols[1].dot(&gs[1]).abs()
    };
    assert!(xtg < 1e-12, "Riemannian gradient must be horizontal {xtg}");
    assert!(
        rgrad.iter().map(|v| v * v).sum::<f64>().sqrt() > 1e-6,
        "need a non-critical start"
    );
    let want = gman.retract(&x, &rgrad.mapv(|v| -v));

    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            maxiter: 20,
            gtol: 1e-14,
            istep: 0.1,
            maxmove: None,
            ftol_rel: None,
        },
        8,
    );
    solver.set_manifold(ManifoldKind::Grassmann);
    solver.set_factor_shape(4, 2);
    solver.set_accept(Accept::Step);
    let f0 = Brockett.eval(x.view());
    let _ = solver.step(&Brockett, &mut x).unwrap();
    let cols = gman.unpack(&x).unwrap();
    assert!(
        (cols[0].dot(&cols[0]) - 1.0).abs() < 1e-10,
        "left Gr(4,2) {x:?}"
    );
    assert!(cols[0].dot(&cols[1]).abs() < 1e-10, "not orthogonal {x:?}");
    assert!(
        (cols[1].dot(&cols[1]) - 1.0).abs() < 1e-10,
        "left Gr(4,2) {x:?}"
    );
    assert!(
        (&x - &want).mapv(f64::abs).sum() < 1e-10,
        "first L-BFGS step must be Retr_x(-grad_R), got {x:?} want {want:?}"
    );
    let f1 = Brockett.eval(x.view());
    assert!(
        f1 < f0,
        "Riemannian step must decrease Brockett, {f1} vs {f0}"
    );
    for _ in 0..12 {
        let _ = solver.step(&Brockett, &mut x).unwrap();
        let cols = gman.unpack(&x).unwrap();
        assert!(
            (cols[0].dot(&cols[0]) - 1.0).abs() < 1e-8,
            "left Gr(4,2) {x:?}"
        );
        assert!(cols[0].dot(&cols[1]).abs() < 1e-8, "not orthogonal {x:?}");
        assert!(
            (cols[1].dot(&cols[1]) - 1.0).abs() < 1e-8,
            "left Gr(4,2) {x:?}"
        );
    }
}

#[test]
fn set_factor_shape_drops_stale_lbfgs_pairs() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::{Accept, Grassmann, Manifold, ManifoldKind};

    struct Brockett;
    impl Objective<f64> for Brockett {
        fn dim(&self) -> usize {
            8
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(8, -8.0), Array1::from_elem(8, 8.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            let a = [1.0, 2.0, 3.0, 4.0];
            let nmu = [1.0, 2.0];
            let mut f = 0.0;
            for j in 0..2 {
                let mut q = 0.0;
                for i in 0..4 {
                    let xi = x[j * 4 + i];
                    q += a[i] * xi * xi;
                }
                f += 0.5 * nmu[j] * q;
            }
            f
        }
    }
    impl Gradient<f64> for Brockett {
        fn dim(&self) -> usize {
            8
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let a = [1.0, 2.0, 3.0, 4.0];
            let nmu = [1.0, 2.0];
            let mut g = Array1::zeros(8);
            for j in 0..2 {
                for i in 0..4 {
                    g[j * 4 + i] = a[i] * nmu[j] * x[j * 4 + i];
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

    let gman = Grassmann { n: 4, p: 2 };
    let s2 = 0.5_f64.sqrt();
    let mut x = array![s2, s2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            maxiter: 20,
            gtol: 1e-14,
            istep: 0.1,
            maxmove: None,
            ftol_rel: None,
        },
        8,
    );
    solver.set_manifold(ManifoldKind::Grassmann);
    solver.set_factor_shape(4, 2);
    solver.set_accept(Accept::Step);
    let _ = solver.step(&Brockett, &mut x).unwrap();
    let rgrad = gman.project(&x, &Brockett.grad(x.view()));
    let want = gman.retract(&x, &rgrad.mapv(|v| -v));
    // A different (n, p) is a different geometry. Stale two-loop
    // pairs are not tangent there. Changing back must cold-start.
    solver.set_factor_shape(8, 1);
    solver.set_factor_shape(4, 2);
    let _ = solver.step(&Brockett, &mut x).unwrap();
    assert!(
        (&x - &want).mapv(f64::abs).sum() < 1e-10,
        "changing p must forget L-BFGS pairs so the next step is Retr(-grad_R), got {x:?} want {want:?}"
    );
    let cols = gman.unpack(&x).unwrap();
    assert!(
        (cols[0].dot(&cols[0]) - 1.0).abs() < 1e-10,
        "left Gr(4,2) {x:?}"
    );
    assert!(cols[0].dot(&cols[1]).abs() < 1e-10, "not orthogonal {x:?}");
    assert!(
        (cols[1].dot(&cols[1]) - 1.0).abs() < 1e-10,
        "left Gr(4,2) {x:?}"
    );
}


#[test]
fn direct_spd_steps_retract_once_and_evaluate_the_accepted_point_once() {
    use rgmin::{Accept, ManifoldKind, Oracle};
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    for (method, scale) in [(Method::Bb, 0.1), (Method::lbfgs(), 1.0)] {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let objective = Oracle::unbounded(4, move |x| {
            observed.fetch_add(1, Ordering::Relaxed);
            let gradient = &x - &array![1.0, 0.0, 0.0, 1.0];
            (0.5 * gradient.dot(&gradient), gradient)
        });
        let initial = 1.1_f64;
        let mut x = array![initial, 0.0, 0.0, initial];
        let mut solver = Solver::new(method.clone(), Control {
            maxiter: 20,
            gtol: 1e-12,
            istep: 0.1,
            maxmove: None,
            ftol_rel: None,
        }, 4);
        solver.set_manifold(ManifoldKind::Spd);
        solver.set_accept(Accept::Step);
        let tangent = -scale * initial * initial * (initial - 1.0);
        let expected = initial + tangent + tangent * tangent / (2.0 * initial);
        let report = solver.step(&objective, &mut x).unwrap();
        assert!((x[0] - expected).abs() < 1e-12, "{method:?}: {x:?}");
        assert!((x[3] - expected).abs() < 1e-12, "{method:?}: {x:?}");
        assert_eq!(x[1], 0.0);
        assert_eq!(x[2], 0.0);
        assert!((report.value - (expected - 1.0).powi(2)).abs() < 1e-12);
        assert_eq!(calls.load(Ordering::Relaxed), 2, "{method:?}");
    }
}
