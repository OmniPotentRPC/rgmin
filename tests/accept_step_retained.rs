use ndarray::{Array1, array};
use rgmin::{Control, Method, Solver};

fn control() -> Control {
    Control {
        maxiter: 80,
        gtol: 1e-8,
        istep: 0.1,
        maxmove: None,
        ftol_rel: None,
    }
}

#[test]
fn lbfgs_accept_step_moves_when_energy_is_flat() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::Accept;
    use std::sync::OnceLock;

    struct FlatEnergyBowl;
    impl Objective<f64> for FlatEnergyBowl {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
        }
        fn eval(&self, _: ArrayView1<f64>) -> f64 {
            0.0
        }
    }
    impl Gradient<f64> for FlatEnergyBowl {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            &x * 2.0
        }
    }
    impl DifferentiableObjective<f64> for FlatEnergyBowl {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (0.0, self.grad(x))
        }
    }

    let obj = FlatEnergyBowl;
    let mut x = array![3.0, -4.0];
    let start = x.clone();
    let mut ctrl = control();
    ctrl.maxmove = Some(1.0);
    let mut solver = Solver::new(Method::lbfgs(), ctrl, 2);
    solver.set_accept(Accept::Step);
    solver.step(&obj, &mut x).unwrap();
    assert!(
        (x[0] - start[0]).abs() + (x[1] - start[1]).abs() > 1e-9,
        "LBFGS Accept::Step stayed put {x:?}"
    );
    let n0 = (start[0] * start[0] + start[1] * start[1]).sqrt();
    let n1 = (x[0] * x[0] + x[1] * x[1]).sqrt();
    assert!(n1 < n0, "clipped two-loop step did not descend {x:?}");
}

#[test]
fn first_order_accept_step_moves_a_nonconservative_force() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::Accept;
    use std::sync::OnceLock;

    struct ConstantForce;
    impl Objective<f64> for ConstantForce {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
        }
        fn eval(&self, _: ArrayView1<f64>) -> f64 {
            1.0
        }
    }
    impl Gradient<f64> for ConstantForce {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, _: ArrayView1<f64>) -> Array1<f64> {
            array![0.4, 0.0]
        }
    }
    impl DifferentiableObjective<f64> for ConstantForce {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (1.0, self.grad(x))
        }
    }

    for method in [
        Method::Steepest,
        Method::Bfgs,
        Method::lbfgs(),
        Method::polak_ribiere(),
        Method::Sr1,
        Method::Sr2,
    ] {
        let obj = ConstantForce;
        let mut x = array![0.0, 0.0];
        let mut ctrl = control();
        ctrl.maxmove = Some(0.2);
        let mut solver = Solver::new(method.clone(), ctrl, 2);
        solver.set_accept(Accept::Step);
        solver.step(&obj, &mut x).unwrap();
        assert!(
            x[0].abs() > 1e-9,
            "{method:?} Accept::Step froze on a non-conservative force {x:?}"
        );
    }
}
