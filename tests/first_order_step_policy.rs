use std::sync::{OnceLock, atomic::{AtomicUsize, Ordering}};

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, Method, Solver};

struct ConstantForce {
    calls: AtomicUsize,
    invalid_gradient: bool,
    invalid_value: bool,
}
impl Objective<f64> for ConstantForce {
    fn dim(&self) -> usize { 2 }
    fn bounds(&self) -> &Bounds<f64> {
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-10.0, -10.0], array![10.0, 10.0], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if self.invalid_value && x[0] < 0.0 { f64::NAN } else { 1.0 }
    }
}
impl Gradient<f64> for ConstantForce {
    fn dim(&self) -> usize { 2 }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![if self.invalid_gradient && x[0] < 0.0 { f64::NAN } else { 0.4 }, 0.0]
    }
}
impl DifferentiableObjective<f64> for ConstantForce {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        (self.eval(x), self.grad(x))
    }
}
fn methods() -> Vec<Method> {
    vec![Method::Steepest, Method::Bfgs, Method::lbfgs(), Method::polak_ribiere(),
         Method::Sr1, Method::Sr2, Method::adam(), Method::Bb]
}
fn solver(method: Method) -> Solver {
    let mut solver=Solver::new(method, Control { maxiter: 8, gtol: 1e-12, istep: 0.1,
        maxmove: Some(0.05), ftol_rel: None }, 2);
    solver.set_accept(Accept::Step);
    solver
}
#[test]
fn every_first_order_step_policy_moves_with_one_trial_and_keeps_the_cap() {
    for method in methods() {
        let obj=ConstantForce { calls: AtomicUsize::new(0), invalid_gradient: false, invalid_value: false };
        let mut solver=solver(method.clone());
        let mut x=array![0.0,0.0];
        for iteration in 0..3 {
            let previous=x.clone();
            let report=solver.step(&obj,&mut x).unwrap();
            assert!(x[0] < previous[0], "{method:?} stalled at {x:?}");
            assert!((x[0]-previous[0]).abs() <= 0.05 + 1e-14, "{method:?}: {x:?}");
            assert_eq!(x[1],0.0);
            assert_eq!(report.coords,x);
            assert_eq!(report.value,1.0);
            assert!((report.grad_norm-0.4).abs() < 1e-14);
            assert_eq!(obj.calls.load(Ordering::Relaxed),iteration+2,"{method:?}");
        }
    }
}
#[test]
fn rejected_nonfinite_step_keeps_the_measured_point_and_cached_gradient() {
    for method in methods() {
        for (invalid_gradient,invalid_value) in [(true,false),(false,true)] {
            let obj=ConstantForce { calls: AtomicUsize::new(0), invalid_gradient, invalid_value };
            let mut solver=solver(method.clone());
            let mut x=array![0.0,0.0];
            for iteration in 0..2 {
                let report=solver.step(&obj,&mut x).unwrap();
                assert_eq!(x,array![0.0,0.0],"{method:?}");
                assert_eq!(report.coords,x);
                assert_eq!(report.value,1.0);
                assert!((report.grad_norm-0.4).abs() < 1e-14);
                assert_eq!(obj.calls.load(Ordering::Relaxed),iteration+2,"{method:?}");
                assert_eq!(solver.pair_count(),0,"{method:?}");
            }
        }
    }
}
