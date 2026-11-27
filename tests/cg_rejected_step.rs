//! A rejected line search restarts nonlinear CG at steepest descent.

use std::sync::{Arc, Mutex};

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Conjugacy, Control, LineSearch, Method, Oracle, Restart, Solver, minimize};

fn bowl(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let g = array![x[0], 4.0 * x[1]];
    (0.5 * x[0] * x[0] + 2.0 * x[1] * x[1], g)
}

fn close(x: ArrayView1<f64>, y: &Array1<f64>) -> bool {
    x.iter()
        .zip(y.iter())
        .all(|(a, b)| (*a - *b).abs() <= 1e-12)
}

struct Gate {
    x1: Option<Array1<f64>>,
    probe: Option<Array1<f64>>,
}

type StepFn = Box<dyn for<'a> Fn(ArrayView1<'a, f64>) -> (f64, Array1<f64>) + Send + Sync>;

fn gated(start: Array1<f64>) -> (Oracle<StepFn>, Arc<Mutex<Gate>>) {
    let gate = Arc::new(Mutex::new(Gate {
        x1: None,
        probe: None,
    }));
    let shared = Arc::clone(&gate);
    let step: StepFn = Box::new(move |x: ArrayView1<f64>| {
        let mut gate = shared.lock().expect("gate");
        let (f, g) = bowl(x);
        let Some(x1) = gate.x1.clone() else {
            if !close(x, &start) {
                gate.x1 = Some(x.to_owned());
            }
            return (f, g);
        };
        if close(x, &x1) {
            return (f, g);
        }
        if gate.probe.is_none() {
            // First trial after the accepted step: refuse it.
            gate.probe = Some(Array1::zeros(0));
            return (f + 100.0, g);
        }
        if gate.probe.as_ref().is_some_and(|p| p.is_empty()) {
            gate.probe = Some(x.to_owned());
        }
        (f + 100.0, g)
    });
    let obj = Oracle::unbounded(2, step);
    (obj, gate)
}

fn assert_steepest(gate: &Gate) {
    let x1 = gate.x1.as_ref().expect("accepted step");
    let probe = gate.probe.as_ref().expect("probe after the rejection");
    assert!(!probe.is_empty(), "rejection was not followed by a search");
    let steep = array![-x1[0], -4.0 * x1[1]];
    let disp = probe - x1;
    let cross = (disp[0] * steep[1] - disp[1] * steep[0]).abs();
    let scale = disp.dot(&disp).sqrt() * steep.dot(&steep).sqrt();
    assert!(
        cross <= 1e-9 * scale.max(1.0),
        "direction {disp:?} is not steepest {steep:?} at {x1:?}"
    );
}

#[test]
fn a_rejected_step_restarts_the_one_shot_search() {
    let start = array![1.0, 1.0];
    let (obj, gate) = gated(start.clone());
    let control = Control::default()
        .with_maxiter(3)
        .with_gtol(1e-14)
        .with_istep(0.1)
        .with_maxmove(Some(-1.0));
    minimize(
        &obj,
        start,
        &control,
        Conjugacy::FletcherReeves,
        Restart::Never,
        LineSearch::backtracking(1e-4, 0.5, 1),
    )
    .expect("minimize");
    assert_steepest(&gate.lock().expect("gate"));
}

#[test]
fn a_rejected_step_restarts_the_session_search() {
    let start = array![1.0, 1.0];
    let (obj, gate) = gated(start.clone());
    let control = Control::default()
        .with_istep(0.1)
        .with_gtol(1e-14)
        .with_maxmove(Some(-1.0));
    let mut solver = Solver::new(Method::nlcg(Conjugacy::FletcherReeves), control, 2);
    solver.set_linesearch(LineSearch::backtracking(1e-4, 0.5, 1));
    let mut x = start;
    for _ in 0..3 {
        solver.step(&obj, &mut x).expect("step");
    }
    assert_steepest(&gate.lock().expect("gate"));
}
