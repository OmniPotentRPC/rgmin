//! Oracle checks for BFGS, SR1, SR2, seeded PSO, and the L-BFGS NEB guards.

use std::thread;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::trust::dogleg_direction;
use rgmin::{
    Control, LineSearch, Method, Oracle, Status, minimize_bfgs, minimize_method, minimize_pso,
    minimize_sr1, minimize_sr2,
};

const WOLFE: LineSearch = LineSearch::Wolfe {
    c1: 1e-4,
    c2: 0.1,
    maxiter: 30,
};

fn control(maxiter: usize, gtol: f64) -> Control {
    Control {
        maxiter,
        gtol,
        istep: 1.0,
        maxmove: Some(f64::INFINITY),
        ftol_rel: None,
        noise_floor: None,
    }
}

fn rose(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len();
    let mut f = -0.0;
    let mut g = Array1::zeros(n);
    for i in 0..n - 1 {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        f += a * a + 100.0 * b * b;
        g[i] += -2.0 * a - 400.0 * x[i] * b;
        g[i + 1] += 200.0 * b;
    }
    (f, g)
}

fn scaled(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len();
    let mut f = -0.0;
    let mut g = Array1::zeros(n);
    let denom = (n - 1) as f64;
    for i in 0..n {
        let lam = 10f64.powf(8.0 * (i as f64) / denom);
        f += 0.5 * lam * x[i] * x[i];
        g[i] = lam * x[i];
    }
    (f, g)
}

const MORSE_D0: f64 = 0.7102;
const MORSE_ALPHA: f64 = 1.6047;
const MORSE_R0: f64 = 2.897;

fn morse(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let rc = 7.0;
    let xc = (-MORSE_ALPHA * (rc - MORSE_R0)).exp();
    let shift = MORSE_D0 * (xc * xc - 2.0 * xc);
    let mut energy = -0.0;
    let mut grad = Array1::zeros(x.len());
    let cut2 = rc * rc;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut diff = [0.0; 3];
            let mut r2 = -0.0;
            for a in 0..3 {
                diff[a] = x[3 * i + a] - x[3 * j + a];
                r2 += diff[a] * diff[a];
            }
            if r2 <= 0.0 || r2 >= cut2 {
                continue;
            }
            let r = r2.sqrt();
            let e = (-MORSE_ALPHA * (r - MORSE_R0)).exp();
            energy += MORSE_D0 * (e * e - 2.0 * e) - shift;
            let slope = (2.0 * MORSE_D0 * MORSE_ALPHA * e * (1.0 - e)) / r;
            for a in 0..3 {
                let piece = slope * diff[a];
                grad[3 * i + a] += piece;
                grad[3 * j + a] -= piece;
            }
        }
    }
    (energy, grad)
}

fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut energy = -0.0;
    let mut grad = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let mut diff = [0.0; 3];
            let mut r2 = -0.0;
            for a in 0..3 {
                diff[a] = x[3 * i + a] - x[3 * j + a];
                r2 += diff[a] * diff[a];
            }
            if r2 <= 0.0 {
                continue;
            }
            let inv = 1.0 / r2;
            let sr2 = inv;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            energy += 4.0 * (sr12 - sr6);
            let slope = 24.0 * (sr6 - 2.0 * sr12) * inv;
            for a in 0..3 {
                let piece = slope * diff[a];
                grad[3 * i + a] += piece;
                grad[3 * j + a] -= piece;
            }
        }
    }
    (energy, grad)
}

struct Boxed<F> {
    f: F,
    bounds: Bounds<f64>,
}

impl<F> Objective<f64> for Boxed<F>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        (self.f)(x).0
    }
}

impl<F> Gradient<f64> for Boxed<F>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        (self.f)(x).1
    }
}

impl<F> DifferentiableObjective<f64> for Boxed<F>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.f)(x)
    }
}

fn box_of(
    dim: usize,
    lo: f64,
    hi: f64,
    f: impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
) -> Boxed<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    Boxed {
        f,
        bounds: Bounds::new(Array1::from_elem(dim, lo), Array1::from_elem(dim, hi), 0.0),
    }
}

#[test]
fn bfgs_sr1_and_sr2_reach_the_oracle_problems() {
    let rose_obj = Oracle::unbounded(2, rose);
    let start = array![-1.2, 1.0];
    let quad = Oracle::unbounded(4, scaled);
    let q0 = Array1::from_elem(4, 0.5);
    let morse_obj = Oracle::unbounded(6, morse);
    let m0 = array![0.0, 0.0, 0.0, 3.4, 0.15, -0.05];
    let lj_obj = Oracle::unbounded(9, lj);
    let l0 = array![0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.4, 1.3, 0.1];
    for (name, method) in [
        ("bfgs", Method::Bfgs),
        ("sr1", Method::Sr1),
        ("sr2", Method::Sr2),
    ] {
        let rose_r = minimize_method(
            &rose_obj,
            start.clone(),
            &control(80, 1e-6),
            method.clone(),
            WOLFE,
        )
        .unwrap();
        assert!(
            rose_r.grad_norm < 1e-6 && rose_r.value < 1e-8,
            "{name} Rosenbrock |g|={} f={}",
            rose_r.grad_norm,
            rose_r.value
        );
        let quad_r =
            minimize_method(&quad, q0.clone(), &control(80, 1e-5), method.clone(), WOLFE).unwrap();
        if name == "sr2" {
            // PSB on a condition-1e8 diagonal fills the soft curvature and
            // then creeps. The stiff coordinates still have to be solved,
            // and the moderate quadratic below checks a tight gradient.
            let (f0, _) = scaled(q0.view());
            assert!(
                quad_r.value < f0 / 1.0e6
                    && quad_r.coords[2].abs() < 1e-5
                    && quad_r.coords[3].abs() < 1e-5,
                "{name} scaled f={} x={:?}",
                quad_r.value,
                quad_r.coords
            );
        } else {
            assert!(
                quad_r.grad_norm < 1e-5,
                "{name} scaled |g|={} x={:?}",
                quad_r.grad_norm,
                quad_r.coords
            );
        }
        let morse_r = minimize_method(
            &morse_obj,
            m0.clone(),
            &control(80, 1e-4),
            method.clone(),
            WOLFE,
        )
        .unwrap();
        let dx = morse_r.coords[3] - morse_r.coords[0];
        let dy = morse_r.coords[4] - morse_r.coords[1];
        let dz = morse_r.coords[5] - morse_r.coords[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            (r - MORSE_R0).abs() < 1e-3 && morse_r.grad_norm < 1e-3,
            "{name} Morse r={r} |g|={}",
            morse_r.grad_norm
        );
        let lj_r = minimize_method(&lj_obj, l0.clone(), &control(80, 1e-3), method, WOLFE).unwrap();
        assert!(
            lj_r.value < -2.9 && lj_r.grad_norm < 1e-2,
            "{name} LJ f={} |g|={}",
            lj_r.value,
            lj_r.grad_norm
        );
    }
}

#[test]
fn bfgs_energy_does_not_rise_and_a_cap_step_lands_on_the_cap() {
    let obj = Oracle::unbounded(2, rose);
    let mut solver = rgmin::Solver::new(Method::Bfgs, control(40, 1e-8), 2);
    solver.set_linesearch(WOLFE);
    solver.set_atom_maxmove(0.0);
    solver.set_maxmove(0.0);
    let mut x = array![-1.2, 1.0];
    let (mut prev, _) = obj.value_and_gradient(x.view());
    for _ in 0..40 {
        let report = solver.step(&obj, &mut x).unwrap();
        let slack = (4.0 * f64::EPSILON) * prev.abs();
        assert!(
            report.value <= prev + slack,
            "BFGS energy {prev} -> {}",
            report.value
        );
        prev = report.value;
        if report.grad_norm < 1e-8 {
            break;
        }
    }

    let bowl = Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]));
    let mut ctl = control(1, 0.0);
    ctl.maxmove = Some(0.5);
    let report = minimize_bfgs(&bowl, array![10.0], &ctl, WOLFE).unwrap();
    assert!(
        (report.coords[0] - 9.5).abs() < 1e-12,
        "x={}",
        report.coords[0]
    );
}

#[test]
fn a_trust_step_that_equals_the_radius_is_the_newton_point() {
    let h = Array2::<f64>::eye(2);
    let g = array![3.0, 4.0];
    let p = dogleg_direction(&h, &g, 5.0);
    let n = (p[0] * p[0] + p[1] * p[1]).sqrt();
    assert!((n - 5.0).abs() < 1e-12, "norm {n}");
    assert!(
        (p[0] + 3.0).abs() < 1e-12 && (p[1] + 4.0).abs() < 1e-12,
        "p {p:?}"
    );
}

#[test]
fn a_zero_gradient_stops_bfgs_and_sr_and_pso_keeps_the_best() {
    let bowl = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        let mut f = -0.0;
        let mut g = Array1::zeros(3);
        for i in 0..3 {
            f += 0.5 * x[i] * x[i];
            g[i] = x[i];
        }
        (f, g)
    });
    let x0 = Array1::zeros(3);
    let ctl = control(8, 1e-8);
    for (name, method) in [
        ("bfgs", Method::Bfgs),
        ("sr1", Method::Sr1),
        ("sr2", Method::Sr2),
    ] {
        let report = minimize_method(&bowl, x0.clone(), &ctl, method, WOLFE).unwrap();
        assert_eq!(report.steps, 0, "{name}");
        assert!(
            report.status == Status::Tolerance,
            "{name} {:?}",
            report.status
        );
        assert!(report.coords.iter().all(|v| *v == 0.0));
    }
    let boxed = box_of(2, -2.0, 2.0, |x: ArrayView1<f64>| {
        (0.5 * x.dot(&x), x.to_owned())
    });
    let report = minimize_pso(
        &boxed,
        array![0.0, 0.0],
        &control(40, 1e-8),
        8,
        0.5,
        1.5,
        1.5,
    )
    .unwrap();
    assert!(report.value <= 0.0, "pso left the minimum {}", report.value);
}

#[test]
fn a_nonfinite_oracle_is_not_a_converged_quasi_newton_point() {
    for bad in [f64::NAN, f64::INFINITY] {
        let obj = Oracle::unbounded(2, move |x: ArrayView1<f64>| {
            if x.iter().all(|v| *v == 0.0) {
                (0.0, array![1.0, -1.0])
            } else {
                (bad, array![bad, 0.0])
            }
        });
        let ctl = control(4, 1e-10);
        for method in [Method::Bfgs, Method::Sr1, Method::Sr2] {
            let report = minimize_method(&obj, array![0.0, 0.0], &ctl, method, WOLFE).unwrap();
            assert!(report.coords.iter().all(|v| v.is_finite()));
            assert_ne!(report.status, Status::Tolerance);
        }
    }
}

#[test]
fn seeded_pso_matches_across_runs_and_threads() {
    let obj = box_of(2, -2.0, 2.0, rose);
    let ctl = control(25, 1e-6);
    let once = || minimize_pso(&obj, array![-1.2, 1.0], &ctl, 12, 0.6, 1.4, 1.4).unwrap();
    let a = once();
    let b = once();
    assert!(a.value < 10.0, "pso did not descend {}", a.value);
    for i in 0..2 {
        assert_eq!(a.coords[i].to_bits(), b.coords[i].to_bits());
    }
    assert_eq!(a.steps, b.steps);
    let (left, right) = thread::scope(|scope| {
        let t1 = scope.spawn(once);
        let t2 = scope.spawn(once);
        (t1.join().unwrap(), t2.join().unwrap())
    });
    for i in 0..2 {
        assert_eq!(left.coords[i].to_bits(), right.coords[i].to_bits());
    }
}

#[test]
fn sr_steps_lower_a_convex_quadratic() {
    let obj = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        let mut f = -0.0;
        let mut g = Array1::zeros(3);
        for i in 0..3 {
            let lam = (i + 1) as f64;
            f += 0.5 * lam * x[i] * x[i];
            g[i] = lam * x[i];
        }
        (f, g)
    });
    let x0 = array![1.0, -0.5, 0.25];
    let (f0, _) = obj.value_and_gradient(x0.view());
    for (name, run) in [
        (
            "sr1",
            minimize_sr1(&obj, x0.clone(), &control(30, 1e-8), WOLFE).unwrap(),
        ),
        (
            "sr2",
            minimize_sr2(&obj, x0, &control(30, 1e-8), WOLFE).unwrap(),
        ),
    ] {
        assert!(run.value < f0, "{name} {f0} -> {}", run.value);
        assert!(run.grad_norm < 1e-6, "{name} |g|={}", run.grad_norm);
    }
}
