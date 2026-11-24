//! A Euclidean `maxmove` that is not positive and finite is no cap: the
//! run is the run with `maxmove = None`, and no step is turned around.
//!
//! Zero, -0 and NaN also keep the default per-atom cap. A negative or
//! infinite value is the caller's request for no cap at all, so the
//! default per-atom cap is off as well; the reference for those runs
//! clears it with `set_atom_maxmove(0.0)`.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::{
    Accept, Control, DistanceConstraints, EonCg, FireKind, HessianObjective, HessianOracle,
    HessianVector, LbfgsStep, LineSearch, ManifoldKind, Method, NewtonKind, NoiseFloor, Solver,
    minimize_eon_cg, minimize_method, minimize_method_hess, minimize_method_hvp,
    minimize_newton_cg,
};
use std::sync::OnceLock;

const DIM: usize = 6;
const WEIGHT: [f64; DIM] = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

struct Bowl;

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        DIM
    }

    fn bounds(&self) -> &Bounds<f64> {
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| {
            Bounds::new(
                Array1::from_elem(DIM, -1e6),
                Array1::from_elem(DIM, 1e6),
                0.0,
            )
        })
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * (0..DIM).map(|i| WEIGHT[i] * x[i] * x[i]).sum::<f64>()
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        DIM
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter((0..DIM).map(|i| WEIGHT[i] * x[i]))
    }
}

impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

impl HessianObjective for Bowl {
    fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
        Array2::from_diag(&Array1::from(WEIGHT.to_vec()))
    }
}

impl HessianVector for Bowl {
    fn hessian_vector(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter((0..DIM).map(|i| WEIGHT[i] * v[i]))
    }
}

/// Values that keep the defaults: the run is the `None` run.
const KEEPS_DEFAULTS: [f64; 3] = [0.0, -0.0, f64::NAN];
/// Values that turn every cap off, the default per-atom cap included.
const ALL_OFF: [f64; 3] = [-1.0, f64::NEG_INFINITY, f64::INFINITY];

fn start() -> Array1<f64> {
    array![3.0, -4.0, 1.5, 2.5, -2.0, 0.5]
}

#[derive(Clone, Copy, Debug)]
enum Oracle {
    Gradient,
    Hessian,
    Product,
}

fn methods() -> Vec<(Method, Oracle)> {
    vec![
        (Method::polak_ribiere_plus(), Oracle::Gradient),
        (Method::Bfgs, Oracle::Gradient),
        (Method::lbfgs(), Oracle::Gradient),
        (Method::Sr1, Oracle::Gradient),
        (Method::Sr2, Oracle::Gradient),
        (Method::adam(), Oracle::Gradient),
        (Method::Steepest, Oracle::Gradient),
        (Method::fire(FireKind::V1), Oracle::Gradient),
        (Method::fire(FireKind::V2), Oracle::Gradient),
        (Method::Bb, Oracle::Gradient),
        (Method::QuickMin, Oracle::Gradient),
        (Method::Ode12r, Oracle::Gradient),
        (Method::diis(), Oracle::Gradient),
        (Method::newton(NewtonKind::Shifted), Oracle::Hessian),
        (Method::newton(NewtonKind::Rfo), Oracle::Hessian),
        (Method::Dogleg, Oracle::Hessian),
        (Method::NewtonCg, Oracle::Product),
        (Method::NewtonPcg, Oracle::Product),
    ]
}

fn control(maxmove: Option<f64>) -> Control {
    Control::default()
        .with_maxiter(6)
        .with_gtol(1e-14)
        .with_maxmove(maxmove)
}

type Setup<'a> = &'a dyn Fn(&mut Solver);

/// Points and values after each of six steps, or the error that ended them.
fn trajectory(
    method: &Method,
    oracle: Oracle,
    control: Control,
    start: &Array1<f64>,
    setup: Setup,
    atom_off: bool,
) -> String {
    let mut solver = Solver::new(method.clone(), control, start.len());
    setup(&mut solver);
    if atom_off {
        solver.set_atom_maxmove(0.0);
    }
    let mut x = start.clone();
    let mut seen = Vec::new();
    for _ in 0..6 {
        let report = match oracle {
            Oracle::Gradient => solver.step(&Bowl, &mut x),
            Oracle::Hessian => solver.step_hess(&Bowl, &mut x),
            Oracle::Product => solver.step_hvp(&Bowl, &mut x),
        };
        match report {
            Ok(report) => seen.push(format!("{:?} {:?}", x.as_slice().unwrap(), report.value)),
            Err(err) => {
                seen.push(format!("error {err}"));
                break;
            }
        }
    }
    seen.join("; ")
}

/// Every non-binding cap, given in the `Control` and through
/// `set_maxmove`, runs as its reference.
fn assert_uncapped(
    label: &str,
    method: &Method,
    oracle: Oracle,
    base: &Control,
    start: &Array1<f64>,
    setup: Setup,
) {
    let with = |maxmove| base.clone().with_maxmove(maxmove);
    for (caps, atom_off) in [(KEEPS_DEFAULTS, false), (ALL_OFF, true)] {
        let free = trajectory(method, oracle, with(None), start, setup, atom_off);
        for cap in caps {
            let given = trajectory(method, oracle, with(Some(cap)), start, setup, false);
            assert_eq!(given, free, "{label} {method:?} Control maxmove {cap}");
            let set = |s: &mut Solver| {
                setup(s);
                s.set_maxmove(cap);
            };
            let late = trajectory(method, oracle, with(None), start, &set, false);
            assert_eq!(late, free, "{label} {method:?} set_maxmove({cap})");
        }
    }
}

#[test]
fn a_negative_maxmove_does_not_turn_a_descent_step_uphill() {
    let x0 = start();
    let f0 = Bowl.eval(x0.view());
    let g = Bowl.grad(x0.view());
    for (method, oracle) in methods() {
        for cap in [-1.0, -0.5, f64::NEG_INFINITY] {
            let mut solver = Solver::new(method.clone(), control(Some(cap)), DIM);
            let mut x = x0.clone();
            let report = match oracle {
                Oracle::Gradient => solver.step(&Bowl, &mut x),
                Oracle::Hessian => solver.step_hess(&Bowl, &mut x),
                Oracle::Product => solver.step_hvp(&Bowl, &mut x),
            }
            .unwrap_or_else(|err| panic!("{method:?} maxmove {cap}: {err}"));
            assert!(
                report.value <= f0,
                "{method:?} with maxmove {cap} went from {f0} to {}",
                report.value
            );
            assert!(
                (&x - &x0).dot(&g) <= 0.0,
                "{method:?} with maxmove {cap} stepped against -g"
            );
        }
    }
}

#[test]
fn a_session_runs_uncapped_under_a_non_binding_maxmove() {
    let x0 = start();
    for (method, oracle) in methods() {
        for accept in [
            Accept::None,
            Accept::Energy,
            Accept::Nonmonotone,
            Accept::Step,
        ] {
            assert_uncapped(
                &format!("{accept:?}"),
                &method,
                oracle,
                &control(None),
                &x0,
                &|s| s.set_accept(accept),
            );
        }
    }
}

#[test]
fn the_session_variants_run_uncapped_under_a_non_binding_maxmove() {
    let x0 = start();
    let plain = control(None);
    let fire2 = Method::fire(FireKind::V2);
    assert_uncapped("abc", &fire2, Oracle::Gradient, &plain, &x0, &|s| {
        s.set_fire_abc(true).unwrap()
    });
    let lbfgs = Method::lbfgs();
    assert_uncapped("eon capped", &lbfgs, Oracle::Gradient, &plain, &x0, &|_| {});
    assert_uncapped("wolfe", &lbfgs, Oracle::Gradient, &plain, &x0, &|s| {
        s.set_lbfgs_step(LbfgsStep::Wolfe).unwrap()
    });
    let wolfe = LineSearch::wolfe(1e-4, 0.9, 20);
    for method in [
        Method::lbfgs(),
        Method::Steepest,
        Method::polak_ribiere_plus(),
        Method::Bfgs,
    ] {
        assert_uncapped(
            "wolfe line search",
            &method,
            Oracle::Gradient,
            &plain,
            &x0,
            &|s| s.set_linesearch(wolfe),
        );
    }
    let noisy = plain
        .clone()
        .with_noise_floor(NoiseFloor::new(1e-3))
        .with_ftol_rel(1e-6);
    for (method, oracle) in methods() {
        assert_uncapped("noise floor", &method, oracle, &noisy, &x0, &|_| {});
    }
}

#[test]
fn a_manifold_session_runs_uncapped_under_a_non_binding_maxmove() {
    let on_sphere = start() / start().dot(&start()).sqrt();
    for method in [
        Method::lbfgs(),
        Method::Steepest,
        Method::polak_ribiere_plus(),
        Method::fire(FireKind::V1),
        Method::Bfgs,
    ] {
        for accept in [Accept::None, Accept::Energy, Accept::Step] {
            assert_uncapped(
                &format!("sphere {accept:?}"),
                &method,
                Oracle::Gradient,
                &control(None),
                &on_sphere,
                &|s| {
                    s.set_manifold(ManifoldKind::Sphere);
                    s.set_accept(accept);
                },
            );
        }
    }
}

#[test]
fn a_filled_run_is_uncapped_under_a_non_binding_maxmove() {
    let fill = |x: ArrayView1<f64>, g: &mut Array1<f64>| {
        g.assign(&Bowl.grad(x));
        Bowl.eval(x)
    };
    let methods = [
        Method::fire(FireKind::V1),
        Method::fire(FireKind::V2),
        Method::lbfgs(),
        Method::QuickMin,
        Method::Steepest,
    ];
    for method in methods {
        let run = |maxmove, atom_off: bool| {
            let mut solver = Solver::new(method.clone(), control(maxmove), DIM);
            if atom_off {
                solver.set_atom_maxmove(0.0);
            }
            let mut x = start();
            let report = solver.run_fill(&mut x, fill).map_err(|e| e.to_string());
            format!("{:?} {:?}", x.as_slice().unwrap(), report.map(|r| r.value))
        };
        for (caps, atom_off) in [(KEEPS_DEFAULTS, false), (ALL_OFF, true)] {
            let free = run(None, atom_off);
            for cap in caps {
                assert_eq!(run(Some(cap), false), free, "{method:?} maxmove {cap}");
            }
        }
    }
}

#[test]
fn a_one_shot_is_uncapped_under_a_non_binding_maxmove() {
    let run = |method: &Method, oracle: Oracle, maxmove| {
        let control = control(maxmove);
        let ls = LineSearch::default();
        let report = match oracle {
            Oracle::Gradient => minimize_method(&Bowl, start(), &control, method.clone(), ls),
            Oracle::Hessian => minimize_method_hess(&Bowl, start(), &control, method.clone(), ls),
            Oracle::Product => minimize_method_hvp(&Bowl, start(), &control, method.clone(), ls),
        };
        report
            .map(|r| {
                format!(
                    "{:?} {:?} {}",
                    r.coords.as_slice().unwrap(),
                    r.value,
                    r.steps
                )
            })
            .map_err(|err| err.to_string())
    };
    for (method, oracle) in methods() {
        let free = run(&method, oracle, None);
        assert!(free.is_ok(), "{method:?} uncapped: {free:?}");
        for cap in KEEPS_DEFAULTS {
            assert_eq!(
                run(&method, oracle, Some(cap)),
                free,
                "{method:?} maxmove {cap}"
            );
        }
        let off = run(&method, oracle, Some(f64::INFINITY));
        assert!(off.is_ok(), "{method:?} maxmove inf: {off:?}");
        for cap in ALL_OFF {
            assert_eq!(
                run(&method, oracle, Some(cap)),
                off,
                "{method:?} maxmove {cap}"
            );
        }
    }
    let newton_cg = |maxmove| {
        minimize_newton_cg(&Bowl, start(), &control(maxmove))
            .map(|r| format!("{:?} {:?}", r.coords.as_slice().unwrap(), r.value))
            .map_err(|err| err.to_string())
    };
    let free = newton_cg(None);
    assert!(free.is_ok(), "newton-cg uncapped: {free:?}");
    for cap in KEEPS_DEFAULTS.into_iter().chain(ALL_OFF) {
        assert_eq!(newton_cg(Some(cap)), free, "newton-cg maxmove {cap}");
    }
}

#[test]
fn set_atom_maxmove_takes_only_a_positive_finite_cap() {
    let x0 = start();
    for (method, oracle) in methods() {
        let off = |s: &mut Solver| s.set_atom_maxmove(0.0);
        let free = trajectory(&method, oracle, control(None), &x0, &off, false);
        for cap in [-1.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let set = |s: &mut Solver| s.set_atom_maxmove(cap);
            let got = trajectory(&method, oracle, control(None), &x0, &set, false);
            assert_eq!(got, free, "{method:?} set_atom_maxmove({cap})");
        }
    }
}

/// The off sentinel is the request for no cap at all, so it also clears
/// a per-atom cap the caller set before it, not only the default one.
#[test]
fn an_off_maxmove_clears_a_per_atom_cap_set_before_it() {
    let x0 = start();
    let none = |s: &mut Solver| s.set_atom_maxmove(0.0);
    let held = |s: &mut Solver| s.set_atom_maxmove(0.05);
    let run = |setup: Setup| {
        trajectory(
            &Method::Steepest,
            Oracle::Gradient,
            control(None),
            &x0,
            setup,
            false,
        )
    };
    assert_ne!(
        run(&held),
        run(&none),
        "a 0.05 per-atom cap must change the steepest descent run"
    );
    for (method, oracle) in methods() {
        let free = trajectory(&method, oracle, control(None), &x0, &none, false);
        for cap in ALL_OFF {
            let set = |s: &mut Solver| {
                s.set_atom_maxmove(0.05);
                s.set_maxmove(cap);
            };
            let got = trajectory(&method, oracle, control(None), &x0, &set, false);
            assert_eq!(
                got, free,
                "{method:?} set_atom_maxmove(0.05), set_maxmove({cap})"
            );
        }
    }
}

#[test]
fn eon_cg_runs_uncapped_under_a_non_binding_max_move() {
    let fill = |x: &[f64], g: &mut [f64]| {
        let x = ArrayView1::from(x);
        g.copy_from_slice(Bowl.grad(x).as_slice().unwrap());
        Bowl.eval(x)
    };
    let run = |max_move| {
        let mut x = start().to_vec();
        let opts = EonCg::default()
            .with_max_move(max_move)
            .with_max_iter(6)
            .with_gtol(1e-14);
        let report = minimize_eon_cg(&mut x, fill, &opts);
        format!("{x:?} {:?} {}", report.energy, report.evals)
    };
    let free = run(1e300);
    for cap in KEEPS_DEFAULTS.into_iter().chain(ALL_OFF) {
        assert_eq!(run(cap), free, "max_move {cap}");
    }
    let near = start() * 0.05;
    let mut x = near.to_vec();
    let f0 = Bowl.eval(near.view());
    let opts = EonCg::default().with_max_move(-1.0).with_max_iter(1);
    let report = minimize_eon_cg(&mut x, fill, &opts);
    assert!(
        report.energy < f0,
        "max_move -1 went from {f0} to {}",
        report.energy
    );
}

#[test]
fn a_positive_maxmove_still_caps_the_step() {
    let x0 = start();
    let mut solver = Solver::new(Method::Steepest, control(None), DIM);
    solver.set_accept(Accept::None);
    solver.set_atom_maxmove(0.0);
    solver.set_maxmove(0.5);
    let mut x = x0.clone();
    solver.step(&Bowl, &mut x).unwrap();
    let d = &x - &x0;
    let length = d.dot(&d).sqrt();
    assert!((length - 0.5).abs() < 1e-12, "a 0.5 cap gave {length}");
    assert!(
        d.dot(&Bowl.grad(x0.view())) < 0.0,
        "the capped step is downhill"
    );
}

#[test]
fn a_constrained_session_runs_uncapped_under_a_non_binding_maxmove() {
    let angle = 20f64.to_radians();
    let target = array![0.0, 0.0, 0.0, angle.cos(), angle.sin(), 0.0];
    let obj = HessianOracle::unbounded(
        6,
        move |p: ArrayView1<f64>| {
            let d = &p - &target;
            (0.5 * d.dot(&d), d)
        },
        |_| Array2::eye(6),
    );
    for (method, oracle) in [
        (Method::Steepest, Oracle::Gradient),
        (Method::fire(FireKind::V1), Oracle::Gradient),
        (Method::QuickMin, Oracle::Gradient),
        (Method::Bfgs, Oracle::Gradient),
        (Method::lbfgs(), Oracle::Gradient),
        (Method::polak_ribiere_plus(), Oracle::Gradient),
        (Method::Dogleg, Oracle::Hessian),
    ] {
        let run = |maxmove, atom_off: bool| {
            let mut solver = Solver::new(method.clone(), control(maxmove), 6);
            if atom_off {
                solver.set_atom_maxmove(0.0);
            }
            let bond = DistanceConstraints::new(vec![(0, 1)], vec![1.0]).unwrap();
            solver.set_distance_constraints(bond).unwrap();
            let mut x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
            let mut seen = Vec::new();
            for _ in 0..6 {
                let report = match oracle {
                    Oracle::Hessian => solver.step_hess(&obj, &mut x),
                    _ => solver.step(&obj, &mut x),
                };
                match report {
                    Ok(report) => {
                        seen.push(format!("{:?} {:?}", x.as_slice().unwrap(), report.value))
                    }
                    Err(err) => {
                        seen.push(format!("error {err}"));
                        break;
                    }
                }
            }
            seen.join("; ")
        };
        for (caps, atom_off) in [(KEEPS_DEFAULTS, false), (ALL_OFF, true)] {
            let free = run(None, atom_off);
            assert!(!free.contains("error"), "{method:?} uncapped: {free}");
            for cap in caps {
                assert_eq!(run(Some(cap), false), free, "{method:?} maxmove {cap}");
            }
        }
    }
}

#[cfg(feature = "highs")]
#[test]
fn a_highs_box_step_runs_uncapped_under_a_non_binding_maxmove() {
    for method in [Method::lbfgs(), Method::Bfgs, Method::Steepest] {
        let run = |maxmove, atom_off: bool| {
            let mut solver = Solver::new(method.clone(), control(maxmove), DIM);
            if atom_off {
                solver.set_atom_maxmove(0.0);
            }
            solver.set_highs(true);
            assert!(solver.set_box(Some(vec![-10.0; DIM]), Some(vec![10.0; DIM])));
            let mut x = start();
            let mut seen = Vec::new();
            for _ in 0..6 {
                match solver.step(&Bowl, &mut x) {
                    Ok(report) => {
                        seen.push(format!("{:?} {:?}", x.as_slice().unwrap(), report.value))
                    }
                    Err(err) => {
                        seen.push(format!("error {err}"));
                        break;
                    }
                }
            }
            seen.join("; ")
        };
        for (caps, atom_off) in [(KEEPS_DEFAULTS, false), (ALL_OFF, true)] {
            let free = run(None, atom_off);
            for cap in caps {
                assert_eq!(run(Some(cap), false), free, "{method:?} maxmove {cap}");
            }
        }
    }
}
