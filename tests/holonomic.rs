//! Distance constraints through the session: every method and
//! acceptance mode on closed-form minima with every iterate on the set,
//! real masses, refusals and failures, the regression cases of the
//! first review, and the minima ASE `FixBondLengths` reaches.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::{
    Accept, ConstraintError, ConstraintSolver, Control, DistanceConstraints, Error, ExpParams,
    FireKind, FirePreset, HessianObjective, HessianOracle, LineSearch, ManifoldKind, Method,
    Neighbours, NewtonKind, QnStep, Report, Solver, StopNorm,
};

const U: f64 = f64::EPSILON / 2.0;

/// Higham (2002) Lemma 3.1.
fn gamma(n: f64) -> f64 {
    n * U / (1.0 - n * U)
}

fn l2(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

fn join(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    a.iter().chain(b).copied().collect()
}

/// The UNCMIN gradient tolerance (Schnabel, Koontz and Weiss 1982), as
/// in `tests/ase_fire2_gold.py`.
fn gtol() -> f64 {
    f64::EPSILON.powf(1.0 / 3.0)
}

/// ASE FIRE defaults: `dt` 0.1 and `maxstep` 0.2 on the step norm.
/// `maxiter` caps the loop and fails the test when reached.
fn control() -> Control {
    Control {
        maxiter: 4000,
        gtol: gtol(),
        istep: 0.1,
        maxmove: Some(0.2),
        ..Default::default()
    }
}

/// Water: O at the origin, O-H 0.9572 Angstrom, H-O-H 104.52 degrees.
fn water() -> Array1<f64> {
    let half = 104.52_f64.to_radians() / 2.0;
    let (s, c) = (0.9572 * half.sin(), 0.9572 * half.cos());
    array![0.0, 0.0, 0.0, s, c, 0.0, -s, c, 0.0]
}

/// A step of about 0.07 per atom with in-plane and out-of-plane parts.
fn kick() -> Array1<f64> {
    array![0.05, -0.02, 0.03, -0.04, 0.06, 0.01, 0.02, 0.03, -0.05]
}

/// Lennard-Jones with epsilon = sigma = 1 over all pairs.
fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in i + 1..n {
            let d = [0, 1, 2].map(|c| x[3 * i + c] - x[3 * j + c]);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let s6 = r2.powi(-3);
            e += 4.0 * (s6 * s6 - s6);
            let f = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
            for c in 0..3 {
                g[3 * i + c] += f * d[c];
                g[3 * j + c] -= f * d[c];
            }
        }
    }
    (e, g)
}

/// The Lennard-Jones Hessian. With `u = r^2` the pair gradient is
/// `f(u) d`, `f = 24 (u^-4 - 2 u^-7)`, so each block is
/// `f I + 2 f'(u) d d^T`.
fn lj_hessian(x: ArrayView1<f64>) -> Array2<f64> {
    let n = x.len() / 3;
    let mut h = Array2::zeros((x.len(), x.len()));
    for i in 0..n {
        for j in i + 1..n {
            let d = [0, 1, 2].map(|c| x[3 * i + c] - x[3 * j + c]);
            let u = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let f = 24.0 * (u.powi(-4) - 2.0 * u.powi(-7));
            let df = 24.0 * (-4.0 * u.powi(-5) + 14.0 * u.powi(-8));
            for a in 0..3 {
                for b in 0..3 {
                    let v = if a == b { f } else { 0.0 } + 2.0 * df * d[a] * d[b];
                    h[(3 * i + a, 3 * i + b)] += v;
                    h[(3 * j + a, 3 * j + b)] += v;
                    h[(3 * i + a, 3 * j + b)] -= v;
                    h[(3 * j + a, 3 * i + b)] -= v;
                }
            }
        }
    }
    h
}

/// A session under test and whether it steps with the Hessian.
struct Case {
    name: String,
    solver: Solver,
    hessian: bool,
}

fn case(name: &str, method: Method, dim: usize, accept: Accept) -> Case {
    let mut solver = Solver::new(method, control(), dim);
    solver.set_accept(accept);
    Case {
        name: format!("{name} {accept:?}"),
        solver,
        hessian: false,
    }
}

/// Every method of the session, under every acceptance mode where the
/// method reads one, and every line search under the energy test.
fn cases(dim: usize) -> Vec<Case> {
    let mut out = Vec::new();
    for (name, kind) in [("fire", FireKind::V1), ("fire2", FireKind::V2)] {
        out.push(case(name, Method::Fire { kind }, dim, Accept::None));
    }
    let mut guenole = case(
        "fire2 guenole2020",
        Method::Fire { kind: FireKind::V2 },
        dim,
        Accept::None,
    );
    guenole
        .solver
        .set_fire_preset(FirePreset::Guenole2020)
        .unwrap();
    out.push(guenole);
    let mut abc = case(
        "fire2 abc",
        Method::Fire { kind: FireKind::V2 },
        dim,
        Accept::None,
    );
    abc.solver.set_fire_abc(true).unwrap();
    out.push(abc);
    for accept in [Accept::None, Accept::Energy, Accept::Nonmonotone] {
        out.push(case("lbfgs", Method::lbfgs(), dim, accept));
        out.push(case("steepest", Method::Steepest, dim, accept));
        out.push(case("cg", Method::polak_ribiere(), dim, accept));
        out.push(case("bfgs", Method::Bfgs, dim, accept));
        out.push(case("sr1", Method::Sr1, dim, accept));
        out.push(case("sr2", Method::Sr2, dim, accept));
        out.push(case("bb", Method::Bb, dim, accept));
    }
    let searches = [
        LineSearch::default(),
        LineSearch::Backtracking {
            c: 1e-4,
            beta: 0.5,
            maxiter: 40,
        },
        LineSearch::Goldstein {
            c: 0.25,
            beta: 0.5,
            maxiter: 40,
        },
    ];
    for search in searches {
        let mut c = case("lbfgs", Method::lbfgs(), dim, Accept::Energy);
        c.solver.set_linesearch(search);
        c.name = format!("lbfgs {search:?}");
        out.push(c);
    }
    for accept in [Accept::None, Accept::Energy] {
        let mut atom = case("lbfgs atom cap", Method::lbfgs(), dim, accept);
        atom.solver.set_atom_maxmove(0.2);
        out.push(atom);
    }
    out.push(case("newton-cg", Method::NewtonCg, dim, Accept::None));
    out.push(case("newton-pcg", Method::NewtonPcg, dim, Accept::None));
    let hessian = [
        (
            "newton",
            Method::Newton {
                kind: NewtonKind::Shifted,
            },
            None,
        ),
        (
            "rfo",
            Method::Newton {
                kind: NewtonKind::Rfo,
            },
            None,
        ),
        ("dogleg", Method::Dogleg, None),
        ("lbfgs qn newton", Method::lbfgs(), Some(QnStep::Newton)),
        ("lbfgs qn rfo", Method::lbfgs(), Some(QnStep::Rfo)),
        ("lbfgs qn two-loop", Method::lbfgs(), Some(QnStep::TwoLoop)),
    ];
    for (name, method, qn) in hessian {
        let mut c = case(name, method, dim, Accept::None);
        if let Some(qn) = qn {
            c.solver.set_qn_step(qn);
        }
        c.hessian = true;
        out.push(c);
    }
    out
}

/// Adam takes a fixed rate, so its iterates settle within about that rate
/// of the minimum rather than at the gradient stop; it is run for a fixed
/// count and checked on the set.
fn adam(dim: usize) -> Case {
    case("adam", Method::adam(), dim, Accept::None)
}

/// Steps to the projected-gradient stop, asserting every iterate on the
/// set at its tolerance.
fn run<F, H>(
    c: &mut Case,
    obj: &HessianOracle<F, H>,
    set: &DistanceConstraints,
    x: &mut Array1<f64>,
) -> Report
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    c.solver.set_distance_constraints(set.clone()).unwrap();
    for _ in 0..control().maxiter {
        let rep = if c.hessian {
            c.solver.step_hess(obj, x)
        } else {
            c.solver.step(obj, x)
        };
        let rep = rep.unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let excess = set.excess(x.view());
        assert!(
            excess <= 1.0,
            "{} step {}: excess {excess}",
            c.name,
            rep.steps
        );
        if rep.grad_norm < gtol() {
            return rep;
        }
    }
    panic!("{}: no stop in {} steps", c.name, control().maxiter);
}

/// Whether `c` stops on `obj` from `x` without a set.
fn stops_without_set<O: HessianObjective>(c: &mut Case, obj: &O, x: &Array1<f64>) -> bool {
    let mut x = x.clone();
    for _ in 0..control().maxiter {
        let rep = if c.hessian {
            c.solver.step_hess(obj, &mut x)
        } else {
            c.solver.step(obj, &mut x)
        };
        match rep {
            Ok(rep) if rep.grad_norm < gtol() => return true,
            Ok(_) => {}
            Err(_) => return false,
        }
    }
    false
}

/// Runs every case on the set. A case that stops on the energy without
/// the set must stop on it with the set and is returned with its point;
/// one that does not is run for the same budget with every iterate
/// checked on the set.
fn run_where_free_stops<F, H>(
    cases: impl Fn() -> Vec<Case>,
    obj: &HessianOracle<F, H>,
    set: &DistanceConstraints,
    start: &Array1<f64>,
) -> Vec<(Case, Report, Array1<f64>)>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    let mut out = Vec::new();
    for (mut free, mut c) in cases().into_iter().zip(cases()) {
        let mut x = start.clone();
        if stops_without_set(&mut free, obj, start) {
            let rep = run(&mut c, obj, set, &mut x);
            out.push((c, rep, x));
            continue;
        }
        println!("{}: no stop without the set, checked on it only", c.name);
        c.solver.set_distance_constraints(set.clone()).unwrap();
        for _ in 0..control().maxiter {
            let rep = if c.hessian {
                c.solver.step_hess(obj, &mut x)
            } else {
                c.solver.step(obj, &mut x)
            };
            let rep = rep.unwrap_or_else(|e| panic!("{}: {e}", c.name));
            let excess = set.excess(x.view());
            assert!(
                excess <= 1.0,
                "{} step {}: excess {excess}",
                c.name,
                rep.steps
            );
        }
    }
    out
}

type Energy = Box<dyn Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync>;
type Curvature = Box<dyn Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync>;

/// Two atoms tied to `a` and `b` by springs of stiffness `k`, bond `L`.
/// With `D = |a - b|`, `u = (a - b) / D` and `c = (a + b) / 2` the
/// minimum is `c + L u / 2` and `c - L u / 2`, whatever the masses.
struct Dumbbell {
    k: f64,
    bond: f64,
    a: Array1<f64>,
    b: Array1<f64>,
}

impl Dumbbell {
    fn new() -> Self {
        Self {
            k: 2.0,
            bond: 1.2,
            a: array![0.3, -0.1, 0.2],
            b: array![-0.1, 0.2, -0.05],
        }
    }

    fn oracle(&self) -> HessianOracle<Energy, Curvature> {
        let (k, ta, tb) = (self.k, self.a.clone(), self.b.clone());
        HessianOracle::unbounded(
            6,
            Box::new(move |x: ArrayView1<f64>| {
                let (d1, d2) = (
                    &x.slice(ndarray::s![..3]) - &ta,
                    &x.slice(ndarray::s![3..]) - &tb,
                );
                let e = 0.5 * k * (d1.dot(&d1) + d2.dot(&d2));
                (e, join(&(d1 * k), &(d2 * k)))
            }),
            Box::new(move |_| Array2::eye(6) * k),
        )
    }

    fn minimum(&self) -> Array1<f64> {
        let ab = &self.a - &self.b;
        let u = &ab / l2(&ab);
        let centre = (&self.a + &self.b) / 2.0;
        join(
            &(&centre + &(&u * (self.bond / 2.0))),
            &(&centre - &(&u * (self.bond / 2.0))),
        )
    }

    fn start(&self) -> Array1<f64> {
        let tilt = 30_f64.to_radians();
        let w = [tilt.cos(), tilt.sin(), 0.0];
        let h = self.bond / 2.0;
        array![
            0.5 + h * w[0],
            0.4 + h * w[1],
            0.1,
            0.5 - h * w[0],
            0.4 - h * w[1],
            0.1
        ]
    }

    /// Write the pair as centre offset `p` and bond at angle `theta` to
    /// `u`. In the metric of masses `m_1, m_2` the projected gradient
    /// bounds `|x - x*|` by `|g| / k_min` times `sqrt(2)` on the near side,
    /// with `k_min = min(k, k D / L)`; forming `x*` and `x - x*` rounds each
    /// coordinate at most 9.5 times at `scale`.
    fn tolerance(&self, x: &Array1<f64>, grad: f64) -> f64 {
        let dist = l2(&(&self.a - &self.b));
        let k_min = self.k.min(self.k * dist / self.bond);
        let scale = x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        2_f64.sqrt() * grad / k_min + (x.len() as f64).sqrt() * gamma(10.0) * scale
    }
}

#[test]
fn every_method_reaches_the_tethered_dumbbell_minimum_on_the_set() {
    let dumbbell = Dumbbell::new();
    let obj = dumbbell.oracle();
    let want = dumbbell.minimum();
    for masses in [None, Some(vec![1.0, 7.0])] {
        let mut set = DistanceConstraints::at(vec![(0, 1)], dumbbell.start().view()).unwrap();
        if let Some(m) = &masses {
            set = set.with_masses(m.clone()).unwrap();
        }
        for mut c in cases(6) {
            let mut x = dumbbell.start();
            let rep = run(&mut c, &obj, &set, &mut x);
            let off = l2(&(&x - &want));
            let tol = dumbbell.tolerance(&x, rep.grad_norm);
            assert!(off <= tol, "{} {masses:?}: |x - x*| {off} > {tol}", c.name);
        }
        let mut c = adam(6);
        c.solver.set_distance_constraints(set.clone()).unwrap();
        let mut x = dumbbell.start();
        let e0 = obj_value(&obj, &x);
        for _ in 0..200 {
            c.solver.step(&obj, &mut x).unwrap();
            assert!(set.excess(x.view()) <= 1.0, "adam off the set");
        }
        assert!(obj_value(&obj, &x) < e0, "adam did not descend");
    }
}

fn obj_value<O: Objective<f64>>(obj: &O, x: &Array1<f64>) -> f64 {
    obj.eval(x.view())
}

/// Lennard-Jones water with both O-H bonds fixed. Only `V(r)` of the H-H
/// distance moves the surface, so its minimum is `r_m = 2^(1/6)` with
/// any masses. To first order `|r - r_m| = |g| / (V''(r_m) rho)` with
/// `rho` the tangent part of the gradient of `r`, at least
/// `sqrt(2) cos(phi)` in the metric of masses at least one, and
/// `V''(r_m) = 144 2^(-4/3)`; the factor 2 covers the fall of the
/// secant within `V''(r_m) / (2 |V'''(r_m)|)` of `r_m`.
#[test]
fn every_method_closes_water_to_the_pair_minimum_with_real_masses() {
    let start = water();
    let r_m = 2_f64.powf(1.0 / 6.0);
    let v2 = 144.0 * 2_f64.powf(-4.0 / 3.0);
    let cos = (1.0 - (r_m / (2.0 * 0.9572)).powi(2)).sqrt();
    let obj = HessianOracle::unbounded(9, lj, lj_hessian);
    for masses in [None, Some(vec![15.999, 1.008, 1.008])] {
        let mut set = DistanceConstraints::at(vec![(0, 1), (0, 2)], start.view()).unwrap();
        if let Some(m) = &masses {
            set = set.with_masses(m.clone()).unwrap();
        }
        let floor = masses
            .as_ref()
            .map_or(1.0, |m| m.iter().fold(f64::MAX, |a, &b| a.min(b)));
        let rho = (2.0_f64).sqrt() * cos * floor.sqrt().min(1.0);
        for (c, rep, x) in run_where_free_stops(|| cases(9), &obj, &set, &start) {
            let r = l2(&array![x[3] - x[6], x[4] - x[7], x[5] - x[8]]);
            let off = (r - r_m).abs();
            let tol = 2.0 * rep.grad_norm / (v2 * rho) + gamma(6.0) * r_m;
            assert!(off <= tol, "{} {masses:?}: |r - r_m| {off} > {tol}", c.name);
        }
    }
}

/// One steepest step of length `t` moves `x` by `-t W Pi^T g` and then
/// back to the set along the bonds; that return is second order in `t`.
/// The heavy end of a bond with masses 1 and 3 moves a third as far.
#[test]
fn a_steepest_step_is_mass_weighted() {
    let x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let set = DistanceConstraints::new(vec![(0, 1)], vec![1.0])
        .unwrap()
        .with_masses(vec![1.0, 3.0])
        .unwrap();
    let force = array![0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let obj = HessianOracle::unbounded(
        6,
        move |p: ArrayView1<f64>| (-force.dot(&p), -force.clone()),
        |_| Array2::zeros((6, 6)),
    );
    let t = 1e-4;
    let control = Control {
        istep: t,
        maxmove: None,
        ..control()
    };
    let mut solver = Solver::new(Method::Steepest, control, 6);
    solver.set_distance_constraints(set.clone()).unwrap();
    let mut y = x.clone();
    solver.step(&obj, &mut y).unwrap();
    let dy = &y - &x;
    assert!((dy[1] - t).abs() <= t * t * 4.0, "light end {}", dy[1]);
    assert!(
        (dy[4] - t / 3.0).abs() <= t * t * 4.0,
        "heavy end {}",
        dy[4]
    );
    assert!(set.excess(y.view()) <= 1.0);
}

#[test]
fn refused_settings_are_typed() {
    let x = water();
    let set = DistanceConstraints::at(vec![(0, 1), (0, 2)], x.view()).unwrap();
    let obj = HessianOracle::unbounded(9, lj, lj_hessian);
    let mut pso = Solver::new(Method::pso(), control(), 9);
    assert!(matches!(
        pso.set_distance_constraints(set.clone()),
        Err(Error::Constraint(ConstraintError::Unsupported { .. }))
    ));
    let mut short = Solver::new(Method::lbfgs(), control(), 6);
    assert!(matches!(
        short.set_distance_constraints(set.clone()),
        Err(Error::Constraint(ConstraintError::OutOfRange { .. }))
    ));
    for kind in [
        ManifoldKind::Sphere,
        ManifoldKind::MwRigid,
        ManifoldKind::RigidQuotient,
    ] {
        let mut s = Solver::new(Method::lbfgs(), control(), 9);
        s.set_distance_constraints(set.clone()).unwrap();
        s.set_manifold(kind);
        let mut y = x.clone();
        assert!(unsupported(s.step(&obj, &mut y)), "{kind:?}");
        assert_eq!(y, x);
    }
}

/// A periodic cell needs the lattice for a minimum image, which the
/// session does not carry: refused when set before or after the set.
#[test]
fn a_periodic_cell_is_refused_either_way() {
    let x = water();
    let set = DistanceConstraints::at(vec![(0, 1), (0, 2)], x.view()).unwrap();
    let obj = HessianOracle::unbounded(9, lj, lj_hessian);
    let mut before = Solver::new(Method::Fire { kind: FireKind::V2 }, control(), 9);
    before.set_periodic(true);
    assert!(matches!(
        before.set_distance_constraints(set.clone()),
        Err(Error::Constraint(ConstraintError::Unsupported { .. }))
    ));
    let mut after = Solver::new(Method::Fire { kind: FireKind::V2 }, control(), 9);
    after.set_distance_constraints(set).unwrap();
    after.set_periodic(true);
    let mut y = x.clone();
    assert!(matches!(
        after.step(&obj, &mut y),
        Err(Error::Constraint(ConstraintError::Unsupported { .. }))
    ));
}

fn unsupported(r: Result<Report, Error>) -> bool {
    matches!(
        r,
        Err(Error::Constraint(ConstraintError::Unsupported { .. }))
    )
}

/// The exponential preconditioner is refused when set before or after
/// the set, and the refused step leaves `x` as it was.
#[test]
fn the_exponential_preconditioner_is_refused_either_way() {
    let x = water();
    let set = DistanceConstraints::at(vec![(0, 1), (0, 2)], x.view()).unwrap();
    let obj = HessianOracle::unbounded(9, lj, lj_hessian);
    let neighbours = Neighbours {
        i: vec![0, 0, 1],
        j: vec![1, 2, 2],
        ..Neighbours::default()
    };
    let params = ExpParams {
        r_nn: Some(1.0),
        mu: Some(1.0),
        ..ExpParams::default()
    };
    let mut before = Solver::new(Method::lbfgs(), control(), 9);
    before.set_exp(x.view(), &neighbours, &params).unwrap();
    assert!(matches!(
        before.set_distance_constraints(set.clone()),
        Err(Error::Constraint(ConstraintError::Unsupported { .. }))
    ));
    let mut after = Solver::new(Method::lbfgs(), control(), 9);
    after.set_distance_constraints(set).unwrap();
    after.set_exp(x.view(), &neighbours, &params).unwrap();
    let mut y = x.clone();
    assert!(unsupported(after.step(&obj, &mut y)));
    assert_eq!(y, x);
}

/// The HiGHS feasible-set step is refused when set before or after the
/// set, and the refused step leaves `x` as it was.
#[cfg(feature = "highs")]
#[test]
fn the_highs_step_is_refused_either_way() {
    let x = water();
    let set = DistanceConstraints::at(vec![(0, 1), (0, 2)], x.view()).unwrap();
    let obj = HessianOracle::unbounded(9, lj, lj_hessian);
    let mut before = Solver::new(Method::lbfgs(), control(), 9);
    before.set_highs(true);
    assert!(matches!(
        before.set_distance_constraints(set.clone()),
        Err(Error::Constraint(ConstraintError::Unsupported { .. }))
    ));
    let mut after = Solver::new(Method::lbfgs(), control(), 9);
    after.set_distance_constraints(set).unwrap();
    after.set_highs(true);
    let mut y = x.clone();
    assert!(unsupported(after.step(&obj, &mut y)));
    assert_eq!(y, x);
}

/// Rigid projection with a set and real masses: the projected gradient
/// carries no net force and no torque about the centre of mass, and the
/// iterates keep the centre of mass and every bond.
#[test]
fn rigid_projection_commutes_with_the_set() {
    let masses = vec![15.999, 1.008, 1.008, 4.0];
    let mut x = join(&water(), &array![0.4, 1.9, 0.3]);
    let set = DistanceConstraints::at(vec![(0, 1), (0, 2)], x.view())
        .unwrap()
        .with_masses(masses.clone())
        .unwrap();
    let obj = HessianOracle::unbounded(12, lj, lj_hessian);
    let com = |x: &Array1<f64>| {
        let total: f64 = masses.iter().sum();
        [0, 1, 2].map(|c| (0..4).map(|a| masses[a] * x[3 * a + c]).sum::<f64>() / total)
    };
    let c0 = com(&x);
    for method in [Method::lbfgs(), Method::Fire { kind: FireKind::V2 }] {
        let mut solver = Solver::new(method, control(), 12);
        solver.set_project_rigid(true);
        solver.set_distance_constraints(set.clone()).unwrap();
        for _ in 0..50 {
            let rep = solver.step(&obj, &mut x).unwrap();
            assert!(set.excess(x.view()) <= 1.0);
            let g = rep.grad.unwrap();
            let net = [0, 1, 2].map(|c| (0..4).map(|a| g[3 * a + c]).sum::<f64>());
            let scale = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(
                net.iter().all(|f| f.abs() <= 1e3 * U * scale.max(1.0)),
                "net {net:?}"
            );
        }
        let c = com(&x);
        for k in 0..3 {
            assert!(
                (c[k] - c0[k]).abs() <= 1e-6,
                "centre of mass moved {c:?} from {c0:?}"
            );
        }
    }
}

/// The linear energy `-pull . x` in a box.
struct Pull {
    pull: Array1<f64>,
    bounds: Bounds<f64>,
}

impl Objective<f64> for Pull {
    fn dim(&self) -> usize {
        self.pull.len()
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        -self.pull.dot(&x)
    }
}

impl Gradient<f64> for Pull {
    fn dim(&self) -> usize {
        self.pull.len()
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        -self.pull.clone()
    }
}

impl DifferentiableObjective<f64> for Pull {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

/// A constrained atom is never clipped: a trial that takes it to its box
/// is a typed error and leaves the point inside. A bound on a free atom
/// binds while every iterate stays on the set.
#[test]
fn bounds_and_the_set_act_on_different_coordinates() {
    let start = join(&water(), &array![0.4, 1.9, 0.3]);
    let set = DistanceConstraints::at(vec![(0, 1), (0, 2)], start.view()).unwrap();
    let bounded = |k: usize, sign: f64, low: Array1<f64>, high: Array1<f64>| {
        let mut pull = Array1::zeros(12);
        pull[k] = sign;
        Pull {
            pull,
            bounds: Bounds::new(low, high, 0.0),
        }
    };
    let mut low = Array1::from_elem(12, -10.0);
    let mut high = Array1::from_elem(12, 10.0);
    high[9] = 0.6;
    let free = bounded(9, 1.0, low.clone(), high.clone());
    for kind in [FireKind::V1, FireKind::V2] {
        let mut solver = Solver::new(Method::Fire { kind }, control(), 12);
        solver.set_distance_constraints(set.clone()).unwrap();
        let mut x = start.clone();
        for _ in 0..100 {
            solver.step(&free, &mut x).unwrap();
            assert!(set.excess(x.view()) <= 1.0);
            assert!(x[9] <= 0.6);
        }
        assert_eq!(x[9], 0.6, "{kind:?}: the free atom did not reach its bound");
    }
    low[3] = start[3] - 0.05;
    let tight = bounded(3, -1.0, low, high);
    let mut solver = Solver::new(Method::Steepest, control(), 12);
    solver.set_distance_constraints(set.clone()).unwrap();
    let mut x = start.clone();
    let mut hit = None;
    for _ in 0..100 {
        match solver.step(&tight, &mut x) {
            Ok(_) => assert!(x[3] > start[3] - 0.05),
            Err(e) => {
                hit = Some(e);
                break;
            }
        }
    }
    assert!(
        matches!(
            hit,
            Some(Error::Constraint(ConstraintError::Bound { atom: 1 }))
        ),
        "{hit:?}"
    );
    assert!(x[3] > start[3] - 0.05);
}

/// Rigid projection moves every atom, so at an active bound it and the
/// box projection would not commute: the step is refused and `x` stays.
#[test]
fn rigid_projection_at_an_active_bound_is_refused() {
    let x = join(&water(), &array![0.4, 1.9, 0.3]);
    let set = DistanceConstraints::at(vec![(0, 1), (0, 2)], x.view()).unwrap();
    let mut high = Array1::from_elem(12, 10.0);
    high[9] = x[9];
    let mut pull = Array1::zeros(12);
    pull[9] = 1.0;
    let obj = Pull {
        pull,
        bounds: Bounds::new(Array1::from_elem(12, -10.0), high, 0.0),
    };
    for method in [Method::lbfgs(), Method::Fire { kind: FireKind::V2 }] {
        let mut solver = Solver::new(method, control(), 12);
        solver.set_project_rigid(true);
        solver.set_distance_constraints(set.clone()).unwrap();
        let mut y = x.clone();
        assert!(unsupported(solver.step(&obj, &mut y)));
        assert_eq!(y, x);
    }
}

#[test]
fn a_start_off_the_set_is_returned_to_it() {
    let x = water();
    let set = DistanceConstraints::at(vec![(0, 1), (0, 2), (1, 2)], x.view()).unwrap();
    let obj = HessianOracle::unbounded(9, lj, lj_hessian);
    for solver in [
        ConstraintSolver::Auto,
        ConstraintSolver::Shake,
        ConstraintSolver::Lincs,
    ] {
        let set = set.clone().with_settle(false).with_solver(solver);
        let mut s = Solver::new(Method::Fire { kind: FireKind::V2 }, control(), 9);
        s.set_distance_constraints(set.clone()).unwrap();
        let mut y = &x + &(kick() * 0.5);
        assert!(set.excess(y.view()) > 1.0);
        s.step(&obj, &mut y).unwrap();
        assert!(set.excess(y.view()) <= 1.0, "{solver:?}");
    }
}

/// Unit bonds from a hub at the origin to three ends on a cone of
/// half-angle `cone`, in degrees.
fn hub(cone: f64) -> Array1<f64> {
    let cone = cone.to_radians();
    let mut x = Array1::zeros(12);
    for k in 0..3 {
        let phi = 2.0 * std::f64::consts::PI * k as f64 / 3.0;
        x[3 + 3 * k] = cone.sin() * phi.cos();
        x[4 + 3 * k] = cone.sin() * phi.sin();
        x[5 + 3 * k] = cone.cos();
    }
    x
}

/// The hub with mass 1 and its ends with mass 100. The coupling of the
/// three bonds is about `cos(theta)` between bonds at angle `theta`: at
/// a cone of 30 degrees it has an eigenvalue of 2.24 and the LINCS
/// series diverges, and at the tetrahedral cone, 70.53 degrees, the
/// series converges with spectral radius 0.66.
fn light_hub_set(x: &Array1<f64>) -> DistanceConstraints {
    DistanceConstraints::at(vec![(0, 1), (0, 2), (0, 3)], x.view())
        .unwrap()
        .with_masses(vec![1.0, 100.0, 100.0, 100.0])
        .unwrap()
}

/// LINCS where its expansion diverges is refused; `Auto` takes SHAKE.
#[test]
fn an_unsafe_lincs_block_is_a_typed_error_and_auto_takes_shake() {
    let x = hub(30.0);
    let set = light_hub_set(&x);
    let obj = HessianOracle::unbounded(12, lj, lj_hessian);
    let mut lincs = Solver::new(Method::lbfgs(), control(), 12);
    lincs
        .set_distance_constraints(set.clone().with_solver(ConstraintSolver::Lincs))
        .unwrap();
    let mut y = &x + &(Array1::from_elem(12, 0.01));
    assert!(matches!(
        lincs.step(&obj, &mut y),
        Err(Error::Constraint(ConstraintError::LincsUnsafe { .. }))
    ));
    let mut auto = Solver::new(Method::lbfgs(), control(), 12);
    auto.set_distance_constraints(set.clone()).unwrap();
    let mut y = &x + &(Array1::from_elem(12, 0.01));
    auto.step(&obj, &mut y).unwrap();
    assert!(set.excess(y.view()) <= 1.0);
}

/// Corrections run along the anchor bonds, so a step that moves the ends
/// of a unit bond sideways by more than half its length apart leaves a
/// bond no correction along the anchor can shorten to its length: under
/// `Accept::None` it is a typed error and `x` stays where it was.
#[test]
fn a_step_the_anchor_bonds_cannot_close_is_a_typed_error() {
    let x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let set = DistanceConstraints::at(vec![(0, 1)], x.view()).unwrap();
    let force = array![0.0, 3.0, 0.0, 0.0, -3.0, 0.0];
    let obj = HessianOracle::unbounded(
        6,
        move |p: ArrayView1<f64>| (-force.dot(&p), -force.clone()),
        |_| Array2::zeros((6, 6)),
    );
    let control = Control {
        istep: 1.0,
        maxmove: None,
        ..control()
    };
    let mut solver = Solver::new(Method::Steepest, control, 6);
    // The default per-atom cap would shorten the sideways step enough
    // that the anchor correction can still close the bond.
    solver.set_atom_maxmove(0.0);
    solver.set_distance_constraints(set).unwrap();
    let mut y = x.clone();
    let r = solver.step(&obj, &mut y);
    assert!(matches!(r, Err(Error::Constraint(_))), "{r:?}");
    assert_eq!(y, x);
}

fn step<F, H>(c: &mut Case, obj: &HessianOracle<F, H>, x: &mut Array1<f64>) -> Result<Report, Error>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    if c.hessian {
        c.solver.step_hess(obj, x)
    } else {
        c.solver.step(obj, x)
    }
}

/// A start whose disjoint unit bond is stretched from 1 to 1.4, ahead of
/// a hub where LINCS diverges, under a zero objective. Returning the start
/// to the set closes the bond, its free end going from 3.4 to 3.2, and
/// then fails on the hub. The step is a typed error, `x` is the start as
/// the caller passed it, and the session memory is unchanged.
#[test]
fn a_failed_return_of_the_start_leaves_x_and_the_memory() {
    let start = join(&array![2.0, 0.0, 0.0, 3.0, 0.0, 0.0], &hub(30.0));
    let pairs = vec![(0, 1), (2, 3), (2, 4), (2, 5)];
    let set = DistanceConstraints::at(pairs, start.view())
        .unwrap()
        .with_masses(vec![1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
        .unwrap()
        .with_solver(ConstraintSolver::Lincs);
    let zero = HessianOracle::unbounded(
        18,
        |p: ArrayView1<f64>| (0.0, Array1::zeros(p.len())),
        |_| Array2::zeros((18, 18)),
    );
    let mut stretched = start.clone();
    stretched[3] = 3.4;
    let g = Array1::from_shape_fn(18, |k| (k as f64 + 1.0).cos());
    for method in [Method::Steepest, Method::lbfgs()] {
        let mut solver = Solver::new(method, control(), 18);
        solver.set_distance_constraints(set.clone()).unwrap();
        let s = Array1::from_shape_fn(18, |k| (k as f64).sin());
        solver.push_pair(s.view(), s.view());
        let pairs = solver.pair_count();
        let dir = solver.search_direction(g.view()).unwrap();
        let mut x = stretched.clone();
        let r = solver.step(&zero, &mut x);
        assert!(
            matches!(
                r,
                Err(Error::Constraint(ConstraintError::LincsUnsafe { .. }))
            ),
            "{r:?}"
        );
        assert_eq!(x, stretched);
        assert_eq!(solver.pair_count(), pairs);
        assert_eq!(solver.search_direction(g.view()).unwrap(), dir);
    }
}

/// The caller narrows a hub, on the set, until LINCS diverges there. The
/// next step carries the memory to the new point and records the move as
/// a pair, and then no trial can be placed. Every method whose step fails
/// leaves `x` as the caller passed it, and its next step from the point
/// before the move matches, bit for bit, a twin session that never took
/// the failed step.
#[test]
fn a_failed_step_after_a_move_leaves_x_and_the_memory() {
    let wide = hub(70.528_779_365_509_31);
    let set = light_hub_set(&wide).with_solver(ConstraintSolver::Lincs);
    let mut narrow = hub(30.0);
    let anchor = narrow.clone();
    set.clone()
        .with_solver(ConstraintSolver::Shake)
        .constrain(anchor.view(), &mut narrow)
        .unwrap();
    let target = &wide + &Array1::from_shape_fn(12, |k| 0.05 * (0.7 * k as f64).sin());
    let pull = move |p: ArrayView1<f64>| {
        let d = &p - &target;
        (0.5 * d.dot(&d), d)
    };
    let obj = HessianOracle::unbounded(12, pull, |_| Array2::eye(12));
    let g = Array1::from_shape_fn(12, |k| (k as f64 + 1.0).cos());
    let mut failed = Vec::new();
    let twins = cases(12).into_iter().zip(cases(12));
    for (mut a, mut b) in twins.chain([(adam(12), adam(12))]) {
        a.solver.set_distance_constraints(set.clone()).unwrap();
        b.solver.set_distance_constraints(set.clone()).unwrap();
        let (mut xa, mut xb) = (wide.clone(), wide.clone());
        for _ in 0..3 {
            step(&mut a, &obj, &mut xa).unwrap_or_else(|e| panic!("{}: {e}", a.name));
            step(&mut b, &obj, &mut xb).unwrap();
        }
        let pairs = a.solver.pair_count();
        let dir = a.solver.search_direction(g.view()).unwrap();
        let mut moved = narrow.clone();
        match step(&mut a, &obj, &mut moved) {
            Err(Error::Constraint(_)) => failed.push(a.name.clone()),
            // Trust-region steps count a trial the set refuses as a
            // rejected model step.
            Ok(_) => continue,
            Err(e) => panic!("{}: {e}", a.name),
        }
        assert_eq!(moved, narrow, "{}", a.name);
        assert_eq!(a.solver.pair_count(), pairs, "{}", a.name);
        assert_eq!(
            a.solver.search_direction(g.view()).unwrap(),
            dir,
            "{}",
            a.name
        );
        let ra = step(&mut a, &obj, &mut xa).unwrap();
        let rb = step(&mut b, &obj, &mut xb).unwrap();
        assert_eq!(xa, xb, "{}", a.name);
        assert_eq!((ra.value, ra.steps), (rb.value, rb.steps), "{}", a.name);
    }
    for name in [
        "lbfgs None",
        "lbfgs Energy",
        "lbfgs Nonmonotone",
        "lbfgs qn two-loop None",
        "cg None",
        "fire2 None",
    ] {
        assert!(failed.iter().any(|f| f == name), "{name}: {failed:?}");
    }
}

#[test]
fn projections_report_dependent_and_non_finite_anchors() {
    let line = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let set = DistanceConstraints::new(vec![(0, 1), (1, 2), (0, 2)], vec![1.0, 1.0, 1.9]).unwrap();
    let mut v = kick();
    assert!(matches!(
        set.project_vector(line.view(), &mut v),
        Err(Error::Constraint(ConstraintError::Dependent { .. }))
    ));
    let mut g = kick();
    assert!(matches!(
        set.project_gradient(line.view(), &mut g),
        Err(Error::Constraint(ConstraintError::Dependent { .. }))
    ));
    let mut nan = water();
    nan[4] = f64::NAN;
    let set = DistanceConstraints::at(vec![(0, 1), (0, 2)], water().view()).unwrap();
    let mut v = kick();
    assert!(matches!(
        set.project_vector(nan.view(), &mut v),
        Err(Error::Constraint(ConstraintError::NonFinite))
    ));
    let mut y = nan.clone();
    assert!(matches!(
        set.constrain(water().view(), &mut y),
        Err(Error::Constraint(ConstraintError::NonFinite))
    ));
}

/// A step whose pair fails the curvature test must leave the pair before
/// it in place, with or without a set.
#[test]
fn a_dropped_pair_keeps_the_previous_one() {
    let concave = |x: ArrayView1<f64>| {
        let e = x[6].cos();
        let mut g = Array1::zeros(x.len());
        g[6] = -x[6].sin();
        (e, g)
    };
    let start = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 3.0, 0.0];
    let set = DistanceConstraints::at(vec![(0, 1)], start.view()).unwrap();
    for constrained in [false, true] {
        let obj = HessianOracle::unbounded(9, concave, |_| Array2::zeros((9, 9)));
        let mut solver = Solver::new(Method::lbfgs(), control(), 9);
        if constrained {
            solver.set_distance_constraints(set.clone()).unwrap();
        }
        let mut s = Array1::zeros(9);
        s[6] = 1.0;
        assert!(solver.push_pair(s.view(), s.view()));
        assert_eq!(solver.pair_count(), 1);
        let mut x = start.clone();
        solver.step(&obj, &mut x).unwrap();
        assert!(x[6] > 0.5, "constrained {constrained}: no move");
        assert_eq!(solver.pair_count(), 1, "constrained {constrained}");
    }
}

/// The how-to row for the per-atom cap names what a set changes.
#[test]
fn the_session_how_to_describes_the_cap_with_a_set() {
    let rst = include_str!("../docs/source/howto/session.rst");
    let org = include_str!("../docs/orgmode/howto/session.org");
    for text in [rst, org] {
        assert!(text.contains("set_distance_constraints"));
        assert!(
            !text.contains("Per-atom cap. Euclidean L-BFGS takes the geometry step and ignores")
        );
    }
}

struct AseRun {
    label: String,
    steps: usize,
    calls: usize,
    grad: f64,
    end: Array1<f64>,
}

struct AseSystem {
    name: String,
    k_min: f64,
    start: Array1<f64>,
    pairs: Vec<(usize, usize)>,
    /// `(i, j, r0)` of unit springs, or none for Lennard-Jones.
    springs: Vec<(usize, usize, f64)>,
    /// Real masses where they differ from one another.
    masses: Option<Vec<f64>>,
    runs: Vec<AseRun>,
}

/// `E = sum (r_ij - r0)^2 / 2` over `springs`.
fn springs(x: ArrayView1<f64>, springs: &[(usize, usize, f64)]) -> (f64, Array1<f64>) {
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for &(i, j, r0) in springs {
        let d = [0, 1, 2].map(|c| x[3 * i + c] - x[3 * j + c]);
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        e += 0.5 * (r - r0) * (r - r0);
        for c in 0..3 {
            g[3 * i + c] += (r - r0) * d[c] / r;
            g[3 * j + c] -= (r - r0) * d[c] / r;
        }
    }
    (e, g)
}

/// `tests/data/ase_fix_bond_lengths.txt`, written by
/// `tests/ase_fix_bond_lengths.py`.
fn ase_systems() -> (f64, Vec<AseSystem>) {
    let text = include_str!("data/ase_fix_bond_lengths.txt");
    let floats = |w: &[&str]| -> Array1<f64> { w.iter().map(|v| v.parse().unwrap()).collect() };
    let mut fmax = f64::NAN;
    let mut systems: Vec<AseSystem> = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let w: Vec<&str> = line.split_whitespace().collect();
        let last = systems.last_mut();
        match (w[0], last) {
            ("fmax", _) => fmax = w[1].parse().unwrap(),
            ("system", _) => systems.push(AseSystem {
                name: w[1].to_string(),
                k_min: w[4].parse().unwrap(),
                start: Array1::zeros(0),
                pairs: Vec::new(),
                springs: Vec::new(),
                masses: None,
                runs: Vec::new(),
            }),
            ("start", Some(s)) => s.start = floats(&w[1..]),
            ("pairs", Some(s)) => {
                let ids: Vec<usize> = w[1..].iter().map(|v| v.parse().unwrap()).collect();
                s.pairs = ids.chunks(2).map(|p| (p[0], p[1])).collect();
            }
            ("springs", Some(s)) => {
                let t = &w[1..];
                let parse = |k: usize| t[k].parse::<usize>().unwrap();
                s.springs = (0..t.len() / 3)
                    .map(|k| {
                        (
                            parse(3 * k),
                            parse(3 * k + 1),
                            t[3 * k + 2].parse().unwrap(),
                        )
                    })
                    .collect();
            }
            ("lj", Some(_)) => {}
            ("masses", Some(s)) => s.masses = Some(floats(&w[1..]).to_vec()),
            ("run", Some(s)) => s.runs.push(AseRun {
                label: w[1].to_string(),
                steps: w[2].parse().unwrap(),
                calls: w[3].parse().unwrap(),
                grad: w[4].parse().unwrap(),
                end: Array1::zeros(0),
            }),
            ("end", Some(s)) => s.runs.last_mut().unwrap().end = floats(&w[1..]),
            _ => panic!("unexpected line {line}"),
        }
    }
    (fmax, systems)
}

fn pair_distances(x: &Array1<f64>) -> Vec<f64> {
    let n = x.len() / 3;
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let d = [0, 1, 2].map(|c| x[3 * i + c] - x[3 * j + c]);
            out.push((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        }
    }
    out
}

/// Each rgmin run, with unit and with real masses, against the ASE FIRE
/// minimum, both stopped below `fmax` in the largest per-atom norm. To
/// first order a stop at unit-mass projected gradient `g` lies within
/// `|g|_2 / k_min` of the orbit of minima under rigid motion, with
/// `k_min` the smallest Lagrangian curvature off the rigid motions. A
/// pair distance is invariant on the orbit and moves at most `sqrt(2)`
/// times as far as the point. The factor 2 lets `k_min` halve between
/// the two ends.
#[test]
fn minima_match_ase_fix_bond_lengths() {
    let (fmax, systems) = ase_systems();
    assert_eq!(fmax, gtol());
    for sys in &systems {
        let reference = sys.runs.iter().find(|r| r.label == "fire-matched").unwrap();
        assert!(sys.k_min > 0.0);
        let want = pair_distances(&reference.end);
        let unit = DistanceConstraints::at(sys.pairs.clone(), sys.start.view()).unwrap();
        let dim = sys.start.len();
        let toy = sys.springs.clone();
        let f = move |x: ArrayView1<f64>| {
            if toy.is_empty() {
                lj(x)
            } else {
                springs(x, &toy)
            }
        };
        let obj = HessianOracle::unbounded(dim, f.clone(), move |_| Array2::zeros((dim, dim)));
        let first_order = || {
            let mut out: Vec<Case> = cases(dim).into_iter().filter(|c| !c.hessian).collect();
            out.iter_mut()
                .for_each(|c| c.solver.set_stop_norm(StopNorm::MaxAtom));
            out
        };
        let mut sets = vec![("unit", unit.clone())];
        if let Some(m) = &sys.masses {
            sets.push(("real", unit.clone().with_masses(m.clone()).unwrap()));
        }
        for (masses, set) in &sets {
            for (c, rep, x) in run_where_free_stops(first_order, &obj, set, &sys.start) {
                let mut g = f(x.view()).1;
                unit.project_gradient(x.view(), &mut g).unwrap();
                let tol = 2.0 * 2_f64.sqrt() * (l2(&g) + reference.grad) / sys.k_min;
                let off = pair_distances(&x)
                    .iter()
                    .zip(&want)
                    .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
                assert!(
                    off <= tol,
                    "{} {} {masses}: pair distances off by {off} > {tol}",
                    sys.name,
                    c.name
                );
                println!(
                    "{:7} {:28} {masses} {:5} | fire-matched {:5} steps {:5} calls",
                    sys.name, c.name, rep.steps, reference.steps, reference.calls
                );
            }
        }
    }
}
