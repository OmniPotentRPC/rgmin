//! Preconditioned FIRE, ASE `PreconFIRE` (Packwood et al. 2016).
//!
//! The Euclidean integrator is what a new session runs. Installing a
//! preconditioner opts into the metric update. `set_fire_precon(false)`
//! turns that update off again.

use eindir_core::{DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, Exp, ExpParams, FireKind, IdentityPrecond, Method, Neighbours, Oracle, Preconditioner,
    Solver,
};

fn rosenbrock(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (mut e, mut g) = (0.0, Array1::zeros(x.len()));
    for i in 0..x.len() - 1 {
        let t = x[i + 1] - x[i] * x[i];
        let u = 1.0 - x[i];
        e += 100.0 * t * t + u * u;
        g[i] += -400.0 * x[i] * t - 2.0 * u;
        g[i + 1] += 200.0 * t;
    }
    (e, g)
}

fn bits(x: &Array1<f64>) -> String {
    x.iter()
        .map(|v| format!("{:016x}", v.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn assert_bits(x: &Array1<f64>, hexes: &str) {
    assert_eq!(bits(x), hexes, "{x:?}");
}

/// Recorded from the Euclidean integrator. A change to plain FIRE or
/// FIRE 2.0 moves these bits.
const V1_DEFAULT: [&str; 4] = [
    "406acccccccccccd,c088b80000000001,407b6cccccccccce",
    "c1cde0761d59999a,422707f8c2ae0003,417dd0754ccccccf",
    "426d1a94a2000000,c26d1a94a2000000,426d1a94a2000000",
    "c26d1a94a2000000,426d1a94a2000000,426d1a94a2000000",
];
const V2_DEFAULT: [&str; 4] = [
    "406acccccccccccd,c088b80000000001,407b6cccccccccce",
    "c1bde2878f000000,4216edcd2bf40004,417db7c72cccccd1",
    "426d1a94a2000000,c26d1a94a2000000,426d1a94a2000000",
    "c26d1a94a2000000,426d1a94a2000000,426d1a94a2000000",
];
const V1_CAPPED: [&str; 8] = [
    "bff2758dfe5e0df2,3fea8eaff401613c,bff1b02b2fde3ffd",
    "bff18585eb244c4d,3fe5618e87199baf,bff00e75b406bf02",
    "bff05cefdfb8609c,3fe08fc2d6a83a5e,bfec98c7eb479652",
    "bfedf2f967876dca,3fd865bb01e37471,bfe8d3a0d3702f12",
    "bfeabdb9a66bf298,3fd0c280caed2d78,bfe4d0e5c13928f4",
    "bfe72c5610bb16c6,3fc4a6994630520c,bfe0966bcef15f4e",
    "bfe354486f61bb19,3fb4c0aa16fabe1c,bfd852ebbfb2e04e",
    "bfde9aa62cbcf4b6,3f95ddecd4e2b426,bfce47236debd928",
];
const V2_CAPPED: [&str; 8] = [
    "bff2758dfe5e0df2,3fea8eaff401613c,bff1b02b2fde3ffd",
    "bff151bd9bc2c482,3fe5b428ef347d6a,bfefe1fd429f6bff",
    "bfefb045f78341fc,3fe1a228dbed2886,bfebebf6575ffa36",
    "bfec6c226dcf7436,3fdcaee9e2e7da56,bfe782c4c0846b93",
    "bfe92a8291c8d64b,3fd772d880f975f1,bfe2a99bab390c1d",
    "bfe6191e67db9644,3fd365213b3449b5,bfdad9336a696a5a",
    "bfe342b348df31ac,3fd06d1ff54c2241,bfcf88391ca60e25",
    "bfe095a8d4561bd2,3fccf81560604b80,bfb135248ae3e1a0",
];
const V1_FILL: &str = "bfde9aa62cbcf4b4,3f95ddecd4e2b3f2,bfce47236debd92c";
const V2_FILL: &str = "bfe0b400a0738993,3fcdb408576b9a0d,bfafd67898cf2c84";

fn control(istep: f64, maxmove: Option<f64>, steps: usize) -> Control {
    Control {
        maxiter: steps,
        gtol: 0.0,
        istep,
        maxmove,
        ..Control::default()
    }
}

fn replay(kind: FireKind, istep: f64, maxmove: Option<f64>, expect: &[&str]) {
    let mut solver = Solver::new(
        Method::Fire { kind },
        control(istep, maxmove, expect.len()),
        3,
    );
    let obj = Oracle::unbounded(3, rosenbrock);
    let mut x = array![-1.2, 1.0, -1.2];
    for hexes in expect {
        solver.step(&obj, &mut x).unwrap();
        assert_bits(&x, hexes);
    }
}

#[test]
fn default_fire_trajectories_stay_pinned() {
    replay(FireKind::V1, 1.0, None, &V1_DEFAULT);
    replay(FireKind::V2, 1.0, None, &V2_DEFAULT);
    replay(FireKind::V1, 0.1, Some(0.2), &V1_CAPPED);
    replay(FireKind::V2, 0.1, Some(0.2), &V2_CAPPED);
    for (kind, hexes) in [(FireKind::V1, V1_FILL), (FireKind::V2, V2_FILL)] {
        let mut solver = Solver::new(Method::Fire { kind }, control(1.0, Some(0.2), 8), 3);
        let mut x = array![-1.2, 1.0, -1.2];
        solver
            .run_fill(&mut x, |v, g| {
                let (e, grad) = rosenbrock(v);
                g.assign(&grad);
                e
            })
            .unwrap();
        assert_bits(&x, hexes);
    }
}

struct Diag(Vec<f64>);

impl Preconditioner for Diag {
    fn solve(&self, r: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter(r.iter().zip(self.0.iter()).map(|(v, d)| v / d))
    }
    fn apply(&self, v: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter(v.iter().zip(self.0.iter()).map(|(x, d)| x * d))
    }
}

fn capped_walk(kind: FireKind, install: impl FnOnce(&mut Solver)) -> Vec<Array1<f64>> {
    let mut solver = Solver::new(Method::Fire { kind }, control(0.1, Some(0.2), 8), 3);
    install(&mut solver);
    let obj = Oracle::unbounded(3, rosenbrock);
    let mut x = array![-1.2, 1.0, -1.2];
    let mut out = Vec::new();
    for _ in 0..8 {
        solver.step(&obj, &mut x).unwrap();
        out.push(x.clone());
    }
    out
}

#[test]
fn the_option_is_honoured_and_identity_matches_plain_fire() {
    for kind in [FireKind::V1, FireKind::V2] {
        let plain = capped_walk(kind, |_| {});
        let identity = capped_walk(kind, |s| {
            s.set_preconditioner(IdentityPrecond).unwrap();
        });
        let ones = capped_walk(kind, |s| {
            s.set_preconditioner(Diag(vec![1.0, 1.0, 1.0])).unwrap();
        });
        let stretched = capped_walk(kind, |s| {
            s.set_preconditioner(Diag(vec![1.0, 4.0, 9.0])).unwrap();
        });
        let turned_off = capped_walk(kind, |s| {
            s.set_preconditioner(Diag(vec![1.0, 4.0, 9.0])).unwrap();
            s.set_fire_precon(false).unwrap();
        });
        assert_eq!(identity, plain, "{kind:?} identity");
        assert_eq!(ones, plain, "{kind:?} diagonal identity");
        assert_eq!(turned_off, plain, "{kind:?} option off");
        assert_ne!(stretched, plain, "{kind:?} option on");
    }
    let mut lbfgs = Solver::new(Method::Lbfgs { memory: 3 }, Control::default(), 2);
    assert!(lbfgs.set_fire_precon(true).is_err());
    assert!(lbfgs.set_preconditioner(IdentityPrecond).is_err());
}

fn quad(d: [f64; 2]) -> impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) {
    move |x: ArrayView1<f64>| {
        let g = array![d[0] * x[0], d[1] * x[1]];
        (0.5 * (d[0] * x[0] * x[0] + d[1] * x[1] * x[1]), g)
    }
}

#[test]
fn the_kick_uses_the_preconditioned_force_and_the_cap_stays_cartesian() {
    let d = [100.0, 1.0];
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V1 },
        Control {
            maxiter: 1,
            gtol: 0.0,
            istep: 0.2,
            maxmove: Some(0.01),
            ..Control::default()
        },
        2,
    );
    solver.set_preconditioner(Diag(d.to_vec())).unwrap();
    let obj = Oracle::unbounded(2, quad(d));
    let mut x = array![1.0, 0.0];
    solver.step(&obj, &mut x).unwrap();
    // P^{-1} F = -x, so the uncapped kick is dt^2 (-1, 0). The Euclidean
    // cap of 0.01 shortens that to (-0.01, 0). The P-norm of the same
    // kick is 0.4, and capping there would land at 0.999.
    assert!((x[0] - 0.99).abs() < 1e-12, "cartesian cap {x:?}");
    assert!(x[1].abs() < 1e-15, "{x:?}");
    assert!((x[0] - 0.999).abs() > 1e-4, "capped in the P norm {x:?}");
}

fn chain(n: usize) -> (Array1<f64>, Neighbours, ExpParams) {
    let mut pos = Array1::zeros(3 * n);
    let mut neigh = Neighbours::default();
    for i in 0..n {
        pos[3 * i] = i as f64;
        if i + 1 < n {
            neigh.i.push(i);
            neigh.j.push(i + 1);
            neigh.r.push(1.0);
        }
    }
    let params = ExpParams {
        r_nn: Some(1.0),
        mu: Some(1.0),
        ..ExpParams::default()
    };
    (pos, neigh, params)
}

struct FixedMetric {
    p: Exp,
    bounds: eindir_core::Bounds<f64>,
}

impl Objective<f64> for FixedMetric {
    fn dim(&self) -> usize {
        self.p.dim()
    }
    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * x.dot(&self.p.apply(x))
    }
}

impl Gradient<f64> for FixedMetric {
    fn dim(&self) -> usize {
        self.p.dim()
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.p.apply(x)
    }
}

impl DifferentiableObjective<f64> for FixedMetric {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let g = self.p.apply(x);
        (0.5 * x.dot(&g), g)
    }
}

fn chain_run(precon: bool, enabled: bool) -> (usize, f64) {
    let n = 12;
    let (geom, neigh, params) = chain(n);
    let metric = Exp::assemble(geom.view(), &neigh, &params, 1.0, 1.0).unwrap();
    let obj = FixedMetric {
        p: metric,
        bounds: eindir_core::Bounds::new(
            Array1::from_elem(3 * n, -1e6),
            Array1::from_elem(3 * n, 1e6),
            0.0,
        ),
    };
    let mut x = geom.clone();
    x[0] = 0.4;
    x[7] = -0.25;
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V1 },
        Control {
            maxiter: 1,
            gtol: 1e-8,
            istep: 0.2,
            maxmove: None,
            ..Control::default()
        },
        x.len(),
    );
    if precon {
        solver.set_exp(geom.view(), &neigh, &params).unwrap();
        solver.set_fire_precon(enabled).unwrap();
    }
    let mut taken = 0;
    let mut gnorm = 0.0;
    for _ in 0..250 {
        let g = obj.grad(x.view());
        gnorm = g.dot(&g).sqrt();
        if gnorm < 1e-8 {
            break;
        }
        solver.step(&obj, &mut x).unwrap();
        taken += 1;
    }
    (taken, gnorm)
}

#[test]
fn exp_preconditioner_converges_and_switching_it_off_matches_plain_fire() {
    let n = 12;
    let (geom, neigh, params) = chain(n);
    let metric = Exp::assemble(geom.view(), &neigh, &params, 1.0, 1.0).unwrap();
    let obj = FixedMetric {
        p: metric,
        bounds: eindir_core::Bounds::new(
            Array1::from_elem(3 * n, -1e6),
            Array1::from_elem(3 * n, 1e6),
            0.0,
        ),
    };
    let mut plain_x = geom.clone();
    let mut off_x = geom.clone();
    plain_x[0] = 0.4;
    plain_x[7] = -0.25;
    off_x[0] = 0.4;
    off_x[7] = -0.25;
    let ctrl = Control {
        maxiter: 1,
        gtol: 0.0,
        istep: 0.2,
        maxmove: None,
        ..Control::default()
    };
    let mut plain = Solver::new(
        Method::Fire { kind: FireKind::V1 },
        ctrl.clone(),
        plain_x.len(),
    );
    let mut off = Solver::new(Method::Fire { kind: FireKind::V1 }, ctrl, off_x.len());
    off.set_exp(geom.view(), &neigh, &params).unwrap();
    off.set_fire_precon(false).unwrap();
    for _ in 0..6 {
        plain.step(&obj, &mut plain_x).unwrap();
        off.step(&obj, &mut off_x).unwrap();
        assert_eq!(plain_x, off_x);
    }

    let (plain_n, plain_g) = chain_run(false, false);
    let (off_n, off_g) = chain_run(true, false);
    let (on_n, on_g) = chain_run(true, true);
    assert!(plain_g < 1e-8 && off_g < 1e-8 && on_g < 1e-8);
    assert_eq!(off_n, plain_n, "option off changed the iteration count");
    assert!(on_n < plain_n, "exp FIRE took {on_n}, plain {plain_n}");
}
