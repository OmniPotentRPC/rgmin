//! Exponential preconditioner: the paper's μ, and fewer iterations
//! when H0 is that matrix.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rgmin::{Control, Exp, ExpParams, Method, Neighbours, Solver, estimate_mu};

struct FixedMetric {
    p: Exp,
    bounds: Bounds<f64>,
}

impl Objective<f64> for FixedMetric {
    fn dim(&self) -> usize {
        self.p.dim()
    }
    fn bounds(&self) -> &Bounds<f64> {
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
    let mut params = ExpParams::default();
    params.r_nn = Some(1.0);
    params.mu = Some(1.0);
    (pos, neigh, params)
}

fn steps_until(precon: bool, estimate: bool) -> (usize, Option<f64>) {
    let n = 30;
    let (geom, neigh, mut params) = chain(n);
    let metric = Exp::assemble(geom.view(), &neigh, &params, 1.0, 1.0).unwrap();
    let obj = FixedMetric {
        p: metric,
        bounds: Bounds::new(
            Array1::from_elem(3 * n, -1e6),
            Array1::from_elem(3 * n, 1e6),
            0.0,
        ),
    };
    let mut x = geom.clone();
    x[1] = 0.2;
    x[4] = -0.15;
    let control = Control {
        maxiter: 1,
        gtol: 1e-8,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 5 }, control, x.len());
    if precon {
        if estimate {
            params.mu = None;
        }
        solver.set_exp(geom.view(), &neigh, &params).unwrap();
    }
    let mut taken = 0;
    for _ in 0..40 {
        let gnorm = obj.grad(x.view()).dot(&obj.grad(x.view())).sqrt();
        if gnorm < 1e-8 {
            break;
        }
        solver.step(&obj, &mut x).unwrap();
        taken += 1;
    }
    (taken, solver.precon_scale())
}

#[test]
fn preconditioned_lbfgs_takes_fewer_iterations_on_a_chain() {
    let (with, mu) = steps_until(true, false);
    let (without, _) = steps_until(false, false);
    assert!(
        with < without,
        "preconditioned {with} iterations, plain {without}"
    );
    assert!(with <= 3, "expected a short Newton-like run, took {with}");
    assert!((mu.unwrap() - 1.0).abs() < 1e-12);

    let (estimated_steps, estimated_mu) = steps_until(true, true);
    assert!(
        (estimated_mu.unwrap() - 1.0).abs() < 1e-6,
        "on-the-fly mu {}",
        estimated_mu.unwrap()
    );
    assert!(estimated_steps <= 3, "estimated run took {estimated_steps}");
}

fn steps_of(method: Method, precon: bool, limit: usize, istep: f64) -> (usize, f64) {
    let n = 12;
    let (geom, neigh, params) = chain(n);
    let metric = Exp::assemble(geom.view(), &neigh, &params, 1.0, 1.0).unwrap();
    let obj = FixedMetric {
        p: metric,
        bounds: Bounds::new(
            Array1::from_elem(3 * n, -1e6),
            Array1::from_elem(3 * n, 1e6),
            0.0,
        ),
    };
    let mut x = geom.clone();
    x[0] = 0.4;
    x[7] = -0.25;
    let control = Control {
        maxiter: 1,
        gtol: 1e-8,
        istep,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    };
    let mut solver = Solver::new(method, control, x.len());
    if precon {
        solver.set_exp(geom.view(), &neigh, &params).unwrap();
    }
    let mut taken = 0;
    let mut gnorm = 0.0;
    for _ in 0..limit {
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
fn preconditioned_fire_and_nlcg_take_fewer_iterations_on_a_chain() {
    let fire = Method::Fire {
        kind: rgmin::FireKind::V1,
    };
    let cg = Method::nlcg(rgmin::Conjugacy::PolakRibiere);
    let fire_with = steps_of(fire.clone(), true, 250, 0.2);
    let fire_without = steps_of(fire, false, 250, 0.2);
    assert!(
        fire_with.1 < 1e-8 && fire_without.1 < 1e-8,
        "FIRE metric {fire_with:?}, plain {fire_without:?}"
    );
    assert!(
        fire_with.0 < fire_without.0,
        "FIRE metric {fire_with:?}, plain {fire_without:?}"
    );
    let cg_with = steps_of(cg.clone(), true, 40, 1.0);
    let cg_without = steps_of(cg, false, 40, 1.0);
    assert!(
        cg_with.0 < cg_without.0,
        "CG metric {cg_with:?}, plain {cg_without:?}"
    );
    assert!(cg_with.0 <= 4, "metric CG took {}", cg_with.0);
}

#[test]
fn fire_and_nlcg_move_in_the_metric() {
    let n = 8;
    let (geom, neigh, params) = chain(n);
    let metric = Exp::assemble(geom.view(), &neigh, &params, 1.0, 1.0).unwrap();
    let obj = FixedMetric {
        p: metric,
        bounds: Bounds::new(
            Array1::from_elem(3 * n, -1e6),
            Array1::from_elem(3 * n, 1e6),
            0.0,
        ),
    };
    let control = Control {
        maxiter: 1,
        gtol: 1e-10,
        istep: 0.05,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    };
    for method in [
        Method::Fire {
            kind: rgmin::FireKind::V1,
        },
        Method::nlcg(rgmin::Conjugacy::PolakRibiere),
    ] {
        let mut plain = geom.clone();
        plain[0] = 0.3;
        let mut pre = plain.clone();
        let mut a = Solver::new(method.clone(), control.clone(), plain.len());
        let mut b = Solver::new(method, control.clone(), pre.len());
        b.set_exp(geom.view(), &neigh, &params).unwrap();
        a.step(&obj, &mut plain).unwrap();
        b.step(&obj, &mut pre).unwrap();
        let delta: f64 = plain
            .iter()
            .zip(pre.iter())
            .map(|(u, v)| (u - v).abs())
            .sum();
        assert!(delta > 1e-8, "metric did not change the step ({delta})");
    }
}

#[test]
fn estimate_mu_recovers_a_scaled_metric() {
    let (geom, neigh, mut params) = chain(6);
    params.lengths = Some([8.0, 2.0, 2.0]);
    let mu_true = 3.5;
    let p1 = Exp::assemble(geom.view(), &neigh, &params, 1.0, 1.0).unwrap();
    let grad = |x: ArrayView1<f64>| mu_true * p1.apply(x);
    let mu = estimate_mu(geom.view(), &neigh, &params, grad).unwrap();
    assert!((mu - mu_true).abs() < 1e-8, "mu {mu}");
}

struct ModelQuad<P> {
    apply: P,
    dim: usize,
    bounds: Bounds<f64>,
}

impl<P> Objective<f64> for ModelQuad<P>
where
    P: Fn(ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * x.dot(&(self.apply)(x))
    }
}

impl<P> Gradient<f64> for ModelQuad<P>
where
    P: Fn(ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.dim
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        (self.apply)(x)
    }
}

impl<P> DifferentiableObjective<f64> for ModelQuad<P>
where
    P: Fn(ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let g = (self.apply)(x);
        (0.5 * x.dot(&g), g)
    }
}

fn water() -> Array1<f64> {
    let ang = 104.5_f64.to_radians();
    ndarray::array![
        0.0,
        0.0,
        0.0,
        0.96,
        0.0,
        0.0,
        0.96 * ang.cos(),
        0.96 * ang.sin(),
        0.0
    ]
}

fn steps_on(
    mut solver: Solver,
    obj: &impl DifferentiableObjective<f64>,
    x0: &Array1<f64>,
) -> usize {
    let mut x = x0.clone();
    let mut taken = 0;
    for _ in 0..40 {
        let g = obj.grad(x.view());
        if g.dot(&g).sqrt() < 1e-8 {
            break;
        }
        solver.step(obj, &mut x).unwrap();
        taken += 1;
    }
    taken
}

#[test]
fn lindh_and_ff_shorten_lbfgs_on_a_molecule() {
    use rgmin::{BondTerm, FfParams, FfTerms, Lindh, LindhParams, select_precon};
    use rgmin::{GeometryKind, PreconChoice};

    let geom = water();
    let z = [8, 1, 1];
    let mut lindh_params = LindhParams::default();
    lindh_params.mu = Some(1.0);
    let model = Lindh::assemble(geom.view(), &z, &[], &lindh_params).unwrap();
    let obj = ModelQuad {
        apply: |x: ArrayView1<f64>| model.apply(x),
        dim: geom.len(),
        bounds: Bounds::new(Array1::from_elem(9, -1e6), Array1::from_elem(9, 1e6), 0.0),
    };
    let control = Control {
        maxiter: 1,
        gtol: 1e-8,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    };
    let plain = Solver::new(Method::Lbfgs { memory: 5 }, control.clone(), geom.len());
    let mut with = Solver::new(Method::Lbfgs { memory: 5 }, control, geom.len());
    with.set_lindh(geom.view(), &z, &[], &lindh_params).unwrap();
    let without = steps_on(plain, &obj, &geom);
    let taken = steps_on(with, &obj, &geom);
    assert!(taken < without, "lindh {taken}, plain {without}");
    assert!(taken <= 3, "lindh took {taken}");

    let bonds = [
        BondTerm { i: 0, j: 1, k: 1.0 },
        BondTerm { i: 0, j: 2, k: 1.0 },
    ];
    let mut ff_params = FfParams::default();
    ff_params.mu = Some(1.0);
    let terms = FfTerms {
        bonds: &bonds,
        ..FfTerms::default()
    };
    let ff = rgmin::Ff::assemble(geom.view(), terms, &ff_params).unwrap();
    let ff_obj = ModelQuad {
        apply: |x: ArrayView1<f64>| ff.apply(x),
        dim: geom.len(),
        bounds: Bounds::new(Array1::from_elem(9, -1e6), Array1::from_elem(9, 1e6), 0.0),
    };
    let control = Control {
        maxiter: 1,
        gtol: 1e-8,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    };
    let plain = Solver::new(Method::Lbfgs { memory: 5 }, control.clone(), geom.len());
    let mut with = Solver::new(Method::Lbfgs { memory: 5 }, control, geom.len());
    with.set_ff(geom.view(), terms, &ff_params).unwrap();
    let without = steps_on(plain, &ff_obj, &geom);
    let taken = steps_on(with, &ff_obj, &geom);
    assert!(taken < without, "ff {taken}, plain {without}");
    assert!(taken <= 3, "ff took {taken}");
    assert_eq!(
        select_precon(
            GeometryKind::Molecule,
            &[8, 1, 1],
            false,
            rgmin::OptimizerFamily::QuasiNewton
        ),
        PreconChoice::Lindh
    );
    assert_eq!(
        select_precon(
            GeometryKind::Molecule,
            &[8, 1, 1],
            true,
            rgmin::OptimizerFamily::Cg
        ),
        PreconChoice::Ff
    );
}

#[test]
fn lindh_estimates_the_energy_scale() {
    use rgmin::LindhParams;

    let geom = water();
    let z = [8, 1, 1];
    let mut params = LindhParams::default();
    params.mu = Some(1.0);
    let model = rgmin::Lindh::assemble(geom.view(), &z, &[], &params).unwrap();
    let scale = 2.5;
    let obj = ModelQuad {
        apply: move |x: ArrayView1<f64>| scale * model.apply(x),
        dim: geom.len(),
        bounds: Bounds::new(Array1::from_elem(9, -1e6), Array1::from_elem(9, 1e6), 0.0),
    };
    let mut estimated = LindhParams::default();
    estimated.lengths = Some([4.0, 4.0, 4.0]);
    let control = Control {
        maxiter: 1,
        gtol: 1e-10,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 4 }, control, geom.len());
    solver.set_lindh(geom.view(), &z, &[], &estimated).unwrap();
    let mut x = geom.clone();
    solver.step(&obj, &mut x).unwrap();
    let mu = solver.precon_scale().unwrap();
    assert!((mu - scale).abs() < 1e-4, "mu {mu}");
}

fn lattice(n: usize) -> (Array1<f64>, Vec<i32>) {
    let mut pos = Array1::zeros(3 * n);
    let mut z = vec![6i32; n];
    z[0] = 1;
    for i in 0..n {
        pos[3 * i] = (i % 5) as f64 * 1.45;
        pos[3 * i + 1] = (i / 5) as f64 * 1.5;
        pos[3 * i + 2] = ((i * 3) % 4) as f64 * 0.35;
    }
    (pos, z)
}

fn bits_of(v: &Array1<f64>) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// One past the gate, or 168 when the kernel stays serial.
fn size_above(gate: usize) -> usize {
    if gate >= usize::MAX / 2 {
        168
    } else {
        gate.saturating_add(8).max(24)
    }
}

fn neighbours_cutoff(pos: &Array1<f64>, cut: f64) -> Neighbours {
    let n = pos.len() / 3;
    let mut origin = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for i in 0..n {
        for k in 0..3 {
            origin[k] = origin[k].min(pos[3 * i + k]);
            hi[k] = hi[k].max(pos[3 * i + k]);
        }
    }
    let _ = hi;
    let mut cells: std::collections::BTreeMap<(i32, i32, i32), Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in 0..n {
        let key = (
            ((pos[3 * i] - origin[0]) / cut).floor() as i32,
            ((pos[3 * i + 1] - origin[1]) / cut).floor() as i32,
            ((pos[3 * i + 2] - origin[2]) / cut).floor() as i32,
        );
        cells.entry(key).or_default().push(i);
    }
    let mut neigh = Neighbours::default();
    for (&key, atoms) in &cells {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nb = (key.0 + dx, key.1 + dy, key.2 + dz);
                    let Some(others) = cells.get(&nb) else {
                        continue;
                    };
                    for &i in atoms {
                        for &j in others {
                            if j <= i {
                                continue;
                            }
                            let dx = pos[3 * i] - pos[3 * j];
                            let dy = pos[3 * i + 1] - pos[3 * j + 1];
                            let dz = pos[3 * i + 2] - pos[3 * j + 2];
                            let r = (dx * dx + dy * dy + dz * dz).sqrt();
                            if r < cut {
                                neigh.i.push(i);
                                neigh.j.push(j);
                                neigh.r.push(r);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut pairs: Vec<(usize, usize, f64)> = neigh
        .i
        .iter()
        .zip(neigh.j.iter())
        .zip(neigh.r.iter())
        .map(|((i, j), r)| (*i, *j, *r))
        .collect();
    pairs.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    pairs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    Neighbours {
        i: pairs.iter().map(|p| p.0).collect(),
        j: pairs.iter().map(|p| p.1).collect(),
        r: pairs.iter().map(|p| p.2).collect(),
        anchor: Vec::new(),
    }
}

/// Assembly, apply, and solve above each kernel's gate. The thread
/// pool is the only thing that changes.
fn precon_fingerprint() -> Vec<u64> {
    use rgmin::{Ff, FfParams, FfTerms, Lindh, LindhParams};

    let n = size_above(rgmin::N_PAR_LINDH_SOLVE)
        .max(size_above(rgmin::N_PAR_LINDH_APPLY))
        .max(size_above(rgmin::N_PAR_ASSEMBLE));
    let (pos, z) = lattice(n);
    let mut lp = LindhParams::default();
    lp.mu = Some(1.0);
    lp.r_cut = Some(3.0);
    let lindh = Lindh::assemble(pos.view(), &z, &[], &lp).unwrap();
    let rhs = Array1::from_iter((0..pos.len()).map(|i| ((i % 5) as f64 - 2.0) * 0.1));
    let mut out = Vec::new();
    out.extend(bits_of(&lindh.apply(rhs.view())));
    out.extend(bits_of(&lindh.solve(rhs.view()).unwrap()));

    let (bonds, angles) = rgmin::bonded_terms(pos.view(), 2.2, 1.0, 0.2, None).unwrap();
    let bonds = bonds;
    let mut fp = FfParams::default();
    fp.mu = Some(1.0);
    fp.r_cut = Some(3.0);
    let terms = FfTerms {
        bonds: &bonds,
        angles: &angles,
        ..FfTerms::default()
    };
    let ff = Ff::assemble(pos.view(), terms, &fp).unwrap();
    out.extend(bits_of(&ff.apply(rhs.view())));
    out.extend(bits_of(&ff.solve(rhs.view()).unwrap()));

    let exp_n = size_above(rgmin::N_PAR_EXP_SOLVE).max(size_above(rgmin::N_PAR_EXP_APPLY));
    let (pos, _z) = if exp_n == n { (pos, z) } else { lattice(exp_n) };
    let neigh = neighbours_cutoff(&pos, 2.6);
    let rhs = Array1::from_iter((0..pos.len()).map(|i| ((i % 5) as f64 - 2.0) * 0.1));
    let mut ep = ExpParams::default();
    ep.mu = Some(1.0);
    ep.r_nn = Some(1.45);
    let exp = Exp::assemble(pos.view(), &neigh, &ep, 1.0, 1.45).unwrap();
    out.extend(bits_of(&exp.apply(rhs.view())));
    out.extend(bits_of(&exp.solve(rhs.view()).unwrap()));
    out
}

#[test]
fn precon_is_bit_identical_at_1_2_3_and_8_threads() {
    let mut seen = Vec::new();
    for threads in [1usize, 2, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let bits = pool.install(precon_fingerprint);
        seen.push(bits);
    }
    assert_eq!(seen[0], seen[1], "1 thread vs 2 threads");
    assert_eq!(seen[0], seen[2], "1 thread vs 3 threads");
    assert_eq!(seen[0], seen[3], "1 thread vs 8 threads");
}
