//! Noise-tolerant L-BFGS: zero-noise identity, lengthening, and the stops.

use eindir_core::objectives::Rosenbrock;
use eindir_core::{DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, DEFAULT_LINE_FAILURES, LineSearch, Method, NoiseFloor, NoiseModel, Oracle, Solver,
    Status, minimize_method,
};

const ROSENBROCK: &[(u64, u64)] = &[
    (0x3ff7c00b4157d869, 0x4000624dd2f1a9fc),
    (0x3ff691bfdf78930d, 0x4000624dd2f1a9fc),
    (0x3ff6dd4c92e99890, 0x40005fb324fbe6ef),
    (0x3ff6e104e680f90e, 0x40005e7934de91f7),
    (0x3ff6ddf998d36c64, 0x4000582c65448c81),
    (0x3ff6bd43f100b990, 0x400021b4bbf44c07),
    (0x3ff664c1e24d8aeb, 0x3fff339b6830c417),
    (0x3ff5cfac2783a300, 0x3ffd81e6221819c0),
    (0x3ff530040a5031d6, 0x3ffbcf270bf5991c),
    (0x3ff430977be2f247, 0x3ff94cb02663e5e4),
    (0x3ff36b7c6a6d7188, 0x3ff7860ed0d7f9a4),
    (0x3ff286156eac6998, 0x3ff54c03e9eb280d),
    (0x3ff1e3b939ccf01b, 0x3ff3eab3983605c5),
    (0x3ff1522f65d618d9, 0x3ff2b9375e5af448),
    (0x3ff0d22b3f0a2af5, 0x3ff19d1e5858872a),
    (0x3ff05a24c06d42bc, 0x3ff0b3a4c2bd04c6),
    (0x3ff019c191fd4931, 0x3ff02fa740d5dbd5),
    (0x3ff0064c0a2cbbff, 0x3ff00dacf9fc0b1f),
    (0x3feffe954b609a04, 0x3feffd05b6e1d2b6),
    (0x3ff0000267d4691f, 0x3ff0000569fd3962),
    (0x3fefffffb2ee6c16, 0x3fefffff6e402965),
    (0x3feffffff8290649, 0x3feffffff0a9965e),
    (0x3feffffffff4e1f3, 0x3fefffffffe8ced9),
];

fn control(gtol: f64, maxiter: usize) -> Control {
    Control {
        maxiter,
        gtol,
        ..Control::default()
    }
}

fn rosenbrock_bits(noise: Option<NoiseModel>) -> Vec<(u64, u64)> {
    let obj = Rosenbrock::<2>::new();
    let mut x = array![-1.2, 1.0];
    let mut solver = Solver::new(Method::lbfgs(), control(1e-8, 80), x.len());
    if let Some(noise) = noise {
        solver.set_lbfgs_noise(noise);
    }
    let mut bits = Vec::new();
    for _ in 0..80 {
        let report = solver.step(&obj, &mut x).unwrap();
        bits.push((report.coords[0].to_bits(), report.coords[1].to_bits()));
        if report.status != Status::Iterating {
            assert_eq!(report.status, Status::Tolerance);
            break;
        }
    }
    bits
}

#[test]
fn zero_noise_matches_the_main_rosenbrock_iterates() {
    let plain = rosenbrock_bits(None);
    let explicit = rosenbrock_bits(Some(NoiseModel::new()));
    assert_eq!(plain, explicit);
    assert_eq!(plain, ROSENBROCK);
}

fn lj_pair(dx: f64, dy: f64, dz: f64, f32_arith: bool) -> (f64, f64, f64, f64) {
    if f32_arith {
        let dx = dx as f32;
        let dy = dy as f32;
        let dz = dz as f32;
        let r2 = dx * dx + dy * dy + dz * dz;
        let sr2 = 1.0 / r2;
        let s6 = sr2 * sr2 * sr2;
        let s12 = s6 * s6;
        let e = 4.0 * (s12 - s6);
        let df = 24.0 / r2 * (2.0 * s12 - s6);
        return (
            e as f64,
            (df * dx) as f64,
            (df * dy) as f64,
            (df * dz) as f64,
        );
    }
    let r2 = dx * dx + dy * dy + dz * dz;
    let sr2 = 1.0 / r2;
    let s6 = sr2 * sr2 * sr2;
    let s12 = s6 * s6;
    let e = 4.0 * (s12 - s6);
    let df = 24.0 / r2 * (2.0 * s12 - s6);
    (e, df * dx, df * dy, df * dz)
}

struct Lj {
    bounds: eindir_core::Bounds<f64>,
    f32_arith: bool,
    calls: std::sync::atomic::AtomicUsize,
}

impl Lj {
    fn new(f32_arith: bool) -> Self {
        Self {
            bounds: eindir_core::Bounds::new(
                Array1::from_elem(114, -1.0e3),
                Array1::from_elem(114, 1.0e3),
                0.0,
            ),
            f32_arith,
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Objective<f64> for Lj {
    fn dim(&self) -> usize {
        114
    }
    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Lj {
    fn dim(&self) -> usize {
        114
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for Lj {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let n = x.len() / 3;
        let mut energy = 0.0;
        let mut g = vec![0.0; x.len()];
        for i in 0..n {
            for j in (i + 1)..n {
                let (e, fx, fy, fz) = lj_pair(
                    x[3 * i] - x[3 * j],
                    x[3 * i + 1] - x[3 * j + 1],
                    x[3 * i + 2] - x[3 * j + 2],
                    self.f32_arith,
                );
                energy += e;
                g[3 * i] -= fx;
                g[3 * i + 1] -= fy;
                g[3 * i + 2] -= fz;
                g[3 * j] += fx;
                g[3 * j + 1] += fy;
                g[3 * j + 2] += fz;
            }
        }
        (energy, Array1::from_vec(g))
    }
}

fn lj38_start() -> Array1<f64> {
    let nn = 2.0_f64.powf(1.0 / 6.0);
    let a = nn * 2.0_f64.sqrt();
    let mut pts = Vec::new();
    for i in -3..=3 {
        for j in -3..=3 {
            for k in -3..=3 {
                for b in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    let p = [
                        (i as f64 + b[0]) * a,
                        (j as f64 + b[1]) * a,
                        (k as f64 + b[2]) * a,
                    ];
                    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    pts.push((r, p[0], p[1], p[2]));
                }
            }
        }
    }
    pts.sort_by(|u, v| {
        u.0.partial_cmp(&v.0)
            .unwrap()
            .then(u.1.partial_cmp(&v.1).unwrap())
            .then(u.2.partial_cmp(&v.2).unwrap())
            .then(u.3.partial_cmp(&v.3).unwrap())
    });
    let mut x = Array1::zeros(114);
    for (i, p) in pts.into_iter().take(38).enumerate() {
        x[3 * i] = p.1;
        x[3 * i + 1] = p.2;
        x[3 * i + 2] = p.3;
    }
    x
}

fn lj_hash(noise: Option<NoiseModel>) -> (u64, Status, usize) {
    let obj = Lj::new(false);
    let mut x = lj38_start();
    let mut solver = Solver::new(Method::lbfgs(), control(1e-2, 80), x.len());
    if let Some(noise) = noise {
        solver.set_lbfgs_noise(noise);
    }
    let mut h = 0xA5A5_A5A5_A5A5_A5A5u64;
    let mut last = Status::Iterating;
    let mut steps = 0usize;
    for _ in 0..80 {
        let report = solver.step(&obj, &mut x).unwrap();
        for c in report.coords.iter() {
            h = h
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(c.to_bits());
        }
        last = report.status;
        steps = report.steps;
        if report.status != Status::Iterating {
            break;
        }
    }
    (h, last, steps)
}

#[test]
fn zero_noise_matches_the_main_lj38_trajectory() {
    let (plain, status, steps) = lj_hash(None);
    let (explicit, status_z, steps_z) = lj_hash(Some(NoiseModel::new()));
    assert_eq!(plain, explicit);
    assert_eq!(status, status_z);
    assert_eq!(steps, steps_z);
    assert_eq!(status, Status::Tolerance);
    assert_eq!(steps, 35);
    assert_eq!(plain, 0xe8ecb21a6fa560a8);
}

fn nrm2(x: &Array1<f64>) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[test]
fn lengthening_fires_only_on_a_short_step() {
    let quadratic = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        (0.5 * (x[0] * x[0] + x[1] * x[1]), array![x[0], x[1]])
    });
    let mut capped = control(1e-12, 4);
    capped.maxmove = Some(0.05);
    let mut solver = Solver::new(Method::lbfgs(), capped, 2);
    solver.set_lbfgs_noise(NoiseModel::with_noise(0.0, 1.0));
    let mut x = array![1.0, -0.25];
    let _ = solver.step(&quadratic, &mut x).unwrap();
    assert!(
        solver.lbfgs_lengthened() >= 1,
        "short step was not lengthened"
    );

    let mut wide = control(1e-12, 4);
    wide.maxmove = Some(10.0);
    let mut quiet = Solver::new(Method::lbfgs(), wide, 2);
    quiet.set_lbfgs_noise(NoiseModel::with_noise(0.0, 1e-18));
    let mut y = array![1.0, -0.25];
    let _ = quiet.step(&quadratic, &mut y).unwrap();
    assert_eq!(quiet.lbfgs_lengthened(), 0);
}

#[test]
fn repeated_line_search_failures_stop_as_step_tolerance() {
    let objective = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let start = (x[0] - 1.0).abs() < 1e-15 && (x[1] + 0.5).abs() < 1e-15;
        if start {
            (1.0, array![1.0, 1.0])
        } else {
            (f64::NAN, array![f64::NAN, f64::NAN])
        }
    });
    let report = minimize_method(
        &objective,
        array![1.0, -0.5],
        &control(1e-12, 200),
        Method::lbfgs(),
        LineSearch::Wolfe {
            c1: 1e-4,
            c2: 0.9,
            maxiter: 20,
        },
    )
    .unwrap();
    assert_eq!(report.status, Status::StepTolerance);
    assert_eq!(report.steps, DEFAULT_LINE_FAILURES);
    assert!(report.steps < 200);
}

#[test]
fn energy_noise_stops_before_the_budget() {
    let objective = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let f = 1e-8 * (x[0] * x[0] + x[1] * x[1]);
        (f, array![2e-8 * x[0], 2e-8 * x[1]])
    });
    let mut control = control(1e-20, 40);
    control.noise_floor = Some(NoiseFloor::new(0.0).with_eps_f(1e-4));
    let report = minimize_method(
        &objective,
        array![1.0, -0.4],
        &control,
        Method::lbfgs(),
        LineSearch::Wolfe {
            c1: 1e-4,
            c2: 0.9,
            maxiter: 20,
        },
    )
    .unwrap();
    assert_eq!(report.status, Status::NoiseFloor);
    assert!(report.steps < 10, "steps {}", report.steps);
}

fn uniform(seed: u64, index: u64) -> f64 {
    let mut z = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let mant = (z >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
    mant.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
}

fn normal(seed: u64, index: u64) -> f64 {
    let u1 = uniform(seed, 2 * index);
    let u2 = uniform(seed, 2 * index + 1);
    (-2.0 * libm::log(u1)).sqrt() * libm::cos(std::f64::consts::TAU * u2)
}

fn noisy_radius(eps: f64, gaussian: bool) -> f64 {
    let objective = Oracle::unbounded(2, move |x: ArrayView1<f64>| {
        let f = 0.5 * (x[0] * x[0] + x[1] * x[1]);
        let mut g = array![x[0], x[1]];
        for (i, component) in g.iter_mut().enumerate() {
            let draw = if gaussian {
                normal(0x51, i as u64)
            } else {
                2.0 * uniform(0x51, i as u64) - 1.0
            };
            *component += eps * draw;
        }
        (f, g)
    });
    let mut control = control(1e-14, 80);
    control.noise_floor = Some(NoiseFloor::new(eps).with_eps_f(eps));
    let mut solver = Solver::new(Method::lbfgs(), control, 2);
    solver.set_lbfgs_noise(NoiseModel::with_noise(eps, eps));
    let mut x = array![0.8, -0.6];
    let mut last = x.clone();
    for _ in 0..80 {
        let report = solver.step(&objective, &mut x).unwrap();
        last.assign(&report.coords);
        if report.status != Status::Iterating {
            break;
        }
    }
    nrm2(&last)
}

#[test]
fn the_neighbourhood_radius_scales_with_the_injected_noise() {
    let levels = [1e-3, 3e-3, 1e-2, 3e-2];
    for gaussian in [false, true] {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for eps in levels {
            let radius = noisy_radius(eps, gaussian);
            assert!(radius.is_finite() && radius > 0.0, "radius {radius}");
            xs.push(eps.ln());
            ys.push(radius.ln());
        }
        let n = xs.len() as f64;
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut den = 0.0;
        for (x, y) in xs.iter().zip(ys.iter()) {
            num += (x - mx) * (y - my);
            den += (x - mx) * (x - mx);
        }
        let slope = num / den;
        assert!(
            (0.4..=1.6).contains(&slope),
            "gaussian {gaussian} slope {slope} radii {ys:?}"
        );
    }
}

#[test]
#[ignore]
fn strong_scale_two_loop() {
    use std::hint::black_box;
    use std::time::Instant;
    let n = 3_000_000usize;
    let mut solver = rgmin::Lbfgs::with_capacity(8);
    for k in 0..8 {
        let scale = 1e-3 * (k as f64 + 1.0);
        solver.record(Array1::from_elem(n, scale), Array1::from_elem(n, scale));
    }
    let g = Array1::from_elem(n, 1e-4);
    black_box(solver.two_loop(g.view()));
    let mut samples = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        black_box(solver.two_loop(g.view()));
        samples.push(start.elapsed().as_secs_f64());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into());
    println!(
        "threads {threads} median {:.6} min {:.6} max {:.6}",
        samples[2], samples[0], samples[4]
    );
}

#[test]
fn f32_lj38_does_not_spend_the_iteration_budget() {
    let obj = Lj::new(true);
    let mut x = lj38_start();
    x[0] += 0.15;
    x[4] -= 0.1;
    let mut solver = Solver::new(Method::lbfgs(), control(1e-2, 200), x.len());
    let mut last = Status::Iterating;
    for _ in 0..200 {
        let report = solver.step(&obj, &mut x).unwrap();
        last = report.status;
        if report.status != Status::Iterating {
            break;
        }
    }
    let calls = obj.calls.load(std::sync::atomic::Ordering::Relaxed);
    assert_ne!(last, Status::Budget);
    assert!(calls < 2_000, "calls {calls} status {last:?}");
}
