//! Stiefel, SPD and product manifolds on column-major buffers.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ndarray::Array1;
use rgmin::manifold::Spd as SpdSecondOrder;
use rgmin::{Manifold, RiemannianManifold, SpdMetric, StiefelRetraction};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn gram_schmidt(n: usize, p: usize, a: &mut [f64]) {
    for j in 0..p {
        for _ in 0..2 {
            for k in 0..j {
                let r: f64 = (0..n).map(|i| a[i + n * k] * a[i + n * j]).sum();
                for i in 0..n {
                    a[i + n * j] -= r * a[i + n * k];
                }
            }
        }
        let nrm: f64 = (0..n).map(|i| a[i + n * j].powi(2)).sum::<f64>().sqrt();
        for i in 0..n {
            a[i + n * j] /= nrm;
        }
    }
}

fn stiefel_point(rng: &mut Lcg, n: usize, p: usize) -> Vec<f64> {
    let mut a = rng.vec(n * p);
    gram_schmidt(n, p, &mut a);
    a
}

fn spd_point(rng: &mut Lcg, p: usize) -> Vec<f64> {
    let a = rng.vec(p * p);
    let mut s = vec![0.0; p * p];
    for j in 0..p {
        for i in 0..p {
            s[i + p * j] = (0..p).map(|k| a[i + p * k] * a[j + p * k]).sum::<f64>();
        }
        s[j + p * j] += 0.5;
    }
    s
}

fn sym_dir(rng: &mut Lcg, p: usize) -> Vec<f64> {
    let a = rng.vec(p * p);
    let mut s = vec![0.0; p * p];
    for j in 0..p {
        for i in 0..p {
            s[i + p * j] = 0.5 * (a[i + p * j] + a[j + p * i]);
        }
    }
    s
}

/// `max |U^T U - I|`.
fn orthonormality_error(n: usize, p: usize, u: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for a in 0..p {
        for b in 0..p {
            let g: f64 = (0..n).map(|i| u[i + n * a] * u[i + n * b]).sum();
            let want = if a == b { 1.0 } else { 0.0 };
            worst = worst.max((g - want).abs());
        }
    }
    worst
}

/// `max |X^T T + T^T X|`: zero when `T` is tangent at `X`.
fn stiefel_normal_part(n: usize, p: usize, x: &[f64], t: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for a in 0..p {
        for b in 0..p {
            let xt: f64 = (0..n).map(|i| x[i + n * a] * t[i + n * b]).sum();
            let tx: f64 = (0..n).map(|i| t[i + n * a] * x[i + n * b]).sum();
            worst = worst.max((xt + tx).abs());
        }
    }
    worst
}

fn asymmetry(p: usize, s: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for j in 0..p {
        for i in 0..p {
            worst = worst.max((s[i + p * j] - s[j + p * i]).abs());
        }
    }
    worst
}

/// Cholesky succeeds with every pivot above `floor`.
fn cholesky_pivots_above(p: usize, s: &[f64], floor: f64) -> bool {
    let mut l = vec![0.0f64; p * p];
    for j in 0..p {
        let d = s[j + p * j] - (0..j).map(|k| l[j + p * k].powi(2)).sum::<f64>();
        if d.is_nan() || d <= floor {
            return false;
        }
        l[j + p * j] = d.sqrt();
        for i in j + 1..p {
            let v = s[i + p * j] - (0..j).map(|k| l[i + p * k] * l[j + p * k]).sum::<f64>();
            l[i + p * j] = v / l[j + p * j];
        }
    }
    true
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn norm(a: &[f64]) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn tangent(m: &RiemannianManifold, x: &[f64], v: &[f64]) -> Vec<f64> {
    let mut t = vec![0.0; v.len()];
    m.project(x, v, &mut t).unwrap();
    t
}

#[test]
fn retracted_points_stay_on_the_manifold() {
    let mut rng = Lcg(7);
    let (n, p) = (7, 3);
    for retraction in [StiefelRetraction::Qr, StiefelRetraction::Polar] {
        let m = RiemannianManifold::stiefel(n, p, retraction).unwrap();
        for scale in [1e-3, 0.3, 1.0, 4.0] {
            let x = stiefel_point(&mut rng, n, p);
            let v: Vec<f64> = tangent(&m, &x, &rng.vec(n * p))
                .iter()
                .map(|t| scale * t)
                .collect();
            let mut y = vec![0.0; n * p];
            m.retract(&x, &v, &mut y).unwrap();
            let err = orthonormality_error(n, p, &y);
            assert!(
                err < 1e-12,
                "{retraction:?} at scale {scale}: |U^T U - I| = {err}"
            );
            m.exp(&x, &v, &mut y).unwrap();
            let err = orthonormality_error(n, p, &y);
            assert!(err < 1e-12, "exp at scale {scale}: |U^T U - I| = {err}");
        }
    }
    let p = 4;
    for metric in [SpdMetric::AffineInvariant, SpdMetric::LogEuclidean] {
        let m = RiemannianManifold::spd(p, metric).unwrap();
        for scale in [1e-3, 0.3, 1.0, 4.0] {
            let x = spd_point(&mut rng, p);
            let v: Vec<f64> = sym_dir(&mut rng, p).iter().map(|t| scale * t).collect();
            let mut y = vec![0.0; p * p];
            m.retract(&x, &v, &mut y).unwrap();
            assert!(asymmetry(p, &y) <= 1e-12, "{metric:?}: asymmetric {y:?}");
            assert!(
                cholesky_pivots_above(p, &y, 1e-12),
                "{metric:?} at scale {scale}: not positive definite {y:?}"
            );
        }
    }
}

/// `|R(t v) - Exp(t v)|` at `t` and `t / 2`. A second-order retraction
/// has error `O(t^3)`, so halving `t` divides it by about 8.
fn halving_ratios(err: impl Fn(f64) -> f64) -> Vec<f64> {
    [0.2, 0.1, 0.05, 0.025]
        .iter()
        .map(|&t| err(t) / err(0.5 * t))
        .collect()
}

#[test]
fn polar_and_spd_retractions_agree_with_the_exponential_to_second_order() {
    let mut rng = Lcg(11);
    let (n, p) = (6, 3);
    let polar = RiemannianManifold::stiefel(n, p, StiefelRetraction::Polar).unwrap();
    let x = stiefel_point(&mut rng, n, p);
    let v = tangent(&polar, &x, &rng.vec(n * p));
    let gap = |m: &RiemannianManifold, t: f64| {
        let tv: Vec<f64> = v.iter().map(|a| t * a).collect();
        let (mut r, mut e) = (vec![0.0; n * p], vec![0.0; n * p]);
        m.retract(&x, &tv, &mut r).unwrap();
        m.exp(&x, &tv, &mut e).unwrap();
        norm(&r.iter().zip(&e).map(|(a, b)| a - b).collect::<Vec<_>>())
    };
    for ratio in halving_ratios(|t| gap(&polar, t)) {
        assert!((7.0..9.0).contains(&ratio), "polar ratio {ratio}");
    }
    let qr = RiemannianManifold::stiefel(n, p, StiefelRetraction::Qr).unwrap();
    for ratio in halving_ratios(|t| gap(&qr, t)) {
        assert!(
            (3.5..5.0).contains(&ratio),
            "QR agrees with exp to first order, error O(t^2): {ratio}"
        );
    }

    let p = 3;
    let ai = RiemannianManifold::spd(p, SpdMetric::AffineInvariant).unwrap();
    let x = spd_point(&mut rng, p);
    let v = sym_dir(&mut rng, p);
    let spd_gap = |t: f64| {
        let tv: Vec<f64> = v.iter().map(|a| t * a).collect();
        let mut e = vec![0.0; p * p];
        ai.exp(&x, &tv, &mut e).unwrap();
        let r = SpdSecondOrder.retract(&Array1::from(x.clone()), &Array1::from(tv));
        max_abs_diff(r.as_slice().unwrap(), &e)
    };
    for ratio in halving_ratios(spd_gap) {
        assert!(
            (7.0..9.0).contains(&ratio),
            "SPD second-order ratio {ratio}"
        );
    }
}

#[test]
fn vector_transport_keeps_vectors_tangent() {
    let mut rng = Lcg(13);
    let (n, p) = (5, 2);
    let st = RiemannianManifold::stiefel(n, p, StiefelRetraction::Qr).unwrap();
    let x = stiefel_point(&mut rng, n, p);
    let y = stiefel_point(&mut rng, n, p);
    let u = tangent(&st, &x, &rng.vec(n * p));
    let mut t = vec![0.0; n * p];
    st.transport(&x, &y, &u, &mut t).unwrap();
    assert!(stiefel_normal_part(n, p, &y, &t) < 1e-13);
    assert!(stiefel_normal_part(n, p, &x, &u) < 1e-13);

    let q = 3;
    for metric in [SpdMetric::AffineInvariant, SpdMetric::LogEuclidean] {
        let m = RiemannianManifold::spd(q, metric).unwrap();
        let x = spd_point(&mut rng, q);
        let y = spd_point(&mut rng, q);
        let (a, b) = (sym_dir(&mut rng, q), sym_dir(&mut rng, q));
        let (mut ta, mut tb) = (vec![0.0; q * q], vec![0.0; q * q]);
        m.transport(&x, &y, &a, &mut ta).unwrap();
        m.transport(&x, &y, &b, &mut tb).unwrap();
        assert!(asymmetry(q, &ta) == 0.0 && asymmetry(q, &tb) == 0.0);
        let before = m.inner(&x, &a, &b).unwrap();
        let after = m.inner(&y, &ta, &tb).unwrap();
        assert!(
            (before - after).abs() <= 1e-10 * before.abs().max(1.0),
            "{metric:?} parallel transport is not an isometry: {before} vs {after}"
        );
    }

    let prod = RiemannianManifold::product([
        st.clone(),
        RiemannianManifold::spd(q, SpdMetric::AffineInvariant).unwrap(),
    ])
    .unwrap();
    let xs = [stiefel_point(&mut rng, n, p), spd_point(&mut rng, q)].concat();
    let ys = [stiefel_point(&mut rng, n, p), spd_point(&mut rng, q)].concat();
    let raw = rng.vec(prod.packed_len());
    let mut moved = vec![0.0; raw.len()];
    prod.transport(&xs, &ys, &raw, &mut moved).unwrap();
    let r = prod.ranges();
    assert!(stiefel_normal_part(n, p, &ys[r[0].clone()], &moved[r[0].clone()]) < 1e-13);
    assert_eq!(asymmetry(q, &moved[r[1].clone()]), 0.0);
}

#[test]
fn spd_exp_and_log_round_trip() {
    let mut rng = Lcg(17);
    let p = 4;
    for metric in [SpdMetric::AffineInvariant, SpdMetric::LogEuclidean] {
        let m = RiemannianManifold::spd(p, metric).unwrap();
        let x = spd_point(&mut rng, p);
        let y = spd_point(&mut rng, p);
        let v = sym_dir(&mut rng, p);
        let (mut e, mut back) = (vec![0.0; p * p], vec![0.0; p * p]);
        m.exp(&x, &v, &mut e).unwrap();
        m.log(&x, &e, &mut back).unwrap();
        assert!(
            max_abs_diff(&back, &v) < 1e-10,
            "{metric:?} log(exp(v)) != v by {}",
            max_abs_diff(&back, &v)
        );
        m.log(&x, &y, &mut back).unwrap();
        m.exp(&x, &back, &mut e).unwrap();
        assert!(
            max_abs_diff(&e, &y) < 1e-10,
            "{metric:?} exp(log(y)) != y by {}",
            max_abs_diff(&e, &y)
        );
    }
    let st = RiemannianManifold::stiefel(3, 2, StiefelRetraction::Qr).unwrap();
    let x = stiefel_point(&mut rng, 3, 2);
    assert!(st.log(&x, &x, &mut [0.0; 6]).is_err());
}

#[test]
fn riemannian_gradient_represents_the_euclidean_differential() {
    let mut rng = Lcg(19);
    let m = RiemannianManifold::product([
        RiemannianManifold::stiefel(5, 2, StiefelRetraction::Polar).unwrap(),
        RiemannianManifold::spd(3, SpdMetric::AffineInvariant).unwrap(),
        RiemannianManifold::spd(2, SpdMetric::LogEuclidean).unwrap(),
    ])
    .unwrap();
    let x = [
        stiefel_point(&mut rng, 5, 2),
        spd_point(&mut rng, 3),
        spd_point(&mut rng, 2),
    ]
    .concat();
    let g = rng.vec(m.packed_len());
    let v = tangent(&m, &x, &rng.vec(m.packed_len()));
    let mut rgrad = vec![0.0; g.len()];
    m.egrad_to_rgrad(&x, &g, &mut rgrad).unwrap();
    let riemannian = m.inner(&x, &rgrad, &v).unwrap();
    let euclidean: f64 = g.iter().zip(&v).map(|(a, b)| a * b).sum();
    assert!(
        (riemannian - euclidean).abs() < 1e-12 * euclidean.abs().max(1.0),
        "<grad f, v>_x = {riemannian}, Df[v] = {euclidean}"
    );
}

/// `f(U, S) = |U - U*|^2 / 2 + |S - S*|^2 / 2`, convex in the ambient space.
fn fit_cost(x: &[f64], target: &[f64]) -> (f64, Vec<f64>) {
    let g: Vec<f64> = x.iter().zip(target).map(|(a, b)| a - b).collect();
    (0.5 * g.iter().map(|v| v * v).sum::<f64>(), g)
}

#[test]
fn gradient_steps_decrease_a_convex_cost_monotonically() {
    let mut rng = Lcg(23);
    let (n, p, q) = (6, 2, 3);
    for (retraction, metric) in [
        (StiefelRetraction::Qr, SpdMetric::AffineInvariant),
        (StiefelRetraction::Polar, SpdMetric::LogEuclidean),
    ] {
        let m = RiemannianManifold::product([
            RiemannianManifold::stiefel(n, p, retraction).unwrap(),
            RiemannianManifold::spd(q, metric).unwrap(),
        ])
        .unwrap();
        let target = [stiefel_point(&mut rng, n, p), spd_point(&mut rng, q)].concat();
        let mut x = [stiefel_point(&mut rng, n, p), spd_point(&mut rng, q)].concat();
        let (start, _) = fit_cost(&x, &target);
        let mut last = start;
        let (mut rgrad, mut next) = (vec![0.0; x.len()], vec![0.0; x.len()]);
        for step in 0..200 {
            let (_, g) = fit_cost(&x, &target);
            m.gradient_step(&x, &g, 0.1, &mut rgrad, &mut next).unwrap();
            let (cost, _) = fit_cost(&next, &target);
            assert!(
                cost < last,
                "{retraction:?}/{metric:?} step {step}: {cost} did not decrease from {last}"
            );
            let r = m.ranges();
            assert!(orthonormality_error(n, p, &next[r[0].clone()]) < 1e-12);
            assert!(cholesky_pivots_above(q, &next[r[1].clone()], 0.0));
            last = cost;
            std::mem::swap(&mut x, &mut next);
        }
        assert!(
            last < 1e-3 * start,
            "{retraction:?}/{metric:?}: {start} -> {last}"
        );
    }
}

#[test]
fn trait_methods_match_the_slice_methods() {
    let mut rng = Lcg(29);
    let m = RiemannianManifold::product([
        RiemannianManifold::stiefel(4, 2, StiefelRetraction::Qr).unwrap(),
        RiemannianManifold::spd(2, SpdMetric::AffineInvariant).unwrap(),
    ])
    .unwrap();
    let x = [stiefel_point(&mut rng, 4, 2), spd_point(&mut rng, 2)].concat();
    let v = tangent(&m, &x, &rng.vec(m.packed_len()));
    let mut y = vec![0.0; x.len()];
    m.retract(&x, &v, &mut y).unwrap();
    let ya = Manifold::retract(&m, &Array1::from(x.clone()), &Array1::from(v.clone()));
    assert_eq!(ya.to_vec(), y);
    assert!(m.required_dim(12).is_ok());
    assert_eq!(m.required_dim(11), Err(12));
}

#[test]
fn bad_shapes_points_and_steps_are_refused() {
    assert!(RiemannianManifold::stiefel(2, 3, StiefelRetraction::Qr).is_err());
    assert!(RiemannianManifold::stiefel(3, 0, StiefelRetraction::Qr).is_err());
    assert!(RiemannianManifold::spd(0, SpdMetric::AffineInvariant).is_err());
    assert!(RiemannianManifold::product(Vec::new()).is_err());
    let spd = RiemannianManifold::spd(2, SpdMetric::AffineInvariant).unwrap();
    let mut out = [0.0; 4];
    assert!(
        spd.retract(&[1.0, 0.0, 0.0, 1.0], &[0.0; 3], &mut out)
            .is_err()
    );
    assert!(
        spd.retract(&[1.0, 0.0, 0.0, -1.0], &[0.0; 4], &mut out)
            .is_err()
    );
    assert!(
        spd.retract(&[1.0, 0.5, 0.0, 1.0], &[0.0; 4], &mut out)
            .is_err()
    );
    assert!(
        spd.retract(&[1.0, 0.0, 0.0, f64::NAN], &[0.0; 4], &mut out)
            .is_err()
    );
    let x = [1.0, 0.0, 0.0, 1.0];
    assert!(
        spd.gradient_step(&x, &[0.0; 4], f64::INFINITY, &mut [0.0; 4], &mut out)
            .is_err()
    );
    let st = RiemannianManifold::stiefel(2, 2, StiefelRetraction::Qr).unwrap();
    assert!(st.retract(&x, &[-1.0, 0.0, 0.0, 0.0], &mut out).is_err());
}

// The fixture is shared with python/tests/test_manifold.py. Set
// RGMIN_REGEN_MANIFOLD_FIXTURE=1 to rewrite it from this build.

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/riemannian_manifold.txt")
}

struct Case {
    name: &'static str,
    manifold: RiemannianManifold,
    x: Vec<f64>,
    y: Vec<f64>,
    v: Vec<f64>,
    g: Vec<f64>,
}

fn fixture_cases() -> Vec<Case> {
    let mut rng = Lcg(31);
    let (n, p, q) = (5, 2, 3);
    let st_qr = RiemannianManifold::stiefel(n, p, StiefelRetraction::Qr).unwrap();
    let st_polar = RiemannianManifold::stiefel(n, p, StiefelRetraction::Polar).unwrap();
    let ai = RiemannianManifold::spd(q, SpdMetric::AffineInvariant).unwrap();
    let le = RiemannianManifold::spd(q, SpdMetric::LogEuclidean).unwrap();
    let product = RiemannianManifold::product([st_qr.clone(), ai.clone()]).unwrap();
    let mut cases = Vec::new();
    for (name, manifold) in [
        ("stiefel_qr", st_qr),
        ("stiefel_polar", st_polar),
        ("spd_ai", ai),
        ("spd_le", le),
        ("product", product),
    ] {
        let point = |rng: &mut Lcg| -> Vec<f64> {
            manifold
                .factors()
                .iter()
                .flat_map(|f| match *f {
                    rgmin::ManifoldFactor::Stiefel { n, p, .. } => stiefel_point(rng, n, p),
                    rgmin::ManifoldFactor::Spd { p, .. } => spd_point(rng, p),
                })
                .collect()
        };
        let x = point(&mut rng);
        let y = point(&mut rng);
        let v = tangent(&manifold, &x, &rng.vec(manifold.packed_len()));
        let g = rng.vec(manifold.packed_len());
        cases.push(Case {
            name,
            manifold,
            x,
            y,
            v,
            g,
        });
    }
    cases
}

fn fixture_rows() -> BTreeMap<String, Vec<f64>> {
    let mut rows = BTreeMap::new();
    for c in fixture_cases() {
        let m = &c.manifold;
        let len = m.packed_len();
        let put = |rows: &mut BTreeMap<String, Vec<f64>>, key: &str, v: Vec<f64>| {
            rows.insert(format!("{}.{key}", c.name), v);
        };
        put(&mut rows, "x", c.x.clone());
        put(&mut rows, "y", c.y.clone());
        put(&mut rows, "v", c.v.clone());
        put(&mut rows, "g", c.g.clone());
        let mut out = vec![0.0; len];
        m.project(&c.x, &c.g, &mut out).unwrap();
        put(&mut rows, "project", out.clone());
        m.egrad_to_rgrad(&c.x, &c.g, &mut out).unwrap();
        put(&mut rows, "rgrad", out.clone());
        m.retract(&c.x, &c.v, &mut out).unwrap();
        put(&mut rows, "retract", out.clone());
        m.exp(&c.x, &c.v, &mut out).unwrap();
        put(&mut rows, "exp", out.clone());
        m.transport(&c.x, &c.y, &c.v, &mut out).unwrap();
        put(&mut rows, "transport", out.clone());
        put(&mut rows, "inner", vec![m.inner(&c.x, &c.v, &c.v).unwrap()]);
        let mut x_out = vec![0.0; len];
        m.gradient_step(&c.x, &c.g, 0.1, &mut out, &mut x_out)
            .unwrap();
        put(&mut rows, "step_rgrad", out.clone());
        put(&mut rows, "step_x", x_out);
        if m.log(&c.x, &c.y, &mut out).is_ok() {
            put(&mut rows, "log", out.clone());
        }
    }
    rows
}

#[test]
fn rust_results_match_the_shared_fixture() {
    let rows = fixture_rows();
    let path = fixture_path();
    if std::env::var_os("RGMIN_REGEN_MANIFOLD_FIXTURE").is_some() {
        let mut text = String::from(
            "# name value... ; inputs x, y, v, g, then outputs. Written by tests/riemannian_manifold.rs.\n",
        );
        for (key, values) in &rows {
            text.push_str(key);
            for v in values {
                text.push_str(&format!(" {v:.17e}"));
            }
            text.push('\n');
        }
        std::fs::write(&path, text).unwrap();
    }
    let text = std::fs::read_to_string(&path).unwrap();
    let mut seen = 0;
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let mut parts = line.split_whitespace();
        let key = parts.next().unwrap();
        let want: Vec<f64> = parts.map(|s| s.parse().unwrap()).collect();
        let got = rows
            .get(key)
            .unwrap_or_else(|| panic!("{key} not computed"));
        assert_eq!(got.len(), want.len(), "{key}");
        let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        assert!(
            max_abs_diff(got, &want) <= 1e-13 * scale,
            "{key}: {got:?} vs {want:?}"
        );
        seen += 1;
    }
    assert_eq!(seen, rows.len());
}
