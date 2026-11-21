//! Rigid quotient against an independent dense reference, across scales,
//! strata and the retraction and transport.

use rgmin::RiemannianManifold;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn quotient(n_atoms: usize, periodic: bool) -> RiemannianManifold {
    RiemannianManifold::rigid_quotient(n_atoms, periodic).unwrap()
}

fn project(m: &RiemannianManifold, x: &[f64], v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    m.project(x, v, &mut out).unwrap();
    out
}

fn rigid_motions(x: &[f64], periodic: bool) -> Vec<Vec<f64>> {
    let n = x.len() / 3;
    let mut centre = [0.0; 3];
    for atom in x.chunks(3) {
        for (c, a) in centre.iter_mut().zip(atom) {
            *c += a / n as f64;
        }
    }
    let mut motions = Vec::new();
    for axis in 0..3 {
        let mut t = vec![0.0; x.len()];
        for atom in 0..n {
            t[3 * atom + axis] = 1.0;
        }
        motions.push(t);
    }
    if !periodic {
        for axis in 0..3 {
            let mut omega = [0.0; 3];
            omega[axis] = 1.0;
            let mut r = vec![0.0; x.len()];
            for atom in 0..n {
                let d = [
                    x[3 * atom] - centre[0],
                    x[3 * atom + 1] - centre[1],
                    x[3 * atom + 2] - centre[2],
                ];
                r[3 * atom] = omega[1] * d[2] - omega[2] * d[1];
                r[3 * atom + 1] = omega[2] * d[0] - omega[0] * d[2];
                r[3 * atom + 2] = omega[0] * d[1] - omega[1] * d[0];
            }
            motions.push(r);
        }
    }
    motions
}

/// Orthonormal basis of the span of `motions`, dropping dependent vectors
/// relative to the size of the largest one.
fn orthonormal(motions: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let peak = motions.iter().map(|m| norm(m)).fold(0.0, f64::max);
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for m in motions {
        let mut w = m.clone();
        for _ in 0..2 {
            for q in &basis {
                let overlap = dot(&w, q);
                for (wi, qi) in w.iter_mut().zip(q) {
                    *wi -= overlap * qi;
                }
            }
        }
        let len = norm(&w);
        if len > 1e-9 * peak.max(f64::MIN_POSITIVE) {
            basis.push(w.iter().map(|c| c / len).collect());
        }
    }
    basis
}

fn reference_projection(x: &[f64], v: &[f64], periodic: bool) -> Vec<f64> {
    let basis = orthonormal(&rigid_motions(x, periodic));
    let mut out = v.to_vec();
    for q in &basis {
        let overlap = dot(v, q);
        for (o, qi) in out.iter_mut().zip(q) {
            *o -= overlap * qi;
        }
    }
    out
}

fn horizontal_rank(m: &RiemannianManifold, x: &[f64]) -> usize {
    let mut trace = 0.0;
    for i in 0..x.len() {
        let mut e = vec![0.0; x.len()];
        e[i] = 1.0;
        trace += project(m, x, &e)[i];
    }
    trace.round() as usize
}

const CLUSTER: [f64; 15] = [
    0.0, 0.0, 0.0, 1.1, 0.1, -0.2, 0.3, 0.9, 0.4, -0.5, 0.2, 1.0, 0.6, -0.7, 0.8,
];

fn scaled(x: &[f64], scale: f64) -> Vec<f64> {
    x.iter().map(|c| c * scale).collect()
}

fn pseudo_random(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn projection_matches_an_independent_dense_reference_at_every_scale() {
    for periodic in [false, true] {
        let m = quotient(5, periodic);
        for scale in [1e-8, 1e-4, 1.0, 1e4, 1e8] {
            let x = scaled(&CLUSTER, scale);
            for seed in 1..4 {
                let v = pseudo_random(x.len(), seed);
                let got = project(&m, &x, &v);
                let want = reference_projection(&x, &v, periodic);
                let err = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(
                    err < 1e-10,
                    "periodic {periodic} scale {scale}: error {err}"
                );
            }
        }
    }
}

#[test]
fn horizontal_rank_of_a_non_collinear_cluster_is_exact_at_every_scale() {
    for scale in [1e-10, 1e-5, 1.0, 1e5, 1e10] {
        let x = scaled(&CLUSTER, scale);
        assert_eq!(horizontal_rank(&quotient(5, false), &x), 15 - 6, "{scale}");
        assert_eq!(horizontal_rank(&quotient(5, true), &x), 15 - 3, "{scale}");
    }
}

#[test]
fn the_projection_commutes_with_uniform_scaling() {
    let m = quotient(5, false);
    let v = pseudo_random(15, 9);
    let base = project(&m, &CLUSTER, &v);
    for scale in [1e-9, 1e-3, 1e3, 1e9] {
        let got = project(&m, &scaled(&CLUSTER, scale), &v);
        let err = got
            .iter()
            .zip(&base)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-10, "scale {scale}: error {err}");
    }
}

#[test]
fn single_atom_and_coincident_atoms_have_rank_three() {
    let single = [0.3, -0.2, 0.9];
    assert_eq!(horizontal_rank(&quotient(1, false), &single), 0);

    let coincident = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4];
    let m = quotient(3, false);
    assert_eq!(horizontal_rank(&m, &coincident), 9 - 3);
    let v = pseudo_random(9, 4);
    let got = project(&m, &coincident, &v);
    let want = reference_projection(&coincident, &v, false);
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn collinear_atoms_have_rank_five_at_every_scale() {
    for scale in [1e-8, 1.0, 1e8] {
        let line: Vec<f64> = (0..4)
            .flat_map(|i| [0.5 * f64::from(i) * scale, 0.0, 0.0])
            .collect();
        let m = quotient(4, false);
        assert_eq!(horizontal_rank(&m, &line), 12 - 5, "{scale}");
        let v = pseudo_random(12, 5);
        let got = project(&m, &line, &v);
        let want = reference_projection(&line, &v, false);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-10, "scale {scale}: error {err}");
        for motion in rigid_motions(&line, false) {
            let left = norm(&project(&m, &line, &motion));
            assert!(left <= 1e-9 * norm(&motion).max(1e-300), "scale {scale}");
        }
    }
}

#[test]
fn the_retraction_adds_the_step_and_exp_adds_its_horizontal_part() {
    let m = quotient(5, false);
    let v = pseudo_random(15, 11);
    let mut retracted = vec![0.0; 15];
    m.retract(&CLUSTER, &v, &mut retracted).unwrap();
    for i in 0..15 {
        assert!((retracted[i] - CLUSTER[i] - v[i]).abs() < 1e-14);
    }
    let h = project(&m, &CLUSTER, &v);
    let mut exped = vec![0.0; 15];
    m.exp(&CLUSTER, &v, &mut exped).unwrap();
    for i in 0..15 {
        assert!((exped[i] - CLUSTER[i] - h[i]).abs() < 1e-12);
    }
    let mut log = vec![0.0; 15];
    assert!(m.log(&CLUSTER, &exped, &mut log).is_err());
}

#[test]
fn transport_lands_in_the_horizontal_space_at_the_arrival_point() {
    let m = quotient(5, false);
    let from = CLUSTER.to_vec();
    let to: Vec<f64> = CLUSTER
        .iter()
        .zip(pseudo_random(15, 21))
        .map(|(a, d)| a + 0.2 * d)
        .collect();
    let v = project(&m, &from, &pseudo_random(15, 22));
    let mut carried = vec![0.0; 15];
    m.transport(&from, &to, &v, &mut carried).unwrap();
    for motion in rigid_motions(&to, false) {
        let along = dot(&carried, &motion) / norm(&motion);
        assert!(
            along.abs() < 1e-12,
            "component along a rigid motion {along}"
        );
    }
    let want = reference_projection(&to, &v, false);
    for (a, b) in carried.iter().zip(&want) {
        assert!((a - b).abs() < 1e-12);
    }
    let mut again = vec![0.0; 15];
    m.transport(&to, &to, &carried, &mut again).unwrap();
    for (a, b) in again.iter().zip(&carried) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn the_inner_product_is_euclidean_and_the_gradient_is_the_projection() {
    let m = quotient(5, false);
    let u = pseudo_random(15, 31);
    let v = pseudo_random(15, 32);
    assert!((m.inner(&CLUSTER, &u, &v).unwrap() - dot(&u, &v)).abs() < 1e-12);
    let mut rgrad = vec![0.0; 15];
    m.egrad_to_rgrad(&CLUSTER, &u, &mut rgrad).unwrap();
    let want = project(&m, &CLUSTER, &u);
    assert_eq!(rgrad, want);
}
