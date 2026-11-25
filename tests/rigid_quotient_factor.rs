//! Rigid quotient projection and tangent-projected eigenpairs.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::{ApplyHessian, RiemannianManifold, lowest_eigenpairs_projected};

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

fn project(periodic: bool, x: &[f64], v: &[f64]) -> Vec<f64> {
    let m = RiemannianManifold::rigid_quotient(x.len() / 3, periodic).unwrap();
    let mut out = vec![0.0; x.len()];
    m.project(x, v, &mut out).unwrap();
    out
}

#[test]
fn isolated_quotient_removes_translation_and_rotation() {
    let x = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let trans = [0.4, -0.2, 0.7, 0.4, -0.2, 0.7];
    let out = project(false, &x, &trans);
    assert!(norm(&out) < 1e-12, "translation survived {out:?}");

    let rot = [0.0, -1.0, 0.0, 0.0, 1.0, 0.0];
    let out = project(false, &x, &rot);
    assert!(norm(&out) < 1e-12, "rotation survived {out:?}");

    let stretch = [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let out = project(false, &x, &stretch);
    assert!((norm(&out) - norm(&stretch)).abs() < 1e-12, "{out:?}");
    assert!((out[0] + 1.0).abs() < 1e-12 && (out[3] - 1.0).abs() < 1e-12);
}

#[test]
fn periodic_quotient_removes_translation_only() {
    let x = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let trans = [0.4, -0.2, 0.7, 0.4, -0.2, 0.7];
    let out = project(true, &x, &trans);
    assert!(norm(&out) < 1e-12, "periodic translation survived {out:?}");

    let rot = [0.0, -1.0, 0.0, 0.0, 1.0, 0.0];
    let out = project(true, &x, &rot);
    for (got, want) in out.iter().zip(rot) {
        assert!(
            (got - want).abs() < 1e-12,
            "periodic rotation changed {out:?}"
        );
    }
}

#[test]
fn rigid_quotient_refuses_an_empty_cluster() {
    assert!(RiemannianManifold::rigid_quotient(0, false).is_err());
}

struct Spectrum {
    translation: Array1<f64>,
}

impl ApplyHessian for Spectrum {
    fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let along = v.dot(&self.translation);
        let mut hv = v.to_owned();
        for (i, entry) in hv.iter_mut().enumerate() {
            *entry *= 0.5 + 0.25 * i as f64;
        }
        hv - &self.translation * (along * 50.0)
    }
}

fn jacobi_eigs(a: &mut [Vec<f64>]) -> Vec<f64> {
    let n = a.len();
    for _ in 0..80 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let aij = a[i][j];
                off += aij * aij;
                if aij.abs() <= 1e-15 {
                    continue;
                }
                let tau = (a[j][j] - a[i][i]) / (2.0 * aij);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let aii = a[i][i];
                let ajj = a[j][j];
                a[i][i] = c * c * aii - 2.0 * s * c * aij + s * s * ajj;
                a[j][j] = s * s * aii + 2.0 * s * c * aij + c * c * ajj;
                a[i][j] = 0.0;
                a[j][i] = 0.0;
                for k in (0..n).filter(|&k| k != i && k != j) {
                    let aik = a[i][k];
                    let ajk = a[j][k];
                    let ik = c * aik - s * ajk;
                    let jk = s * aik + c * ajk;
                    a[i][k] = ik;
                    a[k][i] = ik;
                    a[j][k] = jk;
                    a[k][j] = jk;
                }
            }
        }
        if off.sqrt() <= 1e-14 {
            break;
        }
    }
    (0..n).map(|i| a[i][i]).collect()
}

#[test]
fn projected_eigenpairs_match_the_deflated_dense_spectrum() {
    let m = RiemannianManifold::rigid_quotient(3, false).unwrap();
    let x = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.8, 0.1];
    let n = x.len();
    let project_vec = |v: &Array1<f64>| {
        let mut out = vec![0.0; n];
        m.project(&x, v.as_slice().expect("contiguous"), &mut out)
            .unwrap();
        Array1::from(out)
    };
    let translation = Array1::from(vec![
        1.0 / 3.0_f64.sqrt(),
        0.0,
        0.0,
        1.0 / 3.0_f64.sqrt(),
        0.0,
        0.0,
        1.0 / 3.0_f64.sqrt(),
        0.0,
        0.0,
    ]);
    let mut basis = Vec::new();
    for axis in 0..n {
        let mut e = Array1::zeros(n);
        e[axis] = 1.0;
        let mut w = project_vec(&e);
        for q in &basis {
            let overlap = w.dot(q);
            w = &w - q * overlap;
        }
        let len = w.dot(&w);
        let len = len.sqrt();
        if len > 1e-10 {
            basis.push(&w / len);
        }
    }
    let rank = basis.len();
    assert_eq!(rank, 3, "horizontal rank of a non-collinear triangle");
    let op = Spectrum {
        translation: translation.clone(),
    };
    let mut gram = vec![vec![0.0; rank]; rank];
    for i in 0..rank {
        let image = project_vec(&op.apply_hessian(ArrayView1::from(&x), basis[i].view()));
        for j in 0..=i {
            let value = basis[j].dot(&image);
            gram[i][j] = value;
            gram[j][i] = value;
        }
    }
    let mut want = jacobi_eigs(&mut gram);
    want.sort_by(|a, b| a.total_cmp(b));
    let k = 2;
    let mut seed = Array2::zeros((n, k));
    for (j, column) in basis.iter().take(k).enumerate() {
        seed.column_mut(j).assign(column);
    }
    struct NoPrecond;
    impl rgmin::ApplyBlockPreconditioner for NoPrecond {
        fn apply_block_preconditioner(&self, block: ArrayView2<f64>) -> Array2<f64> {
            block.to_owned()
        }
    }
    let mut project = |v: &mut Array1<f64>| {
        let projected = project_vec(v);
        v.assign(&projected);
    };
    let found = lowest_eigenpairs_projected(
        &op,
        None::<&NoPrecond>,
        ArrayView1::from(&x),
        seed.view(),
        1e-10,
        40,
        &mut project,
    )
    .unwrap();
    assert_eq!(found.pairs.len(), k);
    for (pair, truth) in found.pairs.iter().zip(want.iter().take(k)) {
        assert!(
            (pair.value - truth).abs() < 1e-6,
            "pair {} vs dense {truth} (spectrum {want:?})",
            pair.value
        );
        let drift = (&project_vec(&pair.vector) - &pair.vector)
            .iter()
            .map(|c| c * c)
            .sum::<f64>()
            .sqrt();
        assert!(drift < 1e-8, "mode left the tangent space by {drift}");
        assert!(pair.vector.dot(&translation).abs() < 1e-8);
    }
    let lowest = found.pairs[0].value;
    assert!(
        (lowest - want[0]).abs() < 1e-6,
        "lowest {lowest} vs deflated {want:?}"
    );
}
