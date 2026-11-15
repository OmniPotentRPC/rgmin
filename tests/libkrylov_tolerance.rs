#![cfg(rgmin_has_libkrylov)]

use ndarray::{Array1, ArrayView1};
use rgmin::{ApplyHessian, EigenParams, EigensolverKind, lowest_mode};

struct Diagonal(Array1<f64>);
impl ApplyHessian for Diagonal {
    fn apply_hessian(&self, _: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        &self.0 * &v
    }
}

#[test]
fn requested_residual_tolerance_controls_basis_pruning() {
    let n = 32;
    let diagonal = Array1::from_iter((0..n).map(|i| if i == 0 { -2.5 } else { i as f64 + 1.0 }));
    let h = Diagonal(diagonal.clone());
    let x = Array1::zeros(n);
    let seed = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
    let tolerance = 1e-9;
    let mode = lowest_mode(
        &h,
        x.view(),
        seed.view(),
        &EigenParams::default()
            .with_kind(EigensolverKind::Libkrylov)
            .with_krylov(16)
            .with_max_iter(256)
            .with_tol(tolerance)
            .with_nev(1),
    )
    .expect("the configured backend must resolve the requested pair");
    assert!((mode.value + 2.5).abs() < 1e-12, "{}", mode.value);
    assert!(mode.vector[0].abs() > 1.0 - 1e-10, "{:?}", mode.vector);
    let residual = &diagonal * &mode.vector - &mode.vector * mode.value;
    assert!(residual.dot(&residual).sqrt() <= tolerance, "{residual:?}");
}
