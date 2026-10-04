#![cfg(rgmin_has_libkrylov)]

use ndarray::{Array1, ArrayView1};
use rgmin::{ApplyHessian, EigenParams, EigensolverKind, lowest_mode};
use std::sync::{Arc, Barrier};
struct Diagonal;
impl ApplyHessian for Diagonal {
    fn apply_hessian(&self, _: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        std::thread::yield_now();
        Array1::from_iter(
            v.iter()
                .enumerate()
                .map(|(i, x)| x * if i == 0 { -2.5 } else { i as f64 + 1.0 }),
        )
    }
}
#[test]
fn independent_concurrent_pairs_are_valid() {
    assert!(EigensolverKind::Libkrylov.is_linked());
    let gate = Arc::new(Barrier::new(4));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let gate = gate.clone();
            std::thread::spawn(move || {
                let x = Array1::zeros(32);
                let seed = Array1::from_elem(32, 1.0 / 32.0_f64.sqrt());
                gate.wait();
                for _ in 0..8 {
                    let mode = lowest_mode(
                        &Diagonal,
                        x.view(),
                        seed.view(),
                        &EigenParams {
                            kind: EigensolverKind::Libkrylov,
                            krylov: 16,
                            max_iter: 256,
                            tol: 1e-9,
                            nev: 1,
                            ..EigenParams::default()
                        },
                    )
                    .unwrap();
                    assert!((mode.value + 2.5).abs() < 1e-12);
                    let r = Diagonal.apply_hessian(x.view(), mode.vector.view())
                        - &mode.vector * mode.value;
                    assert!(r.dot(&r).sqrt() <= 1e-9);
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

#[test]
fn recursive_callback_is_rejected_without_disturbing_the_outer_solve() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Recursive(AtomicUsize);
    impl ApplyHessian for Recursive {
        fn apply_hessian(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
            let error = lowest_mode(
                &Diagonal,
                x,
                v,
                &EigenParams {
                    kind: EigensolverKind::Libkrylov,
                    max_iter: 256,
                    tol: 1e-9,
                    ..EigenParams::default()
                },
            )
            .unwrap_err();
            assert!(matches!(
                error,
                rgmin::Error::Libkrylov {
                    what: "recursive eigensolver call"
                }
            ));
            self.0.fetch_add(1, Ordering::Relaxed);
            Diagonal.apply_hessian(x, v)
        }
    }
    let h = Recursive(AtomicUsize::new(0));
    let x = Array1::zeros(32);
    let seed = Array1::from_elem(32, 1.0 / 32.0_f64.sqrt());
    let mode = lowest_mode(
        &h,
        x.view(),
        seed.view(),
        &EigenParams {
            kind: EigensolverKind::Libkrylov,
            max_iter: 256,
            tol: 1e-9,
            ..EigenParams::default()
        },
    )
    .unwrap();
    assert!((mode.value + 2.5).abs() < 1e-12);
    let residual = Diagonal.apply_hessian(x.view(), mode.vector.view()) - &mode.vector * mode.value;
    assert!(residual.dot(&residual).sqrt() <= 1e-9);
    assert!(h.0.load(Ordering::Relaxed) > 0);
}
