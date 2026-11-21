//! Riemannian conjugate gradients on a Euclidean quadratic.
//! The transport is the identity.
//!
//! `cargo run --example tutorial_rcg`

use ndarray::{ArrayView1, array};
use rgmin::manifold::Euclidean;
use rgmin::{RcgOptions, RcgTransport, minimize_rcg};

fn main() {
    let out = minimize_rcg(
        &Euclidean,
        array![0.0, 0.0],
        |x: ArrayView1<f64>| {
            let f = (x[0] - 1.0).powi(2) + 4.0 * (x[1] + 2.0).powi(2);
            let g = array![2.0 * (x[0] - 1.0), 8.0 * (x[1] + 2.0)];
            (f, g)
        },
        &RcgOptions::default()
            .with_transport(RcgTransport::Identity)
            .with_max_iter(40)
            .with_grad_tol(1e-8),
    )
    .unwrap();
    assert!((out.x[0] - 1.0).abs() < 1e-4, "{}", out.x);
    assert!((out.x[1] + 2.0).abs() < 1e-4, "{}", out.x);
}
