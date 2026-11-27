

Conjugate gradients
-------------------

Two steps share the Polak-Ribière coefficient. ``Method::eon_cg`` selects that coefficient and ``Restart::powell`` at threshold 0.5, and the session still line-searches. ``should_restart`` compares the gradient with the previous gradient. A session step calls ``should_restart_metric`` with ``z = P^{-1} g`` when a metric is installed, and with ``z = g`` otherwise. ``minimize_eon_cg`` is the client step: the previous direction is kept only when |g_k . g_{k-1}| < 0.5 |g_{k-1}|^2, one curvature probe, no line search, then the per-atom cap.

The listing is ``examples/tutorial_cg.rs``, whole. Continuous integration runs it, so it is the program as it builds.

.. code:: rust

    //! Line-search-free conjugate gradients, the eOn client step.
    //!
    //! `cargo run --example tutorial_cg`

    use rgmin::{EonCg, minimize_eon_cg};

    fn main() {
        let mut x = vec![1.0, -2.0, 0.5];
        let opts = EonCg::default().with_gtol(1e-8).with_max_iter(200);
        let report = minimize_eon_cg(
            &mut x,
            |xs, gs| {
                let mut f = 0.0;
                for (x, g) in xs.iter().zip(gs.iter_mut()) {
                    f += x * x;
                    *g = 2.0 * x;
                }
                f
            },
            &opts,
        );
        assert!(report.converged, "grad {}", report.grad);
        assert!(report.energy < 1e-12, "energy {}", report.energy);
        assert!(x.iter().all(|v| v.abs() < 1e-5), "{x:?}");
    }

``Method::polak_ribiere_plus`` is SciPy's ``method="CG"``: ``max(0, β)``. The Python session named ``cg`` uses that coefficient with a strong Wolfe search, because SciPy does. The eOn client step does not.
