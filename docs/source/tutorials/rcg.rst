

Riemannian conjugate gradients
------------------------------

``minimize_rcg`` runs Fletcher-Reeves, PRP+ or Hager-Zhang on a manifold the crate already retracts.
The vector transport is that manifold's, scaled back to its original length when the image is longer (Sato and Iwai, `10.1080/02331934.2013.836650 <https://doi.org/10.1080/02331934.2013.836650>`_).
``RcgTransport::Identity`` is the Euclidean case.
The default line search is strong Wolfe with curvature constant 0.1.
``LineSearch::default`` selects Brent instead.

The listing is ``examples/tutorial_rcg.rs``, whole. Continuous integration runs it, so it is the program as it builds.

.. code:: rust

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

The example calls ``minimize_rcg`` on a quadratic and checks both coordinates.

``RcgOptions::validate`` refuses a gradient tolerance that is not finite or is negative, and an initial step that is not finite and positive, with ``Error::InvalidControl``.
``minimize_rcg`` calls it before the first evaluation.
A projection or retraction that cannot be carried out is returned as its typed error rather than skipped.
A start off the manifold is retracted by a zero step, so the run begins on the manifold.

In a session
~~~~~~~~~~~~

``Method::rcg()`` runs the same step inside a ``Solver``.
There it keeps the session's rules:

- ``Control::maxmove`` and the per-atom cap bound the move that is taken.
  A trial beyond a cap lies outside the line-search domain, so the search contracts to the cap.
  The default per-atom cap stays off for this method.

- A distance constraint set composes with the configured manifold.
  The tangent space is the intersection of the manifold's and the bonds', and the retraction alternates between the two until both hold.
  A set with masses steps along the cut gradient raised by the inverse masses, which is tangent to the bonds, and its line search measures that direction in the mass metric.

- The stop and the report read the gradient cut by the bonds, so a constrained minimum with a pure constraint force meets the tolerance.
  A start that misses its bonds is retracted onto them first, so a reported tolerance holds at a point that keeps every bond.

- A level-set retraction that moves the point clears the held direction.
  The next step opens on steepest descent.

- The step reads the field the stop reads.
  The gradient, the tangent projection and the transport are the session's: on a rigid quotient, a product with rigid blocks, or a periodic session they carry the session masses, the periodic flag and the removal of the declared rigid motion.
  The Wolfe slopes, the conjugacy coefficient and the stop are formed in that one field.

- A start off the manifold is projected onto it, as ``minimize_rcg`` does with a zero retraction, so the first evaluation, the held value and the point handed back are on the manifold.
  A zero-budget run judges the start at that same point.
  Nothing is refused for it.

- Every trial stays in the objective's box and on the manifold.
  Where the manifold fills the ambient coordinates, the trial is the retraction clipped to the box, which is the projection onto that intersection.
  On every other manifold the trial is the retraction of a tangent step in the box's tangent cone, and a retraction that would leave the box is refused so the search shortens.
  The slope is the slope of the curve that is walked.

- A step whose best finite decrease misses the energy noise of ``NoiseFloor::with_eps_f`` keeps its start and reports ``Status::NoiseFloor``, as the other methods do.
