

L-BFGS quadratic model through HiGHS
------------------------------------

The two-loop recursion is the unconstrained L-BFGS step. HiGHS is
used only when that step must sit in a feasible set: an L\ :sub:`inf`\ trust
region, a box on ``x + p``, and optional linear equalities.

The QP is ``min 1/2 ||p - d||^2`` with ``d = -H g`` and ``Q = I``. A dense
compact Hessian is not formed; that matrix is indefinite after a tiny
accepted step and ``Highs_run`` does not return.

Unconstrained, ``highs_step`` is the two-loop direction. A box or trust
region scales the whole increment (direction preserved). Packed
centering is a mean subtract. Arbitrary equalities still go to HiGHS.

When a host Hessian is present (``rgmin_solver_step_hess``) and
``rgmin_solver_set_highs`` is on, HiGHS solves the convex Newton QP
``min 1/2 p^T P p + g^T p`` with per-coordinate boxes from
``rgmin_solver_set_box`` (a NULL side is unbounded) and, if no box
was set, from atom maxmove. ``rgmin_solver_set_trust`` sets the
L\ :sub:`inf`\ radius on the step; ``rgmin_solver_add_equality``
appends ``a · p = rhs``. That is the path eOn / rgpot / any eindir
objective share. Without the ``highs`` cargo feature every setter
returns 1.

Enable the feature.
Its ``highs-sys`` crate builds HiGHS from source with CMake and a C++ compiler.
The tests also need SciPy (:doc:`install`):

.. code:: bash

    cargo test --features highs

.. code:: rust

    use rgmin::{HighsStep, Lbfgs};

    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep {
        trust: Some(0.5),
        lo: Some(vec![-1.0]),
        hi: Some(vec![1.0]),
        ..Default::default()
    });

On ``develop``, ``lo`` and ``hi`` give one bound per coordinate.
The 0.5.2 release takes one ``f64`` for all coordinates (:doc:`install`).

landfold ``--highs`` is this step on chi, not sequential LP.
