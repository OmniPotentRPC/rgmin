

Quickstart
----------

This tutorial minimizes the 2-D Rosenbrock banana from the classic start
``(-1.2, 1.0)`` with L-BFGS and the strong Wolfe line search.

What you will need
~~~~~~~~~~~~~~~~~~

A checkout of ``rgmin`` next to ``eindir`` (the crate path-depends on
``../eindir``). Build on the remote builder.

Rust
~~~~

.. code:: rust

    use eindir_core::objectives::Rosenbrock;
    use ndarray::array;
    use rgmin::{Control, LineSearch, Method, minimize_method};

    let obj = Rosenbrock::<2>::new();
    let report = minimize_method(
        &obj,
        array![-1.2, 1.0],
        &Control { maxiter: 200, gtol: 1e-8, istep: 1.0, maxmove: None, ftol_rel: None },
        Method::lbfgs(),
        LineSearch::Wolfe { c1: 1e-4, c2: 0.9, maxiter: 20 },
    )
    .unwrap();
    assert!(report.value < 1e-8);

The same pair history can survive between calls when the start of the next
relaxation is a perturbation of the last minimum:

.. code:: rust

    use ndarray::array;
    use rgmin::Lbfgs;

    let mut opt = Lbfgs::default();
    let x0 = array![-1.2, 1.0];
    let (_f, x, _evals) = opt.minimize(x0.view(), 200, |x| {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        Some((a * a + 100.0 * b * b, {
            use ndarray::array;
            array![
                -2.0 * a - 400.0 * x[0] * b,
                200.0 * b,
            ]
        }))
    });

Session
~~~~~~~

Hosts that already own an outer loop hold a ``Solver`` and call
``step`` once per iteration:

.. code:: rust

    use rgmin::{Control, Method, Solver};

    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2);
    let rep = solver.step(&obj, &mut x).unwrap();

``set_manifold`` retracts that step. Isolated molecules use
``RigidQuotient`` (``R^{3N}/SE(3)``) or ``MwRigid`` (mass-weighted
Eckart). Sphere, SO(3), Stiefel, and SE(3) are matrix-manifold
embeddings. Euclidean is the default.

What just happened
~~~~~~~~~~~~~~~~~~

1. ``Method::lbfgs()`` dispatched the cold-start two-loop recursion.

2. ``LineSearch::Wolfe`` enforced both sufficient decrease and curvature,
   so every stored ``(s, y)`` pair describes a measured Hessian action.

3. ``Lbfgs`` keeps those pairs so a nearby restart does not rebuild the
   inverse-Hessian approximation from identity.
