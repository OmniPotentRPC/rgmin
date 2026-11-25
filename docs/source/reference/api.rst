

API reference
-------------

Two published surfaces, one implementation.

Rust
~~~~

``cargo doc --no-deps`` in a checkout.
The crate root is ``rgmin``.
The types a host works with:

.. table::

    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | Type                                                  | Role                                                                                                            |
    +=======================================================+=================================================================================================================+
    | ``Solver``                                            | One stepper step. Hosts own the loop.                                                                           |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``Lbfgs``                                             | Persistent limited-memory quasi-Newton; ``minimize_box`` keeps iterates in bounds.                              |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``Method`` / ``Control`` / ``LineSearch``             | Which step, and how far.                                                                                        |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``ManifoldKind``                                      | ``project`` / ``retract`` / ``transport``.                                                                      |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``RiemannianManifold``                                | Stiefel, SPD and product handles; see the manifold page.                                                        |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``IrcTrust``                                          | GS2 / Sella MW-sphere equality.                                                                                 |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``lowest_eigenpair`` / ``lowest_mode``                | Matrix-free lowest Hessian pair.                                                                                |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``lowest_eigenpairs``                                 | Block LOBPCG. Residuals and an optional ``T`` (Knyazev 2001).                                                   |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``EigensolverKind`` / ``EigenParams``                 | Closed eigen waist.                                                                                             |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``HessianVector`` / ``FdHvp``                         | ``H v`` without forming ``H``.                                                                                  |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``TryHessianVector``                                  | The same callbacks, each one allowed to fail.                                                                   |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``Solver::step_hvp``                                  | Analytic product for ``newton-cg`` / ``newton-pcg``.                                                            |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``Solver::try_step_hvp``                              | That step. The first host error is ``Error::Host``.                                                             |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``TryApplyHessian``                                   | A block-eigen product that can fail.                                                                            |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``try_lowest_eigenpairs``                             | ``lowest_eigenpairs`` on that product, without ``T``.                                                           |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``try_lowest_eigenpairs_precond``                     | ``lowest_eigenpairs_precond`` on that product, with an optional ``T``.                                          |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``kantorovich_ball`` / ``KantorovichBall``            | Newton--Kantorovich existence ball from supplied majorants (Kantorovich and Akilov 1982; Gragg and Tapia 1974). |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``adjoint_functional_bar`` / ``AdjointFunctionalBar`` | Dual-weighted residual functional bound and remainder (Becker and Rannacher 2001).                              |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
    | ``certify_morse_gap`` / ``MorseGap``                  | Self-adjoint spectral gap certified by Weyl's inequality (Weyl 1912).                                           |
    +-------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+

``newton-cg`` and ``newton-pcg`` are ``Method::from_name`` tokens. The
session counts ``gradient_calls`` and ``hvp_calls`` separately. A
forward-difference product on ``step`` costs one gradient and is
charged as a product.

``kantorovich_ball``, ``adjoint_functional_bar``, and ``certify_morse_gap``
combine caller-supplied majorants in standard ``f64`` floating-point
arithmetic without outward rounding. Given a primal norm and its
induced operator norm, a Jacobian Lipschitz bound on ``B(x0, 2 eta)``
suffices for the Kantorovich ball (`10.1016/B978-0-08-023036-8.50024-2 <https://doi.org/10.1016/B978-0-08-023036-8.50024-2>`_,
`10.1137/0711002 <https://doi.org/10.1137/0711002>`_) :cite:`kantorovichNewtonsMethod1982,graggOptimalErrorBounds1974`.
``adjoint_functional_bar`` pairs ``first_order = |lambda^T F(x0)|`` for an
exact adjoint ``J(x0)^T lambda = grad q(x0)`` with ``gradient_norm`` in the
dual norm and ``hessian_bound`` in the induced operator norm from the
primal space to its dual (`10.1017/S0962492901000010 <https://doi.org/10.1017/S0962492901000010>`_)
:cite:`beckerOptimalControlApproach2001`. ``certify_morse_gap``
requires ``operator_lipschitz`` to bound the spectral-norm Lipschitz
constant of the projected self-adjoint operator and subtracts that
perturbation plus ``eigensolver_roundoff`` from ``computed_gap`` by Weyl's
inequality (`10.1007/BF01456804 <https://doi.org/10.1007/BF01456804>`_)
:cite:`weylAsymptotischeVerteilungsgesetzEigenwerte1912`.

The generated Sphinx pages under ``crates/rgmin/`` (when the docs
environment is built with ``sphinxcontrib-rust``) are the same
rustdoc, exported.

.. table::

    +---------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+
    | Type                                                                                                    | Role                                                              |
    +=========================================================================================================+===================================================================+
    | ``magnus::magnus_step`` / ``magnus::MagnusStep``                                                        | Sixth-order Magnus map with an embedded fourth-order exponential. |
    +---------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+
    | ``magnus::expm`` / ``magnus::Exponential`` / ``magnus::gauss_legendre_three`` / ``magnus::MagnusError`` | Padé matrix exponential, Gauss-Legendre nodes and typed refusal.  |
    +---------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+

C
~

``include/rgmin.h`` is the C ABI. ``cargo cinstall`` installs that
header and ``dlpack/dlpack.h``. ``include/xts.h`` is the compatibility
alias. The hourglass is ``rgmin_solver_create`` /
``rgmin_solver_step`` / ``rgmin_solver_free``, plus ``rgmin_minimize``.
Tensors are dlpk ``DLManagedTensorVersioned``. HTML for this waist
is Doxygen plus doxyYoda; see :doc:`howto/doxygen <../howto/doxygen>`.

``rgmin_minimize_eindir`` takes an ``eindir_objective_t`` and an
``eindir_abi_stamp_t`` that no ``rgmin_`` function builds. librgmin
therefore re-exports the C ABI of eindir-core as public API:
``eindir_core_abi_compatible``, ``eindir_core_abi_family``,
``eindir_core_abi_stamp``, ``eindir_core_version``, ``eindir_last_error`` and
the ``eindir_objective_`` functions ``new``, ``free``, ``eval``, ``grad``,
``has_grad``, ``dim``, ``bounds_low`` and ``bounds_high``. A C caller that links
only librgmin builds the objective with ``eindir_objective_new`` and the
stamp with ``eindir_core_abi_stamp``. These names have been exported
since 0.7.0.

What belongs next door
~~~~~~~~~~~~~~~~~~~~~~

Band, min-mode, and IRC **sessions** are `rgsaddle <https://github.com/OmniPotentRPC/rgsaddle>`_. Potentials
are `rgpot <https://github.com/OmniPotentRPC/rgpot>`_.
