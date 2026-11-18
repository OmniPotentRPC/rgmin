

API reference
-------------

Two published surfaces, one implementation.

Rust
~~~~

``cargo doc --no-deps`` in a checkout.
The crate root is ``rgmin``.
The types a host works with:

.. table::

    +-------------------------------------------+------------------------------------------------------------------------------------+
    | Type                                      | Role                                                                               |
    +===========================================+====================================================================================+
    | ``Solver``                                | One stepper step. Hosts own the loop.                                              |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``Lbfgs``                                 | Persistent limited-memory quasi-Newton; ``minimize_box`` keeps iterates in bounds. |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``Method`` / ``Control`` / ``LineSearch`` | Which step, and how far.                                                           |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``ManifoldKind``                          | ``project`` / ``retract`` / ``transport``.                                         |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``RiemannianManifold``                    | Stiefel, SPD and product handles; see the manifold page.                           |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``IrcTrust``                              | GS2 / Sella MW-sphere equality.                                                    |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``lowest_eigenpair`` / ``lowest_mode``    | Matrix-free lowest Hessian pair.                                                   |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``lowest_eigenpairs``                     | Block LOBPCG. Residuals and an optional ``T`` (Knyazev 2001).                      |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``EigensolverKind`` / ``EigenParams``     | Closed eigen waist.                                                                |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``HessianVector`` / ``FdHvp``             | ``H v`` without forming ``H``.                                                     |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``TryHessianVector``                      | The same callbacks, each one allowed to fail.                                      |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``Solver::step_hvp``                      | Analytic product for ``newton-cg`` / ``newton-pcg``.                               |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``Solver::try_step_hvp``                  | That step. The first host error is ``Error::Host``.                                |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``TryApplyHessian``                       | A block-eigen product that can fail.                                               |
    +-------------------------------------------+------------------------------------------------------------------------------------+
    | ``try_lowest_eigenpairs``                 | ``lowest_eigenpairs`` on that product, with or without ``T``.                      |
    +-------------------------------------------+------------------------------------------------------------------------------------+

``newton-cg`` and ``newton-pcg`` are ``Method::from_name`` tokens. The
session counts ``gradient_calls`` and ``hvp_calls`` separately. A
forward-difference product on ``step`` costs one gradient and is
charged as a product.

The generated Sphinx pages under ``crates/rgmin/`` (when the docs
environment is built with ``sphinxcontrib-rust``) are the same
rustdoc, exported.

C
~

``include/rgmin.h`` is the C ABI. ``cargo cinstall`` installs that
header and ``dlpack/dlpack.h``. ``include/xts.h`` is the compatibility
alias. The hourglass is ``rgmin_solver_create`` /
``rgmin_solver_step`` / ``rgmin_solver_free``, plus ``rgmin_minimize``.
Tensors are dlpk ``DLManagedTensorVersioned``. HTML for this waist
is Doxygen plus doxyYoda; see :doc:`howto/doxygen <../howto/doxygen>`.

What belongs next door
~~~~~~~~~~~~~~~~~~~~~~

Band, min-mode, and IRC **sessions** are `rgsaddle <https://github.com/OmniPotentRPC/rgsaddle>`_. Potentials
are `rgpot <https://github.com/OmniPotentRPC/rgpot>`_.
