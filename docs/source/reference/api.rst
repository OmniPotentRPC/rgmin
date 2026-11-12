

API reference
-------------

Two published surfaces, one implementation.

Rust
~~~~

``cargo doc --no-deps`` in a checkout.
The crate root is ``rgmin``.
The types a host works with:

.. table::

    +-------------------------------------------+------------------------------------------------------+
    | Type                                      | Role                                                 |
    +===========================================+======================================================+
    | ``Solver``                                | One stepper step. Hosts own the loop.                |
    +-------------------------------------------+------------------------------------------------------+
    | ``relax_cell`` / ``PeriodicPotential``    | Atoms plus cell strain. Stress on the minimum.       |
    +-------------------------------------------+------------------------------------------------------+
    | ``Method`` / ``Control`` / ``LineSearch`` | Which step, and how far.                             |
    +-------------------------------------------+------------------------------------------------------+
    | ``ManifoldKind``                          | ``project`` / ``retract`` / ``transport``.           |
    +-------------------------------------------+------------------------------------------------------+
    | ``IrcTrust``                              | GS2 / Sella MW-sphere equality.                      |
    +-------------------------------------------+------------------------------------------------------+
    | ``lowest_eigenpair`` / ``lowest_mode``    | Matrix-free lowest Hessian pair.                     |
    +-------------------------------------------+------------------------------------------------------+
    | ``EigensolverKind`` / ``EigenParams``     | Closed eigen waist.                                  |
    +-------------------------------------------+------------------------------------------------------+
    | ``HessianVector`` / ``FdHvp``             | ``H v`` without forming ``H``.                       |
    +-------------------------------------------+------------------------------------------------------+
    | ``Solver::step_hvp``                      | Analytic product for ``newton-cg`` / ``newton-pcg``. |
    +-------------------------------------------+------------------------------------------------------+

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
