

API reference
-------------

Two published surfaces, one implementation.

Rust
~~~~

``cargo doc --no-deps`` on the remote builder. The crate root is
``rgmin``. The types a host actually holds:

.. table::

    +-------------------------------------------+--------------------------------------------+
    | Type                                      | Role                                       |
    +===========================================+============================================+
    | ``Solver``                                | One stepper step. Hosts own the loop.      |
    +-------------------------------------------+--------------------------------------------+
    | ``Method`` / ``Control`` / ``LineSearch`` | Which step, and how far.                   |
    +-------------------------------------------+--------------------------------------------+
    | ``ManifoldKind``                          | ``project`` / ``retract`` / ``transport``. |
    +-------------------------------------------+--------------------------------------------+
    | ``IrcTrust``                              | GS2 / Sella MW-sphere equality.            |
    +-------------------------------------------+--------------------------------------------+
    | ``lowest_eigenpair`` / ``lowest_mode``    | Matrix-free lowest Hessian pair.           |
    +-------------------------------------------+--------------------------------------------+
    | ``lowest_mode_slepc`` / ``SlepcParams``   | Typed SLEPc EPS extras (Pmat / ST).        |
    +-------------------------------------------+--------------------------------------------+
    | ``EigensolverKind`` / ``EigenParams``     | Closed eigen waist.                        |
    +-------------------------------------------+--------------------------------------------+
    | ``HessianVector`` / ``FdHvp``             | ``H v`` without forming ``H``.             |
    +-------------------------------------------+--------------------------------------------+

The generated Sphinx pages under ``crates/rgmin/`` (when the docs
environment is built with ``sphinxcontrib-rust``) are the same
rustdoc, exported.

C
~

``include/xts.h`` (and the ``rgmin.h`` cbindgen name). The hourglass
is ``rgmin_solver_create`` / ``rgmin_solver_step`` /
``rgmin_solver_free``, plus ``rgmin_minimize``. Tensors are dlpk
``DLManagedTensorVersioned``. HTML for this waist is Doxygen plus
doxyYoda; see `howto/doxygen <../howto/doxygen.rst>`_.

What belongs next door
~~~~~~~~~~~~~~~~~~~~~~

Band, min-mode, and IRC **sessions** are `rgsaddle <https://github.com/OmniPotentRPC/rgsaddle>`_. Potentials
are `rgpot <https://github.com/OmniPotentRPC/rgpot>`_.
