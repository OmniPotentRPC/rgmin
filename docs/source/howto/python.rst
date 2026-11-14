Python
======

The package is ``python/``. Install it with maturin. ASE is optional and
imported when an optimizer is constructed.

.. code:: bash

   pip install "python/[ase]"

.. code:: python

   from ase.build import molecule
   from ase.calculators.emt import EMT
   from rgmin import FIRE, LBFGS

   atoms = molecule("H2")
   atoms.calc = EMT()
   opt = LBFGS(atoms, trajectory="opt.traj", logfile=None)
   opt.attach(lambda: atoms.get_potential_energy())
   opt.run(fmax=0.05)

The Rust L-BFGS default is the eOn cap, a max-move step with no line
search. ``LBFGS`` matches ASE's length control, ``LBFGS(eon_cap=True)``
selects the Rust default, and ``LBFGS(use_line_search=True)`` opts into
the strong Wolfe search. Stubs live in ``python/rgmin`` and
``py.typed`` marks the package.

``lowest_eigenpairs(hvp, x0_block)`` returns the k lowest pairs of a
symmetric product as NumPy arrays. ``hvp(x, v)`` is ``H(x) v``.
:doc:`../tutorials/eigenpairs` is the short example, and continuous
integration runs it.

C and C++ use ``include/rgmin.h`` and ``include/rgmin/optimize.hpp``.
The C session is a handle: ``rgmin_solver_create``, ``rgmin_solver_step``,
``rgmin_solver_free``. ``rgmin_solver_dim`` and ``rgmin_solver_steps``
read that handle. ``rgmin_last_error_message`` is the same borrowed
string as ``rgmin_last_error``.
