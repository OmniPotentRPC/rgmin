

Moving an ASE script to rgmin.optimize
--------------------------------------

The maturin package ``python/`` provides ``rgmin.optimize``. Replace

.. code:: python

    from ase.optimize import BFGS, LBFGS, FIRE

with

.. code:: python

    from rgmin.optimize import BFGS, LBFGS, FIRE

``LBFGSLineSearch``, ``FIRE2``, and ``GPMin`` match those ASE names (see
:doc:`gpmin` for GPMin hyperparameters and step-cap behavior).
The other gradient methods of ``Session`` are ``SR1``, ``SR2``, ``SteepestDescent``,
``BarzilaiBorwein``, ``Adam``, the nonlinear CG family, ``EonCG``, ``NewtonCG``
and ``NewtonPCG``.
Importing ``rgmin`` does not import ASE. The classes do, when they are constructed.

Each class is an ``ase.optimize.optimize.Optimizer``. ``run``, ``irun``, ``attach``,
``trajectory`` and ``logfile`` are ASE's. The step reads and writes
``self.optimizable``, so ``FixAtoms``, ``FixedLine``, ``FixCartesian``, ``FixedPlane``
and the cell filters work.

``LBFGS`` uses ASE's length control unless ``eon_cap=True``. ``Adam`` takes
``learning_rate`` (default 0.02) as the session step scale. On this line the
session normalizes Adam's direction to unit per-atom length before that
scale, and ``maxstep`` still caps the longest atom. ``BFGS`` and ``FIRE`` on a
cell filter reach the same minimum as ASE; their iterates are not ASE's,
because the 0.6.0 session rescales the first BFGS step and integrates FIRE
on its own schedule. ``LBFGS`` on the same filters keeps ASE's iterates.

A restart file replays the recorded evaluations into a fresh session and
refuses a file written by another class or with parameters that no longer
match. The nanobind classes live in ``rgmin.ase``. They use the same
evaluation log (``rgmin-replay-1``), and the same ASE constructor contract,
as this package. The conformance table is on the migration page.
