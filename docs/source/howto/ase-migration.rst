

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
match. The nanobind classes live in ``rgmin.ase`` and are a separate binding.

``rgmin.ase.BFGSLineSearch`` is also exported as ``QuasiNewton``.
``rgmin.ase.LBFGSLineSearch`` subclasses ``rgmin.ase.LBFGS``. Both match the
ASE 3.29.0 constructors
(`BFGSLineSearch <https://gitlab.com/ase/ase/-/blob/3.29.0/ase/optimize/bfgslinesearch.py>`__,
`LBFGSLineSearch <https://gitlab.com/ase/ase/-/blob/3.29.0/ase/optimize/lbfgs.py>`__,
`QuasiNewton <https://gitlab.com/ase/ase/-/blob/3.29.0/ase/optimize/__init__.py>`__).
``c1=0.23``, ``c2=0.46``, ``alpha``, ``stpmax=50``, ``memory=100``, and ``maxstep``
are the ASE values. ``alpha`` is the fixed inverse-Hessian scale ``1/alpha``
on the Rust BFGS or L-BFGS session. ``stpmax`` bounds the line-search
multiplier when that bound is tighter than the per-atom cap. ``damping``
is stored and, as in ASE, does not scale a line-search step. The first
trial is the full quasi-Newton step. ``restart`` replays the rgmin
evaluation log; it does not read ASE's Hessian JSON, and
``replay_trajectory`` is not ASE's Hessian replay. The BFGS line-search
log adds ASE's force-call column. The L-BFGS line-search log keeps the
shared ``Step``, ``Time``, ``Energy``, ``fmax`` columns.
