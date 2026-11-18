

GROMACS steepest descent
------------------------

``Method::GromacsSteep`` implements the update documented in the
`GROMACS reference manual, Energy Minimization: Steepest Descent <https://manual.gromacs.org/current/reference-manual/algorithms/energy-minimization.html#steepest-descent>`_:



.. math::

    r_{n+1} = r_n + h_n F_n / \max_i |F_{n,i}|.

The maximum is the largest Euclidean norm of one consecutive xyz atom
triple. A trailing partial triple is one atom. A strict energy decrease
accepts the trial and sets ``h <- 1.2 h``. An equal or larger energy rejects
the trial, keeps the old point and force, and sets ``h <- shrink * h``.
``GromacsSteepConfig`` defaults ``shrink`` to 0.5.

The implementation follows the state-swap structure and strict energy
comparison in GROMACS 2025.4
`do_steep <https://github.com/gromacs/gromacs/blob/v2025.4/src/gromacs/mdrun/minimize.cpp#L3104>`_.
That function multiplies a rejected ``ustep`` by 0.5 (``minimize.cpp``
line 3104) and an accepted one by 1.2 when ``count > 0`` (line 3092).
The reference manual still says 0.2. ``GromacsSteepConfig::with_shrink``
and ``Solver::set_gromacs_steep_config`` select that factor, or any other
finite value in ``(0, 1)``. The Python ``Session`` keyword ``shrink``
does the same for ``gromacs-steep``.

Stops and reports
~~~~~~~~~~~~~~~~~

.. table::

    +--------------------------------------------------------+----------------------+----------------------------------+
    | condition                                              | ``Status``           | C ABI                            |
    +========================================================+======================+==================================+
    | largest per-atom force is below ``Control::gtol``      | ``Tolerance``        | ``RGMIN_STOP_TOLERANCE``         |
    +--------------------------------------------------------+----------------------+----------------------------------+
    | ``Control::maxiter`` trials were attempted             | ``Budget``           | ``RGMIN_STOP_BUDGET``            |
    +--------------------------------------------------------+----------------------+----------------------------------+
    | the proposed displacement cannot change any coordinate | ``MachinePrecision`` | ``RGMIN_STOP_MACHINE_PRECISION`` |
    +--------------------------------------------------------+----------------------+----------------------------------+

``Report::stop_norm`` is ``StopNorm::MaxAtom`` by default for this method,
even when the input ``Control`` names another norm. An explicit
``Solver::set_stop_norm`` call after construction overrides that default,
the same as for any other method. Rejected trials count as steps.
``Solver::gromacs_step_size`` returns the displacement that the next trial
will use.

Non-finite callback values or gradients are runtime errors
(``Error::Oracle`` in Rust, ``RGMIN_INTERNAL_ERROR`` in C with
``rgmin_last_error`` text). A distance-constraint failure returns
``Error::Constraint`` in Rust and ``RGMIN_CONSTRAINT_ERROR`` in C, with
``rgmin_last_constraint_error`` naming the reason.
This differs from GROMACS ``do_steep`` in two places by design: a
non-finite trial is an immediate oracle error here, where GROMACS
refuses and shrinks, and a constraint-failure run returns
``Error::Constraint`` once the shrink budget is spent, where GROMACS
ends the minimization loop gracefully once ``ustep < 1e-12``.

Session hooks
~~~~~~~~~~~~~

The core state contains ``h`` and the 1.2 / ``shrink`` update. The session
projects rigid motion from the force before normalization, applies an
explicit per-atom or Euclidean ``maxmove`` cap to the proposed displacement,
and uses the ordinary manifold, coordinate-bound, and holonomic-distance
constraint placement. A constraint the trial cannot meet is a refused
step: the old point is kept and ``h`` is multiplied by ``shrink``, the
same branch GROMACS takes by setting the trial energy to +infinity
(``minimize.cpp`` lines 3008-3009 and 3104).
``GromacsSteepConfig::max_constraint_shrinks`` bounds that run. The
session derives its default from the actual ``emstep`` and shrink factor:
the number of shrinks until ``h`` reaches floating-point epsilon
(``h * shrink^k <= eps``). At the canonical ``emstep = 0.01`` and
``shrink = 0.5``, that gives 46 in f64 arithmetic. The next failure
returns the constraint error. Zero restores an error on the first
failure and does not shrink. A placement that meets the constraints
restarts the count. The trial energy is tested only after those hooks,
so an accepted step and its reported gradient describe the point the host
receives.

There is no implicit rgmin 0.2 per-atom cap: ``h`` is already the GROMACS
maximum displacement. An explicit cap still applies.

GROMACS-unit mapping
~~~~~~~~~~~~~~~~~~~~

The Python ``gromacs_mdp_options`` layer converts:

.. table::

    +------------+----------------------------------------+---------------------------------+
    | MDP key    | default                                | native option                   |
    +============+========================================+=================================+
    | ``emtol``  | 10.0 kJ mol\ :sup:`-1`\ nm\ :sup:`-1`\ | ``gtol`` in eV Å\ :sup:`-1`\    |
    +------------+----------------------------------------+---------------------------------+
    | ``emstep`` | 0.01 nm                                | ``istep`` in Å                  |
    +------------+----------------------------------------+---------------------------------+
    | ``nsteps`` | -1                                     | no limit; otherwise ``maxiter`` |
    +------------+----------------------------------------+---------------------------------+

The Rust stepper, C ABI, and C++ wrapper do no unit conversion. C and
C++ expose both ``shrink`` and ``max_constraint_shrinks`` through
``rgmin_solver_set_gromacs_steep`` / ``Solver::set_gromacs_steep``.
``RGMIN_GROMACS_STEEP`` and ``RGMIN_STOP_MACHINE_PRECISION`` belong to
ABI minor 50, the minor named by ``include/rgmin.h``.
