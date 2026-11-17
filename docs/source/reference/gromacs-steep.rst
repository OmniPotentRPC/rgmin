

GROMACS steepest descent
------------------------

``Method::GromacsSteep`` implements the update documented in the
`GROMACS reference manual, Energy Minimization: Steepest Descent <https://manual.gromacs.org/current/reference-manual/algorithms/energy-minimization.html#steepest-descent>`_:



.. math::

    r_{n+1} = r_n + h_n F_n / \max_i |F_{n,i}|.

The maximum is the largest Euclidean norm of one consecutive xyz atom
triple. A trailing partial triple is one atom. A strict energy decrease
accepts the trial and sets ``h <- 1.2 h``. An equal or larger energy rejects
the trial, keeps the old point and force, and sets ``h <- 0.2 h``.

The implementation also follows the state-swap structure and strict
energy comparison in GROMACS
`do\_steep in src/gromacs/mdrun/minimize.cpp <https://github.com/gromacs/gromacs/blob/main/src/gromacs/mdrun/minimize.cpp#L2881-L3178>`_.
That source currently multiplies a rejected displacement by 0.5, while
the reference-manual algorithm says 0.2. rgmin follows the documented
0.2 rule.

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

``Report::stop_norm`` is always ``StopNorm::MaxAtom`` for this method, even
when the input ``Control`` names another norm. Rejected trials count as
steps. ``Solver::gromacs_step_size`` returns the displacement that the next
trial will use.

Session hooks
~~~~~~~~~~~~~

The core state contains only ``h`` and the 1.2/0.2 update. The session
projects rigid motion from the force before normalization, applies an
explicit per-atom or Euclidean ``maxmove`` cap to the proposed displacement,
and uses the ordinary manifold, coordinate-bound, and holonomic-distance
constraint placement. The trial energy is tested only after those hooks,
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

The Rust stepper, C ABI, and C++ wrapper do no unit conversion.
ABI minor 48 carries ``RGMIN_GROMACS_STEEP`` and
``RGMIN_STOP_MACHINE_PRECISION``. Minor 47 is the level-set constraint
setters.
