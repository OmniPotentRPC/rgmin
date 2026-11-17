

LAMMPS min\ :sub:`style`\ rgmin
-------------------------------

This page lists the ``min_modify`` keywords of the LAMMPS plugin in ``lammps/``, the rgmin C ABI call behind each one, and the errors the plugin raises.
:doc:`../howto/lammps-migration` shows how to build and load it.

.. code:: text

    plugin load rgminplugin.so
    min_style rgmin
    min_modify keyword value ...
    minimize etol ftol maxiter maxeval

``min_modify`` applies to the current minimizer, so it comes after ``min_style rgmin``.
A new ``min_style`` command starts again from the defaults.

Keywords of the plugin
~~~~~~~~~~~~~~~~~~~~~~

.. table::

    +----------------+----------------------------------------------------+-------------+-----------------------------------------------------+
    | keyword        | values                                             | default     | C ABI                                               |
    +================+====================================================+=============+=====================================================+
    | ``method``     | ``lbfgs``, ``fire``, ``fire2``, ``sd``, ``bb``     | ``lbfgs``   | ``rgmin_method_from_name``, ``rgmin_solver_create`` |
    +----------------+----------------------------------------------------+-------------+-----------------------------------------------------+
    | ``memory``     | integer, at least 1                                | 10          | ``rgmin_control_t.memory``                          |
    +----------------+----------------------------------------------------+-------------+-----------------------------------------------------+
    | ``trust``      | radius, at least 0 (distance units)                | 0, off      | ``rgmin_solver_set_lbfgs_trust``                    |
    +----------------+----------------------------------------------------+-------------+-----------------------------------------------------+
    | ``linesearch`` | ``none``, ``wolfe``                                | ``none``    | ``rgmin_solver_set_lbfgs_wolfe``                    |
    +----------------+----------------------------------------------------+-------------+-----------------------------------------------------+
    | ``accept``     | ``default``, ``energy``, ``none``, ``nonmonotone`` | ``default`` | ``rgmin_solver_set_accept``                         |
    +----------------+----------------------------------------------------+-------------+-----------------------------------------------------+
    | ``istep``      | step scale, at least 0                             | 0           | ``rgmin_control_t.istep``                           |
    +----------------+----------------------------------------------------+-------------+-----------------------------------------------------+

- ``method`` takes rgmin's method names, so ``steepest`` and ``barzilai-borwein`` are ``sd`` and ``bb``.
  Any other rgmin method is an error.

- ``memory`` is the number of L-BFGS pairs.
  Other methods ignore it.

- ``trust`` is the Burdakov ray-truncation radius of L-BFGS, which replaces the per-atom cap and the acceptance test while it is on.
  It needs ``method lbfgs`` and excludes ``linesearch wolfe`` and ``fix box/relax``.

- ``linesearch wolfe`` needs ``method lbfgs``.
  With a positive ``dmax`` it is rgmin's geometry L-BFGS step, whose trials the per-atom cap bounds; with ``dmax 0`` it is the strong Wolfe search of Nocedal and Wright, algorithms 3.5 and 3.6.

- ``accept`` is how a session takes a proposed step.
  ``default`` keeps the C ABI default, ``energy``, which refuses a trial that raises the energy.

- ``istep`` is the FIRE time step of ``fire`` and ``fire2``, 0.1 when it is 0.
  For ``sd`` and ``bb``, 0 is the C ABI default: a unit step scale, with ``min(1, 1/|g|)`` on the first step.

LAMMPS keywords the plugin reads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. table::

    +-------------+---------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | keyword     | default | use                                                                                                                                              |
    +=============+=========+==================================================================================================================================================+
    | ``dmax``    | 0.1     | ``rgmin_solver_set_atom_maxmove``: the largest displacement of one atom in one step; 0 leaves the cap off and is an error with ``fix box/relax`` |
    +-------------+---------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``norm``    | ``two`` | ``rgmin_solver_set_stop_norm``: ``two`` is ``TWO``, ``max`` is ``MAX_ATOM``, ``inf`` is ``INF``                                                  |
    +-------------+---------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``abcfire`` | ``no``  | ``rgmin_solver_set_fire_abc`` on a ``fire2`` session                                                                                             |
    +-------------+---------+--------------------------------------------------------------------------------------------------------------------------------------------------+

``method fire2`` selects ``RGMIN_FIRE_PRESET_GUENOLE2020``, the parameter set of the LAMMPS ``min_style fire`` defaults.
``delaystep``, ``dtgrow``, ``dtshrink``, ``alpha0``, ``alphashrink``, ``tmax``, ``tmin``, ``integrator``, ``halfstepback``, ``initialdelay`` and ``vdfmax`` must keep those defaults, and ``line`` must stay ``quadratic``.
The ``minimize`` command checks them and stops with an error otherwise.

Stopping rules
~~~~~~~~~~~~~~

.. table::

    +-------------+---------------------------------------------------------------------------------------------------------+
    | LAMMPS      | plugin                                                                                                  |
    +=============+=========================================================================================================+
    | ``etol``    | ``abs(E_k - E_{k-1}) < etol (abs(E_k) + abs(E_{k-1}) + 1e-8) / 2`` after a step that moved the atoms    |
    +-------------+---------------------------------------------------------------------------------------------------------+
    | ``ftol``    | ``rgmin_report_t.grad_norm < ftol``, with ``rgmin_control_t.gtol = ftol`` and the stop norm of ``norm`` |
    +-------------+---------------------------------------------------------------------------------------------------------+
    | ``maxiter`` | number of ``rgmin_solver_step_fg`` calls                                                                |
    +-------------+---------------------------------------------------------------------------------------------------------+
    | ``maxeval`` | number of ``Min::energy_force`` calls the plugin made, checked after each step                          |
    +-------------+---------------------------------------------------------------------------------------------------------+

``E`` is the energy LAMMPS prints, per atom under ``thermo_modify norm yes``.
The gradient rgmin sees is ``-F`` over all ``3N`` coordinates after every fix, so ``grad_norm`` is the force norm that ``min_style cg`` compares with ``ftol``.
With ``fix box/relax`` it also holds the fix forces, as below.
A failed rgmin step ends the minimization with the stopping criterion ``rgmin step failed:`` and the text of ``rgmin_last_error``.

Box relaxation
~~~~~~~~~~~~~~

``fix box/relax`` and other fixes that add global degrees of freedom work with every method except ``fire`` and ``fire2``.

.. code:: text

    fix relax all box/relax tri 0.0 vmax 0.001
    min_style rgmin
    minimize 0.0 1.0e-8 10000 100000

.. table::

    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | LAMMPS                                   | plugin                                                                                      |
    +==========================================+=============================================================================================+
    | ``nextra_global`` box coordinates ``ds`` | appended to the ``3N`` atom coordinates: 1 for ``iso``, 3 for ``aniso``, 6 for ``tri``      |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``fextra``                               | ``-fextra`` is the gradient of the box coordinates                                          |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``Modify::min_store``                    | called once, when the session starts; ``ds`` is measured from that box                      |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``Modify::min_step``                     | each evaluation calls ``min_step(0.0)``, places the atoms, then calls ``min_step(1.0, ds)`` |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``Modify::max_alpha``                    | the cap of each box coordinate in one step, through ``rgmin_solver_set_cell_block``         |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``Modify::min_reset_ref``                | called after each step; a reset evaluates the iterate again                                 |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``Modify::min_energy``                   | part of the energy, as in every LAMMPS minimizer                                            |
    +------------------------------------------+---------------------------------------------------------------------------------------------+

- The atom coordinates are the positions in the box of the session start.
  ``min_step(1.0, ds)`` maps them into the current box.

- ``fix box/relax`` allows a box coordinate to move by ``vmax`` in one step for ``iso``, and by ``min(1, vmax)`` otherwise.
  ``dmax`` still bounds each atom.
  A step that exceeds either bound is scaled down as a whole.

- ``ftol`` uses the norm over the ``3N`` forces and the box forces, as in ``min_style cg``.
  ``norm two`` and ``norm inf`` match LAMMPS.
  ``norm max`` takes the largest per-atom force and the magnitude of each box force.
  LAMMPS ``Min::fnorm_max`` reads only ``fextra[0]`` and ``fextra[3]``, so the plugin's ``max`` norm is never smaller and never stops earlier.

- The gradient of an atom is ``-F``, as ``min_style cg`` uses it.
  The exact gradient in the coordinates of the start box would differ by the box strain; both vanish at the same minimum.

- The log line ``rgmin: K box coordinates from fixes, box step at most C`` reports the box degrees of freedom and their cap.

Errors
~~~~~~

.. table::

    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | message contains                                                      | cause                                                                                        |
    +=======================================================================+==============================================================================================+
    | ``Unknown min_modify keyword X for min_style rgmin``                  | a keyword that neither LAMMPS nor the plugin defines                                         |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``Unknown rgmin method X``                                            | a name ``rgmin_method_from_name`` refuses                                                    |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``stores a dense 3N x 3N matrix``                                     | ``bfgs``, ``sr1`` or ``sr2``                                                                 |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``needs a dense Hessian``                                             | ``newton``, ``rfo`` or ``dogleg``                                                            |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``is not available with min_style rgmin``                             | any other rgmin method                                                                       |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``cannot relax the box of fix box/relax with method X``               | ``fire`` or ``fire2`` with ``fix box/relax``; the message names ``lbfgs``, ``sd`` and ``bb`` |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``min_modify trust does not bound the box step of fix box/relax``     | ``trust`` above 0 with ``fix box/relax``                                                     |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``needs min_modify dmax > 0 with fix box/relax``                      | ``dmax 0`` with ``fix box/relax``                                                            |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``does not support pair styles that add per-atom degrees of freedom`` | for example ``pair_style eff/cut``                                                           |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``requires consecutive atom IDs 1 to N``                              | gaps in the atom IDs                                                                         |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``does not take min_modify line``                                     | ``min_modify line`` other than ``quadratic``                                                 |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``keeps the Guenole 2020 FIRE parameters``                            | a LAMMPS FIRE keyword changed from its default                                               |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | ``replicated rgmin sessions diverged across ranks``                   | ranks that reached different step energies                                                   |
    +-----------------------------------------------------------------------+----------------------------------------------------------------------------------------------+

Keywords with a bad value (``memory 0``, ``linesearch armijo``, ``accept maybe``) and settings that need another method (``trust`` or ``linesearch wolfe`` without ``lbfgs``, ``abcfire yes`` without ``fire2``) also stop with an error that names them.
