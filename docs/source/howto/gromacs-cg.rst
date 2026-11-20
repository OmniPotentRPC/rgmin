

Run GROMACS conjugate-gradient minimization
-------------------------------------------

Use the ``gromacs-cg`` method when a setup should follow GROMACS
``integrator = cg`` (Polak-Ribière conjugate gradient with a periodic
steepest-descent restart every ``nstcgsteep`` steps and convergence on the
largest per-atom force norm) alongside ``gromacs-steep``.

Python and MDP keys
~~~~~~~~~~~~~~~~~~~

``gromacs_mdp_options`` accepts ``integrator = "cg"``, ``emtol``, ``emstep``,
``nsteps``, ``nstcgsteep``, and ``constraints = "none"`` in GROMACS units and
converts the force tolerance and initial step to the eV/Å units used by
ASE calculators.

.. code:: python

    import numpy as np
    import rgmin

    options = rgmin.gromacs_mdp_options(
        {
            "integrator": "cg",
            "emtol": 10.0,
            "emstep": 0.01,
            "nsteps": 500,
            "nstcgsteep": 1000,
        }
    )
    method = options.pop("method")
    report, x = rgmin.minimize(value_and_gradient, np.asarray(x0), method, **options)

The defaults for ``integrator = "cg"`` are ``emtol = 10.0`` kJ mol\ :sup:`-1`\ nm\ :sup:`-1`\,
``emstep = 0.01`` nm, ``nsteps = -1`` (no iteration limit),
``nstcgsteep = 1000``, and ``constraints = "none"``. ``nstcgsteep = 0`` turns the
periodic steepest-descent restart off, as in GROMACS. Any non-``"none"``
value of ``constraints`` raises ``ValueError``: ``gromacs-cg`` has no constrained
line search, so use ``gromacs-steep`` for constrained systems.

For an ASE ``Atoms`` or cell filter, use ``GromacsCg`` in ``rgmin.optimize``:

.. code:: python

    from rgmin.optimize import GromacsCg

    opt = GromacsCg(atoms, emtol=10.0, emstep=0.01, nsteps=500, nstcgsteep=1000)
    opt.run()

An explicit ``fmax`` or ``steps`` passed to ``run`` overrides the mapped ``emtol``
or ``nsteps`` for that call. Holonomic bond/angle constraints on ``atoms`` are
refused with a ``ValueError``.

Rust, C, and C++
~~~~~~~~~~~~~~~~

The core is unit-agnostic. Put the force tolerance and initial step scale in
the units of the objective:

.. code:: rust

    let control = rgmin::Control::default()
        .with_gtol(force_tolerance)
        .with_istep(initial_step_scale)
        .with_maxiter(number_of_steps);
    let mut solver = rgmin::Solver::new(rgmin::Method::GromacsCg, control, x.len());
    solver.set_gromacs_cg_config(
        rgmin::GromacsCgConfig::default().with_nstcgsteep(1000),
    )?;
    let report = solver.run(&objective, &mut x)?;

The C selector is ``RGMIN_GROMACS_CG`` and the C++ selector is
``rgmin::Method::GromacsCg``. Configure ``nstcgsteep`` with
``rgmin_solver_set_gromacs_cg`` in C or ``Solver::set_gromacs_cg`` in C++.
Distance constraints (``Solver::set_distance_constraints`` /
``rgmin_solver_set_distance_constraints``) are refused on ``GromacsCg`` with a
constraint error.

The algorithm, line search, and ``gmx mdrun`` comparison are in
:doc:`../reference/gromacs-cg`.
