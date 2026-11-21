===============================
Run GROMACS L-BFGS minimization
===============================

    :Author: rgmin developers

.. contents::


1 Run GROMACS L-BFGS minimization
---------------------------------

Use ``gromacs-lbfgs`` to carry a GROMACS ``integrator = l-bfgs``
energy-minimization block into rgmin. The defaults are ``emtol = 10.0``
kJ mol\ :sup:`-1`\ nm\ :sup:`-1`\, ``emstep = 0.01`` nm, ``nsteps = -1``, ``nbfgscorr = 10``,
and ``constraints = none``.

1.1 Python and MDP keys
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from rgmin.optimize import GromacsLbfgs

    opt = GromacsLbfgs(
        atoms,
        emtol=10.0,
        emstep=0.01,
        nsteps=500,
        nbfgscorr=10,
    )
    opt.run()

For the functional API, map the MDP keys first:

.. code:: python

    options = rgmin.gromacs_mdp_options(
        {
            "integrator": "l-bfgs",
            "emtol": 10.0,
            "emstep": 0.01,
            "nsteps": 500,
            "nbfgscorr": 10,
            "constraints": "none",
        }
    )
    method = options.pop("method")
    report, x = rgmin.minimize(value_and_gradient, x0, method, **options)

Unknown keys raise ``ValueError`` and name the key. A non-``none``
``constraints`` value is refused, as it is by GROMACS L-BFGS.

1.2 Rust, C, and C++
~~~~~~~~~~~~~~~~~~~~

The Rust core is unit-agnostic:

.. code:: rust

    let control = rgmin::Control::default()
        .with_gtol(force_tolerance)
        .with_istep(emstep)
        .with_maxiter(nsteps);
    let mut solver =
        rgmin::Solver::new(rgmin::Method::GromacsLbfgs, control, x.len());
    solver.set_gromacs_lbfgs_config(
        rgmin::GromacsLbfgsConfig::default().with_nbfgscorr(10),
    )?;
    let report = solver.run(&objective, &mut x)?;

The C selector is ``RGMIN_GROMACS_LBFGS`` and
``rgmin_solver_set_gromacs_lbfgs`` sets the correction count. C++ uses
``rgmin::Method::GromacsLbfgs`` and ``Solver::set_gromacs_lbfgs``.

The exact line search, restart, stopping rules, and documented
differences from GROMACS 2025.4 are in
:doc:`../reference/gromacs-lbfgs`.
