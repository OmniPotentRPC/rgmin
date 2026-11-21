==============
GROMACS L-BFGS
==============

    :Author: rgmin developers

.. contents::


1 GROMACS L-BFGS
----------------

``Method::GromacsLbfgs`` follows ``do_lbfgs`` in the tagged GROMACS 2025.4
source,
`minimize.cpp, lines 2036--2858 <https://github.com/gromacs/gromacs/blob/v2025.4/src/gromacs/mdrun/minimize.cpp#L2036-L2858>`_
(``src/gromacs/mdrun/minimize.cpp``). The MDP defaults come from the same
tag's MDP option documentation for
`nsteps <https://github.com/gromacs/gromacs/blob/v2025.4/docs/user-guide/mdp-options.rst#L206-L210>`_,
`emtol and emstep <https://github.com/gromacs/gromacs/blob/v2025.4/docs/user-guide/mdp-options.rst#L341-L350>`_, and
`nbfgscorr <https://github.com/gromacs/gromacs/blob/v2025.4/docs/user-guide/mdp-options.rst#L358-L362>`_.

1.1 Algorithm
~~~~~~~~~~~~~

The first direction is the force, :math:`\mathbf p_0=-\mathbf g_0`, and the
opening scalar step is :math:`1/\|\mathbf F_0\|_2`. Before evaluation, GROMACS
repeatedly multiplies it by 0.1 until the largest positive coordinate
displacement is no greater than ``emstep``. Although the MDP manual calls
``emstep`` the "initial step-size", ``do_lbfgs`` uses it as this displacement
ceiling (lines 2409--2435); rgmin follows the source.

An opening point :math:`C` is accepted immediately when its energy is lower
than point :math:`A`, or when its directional derivative is still negative
and its energy rise is below
:math:`\sqrt{\epsilon}\lvert E_A\rvert`. Otherwise the search brackets between
:math:`A` and :math:`C`. It interpolates the derivative zero when the endpoint
derivatives have opposite signs, bisects otherwise, and stops at the
first point below both endpoints. The search makes at most 20 bracket
trials (lines 2461--2618).

For an accepted displacement :math:`\mathbf s` and gradient change
:math:`\mathbf y`, rgmin keeps at most ``nbfgscorr`` pairs and uses the same
two-loop recursion and scalar initial inverse Hessian
:math:`H_0=(\mathbf s^T\mathbf y)/(\mathbf y^T\mathbf y) I` as lines
2620--2724. The default correction count is 10.

If the line search cannot resolve a lower energy, nonempty correction
memory is cleared and the next direction is the force. Failure with
empty memory stops at machine precision (lines 2571--2595). Every
accepted step resets the opening scalar step to 1 (line 2786).
Calling ``Solver::rebase``, ``Solver::forget_evaluation``, or supplying
externally moved coordinates retains the correction pairs and rebuilds
the two-loop direction at the new point with opening step
:math:`1/\|\mathbf F\|_2`.

1.2 Stops and reporting
~~~~~~~~~~~~~~~~~~~~~~~

``emtol`` is the largest per-atom force norm, so the default stop norm is
``StopNorm::MaxAtom``. ``nsteps`` maps to ``Control::maxiter`` and ``-1`` maps
to no Python iteration limit. Reports expose outer iterations and
all objective/force evaluations, including line-search trials.

.. table::

    +------------------------------------------------------------------------------------------------------------+----------------------+
    | condition                                                                                                  | status               |
    +============================================================================================================+======================+
    | :math:`F_{\max}<\texttt{emtol}`                                                                            | ``Tolerance``        |
    +------------------------------------------------------------------------------------------------------------+----------------------+
    | ``nsteps`` exhausted                                                                                       | ``Budget``           |
    +------------------------------------------------------------------------------------------------------------+----------------------+
    | line step below the relative-coordinate precision threshold, or an empty-memory search cannot lower energy | ``MachinePrecision`` |
    +------------------------------------------------------------------------------------------------------------+----------------------+
    | callback produces a non-finite value or gradient                                                           | ``Error::Oracle``    |
    +------------------------------------------------------------------------------------------------------------+----------------------+

1.3 Exactness and differences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an unconstrained Euclidean objective without active coordinate
bounds, the correction recursion, displacement check, line search,
reset, and force stop reproduce GROMACS 2025.4 ``do_lbfgs``. The following
differences are deliberate and documented:

- GROMACS refuses L-BFGS with domain decomposition, multiple ranks, or
  constraints (lines 2051--2066). rgmin has no rank/domain-decomposition
  state and refuses holonomic distance constraints, while frozen linear
  coordinate masks remain available.

- rgmin uses binary64 ``f64::EPSILON``. A single-precision GROMACS build
  uses its larger ``GMX_REAL_EPS``.

- GROMACS uses 1 nm as the characteristic coordinate in its
  machine-precision step test (lines 2350--2368). The unit-agnostic Rust
  API uses 1 in the objective's coordinate unit. Consequently the Python
  Å mapping can stop at a slightly different last representable step.

- rgmin checks ``emtol`` at the point supplied to a session before taking
  another step. ``do_lbfgs`` checks it after an accepted step.

- The existing Python MDP helper defaults ``nsteps`` to ``-1`` (unlimited);
  the GROMACS 2025.4 MDP default is 0. An explicit ``nsteps`` value maps
  directly to the rgmin iteration budget. GROMACS ``do_lbfgs`` uses an
  inclusive ``step <= nsteps`` loop and can attempt one additional outer
  iteration; rgmin keeps the exact-budget convention shared by its
  other GROMACS front ends.

- ``do_lbfgs`` divides by curvature without validating it. rgmin returns
  ``Error::GromacsLbfgs`` for zero or non-finite curvature instead of
  propagating a non-finite direction.

- Active objective bounds or non-Euclidean retractions alter trial
  coordinates, so their iterates are extensions rather than an exact
  GROMACS trajectory.

The C ABI selector is ``RGMIN_GROMACS_LBFGS``. ABI minor 54 adds it and
``rgmin_solver_set_gromacs_lbfgs``.

1.4 GROMACS comparison
~~~~~~~~~~~~~~~~~~~~~~

The seven-atom argon fixture in ``tests/data/gromacs_steep_argon.*`` was
run for ``nsteps = 12``, ``emtol = 1e-8``, ``emstep = 0.02`` and
``nbfgscorr = 10``:

.. table::

    +-----------------------------------------------+---------------------------------------------------+---------------------+-------------------+---------------------+---------------------+
    | solver                                        | reported steps                                    | accepted step lines | force evaluations |        final energy |          final Fmax |
    +===============================================+===================================================+=====================+===================+=====================+=====================+
    | ``gmx mdrun`` 2023.3 Ubuntu, single precision | -1 (the upstream machine-precision reporting bug) |                  12 |                13 |        -1.2548940e1 |        2.0878734e-4 |
    +-----------------------------------------------+---------------------------------------------------+---------------------+-------------------+---------------------+---------------------+
    | rgmin ``GromacsLbfgs``                        | 12                                                |                  12 |                14 | -1.25489377088264e1 | 2.73745712994804e-5 |
    +-----------------------------------------------+---------------------------------------------------+---------------------+-------------------+---------------------+---------------------+

The compared ``gmx`` build is single precision, so its ``GMX_REAL_EPS``
line-search and machine-precision decisions differ from rgmin's
binary64 decisions.
