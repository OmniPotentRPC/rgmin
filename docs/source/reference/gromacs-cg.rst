

GROMACS-style conjugate gradient
--------------------------------

``Method::GromacsCg`` pairs the GROMACS-style Polak-Ribière
conjugate-gradient direction update from GROMACS 2025.4
`do\_cg <https://github.com/gromacs/gromacs/blob/v2025.4/src/gromacs/mdrun/minimize.cpp#L1313>`_
(``src/gromacs/mdrun/minimize.cpp``, lines 1313-2035) with ``rgmin``'s
strong-Wolfe line search (``c1 = 1e-4``, ``c2 = 0.1``) rather than
GROMACS's own golden-section/cubic-secant line search:



.. math::

    \mathbf{p}_k = -\mathbf{g}_k + \beta_k \mathbf{p}_{k-1}, \qquad
    \beta_k = \frac{\mathbf{g}_k \cdot (\mathbf{g}_k - \mathbf{g}_{k-1})}{\mathbf{g}_{k-1} \cdot \mathbf{g}_{k-1}},

where :math:`\mathbf{g}_k = -\mathbf{F}_k` is the potential-energy gradient.

Because the line search differs from ``do_cg``, ``Method::GromacsCg`` is not
a step-for-step clone of ``gmx mdrun`` under ``integrator = cg``. What matches
``do_cg`` and what is ``rgmin``-specific is listed below.

Direction update and strong-Wolfe line search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than duplicating ``do_cg``'s bracket/secant/bisection search loop,
``Method::GromacsCg`` runs on top of ``rgmin``'s existing ``Inner::Nlcg``
(``Conjugacy::PolakRibiere``) and ``LineSearch::Wolfe`` infrastructure
(``take_session_step``), adding the direction and acceptance rules from
``do_cg``:

1. **Periodic steepest-descent restart (``nstcgsteep``)**: ``do_cg`` counts its
   iterations with ``for (step = 0; ...; step++)`` and, once iteration
   ``step`` has moved, sets :math:`\beta = 0` for the next direction when
   ``nstcg && ((step % nstcg) == 0)`` (``minimize.cpp`` lines 1497 and
   1868-1871). ``GromacsCg::step`` is that counter and
   ``GromacsCg::filter_beta`` applies the test before ``GromacsCg::advance``,
   so the steepest-descent directions are those of iterations 0, 1,
   :math:`\text{nstcgsteep} + 1`, :math:`2\,\text{nstcgsteep} + 1`, and so on: the
   second step is always steepest descent. As in ``do_cg``, a restart for a
   positive :math:`\mathbf{g} \cdot \mathbf{p}` (``gpa > 0``, which undoes its count
   with ``step--``) keeps the index, while the retry of a stalled conjugate
   step spends one. ``nstcgsteep = 0`` turns the periodic restart off.
   ``Solver::forget`` sets the counter back to zero, so a forgotten session
   restarts like a fresh one. The default is
   ``GROMACS_CG_NSTCGSTEEP_DEFAULT = 1000``
   (`MDP nstcgsteep <https://manual.gromacs.org/current/user-guide/mdp-options.html#mdp-nstcgsteep>`_).

2. **Polak-Ribière multiplier cap**: ``do_cg`` suppresses the conjugate term
   when :math:`|\beta_k| > 5.0` (``minimize.cpp`` lines 1883-1887:
   ``if (std::fabs(beta) > 5.0) { beta = 0.0; }``). ``GromacsCg::filter_beta``
   enforces the same ``GROMACS_CG_BETA_CAP = 5.0`` bound on the raw
   Polak-Ribière ratio from ``Conjugacy::PolakRibiere``.

3. **Unit search direction and opening step scale**: before entering the line
   search, the search direction is normalized to unit Euclidean length via
   overflow-safe scaling. On the first step, the opening trial step is
   ``Control::istep`` (``emstep``, default ``GROMACS_CG_EMSTEP_DEFAULT = 0.01``);
   subsequent steps scale the previous accepted step length through
   rgmin's Nocedal-Wright eq. 3.60 opening estimate (``nlcg_opening_alpha``).
   The session defaults to strong Wolfe with :math:`c_1 = 10^{-4}` and the
   conjugate-gradient curvature parameter :math:`c_2 = 0.1` so cubic Hermite
   zoom locates the 1D directional stationary point before the next
   Polak-Ribière update. Callers can override the line search via
   ``Solver::set_linesearch`` or ``minimize_method``.

4. **Steepest-descent fallback on a stalled conjugate direction**: if a
   line search along a conjugate direction (:math:`\beta_{k-1} \neq 0`) fails to
   decrease the energy or rounds away all coordinates, the session clears
   the CG memory and retries the step along the steepest-descent direction
   :math:`-\mathbf{g}_k`, matching the ``beta = 0.0; continue;`` branch of ``do_cg``
   (``minimize.cpp`` lines 1814-1832). If a steepest-descent step still cannot change any
   coordinate or decrease the energy, the session stops with
   ``Status::MachinePrecision`` (``RGMIN_STOP_MACHINE_PRECISION``).

5. **Downhill energy-rise tolerance near machine precision**: ``do_cg``
   accepts a trial whose energy rises slightly below
   :math:`\sqrt{\varepsilon_{\text{mach}}}\,|E_0|` when the directional
   derivative remains negative (``minimize.cpp`` lines 1665-1672:
   ``tmp = std::sqrt(GMX_REAL_EPS) * std::fabs(s_a->epot); if (s_c->epot < s_a->epot || (gpc < 0 && s_c->epot < (s_a->epot + tmp)))``).
   ``Method::GromacsCg`` enables this same tolerance in ``LineSearchOptions``
   and ``take_step_with_options``.

6. **Max-atom force norm (``Fmax``)**: ``Report::stop_norm`` defaults to
   ``StopNorm::MaxAtom`` (:math:`\max_i \|\mathbf{F}_i\|_2` over consecutive xyz
   atom triples), matching ``gmx_f_norm_max`` in ``minimize.cpp`` and
   ``Method::GromacsSteep``. There is no implicit 0.2 per-atom displacement
   cap unless explicitly requested.

7. **Step counting (``nsteps`` / ``maxiter``)**: ``rgmin`` takes ``Control::maxiter``
   steps (``nsteps = 0`` evaluates only the initial point; Python/MDP
   ``nsteps = -1`` maps to unlimited ``sys.maxsize``). By contrast, upstream
   GROMACS ``do_cg`` loops ``for (step = 0; (number_steps < 0 || step <= number_steps) && !converged; step++)``
   (``minimize.cpp`` line 1497), executing ``nsteps + 1`` iterations on a
   budget stop (13 steps for ``nsteps = 12`` in the committed ``mdrun.txt``)
   and decrementing ``step`` by one when converged.

Refusal of holonomic constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GROMACS 2025.4 runs ``do_cg`` with LINCS constraints applied to each trial
configuration; ``gmx grompp`` refuses only SHAKE with ``integrator = cg``
(`grompp.cpp <https://github.com/gromacs/gromacs/blob/v2025.4/src/gromacs/gmxpreprocess/grompp.cpp#L2371>`_)
and suggests ``lincs-order`` of 8 or more
(`readir.cpp <https://github.com/gromacs/gromacs/blob/v2025.4/src/gromacs/gmxpreprocess/readir.cpp#L5365>`_).
``gromacs-cg`` does not implement that constrained line search: projecting
coordinates back onto a nonlinear constraint manifold after a step breaks
the conjugacy of consecutive search directions that the rest of the
session relies on. Use ``gromacs-steep`` for constrained systems.

``rgmin`` refuses holonomic constraints on ``gromacs-cg`` across every layer:

- Rust / C / C++: ``Solver::set_constraints`` (``rgmin_solver_set_constraints``)
  and stepping with installed ``DistanceConstraint`` pairs return
  ``Error::Constraint(ConstraintError::Unsupported)`` (``RGMIN_CONSTRAINT_ERROR``).

- Python ``gromacs_mdp_options``: any ``constraints`` value other than ``"none"``
  raises ``ValueError``.

- Python ``GromacsCg`` ASE optimizer: holonomic geometry constraints on
  ``atoms`` (such as ``FixBondLength`` / ``FixBondLengths``) or a non-``"none"``
  ``constraints`` keyword raise ``ValueError``. Linear coordinate subspace
  masks (``FixAtoms``, ``FixCartesian``, ``FixedLine``, ``FixedPlane``, analogous
  to GROMACS ``freezegrps`` / ``freezedim``) remain supported.

Stops and reports
~~~~~~~~~~~~~~~~~

.. table::

    +-----------------------------------------------------------------------------+-----------------------+----------------------------------------------------------------+
    | condition                                                                   | ``Status``            | C ABI                                                          |
    +=============================================================================+=======================+================================================================+
    | largest per-atom force is below ``Control::gtol`` (``emtol``)               | ``Tolerance``         | ``RGMIN_STOP_TOLERANCE``                                       |
    +-----------------------------------------------------------------------------+-----------------------+----------------------------------------------------------------+
    | ``Control::maxiter`` (``nsteps``) steps were taken                          | ``Budget``            | ``RGMIN_STOP_BUDGET``                                          |
    +-----------------------------------------------------------------------------+-----------------------+----------------------------------------------------------------+
    | a steepest-descent step cannot change any coordinate or decrease the energy | ``MachinePrecision``  | ``RGMIN_STOP_MACHINE_PRECISION``                               |
    +-----------------------------------------------------------------------------+-----------------------+----------------------------------------------------------------+
    | callback returned a non-finite value or gradient                            | ``Error::Oracle``     | ``RGMIN_INTERNAL_ERROR`` (+ ``rgmin_last_error``)              |
    +-----------------------------------------------------------------------------+-----------------------+----------------------------------------------------------------+
    | holonomic distance constraints were attached                                | ``Error::Constraint`` | ``RGMIN_CONSTRAINT_ERROR`` (+ ``rgmin_last_constraint_error``) |
    +-----------------------------------------------------------------------------+-----------------------+----------------------------------------------------------------+

GROMACS-unit mapping
~~~~~~~~~~~~~~~~~~~~

The Python ``gromacs_mdp_options`` helper converts:

.. table::

    +-----------------+-----------------------------------+----------------------------------------------------------------------+
    | MDP key         | default                           | native option                                                        |
    +=================+===================================+======================================================================+
    | ``integrator``  | ``"steep"`` (``"cg"`` selects CG) | ``method = "gromacs-cg"``                                            |
    +-----------------+-----------------------------------+----------------------------------------------------------------------+
    | ``emtol``       | 10.0 kJ mol^-1 nm^-1              | ``gtol`` in eV Å^-1                                                  |
    +-----------------+-----------------------------------+----------------------------------------------------------------------+
    | ``emstep``      | 0.01 nm                           | ``istep`` in Å                                                       |
    +-----------------+-----------------------------------+----------------------------------------------------------------------+
    | ``nsteps``      | -1                                | no limit; otherwise ``maxiter``                                      |
    +-----------------+-----------------------------------+----------------------------------------------------------------------+
    | ``nstcgsteep``  | 1000                              | ``nstcgsteep`` (:math:`\ge 0`, 0 turns the restart off; ``cg`` only) |
    +-----------------+-----------------------------------+----------------------------------------------------------------------+
    | ``constraints`` | ``"none"``                        | validated; non-``"none"`` raises ``ValueError`` on ``cg``            |
    +-----------------+-----------------------------------+----------------------------------------------------------------------+

The Rust stepper, C ABI, and C++ wrapper do no unit conversion. C and C++
expose ``nstcgsteep`` through ``rgmin_solver_set_gromacs_cg`` /
``Solver::set_gromacs_cg``; ``RGMIN_GROMACS_CG`` and
``rgmin_solver_set_gromacs_cg`` are declared in ``include/rgmin.h``.

Comparison against ``gmx mdrun`` on the 7-atom argon cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 7-atom Lennard-Jones argon cluster fixture
(``tests/data/gromacs_steep_argon.txt``, :math:`\sigma = 0.34\text{ nm}`,
:math:`\epsilon = 1.0\text{ kJ mol}^{-1}`, :math:`r_c = 3.0\text{ nm}`,
:math:`\text{emstep} = 0.02\text{ nm}`, :math:`\text{nstcgsteep} = 1000`) was run with
both double-precision ``gmx_d mdrun`` and single-precision ``gmx mdrun``
(GROMACS 2023.3 built from source with GCC 13.3.0 and AVX2\_256 SIMD,
``-nt 1 -ntmpi 1``) under ``integrator = cg`` and compared against ``rgmin``
(``Method::GromacsCg``). The complete inputs, terminal output, ``em.log`` files,
and machine-readable summary are committed under
``tests/data/gromacs_cg_reference/``.

.. table::

    +-----------------------------------+-------------------------+-------+-------------+------------------------------------------+------------------------------------------+
    | run / tolerance                   | solver                  | steps | force evals | final :math:`E_{\text{pot}}` (kJ mol^-1) | final :math:`F_{\max}` (kJ mol^-1 nm^-1) |
    +===================================+=========================+=======+=============+==========================================+==========================================+
    | ``nsteps = 12``, ``emtol = 1e-8`` | ``gmx_d mdrun`` (f64)   |    12 |          16 | ``-1.25489376279604e+01``                | ``3.54506056338870e-03``                 |
    +-----------------------------------+-------------------------+-------+-------------+------------------------------------------+------------------------------------------+
    | ``nsteps = 12``, ``emtol = 1e-8`` | ``gmx mdrun`` (f32)     |    11 |          25 | ``-1.2548939e+01``                       | ``4.3311492e-03``                        |
    +-----------------------------------+-------------------------+-------+-------------+------------------------------------------+------------------------------------------+
    | ``nsteps = 12``, ``emtol = 1e-8`` | ``rgmin`` ``GromacsCg`` |    12 |          60 | ``-1.25489377088303e+01``                | ``4.735427e-07``                         |
    +-----------------------------------+-------------------------+-------+-------------+------------------------------------------+------------------------------------------+
    | ``emtol = 1e-2``                  | ``gmx_d mdrun`` (f64)   |     9 |          12 | ``-1.25489364597334e+01``                | ``5.54402580304420e-03``                 |
    +-----------------------------------+-------------------------+-------+-------------+------------------------------------------+------------------------------------------+
    | ``emtol = 1e-2``                  | ``rgmin`` ``GromacsCg`` |     6 |          18 | ``-1.25489374174275e+01``                | ``8.694525e-03``                         |
    +-----------------------------------+-------------------------+-------+-------------+------------------------------------------+------------------------------------------+
    | ``emtol = 1e-3``                  | ``gmx_d mdrun`` (f64)   |    13 |          18 | ``-1.25489376639539e+01``                | ``4.98089785018483e-04``                 |
    +-----------------------------------+-------------------------+-------+-------------+------------------------------------------+------------------------------------------+
    | ``emtol = 1e-3``                  | ``rgmin`` ``GromacsCg`` |     7 |          22 | ``-1.25489377085847e+01``                | ``1.529776e-04``                         |
    +-----------------------------------+-------------------------+-------+-------------+------------------------------------------+------------------------------------------+
    | ``emtol = 1e-6``                  | ``gmx_d mdrun`` (f64)   |    57 |          72 | ``-1.25489377088303e+01``                | ``3.48102157458076e-07``                 |
    +-----------------------------------+-------------------------+-------+-------------+------------------------------------------+------------------------------------------+
    | ``emtol = 1e-6``                  | ``rgmin`` ``GromacsCg`` |     9 |          33 | ``-1.25489377088303e+01``                | ``7.015111e-07``                         |
    +-----------------------------------+-------------------------+-------+-------------+------------------------------------------+------------------------------------------+

Both solvers take steepest-descent steps at iterations 0 and 1 and
conjugate steps after that. Because ``rgmin`` uses its strong-Wolfe
bracketing and cubic-zoom line search (:math:`c_1 = 10^{-4}, c_2 = 0.1`) rather
than ``do_cg``'s midpoint/cubic secant heuristic (and stops at ``nsteps``
steps rather than ``nsteps + 1``), step-by-step intermediate
iterates differ while
converging to the same potential-energy minimum
(:math:`-12.548937708830\text{ kJ mol}^{-1}`) on this fixture. Note also that in
the ``nsteps = 12`` run below, ``summary.tsv`` records ``12`` (the ``nsteps``
input / last step index ``Step 12``), while ``gmx_d mdrun`` actually executed
13 steps (``Step 0`` through ``Step 12``) before reporting ``in 13 steps``.
On this fixture, ``rgmin`` uses fewer outer iterations at each tested
tolerance. It uses more force evaluations at ``emtol = 1e-2`` and ``1e-3``,
but fewer at ``emtol = 1e-6``. At the 12-step cap it uses more force
evaluations and reaches a lower :math:`F_{\max}`. The single-precision
GROMACS run stops at machine precision after 11 steps. No GROMACS
trajectory equivalence beyond these tested final-energy and force-norm
comparisons is claimed.

Minimization section of the raw ``gmx_d mdrun -v`` output for
``integrator = cg``, ``nsteps = 12``, ``emtol = 1e-8``, ``emstep = 0.02``,
``nstcgsteep = 1000``:

::

    Polak-Ribiere Conjugate Gradients:
       Tolerance (Fmax)   =  1.00000e-08
       Number of steps    =           12
       F-max             =  7.58904e+00 on atom 3
       F-Norm            =  7.02609e+00

    Step 0, Epot=-1.253293e+01, Fnorm=8.103e-01, Fmax=8.753e-01 (atom 6)
    Step 1, Epot=-1.253484e+01, Fnorm=7.289e-01, Fmax=7.873e-01 (atom 2)
    Step 2, Epot=-1.253491e+01, Fnorm=8.501e-01, Fmax=9.183e-01 (atom 2)
    Step 3, Epot=-1.253735e+01, Fnorm=3.975e-01, Fmax=4.293e-01 (atom 6)
    Step 4, Epot=-1.253801e+01, Fnorm=3.227e-01, Fmax=3.485e-01 (atom 6)
    Step 5, Epot=-1.253901e+01, Fnorm=2.951e-01, Fmax=3.187e-01 (atom 5)
    Step 6, Epot=-1.254064e+01, Fnorm=2.856e-01, Fmax=3.085e-01 (atom 2)
    Step 7, Epot=-1.254246e+01, Fnorm=4.829e-01, Fmax=5.216e-01 (atom 2)
    Step 8, Epot=-1.254889e+01, Fnorm=1.090e-01, Fmax=1.177e-01 (atom 2)
    Step 9, Epot=-1.254894e+01, Fnorm=5.133e-03, Fmax=5.544e-03 (atom 2)
    Step 10, Epot=-1.254894e+01, Fnorm=1.983e-03, Fmax=2.141e-03 (atom 4)
    Step 11, Epot=-1.254894e+01, Fnorm=1.121e-03, Fmax=1.211e-03 (atom 4)
    Step 12, Epot=-1.254894e+01, Fnorm=3.282e-03, Fmax=3.545e-03 (atom 3)

    Energy minimization reached the maximum number of steps before the forces
    reached the requested precision Fmax < 1e-08.

    writing lowest energy coordinates.

    Polak-Ribiere Conjugate Gradients did not converge to Fmax < 1e-08 in 13 steps.
    Potential Energy  = -1.25489376279604e+01
    Maximum force     =  3.54506056338870e-03 on atom 3
    Norm of force     =  3.28204766414723e-03
