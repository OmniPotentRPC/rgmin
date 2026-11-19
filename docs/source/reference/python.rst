

Module ``rgmin``
----------------

``Method`` is an ``enum.IntEnum`` of the C method ids. Members are
UPPER\ :sub:`SNAKE`\, for example ``LBFGS`` and ``POLAK_RIBIERE``. The older
CamelCase spellings are aliases of the same values.
``Stop`` is why a step ended. Its members are ``TOLERANCE``,
``NOISE_FLOOR``, ``STEP_TOLERANCE``, ``BUDGET``, ``LINE_SEARCH``,
``POLISH_UNMET``, ``ENERGY_TOLERANCE``, ``EVAL_BUDGET``,
``MACHINE_PRECISION``, and ``LINE_SEARCH_ROUNDING``, again with CamelCase
aliases.
``Report`` has ``value``, ``steps``, ``grad_norm``, and ``stop``.
``Report.converged`` is true when ``stop`` is ``TOLERANCE``.
``backward_error_floor(value, x)`` is the first-order stationarity
floor ``n * eps * (1 + |value| + sum |x_i|)``.
``NumericalError`` is an ``ArithmeticError``.
``NotConverged`` is a ``RuntimeError`` with a ``report`` attribute holding
the terminal ``Report``.
``SolverConfig(method``"lbfgs", \*\*options)= validates options.
``Solver(n, config)`` is the session. ``step(f_and_grad, x)`` takes one
iteration. ``step_hess(f_and_grad, hessian, x)`` is the same for Newton,
RFO, and dogleg.
``minimize(f_and_grad, x0, method``"lbfgs", \*\*options)= returns
``(report, x)``. ``x0`` must already be a contiguous float64 vector.
``require_convergence``, ``error_on_unmet``, and ``backward_polish`` are flags
and default to off. ``require_convergence`` raises ``NotConverged`` when the
gradient norm is not below ``gtol``. ``error_on_unmet`` raises
``StopUnmetError`` when the run stops on ``BUDGET``, ``LINE_SEARCH``,
``LINE_SEARCH_ROUNDING``, ``POLISH_UNMET``, or ``EVAL_BUDGET``. It is a
``RuntimeError`` whose ``report`` attribute is the
``Report`` of the stop. ``backward_polish`` is stored on the session. This step
loop does not form a Hessian, so the polish itself is the C
``rgmin_minimize_hess_opts`` and ``rgmin_minimize_hvp_opts`` entries.
An ``lbfgs`` run that spends its budget with ``x`` unmoved stops on
``LINE_SEARCH``, as the C ``rgmin_minimize`` does, not on ``BUDGET``.
``__version__`` is the version string of the extension.

``c1``, ``c2``, and ``linesearch_maxiter`` are the line-search constants.
They are accepted only together with ``linesearch``, one of ``brent``,
``backtracking``, ``goldstein``, or ``wolfe``. Passing a non-default
constant alone raises ``ValueError``. So does a ``linesearch`` the chosen
step ignores: any kind with ``accept="step"`` or on a method that takes
no line search, and any kind but ``wolfe`` on ``lbfgs``.

Two packages, one import name
-----------------------------

Both packages install the import package ``rgmin``, so an environment holds one of them.
The PyPI wheel (the repository root, nanobind) has ``SolverConfig``, ``Solver``, ``minimize``, ``Report``, ``Stop``, ``Method``, ``NotConverged`` and ``rgmin.ase``.
It has no manifolds, no ``Session`` and no ``lowest_eigenpairs``.
The maturin package under ``python/`` is not on PyPI and has ``Session``, ``LineSearch``, ``LbfgsPairs``, ``Stiefel``, ``SPD``, ``Product``, ``lowest_eigenpairs`` and ``rgmin.optimize``.
:doc:`../howto/install` shows how to build each one.

Options of ``SolverConfig`` and ``minimize``
--------------------------------------------

Names are lower case, and ``-`` or a space reads as ``_``.
An unknown name raises ``ValueError`` and lists the valid ones.
The PyPI wheel starts from ``maxiter`` 100, ``gtol`` 1e-8, ``istep`` 0.1, ``memory`` 10 and ``maxmove`` 0.
Those differ from the Rust ``Control::default`` (``gtol`` 1e-5, ``istep`` 1) and from the C ``rgmin_control_default``, so a script that moves between bindings should set them.

.. table::

    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Option                               | Value                                                              | Default   | Meaning                                                                                                                                                                                      |
    +======================================+====================================================================+===========+==============================================================================================================================================================================================+
    | ``maxiter``                          | int >= 0                                                           | 100       | Steps ``minimize`` takes. Zero returns a report without a step. A ``Solver`` takes one step per ``step`` call.                                                                               |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``gtol``                             | float                                                              | 1e-8      | Stop when the gradient norm is below it.                                                                                                                                                     |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``istep``                            | float                                                              | 0.1       | Initial step scale. For FIRE it is the initial time step.                                                                                                                                    |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``memory``                           | int >= 0                                                           | 10        | L-BFGS pairs.                                                                                                                                                                                |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``maxmove``                          | float                                                              | 0         | Euclidean cap on the step. Zero keeps the method's own caps, and a negative value leaves every cap off.                                                                                      |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``atom_maxmove``                     | float                                                              | unset     | Per-atom cap on the longest atom displacement. ``maxstep`` is the same option. Unset keeps the method's default, 0.2 for L-BFGS.                                                             |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``accept``                           | ``none``, ``energy``, ``nonmonotone``, ``step``                    | unset     | How a step is taken. Unset keeps the C session's ``energy``.                                                                                                                                 |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``linesearch``                       | ``brent``, ``backtracking`` (``armijo``), ``goldstein``, ``wolfe`` | unset     | Line search. L-BFGS searches none unless this is ``wolfe``.                                                                                                                                  |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``c1``, ``c2``                       | float                                                              | 1e-4, 0.9 | Wolfe and Armijo constants. Need ``linesearch``.                                                                                                                                             |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``linesearch_maxiter``               | int >= 1                                                           | 20        | Trial budget of the line search. Needs ``linesearch``.                                                                                                                                       |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``beta``                             | a CG variant name, or ``cg``                                       | unset     | Conjugacy coefficient. Only for a CG method. ``cg`` is Polak-Ribiere.                                                                                                                        |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``fire_variant``                     | ``rgmin``, ``guenole2020`` (``fire2``)                             | unset     | FIRE schedule.                                                                                                                                                                               |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``qn_step``                          | ``lbfgs``, ``newton``, ``rfo``                                     | unset     | Quasi-Newton step. Newton and RFO need ``Solver.step_hess``.                                                                                                                                 |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``neb_guards``                       | bool                                                               | unset     | Opt-in L-BFGS NEB guards. Other methods ignore it.                                                                                                                                           |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``periodic``                         | bool                                                               | unset     | Periodic cell. Drops rotation.                                                                                                                                                               |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``project_rigid``                    | bool                                                               | unset     | eOn ``lbfgs_project_rigid``.                                                                                                                                                                 |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``extra_updates``                    | int >= 0                                                           | unset     | Al-Baali extra updates. Accepted and without effect.                                                                                                                                         |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``cautious_eps``, ``cautious_alpha`` | float, set together                                                | unset     | Li-Fukushima filter: a pair is dropped when s.y < eps \* norm(s)\ :sup:`2`\ \* norm(g)\ :sup:`alpha`\.                                                                                       |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``noise_eps``, ``noise_c``           | float                                                              | 0, 0      | Gradient noise floor. ``noise_eps`` <= 0 leaves it off, and ``noise_c`` <= 0 selects 3.                                                                                                      |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``objective_roundoff``               | float                                                              | unset     | Relative value window of the approximate Wolfe test, in place of the default four epsilon. Zero permits no rise, and a negative or non-finite value raises ``ValueError``.                   |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``require_convergence``              | bool                                                               | False     | Raise ``NotConverged`` from ``minimize`` when the gradient norm is not below ``gtol``. A first-order session step that took no move and left its state unchanged reports ``STEP_TOLERANCE``. |
    +--------------------------------------+--------------------------------------------------------------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

A count must be an int, and a flag must be a bool.
A kind or constant that the method ignores raises ``ValueError`` rather than being dropped.
Method names are ``adam``, ``bb``, ``bfgs``, ``cg``, ``conjugate_descent``, ``dai_yuan``, ``diis``, ``dogleg``, ``fire``, ``fire2``, ``fletcher_reeves``, ``fr_pr``, ``hager_zhang``, ``hestenes_stiefel``, ``lbfgs``, ``liu_storey``, ``newton``, ``polak_ribiere``, ``pso``, ``quickmin``, ``rfo``, ``sd``, ``sr1``, ``sr2`` and ``steepest``.

Module ``rgmin`` in the maturin package
---------------------------------------

``Session(kind, n, **options)`` is the stepper. ``kind`` is any gradient-only method token; ``cg`` is Polak-Ribiere-Plus, and ``newton``, ``rfo``, ``dogleg`` and ``pso`` are refused.
``step(x, callback)`` takes one outer step and returns ``(x, value, grad_norm)``, where ``callback(x)`` returns ``(value, gradient)``.
``push_pair(s, y)`` stores an L-BFGS pair without an evaluation and returns whether the curvature gate kept it. ``pair_count()``, ``steps``, ``forget()`` and ``set_level_set(target)`` complete it.
The options and their defaults are:

.. table::

    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | Option                                                                                             | Default                                                                    | Meaning                                                                                                  |
    +====================================================================================================+============================================================================+==========================================================================================================+
    | ``max_atom``                                                                                       | 0.2                                                                        | Per-atom cap. Non-positive leaves it off.                                                                |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``maxmove``                                                                                        | ``None``                                                                   | Euclidean cap. Non-positive leaves it off.                                                               |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``wolfe``                                                                                          | False                                                                      | L-BFGS takes the strong Wolfe search.                                                                    |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``eon_cap``                                                                                        | False                                                                      | L-BFGS takes the eOn capped step.                                                                        |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``memory``                                                                                         | 8                                                                          | L-BFGS pairs.                                                                                            |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``h0``                                                                                             | ``None``                                                                   | Fixed L-BFGS inverse-Hessian scale.                                                                      |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``dt``, ``dt_max``, ``n_min``, ``f_inc``, ``f_dec``, ``alpha``, ``f_alpha``                        | 0.1, 1.0, 5, 1.1, 0.5, 0.1, 0.99                                           | FIRE parameters, the ASE set.                                                                            |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``preset``, ``abc``                                                                                | ``None``, False                                                            | ``preset``"guenole2020"= and ``abc`` select FIRE 2.0 parameters and ABC-FIRE.                            |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``accept``                                                                                         | ``None``                                                                   | ``none``, ``energy`` or ``nonmonotone``. ``step`` is not accepted here.                                  |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``downhill``                                                                                       | False                                                                      | The same as ``accept``"energy"=.                                                                         |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``istep``                                                                                          | ``None``                                                                   | Step scale of a method other than FIRE, whose scale is ``dt``.                                           |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``level_set``                                                                                      | ``None``                                                                   | Target of the level-set constraint, as in :doc:`../howto/level-set`.                                     |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``scale``, ``weight``, ``noise``                                                                   | ``0.4``, ``1.0``, ``0.001`` (or ``0.3``, ``2.0``, ``0.004`` when updating) | GPMin kernel length scale, weight, and observation noise.                                                |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``prior``, ``update_prior_strategy``                                                               | ``None``, ``"maximum"``                                                    | Fixed GPMin prior mean constant or adaptive strategy (``"maximum"``, ``"init"``, ``"average"``).         |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
    | ``update_hyperparams``, ``batch_size``, ``bounds``, ``unbounded_hyperparams``, ``max_refinements`` | ``False``, ``1``, ``0.1``, ``False``, ``30``                               | GPMin marginal-likelihood hyperparameter update schedule and refinement cap (see :doc:`../howto/gpmin`). |
    +----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+

``rgmin.optimize.GPMin`` (also re-exported as ``rgmin.GPMin``) wraps ``Session("gpmin", ...)`` with ``ase.optimize.GPMin``'s constructor signature, defaults, and ``run(fmax, steps)`` convergence flag, plus ``maxstep`` (default ``0.2``, or ``0.0`` for ASE's uncapped steps) and ``hyperparameters`` property. ``Session.set_gtol(gtol)`` updates the session gradient tolerance and ``Session.gpmin_hyperparameters`` reports ``(weight, scale, noise)``.

``LineSearch(kind, x, direction, istep=1.0, c1=1e-4, c2=0.9, beta=0.5, maxiter=40, tol=1e-10)`` is the supplied search. ``offer(f, gradient)`` takes the value and gradient at ``x``, then at each trial, and returns ``(x, alpha, value, done)``. ``done`` is False when ``x`` is the next trial.
``LbfgsPairs(n, capacity=10)`` is the pair memory. ``push(s, y, rho=None)`` returns whether the pair was stored, and ``direction(gradient, h0)`` returns ``-H g`` for ``h0`` a float, an array or a callable. ``clear()``, ``dim()`` and ``len()`` complete it.
``Stiefel``, ``SPD`` and ``Product`` are the manifold classes of :doc:`manifold`.
``lowest_eigenpairs(hvp, x0_block, *, x=None, tol=1e-8, max_iter=None, precond=None, strict=True)`` is block LOBPCG, as :doc:`../tutorials/eigenpairs` shows. It raises ``RuntimeError`` for a pair that misses ``tol`` unless ``strict=False``.

Errors
------

``ValueError`` means an option name or an option value is invalid, or a
coordinate array has the wrong shape. A count such as ``memory`` that
does not fit in a signed 64-bit integer is a ``ValueError``. A count
passed as a float or a bool, for example ``memory`` set to ``1.5``, is
also a ``ValueError`` whose message says it must be an int.
``TypeError`` means an array has the wrong dtype or order, an option
value is not a bool, an int, a float, or a str, ``method`` is not a
string or a ``Method``, or ``f_and_grad`` did not return
``(value, gradient)``.
``RuntimeError`` means the solver is already in use on another thread,
or the native step failed. The message is the text reported by the
library. ``Solver.step`` on Newton, RFO, or dogleg, with no Hessian
callback, raises ``RuntimeError``.
``NumericalError`` is an ``ArithmeticError``. It means the value or the
gradient was not finite.
``NotConverged`` is a ``RuntimeError`` raised by ``minimize`` when
``require_convergence=True`` and the final gradient norm is not below
``gtol``. ``error.report`` is the terminal ``Report``.
An exception raised inside the objective is raised again on the
calling thread.

Module ``rgmin.ase``
--------------------

``Optimizer`` subclasses ``ase.optimize.optimize.Optimizer``.
Named classes: ``LBFGS``, ``FIRE``, ``FIRE2``, ``QuickMin``, ``BFGS``, ``SD``,
``Adam``, ``Newton``, ``RFO``, ``SR1``, ``SR2``, ``BB``, ``Dogleg``, ``DIIS``, ``PSO``,
``FletcherReeves``, ``HestenesStiefel``, ``DaiYuan``, ``ConjugateDescent``,
``HagerZhang``, ``LiuStorey``, ``FrPr``, and ``CG``.
``CG`` defaults to Polak-Ribiere. The C ABI has no separate
Polak-Ribiere+ value.

``run`` returns a Python ``bool``. ``True`` means the force criterion was
met.
``NotImplementedError`` means ``restart``, ``read``, ``dump``, or a Hessian
optimizer called without ``hessian``.
