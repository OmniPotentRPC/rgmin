

Stop on energy change or an evaluation budget
---------------------------------------------

A strict gradient tolerance spends many steps on changes of the energy
that no longer matter. Two opt-in stops end a run earlier, with the
meaning they have in LAMMPS ``minimize etol ftol maxiter maxeval``. Both are
off by default, and a run that sets neither takes the steps it always
took.

.. table::

    +-------------------+---------------------------------------+---------------------+-------------+
    | ``Control`` field | stop                                  | ``Status``          | LAMMPS      |
    +===================+=======================================+=====================+=============+
    | ``gtol``          | gradient norm below ``gtol``          | ``Tolerance``       | ``ftol``    |
    +-------------------+---------------------------------------+---------------------+-------------+
    | ``etol``          | relative energy change below ``etol`` | ``EnergyTolerance`` | ``etol``    |
    +-------------------+---------------------------------------+---------------------+-------------+
    | ``maxiter``       | outer iterations spent                | ``Budget``          | ``maxiter`` |
    +-------------------+---------------------------------------+---------------------+-------------+
    | ``max_evals``     | objective evaluations spent           | ``EvalBudget``      | ``maxeval`` |
    +-------------------+---------------------------------------+---------------------+-------------+

The energy test
~~~~~~~~~~~~~~~

``etol`` stops a run when the objective changes by little between two
accepted steps:

.. code:: text

    |E_k - E_{k-1}| < etol * 0.5 * (|E_k| + |E_{k-1}| + eps_E)

This is the test of ``MinCG::iterate`` in LAMMPS ``src/min_cg.cpp``. The code
compares with a strict ``<``, where the documentation says "less than or
equal", and rgmin follows the code. LAMMPS writes ``EPS_ENERGY = 1.0e-8``
for the floor that keeps the test finite at ``E = 0``. rgmin uses
``eps_E = sqrt(f64::EPSILON)``, about 1.5e-8, so the two tests differ only
where the energy is as small as the floor.

- ``etol = Some(0.0)`` never stops, as in LAMMPS. ``None`` is the default.
- A negative or non-finite ``etol`` is ``Error::InvalidControl`` before the
  objective is called.
- An iteration that left the point where it was, because its line search
  refused every trial, is not a test.
- FIRE takes the test only after more than ``n_min`` consecutive steps with
  positive power, as ``min_style fire`` does with its delay steps.
- A gradient stop at the same point is reported as ``Tolerance``.
- Particle swarm has no sequence of accepted energies. It answers
  ``Error::StopUnsupported`` rather than ignoring ``etol``.

.. code:: rust

    use eindir_core::objectives::Rosenbrock;
    use ndarray::array;
    use rgmin::{Control, LineSearch, Method, Status, minimize_method};

    let obj = Rosenbrock::<2>::new();
    let control = Control::default()
        .with_maxiter(10_000)
        .with_gtol(1e-12)
        .with_etol(1e-6);
    let report = minimize_method(
        &obj,
        array![-1.2, 1.0],
        &control,
        Method::lbfgs(),
        LineSearch::default(),
    )
    .unwrap();
    assert_eq!(report.status, Status::EnergyTolerance);

The evaluation budget
~~~~~~~~~~~~~~~~~~~~~

``max_evals`` stops a run after that many calls to the objective. One call
to ``eval``, ``grad`` or ``value_and_gradient`` is one evaluation, a
line-search trial included. A Hessian or a Hessian-vector product is not
one. The count is read when an iteration ends, as LAMMPS reads
``maxeval``, so the last iteration may overshoot by what one iteration
costs. ``Report::evals`` and ``Solver::evals()`` hold the count, and it
equals the number of calls the objective saw. The budget is read before
the energy test, and a gradient stop wins over both.

``max_evals = Some(0)`` is ``Error::InvalidControl``. Every method honours
the budget, particle swarm included.

Sessions
~~~~~~~~

A session carries the stops in its ``Control``. ``Solver::set_etol`` and
``Solver::set_max_evals`` change them between steps, and ``None`` turns one
off. ``Solver::evals()`` counts over the life of the session, so a budget
set later compares with the evaluations already made.

Other languages
~~~~~~~~~~~~~~~

.. table::

    +-------------------------+--------------------------------------------------------+------------------------------------------------------------------+------------------------+
    | surface                 | etol                                                   | max_evals                                                        | count                  |
    +=========================+========================================================+==================================================================+========================+
    | C                       | ``rgmin_solver_set_etol``, ``rgmin_solver_clear_etol`` | ``rgmin_solver_set_max_evals``, ``rgmin_solver_clear_max_evals`` | ``rgmin_solver_evals`` |
    +-------------------------+--------------------------------------------------------+------------------------------------------------------------------+------------------------+
    | C++                     | ``Solver::set_etol``, ``Solver::clear_etol``           | ``Solver::set_max_evals``, ``Solver::clear_max_evals``           | ``Solver::evals``      |
    +-------------------------+--------------------------------------------------------+------------------------------------------------------------------+------------------------+
    | Python ``SolverConfig`` | ``etol=``                                              | ``max_evals=``                                                   | ``Solver.evals``       |
    +-------------------------+--------------------------------------------------------+------------------------------------------------------------------+------------------------+
    | Python ``Session``      | ``etol=``, ``set_etol``                                | ``max_evals=``, ``set_max_evals``                                | ``Session.evals``      |
    +-------------------------+--------------------------------------------------------+------------------------------------------------------------------+------------------------+
    | ``rgmin.ase``           | ``etol=``                                              | ``max_evals=``                                                   | ``Optimizer.stop``     |
    +-------------------------+--------------------------------------------------------+------------------------------------------------------------------+------------------------+
    | ``rgmin.optimize``      | ``etol=``                                              | ``max_evals=``                                                   | ``Optimizer.stop``     |
    +-------------------------+--------------------------------------------------------+------------------------------------------------------------------+------------------------+
    | LAMMPS                  | ``minimize etol``                                      | ``minimize ... maxeval``                                         |                        |
    +-------------------------+--------------------------------------------------------+------------------------------------------------------------------+------------------------+

The C setters return ``RGMIN_INVALID_PARAMETER`` with the reason in
``rgmin_last_error`` for a bad value. The step reports carry
``RGMIN_STOP_ENERGY_TOLERANCE`` and ``RGMIN_STOP_EVAL_BUDGET``, and
``rgmin.Stop`` has ``ENERGY_TOLERANCE`` and ``EVAL_BUDGET``. ABI minor 50
added them. ``rgmin_control_t`` keeps its layout, so the one-shot
``rgmin_minimize`` calls carry no stops; use a session.

The LAMMPS plugin passes ``etol`` to the core. It keeps ``maxeval`` on its
own count of ``Min::energy_force`` calls, which is the ``Force evaluations``
line of the LAMMPS summary and leaves out the evaluation it reuses from
setup.

``rgmin.ase`` classes take ``etol`` and ``max_evals`` like any other solver
option. ASE's ``run`` ends on ``etol`` and on ``max_evals``. It returns ``True``
when the forces or the energy change met their tolerance and ``False``
when the evaluation budget or the step count ran out. ``Optimizer.stop``
is the ``rgmin.Stop`` of the last step. The ``rgmin.optimize`` classes take
the same keywords, and their ``Optimizer.stop`` is the name of the last
status, such as ``"EnergyTolerance"``.

SciPy and LAMMPS
~~~~~~~~~~~~~~~~

.. table::

    +---------------+-------------+------------------------------------------------------+
    | rgmin         | LAMMPS      | SciPy                                                |
    +===============+=============+======================================================+
    | ``gtol``      | ``ftol``    | ``gtol`` (``L-BFGS-B``, ``CG``, ``BFGS``)            |
    +---------------+-------------+------------------------------------------------------+
    | ``etol``      | ``etol``    | ``ftol`` (``L-BFGS-B``), ``fatol`` (``Nelder-Mead``) |
    +---------------+-------------+------------------------------------------------------+
    | ``maxiter``   | ``maxiter`` | ``maxiter``                                          |
    +---------------+-------------+------------------------------------------------------+
    | ``max_evals`` | ``maxeval`` | ``maxfun`` (``L-BFGS-B``), ``maxfev``                |
    +---------------+-------------+------------------------------------------------------+

SciPy ``L-BFGS-B`` stops when
``(f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= ftol``. When the energy
has magnitude 1 or more and changes by little, both denominators are
about ``|E|``, so ``etol = ftol`` is the same test up to the comparison.
Below magnitude 1 SciPy divides by 1 and rgmin by the energy, so a SciPy
``ftol`` is an absolute tolerance there and ``etol`` stays relative.
``maxfun`` counts function evaluations including line-search trials, the
count ``max_evals`` keeps.
