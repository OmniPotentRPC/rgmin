

Run Gaussian-process-accelerated optimization (GPMin)
-----------------------------------------------------

``GPMin`` implements the local Gaussian-process-accelerated geometry
optimizer of Garijo del Río, Mortensen, and Jacobsen
:cite:`garijodelrioLocalBayesianOptimizer2019`.
At every step, ``rgmin`` fits a Gaussian process (GP) surrogate over all
observed energies and gradients with a squared-exponential covariance
kernel and analytic derivative blocks, minimizes the GP posterior mean
from the current point with an inner L-BFGS search, and evaluates the
true objective at the proposed point.
If the true energy does not decrease, the trial observation is appended
to the GP and the surrogate is minimized again from the current point
until a descent step is found, the gradient tolerance is met, or the
refinement budget is reached.

Because each training step factors a dense ``(n * (D + 1))`` covariance
matrix with Cholesky decomposition, ``GPMin`` targets expensive objectives
(such as ab initio calculators) on modest dimensions (roughly up to 100
atoms).

Python drop-in for ASE
~~~~~~~~~~~~~~~~~~~~~~

The maturin package provides ``rgmin.optimize.GPMin`` (also re-exported as
``rgmin.GPMin``), matching the constructor signature, defaults, and
``run(fmax, steps)`` contract of ``ase.optimize.GPMin``:

.. code:: python

    from rgmin.optimize import GPMin

    opt = GPMin(
        atoms,
        logfile="gpmin.log",
        trajectory="gpmin.traj",
        update_prior_strategy="maximum",
        update_hyperparams=False,
    )
    converged = opt.run(fmax=0.01, steps=100)

To enable bounded marginal log-likelihood hyperparameter updates on a
batch schedule:

.. code:: python

    opt = GPMin(
        atoms,
        update_hyperparams=True,
        batch_size=1,
        bounds=0.1,
    )
    converged = opt.run(fmax=0.01)
    weight, scale, noise = opt.hyperparameters

``maxstep`` defaults to ``0.2`` Å, matching the per-atom displacement cap of
the other ``rgmin.optimize`` drop-ins. Pass ``maxstep=0.0`` to disable the
cap and match ASE's uncapped surrogate steps.

For non-ASE objectives, construct ``rgmin.Session("gpmin", n, ...)``
directly and drive ``session.step(x, callback)``.

Rust API
~~~~~~~~

The Rust implementation lives in ``rgmin::gpmin`` and works on any
``D``-dimensional smooth objective, with no assumption that ``D`` is a
multiple of 3.
Use either the standalone ``GpMin`` stepper or a ``Solver`` session with
``Method::GpMin``:

.. code:: rust

    use ndarray::array;
    use rgmin::{Control, GpMin, GpMinConfig, Method, Oracle, PriorStrategy, Solver};

    let obj = Oracle::unbounded(2, |x| {
        (x[0] * x[0] + 3.0 * x[1] * x[1], array![2.0 * x[0], 6.0 * x[1]])
    });

    let cfg = GpMinConfig::default()
        .with_scale(0.4)
        .with_weight(1.0)
        .with_noise(0.001)
        .with_prior(PriorStrategy::Maximum)
        .with_max_step(Some(0.2));

    let mut x = array![0.3, -0.2];
    let mut gpmin = GpMin::new(cfg.clone(), 2)?;
    let report = gpmin.run(&obj, &mut x, &Control::default().with_gtol(1e-4))?;

    // Or through Solver:
    let mut x2 = array![0.3, -0.2];
    let mut solver = Solver::new(Method::GpMin, Control::default().with_gtol(1e-4), 2);
    solver.set_gpmin_config(cfg)?;
    let report2 = solver.run(&obj, &mut x2)?;

Configuration and defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~

Every hyperparameter is configured through ``GpMinConfig`` (Rust) or
``GPMin(...)`` / ``Session("gpmin", ...)`` (Python), with defaults matching
ASE's ``ase.optimize.GPMin``:

.. table::

    +---------------------------+------------------------------+-----------------------------+---------------------------------------------------------------------------------------------------------------------+
    | Parameter                 | ``update_hyperparams=False`` | ``update_hyperparams=True`` | Meaning                                                                                                             |
    +===========================+==============================+=============================+=====================================================================================================================+
    | ``scale``                 | ``0.4``                      | ``0.3``                     | Squared-exponential kernel length scale.                                                                            |
    +---------------------------+------------------------------+-----------------------------+---------------------------------------------------------------------------------------------------------------------+
    | ``weight``                | ``1.0``                      | ``2.0``                     | Kernel prefactor (energy scale).                                                                                    |
    +---------------------------+------------------------------+-----------------------------+---------------------------------------------------------------------------------------------------------------------+
    | ``noise``                 | ``0.001``                    | ``0.004``                   | Observation noise regularization (energy variance ``(noise * scale)^2``, gradient variance ``noise^2``).            |
    +---------------------------+------------------------------+-----------------------------+---------------------------------------------------------------------------------------------------------------------+
    | ``update_prior_strategy`` | ``"maximum"``                | ``"maximum"``               | Constant energy prior rule: ``"maximum"``, ``"init"``, ``"average"``, or a fixed constant ``prior``.                |
    +---------------------------+------------------------------+-----------------------------+---------------------------------------------------------------------------------------------------------------------+
    | ``batch_size``            | unused (``1``)               | ``1``                       | Number of new observations between marginal-likelihood updates.                                                     |
    +---------------------------+------------------------------+-----------------------------+---------------------------------------------------------------------------------------------------------------------+
    | ``bounds``                | unused (``0.1``)             | ``0.1``                     | Relative box half-width ``[(1 - b) t0, (1 + b) t0]`` for ``(weight, scale)``; ``False`` / ``None`` disables bounds. |
    +---------------------------+------------------------------+-----------------------------+---------------------------------------------------------------------------------------------------------------------+
    | ``max_refinements``       | ``30``                       | ``30``                      | Maximum surrogate refinements per step before failing closed.                                                       |
    +---------------------------+------------------------------+-----------------------------+---------------------------------------------------------------------------------------------------------------------+

Fail-closed error handling
~~~~~~~~~~~~~~~~~~~~~~~~~~

``GPMin`` never adds silent jitter or falls back to another step when the
surrogate fails:

- Non-positive-definite covariance matrices after applying the configured
  ``noise`` regularization return ``Error::GpMin`` in Rust (``ValueError`` in
  Python) naming the Cholesky failure.

- Non-finite energies or gradients from the objective return
  ``Error::Oracle`` (``ValueError`` in Python).

- Exhausting ``max_refinements`` without finding a descent step returns
  ``Error::GpMin`` (``ValueError`` in Python).
