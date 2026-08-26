

Constrain an IRC step to the mass-weighted sphere
-------------------------------------------------

The inner Gonzalez--Schlegel / Sella IRC step
:cite:`gonzalezImprovedAlgorithmReaction1989,hermesSellaOpensourceAutomationfriendly2022`
is not ``ManifoldKind::Sphere`` (the unit sphere about the origin).
It is an equality on ``MwRigid``:



.. math::

    \|(s + d_1)\odot\sqrt{m}\| = dx.

``IrcTrust`` is that restricted step.
``lowest_mode`` is the matrix-free kick (one extremal Hessian
pair from ``H v``, not a full ELPA / SLATE spectrum)
:cite:`ishidaIntrinsicReactionCoordinate1977,pageEvaluatingReactionPath1988,davidsonIterativeCalculationFew1975,sleijpenJacobidavidsonIterationMethod1996,knyazevTowardOptimalPreconditioned2001`.
Dispatch is the closed ``EigensolverKind`` in ``schema/eigen.capnp``:
Lanczos (default), Rayleigh-Ritz, Jacobi-Davidson, LOBPCG. Feature
``slepc`` links SLEPc EPS (MatShell, typed ``EPSSet*`` / ``STSet*``,
no options database) when PETSc/SLEPc are present. PRIMME, ChASE,
ELPA, ELPA2, SLATE, MAGMA, cuSOLVER, DLA-Future, EigenExa, and
unbuilt SLEPc return ``Error::EigenUnavailable`` until linked
:cite:`winkelmannChase2019,marekElpaLibraryScalable2014,yuGpuaccelerationElpa2Distributed2021`.
There is no string key. The session that owns forward and reverse
branches lives in
`rgsaddle <https://github.com/OmniPotentRPC/rgsaddle>`_ (``IrcSession``).

Rust
~~~~

.. code:: rust

    use ndarray::array;
    use rgmin::{lowest_eigenpair, lowest_mode, EigenParams, EigensolverKind, IrcTrust};

    let masses = [1.0, 16.0];
    let d1 = array![0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
    let tr = IrcTrust::from_atom_masses(d1, &masses, 0.2);
    let s = array![0.5, 0.1, -0.2, 0.3, 0.0, 0.4];
    let p = tr.project(&s);
    assert!(tr.on_bound(&p, 1e-12));

The kick uses ``lowest_eigenpair`` (Lanczos) or ``lowest_mode`` with
an ``EigenParams`` on a ``HessianVector`` (analytic or ``FdHvp``).
Default Krylov dimension is small. Do not form the dense Hessian
just to pick the imaginary mode. Unlinked backends fail closed.

What this is not
~~~~~~~~~~~~~~~~

- ``ManifoldKind::Sphere`` -- unit :math:`S^{n-1}` about 0.

- A full ``heev`` of a 3N Hessian. ELPA / SLATE belong behind
  ``n >= 512`` and a Hessian that already exists. See the rgsaddle
  MEP tutorial.
