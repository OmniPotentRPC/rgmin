

Distance constraints
--------------------

This page states what a ``DistanceConstraints`` set computes, which tolerances and budgets it derives, and where it agrees with GROMACS.
:doc:`../howto/constraints` shows how to install a set on a session.

The set and its metric
~~~~~~~~~~~~~~~~~~~~~~

Constraint ``k`` fixes the bond ``r_k = x_i - x_j`` of the pair ``(i, j)`` at the length ``d_k``.
``J`` is the Jacobian of ``(|r_k|^2 - d_k^2) / 2``, so row ``k`` has ``r_k`` at atom ``i`` and ``-r_k`` at atom ``j``.
``M`` is the diagonal matrix of atom masses, three entries per atom, and ``W = M^{-1}``.
Without ``with_masses``, ``M = I``.

These masses weight the coordinates of the session, ``q = M^{1/2} x``.
A vector ``v`` of ``x`` is ``M^{1/2} v`` in ``q``, and a gradient ``g`` of ``x`` is ``W^{1/2} g`` in ``q``.
Steps, stored pairs and the stop test all live in ``q``, so a quasi-Newton method works on the set with the inner product ``<u, v>_M = u^T M v``.

Tangent projection
~~~~~~~~~~~~~~~~~~

Tangent vectors at a point ``x`` of the set come from the projector

``Pi = I - W J^T (J W J^T)^{-1} J``,

which RATTLE applies to velocities (Andersen `1983 <https://doi.org/10.1016/0021-9991(83)90014-1>`_).
A direction is projected with ``Pi`` and a gradient with ``Pi^T g = g - J^T lambda``, ``lambda = (J W J^T)^{-1} J W g``.
In ``q``, ``Pi_q = M^{1/2} Pi M^{-1/2}`` is symmetric, so it is the orthogonal projector of that space.
The gradient of a step is ``g_q = W^{1/2} Pi^T g``, and ``grad_norm`` reports ``|Pi^T g|`` in the session's stop norm.
At a constrained minimum ``g = J^T lambda``, with ``lambda`` the vector of Lagrange multipliers.

The projector and the multipliers need one solve with ``J W J^T`` per block of coupled constraints, a connected part of the constraint graph.
For one block the solve uses the scaled coupling ``S = D^{-1/2} J W J^T D^{-1/2}`` with ``D = diag(J W J^T)``, which has a unit diagonal.
Its extreme eigenvalues come from cyclic Jacobi rotations, largest entry first, to ``U ||S||_F``, with ``U`` the unit roundoff.
A backward-stable Cholesky solve with ``S`` gains ``log2(1 / (b kappa gamma(6)))`` bits per pass, with ``b`` the block size, ``kappa = lambda_max / lambda_min`` and ``gamma(n) = n U / (1 - n U)`` (Higham 2002, Lemma 3.1).
A block where that is under one bit is ``ConstraintError::Dependent``.
The projection repeats the solve until every ``|J_k u|`` is within ``gamma(6)`` of the magnitudes it was formed from, a budget of the bits to gain over the bits per pass.

Return to the set
~~~~~~~~~~~~~~~~~

A step leaves the set at a point ``x``, the anchor, and arrives at a trial ``y``.
The trial returns to the set by corrections along the anchor bonds,

``y' = y - W J(x)^T mu``, with ``mu`` chosen so that ``|r_k(y')| = d_k`` for every ``k``.

Three solvers meet these equations.
Ryckaert, Ciccotti and Berendsen (`1977 <https://doi.org/10.1016/0021-9991(77)90098-5>`_) gave the Gauss-Seidel bond solver (SHAKE).
The linear constraint solver (LINCS) comes from Hess, Bekker, Berendsen and Fraaije (`1997 <https://doi.org/10.1002/(sici)1096-987x(199709)18:12%3C1463::aid-jcc4%3E3.0.co;2-h>`_).
Miyamoto and Kollman (`1992 <https://doi.org/10.1002/jcc.540130805>`_) solved the rigid triangle in closed form with SETTLE.
They differ in how, not in the answer, so ``R_x(v) = constrain(x + v)`` is one retraction whichever solver ran.
The corrections run along ``W J(x)^T``, a fixed complement of the tangent space at ``x``.
Hence ``R_x(0) = x``, and ``R_x`` has the identity on tangent vectors as its derivative at zero.
That makes ``R`` a retraction in the sense of Absil, Mahony and Sepulchre :cite:`absilOptimizationAlgorithmsMatrix2008`, of the projection-like kind that Absil and Malick study (`2012 <https://doi.org/10.1137/100802529>`_).

The residual of constraint ``k`` is the GROMACS SHAKE measure ``e_k = |d_k^2 - |r_k|^2| / (2 d_k^2)``.
Its tolerance is ``max(t, f_k)``, with ``t`` from ``with_tolerance`` and the floor

``f_k = gamma(6) / 2 + U reach_k / d_k``, ``reach_k = || |x_i| + |x_j| ||_2``.

The first term bounds the rounding error of ``e_k`` itself: five operations for ``|r_k|^2`` and one for ``d_k^2``.
The second covers the stored coordinates, each rounded to ``U`` of its magnitude, along the bond.
No constraint can be met more tightly than its floor, so ``t = 0``, the default, asks for the floor.
``excess`` is the largest ``e_k`` over its tolerance.

SHAKE
^^^^^

Gauss-Seidel SHAKE takes the constraints of a block in turn and moves the two atoms of each along its anchor bond by the Newton step of its own length equation, split by the inverse masses.
A constraint within its tolerance is skipped.
SHAKE returns ``ConstraintError::Reversed`` when a trial bond turns a right angle or more from its anchor direction.
No correction along the anchor bond can then restore its length.

Near the set the sweep is linear Gauss-Seidel on ``S``.
With the splitting ``S = (I + L) + L^T``, its contraction in the ``S`` norm is ``rho_GS^2 <= 1 - lambda_min / ||I + L||_F^2``.
The bound follows from the Xu-Zikatanov identity, with the Frobenius norm as an upper bound on the spectral norm.
The sweep budget is the number of sweeps that contraction needs to bring the largest residual to its tolerance, after converting between the residual and the ``S`` norm.
A lone bond adds its Newton sweeps, which double the correct bits each time.
A block that spends the budget is ``ConstraintError::NotConverged``.

LINCS
^^^^^

One LINCS projection sets every ``n_k . r_k`` to a target, with ``n_k`` the unit anchor bond, by summing the expansion ``(I - A)^{-1} = sum_p A^p`` of ``S = I - A`` (Hess et al. 1997, eqs. 18-20).
The expansion converges when ``rho_L = max |1 - lambda(S)| < 1``.
The order is the number of terms that brings the truncation bound ``rho_L^{n + 1} / (1 - rho_L)`` to the tolerance, unless ``with_lincs_order`` fixes it.
The first projection removes ``n_k . r_k - d_k``.
Each later correction removes ``d_k - sqrt(2 d_k^2 - l_k^2)``, the lengthening a rotation leaves (eq. 21, GROMACS ``calc_dist_iter``), until every constraint meets its tolerance.
Each correction gains one bit from the rotation and ``(order + 1) log2(1 / rho_L)`` bits from the expansion, and the budget counts the smaller gain.

``ConstraintSolver::Auto`` takes LINCS on a block when ``rho_L < 1`` and ``rho_L <= rho_GS``, and SHAKE otherwise.
``ConstraintSolver::Lincs`` on a block with ``rho_L >= 1`` is ``ConstraintError::LincsUnsafe``.
A light atom bonded to two or more heavy ones gives such a block: the off-diagonal couplings of ``S`` approach the cosines of the bond angles.

SETTLE
^^^^^^

A block of three atoms and three constraints is a rigid triangle.
Corrections along the anchor bonds lie in the anchor plane and are central forces between the anchor positions.
So every atom keeps its coordinate along the anchor normal, and the centre of mass stays where the trial put it.
The corrections also exert no torque about the normal at the anchor positions.
The rigid triangle that meets these conditions is the converged SHAKE answer.
SETTLE finds it in closed form: the normal coordinates fix the tilt of the triangle, and the torque condition fixes its angle in the plane through ``C cos(theta) + D sin(theta) = E``.
SETTLE takes the root nearer the trial in the mass metric.
When ``|E| > sqrt(C^2 + D^2)`` no root exists, and SETTLE returns ``ConstraintError::SettleDegenerate``.
A SHAKE pass then removes the rounding of the closed form.
Miyamoto and Kollman treat water with equal hydrogen masses and equal O-H lengths; the derivation here needs neither.

Steps on the set
~~~~~~~~~~~~~~~~

Every method projects its direction with ``Pi_q`` and places the trial at ``x + W^{1/2} d``.
The trial is capped like an unconstrained one and then returned to the set.
Coordinate bounds act only on atoms outside every pair, where ``Pi`` is the identity, so the box projection commutes with ``Pi``.
A constrained atom that a trial takes to its bound is ``ConstraintError::Bound``.
What each method stores is moved to the new tangent space by projection, the vector transport of Absil, Mahony and Sepulchre (section 8.1.3) :cite:`absilOptimizationAlgorithmsMatrix2008`:

- L-BFGS stores ``s_q = Pi_q(x+) M^{1/2} (x+ - x)`` and ``y_q = g_q+ - Pi_q(x+) g_q``, and projects every stored pair at each new point.

- Nonlinear conjugate gradients project the previous gradient and direction, and Barzilai-Borwein projects its last pair.

- Broyden-Fletcher-Goldfarb-Shanno (BFGS), SR1 and SR2 update with the same tangent pair as L-BFGS.

- The fast inertial relaxation engine (FIRE) projects its velocity and its force with the same projector.

- Adam keeps its moments, and its direction is projected like every other.

Under ``Accept::Energy`` and ``Accept::Nonmonotone``, L-BFGS runs its line search along the curve ``R(t) = constrain(x + t d)``.
The slope of the curve at a trial ``r = R(t)`` is ``g(r)^T R'(t)``, with ``R'(t) = d - W J(x)^T mu`` and ``J(r) R'(t) = 0``, one oblique block solve per trial.
Newton, the rational function optimization (RFO) step and dogleg use the Hessian of the Lagrangian ``H - sum_k lambda_k nabla^2 c_k`` on the tangent space.
Newton-CG uses the same operator through Hessian-vector products.

Agreement with GROMACS
~~~~~~~~~~~~~~~~~~~~~~

The GROMACS 2026.4 sources ``src/gromacs/mdlib/shake.cpp``, ``lincs.cpp``, ``settle.cpp``, ``constr.cpp`` and ``mdrun/minimize.cpp`` were the working reference.
The code here is written for minimization and is not a translation.

.. table::

    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
    | Topic                                | GROMACS                                                                                                                            | rgmin                                                                                     |
    +======================================+====================================================================================================================================+===========================================================================================+
    | SHAKE residual                       | ``0.5 abs(d^2 - r^2) / d^2`` against ``shake-tol``                                                                                 | Same measure, tolerance at least the rounding floor                                       |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
    | Iteration budget                     | ``maxnit = 1000``                                                                                                                  | Derived from the Gauss-Seidel contraction                                                 |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
    | Bond reversal                        | ``r_ref . r < 1e-10 d^2`` is an error                                                                                              | A right angle between anchor and trial bond is ``Reversed``                               |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
    | LINCS order                          | ``lincs-order``, default 4; ``grompp`` asks for 8 with CG and L-BFGS                                                               | Derived from ``rho_L`` and the tolerance, or fixed                                        |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
    | Rotational corrections               | ``lincs-iter``, default 1                                                                                                          | Until the tolerance, within a derived budget                                              |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
    | Coupled angles                       | Accepted; triangles get extra recursions                                                                                           | ``Auto`` picks SHAKE where the expansion diverges or is slower                            |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
    | SETTLE                               | Water: equal hydrogen masses and O-H lengths                                                                                       | Any rigid triangle, any masses                                                            |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
    | Masses in minimization               | Every mass is one                                                                                                                  | Unit by default, real masses on request                                                   |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
    | SHAKE in minimization                | ``grompp`` refuses it for CG and L-BFGS ("use LINCS"); steepest descent stops in ``mdrun``, which cannot project forces with SHAKE | Supported                                                                                 |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
    | L-BFGS with constraints              | Not implemented; ``mdrun`` stops                                                                                                   | Supported, with transported pairs and a search on the curve                               |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
    | Conjugate gradients with constraints | LINCS and SETTLE only; a step they cannot meet ends the run                                                                        | Supported with every solver; under the energy tests such a trial is refused and shortened |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
