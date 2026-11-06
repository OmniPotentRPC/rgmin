==================================================================
The secant family: BFGS, its limited memory, and the session hooks
==================================================================


.. contents::

Newton's method prices a step at one Hessian solve; the secant family
refuses to pay it. This page derives what the crate's quasi-Newton
solvers (``src/qn.rs``, ``src/lbfgs.rs``) actually compute, and why the
warm-restart and recognition hooks on the session are correctness-
preserving.

1 The secant condition
----------------------

After a step ``s = x_new - x_old`` with gradient change
``y = g_new - g_old``, any model Hessian ``B`` worth keeping should
reproduce the observed curvature: B s = y. In inverse form, with
H = B\ :sup:`-1`\, the condition reads H y = s. One equation cannot determine a
matrix, so each method adds a minimality principle. BFGS chooses the
inverse update closest to the old ``H`` in a weighted Frobenius norm
subject to the secant condition and symmetry, which lands on

H\ :sub:`new`\ = (I - rho s y\ :sup:`T`\) H (I - rho y s\ :sup:`T`\) + rho s s\ :sup:`T`\,
rho   = 1 / (y . s).

The update preserves positive definiteness exactly when the curvature
pair condition y . s > 0 holds, which the strong Wolfe curvature
condition guarantees at an accepted line-search point. That is the
architectural reason the crate pairs ``Lbfgs`` with ``LineSearch::Wolfe``
by default: the line search is not a convenience, it is the
hypothesis of the update's soundness.

2 The two-loop recursion
------------------------

Storing ``H`` costs n\ :sup:`2`\; limited-memory BFGS stores the last ``m`` pairs
(s\ :sub:`i`\, y\ :sub:`i`\) and applies the update implicitly. Unrolling the BFGS
formula ``m`` times around a seed matrix H0 gives the two-loop
recursion (Nocedal and Wright 7.4):

q = g
for i = m-1 .. 0:   alpha\ :sub:`i`\ = rho\ :sub:`i`\ (s\ :sub:`i`\ . q);  q = q - alpha\ :sub:`i`\ y\ :sub:`i`\
r = H0 q
for i = 0 .. m-1:   beta = rho\ :sub:`i`\ (y\ :sub:`i`\ . r);     r = r + (alpha\ :sub:`i`\ - beta) s\ :sub:`i`\
direction = -r

with the standard scalar seed H0 = (s . y) / (y . y) I from the most
recent pair, a one-parameter fit of the average curvature along the
last step; positivity of that scalar is again the pair condition.
An installed exponential preconditioner replaces that scalar.
The middle product of the two-loop solves with the preconditioner,
so the initial inverse Hessian is taken from it: ``H0 = P^{-1}``.
Packwood, Kermode, Mones, Bernstein, Woolley, Gould, Ortner, and
Csányi, *A universal preconditioner for simulating condensed phase
materials*, J. Chem. Phys. 144, 164109 (2016)
(`10.1063/1.4947024 <https://doi.org/10.1063/1.4947024>`_;
:cite:`packwoodUniversalPreconditionerSimulating2016`).
A dense pair or Lindh matrix passed into the two-loop still wins.
With no preconditioner the scalar seed above is unchanged.
Every dot and axpy in the crate's two-loop routes through the
``vecops`` seam, so the recursion is written once and the storage
backend (serial, ``par``, and by design a later device backend) is a
detail behind four functions.

3 The geometry step
-------------------

A per-atom cap (``set_atom_maxmove``) on a Euclidean L-BFGS session
selects this step and ignores ``set_accept``. A HiGHS feasible set
and a non-Euclidean manifold stay on the two-loop path above.
Uncapped L-BFGS is unchanged: the scalar seed, and the strong Wolfe
search unless acceptance is ``none``.

The initial scale is selected from the size of the nominal steepest
trial. A trial that exceeds the per-atom cap by at least 64 times
receives one directional-curvature probe; ordinary trials start from
``H0 = 0.1 I`` without spending that force call. The probe is
``x + 0.01 u``, with ``u`` the unit force, clipped to the objective
bounds before it is evaluated. The quotient divides by the
displacement that remains along ``u``. A probe the box reduces to
nothing, or a non-positive result, clears the memory and the step
is steepest descent scaled to the cap. Later iterations set
``H0 = (s . y) / (y . y)`` when both dots are positive. Negative
curvature clears the stored pairs and uses that same steepest step.
The stiffness probe uses monotone energy acceptance. An ordinary
start admits an energy below the largest of the last five accepted
values.

An installed exponential preconditioner replaces that scalar seed.
The two-loop applies ``P^{-1}`` in its place, and a reset that would
have stepped along ``h0`` times the force steps along ``P^{-1}`` times
the force. The negative-curvature step and the short fallback after
ten halvings keep the coefficients 1000 and one tenth, applied to
``P^{-1}`` times the force. A failed solve with ``P`` falls back to the
scalar seed. The per-atom cap still limits the displacement.

An uncapped direction longer than the cap, or one that opposes the
force, also clears the memory and replaces the direction with that
capped seed. The trial is kept when the energy rises by at most
1e-8. The step is halved at most ten times. If none of those trials
pass, the memory is cleared and a short steepest step, one tenth of
the preconditioned force and then the cap, is taken.

4 Stale pairs and the cautious update
-------------------------------------

A pair with y . s <= 0 would corrupt the inverse model, so the update
skips it; the ``cautious`` knob (Li-Fukushima) additionally requires
y . s >= eps norm(s)\ :sup:`alpha`\ before a pair enters the history, which
keeps far-from-quadratic steps from polluting the memory. Skipping is
always sound: the two-loop over any subset of valid pairs is the BFGS
inverse built from exactly those pairs.

5 Bounds: the generalized Cauchy point
--------------------------------------

``Lbfgs::minimize_box`` keeps the same pairs and replaces the two-loop direction with the step of limited-memory Broyden-Fletcher-Goldfarb-Shanno with bounds (L-BFGS-B).
Byrd, Lu, Nocedal and Zhu, *A limited memory algorithm for bound constrained optimization*, Society for Industrial and Applied Mathematics (SIAM) Journal on Scientific Computing 16, 1190 (1995) (`10.1137/0916069 <https://doi.org/10.1137/0916069>`_; :cite:`byrdLimitedMemoryAlgorithm1995`).
The step needs ``B`` rather than ``H``, so it uses the compact form of Byrd, Nocedal and Schnabel, Mathematical Programming 63, 129 (1994) (`10.1007/BF01582063 <https://doi.org/10.1007/BF01582063>`_; :cite:`byrdRepresentationsQuasinewtonMatrices1994`):

::

    B = theta I - W M W^T,   W = [Y, theta S],
    M^-1 = [[-D, L^T], [L, theta S^T S]],

with theta = (y\ :sup:`T`\ y) / (s\ :sup:`T`\ y) from the newest pair, ``D`` the diagonal of ``s_i^T y_i``, and ``L`` the strictly lower triangle of S\ :sup:`T`\ Y.
Both forms describe the same matrix when the two-loop recursion seeds with 1/theta.

The bounded step built on this matrix has two parts.
The generalized Cauchy point is the first local minimizer of the quadratic model along the projected steepest-descent path clip(x - t g, lo, hi).
That path is piecewise linear and bends where a coordinate meets its bound.
The search visits the bends in order of t and updates the slope and curvature of the model in O(m\ :sup:`2`\) work per bend (algorithm CP of the paper).
The coordinates that the Cauchy point leaves on a bound stay there.
One reduced Newton step moves the other coordinates.
It costs a 2m by 2m solve (section 5.1).
If that point leaves the box, it is projected back.
The projection is kept when it is still a descent direction; otherwise the step is cut back to the box.
That test is the remark of Morales and Nocedal, Association for Computing Machinery (ACM) Transactions on Mathematical Software 38, 1 (2011) (`10.1145/2049662.2049669 <https://doi.org/10.1145/2049662.2049669>`_; :cite:`moralesRemarkAlgorithm7782011`).
The strong Wolfe search runs along the segment from x to that point and never steps outside the box.
The Fortran 77 code of Zhu, Byrd, Lu and Nocedal, ACM Transactions on Mathematical Software 23, 550 (1997) (`10.1145/279232.279236 <https://doi.org/10.1145/279232.279236>`_; :cite:`zhuAlgorithm778Lbfgsb1997`), as translated to C in SciPy 1.18.1 (commit e4e854ea, ``scipy/optimize/src/lbfgsb.c``, routines ``cauchy`` and ``subsm``), is the reference for the details.

With no finite bound the Cauchy point meets no bend and the reduced step covers every coordinate.
The end point is then x - H g, the two-loop direction up to rounding, and the unit test ``unbounded_direction_is_the_two_loop_direction`` checks that to 1e-12.
SciPy skips the Cauchy point when no bound is finite.
The method then differs from ``Lbfgs::minimize`` only in rounding.
:doc:`../reference/sota` measures that cost on a 75-atom Lennard-Jones cluster.

6 Session hooks: forget, watched, recognized
--------------------------------------------

``Solver`` (and anneal's ``WarmLbfgs`` face over it) exposes three
lifecycle hooks whose safety arguments are one sentence each:

- ``forget()`` clears the pair history. The next step is plain steepest
  descent scaled by the seed; every stored invariant is vacuous.

- ``minimize_watched`` runs a caller predicate per accepted iterate and
  stops early when asked. Stopping early returns an accepted iterate,
  so every property of the trajectory up to that point stands.

- ``minimize_recognized`` additionally lets the caller substitute a
  known result (energy and coordinates) when it recognizes the basin
  mid-descent, returning a flag that says the substitution happened.
  The optimizer never manufactures the substitute; it only carries the
  caller's answer out, so the correctness burden sits exactly where
  the knowledge is. The identity and refund tests in
  ``tests/lbfgs_state.rs`` pin both behaviors.

7 Non-finite input is not convergence
-------------------------------------

A gradient with a NaN component has no norm worth trusting; the crate
maps any non-finite gradient to an infinite norm in both the Euclidean
and infinity norms (``vecops::nrminf`` documents the rule as: a broken
vector is never a small one). The test
``a_nan_gradient_is_not_convergence`` holds the L-BFGS driver to it, and
the C ABI returns NaN-filled gradients on callback failure for the
same reason: a consumer must never read half-written storage as a
small residual.
