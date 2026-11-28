
# Table of Contents

1.  [The secant condition](#orgbe0b808)
2.  [The two-loop recursion](#org8e012fd)
3.  [The geometry step](#org7a5d4ac)
4.  [The capped step](#orgfd8fa33)
5.  [Stale pairs and the cautious update](#org19f1589)
6.  [Bounds: the generalized Cauchy point](#orgcbfe966)
7.  [Session hooks: forget, watched, recognized](#org9ddddcb)
8.  [Non-finite input is not convergence](#orgd564012)

Newton's method prices a step at one Hessian solve; the secant family
refuses to pay it. This page derives what the crate's quasi-Newton
solvers (`src/qn.rs`, `src/lbfgs.rs`) actually compute, and why the
warm-restart and recognition hooks on the session are correctness-
preserving.


<a id="orgbe0b808"></a>

# The secant condition

After a step `s = x_new - x_old` with gradient change
`y = g_new - g_old`, any model Hessian `B` worth keeping should
reproduce the observed curvature: B s = y. In inverse form, with
H = B<sup>-1</sup>, the condition reads H y = s. One equation cannot determine a
matrix, so each method adds a minimality principle. BFGS chooses the
inverse update closest to the old `H` in a weighted Frobenius norm
subject to the secant condition and symmetry, which lands on

H<sub>new</sub> = (I - rho s y<sup>T</sup>) H (I - rho y s<sup>T</sup>) + rho s s<sup>T</sup>,
rho   = 1 / (y . s).

The update preserves positive definiteness exactly when the curvature
pair condition y . s > 0 holds, which the strong Wolfe curvature
condition guarantees at an accepted line-search point. The default
step does not search the line. It is the eOn cap: the first step
probes curvature, and a proposal at least as long as the cap clears
the memory. `LbfgsStep::Wolfe` is the opt-in search that enforces the
condition before the step is accepted. `LbfgsStep::Scaled` shortens
the two-loop step and keeps the pair.


<a id="org8e012fd"></a>

# The two-loop recursion

Storing `H` costs n<sup>2</sup>; limited-memory BFGS stores the last `m` pairs
(s<sub>i</sub>, y<sub>i</sub>) and applies the update implicitly. Unrolling the BFGS
formula `m` times around a seed matrix H0 gives the two-loop
recursion (Nocedal and Wright 7.4):

q = g
for i = m-1 .. 0:   alpha<sub>i</sub> = rho<sub>i</sub> (s<sub>i</sub> . q);  q = q - alpha<sub>i</sub> y<sub>i</sub>
r = H0 q
for i = 0 .. m-1:   beta = rho<sub>i</sub> (y<sub>i</sub> . r);     r = r + (alpha<sub>i</sub> - beta) s<sub>i</sub>
direction = -r

with the standard scalar seed H0 = (s . y) / (y . y) I from the most
recent pair, a one-parameter fit of the average curvature along the
last step; positivity of that scalar is again the pair condition.
An installed exponential preconditioner replaces that scalar.
The middle product of the two-loop solves with the preconditioner,
so the initial inverse Hessian is taken from it: `H0 = P^{-1}`.
Packwood, Kermode, Mones, Bernstein, Woolley, Gould, Ortner, and
Csányi, *A universal preconditioner for simulating condensed phase
materials*, J. Chem. Phys. 144, 164109 (2016)
([10.1063/1.4947024](https://doi.org/10.1063/1.4947024);
).
A dense pair or Lindh matrix passed into the two-loop still wins.
With no preconditioner the scalar seed above is unchanged.
Every dot and axpy in the crate's two-loop routes through the
`vecops` seam, so the recursion is written once and the storage
backend (serial, `par`, and by design a later device backend) is a
detail behind four functions.


<a id="org7a5d4ac"></a>

# The geometry step

This step runs when the caller has opted into `LbfgsStep::Wolfe` and
also set a per-atom cap (`set_atom_maxmove`). It ignores `set_accept`.
The default session step is the line-search-free cap, not this
geometry step. A HiGHS feasible set and a non-Euclidean manifold
stay on their own paths.

The initial scale is selected from the size of the nominal steepest
trial. A trial that exceeds the per-atom cap by at least 64 times
receives one directional-curvature probe; ordinary trials start from
`H0 = 0.1 I` without spending that force call. The probe is
`x + 0.01 u`, with `u` the unit force, clipped to the objective
bounds before it is evaluated. The quotient divides by the
displacement that remains along `u`. A probe the box reduces to
nothing, or a non-positive result, clears the memory and the step
is steepest descent scaled to the cap. Later iterations set
`H0 = (s . y) / (y . y)` when both dots are positive. Negative
curvature clears the stored pairs and uses that same steepest step.
The stiffness probe uses monotone energy acceptance. An ordinary
start admits an energy below the largest of the last five accepted
values.

An installed exponential preconditioner replaces that scalar seed.
The two-loop applies `P^{-1}` in its place, and a reset that would
have stepped along `h0` times the force steps along `P^{-1}` times
the force. The negative-curvature step and the short fallback after
ten halvings keep the coefficients 1000 and one tenth, applied to
`P^{-1}` times the force. A failed solve with `P` falls back to the
scalar seed. The per-atom cap still limits the displacement.

An uncapped direction longer than the cap, or one that opposes the
force, also clears the memory and replaces the direction with that
capped seed. The trial is kept when the energy rises by at most
1e-8. The step is halved at most ten times. If none of those trials
pass, the memory is cleared and a short steepest step, one tenth of
the preconditioned force and then the cap, is taken.


<a id="orgfd8fa33"></a>

# The capped step

`LbfgsStep::Capped` is the default. `LbfgsStep::Wolfe` opts into
the line search. `set_atom_maxmove` replaces the cap while the
step is capped. The geometry step above runs when the caller has
opted into `Wolfe` and also set a per-atom cap.

The direction is the same two-loop. The initial inverse Hessian is
the Nocedal-Wright scale `(s · y) / (y · y)` on the latest stored
pair (equation 7.20)
.
An installed exponential preconditioner still replaces that scalar.
The step is then shortened so the longest per-atom displacement is
at most the caller's cap. The cap is the only length in the rule.
An empty memory has no curvature yet, so the step follows the force
and has that length. A direction that is not a descent direction
clears the memory and does the same. A pair with `s · y` below the
usual relative-curvature gate is not stored. A pair that passes the
gate is kept when the step was shortened: the cap changes the
length, and the stored pairs still describe the curvature.

One evaluation is taken at the trial. A non-finite quasi-Newton
trial is retried once along the force. A second non-finite trial
leaves the point and the pairs as they were. `Lbfgs::ftol`, when
positive, still ends the relaxation after a step that decreases
the value by little.


<a id="org19f1589"></a>

# Stale pairs and the cautious update

A pair with y . s <= 0 would corrupt the inverse model, so the update
skips it; the `cautious` knob (Li-Fukushima) additionally requires
y . s >= eps norm(s)<sup>alpha</sup> before a pair enters the history, which
keeps far-from-quadratic steps from polluting the memory. Skipping is
always sound: the two-loop over any subset of valid pairs is the BFGS
inverse built from exactly those pairs.


<a id="orgcbfe966"></a>

# Bounds: the generalized Cauchy point

`Lbfgs::minimize_box` keeps the same pairs and replaces the two-loop direction with the step of limited-memory Broyden-Fletcher-Goldfarb-Shanno with bounds (L-BFGS-B).
Byrd, Lu, Nocedal and Zhu, *A limited memory algorithm for bound constrained optimization*, Society for Industrial and Applied Mathematics (SIAM) Journal on Scientific Computing 16, 1190 (1995) ([10.1137/0916069](https://doi.org/10.1137/0916069); ).
The step needs `B` rather than `H`, so it uses the compact form of Byrd, Nocedal and Schnabel, Mathematical Programming 63, 129 (1994) ([10.1007/BF01582063](https://doi.org/10.1007/BF01582063); ):

    B = theta I - W M W^T,   W = [Y, theta S],
    M^-1 = [[-D, L^T], [L, theta S^T S]],

with theta = (y<sup>T</sup> y) / (s<sup>T</sup> y) from the newest pair, `D` the diagonal of `s_i^T y_i`, and `L` the strictly lower triangle of S<sup>T</sup> Y.
Both forms describe the same matrix when the two-loop recursion seeds with 1/theta.

The bounded step built on this matrix has two parts.
The generalized Cauchy point is the first local minimizer of the quadratic model along the projected steepest-descent path clip(x - t g, lo, hi).
That path is piecewise linear and bends where a coordinate meets its bound.
The search visits the bends in order of t and updates the slope and curvature of the model in O(m<sup>2</sup>) work per bend (algorithm CP of the paper).
The coordinates that the Cauchy point leaves on a bound stay there.
One reduced Newton step moves the other coordinates.
It costs a 2m by 2m solve (section 5.1).
If that point leaves the box, it is projected back.
The projection is kept when it is still a descent direction; otherwise the step is cut back to the box.
That test is the remark of Morales and Nocedal, Association for Computing Machinery (ACM) Transactions on Mathematical Software 38, 1 (2011) ([10.1145/2049662.2049669](https://doi.org/10.1145/2049662.2049669); ).
The strong Wolfe search runs along the segment from x to that point and never steps outside the box.
The Fortran 77 code of Zhu, Byrd, Lu and Nocedal, ACM Transactions on Mathematical Software 23, 550 (1997) ([10.1145/279232.279236](https://doi.org/10.1145/279232.279236); ), as translated to C in SciPy 1.18.1 (commit e4e854ea, `scipy/optimize/src/lbfgsb.c`, routines `cauchy` and `subsm`), is the reference for the details.

With no finite bound the Cauchy point meets no bend and the reduced step covers every coordinate.
The end point is then x - H g, the two-loop direction up to rounding, and the unit test `unbounded_direction_is_the_two_loop_direction` checks that to 1e-12.
`Lbfgs::minimize` stays on its own step rule. The default is the eOn cap.
`LbfgsStep::Wolfe` opts into the search `minimize_box` uses.


<a id="org9ddddcb"></a>

# Session hooks: forget, watched, recognized

`Solver` (and anneal's `WarmLbfgs` face over it) exposes three
lifecycle hooks whose safety arguments are one sentence each:

-   `forget()` clears the pair history. The next step is plain steepest
    descent scaled by the seed; every stored invariant is vacuous.
-   `minimize_watched` runs a caller predicate per accepted iterate and
    stops early when asked. Stopping early returns an accepted iterate,
    so every property of the trajectory up to that point stands.
-   `minimize_recognized` additionally lets the caller substitute a
    known result (energy and coordinates) when it recognizes the basin
    mid-descent, returning a flag that says the substitution happened.
    The optimizer never manufactures the substitute; it only carries the
    caller's answer out, so the correctness burden sits exactly where
    the knowledge is. The identity and refund tests in
    `tests/lbfgs_state.rs` pin both behaviors.


<a id="orgd564012"></a>

# Non-finite input is not convergence

A gradient with a NaN component has no norm worth trusting; the crate
maps any non-finite gradient to an infinite norm in both the Euclidean
and infinity norms (`vecops::nrminf` documents the rule as: a broken
vector is never a small one). The test
`a_nan_gradient_is_not_convergence` holds the L-BFGS driver to it, and
the C ABI returns NaN-filled gradients on callback failure for the
same reason: a consumer must never read half-written storage as a
small residual.
