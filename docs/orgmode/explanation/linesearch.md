
# Table of Contents

1.  [The two Wolfe conditions](#orgbe0b808)
2.  [The bracket and the zoom](#org8e012fd)
3.  [The cubic Hermite proposal](#org7a5d4ac)
4.  [The guard is the termination proof](#orgfd8fa33)
5.  [When rounding closes the bracket](#org19f1589)
6.  [When the interpolant is not used](#orgcbfe966)
7.  [What the accept layer adds](#org9ddddcb)
8.  [A search the caller evaluates](#orgd564012)

Every solver in this crate that follows a descent direction `d` from a
point `x` reduces the problem to one dimension. Write the restriction
of the objective along the ray as phi(a) = f(x + a d), so
phi'(0) = g . d is negative for any descent direction. The line
search's job is to pick a step `a` that provably makes progress
without spending the caller's evaluation budget, and every claim below
is about where those two guarantees come from.


<a id="orgbe0b808"></a>

# The two Wolfe conditions

Sufficient decrease (Armijo) demands phi(a) <= phi(0) + c1 a phi'(0)
with 0 < c1 < 1: the step must capture at least a `c1` fraction of the
decrease the initial slope promised. Because phi'(0) < 0 and a > 0,
the right-hand side sits strictly below phi(0), so an accepted point is
a real improvement, not a rounding story. That one-line consequence is
the Lean theorem `Rgmin.armijo_strict` in
[proofs/lean/Rgmin/Zoom.lean](https://github.com/OmniPotentRPC/rgmin/blob/develop/proofs/lean/Rgmin/Zoom.lean).

The curvature condition demands abs(phi'(a)) <= c2 abs(phi'(0)) with
c1 < c2 < 1: the slope must have flattened enough that the quasi-Newton
update built from this step carries curvature information. Nocedal and
Wright's algorithms 3.5 and 3.6 (the bracket and the zoom) find a point
satisfying both.

Near a minimum the decrease test can fail on rounding alone. `Lbfgs::approx_wolfe`,
zero by default, is a fraction of abs(f) at the start of a strong Wolfe search.
A positive value accepts a trial inside that band when the strong curvature
test holds (Hager and Zhang 2005, stop T2) and places a trial inside the band
that fails the curvature test by the sign of its slope. The eOn cap does not
consult it. `LbfgsStep::Wolfe` and `Lbfgs::minimize_box` do. `src/linesearch/known.rs` implements them from a
point whose value and gradient the caller already holds: the start is
never re-evaluated, the accepted point comes back with its gradient,
and a displacement cap bounds every trial (`alpha_max`).
`src/linesearch/zoom.rs` keeps a standalone bisection zoom.


<a id="org8e012fd"></a>

# The bracket and the zoom

The outer loop (`known.rs::wolfe`) expands a trial step, within
More-Thuente's bounds of 1.1 to 4 times the last increment, until it either
satisfies both conditions, overshoots (value above the Armijo line, or
above the previous trial), or the slope turns nonnegative. Overshoot
and slope-reversal both certify that a Wolfe point lies between the
last two trials: that pair becomes the bracket `[lo, hi]`, ordered so
`lo` carries the lower value.

The zoom then shrinks the bracket. Each iteration proposes a trial
inside `[lo, hi]`, evaluates it, and replaces whichever end the Wolfe
tests disqualify. The proposal is where the evaluation budget is won
or lost, and the guard is where termination is proved.


<a id="org7a5d4ac"></a>

# The cubic Hermite proposal

At the two ends the zoom holds four numbers: phi(lo), phi'(lo),
phi(hi), phi'(hi). There is exactly one cubic through both points with
both slopes, and its interior minimizer has the closed form (Nocedal
and Wright eq. 3.59)

d1 = phi'(lo) + phi'(hi) - 3 (phi(lo) - phi(hi)) / (lo - hi)
d2 = sign(hi - lo) sqrt(d1<sup>2</sup> - phi'(lo) phi'(hi))
t  = hi - (hi - lo) (phi'(hi) + d2 - d1) / (phi'(hi) - phi'(lo) + 2 d2)

On a well-behaved function the cubic model is third-order accurate, so
`t` lands close to the true minimizer and the zoom converges in a
handful of evaluations; the measured effect in anneal was 46 to 43
force calls per hop across the whole hopping battery, purely from this
proposal replacing bisection. When the discriminant goes negative or a
slope is non-finite the code falls back to the quadratic through
phi(lo), phi'(lo), phi(hi), and below that to bisection: the ladder
degrades in accuracy, never in safety.


<a id="orgfd8fa33"></a>

# The guard is the termination proof

A cubic through awkward data can propose a point arbitrarily close to
either end, and a zoom that accepts such proposals can stall. The code
clamps every proposal into the middle eighty percent of the bracket:
no closer to an end than a tenth of the width. Two exact rational
facts follow, proved in
[Zoom.lean](https://github.com/OmniPotentRPC/rgmin/blob/develop/proofs/lean/Rgmin/Zoom.lean):

-   `Rgmin.guarded_above` / `Rgmin.guarded_below`: the clamp really does
    land in the interior band.
-   `Rgmin.zoom_shrinks`: whichever end the trial replaces, the surviving
    bracket is at most `9/10` of the old width.

The width therefore decays geometrically (`Rgmin.width_envelope` is
the bound, `width_envelope_tendsto` takes it to zero in the limit),
and the evaluation budget of the zoom is logarithmic in the
requested tolerance regardless of what the cubic does. The Rust test
`the_cubic_zoom_keeps_the_evaluation_budget` in
`tests/lbfgs_state.rs` holds the crate to that budget on a quadratic
bowl.


<a id="org19f1589"></a>

# When rounding closes the bracket

Geometric contraction is not useful after floating-point arithmetic
can no longer distinguish the bracket. Following the relative-width
warning in More and Thuente's `dcsrch`
([10.1145/192115.192132](https://doi.org/10.1145/192115.192132)), Wolfe zoom stops when either the bracket
width is at most eight machine epsilons times its endpoint scale or a
sequence of eight trial values differ from the Armijo endpoint by
at most eight epsilons times
`max(abs(f_trial), abs(f_lo))` while Armijo fails. One unresolved
trial is not enough: an ordinary Wolfe search can cross a short run of
rounded values before finding curvature. A trial derivative that
crosses zero and grows past thirty-two times the initial descent slope
also keeps the bracket active; its sharp transition still carries
resolvable information. The named factors allow endpoint arithmetic
and one subtraction to round.

The reusable line search and the dedicated strict `Lbfgs::minimize`
Wolfe loop both wait for the sequence and honor the slope crossing:
callers may have non-smooth objectives, and a narrow basin can sit
past a run of rounded values. When the L-BFGS loop does stop, it keeps
its pairs instead of letting its fallback discard them and repeat a
full search from steepest descent. An explicit approximate-Wolfe
accuracy window remains the acceptance policy and does not take this
strict-value stop.

The search returns the lowest measured Armijo decrease, when one
exists, together with `LineSearchStopReason::RoundingBlocked`. If no
trial has a resolved decrease, the start remains the answer. An
L-BFGS session reports `StopReason::LineSearchRounding` and ends there;
it does not discard the curvature pairs and spend another Wolfe
budget along steepest descent. The rule depends only on the
one-dimensional step, value, and slope, so the line-search API remains
independent of the objective's physical system.


<a id="orgcbfe966"></a>

# When the interpolant is not used

The cubic and the quadratic model the objective on the bracket
$[\ell, h]$ from $\varphi(\ell)$, $\varphi'(\ell)$, $\varphi(h)$ and
$\varphi'(h)$. They carry information about the minimizer only when
$\varphi(h)$ is on the scale of the data at $\ell$. The zoom
interpolates when

$$|\varphi(h) - \varphi(\ell)| \le 10^3 \left(1 + |\varphi(\ell)| + |\varphi'(\ell)|\,(h - \ell)\right),$$

with $\varphi(h)$ finite, and takes the midpoint otherwise. The right
side adds the value magnitude to the change a model with unit
curvature predicts from the slope over the bracket; the factor
$10^3$ admits curvature three orders above that. A first trial worth
$4\cdot 10^{45}$ against $\varphi(\ell) = -24.6$ and
$\varphi'(\ell) \approx -5\cdot 10^3$ fails the test by forty
orders, and the fit through it returns the clamp of the bracket
rather than a point chosen by the objective. Bisection halves the
bracket until the values become comparable, after which the
interpolant resumes.


<a id="org9ddddcb"></a>

# What the accept layer adds

Above the line search, `src/accept.rs` offers three policies: take the
clipped step (one oracle call), refuse any energy rise (`Accept::Energy`,
up to ten halvings), or the Grippo-Lampariello-Lucidi nonmonotone
window. The fallback steepest-descent probe inside the energy policies
faces the same energy test as the step it replaces: a fallback that
also fails reports the position unmoved, and the caller's stall
machinery owns what happens next. The test
`accept_energy_uphill_does_not_take_ten_steepest_retries` pins the
oracle count of that path.


<a id="orgd564012"></a>

# A search the caller evaluates

`SuppliedSearch` takes the energy and gradient the caller already
holds. `offer` either names the next trial or returns the accepted
step. It does not call an objective, so a host that counts its own
evaluations (the C entry `rgmin_linesearch`, the Python class
`rgmin.LineSearch`) does not gain a second call inside
`rgmin_solver_step`. The trials are the ones `LineSearch::search`
makes on the same answers. `examples/tutorial_linesearch.rs` is that
comparison, and `tests/supplied_linesearch.rs` checks Wolfe,
backtracking, Goldstein and Brent one trial at a time.
