
# Table of Contents

1.  [GROMACS L-BFGS](#org19f1589)
    1.  [Algorithm](#orgbe0b808)
    2.  [Stops and reporting](#org8e012fd)
    3.  [Exactness and differences](#org7a5d4ac)
    4.  [GROMACS comparison](#orgfd8fa33)



<a id="org19f1589"></a>

# GROMACS L-BFGS

`Method::GromacsLbfgs` follows `do_lbfgs` in the tagged GROMACS 2023.3
source,
[minimize.cpp, lines 1978&#x2013;2797](https://github.com/gromacs/gromacs/blob/v2023.3/src/gromacs/mdrun/minimize.cpp#L1978-L2797)
(`src/gromacs/mdrun/minimize.cpp`). The MDP defaults come from the same
tag's MDP option documentation for
[nsteps](https://github.com/gromacs/gromacs/blob/v2023.3/docs/user-guide/mdp-options.rst#L206-L210),
[emtol and emstep](https://github.com/gromacs/gromacs/blob/v2023.3/docs/user-guide/mdp-options.rst#L328-L337), and
[nbfgscorr](https://github.com/gromacs/gromacs/blob/v2023.3/docs/user-guide/mdp-options.rst#L345-L349).


<a id="orgbe0b808"></a>

## Algorithm

The first direction is the force, $\mathbf p_0=-\mathbf g_0$, and the
opening scalar step is $1/\|\mathbf F_0\|_2$. Before evaluation, GROMACS
repeatedly multiplies it by 0.1 until the largest positive coordinate
displacement is no greater than `emstep`. Although the MDP manual calls
`emstep` the "initial step-size", `do_lbfgs` uses it as this displacement
ceiling (lines 2348&#x2013;2374); rgmin follows the source.

An opening point $C$ is accepted immediately when its energy is lower
than point $A$, or when its directional derivative is still negative
and its energy rise is below
$\sqrt{\epsilon}\lvert E_A\rvert$. Otherwise the search brackets between
$A$ and $C$. It interpolates the derivative zero when the opening
derivatives at $A$ and $C$ have opposite signs, bisects otherwise, and
stops at the
first point below both endpoints. The search makes at most 20 bracket
trials (lines 2400&#x2013;2557).

For an accepted displacement $\mathbf s$ and gradient change
$\mathbf y$, rgmin keeps at most `nbfgscorr` pairs and uses the same
two-loop recursion and scalar initial inverse Hessian
$H_0=(\mathbf s^T\mathbf y)/(\mathbf y^T\mathbf y) I$ as lines
2559&#x2013;2663. The default correction count is 10.

If the line search cannot resolve a lower energy, nonempty correction
memory is cleared and the next direction is the force. That clear drops
every stored pair. GROMACS leaves the ring index in place; the
difference is documented below. Failure with empty memory stops at
machine precision (lines 2510&#x2013;2534). Every accepted step resets the
opening scalar step to 1 (line 2725).
Calling `Solver::rebase`, `Solver::forget_evaluation`, or supplying
externally moved coordinates retains the correction pairs and rebuilds
the two-loop direction at the new point with opening step
$1/\|\mathbf F\|_2$.


<a id="org8e012fd"></a>

## Stops and reporting

`emtol` is the largest per-atom force norm, so the default stop norm is
`StopNorm::MaxAtom`. `nsteps` maps to `Control::maxiter` and `-1` maps
to no Python iteration limit. Reports expose outer iterations and
all objective/force evaluations, including line-search trials.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">condition</th>
<th scope="col" class="org-left">status</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">\(F_{\max}<\texttt{emtol}\)</td>
<td class="org-left"><code>Tolerance</code></td>
</tr>

<tr>
<td class="org-left"><code>nsteps</code> exhausted</td>
<td class="org-left"><code>Budget</code></td>
</tr>

<tr>
<td class="org-left">line step below the relative-coordinate precision threshold, or an empty-memory search cannot lower energy</td>
<td class="org-left"><code>MachinePrecision</code></td>
</tr>

<tr>
<td class="org-left">callback produces a non-finite value or gradient</td>
<td class="org-left"><code>Error::Oracle</code></td>
</tr>
</tbody>
</table>


<a id="org7a5d4ac"></a>

## Exactness and differences

For an unconstrained Euclidean objective without active coordinate
bounds, no frozen coordinates, and no line-search reset, the correction
recursion, displacement check, line search, and force stop reproduce
GROMACS 2023.3 `do_lbfgs`. `do_lbfgs` is the same algorithm in the tagged
2023.3 and 2025.4 sources. The following differences are deliberate:

-   After a line search that cannot lower the energy, GROMACS sets
    `ncorr = 0` and leaves the ring index `point` unchanged (lines
    2524&#x2013;2533). The two-loop recursion then wraps a negative index to
    `ncorr - 1`, not to `nbfgscorr - 1` (lines 2600&#x2013;2609). Once new
    corrections fill past the end of the buffer, that walk reads pairs
    stored before the reset. rgmin clears the correction deque in
    `restart_steepest`, and `apply_pairs` walks only pairs stored since
    that clear.
-   rgmin leaves the frozen-coordinate force in the L-BFGS direction, the
    stored pairs, and the curvature, so a frozen atom moves on the first
    iteration. `project_vec` removes only rigid modes that already vanish
    on the frozen coordinates. GROMACS writes zero into the stored search
    direction on frozen components (lines 2205&#x2013;2216 and 2654&#x2013;2663) but
    keeps the raw force in `p`, `dg`, and `dgdg` (lines 2570&#x2013;2598).
    `get_f_norm_max` excludes frozen components from `fnorm` and `fmax`
    (lines 279&#x2013;298). A GROMACS reset copies the raw force, frozen
    components included, into the next direction (line 2529). The two
    trajectories therefore differ from iteration 1.
-   GROMACS refuses L-BFGS with domain decomposition, multiple ranks, or
    constraints (lines 1993&#x2013;2008). rgmin has no rank/domain-decomposition
    state and refuses holonomic distance constraints. Frozen linear
    coordinate masks stay available, with the force treatment above.
-   rgmin uses binary64 `f64::EPSILON`. A single-precision GROMACS build
    uses its larger `GMX_REAL_EPS`.
-   GROMACS uses 1 nm as the characteristic coordinate in its
    machine-precision step test (lines 2289&#x2013;2307). The unit-agnostic Rust
    API uses 1 in the objective's coordinate unit. Consequently the Python
    Å mapping can stop at a slightly different last representable step.
-   rgmin checks `emtol` at the point supplied to a session before taking
    another step. `do_lbfgs` checks it after an accepted step.
-   The existing Python MDP helper defaults `nsteps` to `-1` (unlimited);
    the GROMACS 2023.3 MDP default is 0. An explicit `nsteps` value maps
    directly to the rgmin iteration budget. GROMACS `do_lbfgs` uses an
    inclusive `step <= nsteps` loop and can attempt one additional outer
    iteration; rgmin keeps the exact-budget convention shared by its
    other GROMACS front ends.
-   `do_lbfgs` divides by curvature without validating it. rgmin returns
    `Error::GromacsLbfgs` for zero or non-finite curvature instead of
    propagating a non-finite direction.
-   Active objective bounds or non-Euclidean retractions alter trial
    coordinates, so their iterates are extensions rather than an exact
    GROMACS trajectory.

The C ABI selector is `RGMIN_GROMACS_LBFGS`. ABI minor 67 adds it and
`rgmin_solver_set_gromacs_lbfgs`.


<a id="orgfd8fa33"></a>

## GROMACS comparison

The seven-atom argon fixture in `tests/data/gromacs_steep_argon.*` was
minimized with double-precision `gmx_d` from Ubuntu's `gromacs` 2023.3
package (`2023.3-Ubuntu_2023.3_1ubuntu3`, SSE4.1, one thread). The run
uses `integrator = l-bfgs`, `emtol = 1e-8`, `emstep = 0.02`,
`nbfgscorr = 10`, and `nsteps = 12`. GROMACS treats that limit as
inclusive (`step <= nsteps`), so the footer is the state after 13
accepted iterations:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">solver</th>
<th scope="col" class="org-right">accepted steps</th>
<th scope="col" class="org-right">force evaluations</th>
<th scope="col" class="org-right">final energy</th>
<th scope="col" class="org-right">final Fmax</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>gmx_d</code> 2023.3 double</td>
<td class="org-right">13</td>
<td class="org-right">15</td>
<td class="org-right">-1.25489377088304e+01</td>
<td class="org-right">7.27914686102068e-08</td>
</tr>

<tr>
<td class="org-left">rgmin <code>GromacsLbfgs</code>, <code>maxiter = 13</code></td>
<td class="org-right">13</td>
<td class="org-right">15</td>
<td class="org-right">-1.25489377088303e+01</td>
<td class="org-right">7.28008804951526e-08</td>
</tr>
</tbody>
</table>

The evaluation counts match. The energies agree to `1e-11`, which is
tighter than the change of the last accepted step. `Fmax` differs in
the last digits because a coordinate difference far below the printed
energy moves the force. `tests/data/gromacs_lbfgs_argon.txt`, produced
by `tests/data/gromacs_lbfgs_argon_driver.py`, records the same footer
energy, maximum force, and evaluation count at every inclusive
`nsteps` for several `nbfgscorr` and `emstep` settings.
`tests/gromacs_lbfgs.rs` asserts that file. A single-precision GROMACS
build still uses a larger `GMX_REAL_EPS` than `f64::EPSILON`, so this
table does not describe that build.
