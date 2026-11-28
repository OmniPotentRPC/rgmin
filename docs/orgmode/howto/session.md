

# Hold a stateful solver

Geometry hosts such as eOn call `Optimizer::step()` once per outer
iteration. `rgmin_minimize` is a full solve. The session type
`rgmin_solver_t` (Rust `Solver`) is the matching handle: create once,
step many times, free at the end.


## Rust

```rust
use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use rgmin::{Accept, Control, Method, QnStep, Solver};

let obj = Rosenbrock::<2>::new();
let mut x = array![-1.2, 1.0];
let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2)
    .with_gtol(1e-8);
solver.set_qn_step(QnStep::Newton);
solver.set_accept(Accept::None);
let report = solver.step(&obj, &mut x).unwrap();
```

`step` is one outer iteration. `step_hess` / `step_hess_fg` take a
host Hessian. `forget` drops pair history, conjugacy, and moments.
`set_atom_maxmove` is the eOn per-atom cap. The L-BFGS default is
`LbfgsStep::Capped` at 0.2, the eOn step: no line search and no energy
test. `LbfgsStep::Wolfe` opts into the strong Wolfe search.
`set_geometry_step(true)`, or a Wolfe session with an explicit per-atom
cap, takes the geometry step instead. That step uses this cap and
ignores `set_accept`. A nominal steepest trial at least 64 times the
cap takes one finite-difference curvature probe, clipped to the
objective bounds. Ordinary starts use `H0 = 0.1 I` and a five-value
nonmonotone window. Later iterations use `s.y / y.y`. Negative
curvature clears the stored pairs. A trial is kept when the energy
rises by at most 1e-8, with at most ten halvings; otherwise a short
steepest step is taken. When `set_exp` is installed, that
preconditioner replaces the scalar seed, including those steepest
fallbacks, and the cap still limits the displacement. A HiGHS
feasible set and a non-Euclidean manifold keep the two-loop step.

`set_atom_maxmove` is the eOn per-atom clip. A fresh Cartesian
session starts with that clip at 0.2, which is ASE
`Optimizer.defaults` `maxstep` at tag 3.29.0
(<https://gitlab.com/ase/ase/-/blob/3.29.0/ase/optimize/optimize.py>)
and eOn `max_move`
(<https://henkelmanlab.org/eon/optimizer.html>; `OptimizerOptions.maxMove`
in `schema/eon_params.capnp` at `4be157d4`,
<https://github.com/TheochemUI/eOn/blob/4be157d43265af6b1140aaac0e52e42add62693f/schema/eon_params.capnp>).
The number is in the coordinate unit, so a geometry stored in
angstroms is capped at 0.2 Å. `scale_step_atom` limits the largest
per-atom displacement. It does not limit the Euclidean norm of the
whole step. A positive argument replaces the default. A non-positive
argument clears the cap, and that choice stays in force if
`set_maxmove` or `set_manifold` is called later. A positive
`Control.maxmove`, with no call to `set_atom_maxmove`, is the
Euclidean step limit and leaves the per-atom default off. A
non-Cartesian manifold has no default.

`Method::QuickMin` is the projection-then-Euler step in
[quick-min](../explanation/quickmin.md). `set_quickmin_adapt(false)`
keeps the reference fixed time step. `set_quickmin_cell(at)` gives the
atoms and the cell their own velocities. The displacement cap is the
same `set_atom_maxmove` the other methods use.

To run eOn's quick-min, or the per-atom variant, set the options after
building the session:

```rust
use rgmin::{Control, Method, QuickMinOptions, Solver};

let control = Control::default().with_istep(0.1);
let mut solver = Solver::new(Method::QuickMin, control, x.len());
solver.set_atom_maxmove(0.2);
solver.set_quickmin_options(QuickMinOptions::per_atom()); // or ::eon()
let report = solver.step(&obj, &mut x).unwrap();
```

From C, `rgmin_solver_set_quickmin_variant(s, RGMIN_QUICKMIN_PER_ATOM)`
(or `RGMIN_QUICKMIN_EON`, `RGMIN_QUICKMIN_RGMIN`) on a session created
with `RGMIN_QUICKMIN`.

`set_project_rigid` projects translations and rotations on isolated
clusters.
`set_distance_constraints` keeps atom pairs at fixed distances on
every later step; see .
With a set, L-BFGS keeps its two-loop step under `set_accept`, and
the per-atom cap scales each trial before it returns to the set.


## Stop norm

```rust
use rgmin::{Control, Method, Solver, StopNorm};

let n_atoms = 7;
let control = Control::default()
    .with_gtol(0.01)
    .with_stop_norm(StopNorm::MaxAtom);
let solver = Solver::new(Method::lbfgs(), control, 3 * n_atoms);
```

`Control::stop_norm` is the norm a session compares to `gtol` and writes to `Report::grad_norm`.
`Report::stop_norm` names it, and `set_stop_norm` replaces it on a live session.
The default is `StopNorm::Two`, so a default `Control` stops where it did before the field existed.
Only the step at which the session stops changes: every norm takes the same steps up to that point.
The one-shot drivers that run no session, such as `minimize_lbfgs`, keep `norm2(g)` and report `StopNorm::Two`.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left"><code>StopNorm</code></th>
<th scope="col" class="org-left">Value</th>
<th scope="col" class="org-left">ASE</th>
<th scope="col" class="org-left">eOn <code>convergence_metric</code></th>
<th scope="col" class="org-left">LAMMPS <code>min_modify norm</code></th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>Two</code> (default)</td>
<td class="org-left"><code>norm2(g)</code></td>
<td class="org-left">&#xa0;</td>
<td class="org-left"><code>norm</code> (default)</td>
<td class="org-left"><code>two</code> (default)</td>
</tr>

<tr>
<td class="org-left"><code>MaxAtom</code></td>
<td class="org-left">largest <code>norm2</code> over <code>(x, y, z)</code> triples</td>
<td class="org-left"><code>fmax</code></td>
<td class="org-left"><code>max_atom</code></td>
<td class="org-left"><code>max</code></td>
</tr>

<tr>
<td class="org-left"><code>Inf</code></td>
<td class="org-left">largest <code>abs(g_k)</code></td>
<td class="org-left">&#xa0;</td>
<td class="org-left"><code>max_component</code></td>
<td class="org-left"><code>inf</code></td>
</tr>

<tr>
<td class="org-left"><code>Rms</code></td>
<td class="org-left"><code>norm2(g) / sqrt(n)</code></td>
<td class="org-left">&#xa0;</td>
<td class="org-left"><code>rms</code></td>
<td class="org-left">&#xa0;</td>
</tr>
</tbody>
</table>

eOn takes `n` in `rms` as the number of free coordinates, so a host that passes only the free atoms gets the same value.
C sessions select the norm with `rgmin_solver_set_stop_norm`.
`RGMIN_STOP_NORM_RMS` is 3.


## C

```c
#include "rgmin.h"

rgmin_control_t ctrl = rgmin_control_default();
ctrl.maxiter = 200;
ctrl.gtol = 1e-8;
ctrl.memory = 8;
rgmin_solver_t *s = rgmin_solver_create(RGMIN_LBFGS, &ctrl, n);
rgmin_report_t report;
while (!done) {
    rgmin_solver_set_maxmove(s, max_move);
    rgmin_solver_set_qn_step(s, RGMIN_QN_NEWTON);
    rgmin_solver_set_accept(s, RGMIN_ACCEPT_NONE);
    rgmin_solver_step_hess_fg(s, evalgrad, hess, user, x, &report);
}
rgmin_solver_free(s);
```

Callbacks are arguments of each step, not stored on the handle.
`rgmin_solver_step_fg` / `rgmin_solver_step_hess_fg` take one fused
`(f, g)` callback so a host that already bills energy and forces
together is not asked twice.


## Setters

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Setter</th>
<th scope="col" class="org-left">Role</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>set_qn_step</code></td>
<td class="org-left">Two-loop, Newton, or RFO when a host Hessian is present</td>
</tr>

<tr>
<td class="org-left"><code>set_exp</code></td>
<td class="org-left">Packwood preconditioner from a caller neighbour list. An opt-in geometry step takes <code>H0 = P^{-1}</code></td>
</tr>

<tr>
<td class="org-left"><code>clear_exp</code></td>
<td class="org-left">Drop that preconditioner</td>
</tr>

<tr>
<td class="org-left"><code>exp_mu</code></td>
<td class="org-left">Its energy scale after estimation, or none when it is not installed</td>
</tr>

<tr>
<td class="org-left"><code>set_accept</code></td>
<td class="org-left"><code>none</code> / <code>energy</code> / <code>nonmonotone</code> / <code>step</code></td>
</tr>

<tr>
<td class="org-left"><code>set_stop_norm</code></td>
<td class="org-left">Replace <code>Control::stop_norm</code>: <code>two</code> / <code>max_atom</code> / <code>inf</code> / <code>rms</code></td>
</tr>

<tr>
<td class="org-left"><code>set_linesearch</code></td>
<td class="org-left">Line search for the line-searched arms; default strong Wolfe. The default L-BFGS step ignores it</td>
</tr>

<tr>
<td class="org-left"><code>set_lbfgs_step</code></td>
<td class="org-left"><code>Capped</code> at 0.2, the eOn step, is the default. <code>Wolfe</code> opts into the strong Wolfe search. A non-positive cap restores 0.2 and a non-finite one is an error. C: <code>rgmin_solver_set_lbfgs_capped</code>, <code>_scaled</code>, <code>_wolfe</code></td>
</tr>

<tr>
<td class="org-left"><code>set_fire_variant</code></td>
<td class="org-left">FIRE schedule; <code>Guenole2020</code> is FIRE 2.0 as published</td>
</tr>

<tr>
<td class="org-left"><code>forget_evaluation</code></td>
<td class="org-left">Re-evaluate the next start, keep the method memory</td>
</tr>

<tr>
<td class="org-left"><code>set_atom_maxmove</code></td>
<td class="org-left">Per-atom clip; Cartesian default 0.2. A positive <code>maxmove</code> is the Euclidean cap and leaves this default off. A negative <code>maxmove</code> clears every cap</td>
</tr>

<tr>
<td class="org-left"><code>set_project_rigid</code></td>
<td class="org-left">Drop rigid modes on isolated clusters</td>
</tr>

<tr>
<td class="org-left"><code>set_lbfgs_neb_guards</code></td>
<td class="org-left">L-BFGS distance, angle and curvature resets; empty-memory scale 0.01</td>
</tr>

<tr>
<td class="org-left"><code>set_extra_updates</code></td>
<td class="org-left">Accepted, no effect: a replayed newest pair leaves BFGS unchanged</td>
</tr>

<tr>
<td class="org-left"><code>set_cautious</code></td>
<td class="org-left">Li&#x2013;Fukushima pair filter</td>
</tr>

<tr>
<td class="org-left"><code>set_highs</code></td>
<td class="org-left">Feasible-set QP; returns 1 without <code>--features highs</code></td>
</tr>

<tr>
<td class="org-left"><code>set_box</code></td>
<td class="org-left">Per-coordinate box on <code>x + p</code>; NULL side unbounded</td>
</tr>

<tr>
<td class="org-left"><code>set_trust</code></td>
<td class="org-left">L<sub>inf</sub> HiGHS trust radius; non-positive clears</td>
</tr>

<tr>
<td class="org-left"><code>add_equality</code> / <code>clear_equalities</code></td>
<td class="org-left">Sparse <code>a · p = rhs</code> on the HiGHS set</td>
</tr>

<tr>
<td class="org-left"><code>set_manifold</code></td>
<td class="org-left">Embedded geometry; <code>rigid_quotient</code> for 3N clusters</td>
</tr>

<tr>
<td class="org-left"><code>set_masses</code></td>
<td class="org-left">Per-atom masses for <code>mw_rigid</code> (Page–McIver)</td>
</tr>

<tr>
<td class="org-left"><code>set_fire_preset</code></td>
<td class="org-left"><code>fire2</code> only: Guenole 2020 Table 1 parameters; restarts the integrator</td>
</tr>

<tr>
<td class="org-left"><code>set_fire_abc</code></td>
<td class="org-left"><code>fire2</code> only: ABC-FIRE bias correction with a per-component cap</td>
</tr>

<tr>
<td class="org-left"><code>set_distance_constraints</code></td>
<td class="org-left">Keep atom pairs at fixed distances with SHAKE, LINCS or SETTLE; a setting the set cannot run with is a typed error</td>
</tr>

<tr>
<td class="org-left"><code>clear_distance_constraints</code></td>
<td class="org-left">Drop the set and the method memory</td>
</tr>
</tbody>
</table>

The C entry of each setter is `rgmin_solver_<setter>`, except `set_lbfgs_step`, whose entries the table names.
`rgmin_solver_forget_evaluation` and `rgmin_solver_set_lbfgs_neb_guards`
need `abi_minor` 28.


## When to use `rgmin_minimize` instead

Scripts that want a cold-start solve to `gtol` still call
`rgmin_minimize` / `minimize_method`. Hopping chains that own the
pair history hold `Lbfgs` directly (see
).


## Strict convergence

`Control::with_require_convergence(true)` makes `Solver::run`,
`Solver::run_gauged`, a zero iteration budget, and a spent budget
return `Err(Error::NotConverged(Box<Report>))` when the gradient
norm is not below `gtol`. The boxed report is the terminal point.
`Report.converged` is `grad_norm < gtol`, the comparison every driver
uses to stop. The default is off, so the same stops stay `Ok`.

A strict run ends early as `StepTolerance` only when an accepted step
leaves the point unchanged and the next iteration would repeat it. A
rejected trust region trial or a rejected adaptive step changes the
radius or the step length, so it is not a stall and the run goes on.
A run that converges by default converges identically when strict.
