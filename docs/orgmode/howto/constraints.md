

# Keep bond lengths fixed while a session minimizes

A `DistanceConstraints` set fixes `|x_i - x_j| = d` for a list of atom pairs in packed `3 N` coordinates.
Install it on a `Solver` and every later step keeps each constraint to its tolerance.
Every method of the session runs on the set, under every `Accept` rule, except particle swarm.
The set can carry one mass per atom, and then every correction and projection uses the metric `M^{-1}`.
gives the equations, the derived tolerances and budgets, and how the solvers compare with GROMACS.


## Rust

The sample keeps a three-site water rigid, with real masses, while a harmonic pull moves it.

```rust
use ndarray::{ArrayView1, array};
use rgmin::{Accept, Control, DistanceConstraints, Method, Oracle, Solver};

// Oxygen, hydrogen, hydrogen.
let mut x = array![0.0, 0.0, 0.0, 0.9572, 0.0, 0.0, -0.24, 0.9266, 0.0];
let set = DistanceConstraints::at(vec![(0, 1), (0, 2), (1, 2)], x.view())?
    .with_masses(vec![15.999, 1.008, 1.008])?;
let target = array![0.1, 0.2, 0.0, 1.2, 0.1, 0.0, 0.0, 1.1, 0.3];
let obj = Oracle::unbounded(9, move |p: ArrayView1<f64>| {
    let d = &p - &target;
    (0.5 * d.dot(&d), d)
});
let control = Control::default().with_maxiter(200).with_gtol(1e-8);
let mut solver = Solver::new(Method::lbfgs(), control, 9);
solver.set_accept(Accept::Energy);
solver.set_distance_constraints(set)?;
for _ in 0..200 {
    let report = solver.step(&obj, &mut x)?;
    if report.grad_norm < 1e-8 {
        break;
    }
}
let set = solver.distance_constraints().unwrap();
assert!(set.excess(x.view()) <= 1.0);
```

`DistanceConstraints::new` takes the lengths instead of reading them from `x`.
Both constructors check their input.
They refuse a pair of one atom and a pair given twice.
They also refuse a length that is not positive and finite, and a rigid triangle that breaks the triangle inequality.
`excess` is the largest ratio of a constraint's relative residual `|d^2 - r^2| / (2 d^2)` to its tolerance, so at most one means every constraint is met.
`grad_norm` is the norm of the projected gradient, the force that the set leaves free.

A start off the set is put on it first, by corrections from the start itself.
A step that cannot meet the set returns `Error::Constraint`.
It leaves `x` and the method memory unchanged.


## C

```c
#include "rgmin.h"

const uint32_t pi[3] = {0, 0, 1};
const uint32_t pj[3] = {1, 2, 2};
const double masses[3] = {15.999, 1.008, 1.008};
rgmin_constraints_t *set = rgmin_constraints_create_at(pi, pj, 3, x, 9);
if (set == NULL || rgmin_constraints_set_masses(set, masses, 3) != RGMIN_SUCCESS) {
    /* rgmin_last_error() says why, rgmin_last_constraint_error() names it */
}
rgmin_control_t ctrl = rgmin_control_default();
rgmin_solver_t *s = rgmin_solver_create(RGMIN_LBFGS, &ctrl, 9);
rgmin_solver_set_accept(s, RGMIN_ACCEPT_ENERGY);
if (rgmin_solver_set_distance_constraints(s, set) != RGMIN_SUCCESS) {
    /* RGMIN_CONSTRAINT_ERROR: the session cannot carry the set */
}
rgmin_constraints_free(set);
rgmin_status_t st = rgmin_solver_step(s, eval, grad, user, xt, &report);
if (st == RGMIN_CONSTRAINT_ERROR) {
    /* x is unchanged; rgmin_last_constraint_error() says what failed */
}
```

The session copies the set, so the caller can free it right after `rgmin_solver_set_distance_constraints`.
`rgmin_solver_clear_distance_constraints` drops it.
These functions need ABI minor 31.
`xts.h` spells each one with the `xts_` prefix, and each enumerator with `XTS_`.


## C++

```cpp
#include "rgmin/optimize.hpp"

std::uint32_t const pi[3] = {0, 0, 1};
std::uint32_t const pj[3] = {1, 2, 2};
double const masses[3] = {15.999, 1.008, 1.008};
auto set = rgmin::DistanceConstraints::at(pi, pj, 3, x.data(), 9);
set.with_masses(masses, 3).with_solver(rgmin::ConstraintSolver::Auto);
rgmin::Solver solver(rgmin::Method::Lbfgs, ctrl, 9);
solver.set_accept(RGMIN_ACCEPT_ENERGY);
solver.set_distance_constraints(set);
try {
    solver.step(eval, grad, user, xt);
} catch (rgmin::ConstraintError const& e) {
    // e.kind() is the rgmin_constraint_error_t; x is unchanged
}
```

`rgmin::ConstraintError` derives from `std::runtime_error`, so code that catches that keeps working.
`xts::optimize` names the same three types.


## Choose a solver

Each block of coupled constraints is solved on its own, and a block is a connected part of the constraint graph.
A block of three atoms and three constraints is a rigid triangle, and SETTLE solves it in closed form for any masses and any triangle.
`with_settle(false)` leaves triangles to the iterative solver.
The other blocks go to the solver that `with_solver` names, the Gauss-Seidel bond solver (SHAKE) or the linear constraint solver (LINCS).

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left"><code>ConstraintSolver</code></th>
<th scope="col" class="org-left">Use</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>Auto</code> (default)</td>
<td class="org-left">LINCS where its expansion contracts at least as fast per pass as the Gauss-Seidel bound of SHAKE, SHAKE elsewhere</td>
</tr>

<tr>
<td class="org-left"><code>Shake</code></td>
<td class="org-left">Gauss-Seidel SHAKE on every block, for any coupling the projection can still solve</td>
</tr>

<tr>
<td class="org-left"><code>Lincs</code></td>
<td class="org-left">LINCS on every block; a block where the expansion diverges is <code>ConstraintError::LincsUnsafe</code></td>
</tr>
</tbody>
</table>

`with_lincs_order(Some(n))` fixes the expansion order as GROMACS `lincs-order` does.
`None`, the default, derives it per block from the coupling and the tolerance.
`with_tolerance(t)` asks for the relative residual `t`, the GROMACS `shake-tol` measure.
Zero, the default, is the rounding floor of each constraint, and a smaller `t` is raised to that floor.


## Choose masses

GROMACS energy minimization sets every mass to one.
rgmin uses unit masses until `with_masses` is called.
Masses change the path to the minimum, and the constrained minimum depends on the energy and the set alone.
With real masses a correction moves a light atom more than a heavy one, as in molecular dynamics.


## What a set refuses

A set refuses each setting below with `ConstraintError::Unsupported`.
The refusal comes from `set_distance_constraints`, or from the next step when the setting changes later.

-   Manifolds other than Euclidean coordinates: the set defines its own manifold.
-   The exponential preconditioner: its metric would replace the mass metric of the projector.
-   The HiGHS feasible-set step: it solves a quadratic program over linear equalities and has no place for the distance constraints.
-   A periodic cell: the session carries no lattice vectors and so cannot form a minimum image.
-   Particle swarm: it samples the whole box.
-   Rigid-mode projection while a coordinate sits on its bound.

Coordinate bounds act only on atoms outside every pair.
A constrained coordinate is never clipped, and a trial that takes a constrained atom to its bound is `ConstraintError::Bound`.


## Failures

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Rust <code>ConstraintError</code></th>
<th scope="col" class="org-left">C <code>rgmin_constraint_error_t</code></th>
<th scope="col" class="org-left">Cause</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>Malformed</code></td>
<td class="org-left"><code>RGMIN_CONSTRAINT_MALFORMED</code></td>
<td class="org-left">Pairs, lengths or options that describe no set</td>
</tr>

<tr>
<td class="org-left"><code>OutOfRange</code></td>
<td class="org-left"><code>RGMIN_CONSTRAINT_OUT_OF_RANGE</code></td>
<td class="org-left">A pair names an atom past the end of <code>x</code></td>
</tr>

<tr>
<td class="org-left"><code>Mass</code></td>
<td class="org-left"><code>RGMIN_CONSTRAINT_MASS</code></td>
<td class="org-left">A mass that is not positive and finite</td>
</tr>

<tr>
<td class="org-left"><code>Unsupported</code></td>
<td class="org-left"><code>RGMIN_CONSTRAINT_UNSUPPORTED</code></td>
<td class="org-left">A session setting from the list above</td>
</tr>

<tr>
<td class="org-left"><code>NotConverged</code></td>
<td class="org-left"><code>RGMIN_CONSTRAINT_NOT_CONVERGED</code></td>
<td class="org-left">SHAKE, LINCS or the tangent solve spent its derived budget</td>
</tr>

<tr>
<td class="org-left"><code>Reversed</code></td>
<td class="org-left"><code>RGMIN_CONSTRAINT_REVERSED</code></td>
<td class="org-left">A bond turned too far from its direction at the anchor</td>
</tr>

<tr>
<td class="org-left"><code>Dependent</code></td>
<td class="org-left"><code>RGMIN_CONSTRAINT_DEPENDENT</code></td>
<td class="org-left">The constraints of a block are linearly dependent at the anchor</td>
</tr>

<tr>
<td class="org-left"><code>LincsUnsafe</code></td>
<td class="org-left"><code>RGMIN_CONSTRAINT_LINCS_UNSAFE</code></td>
<td class="org-left"><code>Lincs</code> was asked for where its expansion diverges</td>
</tr>

<tr>
<td class="org-left"><code>SettleDegenerate</code></td>
<td class="org-left"><code>RGMIN_CONSTRAINT_SETTLE_DEGENERATE</code></td>
<td class="org-left">A trial moved a rigid triangle beyond any SETTLE solution</td>
</tr>

<tr>
<td class="org-left"><code>Bound</code></td>
<td class="org-left"><code>RGMIN_CONSTRAINT_BOUND</code></td>
<td class="org-left">A constrained atom reached its coordinate bound</td>
</tr>

<tr>
<td class="org-left"><code>NonFinite</code></td>
<td class="org-left"><code>RGMIN_CONSTRAINT_NON_FINITE</code></td>
<td class="org-left">A coordinate, mass or residual is not finite</td>
</tr>
</tbody>
</table>

Under `Accept::Energy` and `Accept::Nonmonotone` a trial that the set cannot take counts as refused, and the search tries a shorter one.
The error reaches the caller only when no trial could be placed.
The dogleg and Newton-CG trust regions count such a trial as a rejected model step and shrink the radius.
Under `Accept::None` there is no shorter step to try, so the error reaches the caller.
A smaller `istep`, `maxmove` or per-atom cap keeps trials close enough to the set.
