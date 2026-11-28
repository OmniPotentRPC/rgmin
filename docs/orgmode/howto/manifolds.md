

# Retract onto an embedded manifold

The default geometry is ambient Euclidean space. A session can
retract each trial point onto an embedded Riemannian manifold
through `Solver::set_manifold` / `rgmin_solver_set_manifold`.

The contract is manopt<sub>cpp</sub>'s `AbstractManifold` on a rank-1 f64
vector: `project`, `retract`, `transport`. Citations:

-   Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix
    Manifolds*
    ([10.1515/9781400830244](https://doi.org/10.1515/9781400830244)).
-   Boumal, *An Introduction to Optimization on Smooth Manifolds*
    ([10.1017/9781009166164](https://doi.org/10.1017/9781009166164)).
-   Boumal, Mishra, Absil, Sepulchre, Manopt
    ([JMLR 15, 1455 (2014)](https://jmlr.org/papers/v15/boumal14a.html)).
-   Huang, Absil, Gallivan, Hand, ROPTLIB
    ([10.1145/3218822](https://doi.org/10.1145/3218822)).


## Tokens

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Token</th>
<th scope="col" class="org-left">Packing</th>
<th scope="col" class="org-left">Retraction</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>Euclidean</code></td>
<td class="org-left">length <code>n</code></td>
<td class="org-left"><code>x + v</code></td>
</tr>

<tr>
<td class="org-left"><code>RigidQuotient</code></td>
<td class="org-left">3N Cartesians, N &gt;= 2</td>
<td class="org-left">horizontal lift <code>x + v</code></td>
</tr>

<tr>
<td class="org-left"><code>MwRigid</code></td>
<td class="org-left">same; masses on the session</td>
<td class="org-left">same; Eckart inner product</td>
</tr>

<tr>
<td class="org-left"><code>Sphere</code></td>
<td class="org-left">unit vector, length <code>n</code></td>
<td class="org-left"><code>(x+v)/norm(x+v)</code></td>
</tr>

<tr>
<td class="org-left"><code>So3</code></td>
<td class="org-left">row-major <code>R</code>, length 9</td>
<td class="org-left">QR with positive diagonal</td>
</tr>

<tr>
<td class="org-left"><code>Stiefel</code></td>
<td class="org-left"><code>St(n,1)</code>: same as the sphere</td>
<td class="org-left">same as the sphere</td>
</tr>

<tr>
<td class="org-left"><code>StiefelP { n, p }</code></td>
<td class="org-left">column-major <code>n x p</code>, length <code>n p</code></td>
<td class="org-left">thin QR of <code>X + U</code></td>
</tr>

<tr>
<td class="org-left"><code>Se3</code></td>
<td class="org-left">row-major <code>R</code> then <code>t</code>, length 12</td>
<td class="org-left">SO(3) on the rotation, Euclidean on <code>t</code></td>
</tr>

<tr>
<td class="org-left"><code>Oblique { n, m }</code></td>
<td class="org-left">column-major <code>n x m</code>, length <code>n m</code></td>
<td class="org-left">column-wise <code>(x+v)/norm</code></td>
</tr>

<tr>
<td class="org-left"><code>Multinomial</code></td>
<td class="org-left">simplex, length <code>n &gt;= 2</code></td>
<td class="org-left"><code>x odot exp(v oslash x)</code>, renormalize</td>
</tr>

<tr>
<td class="org-left"><code>ComplexCircle { n }</code></td>
<td class="org-left">interleaved <code>(re, im)</code>, length <code>2 n</code></td>
<td class="org-left"><code>sign(z+v)</code> per pair</td>
</tr>

<tr>
<td class="org-left"><code>Symmetric</code></td>
<td class="org-left">row-major <code>n x n</code>, length <code>n^2</code></td>
<td class="org-left"><code>X + U</code> then <code>multisym</code></td>
</tr>

<tr>
<td class="org-left"><code>SkewSymmetric</code></td>
<td class="org-left">row-major <code>n x n</code>, length <code>n^2</code>, <code>n &gt;= 2</code></td>
<td class="org-left"><code>X + U</code> then <code>multiskew</code></td>
</tr>

<tr>
<td class="org-left"><code>EuclideanComplex { n }</code></td>
<td class="org-left">interleaved <code>(re, im)</code>, length <code>2 n</code></td>
<td class="org-left"><code>x + v</code></td>
</tr>

<tr>
<td class="org-left"><code>Constant { n }</code></td>
<td class="org-left">singleton of length <code>n</code></td>
<td class="org-left">the fixed point</td>
</tr>

<tr>
<td class="org-left"><code>Positive { n }</code></td>
<td class="org-left">positive orthant, length <code>n</code></td>
<td class="org-left"><code>x odot exp(v oslash x)</code></td>
</tr>

<tr>
<td class="org-left"><code>Spd</code></td>
<td class="org-left">row-major <code>n x n</code>, length <code>n^2</code>, a perfect square</td>
<td class="org-left"><code>symm(X + U + U X^{-1} U / 2)</code></td>
</tr>

<tr>
<td class="org-left"><code>Grassmann</code>, <code>GrassmannP { n, p }</code></td>
<td class="org-left">column-major <code>n x p</code> frame, length <code>n p</code>; <code>Grassmann</code> is Gr(len, 1) until <code>set_factor_shape</code></td>
<td class="org-left">polar factor of <code>X + U</code></td>
</tr>

<tr>
<td class="org-left"><code>Hyperbolic</code></td>
<td class="org-left">hyperboloid point, time-like entry first, length <code>n &gt;= 2</code></td>
<td class="org-left">the exponential map</td>
</tr>

<tr>
<td class="org-left"><code>PoincareBall</code></td>
<td class="org-left">open unit ball, length <code>k</code></td>
<td class="org-left">the exponential map by Möbius addition</td>
</tr>

<tr>
<td class="org-left"><code>Unitary { n }</code></td>
<td class="org-left">interleaved <code>(re, im)</code> row-major <code>n x n</code>, length <code>2 n^2</code></td>
<td class="org-left">unique QR of <code>U + V</code></td>
</tr>

<tr>
<td class="org-left"><code>MultinomialDoublyStochastic { n }</code>, <code>MultinomialSymmetric { n }</code></td>
<td class="org-left">row-major <code>n x n</code>, length <code>n^2</code></td>
<td class="org-left"><code>X odot exp(V oslash X)</code>, then Sinkhorn</td>
</tr>

<tr>
<td class="org-left"><code>SphereComplex { n }</code></td>
<td class="org-left">interleaved <code>(re, im)</code>, length <code>2 n</code></td>
<td class="org-left"><code>(x+v)/norm(x+v)</code></td>
</tr>

<tr>
<td class="org-left"><code>CenteredMatrix { m, n, rows }</code></td>
<td class="org-left">row-major <code>m x n</code>, length <code>m n</code></td>
<td class="org-left"><code>X + U</code>, then center</td>
</tr>

<tr>
<td class="org-left"><code>Product</code></td>
<td class="org-left">the factor packings end to end</td>
<td class="org-left">each factor's retraction</td>
</tr>
</tbody>
</table>

An isolated molecule or cluster lives on `RigidQuotient`
(`R^{3N}/SE(3)`): Sella Cartesian `fix_translation` plus
`fix_rotation` (Hermes, Sarsfield, Zádor, JCTC 2022,
https://doi.org/10.1021/acs.jctc.2c00395). `MwRigid` is the same quotient
with the Page–McIver mass-weighted metric used by Sella IRC and
gpr<sub>optim</sub> `IRCDriver` (https://doi.org/10.1063/1.454172,
https://doi.org/10.1063/1.434152). Call `set_masses` with N atomic masses;
unit mass makes `MwRigid` identical to `RigidQuotient`.

`Sphere`, `So3`, `Stiefel`, `Se3`, `ComplexCircle`,
`EuclideanComplex`, and `Constant` are matrix-manifold embeddings.
`So3` rejects any length other than 9. `Se3` rejects any length
other than 12. `ComplexCircle { n }` and `EuclideanComplex { n }`
reject any length other than `2 n`. `Constant { n }` rejects any
length other than `n`. They do not pack or prefix-interpret a
3N cluster.

Euclidean is the default. Existing eOn / rgpot / eindir paths do
not change until a host calls the setter.


## Rust

```rust
use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use rgmin::{Control, ManifoldKind, Method, Solver};

let obj = Rosenbrock::<3>::new();
let n = (3.0_f64).sqrt();
let mut x = array![1.0 / n, 1.0 / n, 1.0 / n];
let mut solver = Solver::new(Method::Steepest, Control::default(), 3);
solver.set_manifold(ManifoldKind::Sphere);
let _ = solver.step(&obj, &mut x).unwrap();
assert!((x.dot(&x) - 1.0).abs() < 1e-12);
```

On a curved kind a step projects its increment onto the tangent space.
The step is capped there and then retracted.
The objective sees only retracted points.
A trial that the box would move is refused before any oracle call.
Here `P` is the projection of the kind onto the tangent space.
Newton-CG and Newton-PCG solve the model `P H P + (I - P)`.
Dogleg solves the model `Z^T H Z` with the gradient `Z^T g`, for an orthonormal basis `Z` of the tangent space, and maps its point back through `Z`.
Its Newton and Cauchy points are then tangent, and the trust radius bounds the step that is retracted.
On the sphere dogleg writes the same model in ambient form, `P^T H P + (I - P)^T (I - P)` with `P = I - x x^T`.
Where `P` is an orthogonal projection, the dogleg model equals `P H P + (I - P)` on the tangent space, so dogleg and Newton-CG/PCG solve one model.
That is so on every curved kind but the three multinomial kinds.
Their Fisher projection is oblique, and there the two models differ.
A `fire2` session applies `P` to its velocity.
That projection is at each new point.
A refused trial under `Method::Fire` returns `Error::OffManifold`.
So does a refused L-BFGS trial under `Accept::None`.
Under `Accept::Energy` and `Accept::Nonmonotone`, L-BFGS on the sphere and on Stiefel searches on the set.
On the other curved kinds the L-BFGS line search runs in ambient space.
Its Wolfe probes evaluate `x + a d` off the set.
The step is retracted once, after the search.
Flat kinds keep the old order of retraction, cap and box clip.


## C

```c
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_RIGID_QUOTIENT);
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_MW_RIGID);
rgmin_solver_set_masses(s, masses, n_atoms);
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_SPHERE);
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_SO3);
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_STIEFEL);
rgmin_solver_set_stiefel(s, 5, 2);
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_SE3);
rgmin_solver_set_oblique(s, 3, 2);
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_MULTINOMIAL);
rgmin_solver_set_complex_circle(s, 4);
rgmin_solver_set_euclidean_complex(s, 3);
rgmin_solver_set_constant(s, 4);
rgmin_solver_set_multinomial_ds(s, 3);
rgmin_solver_set_multinomial_sym(s, 3);
rgmin_solver_set_sphere_complex(s, 3);
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_SYMMETRIC);
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_SKEWSYMMETRIC);
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_EUCLIDEAN_COMPLEX);
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_CONSTANT);
rgmin_solver_set_power(s, RGMIN_MANIFOLD_SO3, 9, 2);
rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_EUCLIDEAN);
```

Changing the manifold drops method memory (`forget`).


## L-BFGS on the unit sphere

For `Sphere` (or `Stiefel` with one column), use `Accept::Energy`
to minimize the supplied scalar objective with L-BFGS. The line search
evaluates the normalized extension
$F(z)=f(z/\lVert z\rVert)$, so every objective evaluation lies on
the sphere. Its Euclidean gradient is
$(I-uu^T)\nabla f(u)/\lVert z\rVert$, with
$u=z/\lVert z\rVert$. Omitting the denominator changes the
directional derivative used by the Wolfe conditions.

The session transports every retained curvature pair into the accepted
point's tangent space and forms the new pair there. The caller keeps the
session between steps. `Accept::None` requests the unfiltered step and
does not select a line search.

For a symmetric matrix $H$, minimizing $f(u)=u^T H u/2$ on
this sphere gives a lowest eigenvector at a minimum. An arbitrary
stationary point can be a higher eigenvector; a small projected gradient
alone does not certify that the lowest eigenvalue has been found.


## Products and powers

`ManifoldKind::Product` is the weighted Riemannian product of other kinds, with the metric $g = \sum_i w_i\, g_i$.
`Solver::set_product` takes the factors in packing order, each a kind, a packed length and a weight:

```rust
use rgmin::manifold::ProductFactor;
use rgmin::{Control, ManifoldKind, Method, Solver};

let n = 3;
let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2 * n);
solver.set_product([
    ProductFactor::new(ManifoldKind::Euclidean, n).with_weight(0.25),
    ProductFactor::new(ManifoldKind::Sphere, n),
])?;
```

A point is the factor points end to end, here a position `p` and then a unit normal, of length `2 n`.
The weight of a factor fixes how its steps are measured against those of the others.
For a position in length units next to a unit vector, take $w_p = \ell^{-2}$ for a length $\ell$ over which the objective changes, and the iterates no longer depend on the unit of length.
derives that choice, and lists the operations of every factor kind.

With a weight other than one the session steps in the metric chart $z_i = \sqrt{w_i}\, x_i$.
Points, gradients and reports stay in the coordinates of the factors.
The exponential preconditioner, the Lindh model, `set_project_rigid` and the HiGHS box read raw coordinates, and a step with any of them returns `Error::ProductChart`.
`MwRigid` and `RigidQuotient` factors project block by block with the session's masses and periodic flag.

`Solver::set_power(kind, block, count)` is manopt's `powermanifold`, `count` unit-weight copies of one kind.
It returns `false` and keeps the manifold when the kind cannot hold `block`.
From C, `rgmin_solver_set_power` does the same for the tokens without shape parameters; weighted products are Rust only.

Steepest descent, L-BFGS and the Newton-CG trust region step on a product unchanged.
`rtr::minimize_rtr` runs the Riemannian trust region in the product metric with the full Riemannian Hessian, Weingarten term included.


## Subspace spheres, tangent metrics and gauges

`ManifoldKind::sphere_within(basis)` is the unit sphere of the span of the rows of `basis`, and `ManifoldKind::sphere_excluding(basis)` the unit sphere orthogonal to them.
The rows need not be orthonormal; dependent rows return `Error::SubspaceBasis`.
A unit normal over $N$ atoms that must not move their centre of mass excludes the three translations:

```rust
use std::sync::Arc;

use ndarray::{Array1, Array2};
use rgmin::ManifoldKind;
use rgmin::manifold::{ProductFactor, ProductManifold, TangentMetric};
use rgmin::rtr::{RtrOptions, minimize_rtr_gauged};

let atoms = 4;
let n = 3 * atoms;
let mut translations = Array2::zeros((3, n));
for a in 0..atoms {
    for k in 0..3 {
        translations[[k, 3 * a + k]] = 1.0;
    }
}
let normal = ManifoldKind::sphere_excluding(translations)?;
```

A `TangentMetric` on a product factor makes its part of the metric $w\, g(u, G_x v)$ for an operator $G_x$ that maps each tangent space to itself and is self-adjoint and positive definite there.
`egrad2rgrad` then returns $w^{-1} G_x^{-1}$ times the factor's gradient, and `rtr::minimize_rtr` runs its trust region in this metric, with the iterates of truncated CG in the unweighted metric preconditioned by $(w\, G_x)^{-1}$.
Masses on a position factor are such a metric:

```rust
struct Masses(Array1<f64>);

impl TangentMetric for Masses {
    fn apply(&self, _x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        &self.0 * v
    }
    fn solve(&self, _x: &Array1<f64>, r: &Array1<f64>) -> Array1<f64> {
        r / &self.0
    }
}

let masses = Array1::from_elem(n, 12.0);
let product = ProductManifold::new([
    ProductFactor::new(ManifoldKind::Euclidean, n).with_metric(Arc::new(Masses(masses))),
    ProductFactor::new(normal, n),
])?;
```

A product with a tangent metric returns `None` for the exponential, logarithm, distance and parallel transport, and a session refuses it with `Error::ProductTangentMetric`; run it with `rtr::minimize_rtr`.

A gauge replaces the iterate with another representation of the same point, such as the pivot of a plane moved along the plane.
`rtr::minimize_rtr_gauged` calls it after every accepted step, with `value`, `egrad` and `ehess` the cost and its Euclidean derivatives:

```rust
let out = minimize_rtr_gauged(
    &product,
    x0,
    |x| value(x),
    |x| egrad(x),
    |x, v| ehess(x, v),
    |x| recentre(x),
    RtrOptions::default(),
);
```

The step is accepted on the value before the move, and the gradient is evaluated again at the moved point.
`Solver::run_gauged(&objective, &mut x, gauge)` calls the gauge before every session step but the first and rebases the session when it moves the point.

`egrad` and `ehess` may differentiate any smooth extension of the cost to the ambient space, the same one for both.
The Riemannian Hessian adds the Weingarten term, which removes the dependence on the extension; derives the convention for spheres.


## Packing notes

-   `RigidQuotient` is 3N Cartesians. Isolated: kernel of three
    translations and three infinitesimal rotations
    (`R^{3N}/SE(3)`). Periodic: Sella `proj_rot = false`, so only
    translations (`R^{3N}/T(3)`) via `rgmin_solver_set_periodic`.
    TRICs are a different chart, not a different packing here.
-   `MwRigid` is the same 3N packing. Masses are per atom (length
    N). The IRC path itself is steepest descent in that metric,
    not a separate token.
-   `So3` is a 9-vector, row-major. The tangent projection returns
    the embedded vector `R Omega`, not the skew factor alone.
-   `Se3` is twelve numbers: the same 9-vector, then a translation.
    It is one rigid body, not N atoms.
-   `Stiefel` is `St(n,1)`. `StiefelP { n, p }` / `set_stiefel(n, p)`
    is manopt `stiefelfactory` for `p > 1`: column-major `n p`,
    projection `U - X sym(X^T U)`, QR retraction. A 3N cluster is
    not this packing.
-   `Oblique { n, m }` is manopt `obliquefactory`: product of `m`
    unit spheres in `R^n`, column-major `n m`. It is not the
    sphere and not a 3N cluster.
-   `Multinomial` is manopt `multinomialfactory` at `m = 1`: the
    relative interior of the simplex with the Fisher metric.
    Length `n >= 2`.
-   `ComplexCircle { n }` is manopt `complexcirclefactory(n)`:
    interleaved `(re, im)` pairs, length `2 n`. Each pair is
    independently unit-modulus. It is not the sphere $S^{2n-1}$
    and not a 3N cluster.
-   `Symmetric` is manopt `symmetricfactory`: an `n x n` real
    symmetric matrix packed row-major (`n^2`). Projection is
    symmetrization. Retraction is `X + U`. Transport is the
    identity. It is not the SPD cone and not a 3N cluster.
-   `SkewSymmetric` is manopt `skewsymmetricfactory`: an `n x n`
    real skew-symmetric matrix packed row-major (`n^2`, `n >= 2`).
    Projection is `multiskew` (`0.5 (A - A^T)`). Retraction is
    `X + U`. Transport is the identity. It is not the symmetric
    subspace and not a 3N cluster.
-   `EuclideanComplex { n }` is manopt `euclideancomplexfactory(n)`:
    interleaved `(re, im)` pairs, length `2 n`. Projection is the
    identity. Retraction is `x + v`. Transport is the identity.
    It is not the sphere, not `(S^1)^n`, and not a 3N cluster.
    C token 16 (`rgmin_solver_set_euclidean_complex`).
-   `Constant { n }` is manopt `constantfactory`: a singleton of
    packed length `n`. Projection and transport are the zero
    tangent. Retraction is the fixed point. It is not the sphere
    and not a 3N cluster. C token 17 (`rgmin_solver_set_constant`).
-   `MultinomialDoublyStochastic { n }` is manopt
    `multinomialdoublystochasticfactory`: an `n x n` positive
    doubly-stochastic matrix packed row-major (`n^2`, `n >= 2`).
    Projection is Fisher-orthogonal onto `V 1 = V^T 1 = 0`.
    Retraction is `X ⊙ exp(V ⊘ X)` then Sinkhorn. Transport is
    projection at arrival. It is not the simplex, not the sphere,
    and not a 3N cluster. C token 18
    (`rgmin_solver_set_multinomial_ds`).
-   `MultinomialSymmetric { n }` is manopt
    `multinomialsymmetricfactory`: an `n x n` positive symmetric
    doubly-stochastic matrix packed row-major (`n^2`, `n >= 2`).
    Projection is Fisher-orthogonal onto symmetric `V 1 = 0`.
    Retraction is `X ⊙ exp(V ⊘ X)` then Sinkhorn then symmetrize.
    Transport is projection at arrival. It is not the simplex, not
    the unsigned Birkhoff polytope, not the sphere, and not a 3N
    cluster. C token 19 (`rgmin_solver_set_multinomial_sym`).
-   `SphereComplex { n }` is manopt `spherecomplexfactory`: the
    unit sphere in `C^n`, interleaved `(re, im)`, length `2 n`.
    Projection is `u - Re(x^* u) x`. Retraction is
    `(x+v)/||x+v||`. Transport is projection at arrival. It is
    not the real sphere token, not `(S^1)^n`, and not a 3N
    cluster. C token 20 (`rgmin_solver_set_sphere_complex`).
-   `Positive { n }` is manopt `positivefactory`: the open
    positive orthant of packed length `n`. Projection is the
    identity. Retraction is `x odot exp(v oslash x)`. Transport
    is the identity. It is not the sphere, not the simplex, and
    not a 3N cluster. C token 21 (`rgmin_solver_set_positive`).
-   `CenteredMatrix { m, n, rows }` is manopt
    `centeredmatrixfactory`: an `m x n` matrix packed row-major
    (`m n`) with `X 1 = 0` (`rows = nil`, default) or
    `1^T X = 0` (`rows = t`). Projection is the centering operator.
    Retraction is `X + U` then center. Transport is the identity.
    It is not the sphere and not a 3N cluster. C token 22
    (`rgmin_solver_set_centered_matrix`).
-   `Spd` is manopt `sympositivedefinitefactory` with the
    affine-invariant metric. The Riemannian gradient is
    `X sym(egrad) X`. C token 7.
-   `Grassmann` and `GrassmannP { n, p }` are manopt
    `grassmannfactory`. `Grassmann` reads Gr(len, 1) until
    `set_factor_shape(n, p)` (C `rgmin_solver_set_factor_shape`) names
    the frame; a zero dimension clears it. C token 8.
-   `Hyperbolic` is manopt `hyperbolicfactory`, the hyperboloid model
    in the Minkowski form. C token 9.
-   `PoincareBall` is manopt `poincareballfactory`: the tangent space
    is the ambient space and the metric is conformal with factor
    `2 / (1 - norm(x)^2)`. C token 10.
-   `Unitary { n }` is manopt `unitaryfactory(n)` at one block. C token
    23 selects `n = 1`; `rgmin_solver_set_unitary(s, n)` sets `n`.
-   `Product` packs its factors end to end in the order given. A
    product inside a product is flattened and the weights multiply.
    `Oblique { n, m }` stays the token for a power of spheres.
-   `set_project_rigid` is the same horizontal projection as
    `RigidQuotient` and stays available on Euclidean.
