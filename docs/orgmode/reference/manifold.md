

# Riemannian manifolds

`RiemannianManifold` is a Stiefel manifold `St(n, p)`, a cone `SPD(p)` of symmetric positive definite matrices, the rigid quotient `R^{3N}/SE(3)` of `N` atoms, or a product of them.
Rust, C, C++ and Python call the same code.
fits a point on `St x SPD` by gradient steps.


## Packing

Every buffer is caller-owned, column-major and `len` doubles long.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Factor</th>
<th scope="col" class="org-left">Point</th>
<th scope="col" class="org-left"><code>len</code></th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>St(n, p)</code></td>
<td class="org-left"><code>n x p</code> frame, <code>U^T U = I</code>, <code>1 &lt;= p &lt;= n</code></td>
<td class="org-left"><code>n p</code></td>
</tr>

<tr>
<td class="org-left"><code>SPD(p)</code></td>
<td class="org-left"><code>p x p</code> symmetric positive definite</td>
<td class="org-left"><code>p^2</code></td>
</tr>

<tr>
<td class="org-left">Rigid quotient</td>
<td class="org-left"><code>3 N</code> Cartesians <code>x0, y0, z0, x1, ...</code>, <code>N &gt;= 1</code></td>
<td class="org-left"><code>3 N</code></td>
</tr>

<tr>
<td class="org-left">Product</td>
<td class="org-left">factor blocks in factor order</td>
<td class="org-left">the sum</td>
</tr>
</tbody>
</table>

Tangent vectors and Euclidean gradients use the packing of the point.
A nested product contributes its factors.


## Operations

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
<th scope="col" class="org-left">Operation</th>
<th scope="col" class="org-left">Stiefel (embedded metric)</th>
<th scope="col" class="org-left">SPD affine-invariant</th>
<th scope="col" class="org-left">SPD log-Euclidean</th>
<th scope="col" class="org-left">Rigid quotient (Euclidean metric)</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>project</code></td>
<td class="org-left"><code>V - X sym(X^T V)</code></td>
<td class="org-left"><code>sym(V)</code></td>
<td class="org-left"><code>sym(V)</code></td>
<td class="org-left">remove translations and rotations (rotations kept when periodic)</td>
</tr>

<tr>
<td class="org-left"><code>egrad_to_rgrad</code></td>
<td class="org-left">the projection</td>
<td class="org-left"><code>X sym(G) X</code></td>
<td class="org-left"><code>Dlog(X)^{-2}[sym(G)]</code></td>
<td class="org-left">the projection</td>
</tr>

<tr>
<td class="org-left"><code>retract</code></td>
<td class="org-left">QR (first order) or polar (second order) of <code>X + V</code></td>
<td class="org-left"><code>exp</code></td>
<td class="org-left"><code>exp</code></td>
<td class="org-left"><code>X + V</code></td>
</tr>

<tr>
<td class="org-left"><code>exp</code></td>
<td class="org-left">Edelman, Arias, Smith (1998) eq. 2.41</td>
<td class="org-left"><code>X^{1/2} expm(X^{-1/2} V X^{-1/2}) X^{1/2}</code></td>
<td class="org-left"><code>expm(log X + Dlog(X)[V])</code></td>
<td class="org-left"><code>X + P_X V</code></td>
</tr>

<tr>
<td class="org-left"><code>log</code></td>
<td class="org-left">refused</td>
<td class="org-left">inverse of <code>exp</code></td>
<td class="org-left">inverse of <code>exp</code></td>
<td class="org-left">refused</td>
</tr>

<tr>
<td class="org-left"><code>transport</code></td>
<td class="org-left">projection at the arrival point</td>
<td class="org-left"><code>E V E^T</code>, <code>E = (Y X^{-1})^{1/2}</code></td>
<td class="org-left"><code>Dlog(Y)^{-1}[Dlog(X)[V]]</code></td>
<td class="org-left">projection at the arrival point</td>
</tr>

<tr>
<td class="org-left"><code>inner</code></td>
<td class="org-left"><code>tr(U^T V)</code></td>
<td class="org-left"><code>tr(X^{-1} U X^{-1} V)</code></td>
<td class="org-left"><code>&lt;Dlog(X)[U], Dlog(X)[V]&gt;</code></td>
<td class="org-left"><code>U . V</code></td>
</tr>
</tbody>
</table>

`sym(A) = (A + A^T) / 2`.
`Dlog(X)` is applied in the eigenbasis of `X` with the divided differences of `log` on its eigenvalues (Daleckii-Krein).
`exp` projects `v` to the tangent space first.
The product metric is the sum of the factor metrics.
`P_X` is the horizontal projector at `X`.
`gradient_step(x, egrad, t)` returns `rgrad = egrad_to_rgrad(x, egrad)` and `retract(x, -t rgrad)`.

The affine-invariant metric is Pennec, Fillard and Ayache ([2006](https://doi.org/10.1007/s11263-005-3222-z)).
The log-Euclidean metric is Arsigny, Fillard, Pennec and Ayache ([2007](https://doi.org/10.1137/050637996)).
The Stiefel geodesic is Edelman, Arias and Smith ([1998](https://doi.org/10.1137/S0895479895290954)).


## The rigid quotient and its strata

For an isolated cluster, the factor is a manifold on the regular stratum, `N >= 3` atoms that are not collinear.
The horizontal space has dimension `3N - 6` and the projector has rank 6.
The projector is built from a rank-revealing factorization of the six rigid motions, with every column scaled to unit size, so the rank does not depend on the size of the cluster.

At a singular isolated configuration the quotient is not a manifold.
One atom, or atoms that all coincide, have rank 3: the three rotations move nothing.
Collinear atoms have rank 5: the rotation about the line moves nothing.
The horizontal space is then larger than `3N - 6`, and its dimension changes with the configuration.
The projection stays valid there and removes exactly the motions that move the atoms.
The retraction `X + V` and the transport are well defined, but a step that bends a collinear cluster changes the dimension of the tangent space, so a method that keeps a subspace across the step gets no guarantee from the transport.
Treat a singular configuration as the boundary of the manifold.

When `periodic` is true, the factor quotients translations only.
The translation action is free for every nonempty configuration, so the periodic quotient has dimension `3N - 3`, projector rank 3, and no rotational singular strata.


## Refusals

A wrong length, a non-finite entry, an SPD point that is not symmetric positive definite, a rank-deficient `X + V` under a Stiefel retraction, a non-finite step size, and `log` on a Stiefel or rigid-quotient factor are refused.
A rigid-quotient factor with no atoms is refused too.
Rust returns `Error::ManifoldArgument`, C returns `RGMIN_INVALID_PARAMETER` with `rgmin_last_error`, C++ throws `std::runtime_error`, and Python raises `ValueError`.
C writes the outputs only on success, so an output may alias an input.


## Rust

```rust
use rgmin::{RiemannianManifold, SpdMetric, StiefelRetraction};

let m = RiemannianManifold::product([
    RiemannianManifold::stiefel(6, 2, StiefelRetraction::Polar)?,
    RiemannianManifold::spd(3, SpdMetric::AffineInvariant)?,
    RiemannianManifold::rigid_quotient(5, false)?,
])?;
m.gradient_step(&x, &egrad, 0.1, &mut rgrad, &mut x_next)?;
```

`RiemannianManifold` implements the `Manifold` trait, so generic code such as `rtr::truncated_cg` accepts it.
`Solver` selects its manifold with the closed `ManifoldKind` and does not take this type.


## C

`rgmin_manifold_t` is the session token of `rgmin_solver_set_manifold`.
The handle here is `rgmin_manifold_handle_t`. ABI minor 41.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Function</th>
<th scope="col" class="org-left">Shapes and effect</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>rgmin_manifold_stiefel(n, p, r)</code></td>
<td class="org-left"><code>r</code> is <code>RGMIN_STIEFEL_QR</code> or <code>RGMIN_STIEFEL_POLAR</code>. Null on a bad shape.</td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_spd(p, metric)</code></td>
<td class="org-left"><code>RGMIN_SPD_AFFINE_INVARIANT</code> or <code>RGMIN_SPD_LOG_EUCLIDEAN</code>.</td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_rigid_quotient(n, periodic)</code></td>
<td class="org-left"><code>R^{3N}/SE(3)</code>, or <code>R^{3N}/T(3)</code> when <code>periodic</code> is nonzero. Packed length <code>3 n</code>. ABI minor 56.</td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_product(fs, k)</code></td>
<td class="org-left">Copies <code>k</code> factors; the caller still frees them.</td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_free(m)</code></td>
<td class="org-left">Null is a no-op.</td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_len(m)</code></td>
<td class="org-left">Packed length. Zero for null.</td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_project</code></td>
<td class="org-left"><code>(m, x, v, out, len)</code></td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_egrad_to_rgrad</code></td>
<td class="org-left"><code>(m, x, egrad, out, len)</code></td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_retract</code></td>
<td class="org-left"><code>(m, x, v, out, len)</code></td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_exp</code> / <code>_log</code></td>
<td class="org-left"><code>(m, x, v or y, out, len)</code></td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_transport</code></td>
<td class="org-left"><code>(m, x_from, x_to, v, out, len)</code></td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_inner</code></td>
<td class="org-left"><code>(m, x, u, v, &amp;value, len)</code></td>
</tr>

<tr>
<td class="org-left"><code>rgmin_manifold_gradient_step</code></td>
<td class="org-left"><code>(m, x, egrad, t, rgrad_out, x_out, len)</code></td>
</tr>
</tbody>
</table>


## C++

`rgmin::Manifold` owns the handle: `Manifold::stiefel(n, p, r)`, `Manifold::spd(p, metric)`, `Manifold::rigid_quotient(n_atoms, periodic)` and `Manifold::product(a, b, ...)`.
The members take buffers of `len()` doubles. `xts::optimize::Manifold` is the same class.


## Python

`rgmin.Stiefel(n, p, retraction="qr")`, `rgmin.SPD(p, metric="affine_invariant")`, `rgmin.RigidQuotient(n_atoms, periodic=False)` and `rgmin.Product(*factors)` take and return NumPy arrays.
A Stiefel point is an `(n, p)` array, an SPD point a `(p, p)` array, a rigid-quotient point a `(3 n_atoms, 1)` column of Cartesians (`x.reshape(-1, 3)` gives one atom per row), and a product point a tuple with one array per factor.
Each class has `project`, `egrad_to_rgrad`, `retract`, `exp`, `log`, `transport`, `inner`, `norm` and `gradient_step`, which returns `(rgrad, x_next)`.
