

# Product manifold

The weighted Riemannian product $\mathcal{M}_1 \times \cdots \times \mathcal{M}_k$ with metric $g = \sum_i w_i\, g_i$, the sphere of a subspace, tangent metrics on product factors, and the trust-region and session drivers that run on them.
derives every operation listed here.


## Construction

`rgmin::manifold::ProductFactor` is one factor.
It is `#[non_exhaustive]` with public fields:

-   `kind: ManifoldKind`, the geometry;
-   `len: usize`, the packed length of a factor point;
-   `weight: f64`, the weight $w_i$ of the factor metric.

`ProductFactor::default()` is a Euclidean factor of length 1 and weight 1.
`ProductFactor::new(kind, len)` has weight 1, and `with_kind`, `with_len` and `with_weight` set one field each.
`with_metric(Arc<dyn TangentMetric>)` attaches a tangent metric and `metric()` returns it as `Option<&Arc<dyn TangentMetric>>`; two factors compare equal only when they share the same metric object.

`ProductManifold::new(factors)` builds the product of an iterator of `ProductFactor` in packing order.
A factor whose kind is itself a product is flattened into its blocks, and their weights are multiplied by the factor's weight; the blocks keep their tangent metrics.
It fails with `Error::ProductFactor { index, what }`, where `what` is one of:

-   `"weight must be finite and positive"`;
-   `"packed length must be positive"`;
-   `"packed length does not fit the kind"`, from `Manifold::required_dim` of the kind;
-   `"a nested product takes tangent metrics on its factors"`, for a tangent metric on a factor whose kind is a product;
-   `"packed length overflows"`;
-   `"a product needs at least one factor"`, with index 0.

`ProductManifold::power(kind, block, count)` is manopt's `powermanifold`: `count` unit-weight copies of `kind`, each of packed length `block`.
`count = 0` fails with `"a power needs at least one copy"`.

`ManifoldKind::product(factors)` and `ManifoldKind::power(factor, block, count)` wrap the two constructors in `ManifoldKind::Product(Arc<ProductManifold>)`, whose `as_str` is `"product"`.
`Oblique` stays the dedicated kind for a power of spheres.


## Accessors

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Method</th>
<th scope="col" class="org-left">Returns</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>packed_len()</code></td>
<td class="org-left">packed length of a point</td>
</tr>

<tr>
<td class="org-left"><code>factor_count()</code></td>
<td class="org-left">number of factors after flattening</td>
</tr>

<tr>
<td class="org-left"><code>factors()</code></td>
<td class="org-left">the flattened factors, with their weights in the metric and their tangent metrics</td>
</tr>

<tr>
<td class="org-left"><code>tangent_metric_factor()</code></td>
<td class="org-left"><code>Some</code> index of the first factor with a tangent metric</td>
</tr>

<tr>
<td class="org-left"><code>range(index)</code></td>
<td class="org-left"><code>Some</code> slice of the packed vector holding factor <code>index</code>, <code>None</code> past the end</td>
</tr>

<tr>
<td class="org-left"><code>is_flat()</code></td>
<td class="org-left">every factor keeps a straight step on its set</td>
</tr>

<tr>
<td class="org-left"><code>is_chart()</code></td>
<td class="org-left">every factor has weight one in these coordinates, so <code>chart</code> is the identity</td>
</tr>

<tr>
<td class="org-left"><code>chart_scales()</code></td>
<td class="org-left">\(d\) of \(z = d \odot x\): \(\sqrt{w_i}\) on factor \(i\) of a product from <code>new</code>, 1 on a chart</td>
</tr>

<tr>
<td class="org-left"><code>chart()</code></td>
<td class="org-left">the same manifold in the coordinates \(z = d \odot x\), all weights one</td>
</tr>
</tbody>
</table>


## Packing

A point packs the factor points end to end in the order given, each in its kind's own packing ().
A tangent vector and a Euclidean gradient pack the same way.
The hyperplane $\mathbb{R}^n \times S^{n-1}$ is `[p_1, ..., p_n, n_1, ..., n_n]`:

```rust
use rgmin::manifold::ProductFactor;
use rgmin::ManifoldKind;

let n = 4;
let hyperplane = ManifoldKind::product([
    ProductFactor::new(ManifoldKind::Euclidean, n).with_weight(0.3),
    ProductFactor::new(ManifoldKind::Sphere, n),
])?;
```


## Operations

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left"><code>Manifold</code> method</th>
<th scope="col" class="org-left">On the product</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>required_dim</code></td>
<td class="org-left">the packed length, exactly</td>
</tr>

<tr>
<td class="org-left"><code>project</code></td>
<td class="org-left">each factor's projection; no weights</td>
</tr>

<tr>
<td class="org-left"><code>egrad2rgrad</code></td>
<td class="org-left">each factor's <code>egrad2rgrad</code> of its block, then \(G^{-1}\), divided by \(w_i\)</td>
</tr>

<tr>
<td class="org-left"><code>retract</code></td>
<td class="org-left">each factor's retraction</td>
</tr>

<tr>
<td class="org-left"><code>transport</code></td>
<td class="org-left">each factor's transport</td>
</tr>

<tr>
<td class="org-left"><code>inner</code></td>
<td class="org-left">\(\sum_i w_i\, g_i(u_i, G v_i)\)</td>
</tr>

<tr>
<td class="org-left"><code>norm</code></td>
<td class="org-left">the square root of <code>inner</code></td>
</tr>

<tr>
<td class="org-left"><code>ehess2rhess</code></td>
<td class="org-left">each factor's <code>ehess2rhess</code> of its block of the full action, then \(G^{-1}\), divided by \(w_i\)</td>
</tr>

<tr>
<td class="org-left"><code>exp</code>, <code>log</code></td>
<td class="org-left">each factor's</td>
</tr>

<tr>
<td class="org-left"><code>dist</code></td>
<td class="org-left">\((\sum_i w_i\, d_i^2)^{1/2}\)</td>
</tr>

<tr>
<td class="org-left"><code>parallel_transport_along</code></td>
<td class="org-left">each factor's, along \(t \mapsto \exp_x(t d)\)</td>
</tr>

<tr>
<td class="org-left"><code>parallel_transport</code></td>
<td class="org-left">each factor's, along the geodesic of <code>log</code></td>
</tr>

<tr>
<td class="org-left"><code>transport_is_isometric</code></td>
<td class="org-left">true when it is for every factor and no factor has a tangent metric</td>
</tr>

<tr>
<td class="org-left"><code>typical_dist</code></td>
<td class="org-left">\((\sum_i w_i\, t_i^2)^{1/2}\) over the factor typical distances \(t_i\)</td>
</tr>
</tbody>
</table>

$G$ is the factor's tangent metric, the identity when it has none.
The optional operations, `ehess2rhess`, `exp`, `log`, `dist` and both parallel transports, return `None` as soon as one factor returns `None`; all but `ehess2rhess` also return `None` when a factor has a tangent metric.
Arrays of the wrong length leave the input unchanged: `project`, `egrad2rgrad` and `transport` return the vector, `retract` returns the point, `inner` and `norm` return NaN, and the optional operations return `None`.


## Factor capabilities

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Kind</th>
<th scope="col" class="org-left"><code>ehess2rhess</code></th>
<th scope="col" class="org-left"><code>exp</code></th>
<th scope="col" class="org-left"><code>log</code>, <code>dist</code>, parallel transport</th>
<th scope="col" class="org-left">isometric <code>transport</code></th>
<th scope="col" class="org-left">typical distance</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>Euclidean</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">\(\sqrt{n}\)</td>
</tr>

<tr>
<td class="org-left"><code>Sphere</code>, <code>Stiefel</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">no</td>
<td class="org-left">\(\pi\)</td>
</tr>

<tr>
<td class="org-left"><code>SubspaceSphere</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">no</td>
<td class="org-left">\(\pi\)</td>
</tr>

<tr>
<td class="org-left"><code>StiefelP { n, p }</code></td>
<td class="org-left">yes</td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">\(\sqrt{p}\)</td>
</tr>

<tr>
<td class="org-left"><code>So3</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">no</td>
<td class="org-left">\(\pi \sqrt{3}\)</td>
</tr>

<tr>
<td class="org-left"><code>Se3</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">no</td>
<td class="org-left">\((3 \pi^2 + 3)^{1/2}\)</td>
</tr>

<tr>
<td class="org-left"><code>Oblique { n, m }</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">no</td>
<td class="org-left">\(\pi \sqrt{m}\)</td>
</tr>

<tr>
<td class="org-left"><code>ComplexCircle { n }</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">no</td>
<td class="org-left">\(\pi \sqrt{n}\)</td>
</tr>

<tr>
<td class="org-left"><code>SphereComplex { n }</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">no</td>
<td class="org-left">\(\pi\)</td>
</tr>

<tr>
<td class="org-left"><code>Positive { n }</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">no</td>
<td class="org-left">\(\sqrt{n}\)</td>
</tr>

<tr>
<td class="org-left"><code>Symmetric</code>, <code>SkewSymmetric</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">\(\sqrt{\mathrm{len}}\)</td>
</tr>

<tr>
<td class="org-left"><code>CenteredMatrix { m, n, rows }</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">\(\sqrt{\dim}\)</td>
</tr>

<tr>
<td class="org-left"><code>EuclideanComplex { n }</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">\(\sqrt{n}\)</td>
</tr>

<tr>
<td class="org-left"><code>Constant { n }</code></td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">yes</td>
<td class="org-left">0</td>
</tr>

<tr>
<td class="org-left"><code>Spd</code></td>
<td class="org-left">yes</td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">\(\sqrt{\mathrm{len}}\)</td>
</tr>

<tr>
<td class="org-left"><code>Multinomial</code></td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">\(\sqrt{\mathrm{len}}\)</td>
</tr>

<tr>
<td class="org-left"><code>MultinomialDoublyStochastic</code>, <code>MultinomialSymmetric</code></td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">\(\sqrt{\mathrm{len}}\)</td>
</tr>

<tr>
<td class="org-left"><code>MwRigid</code>, <code>RigidQuotient</code></td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">no</td>
<td class="org-left">\(\sqrt{\mathrm{len}}\)</td>
</tr>
</tbody>
</table>

$\mathrm{len}$ is the packed length and $\dim$ the dimension of the centered matrices, $m n - n$ for centered rows and $m n - m$ for centered columns.
`Positive`, `Spd` and the doubly stochastic kinds carry their own metric through `inner` and `egrad2rgrad`; every other kind uses the dot product of its packed coordinates.
`Multinomial` pairs a Fisher-orthogonal projection with the dot product, so its gradient does not represent the differential in its inner product.


## Subspace sphere

`rgmin::manifold::SubspaceSphere` is the unit sphere $S(U) = \{x \in U : \|x\| = 1\}$ of a linear subspace $U \subset \mathbb{R}^n$, in the ambient Euclidean metric.

-   `SubspaceSphere::within(basis: Array2<f64>) -> Result<Self>` takes $U$ as the span of the rows of `basis`, each of length $n$.
-   `SubspaceSphere::excluding(basis: Array2<f64>) -> Result<Self>` takes $U$ as the orthogonal complement of that span; no row gives the whole sphere $S^{n-1}$.

`ManifoldKind::sphere_within(basis)` and `ManifoldKind::sphere_excluding(basis)` wrap them in `ManifoldKind::SubspaceSphere(Arc<SubspaceSphere>)`, whose `as_str` is `"subspace_sphere"`.
The rows are orthonormalized by modified Gram–Schmidt with a second pass.
Both constructors fail with `Error::SubspaceBasis { what }`, where `what` is one of:

-   `"the ambient dimension must be positive"`;
-   `"a basis entry is not finite"`;
-   `"the basis rows are linearly dependent"`, when the part of a row outside the span of the earlier rows is at most $\sqrt{\varepsilon}$ of its length;
-   `"the subspace needs at least one direction"`, from `within` with no row;
-   `"the excluded directions leave no subspace"`, from `excluding` with $n$ independent rows.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Method</th>
<th scope="col" class="org-left">Returns</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>ambient_dim()</code></td>
<td class="org-left">\(n\), the packed length of a point</td>
</tr>

<tr>
<td class="org-left"><code>subspace_dim()</code></td>
<td class="org-left">\(\dim U\); the sphere has dimension one less</td>
</tr>

<tr>
<td class="org-left"><code>basis()</code></td>
<td class="org-left">the orthonormal rows, spanning \(U\), or \(U^\perp\) when <code>excludes()</code></td>
</tr>

<tr>
<td class="org-left"><code>excludes()</code></td>
<td class="org-left"><code>true</code> for a sphere from <code>excluding</code></td>
</tr>

<tr>
<td class="org-left"><code>project_subspace(v)</code></td>
<td class="org-left">\(P_U v\); a vector of another length comes back unchanged</td>
</tr>

<tr>
<td class="org-left"><code>closest_point(x)</code></td>
<td class="org-left">\(P_U x / \Vert P_U x \Vert\); <code>None</code> for another length or no component in \(U\)</td>
</tr>
</tbody>
</table>

As a `Manifold`, `project` is $P_U - x x^\top$, `retract` is $P_U(x + v) / \Vert P_U(x + v) \Vert$, `transport` is projection at the arrival point and `ehess2rhess` is $(P_U - x x^\top)(\nabla^2 \bar f\, v - (x^\top \nabla \bar f)\, v)$.
The exponential, logarithm, distance and both parallel transports are the sphere's.


## Tangent metrics

```rust
pub trait TangentMetric: Send + Sync {
    fn apply(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;
    fn solve(&self, x: &Array1<f64>, r: &Array1<f64>) -> Array1<f64>;
}
```

`rgmin::manifold::TangentMetric` is an operator $G_x$ on the tangent spaces of one factor, attached with `ProductFactor::with_metric`.
The factor's term of the product metric becomes $w_i\, g_i(u, G_x v)$.
`apply` returns $G_x v$ and `solve` returns $G_x^{-1} r$, for the factor point `x` and tangent vectors in the factor's own packing.
At every `x`, $G_x$ must map the tangent space into itself and be self-adjoint and positive definite in $g_i$, and `solve` must invert `apply`; the product projects both results onto the tangent space.
With a tangent metric the gradient is $w_i^{-1} G_x^{-1} \mathrm{grad}_i f$, and `ehess2rhess` returns the model $w_i^{-1} G_x^{-1} \mathrm{Hess}_i f$, which is self-adjoint in the metric and exact at critical points.
`chart` keeps tangent metrics unchanged.


## Sphere Hessians

`ehess2rhess` of `Sphere`, `SubspaceSphere` and the other embedded kinds that supply it keeps the Weingarten term explicit; on the sphere it is $(I - x x^\top) \nabla^2 \bar f\, v - (x^\top \nabla \bar f)\, v$.
`egrad` and `ehess` may come from any smooth extension $\bar f$ of the cost, and the result does not depend on the choice.
The session's dogleg, Newton-CG and Newton-PCG steps model the Hessian of a curved kind without the term, which is exact for the degree-zero extension $\bar f(x) = f(x / \Vert x \Vert)$ and converges linearly for others; `minimize_rtr` keeps the term.


## Sessions

`Solver::set_product(factors) -> Result<()>` selects a weighted product; on `Error::ProductFactor` the session keeps its manifold.
`Solver::set_power(factor, block, count) -> bool` selects a power, and returns `false`, keeping the manifold, when `factor` cannot hold `block` or `count` is 0.
`Solver::set_manifold` accepts a `ManifoldKind::Product` directly.
A change of manifold clears the session's history.
The session dimension must equal the packed length.

A product whose weights are not all one is stored as its chart.
The caller's points, gradients, Hessians and reports stay in natural coordinates, and the reported gradient norm is the length in the product metric.
A step on such a product fails with `Error::ProductChart { what }` while one of these settings is active:

-   the exponential preconditioner;
-   the Lindh model Hessian;
-   rigid-motion projection of the whole vector (`set_project_rigid`);
-   with the `highs` feature, the HiGHS box and equalities.

`MwRigid` and `RigidQuotient` factors are projected block by block with the session's periodic flag.
`Solver::set_masses` is read by the `MwRigid` blocks as one molecule reused on every block when its length is the atom count of one block, or as every block's masses in order when its length is the atom count of all `MwRigid` blocks; any other length means unit masses.

A step on a product with a tangent metric fails with `Error::ProductTangentMetric { index }`, `index` being the first such factor after flattening; `search_direction` fails the same way and `push_pair` returns `false`.

```rust
pub fn run_gauged<O, G>(&mut self, obj: &O, x: &mut Array1<f64>, gauge: G) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
    G: FnMut(&mut Array1<f64>);
```

`Solver::run_gauged` is `run` with `gauge(x)` called before every step but the first, to replace the iterate with another representation of the same point.
A call that changes `x` rebases the session as `Solver::rebase`: the curvature memory stays, and the next step evaluates the objective afresh instead of reading the move as a step.
The returned report describes the point left in `x`.


## Trust region

```rust
pub fn minimize_rtr<M, FV, GG, HH>(
    manifold: &M,
    x0: Array1<f64>,
    value: FV,
    egrad: GG,
    ehess: HH,
    opts: RtrOptions,
) -> RtrMin
where
    M: Manifold + ?Sized,
    FV: FnMut(&Array1<f64>) -> f64,
    GG: FnMut(&Array1<f64>) -> Array1<f64>,
    HH: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>;
```

`rgmin::rtr::minimize_rtr` is the Absil-Baker-Gallivan trust region with truncated conjugate gradients, run in the metric of `manifold`.
`egrad(x)` is the Euclidean gradient and `ehess(x, v)` the Euclidean Hessian action, both in packed coordinates.
The Hessian is `ehess2rhess`, or `egrad2rgrad` of the action when that returns `None`.

`RtrOptions` is `#[non_exhaustive]`, and its defaults are those of manopt `trustregions`:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Field</th>
<th scope="col" class="org-left">Default</th>
<th scope="col" class="org-left">Setter</th>
<th scope="col" class="org-left">Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>radius_max</code></td>
<td class="org-left"><code>None</code></td>
<td class="org-left"><code>with_radius_max</code></td>
<td class="org-left">\(\bar\Delta\); <code>None</code> takes <code>Manifold::typical_dist</code>, and \(\Delta_0 = \bar\Delta / 8\)</td>
</tr>

<tr>
<td class="org-left"><code>grad_tol</code></td>
<td class="org-left"><code>1e-6</code></td>
<td class="org-left"><code>with_grad_tol</code></td>
<td class="org-left">stop when the Riemannian gradient is at most this long</td>
</tr>

<tr>
<td class="org-left"><code>max_iter</code></td>
<td class="org-left"><code>1000</code></td>
<td class="org-left"><code>with_max_iter</code></td>
<td class="org-left">outer iterations</td>
</tr>

<tr>
<td class="org-left"><code>cg_max</code></td>
<td class="org-left"><code>None</code></td>
<td class="org-left"><code>with_cg_max</code></td>
<td class="org-left">inner iterations; <code>None</code> takes the packed length</td>
</tr>

<tr>
<td class="org-left"><code>theta</code></td>
<td class="org-left"><code>1.0</code></td>
<td class="org-left"><code>with_theta</code></td>
<td class="org-left">\(\theta\) of the inner rule \(\Vert r_j \Vert \le \Vert r_0 \Vert \min(\Vert r_0 \Vert^\theta, \kappa)\)</td>
</tr>

<tr>
<td class="org-left"><code>kappa</code></td>
<td class="org-left"><code>0.1</code></td>
<td class="org-left"><code>with_kappa</code></td>
<td class="org-left">\(\kappa\) of the inner rule</td>
</tr>

<tr>
<td class="org-left"><code>rho_regularization</code></td>
<td class="org-left"><code>1e3</code></td>
<td class="org-left"><code>with_rho_regularization</code></td>
<td class="org-left">both decreases gain \(\max(1, \lvert f \rvert)\, \varepsilon\) times this</td>
</tr>
</tbody>
</table>

A step is accepted when $\rho > 0.1$; the radius is quartered below $\rho = 1/4$ and doubled, up to $\bar\Delta$, above $\rho = 3/4$ on the boundary.
A non-finite or non-positive $\bar\Delta$ is replaced by 1.
`RtrMin` is `#[non_exhaustive]` and holds `x`, `value`, `grad_norm` (in the manifold metric), `iterations`, `accepted` and `inner_iterations`, the conjugate-gradient iterations over all subproblems.

```rust
pub fn minimize_rtr_gauged<M, FV, GG, HH, GA>(
    manifold: &M,
    x0: Array1<f64>,
    value: FV,
    egrad: GG,
    ehess: HH,
    gauge: GA,
    opts: RtrOptions,
) -> RtrMin
where
    M: Manifold + ?Sized,
    FV: FnMut(&Array1<f64>) -> f64,
    GG: FnMut(&Array1<f64>) -> Array1<f64>,
    HH: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    GA: FnMut(&mut Array1<f64>);
```

`rgmin::rtr::minimize_rtr_gauged` is `minimize_rtr` with `gauge(x)` called after every accepted step, to replace the iterate with another representation of the same point.
The step is accepted on the value before the call.
The value is evaluated again when the call changed `x`, and the gradient always is, so the next model, the stopping test and the returned `RtrMin` belong to the moved point.

```rust
pub fn truncated_cg_riemannian<M, H>(
    manifold: &M,
    x: &Array1<f64>,
    egrad: &Array1<f64>,
    ehess: H,
    radius: f64,
    opts: &RtrOptions,
) -> TcgResult
where
    M: Manifold + ?Sized,
    H: FnMut(&Array1<f64>) -> Array1<f64>;
```

`rgmin::rtr::truncated_cg_riemannian` solves one subproblem of `minimize_rtr` at `x` with the trust radius `radius`, using `theta`, `kappa` and `cg_max` from `opts`; on a product with a tangent metric it is the preconditioned subproblem.
`TcgResult` holds the step `eta`, the model decrease `model_decrease`, the stopping reason `stop` and the conjugate-gradient `iterations`.
A driver that measures the decrease itself retracts `eta` and passes the measured and model decreases with `manifold.norm(x, &eta)` to `RtrRadius::update(actual_decrease, model_decrease, eta_norm) -> bool`, which applies the acceptance and radius rule above and returns whether the step is accepted.
`RtrRadius::new(radius_max)` starts at a radius of `radius_max / 8`.


## C ABI

```c
int32_t rgmin_solver_set_power(rgmin_solver_t *solver, int32_t factor, size_t block,
                               size_t count);
```

ABI minor 41.
`factor` is an `rgmin_manifold_t` value for a kind without shape parameters: `RGMIN_MANIFOLD_EUCLIDEAN`, `SPHERE`, `SO3`, `STIEFEL`, `SE3`, `RIGID_QUOTIENT`, `MW_RIGID`, `MULTINOMIAL`, `SYMMETRIC` or `SKEWSYMMETRIC`.
It returns 0, or 1 with the manifold unchanged and `rgmin_last_error` set to:

-   `rgmin_solver_set_power: null solver`;
-   `rgmin_solver_set_power: the factor token needs a shape`, for a token with shape parameters or a value outside `rgmin_manifold_t`;
-   `rgmin_solver_set_power: the block does not fit the factor`, for a block the factor cannot hold or a count of 0.

`MW_RIGID` copies read `rgmin_solver_set_masses` as described under Sessions.
The C++ wrapper is `rgmin::Solver::set_power(rgmin_manifold_t factor, std::size_t block, std::size_t count)`, and `xts.h` aliases it as `xts_solver_set_power`.
Weighted products of arbitrary factors, subspace spheres, tangent metrics and gauges are not exposed to C or Python; gives the reasons.
