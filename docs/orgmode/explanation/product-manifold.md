
# Table of Contents

1.  [Why a product, and why weights](#orgbe0b808)
2.  [The product metric](#org8e012fd)
3.  [The gradient](#org7a5d4ac)
4.  [Projection and egrad2rgrad](#orgfd8fa33)
5.  [The product retraction](#org19f1589)
6.  [Exponential, logarithm, distance and injectivity radius](#orgcbfe966)
7.  [Vector transport](#org9ddddcb)
8.  [Parallel transport and the locking condition](#orgd564012)
9.  [The Riemannian Hessian and ehess2rhess](#org40ddf6e)
10. [Typical distance and the trust-region defaults](#org3afa9f0)
11. [The metric chart and unit invariance](#org061d3cb)
12. [The sphere of a subspace](#org15a882f)
13. [Tangent metrics](#orge4e352f)
14. [Gauge moves](#orgb867241)
15. [Sphere Hessians and the extension of the cost](#orgb84759a)
16. [The optimal dividing plane](#org33428ae)
17. [Comparison with manopt, Manifolds.jl, pymanopt and geomstats](#org725d673)
18. [Reconciliation with the earlier manifold branches](#org26ce8b2)
19. [How the derivation is checked](#orgf817d5e)
20. [Limitations](#orgca561e9)

A point of a product manifold $\mathcal{M} = \mathcal{M}_1 \times \cdots \times \mathcal{M}_k$ is a tuple $x = (x_1, \dots, x_k)$ of factor points.
`ProductManifold` (`src/manifold/product.rs`) packs the tuple end to end in one vector and gives it the metric $g = \sum_i w_i\, g_i$ with constant weights $w_i > 0$.
This page derives every operation the solvers call from that one choice, shows where the weights enter and where they cancel, and compares the result with the products of manopt, Manifolds.jl, pymanopt and geomstats.
The first consumer is the optimal dividing hyperplane of transition-state theory in `rgsaddle`: a point and a unit normal in $\mathbb{R}^n \times S^{n-1}$, or a unit normal and a signed offset in $S(U) \times \mathbb{R}$, with $U$ the directions a sampler moves in.
Later sections derive what that design needs beyond the product: the sphere of a subspace, metrics supplied on a factor's tangent spaces, gauge moves between steps and the convention for sphere Hessians.
The section on the optimal dividing plane puts them together; nothing else is specific to it.
lists the API.

Each factor $\mathcal{M}_i$ sits in the space $\mathcal{E}_i = \mathbb{R}^{n_i}$ of its packed coordinates, with tangent projector $P_i(x_i)$ and metric $g_i$.
For most kinds $g_i$ is the Euclidean inner product of $\mathcal{E}_i$ restricted to tangent vectors, which makes $\mathcal{M}_i$ a Riemannian submanifold of $\mathcal{E}_i$.
`Positive` and the doubly stochastic multinomials carry their own $g_i$, through `Manifold::inner` and `Manifold::egrad2rgrad`.
A caller supplies $\bar f$, the objective on the packed coordinates, with its coordinate gradient $\nabla \bar f = (\nabla_1 \bar f, \dots, \nabla_k \bar f)$ and its Hessian action $\nabla^2 \bar f\, u$.
The function optimized is the restriction $f = \bar f|_{\mathcal{M}}$.


<a id="orgbe0b808"></a>

# Why a product, and why weights

A dividing hyperplane in mass-weighted coordinates $y = M^{1/2} r$ is $\{y : n^\top (y - p) = 0\}$, named by a point $p \in \mathbb{R}^n$ and a unit normal $n \in S^{n-1}$.
The pair $(p, n)$ lives on $\mathbb{R}^n \times S^{n-1}$, a flat factor times a curved one.
Both factors are already kinds of `ManifoldKind`; what has to be derived is their product.

The plain product metric $|\delta p|^2 + |\delta n|^2$ adds a squared mass-weighted length to a pure number.
Its value depends on the unit of length: going from ångström to bohr multiplies the first term by $c^2$, with $c \approx 1.89$, and leaves the second alone.
Every step length, trust radius, secant pair and stopping test measured with it inherits that dependence, so the path of the optimizer changes with the unit system.

Weights make that hidden parameter explicit.
Move the plane by $(\delta p, \delta n)$, with $n^\top \delta n = 0$.
A point $y = p + r e$ of the plane, $e \perp n$ a unit vector, is displaced along the normal by
$$
(n + \delta n)^\top (y - p - \delta p) = r\, e^\top \delta n - n^\top \delta p + O(|\delta|^2).
$$
The shift moves every point of the plane by at most $|\delta p|$, and the turn moves points at distance $r$ by up to $r |\delta n|$.
If the objective feels the plane out to a distance $\ell$, in the units of $p$, the two are comparable when $|\delta p| \approx \ell |\delta n|$, and the natural size of a step is
$$
\|(\delta p, \delta n)\|^2 = \ell^{-2} |\delta p|^2 + |\delta n|^2, \qquad w_p = \ell^{-2}, \quad w_n = 1.
$$
Two scales are natural.
The geometric one is the mass-weighted radius of gyration of the system about $p$.
The kinetic one uses that mass-weighted coordinates make the kinetic energy $\tfrac12 |\dot y|^2$: an in-plane mode of angular frequency $\omega$ at temperature $T$ spreads over $\ell = \sqrt{k_B T} / \omega$, again in the units of $p$.
Either way $\ell$ scales with the unit of length, so $w_p$ carries the unit and the metric does not.
The section on the metric chart turns this into a statement about iterates.

The parameterization is redundant: $p$ can slide inside the plane, and $(p, -n)$ names the same plane.
An objective that depends on the plane alone is constant along the in-plane part of $\delta p$, so its Hessian is singular there.
Trust-region and quasi-Newton steps tolerate that.
A caller that wants a minimal parameterization writes $p = s\, n$ and optimizes over $\mathbb{R} \times S^{n-1}$.


<a id="org8e012fd"></a>

# The product metric

For tangent vectors $u = (u_1, \dots, u_k)$ and $v$ at $x$,
$$
g_x(u, v) = \sum_{i=1}^k w_i\, g_{i, x_i}(u_i, v_i), \qquad w_i > 0.
$$
The tangent space is the direct sum $T_x \mathcal{M} = T_{x_1} \mathcal{M}_1 \oplus \cdots \oplus T_{x_k} \mathcal{M}_k$ (Absil, Mahony and Sepulchre, section 3.1.6; Boumal, equation 3.31) , and $g$ makes the blocks orthogonal.
At $w_i = 1$ this is the product metric of Boumal, Example 3.57.
`ProductManifold::inner` is the weighted sum and `norm` its square root.

The weighted product of the $(\mathcal{M}_i, g_i)$ is the plain product of the scaled factors $(\mathcal{M}_i, w_i g_i)$.
Every fact about plain Riemannian products (Boumal, Table 7.2) therefore applies to it.
What remains is how the objects of one factor change when its metric is multiplied by a constant $w > 0$:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Object</th>
<th scope="col" class="org-left">Under \(g \mapsto w\, g\)</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">Levi-Civita connection, geodesics, exponential, logarithm</td>
<td class="org-left">unchanged</td>
</tr>

<tr>
<td class="org-left">parallel transport</td>
<td class="org-left">unchanged</td>
</tr>

<tr>
<td class="org-left">retraction, vector transport</td>
<td class="org-left">unchanged; they do not involve the metric</td>
</tr>

<tr>
<td class="org-left">inner product</td>
<td class="org-left">times \(w\)</td>
</tr>

<tr>
<td class="org-left">norm, distance, injectivity radius</td>
<td class="org-left">times \(\sqrt{w}\)</td>
</tr>

<tr>
<td class="org-left">gradient, Hessian operator, sectional curvature</td>
<td class="org-left">times \(1 / w\)</td>
</tr>
</tbody>
</table>

The Koszul formula,
$$
2 g(\nabla_X Y, Z) = X g(Y, Z) + Y g(X, Z) - Z g(X, Y) + g([X, Y], Z) - g([X, Z], Y) - g([Y, Z], X),
$$
is linear in $g$ on both sides, so a constant factor leaves the connection unchanged, and with it geodesics, the exponential, the logarithm and parallel transport.
Lengths of curves scale by $\sqrt{w}$, and so do distances and the injectivity radius.
The gradient solves $w\, g(\mathrm{grad}_w f, u) = \mathrm{D} f[u]$, so $\mathrm{grad}_w f = w^{-1} \mathrm{grad} f$, and the Hessian $\nabla_u \mathrm{grad}_w f$ uses the same connection.
The curvature tensor depends on the connection alone, so the sectional curvature $\langle R(u, v) v, u \rangle / (\|u\|^2 \|v\|^2 - \langle u, v \rangle^2)$ gains $w$ above and $w^2$ below.
geomstats' `ScalarProductMetric` rescales the methods of a metric by exactly these rules.


<a id="org7a5d4ac"></a>

# The gradient

The Riemannian gradient is the tangent vector that represents the differential in the metric (Absil, Mahony and Sepulchre, section 3.6): $g_x(\mathrm{grad} f(x), u) = \mathrm{D} f(x)[u]$ for every $u \in T_x \mathcal{M}$.
Write $\mathrm{D} f(x)[u] = \sum_i \mathrm{D}_i f(x)[u_i]$, with $\mathrm{D}_i f$ the differential in the $i$-th argument, and let $\mathrm{grad}_i f(x) \in T_{x_i} \mathcal{M}_i$ represent $\mathrm{D}_i f(x)$ in $g_i$.
Then
$$
\mathrm{grad} f(x) = \bigl( w_1^{-1} \mathrm{grad}_1 f(x), \dots, w_k^{-1} \mathrm{grad}_k f(x) \bigr),
$$
because
$$
g_x(\mathrm{grad} f(x), u) = \sum_i w_i\, g_i\bigl(w_i^{-1} \mathrm{grad}_i f(x), u_i\bigr) = \sum_i \mathrm{D}_i f(x)[u_i] = \mathrm{D} f(x)[u],
$$
and the representer is unique because $g_x$ is nondegenerate.
At unit weights this is Boumal, Exercise 3.67.
A heavy factor gets a short gradient: its coordinates are expensive in the metric, so steepest descent moves them less.


<a id="orgfd8fa33"></a>

# Projection and egrad2rgrad

When every $g_i$ is induced by $\mathcal{E}_i$, $g$ is induced by the weighted inner product
$$
\langle a, b \rangle_W = \sum_i w_i\, a_i^\top b_i = a^\top W b, \qquad W = \mathrm{diag}(w_1 I_{n_1}, \dots, w_k I_{n_k}),
$$
on $\mathcal{E} = \mathcal{E}_1 \times \cdots \times \mathcal{E}_k$, and $(\mathcal{M}, g)$ is a Riemannian submanifold of the Euclidean space $(\mathcal{E}, \langle \cdot, \cdot \rangle_W)$.
Two facts follow.

The $W$-orthogonal projector onto $T_x \mathcal{M}$ is $P(x) = \mathrm{diag}(P_1(x_1), \dots, P_k(x_k))$, free of weights.
The normal space of the product is the product of the factor normal spaces, and a vector normal to $T_{x_i} \mathcal{M}_i$ in $\mathcal{E}_i$ stays $W$-normal because $W$ acts on block $i$ as the scalar $w_i$.
`ProductManifold::project` applies the factor projections block by block.

The $W$-gradient of $\bar f$ is $\nabla_W \bar f = W^{-1} \nabla \bar f$, since $\langle W^{-1} \nabla \bar f, b \rangle_W = \nabla \bar f^\top b$ for every $b$.
On a Riemannian submanifold the Riemannian gradient is the projected ambient gradient (Absil, Mahony and Sepulchre, section 3.6), so
$$
\mathrm{grad} f(x) = P(x)\, W^{-1} \nabla \bar f(x) = \bigl( w_i^{-1} P_i(x_i) \nabla_i \bar f(x) \bigr)_i,
$$
the gradient of the previous section with $\mathrm{grad}_i f = P_i \nabla_i \bar f$.
A factor with its own metric enters the same way through its own `egrad2rgrad`, because the Riesz argument holds for any $g_i$.
`ProductManifold::egrad2rgrad` applies each factor's `egrad2rgrad` to its block of $\nabla \bar f$ and divides by $w_i$.
Of the two maps a caller's gradient passes through, the projection and the gradient conversion, only the second sees the weights.


<a id="org19f1589"></a>

# The product retraction

A retraction is a smooth map $R : T \mathcal{M} \to \mathcal{M}$ with $R_x(0_x) = x$ and $\mathrm{D} R_x(0_x) = \mathrm{id}_{T_x \mathcal{M}}$ (Absil, Mahony and Sepulchre, Definition 4.1.1; Boumal, Definition 3.47).
The product retraction applies each factor's retraction to its block (Boumal, Exercise 3.50):
$$
R_x(v) = \bigl( R^{(1)}_{x_1}(v_1), \dots, R^{(k)}_{x_k}(v_k) \bigr).
$$

*First-order agreement.*
Each block is smooth in $(x_i, v_i)$, so $R$ is smooth on $T \mathcal{M} = T \mathcal{M}_1 \times \cdots \times T \mathcal{M}_k$, and $R_x(0_x) = (R^{(i)}_{x_i}(0))_i = x$.
For $v \in T_x \mathcal{M}$ the curve $c(t) = R_x(t v)$ has the blocks $c_i(t) = R^{(i)}_{x_i}(t v_i)$, so
$$
c'(0) = \bigl( \mathrm{D} R^{(i)}_{x_i}(0)[v_i] \bigr)_i = (v_i)_i = v.
$$
Hence $\mathrm{D} R_x(0_x) = \mathrm{id}$, and in the packed coordinates $R_x(t v) = x + t v + O(t^2)$: the retraction agrees with the step to first order.
The weights appear nowhere, since neither condition involves the metric.

A retraction is second order when its curves also start with zero covariant acceleration, $\frac{\mathrm{D}}{\mathrm{d}t} c'(0) = 0$ (Absil, Mahony and Sepulchre, equation 5.33; Boumal, Definition 5.42).
The connection of a product is the product of the factor connections (Boumal, Exercises 5.4 and 5.13), and the weights do not change it, so the acceleration of $c$ is the tuple of the factor accelerations.
The product retraction is therefore second order exactly when every factor retraction is.
The straight line of a flat factor and the normalized step on the sphere (Boumal, Example 5.43) are second order; the QR retraction of SO(3) is first order only.
The local rate of the trust region does not need a second-order retraction (Absil, Mahony and Sepulchre, Proposition 5.5.6), but the numerical Hessian check does (Boumal, section 6.8).
That is why the tests run the check along the exponential wherever a retraction is first order.


<a id="orgcbfe966"></a>

# Exponential, logarithm, distance and injectivity radius

Geodesics of the product are tuples of factor geodesics (Boumal, Exercise 5.39), and the weights leave them alone, so (Boumal, Exercise 10.32)
$$
\exp_x(v) = \bigl( \exp_{x_i}(v_i) \bigr)_i, \qquad \log_x(y) = \bigl( \log_{x_i}(y_i) \bigr)_i.
$$
If $\exp_x(v) = y$ then $\exp_{x_i}(v_i) = y_i$ for every $i$, and $\|v\|_x^2 = \sum_i w_i \|v_i\|_{x_i}^2$ is smallest when every block is.
So, where each factor logarithm is defined, the tuple of factor logarithms is the logarithm of the product for every choice of positive weights.
The distance follows from Boumal, Exercise 10.14, applied to the scaled factors:
$$
\mathrm{dist}(x, y) = \Bigl( \sum_i w_i\, \mathrm{dist}_i(x_i, y_i)^2 \Bigr)^{1/2}.
$$

The exponential at $x$ is a diffeomorphism on the $g$-ball of radius $r$ exactly when every factor exponential is one on its $w_i g_i$-ball of radius $r$.
A vector supported on one block shows that the condition is necessary.
It is sufficient because the $g$-ball lies inside the product of the factor balls, where the exponential is a product of diffeomorphisms.
So
$$
\mathrm{inj}(x) = \min_i \sqrt{w_i}\, \mathrm{inj}_i(x_i),
$$
and a flat factor, with $\mathrm{inj}_i = \infty$, does not constrain it.
A 2-plane tangent to factor $i$ has sectional curvature $K_i / w_i$, and a plane spanned by vectors from two different factors has curvature zero.
For $k \ge 2$ the curvature bounds are $\min(0, \min_i K_i^{\min} / w_i)$ and $\max(0, \max_i K_i^{\max} / w_i)$.
`Manifold` has no injectivity or curvature hook; the two formulas are stated for step-size arguments and for the comparison below.

`ProductManifold::exp` and `log` return `None` as soon as one factor has no closed form.
`dist` is the weighted root sum of squares.


<a id="org9ddddcb"></a>

# Vector transport

A vector transport with associated retraction $R$ maps $\xi \in T_x \mathcal{M}$ along $\eta \in T_x \mathcal{M}$ to $\mathcal{T}_\eta \xi \in T_{R_x(\eta)} \mathcal{M}$, with $\mathcal{T}_{0_x} \xi = \xi$ and $\mathcal{T}_\eta$ linear (Absil, Mahony and Sepulchre, Definition 8.1.1).
The blockwise transport
$$
\mathcal{T}_\eta \xi = \bigl( \mathcal{T}^{(i)}_{\eta_i} \xi_i \bigr)_i
$$
satisfies the three conditions because each block does.
It lands in $\bigoplus_i T_{R_{x_i}(\eta_i)} \mathcal{M}_i = T_{R_x(\eta)} \mathcal{M}$, it is the identity at $\eta = 0$, and it is linear.
Its associated retraction is the product retraction, and it does not involve the metric.
For a Riemannian submanifold factor `transport` is projection onto the arrival tangent space (Absil, Mahony and Sepulchre, section 8.1.3), so on a product of them it is $P(y)$, the $W$-orthogonal projection.

The product transport is an isometry of $g$ exactly when every factor transport is an isometry of its $g_i$.
If each is, then
$$
g_y(\mathcal{T} \xi, \mathcal{T} \zeta) = \sum_i w_i\, g_i(\mathcal{T}^{(i)} \xi_i, \mathcal{T}^{(i)} \zeta_i) = \sum_i w_i\, g_i(\xi_i, \zeta_i) = g_x(\xi, \zeta).
$$
If one is not, vectors supported on that block break the identity.
The weights can neither repair nor break isometry, so `transport_is_isometric` is the conjunction over the factors.
Projection is an isometry only where neither the tangent space nor the metric changes: Euclidean space and the linear subspaces.
On $\mathbb{R}^n \times S^{n-1}$ it is not.


<a id="orgd564012"></a>

# Parallel transport and the locking condition

Parallel transport along a curve $c = (c_1, \dots, c_k)$ is the tuple of factor parallel transports (Boumal, Exercise 10.39), unchanged by the weights because the connection is.
It is an isometry of $g$: each block is an isometry of $g_i$, and so of $w_i g_i$.
`parallel_transport_along(x, d, v)` carries $v$ along $t \mapsto \exp_x(t d)$ from $t = 0$ to $t = 1$, and `parallel_transport(x, y, v)` takes $d = \log_x(y)$.

The Riemannian Broyden class of Huang, Gallivan and Absil ([SIAM J. Optim. 25, 1660 (2015)](https://doi.org/10.1137/140955483)) pairs a retraction $R$ with an isometric transport $\mathcal{T}_S$ under the locking condition
$$
\mathcal{T}_{S_\xi} \xi = \beta\, \mathcal{T}_{R_\xi} \xi, \qquad
\beta = \frac{\|\xi\|}{\|\mathcal{T}_{R_\xi} \xi\|}, \qquad
\mathcal{T}_{R_\xi} \xi = \frac{\mathrm{d}}{\mathrm{d}t} R_x(t \xi) \Big|_{t = 1}.
$$
Take $R = \exp$ and $\mathcal{T}_S$ parallel transport along the geodesic.
The velocity of a geodesic is parallel along it, so $\mathcal{T}_{R_\xi} \xi = P_{0 \to 1} \xi$, its length is $\|\xi\|$, $\beta = 1$, and the condition holds with equality.
On the product it holds block by block and so for the tuple, for every choice of weights.
A product whose factors all supply the exponential and parallel transport therefore carries a retraction and transport pair that meets the hypotheses of that theory.
The session's L-BFGS moves its pairs with `transport`, projection, and does not rely on them.


<a id="org40ddf6e"></a>

# The Riemannian Hessian and ehess2rhess

The Riemannian Hessian is $\mathrm{Hess} f(x)[u] = \nabla_u \mathrm{grad} f$ (Absil, Mahony and Sepulchre, Definition 5.5.1).
For a plain product, Boumal's Example 5.19 (equation 5.12, stated for two factors and extended to $k$ by grouping) gives, with $G_i = \mathrm{grad}_i f$,
$$
\mathrm{Hess} f(x)[u]_i = \mathrm{Hess}_i f(x)[u_i] + \sum_{j \ne i} \mathrm{D}_j G_i(x)[u_j].
$$
The cross terms $\mathrm{D}_j G_i$ couple the blocks; only the diagonal terms carry the curvature of the factors.
On the scaled factors $G_i$ becomes $w_i^{-1} G_i$ and $\mathrm{Hess}_i$ becomes $w_i^{-1} \mathrm{Hess}_i$, so for the weighted product
$$
\mathrm{Hess} f(x)[u]_i = w_i^{-1} \Bigl( \mathrm{Hess}_i f(x)[u_i] + \sum_{j \ne i} \mathrm{D}_j G_i(x)[u_j] \Bigr).
$$

For Riemannian submanifold factors the weighted ambient space gives the same result with the Weingarten terms explicit.
On a Riemannian submanifold of a Euclidean space (Boumal, Corollary 5.47; Absil, Mahony and Sepulchre, Proposition 5.3.2, for the connection it rests on),
$$
\mathrm{Hess} f(x)[u] = P(x) \bigl( \nabla^2 \bar f(x)[u] \bigr) + \mathfrak{W}_x\bigl(u, P^\perp(x) \nabla \bar f(x)\bigr),
$$
where the Weingarten map $\mathfrak{W}_x$ carries the curvature of the embedding.
In $(\mathcal{E}, \langle \cdot, \cdot \rangle_W)$ the ambient gradient is $W^{-1} \nabla \bar f$ and the ambient Hessian is $W^{-1} \nabla^2 \bar f$.
The projector is blockwise and free of weights, and so is the Weingarten map: the normal spaces and projectors of block $i$ are those of $\mathcal{M}_i \subset \mathcal{E}_i$, and $W$ acts on the block as a scalar, so the defining identity of $\mathfrak{W}_i$ holds with both sides multiplied by $w_i$.
Block $i$ of the Hessian is therefore
$$
\mathrm{Hess} f(x)[u]_i
= w_i^{-1} \Bigl( P_i(x_i) \bigl( \nabla^2 \bar f(x)\, u \bigr)_i + \mathfrak{W}_i\bigl(u_i, P_i^\perp(x_i) \nabla_i \bar f(x)\bigr) \Bigr)
= w_i^{-1}\, \mathrm{ehess2rhess}_i\bigl(x_i, \nabla_i \bar f, (\nabla^2 \bar f\, u)_i, u_i\bigr).
$$
The block $(\nabla^2 \bar f\, u)_i = \sum_j \nabla^2_{ij} \bar f\, u_j$ of the caller's full Hessian action carries the cross terms; the factor adds only its own Weingarten term.
The same holds for every kind in the crate that supplies `ehess2rhess`, including those with their own metric.
Each such map is affine in the Hessian action with linear part the kind's `egrad2rgrad`, and $\mathrm{D}_j G_i(x)[u_j] = \mathrm{egrad2rgrad}_i(\nabla^2_{ij} \bar f\, u_j)$, so the off-diagonal blocks of the action produce exactly the cross terms of equation 5.12.
`ProductManifold::ehess2rhess` applies each factor's map to its block of the full action and divides by $w_i$.

For the hyperplane, with the sphere's Weingarten term $\mathfrak{W}_n(u_n, \nu) = -(n^\top \nu)\, u_n$,
$$
\mathrm{Hess} f(p, n)[u]
= \Bigl( w_p^{-1} \bigl( \nabla^2 \bar f\, u \bigr)_p,\;
w_n^{-1} \bigl( (I - n n^\top) (\nabla^2 \bar f\, u)_n - (n^\top \nabla_n \bar f)\, u_n \bigr) \Bigr).
$$

The weights cancel in the bilinear form,
$$
g_x\bigl(\mathrm{Hess} f(x)[u], v\bigr) = v^\top \nabla^2 \bar f(x)\, u + \sum_i \mathfrak{W}_i\bigl(u_i, P_i^\perp \nabla_i \bar f(x)\bigr)^\top v_i,
$$
which is symmetric in $u$ and $v$ because each Weingarten map is self-adjoint.
They cancel in the Newton equation too: $\mathrm{Hess} f(x)[\eta] = -\mathrm{grad} f(x)$ has $w_i^{-1}$ on both sides of block $i$, so the Newton step does not depend on the weights.
Only the steps that measure, the gradient step and the trust region, see them.


<a id="org3afa9f0"></a>

# Typical distance and the trust-region defaults

manopt's `typicaldist` is the distance scale a trust region starts from.
If every factor moves by its own typical distance $t_i$, the product moves by
$$
t(\mathcal{M}) = \Bigl( \sum_i w_i\, t_i^2 \Bigr)^{1/2},
$$
the weighted form of manopt's root sum of squares and the value of `ProductManifold::typical_dist`.
The factor values follow the manopt factories: $\pi$ for a sphere, $\pi \sqrt{3}$ for SO(3), $\pi \sqrt{m}$ for $m$ oblique columns and $\sqrt{n}$ for $\mathbb{R}^n$.

`rtr::minimize_rtr` is the trust region with truncated conjugate gradients of Absil, Baker and Gallivan ([Found. Comput. Math. 7, 303 (2007)](https://doi.org/10.1007/s10208-005-0179-9); Absil, Mahony and Sepulchre, Algorithms 10 and 11).
It runs in the metric of the manifold: the gradient by `egrad2rgrad`, the Hessian by `ehess2rhess`, and every inner product, norm and the trust ball by `inner`.
`RtrOptions` takes the defaults of manopt `trustregions`:

-   $\bar\Delta$ the typical distance and $\Delta_0 = \bar\Delta / 8$;
-   acceptance above $\rho' = 0.1$, the radius quartered below $\rho = 1/4$ and doubled, up to $\bar\Delta$, above $\rho = 3/4$ on the boundary;
-   the inner rule $\|r_j\| \le \|r_0\| \min(\|r_0\|^\theta, \kappa)$ with $\theta = 1$ and $\kappa = 0.1$ (Absil, Mahony and Sepulchre, equation 7.10), and an inner budget of the packed length, which bounds the dimension;
-   `rho_regularization` $10^3$, which adds $10^3 \varepsilon \max(1, |f|)$ to both decreases so that steps below the rounding of $f$ read $\rho \approx 1$.

With the exact Hessian the local rate is $\min(\theta + 1, 2)$ (Absil, Mahony and Sepulchre, Theorem 7.4.11), quadratic at $\theta = 1$.
A factor without `ehess2rhess` makes the product fall back to `egrad2rgrad` of the ambient action, the Gauss-Newton part of the Hessian, which loses that rate.

The typical distance of a flat factor is the one place a unit enters again.
$\pi$ is intrinsic to a sphere, but $\sqrt{n}$ for $\mathbb{R}^n$ counts coordinates in whatever unit they are written in.
Under $p' = c\, p$ and $w_p' = w_p / c^2$ the term $w_p\, n$ becomes $w_p\, n / c^2$, and the default $\bar\Delta$ moves.
A run that must not depend on the unit sets $\bar\Delta$ from a physical length with `RtrOptions::with_radius_max`, for instance $\bar\Delta = (w_p L^2 + w_n \pi^2)^{1/2}$ with $L$ the largest shift of $p$ worth trusting.


<a id="org061d3cb"></a>

# The metric chart and unit invariance

The session steppers, L-BFGS, conjugate gradients, FIRE, the Newton-CG trust region and the rest, measure steps, curvature pairs and gradients with the dot product of the packed vector.
On a weighted product that is not $g$.
Rather than thread $g$ through every stepper, a session stores a weighted product in the coordinates where $g$ is the dot product.

The map $\phi(x) = (\sqrt{w_1}\, x_1, \dots, \sqrt{w_k}\, x_k)$ is a diffeomorphism from $\mathcal{M}$ onto the product of the scaled sets $\sqrt{w_i}\, \mathcal{M}_i$, with $\mathrm{D} \phi(x)[u] = (\sqrt{w_i}\, u_i)_i$.
For Riemannian submanifold factors
$$
\mathrm{D} \phi(x)[u]^\top \mathrm{D} \phi(x)[v] = \sum_i w_i\, u_i^\top v_i = g_x(u, v),
$$
so $\phi$ is an isometry onto $\prod_i \sqrt{w_i}\, \mathcal{M}_i$ with the Euclidean metric of the packed vector, and a stepper that measures with dot products in $z = \phi(x)$ measures in $g$.
`ProductManifold::chart` is that product and `ProductManifold::chart_scales` is $d$, equal to $\sqrt{w_i}$ on the coordinates of factor $i$.
A block stores $s_i x_i$ for a factor point $x_i$ and carries the metric $c_i\, g_i$, with $c_i s_i^2 = w_i$: $s_i = 1$ and $c_i = w_i$ in natural coordinates, $s_i = \sqrt{w_i}$ and $c_i = 1$ in the chart.
Every operation divides the block by $s_i$, calls the factor and scales back.
The Weingarten term of the scaled factor $s_i \mathcal{M}_i$ is $1 / s_i$ times the unit one, which the code obtains by passing the factor $\nabla_i \bar f / s_i$.

`Solver::set_manifold` and `Solver::set_product` store a product whose weights are not all one as its chart.
The caller keeps natural coordinates: the session reads the objective as $\tilde f(z) = \bar f(z \oslash d)$, so that
$$
\nabla_z \tilde f = \nabla_x \bar f \oslash d, \qquad
\nabla_z^2 \tilde f = D^{-1} \nabla_x^2 \bar f\, D^{-1}, \qquad
\nabla_z^2 \tilde f\, v = D^{-1} \nabla_x^2 \bar f\, (D^{-1} v),
$$
with $D = \mathrm{diag}(d)$.
Box bounds are scaled by $d$, and the point, the reported coordinates and the reported gradient are mapped back.
The reported gradient norm is already its $g$-length.
Factors with their own metric (`Positive`, the doubly stochastic multinomials) are stepped in the chart as a session steps them alone, with their Riemannian gradient and dot products.
`minimize_rtr` needs no chart, since it measures everything with `inner`.

Settings that read raw coordinates as atoms or boxes cannot be rescaled behind the caller's back, and a weighted product refuses them with `Error::ProductChart`.
They are the exponential preconditioner, the Lindh model Hessian, rigid-motion projection of the whole vector, and the HiGHS box and equalities.
Rigid factors (`MwRigid`, `RigidQuotient`) inside a product are projected block by block with the session's masses and periodic flag.
`MwRigid` blocks read the masses as one molecule reused on every block, or as the concatenation over the `MwRigid` blocks.

*Unit invariance.*
Change the unit of a factor, $x_i' = c\, x_i$, and its weight with it, $w_i' = w_i / c^2$.
The chart coordinate does not change, $\sqrt{w_i'}\, x_i' = \sqrt{w_i}\, x_i$, and the objective written in the new unit, $\bar f'(x') = \bar f(x)$, reads in the chart as the same function $\tilde f$.
A deterministic stepper started at the same chart point therefore produces the same chart iterates, and mapping back gives $x_i' = c\, x_i$ at every step.
The iterates do not depend on the unit once the weight carries it; in floating point they agree to rounding.
The test `iterates_do_not_depend_on_the_length_unit_once_the_weight_carries_it` runs fifteen L-BFGS steps on the hyperplane in ångström with $w_p$ and in bohr with $w_p / c^2$, and finds them equal to $10^{-9}$.
Keeping $w_p$ in bohr moves them by more than $10^{-4}$.
The same test runs `minimize_rtr` in both units with $\bar\Delta$ set explicitly, and finds the same iteration count, accepted steps and minimizer.


<a id="org15a882f"></a>

# The sphere of a subspace

A sampler that fixes the centre of mass, or holds some atoms still, moves only in a linear subspace $U \subset \mathbb{R}^n$, and the normal of a plane it samples must lie in $U$ too.
The normal is then a point of the unit sphere of $U$,
$$
S(U) = \{ x \in U : \|x\| = 1 \}.
$$
`SubspaceSphere` (`src/manifold/subspace_sphere.rs`) takes $U$ as the span of the rows of a matrix, `SubspaceSphere::within`, or as the orthogonal complement of that span, `SubspaceSphere::excluding`.
The caller passes the smaller basis: the centre-of-mass directions and the coordinates of fixed atoms are excluded, a rotation subspace of a few modes is kept.
The rows are orthonormalized once into $Q$, so that $P_U$ is $Q^\top Q$ or $I - Q^\top Q$, applied in two passes over the rows rather than as a dense $n \times n$ matrix.
`ManifoldKind::sphere_within` and `ManifoldKind::sphere_excluding` wrap the two constructors.

$U$ with the Euclidean inner product is a Euclidean space, so $S(U)$ is a sphere, a Riemannian submanifold of $U$ and therefore of $\mathbb{R}^n$.
Its tangent space at $x$ is $\{v \in U : x^\top v = 0\}$.
Since $P_U x = x$, the projectors $P_U$ and $x x^\top$ commute, and
$$
P_x = (I - x x^\top) P_U = P_U - x x^\top
$$
is symmetric and idempotent with range $U \cap x^\perp$: it is the orthogonal projector onto the tangent space.
The Riemannian gradient is $P_x \nabla \bar f = P_U \nabla \bar f - (x^\top \nabla \bar f)\, x$.

The retraction normalizes the step inside $U$,
$$
R_x(v) = \frac{P_U (x + v)}{\|P_U (x + v)\|}.
$$
For $x \in S(U)$ and a tangent $v$ the projection changes nothing in exact arithmetic, so $R_x(v)$ is the point of $S(U)$ nearest to $x + v$.
That is the metric projection retraction, which is second order (Boumal, Propositions 5.54 and 5.55, and Example 5.43 for the sphere).
In floating point the projection removes the rounding that would otherwise build up along the excluded directions over a long run.
Transport is projection onto the arrival tangent space.

For the Hessian, extend the gradient off the manifold by $\bar G(y) = P_U \nabla \bar f(y) - (y^\top \nabla \bar f(y))\, y$.
Its differential is
$$
\mathrm{D} \bar G(x)[u] = P_U \nabla^2 \bar f(x)\, u - \bigl( u^\top \nabla \bar f(x) + x^\top \nabla^2 \bar f(x)\, u \bigr)\, x - (x^\top \nabla \bar f(x))\, u,
$$
and $P_x$ removes $x$, keeps a tangent $u$ and satisfies $P_x P_U = P_x$, so Boumal's Corollary 5.16 gives
$$
\mathrm{Hess} f(x)[u] = P_x \bigl( \nabla^2 \bar f(x)\, u - (x^\top \nabla \bar f(x))\, u \bigr),
$$
the sphere's formula with $P_x$ in place of $I - x x^\top$.
$U$ is flat, so the part of $\nabla \bar f$ in $U^\perp$ adds no curvature and only the radial part $x^\top \nabla \bar f$ bends: the Weingarten map of $S(U)$ in $\mathbb{R}^n$ is $\mathfrak{W}_x(u, \nu) = -(x^\top \nu)\, u$ for every normal $\nu \in U^\perp \oplus \mathrm{span}(x)$.

$S(U)$ is a great sphere of $S^{n-1}$.
For $x \in S(U)$ and a tangent $v$, the great circle $\cos(t \|v\|)\, x + \sin(t \|v\|)\, v / \|v\|$ stays in $U$, and a geodesic of $S^{n-1}$ that stays in a submanifold is a geodesic of the submanifold.
The exponential, logarithm, distance and parallel transport are therefore the sphere's, the injectivity radius is $\pi$, and so is `typical_dist`.

Gram–Schmidt runs twice over the rows, which keeps them orthonormal to rounding even when they are nearly dependent.
After the two passes the residual of a row of length $L$ carries an error of order $\varepsilon L$, and normalizing a residual of length $\rho$ leaves an error of order $\varepsilon L / \rho$ in its orthogonality to the earlier rows.
A row whose residual is at most $\sqrt{\varepsilon}\, L$ would leave more than $\sqrt{\varepsilon}$, and is refused as dependent.
The constructors also refuse an ambient dimension of zero, an entry that is not finite, no kept direction and excluded directions that fill $\mathbb{R}^n$, all with `Error::SubspaceBasis`.
A single kept direction $u$ gives the two points $\pm u$, at which every tangent vector is zero.


<a id="orge4e352f"></a>

# Tangent metrics

A weight scales a whole factor.
At the optimal plane the rotation block of the harmonic free energy is $k_B T (I + |\lambda_1| H_\parallel^{-1})$, whose eigenvalues follow the in-plane curvatures of the saddle across decades, and no single scalar matches them.
A `TangentMetric` supplies an operator $G_x$ on the tangent spaces of one factor, self-adjoint and positive definite in $g_i$ and smooth in $x$, and that factor's part of the metric becomes
$$
w_i\, g_i(u, G_x v).
$$
`ProductFactor::with_metric` attaches one; `TangentMetric::apply` returns $G_x v$ and `TangentMetric::solve` returns $G_x^{-1} r$.
The product projects both results onto the tangent space, which removes rounding but does not repair an operator that leaves it.

*Gradient.*
The Riesz condition $w_i\, g_i(\mathrm{grad}_G f, G_x u) = \mathrm{D}_i f(x)[u]$ and the self-adjointness of $G_x$ give
$$
\mathrm{grad}_G f = w_i^{-1} G_x^{-1}\, \mathrm{grad}_i f.
$$
`egrad2rgrad` therefore returns the preconditioned gradient, and a driver that steps along it, deterministic or sampled, is preconditioned without knowing it.

*Hessian model.*
The Riemannian Hessian of the new metric needs its Levi-Civita connection, which involves derivatives of $G_x$ in $x$ that an operator supplied as `apply` and `solve` does not give.
`ehess2rhess` returns instead
$$
H_G[u] = w_i^{-1} G_x^{-1}\, \mathrm{Hess}_i f(x)[u],
$$
with $\mathrm{Hess}_i$ the factor's Hessian in $g_i$.
It has the three properties the trust region needs.
It is self-adjoint in the new metric, since $w_i\, g_i(H_G u, G_x v) = g_i(\mathrm{Hess}_i f(x)[u], v)$.
It is exact at a critical point, where the Hessian depends on the metric only through the inner product on the tangent space (Boumal, Proposition 5.45 and the remark after it): there $g_G(\mathrm{Hess}_G f(x)[u], v) = g(\mathrm{Hess} f(x)[u], v) = g_G(H_G u, v)$.
Away from critical points the bilinear forms of two Riemannian Hessians of one function differ by $\mathrm{D} f(x)[\Gamma_x(u, v)]$, with $\Gamma$ the difference of the two Levi-Civita connections, and the Hessian of the pullback $\hat f_x = f \circ R_x$ at $0$ differs from either by another term linear in $\mathrm{D} f(x)$.
$H_G$ thus meets the condition $\|H_k - \mathrm{Hess} \hat f_{x_k}(0)\| \le \beta_H \|\mathrm{grad} f(x_k)\|$ of Absil, Mahony and Sepulchre, Theorem 7.4.11, and `minimize_rtr` keeps its local rate $\min(\theta + 1, 2)$ in the new metric.

*Preconditioned truncated CG.*
In this metric the truncated conjugate gradients of `minimize_rtr` are manopt's preconditioned `trs_tCG` in the unweighted metric $\langle \cdot, \cdot \rangle$, with the preconditioner $P$ equal to $(w_i G_x)^{-1}$ on block $i$.
The residual of the model in the new metric is $P r$, with $r$ the residual in the unweighted metric.
The step length $\langle P r, P r \rangle_G / \langle \delta, H_G \delta \rangle_G = \langle r, P r \rangle / \langle \delta, \mathrm{Hess} f\, \delta \rangle$ is manopt's `z_r / d_Hd`, and the trust ball $\|\eta\|_G^2 = \langle \eta, P^{-1} \eta \rangle \le \Delta^2$ is manopt's, so the iterates are the same.
Two tests differ.
The inner rule here reads the residual in the new metric, $\langle r, P r \rangle^{1/2}$, where manopt reads $\|r\|$; manopt's source calls the choice arbitrary and notes that Conn, Gould and Toint accept both (, page 206).
The outer test here reads $\|\mathrm{grad}_G f\|_G = \langle \mathrm{grad} f, P\, \mathrm{grad} f \rangle^{1/2}$, where manopt reads $\|\mathrm{grad} f\|$.

*What the metric costs.*
The geodesics of the new metric have no closed form, so the exponential, logarithm, distance and both parallel transports of a product with a tangent metric return `None`.
Projection is not an isometry of the new metric, so `transport_is_isometric` is false.
`typical_dist` ignores $G$, which has no single scale; a caller whose $G$ changes lengths by much sets $\bar\Delta$ with `RtrOptions::with_radius_max`.

*Nesting, the chart and sessions.*
A factor that is itself a product takes tangent metrics on its own factors, and flattening keeps them.
A metric on the whole nested product would couple blocks that the product keeps orthogonal, and `ProductManifold::new` refuses it with `Error::ProductFactor`.
The operator always receives the factor point and tangent vectors in the factor's own packing, so `ProductManifold::chart` keeps it unchanged.
The session steppers measure with the dot product of the packed vector or of the chart, and no fixed change of coordinates turns $g(u, G_x v)$ into a dot product when $G_x$ varies with $x$.
A session therefore refuses a product with a tangent metric: a step fails with `Error::ProductTangentMetric`, `push_pair` returns `false` and `search_direction` fails.
`minimize_rtr`, `minimize_rtr_gauged` and `truncated_cg_riemannian` run in the metric.


<a id="orgb867241"></a>

# Gauge moves

A gauge move replaces the coordinates of the iterate by another representation of the same object, and leaves the objective's value unchanged.
The plane $\{q : n \cdot (q - c) = d\}$ is named by its normal and its offset from a pivot $c$, and the pivot can slide: moving it to $c'$ and the offset to
$$
d' = d - n \cdot (c' - c)
$$
names the same plane.
The pivot is a parameter of the objective, not a coordinate of the manifold, so the move changes both the packed point and the function that reads it.

About a fixed pivot, turning the normal also moves the plane where it crosses the saddle region, so rotation and offset couple.
The two parametrizations differ by the shear $(n, d) \mapsto (n, d - n \cdot \Delta c)$, $\Delta c = c' - c$: a turn $u$ about $c$ is, about $c'$, the same turn with the offset changed by $-u \cdot \Delta c$.
For the harmonic free energy at the optimal plane, the Hessian couples offset and rotation through $\lambda_1 P q_0$, with $q_0$ the saddle relative to the pivot and $P$ the tangent projector of the normal, and adds $|\lambda_1| P q_0 q_0^\top P$ to the rotation block.
Both vanish when the pivot is the in-plane minimum $q^\ddagger + (r/s) H^+ n$ of the harmonic energy, the centre of the harmonic distribution on the plane, which at the optimum is the saddle.
Re-centring the pivot there after every step makes the Hessian at the optimum block-diagonal, with the harmonic metric as its rotation block.

*In the trust region.*
`minimize_rtr_gauged` calls the gauge after every accepted step.
The step is accepted on the value before the move, since the ratio compares the model at $x$ with the objective at the trial point, both in the old parametrization.
The value is evaluated again only when the gauge changed $x$, because a gauge preserves it; the gradient always is, because a new parametrization changes it even when it leaves $x$ as it was.
The next model, the stopping test and the returned `RtrMin` belong to the moved point, and the radius carries over.
A gauge that changes the value still runs, without the monotone decrease.
manopt's `options.hook` in `trustregions` does the same: it runs at the start of every iteration, before the stopping test, and when it moves $x$ the solver evaluates the cost and gradient again.
It also runs after rejected steps, where an idempotent gauge has nothing to do.

*In a session.*
`Solver::run_gauged` calls the gauge before every step but the first, so the returned report describes the point left in $x$.
A session keeps the last point, value and gradient, and its next step builds a secant pair from them, reading any change of $x$ between calls as a step.
Across a move that pair mixes two parametrizations: its change of gradient holds the shear, not curvature.
A gauge that changes $x$ therefore rebases the session, as `Solver::rebase`.
The curvature memory (the L-BFGS pairs, the BFGS matrix, the trust radii) stays, the cached point goes, and the next step evaluates the objective afresh at the moved point.
The retained pairs come from nearby parametrizations, and the shear between them shrinks as the pivot converges.
On the plane test, re-centring before every step and running to a gradient norm of $10^{-7}$, the three ways to treat a move took these numbers of steps:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">At a move</th>
<th scope="col" class="org-right">L-BFGS</th>
<th scope="col" class="org-right">Newton-CG</th>
<th scope="col" class="org-right">BFGS</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">keep the secant across it</td>
<td class="org-right">50</td>
<td class="org-right">34</td>
<td class="org-right">111</td>
</tr>

<tr>
<td class="org-left">forget all memory (<code>Solver::forget</code>)</td>
<td class="org-right">75</td>
<td class="org-right">108</td>
<td class="org-right">not converged in 3000</td>
</tr>

<tr>
<td class="org-left">rebase (<code>Solver::rebase</code>)</td>
<td class="org-right">33</td>
<td class="org-right">34</td>
<td class="org-right">111</td>
</tr>
</tbody>
</table>

L-BFGS without the gauge takes 36 steps.


<a id="orgb84759a"></a>

# Sphere Hessians and the extension of the cost

The caller's $\bar f$ is defined on $\mathbb{R}^n$, and many functions agree with it on the sphere.
`Sphere::ehess2rhess` keeps the Weingarten term explicit,
$$
\mathrm{Hess} f(x)[u] = (I - x x^\top) \nabla^2 \bar f(x)\, u - (x^\top \nabla \bar f(x))\, u,
$$
and every embedded kind that supplies `ehess2rhess` keeps its own term the same way.
The result does not depend on the extension.
If $\bar f_1$ and $\bar f_2$ agree on the sphere, then $h = \bar f_1 - \bar f_2$ vanishes there, so its gradient is normal: $\nabla h(x) = \mu(x)\, x$ with $\mu = x^\top \nabla h$.
Differentiating along a curve on the sphere with velocity $u$ gives $\nabla^2 h(x)\, u = (\mathrm{D} \mu(x)[u])\, x + \mu(x)\, u$, so $(I - x x^\top) \nabla^2 h(x)\, u = \mu(x)\, u$, which the Weingarten term $-(x^\top \nabla h(x))\, u = -\mu(x)\, u$ cancels.
Boumal's Corollary 5.47 states the general fact: the correction depends on the extension only through the normal part of its gradient.
On `SubspaceSphere` the gradient of $h$ also has a part in $U^\perp$, whose derivative stays in $U^\perp$, and $P_x$ removes both.

The degree-zero extension $\bar f(x) = f(x / \|x\|)$ is constant along rays, so $x^\top \nabla \bar f = 0$ on the sphere and the term vanishes: the projected Hessian alone is the Riemannian Hessian.
The session's dogleg step models the Hessian as $Z^\top \nabla^2 \bar f\, Z$, with $Z$ an orthonormal basis of the tangent space, and its Newton-CG and Newton-PCG steps as $P \nabla^2 \bar f\, P + (I - P)$.
Neither has the Weingarten term, so both are exact for the degree-zero extension.
For another extension the model is off by $x^\top \nabla \bar f$ times the identity on the tangent space, which at a minimizer is the Lagrange multiplier and does not vanish, and the steps converge linearly.
For $\bar f(x) = x^\top A x$ the model is $2A$ on the tangent space where the Hessian is $2(A - \lambda_1 I)$, so near the lowest eigenvector each step keeps $\lambda_1 / \lambda_j$ of the error along eigenvector $j$, and the rate is $\lambda_1 / \lambda_2$ when $0 < \lambda_1 < \lambda_2$.
`tests/subspace_sphere.rs` measures that rate, $1/2$ within 0.02, on a subspace sphere where $\lambda_1 / \lambda_2 = 1/2$, against at most ten steps for the degree-zero extension $x^\top A x / x^\top x$.
`minimize_rtr` uses `ehess2rhess` and converges quadratically with either.


<a id="org33428ae"></a>

# The optimal dividing plane

Section 4.6 of the optimal-hyperplane (OH-TST) design in `rgsaddle` lists what it needs from rgmin.
Its plane is a unit normal $n$ on the sphere of the subspace the sampler moves in and a signed offset $d$ from a pivot $c$, so the manifold is $S(U) \times \mathbb{R}$, and the drivers minimize $f = -A$ for the free energy $A$ of the plane.
`tests/plane_free_energy.rs` solves its harmonic model, $A = r^2 / (2s) + \tfrac{k_B T}{2} \ln(-s)$ with $s = n^\top H^+ n$ and $r = d - n \cdot (q^\ddagger - c)$, whose optimum is the unstable mode through the saddle $q^\ddagger$, with $f^* = \tfrac{k_B T}{2} \ln |\lambda_1|$.
The items of section 4.6 map onto the crate as follows.

1.  *A Euclidean factor.*
    `ProductFactor::new(ManifoldKind::Euclidean, 1)` carries the offset.
    SPD(1) is the half-line $d > 0$, and no metric on it lets the offset change sign.
    From a pivot at the origin the optimal offset of the test plane is $e_1 \cdot q^\ddagger = -0.8$.
    On SPD(1) the retraction $d + v + v^2 / (2d)$ never goes below $d / 2$, and in the affine-invariant metric the gradient norm $d\,|\partial f / \partial d|$ vanishes with $d$, so the run stops on the gradient tolerance after 116 iterations at $d \approx 5 \times 10^{-11}$ with $f$ still 0.47 above $f^*$.
2.  *Weights and a preconditioner.*
    `ProductFactor::with_weight` sets the block weights of the design's section 4.2, and `ProductFactor::with_metric` attaches a tangent metric, here the harmonic rotation block $k_B T (I + |\lambda_1| H_\parallel^{-1})$ compressed to the tangent space.
    Both enter `egrad2rgrad` and every inner product of the trust region.
3.  *A sphere of a subspace.*
    `ManifoldKind::sphere_excluding` takes the centre-of-mass directions and fixed atoms, and `ManifoldKind::sphere_within` takes a rotation subspace.
4.  *A gauge hook.*
    `minimize_rtr_gauged` re-centres the pivot after each accepted step, and `Solver::run_gauged` does it between the steps of a session.
5.  *Stochastic drivers.* Not here; see the limitations.
6.  *A Hessian-vector oracle with an error bar and a harmonic preconditioner.* Not here; see the limitations.
7.  *A convention for sphere Hessians.*
    Both options of the design hold: `ehess2rhess` keeps the explicit Weingarten term for any extension, and the session steppers are exact for the degree-zero extension.

On the harmonic model with $D = 24$ coordinates, $k_B T = 0.3$, $\lambda_1 = -1.5$ and in-plane curvatures from 0.05 to 8, `minimize_rtr` from a normal a few degrees off the unstable mode reaches the optimal plane in every metric, to $10^{-10}$ in the gradient norm, the normal and $f$:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Metric on \((n, d)\)</th>
<th scope="col" class="org-right">Outer iterations</th>
<th scope="col" class="org-right">Inner iterations</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">unit weights</td>
<td class="org-right">8</td>
<td class="org-right">40</td>
</tr>

<tr>
<td class="org-left">block weight \(\ell^2 = 0.5\) on the normal</td>
<td class="org-right">7</td>
<td class="org-right">35</td>
</tr>

<tr>
<td class="org-left">harmonic tangent metric, offset weight \(\vert\lambda_1\vert\)</td>
<td class="org-right">6</td>
<td class="org-right">14</td>
</tr>

<tr>
<td class="org-left">the same, with the pivot re-centred after each step</td>
<td class="org-right">6</td>
<td class="org-right">8</td>
</tr>
</tbody>
</table>

Exact second derivatives keep the outer counts close; the metric shows in the inner solves.
At the optimum the harmonic metric makes the Hessian the identity plus the rank-two coupling of rotation and offset, and re-centring removes the coupling.
A session's L-BFGS in the block-weighted chart takes 36 steps, and 33 with `Solver::run_gauged` re-centring the pivot before each step.
The sampled loop of the design is `truncated_cg_riemannian` for the step and `RtrRadius::update` for a decrease measured outside the model; the test runs it with exact decreases in place of estimates.


<a id="org725d673"></a>

# Comparison with manopt, Manifolds.jl, pymanopt and geomstats

At unit weights the product here is the product of:

-   manopt `productmanifold` ([Boumal et al., JMLR 15, 1455 (2014)](https://jmlr.org/papers/v15/boumal14a.html)), source [a149c85](https://github.com/NicolasBoumal/manopt/blob/a149c85ca3cf23ca828d8bc6a6b125a4ac82d656/manopt/tools/productmanifold.m);
-   ManifoldsBase.jl `ProductManifold`, used by Manifolds.jl and by the solvers of Manopt.jl ([Axen et al., ACM TOMS 49, 1 (2023)](https://doi.org/10.1145/3618296); [Bergmann, JOSS 7, 3866 (2022)](https://doi.org/10.21105/joss.03866)), source [b16e9d6](https://github.com/JuliaManifolds/ManifoldsBase.jl/blob/b16e9d6f305a68385135092055cb8698d6c69e89/src/ProductManifold.jl);
-   pymanopt `Product` ([Townsend et al., JMLR 17, 1 (2016)](https://jmlr.org/papers/v17/16-177.html)), source [35b022f](https://github.com/pymanopt/pymanopt/blob/35b022ff6043b893f32c903b428e811dea200dc2/src/pymanopt/manifolds/product.py);
-   geomstats `ProductRiemannianMetric` ([Miolane et al., JMLR 21, 1 (2020)](https://jmlr.org/papers/v21/19-027.html)), source [5dd5cc2](https://github.com/geomstats/geomstats/blob/5dd5cc286a86f71ce9aed4f263539ec4bfa4c586/geomstats/geometry/product_manifold.py).

All five agree on the summed inner product and on the blockwise exponential and logarithm, and those that offer parallel transport carry it per factor.
manopt and pymanopt also convert Euclidean gradients and Hessians factor by factor, passing each factor its block of the full Euclidean Hessian action as here, so the cross terms of equation 5.12 arrive the same way.
ManifoldsBase.jl supplies the Weingarten map per factor for the same purpose, and geomstats has no such conversion.

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
<th scope="col" class="org-left">Aspect</th>
<th scope="col" class="org-left">rgmin</th>
<th scope="col" class="org-left">manopt</th>
<th scope="col" class="org-left">Manifolds.jl</th>
<th scope="col" class="org-left">pymanopt</th>
<th scope="col" class="org-left">geomstats</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">weights</td>
<td class="org-left">\(w_i > 0\) per factor</td>
<td class="org-left">none</td>
<td class="org-left">none</td>
<td class="org-left">none</td>
<td class="org-left">per factor, <code>ScalarProductMetric</code></td>
</tr>

<tr>
<td class="org-left">points</td>
<td class="org-left">one packed vector</td>
<td class="org-left">struct of fields</td>
<td class="org-left"><code>ArrayPartition</code></td>
<td class="org-left">list</td>
<td class="org-left">one vector, or stacked</td>
</tr>

<tr>
<td class="org-left">nested product</td>
<td class="org-left">flattened, weights multiply</td>
<td class="org-left">nested structs</td>
<td class="org-left"><code>×</code> flattens</td>
<td class="org-left">refused</td>
<td class="org-left">an ordinary factor</td>
</tr>

<tr>
<td class="org-left">missing operation</td>
<td class="org-left"><code>None</code> at the call</td>
<td class="org-left">field absent</td>
<td class="org-left">method error</td>
<td class="org-left">factor raises</td>
<td class="org-left">numerical solver, else error</td>
</tr>

<tr>
<td class="org-left">distance</td>
<td class="org-left">weighted \(\ell_2\)</td>
<td class="org-left">\(\ell_2\)</td>
<td class="org-left">\(\ell_r\), default \(r = 2\)</td>
<td class="org-left">\(\ell_2\)</td>
<td class="org-left">\(\ell_2\)</td>
</tr>

<tr>
<td class="org-left">typical distance</td>
<td class="org-left">weighted \(\ell_2\)</td>
<td class="org-left">\(\ell_2\)</td>
<td class="org-left">none</td>
<td class="org-left">\(\ell_2\)</td>
<td class="org-left">none</td>
</tr>

<tr>
<td class="org-left">random tangent</td>
<td class="org-left">none</td>
<td class="org-left">scaled \(k^{-1/2}\)</td>
<td class="org-left">per factor</td>
<td class="org-left">scaled \(k^{-1/2}\)</td>
<td class="org-left">per factor</td>
</tr>

<tr>
<td class="org-left">retraction choice</td>
<td class="org-left">one per kind</td>
<td class="org-left">one per factory</td>
<td class="org-left"><code>ProductRetraction</code></td>
<td class="org-left">one per factor</td>
<td class="org-left">exponential</td>
</tr>

<tr>
<td class="org-left">parallel transport</td>
<td class="org-left">per factor, optional</td>
<td class="org-left">if all factors have it</td>
<td class="org-left">per factor</td>
<td class="org-left">none</td>
<td class="org-left">per factor</td>
</tr>

<tr>
<td class="org-left">injectivity radius</td>
<td class="org-left">not exposed</td>
<td class="org-left">not exposed</td>
<td class="org-left">minimum</td>
<td class="org-left">not exposed</td>
<td class="org-left">minimum</td>
</tr>

<tr>
<td class="org-left">power manifold</td>
<td class="org-left">unit-weight product</td>
<td class="org-left"><code>powermanifold</code></td>
<td class="org-left"><code>PowerManifold</code></td>
<td class="org-left">none</td>
<td class="org-left"><code>NFoldManifold</code></td>
</tr>

<tr>
<td class="org-left">subspace sphere</td>
<td class="org-left">kept or excluded basis</td>
<td class="org-left">two fixed subspaces</td>
<td class="org-left">symmetric matrices</td>
<td class="org-left">kept or excluded basis</td>
<td class="org-left">none</td>
</tr>

<tr>
<td class="org-left">preconditioner</td>
<td class="org-left"><code>TangentMetric</code>, in the metric</td>
<td class="org-left">in truncated CG</td>
<td class="org-left">in truncated CG (Manopt.jl)</td>
<td class="org-left">in truncated CG</td>
<td class="org-left">none</td>
</tr>

<tr>
<td class="org-left">point hook</td>
<td class="org-left">after accepted steps; sessions rebase</td>
<td class="org-left"><code>options.hook</code></td>
<td class="org-left">step callbacks (Manopt.jl)</td>
<td class="org-left">none</td>
<td class="org-left">none</td>
</tr>
</tbody>
</table>

The differences in detail:

-   *Weights.*
    Only this product weighs its factors.
    manopt, Manifolds.jl and pymanopt build the element-wise sum of the factor metrics, and a caller who wants $w_i g_i$ must write a scaled factor.
    geomstats removed the `scales` of `ProductManifold` and `ProductRiemannianMetric` in 2022 ([856c3ac](https://github.com/geomstats/geomstats/commit/856c3ac2104e)).
    It now scales a factor by equipping it with `ScalarProductMetric`, whose rules are the table of the metric section: inner products times $s$; norms, distances and injectivity radius times $\sqrt{s}$; sectional curvature times $1/s$; exponential, logarithm and parallel transport unchanged.
    Its product reshape for such factors was fixed in November 2026 ([pull request 2203](https://github.com/geomstats/geomstats/pull/2203)).
    A weighted product here is a geomstats product of `ScalarProductMetric` factors.
-   *Points.*
    A point here is one `f64` vector holding each factor point in its kind's packing, column-major for `StiefelP` like manopt's `X(:)` and row-major for the symmetric kinds, so the session steppers and the C ABI see one array.
    manopt uses a struct with one named field per factor, Manifolds.jl a RecursiveArrayTools `ArrayPartition`, and pymanopt a list.
    geomstats concatenates the flattened factor points, or stacks them when every factor has the same shape.
-   *Nesting.*
    `ProductManifold::new` flattens a product factor into its blocks and multiplies the weights, so $(\mathcal{M}_1 \times \mathcal{M}_2) \times \mathcal{M}_3$ is a three-factor product.
    Manifolds.jl's `×` also flattens, without weights.
    manopt nests structs, pymanopt refuses a nested product, and geomstats treats it as an ordinary factor.
-   *Optional operations.*
    Here every `Manifold` method exists on every product.
    The exponential, logarithm, distance, parallel transport and `ehess2rhess` return `None` as soon as one factor lacks a closed form, and `minimize_rtr` then uses the Gauss-Newton Hessian.
    manopt defines a field such as `M.exp` or `M.ehess2rhess` only when all elements provide it, so a solver sees the absence before it starts.
    pymanopt always defines the method and the factor raises `NotImplementedError` at the call.
    Manifolds.jl fails with a method error, and geomstats falls back on a factor metric's numerical `exp_solver` and `log_solver` when it has them.
-   *Gradient and Hessian conversion.*
    manopt's and pymanopt's conversions have the blockwise structure of this one without the division by $w_i$.
    manopt warns that a factor's `egrad2rgrad` or `ehess2rhess` written with `storedb` and `key` arguments will not receive them inside a product.
-   *Distance and norm.*
    Here the distance is the weighted root sum of squares, which is the Riemannian distance of $g$.
    manopt, pymanopt and geomstats take the unweighted root sum of squares.
    Manifolds.jl's `distance(M, p, q, r)` and `norm(M, p, X, r)` take any $\ell_r$ norm of the factor values, $r = 2$ by default; for $r \ne 2$ they are not the distance and norm of the product metric.
-   *Typical distance and default radius.*
    Here it is the weighted root sum of squares; manopt and pymanopt take the unweighted one, and both start their trust regions at $\bar\Delta$ equal to it and $\Delta_0 = \bar\Delta / 8$, pymanopt falling back on $\sqrt{\dim \mathcal{M}}$.
    Manopt.jl's `trust_regions` defaults to $\bar\Delta = \sqrt{\dim \mathcal{M}}$ ([01ae0f6](https://github.com/JuliaManifolds/Manopt.jl/blob/01ae0f62e90c9def032e8e32b3e4ffc988ca9288/src/solvers/trust_regions.jl)), and ManifoldsBase.jl and geomstats have no typical distance.
-   *Random tangent vectors.*
    manopt and pymanopt scale each factor's random unit tangent vector by $k^{-1/2}$, so the tuple has unit length in the unweighted metric.
    There is no random point or tangent here.
-   *Retraction and transport choice.*
    Here each kind has one retraction and one transport.
    manopt uses each factory's `retr` and `transp`, with `retr2`, `isotransp` and `paralleltransp` when all elements provide them.
    Manifolds.jl composes `ProductRetraction`, `InverseProductRetraction` and `ProductVectorTransport` from any per-factor methods.
    pymanopt uses each factor's `retraction` and `transport`, and geomstats works with the exponential and logarithm.
-   *Parallel transport and isometry.*
    Here parallel transport is per factor, `None` when a factor has none, and `transport_is_isometric` is the conjunction over the factors.
    manopt offers `paralleltransp` and `isotransp` when every element does, Manifolds.jl and geomstats transport per factor, and pymanopt has no parallel transport.
-   *Injectivity radius and curvature.*
    Neither is exposed here.
    Manifolds.jl takes the minimum injectivity radius of the factors, computes the sectional curvature of a product exactly, bounds it by the factor bounds and 0 when $k \ge 2$, and calls a product flat when every factor is.
    geomstats takes the minimum injectivity radius, and `ScalarProductMetric` scales it by $\sqrt{s}$.
    `ProductManifold::is_flat` here means that every factor keeps a straight step on its set, a different notion from the zero curvature of Manifolds.jl's `is_flat`.
-   *Power manifold.*
    `ProductManifold::power` and `ManifoldKind::power` are unit-weight products, and Oblique stays a dedicated kind with its own column loops.
    manopt's `powermanifold`, Manifolds.jl's `PowerManifold` and geomstats' `NFoldManifold` are separate types, and pymanopt has none.
-   *Metric matrix and signature.*
    geomstats' `ProductRiemannianMetric` also offers a block-diagonal `metric_matrix` and the summed signature.
    The trait here has only `inner` on tangent vectors.
-   *Subspace spheres.*
    pymanopt's `SphereSubspaceIntersection` and `SphereSubspaceComplementIntersection` ([sphere.py at 35b022f](https://github.com/pymanopt/pymanopt/blob/35b022ff6043b893f32c903b428e811dea200dc2/src/pymanopt/manifolds/sphere.py)) are the counterparts of `within` and `excluding`, with the same projector, Weingarten term, exponential, logarithm and distance.
    They build the dense projector $Q Q^\top$ from a QR factorization of the spanning columns and do not refuse a rank-deficient matrix, and their retraction normalizes $x + v$ without projecting back onto the subspace, so rounding off it is not removed.
    manopt and Manifolds.jl have the spheres of fixed subspaces only: unit-norm symmetric matrices (manopt's `spheresymmetricfactory`, Manifolds.jl's `SphereSymmetricMatrices`) and, in manopt, the unit tangent vectors at one point of a manifold (`tangentspherefactory`).
    geomstats has none.
-   *Preconditioners.*
    manopt's `problem.precon` ([trs\_tCG.m](https://github.com/NicolasBoumal/manopt/blob/a149c85ca3cf23ca828d8bc6a6b125a4ac82d656/manopt/solvers/trustregions/trs_tCG.m)), pymanopt's `Problem(preconditioner=...)` and the `preconditioner` keyword of Manopt.jl's `trust_regions` approximate the inverse Hessian inside the truncated CG of the trust region.
    A `TangentMetric` gives the same iterates, but it is part of the manifold: `egrad2rgrad` returns the preconditioned gradient, so a first-order or sampled driver sees it as well, and the inner and outer stopping tests read lengths in it.
    geomstats has no trust region.
-   *Point hooks.*
    manopt's `options.hook` (`applyHook`) replaces the iterate at the start of every iteration of `trustregions` and `steepestdescent`, before the stopping test, and the solver evaluates the cost and gradient again when it moved; the other solvers do not call it.
    `minimize_rtr_gauged` has the same semantics after accepted steps, and `Solver::run_gauged` adds the rebase that a solver with curvature memory needs.
    Manopt.jl's solver states take callbacks before and after each step (`:BeforeStep`, `:Step`), which receive the state and may change it ([callback.jl at 01ae0f6](https://github.com/JuliaManifolds/Manopt.jl/blob/01ae0f62e90c9def032e8e32b3e4ffc988ca9288/src/base/state/callback.jl)).
    pymanopt and geomstats have no hook.


<a id="org26ce8b2"></a>

# Reconciliation with the earlier manifold branches

Two earlier branches carried product designs, and the product kind of this page supersedes both.

`feat/manifold-api-060` added `RiemannianManifold` in `cdc8f5dd`, a product of Stiefel and SPD factors over caller-owned column-major slices.
The OH-TST design pins rgmin at `d3f94d11`, a commit that already has it, and the product kind was built on top of it.
It has QR and polar Stiefel retractions, the Edelman, Arias and Smith exponential, SPD under the affine-invariant and log-Euclidean metrics, and a `gradient_step` helper.
It sits beside `ManifoldKind`, so no session can step it.
It implemented the old `Manifold` trait, whose only metric was the dot product, so a driver would have measured its affine-invariant SPD gradient $X \nabla f X$ in another metric; its trait implementation now forwards `inner` to the factor metrics.
Here the product is a `ManifoldKind` over every kind, with weights, and the trait carries `inner`, so gradient and metric agree.
Stiefel frames keep the column-major packing that both designs use.
`RiemannianManifold` stays, and its polar retraction and log-Euclidean metric have no `ManifoldKind`.
The `Spd` kind already took the affine-invariant gradient $X \nabla f X$, and it now carries the matching inner product $\mathrm{tr}(X^{-1} U X^{-1} V)$ and Hessian, so it enters a product in that metric.
Sessions still form their secant pairs in the dot product of the packed coordinates, as before, while `minimize_rtr` measures `Spd` in its own metric.

The rest of the branch, up to `7ea30e08`, binds `RiemannianManifold` outside Rust.
Its C handles are built by `rgmin_manifold_stiefel`, `rgmin_manifold_spd` and `rgmin_manifold_product`, released by `rgmin_manifold_free`, and passed to one function per slice method, `gradient_step` included.
C++ wraps them in `rgmin::Manifold`, and the Python module `rgmin.manifold` has `Stiefel`, `SPD` and `Product` classes over NumPy arrays, a product point being a tuple of factor arrays.
No function hands a handle to a session, whose manifold stays the unrelated `rgmin_manifold_t` token.
The three bindings stay as they were and reach no `ManifoldKind` product.
The handles hold ABI minor 41 and the line's minors 42 and 43 follow, so `rgmin_solver_set_power` takes minor 44.

A caller of `RiemannianManifold` who wants a session or the trust region moves over as follows:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left"><code>RiemannianManifold</code></th>
<th scope="col" class="org-left"><code>ManifoldKind</code></th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>stiefel(n, 1, _)</code></td>
<td class="org-left"><code>ManifoldKind::Sphere</code>, or <code>sphere_within</code> and <code>sphere_excluding</code> for a subspace</td>
</tr>

<tr>
<td class="org-left"><code>stiefel(n, p, _)</code></td>
<td class="org-left"><code>ManifoldKind::stiefel(n, p)</code>, with the QR retraction</td>
</tr>

<tr>
<td class="org-left"><code>spd(p, _)</code></td>
<td class="org-left"><code>ManifoldKind::Spd</code> at packed length \(p^2\), in the affine-invariant metric</td>
</tr>

<tr>
<td class="org-left">no flat factor</td>
<td class="org-left"><code>ManifoldKind::Euclidean</code></td>
</tr>

<tr>
<td class="org-left"><code>product(parts)</code></td>
<td class="org-left"><code>ManifoldKind::product</code> of <code>ProductFactor</code> values, with weights and tangent metrics</td>
</tr>

<tr>
<td class="org-left">slice methods writing into <code>out</code></td>
<td class="org-left"><code>Manifold</code> methods on <code>Array1</code> that return the result</td>
</tr>

<tr>
<td class="org-left"><code>gradient_step</code></td>
<td class="org-left">a session step, or <code>minimize_rtr</code></td>
</tr>

<tr>
<td class="org-left">C handles, C++ <code>rgmin::Manifold</code></td>
<td class="org-left">kept; <code>rgmin_solver_set_power</code> selects a power in a session</td>
</tr>

<tr>
<td class="org-left">Python <code>rgmin.manifold</code></td>
<td class="org-left">kept, over the slice methods</td>
</tr>
</tbody>
</table>

The plane of the design, `product([stiefel(D, 1, _), spd(1, _)])` with `RiemannianManifold`, becomes
`ManifoldKind::product([ProductFactor::new(ManifoldKind::sphere_excluding(e)?, D).with_weight(l2), ProductFactor::new(ManifoldKind::Euclidean, 1)])`
with `e` the excluded directions and `l2` the block weight $\ell^2$, or `with_metric` in place of `with_weight` for a harmonic metric on the normal.

`feat/power-manifold` (`3bb6f1df`) added `ManifoldKind::Power` over four factors (Euclidean, sphere, MwRigid, rigid quotient), `Solver::set_power`, a C setter with private factor codes 0 to 3 and the token `RGMIN_MANIFOLD_POWER`, and `minimize_rtr` in the dot-product metric.
The power is now a special case of the product.
`ManifoldKind::power` builds a unit-weight `ProductManifold` of any kind, so a power inherits the metric, exponential, logarithm, Hessian and chart instead of repeating them.
`Solver::set_power` keeps its signature, its rule for `MwRigid` masses (one molecule reused on every copy, or every copy's masses in order) and its per-block Eckart projection.
`rgmin_solver_set_power` keeps its signature but reads the factor as an `rgmin_manifold_t` token for any kind without shape parameters, in place of the private codes.
The branch never shipped, so no ABI user sees the change.
`RGMIN_MANIFOLD_POWER` is not added: a token cannot carry the factor, block and count, and the branch's default, two Euclidean blocks of length one, is just $\mathbb{R}^2$.
`minimize_rtr` is ported and now runs in `inner` with `ehess2rhess` and the typical distance.
Oblique stays its own kind, the power of spheres with column loops of its own.


<a id="orgf817d5e"></a>

# How the derivation is checked

`tests/product_manifold.rs` checks each section numerically on a quartic $\bar f(x) = \tfrac12 x^\top A x + b^\top x + \tfrac14 \sum_i x_i^4$ whose dense $A$ couples every block:

-   manopt's `checkgradient` and `checkhessian`: the remainders of the first- and second-order models along $t \mapsto R_x(t v)$ and $t \mapsto \exp_x(t v)$ have slopes 2 and 3 in $\log t$ (Boumal, sections 4.8 and 6.8), on the weighted $\mathbb{R}^4 \times S^3$ along both curves, on $S^2 \times SO(3)$ with the Hessian along the exponential, and on every kind with a closed-form geodesic times a weighted $\mathbb{R}^2$;
-   $g(\mathrm{grad} f, v) = \nabla \bar f^\top v$, tangency of the gradient and of the Hessian action, and symmetry of the Hessian in $g$;
-   the retraction and the exponential stay on the manifold for steps from $10^{-3}$ to $10$;
-   transport lands in the target tangent space, parallel transport preserves $g$, and a product of flat factors with weights 2, 1/4 and 1 transports isometrically;
-   the locking condition with $\beta = 1$: the velocity of $t \mapsto \exp_x(t d)$ at $t = 1$, by central differences, equals `parallel_transport_along(x, d, d)`;
-   $\log_x(\exp_x(v)) = v$, $\mathrm{dist}(x, \exp_x(v)) = \|v\|$ and the weighted typical distance;
-   L-BFGS, steepest descent, the session Newton-CG trust region and `minimize_rtr` reach the minimum of a coupled Rayleigh quotient and quadratic on the hyperplane, $\tfrac12 (p - c)^\top B (p - c) + n^\top A n + \kappa ((p - c)^\top n)^2$, whose value is the lowest eigenvalue of $A$;
-   the unit test of the chart section, and the rigid factors keep their mass-weighted centres.

`tests/subspace_sphere.rs` checks the sphere of a subspace:

-   the `checkgradient` and `checkhessian` slopes along the retraction and the exponential;
-   a kept basis and its excluded complement give the same projector, nearest point, tangent projection and retraction;
-   retraction and exponential steps stay on the sphere and in the subspace, the logarithm inverts the exponential at the right distance, and both transports land in the arrival tangent space;
-   the Hessian is the same for $\bar f$ and for $\bar f + (\|x\|^2 - 1)(c^\top x + 2)$, and equals the projected Hessian of the degree-zero extension by central differences, on `SubspaceSphere` and on `Sphere`;
-   L-BFGS and `minimize_rtr` find the lowest mode inside the subspace of a Rayleigh quotient whose two lowest modes are excluded, which the whole sphere reaches instead;
-   the session Newton-CG rates of the section on sphere Hessians.

`tests/plane_free_energy.rs` solves the harmonic plane of the section on the optimal dividing plane and asserts the comparisons made there.
It also checks that the gradient is the Riesz vector of the harmonic metric, runs the sampled trust-region loop, the gauged trust region and the gauged session, and checks that sessions refuse a tangent metric.


<a id="orgca561e9"></a>

# Limitations

-   The session's dogleg, Newton-CG and Newton-PCG steps model the Hessian of a curved factor without its Weingarten term, which is exact for the degree-zero extension only (see the section on sphere Hessians).
    On the hyperplane test, with $n^\top A n$ on the normal, the Newton-CG trust region converges at the rate $\lambda_1 / \lambda_2 = 1/2$ of the two lowest eigenvalues of $A$.
    `minimize_rtr` uses `ehess2rhess` and converges quadratically on the same problem: its gradient norm goes from $2 \times 10^{-4}$ to $2 \times 10^{-9}$ to $10^{-16}$ in its last three iterations.
-   A tangent metric costs its product the exponential, logarithm, distance, parallel transport and isometric transport, `typical_dist` ignores it, and its Hessian is the model $G^{-1} \mathrm{Hess}$, exact at critical points only.
    Sessions refuse it with `Error::ProductTangentMetric`.
-   The stochastic drivers of the OH-TST design (its item 5) are not here: stochastic gradient ascent with a step schedule and iterate averaging, the norm test on the sample size, a trust region that weighs a measured decrease against its standard error, and `NoiseFloor` driven by a caller's covariance.
    Each needs an interface for noisy values and gradients with their covariance, which the `Manifold` trait and the sessions do not have, and convergence tests on sampled objectives.
    `Control::noise_floor` stops a session at $c \sqrt{n_\mathrm{free}}\, \varepsilon_g$ for a caller's gradient noise $\varepsilon_g$, a scalar in place of a covariance, and `minimize_rtr` has no such stop.
    The deterministic core of such a driver is here: `egrad2rgrad` gives the preconditioned direction, `truncated_cg_riemannian` the trust-region step in the metric and `RtrRadius::update` the radius from a decrease measured outside the model.
-   Item 6, a Hessian-vector product with an error bar and a preconditioner built from the harmonic model, is not here either.
    The truncated CG treats every Hessian-vector product as exact, and its curvature and boundary tests would need a confidence level to take an estimate.
    A low-rank-plus-scalar model can already be supplied as a `TangentMetric`, or as an `hvp::Preconditioner` for the Euclidean `hvp::steihaug_pcg`.
-   The QR retraction of SO(3) is first order, so the numerical Hessian check on $S^2 \times SO(3)$ runs along the exponential.
-   `Multinomial` projects Fisher-orthogonally but keeps the dot product as its inner product, so its gradient does not represent the differential in that inner product, and the derivative tests leave it out.
    The doubly stochastic kinds carry the Fisher inner product and are consistent.
-   `Spd` carries the affine-invariant metric, in which its gradient, inner product and Hessian agree and its retraction $x + v + \tfrac12 v x^{-1} v$ is second order.
    Its transport, the identity on tangents, is not an isometry of that metric.
-   `Multinomial` and its doubly stochastic variants, `MwRigid` and `RigidQuotient` have no Hessian or geodesic closed forms, and `StiefelP` and `Spd` have a Hessian but no exponential, logarithm or parallel transport.
    A product containing one returns `None` for those operations; projection, retraction, transport and the gradient still work.
-   A weighted product refuses the settings that read raw coordinates, with `Error::ProductChart`.
-   The C ABI selects powers only, through `rgmin_solver_set_power`.
    A general weighted product has to carry its parameterized factors (the sizes of Oblique, Stiefel and the complex kinds), their weights and the basis of a subspace sphere across the ABI.
    A descriptor struct or composed handles like those of `RiemannianManifold` can do that, and either is a separate ABI design.
    Tangent metrics and gauges are Rust callbacks: the C ABI exposes sessions, which refuse tangent metrics, and has no entry for `minimize_rtr`, so both wait for a callback design.
    The Python module `rgmin.manifold` binds the slice methods of `RiemannianManifold` and no `ManifoldKind`, so a Python product waits for a binding of `ManifoldKind`.
