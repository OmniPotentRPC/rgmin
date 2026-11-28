

# API reference

Two published surfaces, one implementation.


## Rust

`cargo doc --no-deps` in a checkout.
The crate root is `rgmin`.
The types a host works with:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Type</th>
<th scope="col" class="org-left">Role</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>Solver</code></td>
<td class="org-left">One stepper step. Hosts own the loop.</td>
</tr>

<tr>
<td class="org-left"><code>Lbfgs</code></td>
<td class="org-left">Persistent limited-memory quasi-Newton; <code>minimize_box</code> keeps iterates in bounds.</td>
</tr>

<tr>
<td class="org-left"><code>Method</code> / <code>Control</code> / <code>LineSearch</code></td>
<td class="org-left">Which step, and how far.</td>
</tr>

<tr>
<td class="org-left"><code>ManifoldKind</code></td>
<td class="org-left"><code>project</code> / <code>retract</code> / <code>transport</code>.</td>
</tr>

<tr>
<td class="org-left"><code>RiemannianManifold</code></td>
<td class="org-left">Stiefel, SPD and product handles; see the manifold page.</td>
</tr>

<tr>
<td class="org-left"><code>IrcTrust</code></td>
<td class="org-left">GS2 / Sella MW-sphere equality.</td>
</tr>

<tr>
<td class="org-left"><code>lowest_eigenpair</code> / <code>lowest_mode</code></td>
<td class="org-left">Matrix-free lowest Hessian pair.</td>
</tr>

<tr>
<td class="org-left"><code>lowest_eigenpairs</code></td>
<td class="org-left">Block LOBPCG. Residuals and an optional <code>T</code> (Knyazev 2001).</td>
</tr>

<tr>
<td class="org-left"><code>EigensolverKind</code> / <code>EigenParams</code></td>
<td class="org-left">Closed eigen waist.</td>
</tr>

<tr>
<td class="org-left"><code>HessianVector</code> / <code>FdHvp</code></td>
<td class="org-left"><code>H v</code> without forming <code>H</code>.</td>
</tr>

<tr>
<td class="org-left"><code>TryHessianVector</code></td>
<td class="org-left">The same callbacks, each one allowed to fail.</td>
</tr>

<tr>
<td class="org-left"><code>Solver::step_hvp</code></td>
<td class="org-left">Analytic product for <code>newton-cg</code> / <code>newton-pcg</code>.</td>
</tr>

<tr>
<td class="org-left"><code>Solver::try_step_hvp</code></td>
<td class="org-left">That step. The first host error is <code>Error::Host</code>.</td>
</tr>

<tr>
<td class="org-left"><code>TryApplyHessian</code></td>
<td class="org-left">A block-eigen product that can fail.</td>
</tr>

<tr>
<td class="org-left"><code>try_lowest_eigenpairs</code></td>
<td class="org-left"><code>lowest_eigenpairs</code> on that product, without <code>T</code>.</td>
</tr>

<tr>
<td class="org-left"><code>try_lowest_eigenpairs_precond</code></td>
<td class="org-left"><code>lowest_eigenpairs_precond</code> on that product, with an optional <code>T</code>.</td>
</tr>

<tr>
<td class="org-left"><code>kantorovich_ball</code> / <code>KantorovichBall</code></td>
<td class="org-left">Newton&#x2013;Kantorovich existence ball from supplied majorants (Kantorovich and Akilov 1982; Gragg and Tapia 1974).</td>
</tr>

<tr>
<td class="org-left"><code>adjoint_functional_bar</code> / <code>AdjointFunctionalBar</code></td>
<td class="org-left">Dual-weighted residual functional bound and remainder (Becker and Rannacher 2001).</td>
</tr>

<tr>
<td class="org-left"><code>certify_morse_gap</code> / <code>MorseGap</code></td>
<td class="org-left">Self-adjoint spectral gap certified by Weyl's inequality (Weyl 1912).</td>
</tr>
</tbody>
</table>

`newton-cg` and `newton-pcg` are `Method::from_name` tokens. The
session counts `gradient_calls` and `hvp_calls` separately. A
forward-difference product on `step` costs one gradient and is
charged as a product.

`kantorovich_ball`, `adjoint_functional_bar`, and `certify_morse_gap`
combine caller-supplied majorants in standard `f64` floating-point
arithmetic without outward rounding. Given a primal norm and its
induced operator norm, a Jacobian Lipschitz bound on `B(x0, 2 eta)`
suffices for the Kantorovich ball ([10.1016/B978-0-08-023036-8.50024-2](https://doi.org/10.1016/B978-0-08-023036-8.50024-2),
[10.1137/0711002](https://doi.org/10.1137/0711002)) .
`adjoint_functional_bar` pairs `first_order = |lambda^T F(x0)|` for an
exact adjoint `J(x0)^T lambda = grad q(x0)` with `gradient_norm` in the
dual norm and `hessian_bound` in the induced operator norm from the
primal space to its dual ([10.1017/S0962492901000010](https://doi.org/10.1017/S0962492901000010))
. `certify_morse_gap`
requires `operator_lipschitz` to bound the spectral-norm Lipschitz
constant of the projected self-adjoint operator and subtracts that
perturbation plus `eigensolver_roundoff` from `computed_gap` by Weyl's
inequality ([10.1007/BF01456804](https://doi.org/10.1007/BF01456804))
.

The generated Sphinx pages under `crates/rgmin/` (when the docs
environment is built with `sphinxcontrib-rust`) are the same
rustdoc, exported.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Type</th>
<th scope="col" class="org-left">Role</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>magnus::magnus_step</code> / <code>magnus::MagnusStep</code></td>
<td class="org-left">Sixth-order Magnus map with an embedded fourth-order exponential.</td>
</tr>

<tr>
<td class="org-left"><code>magnus::expm</code> / <code>magnus::Exponential</code> / <code>magnus::gauss_legendre_three</code> / <code>magnus::MagnusError</code></td>
<td class="org-left">Padé matrix exponential, Gauss-Legendre nodes and typed refusal.</td>
</tr>
</tbody>
</table>


## C

`include/rgmin.h` is the C ABI. `cargo cinstall` installs that
header and `dlpack/dlpack.h`. `include/xts.h` is the compatibility
alias. The hourglass is `rgmin_solver_create` /
`rgmin_solver_step` / `rgmin_solver_free`, plus `rgmin_minimize`.
Tensors are dlpk `DLManagedTensorVersioned`. HTML for this waist
is Doxygen plus doxyYoda; see .

`rgmin_minimize_eindir` takes an `eindir_objective_t` and an
`eindir_abi_stamp_t` that no `rgmin_` function builds. librgmin
therefore re-exports the C ABI of eindir-core as public API:
`eindir_core_abi_compatible`, `eindir_core_abi_family`,
`eindir_core_abi_stamp`, `eindir_core_version`, `eindir_last_error` and
the `eindir_objective_` functions `new`, `free`, `eval`, `grad`,
`has_grad`, `dim`, `bounds_low` and `bounds_high`. A C caller that links
only librgmin builds the objective with `eindir_objective_new` and the
stamp with `eindir_core_abi_stamp`. These names have been exported
since 0.7.0, and `scripts/check_exports.py` fails if one goes missing
or if eindir-core exports a name not on this list.


## What belongs next door

Band, min-mode, and IRC **sessions** are [rgsaddle](https://github.com/OmniPotentRPC/rgsaddle). Potentials
are [rgpot](https://github.com/OmniPotentRPC/rgpot).
