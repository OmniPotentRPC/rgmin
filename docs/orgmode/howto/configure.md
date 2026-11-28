

# Configure a solver

Every public configuration type is `#[non_exhaustive]`. Outside this
crate it cannot be written as a struct literal. Start from
`default()` or a named constructor and chain `with_*` setters, one per
field. The fields stay public, so they can still be read.

```rust
use rgmin::{Control, EigenParams, EigensolverKind, Method, Solver, StopNorm};

let control = Control::default()
    .with_maxiter(500)
    .with_gtol(0.01)
    .with_stop_norm(StopNorm::MaxAtom)
    .with_maxmove(0.2);
assert_eq!(control.maxmove, Some(0.2));
let solver = Solver::new(Method::lbfgs_with_memory(8), control, 21);

let eigen = EigenParams::default()
    .with_kind(EigensolverKind::Lanczos)
    .with_krylov(8);
```

A setter for an `Option` field takes `impl Into<Option<T>>`:
`with_maxmove(0.2)` stores `Some(0.2)` and `with_maxmove(None)`
clears it. `Control` is `Clone` and not `Copy`, so a setter chain on a
control that is used again starts from `control.clone()`.


## Types built this way

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Type</td>
<td class="org-left">Start from</td>
</tr>

<tr>
<td class="org-left"><code>Control</code></td>
<td class="org-left"><code>Control::default()</code></td>
</tr>

<tr>
<td class="org-left"><code>NoiseFloor</code></td>
<td class="org-left"><code>NoiseFloor::new(eps_g)</code></td>
</tr>

<tr>
<td class="org-left"><code>EigenParams</code>, <code>SlepcParams</code>, <code>EigenExaParams</code>, <code>DlaFutureParams</code>, <code>ElpaParams</code></td>
<td class="org-left"><code>default()</code></td>
</tr>

<tr>
<td class="org-left"><code>ScgParams</code>, <code>ScgOptions</code>, <code>HighsOptions</code>, <code>HighsStep</code></td>
<td class="org-left"><code>default()</code></td>
</tr>

<tr>
<td class="org-left"><code>ExpParams</code>, <code>LindhParams</code>, <code>FfParams</code></td>
<td class="org-left"><code>default()</code></td>
</tr>

<tr>
<td class="org-left"><code>EonCg</code></td>
<td class="org-left"><code>EonCg::default()</code></td>
</tr>

<tr>
<td class="org-left"><code>QuickMinOptions</code></td>
<td class="org-left"><code>rgmin()</code>, <code>eon()</code>, <code>per_atom()</code></td>
</tr>

<tr>
<td class="org-left"><code>FireConfig</code></td>
<td class="org-left"><code>FireConfig::ase()</code></td>
</tr>

<tr>
<td class="org-left"><code>TrustRegion</code></td>
<td class="org-left"><code>TrustRegion::new(delta)</code></td>
</tr>

<tr>
<td class="org-left"><code>IrcTrust</code></td>
<td class="org-left"><code>IrcTrust::new(d1, sqrtm, dx)</code>, <code>from_atom_masses</code></td>
</tr>

<tr>
<td class="org-left"><code>RtrRadius</code></td>
<td class="org-left"><code>RtrRadius::new(radius_max)</code></td>
</tr>

<tr>
<td class="org-left"><code>LindhSpec</code></td>
<td class="org-left"><code>LindhSpec::new(z, c_stab, cell)</code></td>
</tr>

<tr>
<td class="org-left"><code>LineSearchOptions</code></td>
<td class="org-left"><code>default()</code>, <code>with_objective_roundoff</code></td>
</tr>

<tr>
<td class="org-left"><code>Report</code></td>
<td class="org-left"><code>Report::new(value, coords)</code>, for a host that writes its own</td>
</tr>

<tr>
<td class="org-left"><code>DeflatedPair</code></td>
<td class="org-left"><code>DeflatedPair::new</code>, when supplying a pair to deflation</td>
</tr>
</tbody>
</table>

Results (`LowestMode`, `Eigenpair`, `Eigenpairs`, `RestrictedStep`,
`TcgResult`, `RtrMin`, `EonCgReport`, `LineOutcome`, `NoiseEstimate`,
`ValueGradientNoise`, `Ode12rVerdict`, `ConstraintReport`,
`ElpaSpectrum`, `FireTrace`, `EonFeat`, `KantorovichBall`,
`AdjointFunctionalBar`, `MorseGap`, `TridiagEigh`,
`QuadratureEstimate`, `PcgResult`, `DeflatedPair`,
`ExtremePairsReport`, `SlqLogDetReport`, `GaussLegendre`,
`LobattoShapes`, `MagnusStep`, `Exponential`) are non-exhaustive too.
Read their fields; a destructuring pattern needs `..`.


## Methods and step rules

A variant that carries parameters is non-exhaustive. Build it with its
constructor and match it with `..`:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Variant</td>
<td class="org-left">Constructor</td>
</tr>

<tr>
<td class="org-left"><code>Method::Lbfgs</code></td>
<td class="org-left"><code>Method::lbfgs()</code>, <code>Method::lbfgs_with_memory(m)</code></td>
</tr>

<tr>
<td class="org-left"><code>Method::Diis</code></td>
<td class="org-left"><code>Method::diis()</code>, <code>Method::diis_with_memory(m)</code></td>
</tr>

<tr>
<td class="org-left"><code>Method::Fire</code></td>
<td class="org-left"><code>Method::fire(kind)</code></td>
</tr>

<tr>
<td class="org-left"><code>Method::Newton</code></td>
<td class="org-left"><code>Method::newton(kind)</code></td>
</tr>

<tr>
<td class="org-left"><code>Method::Nlcg</code></td>
<td class="org-left"><code>Method::nlcg(c)</code>, <code>Method::nlcg_with_restart(c, r)</code></td>
</tr>

<tr>
<td class="org-left"><code>Method::Adam</code></td>
<td class="org-left"><code>Method::adam()</code>, <code>Method::adam_with(beta1, beta2, eps)</code></td>
</tr>

<tr>
<td class="org-left"><code>Method::Pso</code></td>
<td class="org-left"><code>Method::pso()</code>, <code>Method::pso_with(n, w, c1, c2)</code></td>
</tr>

<tr>
<td class="org-left"><code>LineSearch::Brent</code> &#x2026; <code>Wolfe</code></td>
<td class="org-left"><code>LineSearch::brent</code>, <code>backtracking</code>, <code>goldstein</code>, <code>wolfe</code></td>
</tr>

<tr>
<td class="org-left"><code>LbfgsStep::Capped</code>, <code>Scaled</code></td>
<td class="org-left"><code>LbfgsStep::capped(max_atom)</code>, <code>scaled(max_atom)</code></td>
</tr>

<tr>
<td class="org-left"><code>Restart::Njws</code>, <code>Powell</code></td>
<td class="org-left"><code>Restart::njws_at(t)</code>, <code>powell_at(t)</code></td>
</tr>

<tr>
<td class="org-left"><code>Conjugacy::Hybrid</code></td>
<td class="org-left"><code>Conjugacy::hybrid(a, b, take_max)</code></td>
</tr>
</tbody>
</table>

`LbfgsStep::default()` is still the eOn cap,
`LbfgsStep::capped(EON_MAX_MOVE)`.

```rust
use rgmin::Method;

let method = Method::lbfgs_with_memory(8);
let memory = match method {
    Method::Lbfgs { memory, .. } => memory,
    _ => 10,
};
assert_eq!(memory, 8);
```


## New knobs are additive

A field added to one of these types gets a default that keeps the
earlier behaviour, and a setter. A parameter added to a method variant
gets a default, and the constructors above keep their signatures. A
caller that builds through `default()`, constructors and setters, and
matches with `..`, compiles against the next minor release unchanged.


## Errors

Every fallible entry point returns `rgmin::Result<T>`, whose error is
`rgmin::Error`: the eigenpair calls (`lowest_eigenpair`, `lowest_mode`
and its backend arms, `lowest_eigenpairs`), the minimizers, FIRE 2.0
(`fire2_displacement` stalls with `Error::FireStalled`), the manifold
factories (`StiefelNp::new`, `Grassmann::new`, `Unitary::new` refuse a
shape with `Error::ManifoldShape`) and `Vector::try_on`
(`Error::UnsupportedDevice`). `Error` is non-exhaustive, so a `match`
on it needs a wildcard arm.
