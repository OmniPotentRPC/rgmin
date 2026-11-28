
# Table of Contents



The Lean development under `proofs/lean/Rgmin/` pins the exact
rational facts the solvers' termination and honesty arguments rest
on. It builds on Mathlib, so the statements are the real ones: widths
and radii over the reals, \`Real.sqrt\` in the boundary root rather
than a named-root hypothesis, actual limits for the termination
envelopes, and the Gram positive-semidefiniteness of the sketch as a
matrix theorem. Check it with `lake exe cache get && lake build` in
`proofs/lean/`; the cache makes Mathlib a download, not a compile.

The division of labor is deliberate. The Lean side proves the exact
algebra and the budget envelopes &#x2013; facts about the formulas, true in
every execution. The Rust tests hold the floating-point
implementation to those facts at concrete tolerances. A claim without
a row in both columns is an opinion.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Theorem (Rgmin.*)</th>
<th scope="col" class="org-left">Pins</th>
<th scope="col" class="org-left">Code</th>
<th scope="col" class="org-left">Rust test</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>guarded_above</code>, <code>guarded_below</code></td>
<td class="org-left">zoom trials land in the interior band</td>
<td class="org-left"><code>linesearch/zoom.rs</code></td>
<td class="org-left"><code>the_cubic_zoom_keeps_the_evaluation_budget</code></td>
</tr>

<tr>
<td class="org-left"><code>zoom_shrinks</code></td>
<td class="org-left">bracket loses &gt;= 10 percent per step</td>
<td class="org-left"><code>linesearch/zoom.rs</code></td>
<td class="org-left">(same)</td>
</tr>

<tr>
<td class="org-left"><code>width_envelope</code>, <code>_tendsto</code></td>
<td class="org-left">geometric width envelope, and it reaches zero</td>
<td class="org-left"><code>linesearch/zoom.rs</code></td>
<td class="org-left">(same)</td>
</tr>

<tr>
<td class="org-left"><code>armijo_strict</code></td>
<td class="org-left">an accepted point strictly improves</td>
<td class="org-left"><code>linesearch/conditions.rs</code></td>
<td class="org-left"><code>armijo_and_curvature_match_xtsci</code></td>
</tr>

<tr>
<td class="org-left"><code>boundary_on_sphere</code></td>
<td class="org-left">the boundary root lands on the trust sphere</td>
<td class="org-left"><code>hvp.rs::boundary_tau</code></td>
<td class="org-left"><code>negative_curvature_walks_to_the_trust_boundary</code></td>
</tr>

<tr>
<td class="org-left"><code>boundary_tau_nonneg</code></td>
<td class="org-left">a boundary exit never retracts</td>
<td class="org-left"><code>hvp.rs::boundary_tau</code></td>
<td class="org-left">(same)</td>
</tr>

<tr>
<td class="org-left"><code>ray_further_is_lower</code></td>
<td class="org-left">negative curvature: boundary is optimal greed</td>
<td class="org-left"><code>hvp.rs::steihaug_pcg</code></td>
<td class="org-left">(same)</td>
</tr>

<tr>
<td class="org-left"><code>step_scale_invariant</code></td>
<td class="org-left">CG step ignores preconditioner scale</td>
<td class="org-left"><code>hvp.rs::steihaug_pcg</code></td>
<td class="org-left"><code>nystrom_flattens_a_decaying_spectrum</code></td>
</tr>

<tr>
<td class="org-left"><code>beta_scale_invariant</code></td>
<td class="org-left">conjugacy weight ignores preconditioner scale</td>
<td class="org-left"><code>hvp.rs::steihaug_pcg</code></td>
<td class="org-left"><code>the_preconditioned_boundary_lives_in_the_sketch_metric</code></td>
</tr>

<tr>
<td class="org-left"><code>metric_update_scales</code></td>
<td class="org-left">tracked M-norm recurrence is scale-consistent</td>
<td class="org-left"><code>hvp.rs::steihaug_pcg</code></td>
<td class="org-left">(same)</td>
</tr>

<tr>
<td class="org-left"><code>shrink_ge_floor</code>, <code>shrink_pos</code></td>
<td class="org-left">the trust radius stays positive</td>
<td class="org-left"><code>trust.rs::update_radius</code></td>
<td class="org-left"><code>hvp.rs</code> suite</td>
</tr>

<tr>
<td class="org-left"><code>shrink_strict</code>, <code>collapse_envelope</code>, <code>collapse_tendsto</code></td>
<td class="org-left">rejections decay the radius to any floor</td>
<td class="org-left"><code>hvp.rs::minimize_newton_cg</code></td>
<td class="org-left"><code>TrustCollapsed</code> paths</td>
</tr>

<tr>
<td class="org-left"><code>grow_le_cap</code>, <code>grow_ge</code></td>
<td class="org-left">growth is capped and never loses ground</td>
<td class="org-left"><code>trust.rs::update_radius</code></td>
<td class="org-left"><code>dogleg_stays_on_the_trust_sphere</code></td>
</tr>

<tr>
<td class="org-left"><code>rejection_propagates</code></td>
<td class="org-left">an uphill step cannot be accepted</td>
<td class="org-left"><code>trust.rs::reduction_ratio</code></td>
<td class="org-left"><code>an_uphill_everywhere_oracle_is_refused_not_moved</code></td>
</tr>

<tr>
<td class="org-left"><code>sketch_posSemidef</code>, <code>sketch_form_nonneg</code>, <code>sumSq_nonneg</code></td>
<td class="org-left">the Gram sketch is PSD, as a matrix theorem</td>
<td class="org-left"><code>hvp.rs::NystromPrecond</code></td>
<td class="org-left"><code>nystrom_flattens_a_decaying_spectrum</code></td>
</tr>

<tr>
<td class="org-left"><code>shifted_weight_pos</code>, <code>equalized_weight_bounds</code></td>
<td class="org-left">the solve's weights are positive, damping only</td>
<td class="org-left"><code>hvp.rs::NystromPrecond::solve</code></td>
<td class="org-left">(same)</td>
</tr>
</tbody>
</table>

The narrative behind each block lives in the explanation pages:
,
,
,
,
.
