

# From ASE, SciPy and eOn

The Rust default is the eOn L-BFGS cap (`EON_MAX_MOVE = 0.2`), not a line search. `LbfgsStep::Wolfe` opts into strong Wolfe. `LbfgsStep::Scaled` is ASE `LBFGS` with `use_line_search=False`: shorten the step, keep the pairs.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Call site</td>
<td class="org-left">rgmin</td>
</tr>

<tr>
<td class="org-left">ASE <code>BFGS(atoms, trajectory=, logfile=)</code></td>
<td class="org-left"><code>rgmin.BFGS</code>. The initial Hessian is the identity, not <code>alpha * I</code>.</td>
</tr>

<tr>
<td class="org-left">ASE <code>LBFGS(..., use_line_search=False)</code></td>
<td class="org-left"><code>rgmin.LBFGS</code>. Same length control.</td>
</tr>

<tr>
<td class="org-left">ASE <code>LBFGS(..., use_line_search=True)</code></td>
<td class="org-left"><code>rgmin.LBFGS(use_line_search=True)</code>, the strong Wolfe search.</td>
</tr>

<tr>
<td class="org-left">ASE <code>FIRE(dt=, dtmax=, Nmin=, finc=, fdec=, astart=, fa=)</code></td>
<td class="org-left"><code>rgmin.FIRE</code>. <code>dtmax</code> is kept. A Rust session would use <code>2.5 * dt</code>.</td>
</tr>

<tr>
<td class="org-left">eOn <code>opt_method = lbfgs</code>, <code>max_move = 0.2</code></td>
<td class="org-left"><code>Lbfgs::default</code> or <code>rgmin.LBFGS(eon_cap=True)</code>.</td>
</tr>

<tr>
<td class="org-left">eOn <code>cg</code></td>
<td class="org-left"><code>minimize_eon_cg</code>. <code>Method::eon_cg</code> is only the coefficient and the restart.</td>
</tr>

<tr>
<td class="org-left">SciPy <code>method="L-BFGS-B"</code></td>
<td class="org-left"><code>Lbfgs::minimize_box</code> for bounds. <code>LBFGSB_FTOL</code> is SciPy's <code>ftol</code>. The stop there is the projected gradient.</td>
</tr>

<tr>
<td class="org-left">SciPy <code>method="CG"</code></td>
<td class="org-left"><code>Method::polak_ribiere_plus</code>, <code>max(0, β)</code>.</td>
</tr>
</tbody>
</table>

`run(fmax, steps)`, `attach`, `trajectory` and `logfile` are ASE's. The classes subclass `ase.optimize.optimize.Optimizer`.

The scaled step is not the one to copy when the function has a flat well. On the 10-dimensional Rosenbrock start, ASE's length control stops near `x0 = -1` with value about 4 and an infinity norm below `1e-5`. The eOn cap and the Wolfe search both reach the minimum at 0. SciPy's L-BFGS-B does too. `tests/scaled_rosenbrock10.rs` and `python/tests/test_ase_optimizer.py` record that. The tolerance is the same; the step rule is what leaves the well.

`Stepper::step` is the one-iteration seam rgsaddle already calls on `Solver`.
