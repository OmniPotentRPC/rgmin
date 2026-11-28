

# Module `rgmin`

`Method` is an `enum.IntEnum` of the C method ids. Members are
UPPER<sub>SNAKE</sub>, for example `LBFGS` and `POLAK_RIBIERE`. The older
CamelCase spellings are aliases of the same values.
`Stop` is why a step ended. Its members are `TOLERANCE`,
`NOISE_FLOOR`, `STEP_TOLERANCE`, `BUDGET`, `LINE_SEARCH`,
`POLISH_UNMET`, `ENERGY_TOLERANCE`, `EVAL_BUDGET`,
`MACHINE_PRECISION`, and `LINE_SEARCH_ROUNDING`, again with CamelCase
aliases.
`Report` has `value`, `steps`, `grad_norm`, and `stop`.
`Report.converged` is true when `stop` is `TOLERANCE`.
`backward_error_floor(value, x)` is the first-order stationarity
floor `n * eps * (1 + |value| + sum |x_i|)`.
`NumericalError` is an `ArithmeticError`.
`NotConverged` is a `RuntimeError` with a `report` attribute holding
the terminal `Report`.
`SolverConfig(method`"lbfgs", \*\*options)= validates options.
`Solver(n, config)` is the session. `step(f_and_grad, x)` takes one
iteration. `step_hess(f_and_grad, hessian, x)` is the same for Newton,
RFO, and dogleg.
`minimize(f_and_grad, x0, method`"lbfgs", \*\*options)= returns
`(report, x)`. `x0` must already be a contiguous float64 vector.
An `lbfgs` run that spends its budget with `x` unmoved stops on
`LINE_SEARCH`, as the C `rgmin_minimize` does, not on `BUDGET`.
`require_convergence`, `error_on_unmet`, and `backward_polish` are flags
and default to off. `require_convergence` raises `NotConverged` when the
gradient norm is not below `gtol`. `error_on_unmet` raises
`StopUnmetError` when the run stops on `BUDGET`, `LINE_SEARCH`,
`LINE_SEARCH_ROUNDING`, `MACHINE_PRECISION`, `POLISH_UNMET`, or
`EVAL_BUDGET`. It is a
`RuntimeError` whose `report` attribute is the `Report` of the stop.
`backward_polish` is stored on the session. This step loop does not form
a Hessian, so
the polish itself is the C `rgmin_minimize_hess_opts` and
`rgmin_minimize_hvp_opts` entries.

`__version__` is the version string of the extension.

`c1`, `c2`, and `linesearch_maxiter` are the line-search constants.
They are accepted only together with `linesearch`, one of `brent`,
`backtracking`, `goldstein`, `wolfe`, or `ase`. Passing a non-default
constant alone raises `ValueError`. So does a `linesearch` the chosen
step ignores: any kind with `accept="step"` or on a method that takes
no line search, any kind but `wolfe` and `ase` on `lbfgs`, and `ase`
on a method other than `bfgs` and `lbfgs`.


# Two packages, one import name

Both packages install the import package `rgmin`, so an environment holds one of them.
The PyPI wheel (the repository root, nanobind) has `SolverConfig`, `Solver`, `minimize`, `Report`, `Stop`, `Method`, `NotConverged` and `rgmin.ase`.
It has no manifolds, no `Session` and no `lowest_eigenpairs`.
The maturin package under `python/` is not on PyPI and has `Session`, `LineSearch`, `LbfgsPairs`, `Stiefel`, `SPD`, `Product`, `lowest_eigenpairs` and `rgmin.optimize`.
shows how to build each one.


# Options of `SolverConfig` and `minimize`

Names are lower case, and `-` or a space reads as `_`.
An unknown name raises `ValueError` and lists the valid ones.
The PyPI wheel starts from `maxiter` 100, `gtol` 1e-8, `istep` 0.1, `memory` 10 and `maxmove` 0.
Those differ from the Rust `Control::default` (`gtol` 1e-5, `istep` 1) and from the C `rgmin_control_default`, so a script that moves between bindings should set them.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Option</th>
<th scope="col" class="org-left">Value</th>
<th scope="col" class="org-left">Default</th>
<th scope="col" class="org-left">Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>maxiter</code></td>
<td class="org-left">int &gt;= 0</td>
<td class="org-left">100</td>
<td class="org-left">Steps <code>minimize</code> takes. Zero returns a report without a step. A <code>Solver</code> takes one step per <code>step</code> call.</td>
</tr>

<tr>
<td class="org-left"><code>gtol</code></td>
<td class="org-left">float</td>
<td class="org-left">1e-8</td>
<td class="org-left">Stop when the gradient norm is below it.</td>
</tr>

<tr>
<td class="org-left"><code>istep</code></td>
<td class="org-left">float</td>
<td class="org-left">0.1</td>
<td class="org-left">Initial step scale. For FIRE it is the initial time step.</td>
</tr>

<tr>
<td class="org-left"><code>memory</code></td>
<td class="org-left">int &gt;= 0</td>
<td class="org-left">10</td>
<td class="org-left">L-BFGS pairs.</td>
</tr>

<tr>
<td class="org-left"><code>nbfgscorr</code></td>
<td class="org-left">int &gt;= 1</td>
<td class="org-left">10</td>
<td class="org-left">Correction pairs for <code>gromacs_lbfgs</code>. Other methods reject it.</td>
</tr>

<tr>
<td class="org-left"><code>maxmove</code></td>
<td class="org-left">float</td>
<td class="org-left">0</td>
<td class="org-left">Euclidean cap on the step. Zero keeps the method's own caps, and a negative value leaves every cap off.</td>
</tr>

<tr>
<td class="org-left"><code>atom_maxmove</code></td>
<td class="org-left">float</td>
<td class="org-left">unset</td>
<td class="org-left">Per-atom cap on the longest atom displacement. <code>maxstep</code> is the same option. Unset keeps the method's default, 0.2 for L-BFGS.</td>
</tr>

<tr>
<td class="org-left"><code>accept</code></td>
<td class="org-left"><code>none</code>, <code>energy</code>, <code>nonmonotone</code>, <code>step</code></td>
<td class="org-left">unset</td>
<td class="org-left">How a step is taken. Unset keeps the C session's <code>energy</code>.</td>
</tr>

<tr>
<td class="org-left"><code>linesearch</code></td>
<td class="org-left"><code>brent</code>, <code>backtracking</code> (<code>armijo</code>), <code>goldstein</code>, <code>wolfe</code>, <code>ase</code></td>
<td class="org-left">unset</td>
<td class="org-left">Line search. L-BFGS searches none unless this is <code>wolfe</code> or <code>ase</code>. <code>ase</code> is ASE 3.29's search, on <code>bfgs</code> and <code>lbfgs</code> only.</td>
</tr>

<tr>
<td class="org-left"><code>h0</code></td>
<td class="org-left">finite float &gt; 0</td>
<td class="org-left">unset</td>
<td class="org-left">Initial inverse-Hessian scale of <code>bfgs</code> and <code>lbfgs</code>. Unset keeps the identity BFGS matrix and the Nocedal-Wright L-BFGS scale.</td>
</tr>

<tr>
<td class="org-left"><code>stpmax</code></td>
<td class="org-left">float</td>
<td class="org-left">unset</td>
<td class="org-left">Step-multiplier bound of the ASE line search. Requires <code>linesearch="ase"</code>. A value below 1 fails each search, as in ASE.</td>
</tr>

<tr>
<td class="org-left"><code>c1</code>, <code>c2</code></td>
<td class="org-left">float</td>
<td class="org-left">1e-4, 0.9</td>
<td class="org-left">Wolfe and Armijo constants. Need <code>linesearch</code>.</td>
</tr>

<tr>
<td class="org-left"><code>linesearch_maxiter</code></td>
<td class="org-left">int &gt;= 1</td>
<td class="org-left">20</td>
<td class="org-left">Trial budget of the line search. Needs <code>linesearch</code>.</td>
</tr>

<tr>
<td class="org-left"><code>beta</code></td>
<td class="org-left">a CG variant name, or <code>cg</code></td>
<td class="org-left">unset</td>
<td class="org-left">Conjugacy coefficient. Only for a CG method. <code>cg</code> is Polak-Ribiere.</td>
</tr>

<tr>
<td class="org-left"><code>fire_variant</code></td>
<td class="org-left"><code>rgmin</code>, <code>guenole2020</code> (<code>fire2</code>)</td>
<td class="org-left">unset</td>
<td class="org-left">FIRE schedule.</td>
</tr>

<tr>
<td class="org-left"><code>qn_step</code></td>
<td class="org-left"><code>lbfgs</code>, <code>newton</code>, <code>rfo</code></td>
<td class="org-left">unset</td>
<td class="org-left">Quasi-Newton step. Newton and RFO need <code>Solver.step_hess</code>.</td>
</tr>

<tr>
<td class="org-left"><code>neb_guards</code></td>
<td class="org-left">bool</td>
<td class="org-left">unset</td>
<td class="org-left">Opt-in L-BFGS NEB guards. Other methods ignore it.</td>
</tr>

<tr>
<td class="org-left"><code>periodic</code></td>
<td class="org-left">bool</td>
<td class="org-left">unset</td>
<td class="org-left">Periodic cell. Drops rotation.</td>
</tr>

<tr>
<td class="org-left"><code>project_rigid</code></td>
<td class="org-left">bool</td>
<td class="org-left">unset</td>
<td class="org-left">eOn <code>lbfgs_project_rigid</code>.</td>
</tr>

<tr>
<td class="org-left"><code>extra_updates</code></td>
<td class="org-left">int &gt;= 0</td>
<td class="org-left">unset</td>
<td class="org-left">Al-Baali extra updates. Accepted and without effect.</td>
</tr>

<tr>
<td class="org-left"><code>cautious_eps</code>, <code>cautious_alpha</code></td>
<td class="org-left">float, set together</td>
<td class="org-left">unset</td>
<td class="org-left">Li-Fukushima filter: a pair is dropped when s.y &lt; eps * norm(s)<sup>2</sup> * norm(g)<sup>alpha</sup>.</td>
</tr>

<tr>
<td class="org-left"><code>noise_eps</code>, <code>noise_c</code></td>
<td class="org-left">float</td>
<td class="org-left">0, 0</td>
<td class="org-left">Gradient noise floor. <code>noise_eps</code> &lt;= 0 leaves it off, and <code>noise_c</code> &lt;= 0 selects 3.</td>
</tr>

<tr>
<td class="org-left"><code>objective_roundoff</code></td>
<td class="org-left">float</td>
<td class="org-left">unset</td>
<td class="org-left">Relative value window of the approximate Wolfe test, in place of the default four epsilon. Zero permits no rise, and a negative or non-finite value raises <code>ValueError</code>.</td>
</tr>

<tr>
<td class="org-left"><code>require_convergence</code></td>
<td class="org-left">bool</td>
<td class="org-left">False</td>
<td class="org-left">Raise <code>NotConverged</code> from <code>minimize</code> when the gradient norm is not below <code>gtol</code>. A first-order session step that took no move and left its state unchanged reports <code>STEP_TOLERANCE</code>.</td>
</tr>
</tbody>
</table>

A count must be an int, and a flag must be a bool.
A kind or constant that the method ignores raises `ValueError` rather than being dropped.
Method names are `adam`, `bb`, `bfgs`, `cg`, `conjugate_descent`, `dai_yuan`, `diis`, `dogleg`, `fire`, `fire2`, `fletcher_reeves`, `fr_pr`, `gromacs_lbfgs`, `hager_zhang`, `hestenes_stiefel`, `lbfgs`, `liu_storey`, `newton`, `polak_ribiere`, `pso`, `quickmin`, `rfo`, `sd`, `sr1`, `sr2` and `steepest`.


# Module `rgmin` in the maturin package

`Session(kind, n, **options)` is the stepper. `kind` is any gradient-only method token; `cg` is Polak-Ribiere-Plus, and `newton`, `rfo`, `dogleg` and `pso` are refused.
`step(x, callback)` takes one outer step and returns `(x, value, grad_norm)`, where `callback(x)` returns `(value, gradient)`.
`push_pair(s, y)` stores an L-BFGS pair without an evaluation and returns whether the curvature gate kept it. `pair_count()`, `steps`, `forget()` and `set_level_set(target)` complete it.
The options and their defaults are:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Option</th>
<th scope="col" class="org-left">Default</th>
<th scope="col" class="org-left">Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>max_atom</code></td>
<td class="org-left">0.2</td>
<td class="org-left">Per-atom cap. Non-positive leaves it off.</td>
</tr>

<tr>
<td class="org-left"><code>maxmove</code></td>
<td class="org-left"><code>None</code></td>
<td class="org-left">Euclidean cap. Non-positive leaves it off.</td>
</tr>

<tr>
<td class="org-left"><code>wolfe</code></td>
<td class="org-left">False</td>
<td class="org-left">L-BFGS takes the strong Wolfe search.</td>
</tr>

<tr>
<td class="org-left"><code>eon_cap</code></td>
<td class="org-left">False</td>
<td class="org-left">L-BFGS takes the eOn capped step.</td>
</tr>

<tr>
<td class="org-left"><code>memory</code></td>
<td class="org-left">8</td>
<td class="org-left">L-BFGS pairs.</td>
</tr>

<tr>
<td class="org-left"><code>nbfgscorr</code></td>
<td class="org-left"><code>None</code></td>
<td class="org-left">Correction pairs for a <code>gromacs-lbfgs</code> session. Omitted keeps 10. Other methods reject it.</td>
</tr>

<tr>
<td class="org-left"><code>h0</code></td>
<td class="org-left"><code>None</code></td>
<td class="org-left">Fixed L-BFGS inverse-Hessian scale.</td>
</tr>

<tr>
<td class="org-left"><code>dt</code>, <code>dt_max</code>, <code>n_min</code>, <code>f_inc</code>, <code>f_dec</code>, <code>alpha</code>, <code>f_alpha</code></td>
<td class="org-left">0.1, 1.0, 5, 1.1, 0.5, 0.1, 0.99</td>
<td class="org-left">FIRE parameters, the ASE set.</td>
</tr>

<tr>
<td class="org-left"><code>preset</code>, <code>abc</code></td>
<td class="org-left"><code>None</code>, False</td>
<td class="org-left"><code>preset="guenole2020"</code> and <code>abc</code> select FIRE 2.0 parameters and ABC-FIRE.</td>
</tr>

<tr>
<td class="org-left"><code>accept</code></td>
<td class="org-left"><code>None</code></td>
<td class="org-left"><code>none</code>, <code>energy</code> or <code>nonmonotone</code>. <code>step</code> is not accepted here.</td>
</tr>

<tr>
<td class="org-left"><code>downhill</code></td>
<td class="org-left">False</td>
<td class="org-left">The same as <code>accept="energy"</code>.</td>
</tr>

<tr>
<td class="org-left"><code>istep</code></td>
<td class="org-left"><code>None</code></td>
<td class="org-left">Step scale of a method other than FIRE, whose scale is <code>dt</code>.</td>
</tr>

<tr>
<td class="org-left"><code>level_set</code></td>
<td class="org-left"><code>None</code></td>
<td class="org-left">Target of the level-set constraint, as in .</td>
</tr>

<tr>
<td class="org-left"><code>scale</code>, <code>weight</code>, <code>noise</code></td>
<td class="org-left"><code>0.4</code>, <code>1.0</code>, <code>0.001</code> (or <code>0.3</code>, <code>2.0</code>, <code>0.004</code> when updating)</td>
<td class="org-left">GPMin kernel length scale, weight, and observation noise.</td>
</tr>

<tr>
<td class="org-left"><code>prior</code>, <code>update_prior_strategy</code></td>
<td class="org-left"><code>None</code>, <code>"maximum"</code></td>
<td class="org-left">Fixed GPMin prior mean constant or adaptive strategy (<code>"maximum"</code>, <code>"init"</code>, <code>"average"</code>).</td>
</tr>

<tr>
<td class="org-left"><code>update_hyperparams</code>, <code>batch_size</code>, <code>bounds</code>, <code>unbounded_hyperparams</code>, <code>max_refinements</code></td>
<td class="org-left"><code>False</code>, <code>1</code>, <code>0.1</code>, <code>False</code>, <code>30</code></td>
<td class="org-left">GPMin marginal-likelihood hyperparameter update schedule and refinement cap (see ).</td>
</tr>
</tbody>
</table>

`rgmin.optimize.GPMin` (also re-exported as `rgmin.GPMin`) wraps `Session("gpmin", ...)` with `ase.optimize.GPMin`'s constructor signature, defaults, and `run(fmax, steps)` convergence flag, plus `maxstep` (default `0.2`, or `0.0` for ASE's uncapped steps) and `hyperparameters` property. `Session.set_gtol(gtol)` updates the session gradient tolerance and `Session.gpmin_hyperparameters` reports `(weight, scale, noise)`.

`LineSearch(kind, x, direction, istep=1.0, c1=1e-4, c2=0.9, beta=0.5, maxiter=40, tol=1e-10)` is the supplied search. `offer(f, gradient)` takes the value and gradient at `x`, then at each trial, and returns `(x, alpha, value, done)`. `done` is False when `x` is the next trial.
`LbfgsPairs(n, capacity=10)` is the pair memory. `push(s, y, rho=None)` returns whether the pair was stored, and `direction(gradient, h0)` returns `-H g` for `h0` a float, an array or a callable. `clear()`, `dim()` and `len()` complete it.
`Stiefel`, `SPD` and `Product` are the manifold classes of .
`lowest_eigenpairs(hvp, x0_block, *, x=None, tol=1e-8, max_iter=None, precond=None, strict=True)` is block LOBPCG, as shows. It raises `RuntimeError` for a pair that misses `tol` unless `strict=False`.


# Errors

`ValueError` means an option name or an option value is invalid, or a
coordinate array has the wrong shape. A count such as `memory` that
does not fit in a signed 64-bit integer is a `ValueError`. A count
passed as a float or a bool, for example `memory` set to `1.5`, is
also a `ValueError` whose message says it must be an int.
`TypeError` means an array has the wrong dtype or order, an option
value is not a bool, an int, a float, or a str, `method` is not a
string or a `Method`, or `f_and_grad` did not return
`(value, gradient)`.
`RuntimeError` means the solver is already in use on another thread,
or the native step failed. The message is the text reported by the
library. `Solver.step` on Newton, RFO, or dogleg, with no Hessian
callback, raises `RuntimeError`.
`NumericalError` is an `ArithmeticError`. It means the value or the
gradient was not finite.
`NotConverged` is a `RuntimeError` raised by `minimize` when
`require_convergence=True` and the final gradient norm is not below
`gtol`. `error.report` is the terminal `Report`.
An exception raised inside the objective is raised again on the
calling thread.


# Module `rgmin.ase`

`Optimizer` subclasses `ase.optimize.optimize.Optimizer`.
Named classes: `LBFGS`, `FIRE`, `FIRE2`, `QuickMin`, `BFGS`, `SD`,
`Adam`, `Newton`, `RFO`, `SR1`, `SR2`, `BB`, `Dogleg`, `DIIS`, `PSO`,
`FletcherReeves`, `HestenesStiefel`, `DaiYuan`, `ConjugateDescent`,
`HagerZhang`, `LiuStorey`, `FrPr`, and `CG`.
`BFGSLineSearch` is also exported as `QuasiNewton`, and
`LBFGSLineSearch` subclasses `LBFGS`. Their constructors match ASE 3.29.0
([BFGSLineSearch](https://gitlab.com/ase/ase/-/blob/3.29.0/ase/optimize/bfgslinesearch.py),
[LBFGS](https://gitlab.com/ase/ase/-/blob/3.29.0/ase/optimize/lbfgs.py)).
`CG` defaults to Polak-Ribiere. The C ABI has no separate
Polak-Ribiere+ value.

`run` returns a Python `bool`. `True` means the force criterion was
met.
`NotImplementedError` means `restart`, `read`, or `dump` on a class
other than `BFGSLineSearch` and `LBFGSLineSearch`, or a Hessian
optimizer called without `hessian`. Those two classes replay an rgmin
evaluation log from `restart`. Both run `linesearch="ase"`, a port of
ASE's `LineSearch`, and raise `RuntimeError("LineSearch failed!")` when
a search fails. `h0` and `linesearch="ase"` are opt-in `SolverConfig`
options: `h0` scales the initial inverse Hessian of `bfgs` and `lbfgs`,
and `linesearch="ase"` runs ASE's search on those two methods, with
`maxstep` limiting each trial increment of an atom. `stpmax`, 50 by
default, bounds its step multiplier and requires `linesearch="ase"`.
A failed search stops the step with `Stop.LINE_SEARCH`. Omitting both
leaves the session defaults.
