

# LAMMPS min<sub>style</sub> rgmin

This page lists the `min_modify` keywords of the LAMMPS plugin in `lammps/`, the rgmin C ABI call behind each one, and the errors the plugin raises.
shows how to build and load it.

```text
plugin load rgminplugin.so
min_style rgmin
min_modify keyword value ...
minimize etol ftol maxiter maxeval
```

`min_modify` applies to the current minimizer, so it comes after `min_style rgmin`.
A new `min_style` command starts again from the defaults.


## Keywords of the plugin

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">keyword</th>
<th scope="col" class="org-left">values</th>
<th scope="col" class="org-left">default</th>
<th scope="col" class="org-left">C ABI</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>method</code></td>
<td class="org-left"><code>lbfgs</code>, <code>fire</code>, <code>fire2</code>, <code>sd</code>, <code>bb</code></td>
<td class="org-left"><code>lbfgs</code></td>
<td class="org-left"><code>rgmin_method_from_name</code>, <code>rgmin_solver_create</code></td>
</tr>

<tr>
<td class="org-left"><code>memory</code></td>
<td class="org-left">integer, at least 1</td>
<td class="org-left">10</td>
<td class="org-left"><code>rgmin_control_t.memory</code></td>
</tr>

<tr>
<td class="org-left"><code>trust</code></td>
<td class="org-left">radius, at least 0 (distance units)</td>
<td class="org-left">0, off</td>
<td class="org-left"><code>rgmin_solver_set_lbfgs_trust</code></td>
</tr>

<tr>
<td class="org-left"><code>linesearch</code></td>
<td class="org-left"><code>none</code>, <code>wolfe</code></td>
<td class="org-left"><code>none</code></td>
<td class="org-left"><code>rgmin_solver_set_lbfgs_wolfe</code></td>
</tr>

<tr>
<td class="org-left"><code>accept</code></td>
<td class="org-left"><code>default</code>, <code>energy</code>, <code>none</code>, <code>nonmonotone</code></td>
<td class="org-left"><code>default</code></td>
<td class="org-left"><code>rgmin_solver_set_accept</code></td>
</tr>

<tr>
<td class="org-left"><code>istep</code></td>
<td class="org-left">step scale, at least 0</td>
<td class="org-left">0</td>
<td class="org-left"><code>rgmin_control_t.istep</code></td>
</tr>
</tbody>
</table>

-   `method` takes rgmin's method names, so `steepest` and `barzilai-borwein` are `sd` and `bb`.
    Any other rgmin method is an error.
-   `memory` is the number of L-BFGS pairs.
    Other methods ignore it.
-   `trust` is the Burdakov ray-truncation radius of L-BFGS, which replaces the per-atom cap and the acceptance test while it is on.
    It needs `method lbfgs` and excludes `linesearch wolfe`.
-   `linesearch wolfe` needs `method lbfgs`.
    With a positive `dmax` it is rgmin's geometry L-BFGS step, whose trials the per-atom cap bounds; with `dmax 0` it is the strong Wolfe search of Nocedal and Wright, algorithms 3.5 and 3.6.
-   `accept` is how a session takes a proposed step.
    `default` keeps the C ABI default, `energy`, which refuses a trial that raises the energy.
-   `istep` is the FIRE time step of `fire` and `fire2`, 0.1 when it is 0.
    For `sd` and `bb`, 0 is the C ABI default: a unit step scale, with `min(1, 1/|g|)` on the first step.


## LAMMPS keywords the plugin reads

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">keyword</th>
<th scope="col" class="org-left">default</th>
<th scope="col" class="org-left">use</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>dmax</code></td>
<td class="org-left">0.1</td>
<td class="org-left"><code>rgmin_solver_set_atom_maxmove</code>: the largest displacement of one atom in one step; 0 leaves the cap off</td>
</tr>

<tr>
<td class="org-left"><code>norm</code></td>
<td class="org-left"><code>two</code></td>
<td class="org-left"><code>rgmin_solver_set_stop_norm</code>: <code>two</code> is <code>TWO</code>, <code>max</code> is <code>MAX_ATOM</code>, <code>inf</code> is <code>INF</code></td>
</tr>

<tr>
<td class="org-left"><code>abcfire</code></td>
<td class="org-left"><code>no</code></td>
<td class="org-left"><code>rgmin_solver_set_fire_abc</code> on a <code>fire2</code> session</td>
</tr>
</tbody>
</table>

`method fire2` selects `RGMIN_FIRE_PRESET_GUENOLE2020`, the parameter set of the LAMMPS `min_style fire` defaults.
`delaystep`, `dtgrow`, `dtshrink`, `alpha0`, `alphashrink`, `tmax`, `tmin`, `integrator`, `halfstepback`, `initialdelay` and `vdfmax` must keep those defaults, and `line` must stay `quadratic`.
The `minimize` command checks them and stops with an error otherwise.


## Stopping rules

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">LAMMPS</th>
<th scope="col" class="org-left">plugin</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>etol</code></td>
<td class="org-left"><code>rgmin_solver_set_etol</code>: <code>abs(E_k - E_{k-1}) &lt; etol (abs(E_k) + abs(E_{k-1}) + sqrt(DBL_EPSILON)) / 2</code> after an accepted step; <code>0</code> is off</td>
</tr>

<tr>
<td class="org-left"><code>ftol</code></td>
<td class="org-left"><code>rgmin_report_t.grad_norm &lt; ftol</code>, with <code>rgmin_control_t.gtol = ftol</code> and the stop norm of <code>norm</code></td>
</tr>

<tr>
<td class="org-left"><code>maxiter</code></td>
<td class="org-left">number of <code>rgmin_solver_step_fg</code> calls</td>
</tr>

<tr>
<td class="org-left"><code>maxeval</code></td>
<td class="org-left">number of <code>Min::energy_force</code> calls the plugin made, checked after each step</td>
</tr>
</tbody>
</table>

`E` is the total energy rgmin minimizes. The relative test is the same on the per-atom energy that LAMMPS prints, apart from the floor, which is `sqrt(DBL_EPSILON)` here and 1e-8 in LAMMPS.
The gradient rgmin sees is `-F` over all `3N` coordinates after every fix, so `grad_norm` is the force norm that `min_style cg` compares with `ftol`.
A failed rgmin step ends the minimization with the stopping criterion `rgmin step failed:` and the text of `rgmin_last_error`.


## Errors

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">message contains</th>
<th scope="col" class="org-left">cause</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>Unknown min_modify keyword X for min_style rgmin</code></td>
<td class="org-left">a keyword that neither LAMMPS nor the plugin defines</td>
</tr>

<tr>
<td class="org-left"><code>Unknown rgmin method X</code></td>
<td class="org-left">a name <code>rgmin_method_from_name</code> refuses</td>
</tr>

<tr>
<td class="org-left"><code>stores a dense 3N x 3N matrix</code></td>
<td class="org-left"><code>bfgs</code>, <code>sr1</code> or <code>sr2</code></td>
</tr>

<tr>
<td class="org-left"><code>needs a dense Hessian</code></td>
<td class="org-left"><code>newton</code>, <code>rfo</code> or <code>dogleg</code></td>
</tr>

<tr>
<td class="org-left"><code>is not available with min_style rgmin</code></td>
<td class="org-left">any other rgmin method</td>
</tr>

<tr>
<td class="org-left"><code>does not support fixes that add global degrees of freedom such as fix box/relax</code></td>
<td class="org-left"><code>fix box/relax</code> or another fix with <code>min_dof</code></td>
</tr>

<tr>
<td class="org-left"><code>does not support pair styles that add per-atom degrees of freedom</code></td>
<td class="org-left">for example <code>pair_style eff/cut</code></td>
</tr>

<tr>
<td class="org-left"><code>requires consecutive atom IDs 1 to N</code></td>
<td class="org-left">gaps in the atom IDs</td>
</tr>

<tr>
<td class="org-left"><code>does not take min_modify line</code></td>
<td class="org-left"><code>min_modify line</code> other than <code>quadratic</code></td>
</tr>

<tr>
<td class="org-left"><code>keeps the Guenole 2020 FIRE parameters</code></td>
<td class="org-left">a LAMMPS FIRE keyword changed from its default</td>
</tr>

<tr>
<td class="org-left"><code>replicated rgmin sessions diverged across ranks</code></td>
<td class="org-left">ranks that reached different step energies</td>
</tr>
</tbody>
</table>

Keywords with a bad value (`memory 0`, `linesearch armijo`, `accept maybe`) and settings that need another method (`trust` or `linesearch wolfe` without `lbfgs`, `abcfire yes` without `fire2`) also stop with an error that names them.
