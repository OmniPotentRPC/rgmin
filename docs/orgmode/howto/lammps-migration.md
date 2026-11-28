

# Migrate from LAMMPS minimizers

`min_style rgmin` is a LAMMPS plugin that runs rgmin's L-BFGS, FIRE, FIRE 2.0, steepest descent and Barzilai-Borwein sessions inside the LAMMPS `minimize` command.
Load the plugin and change the `min_style` line; the rest of the input stays as it is.

```text
plugin load rgminplugin.so
min_style rgmin              # was: min_style cg
minimize 0.0 1.0e-6 10000 100000
```

The plugin calls rgmin only through its C ABI, `include/rgmin.h`.
It works with any pair style, bond style, kspace style and fix that contributes energy and forces to a LAMMPS minimization.
lists every `min_modify` keyword, the exact stopping rules and the errors.


## Build

The plugin needs a LAMMPS built with CMake, the `PLUGIN` package and the packages of your potentials.
LAMMPS can load a min style from a plugin from the 22Jul2025 stable release on.
The plugin was tested with the stable releases 22Jul2025 (update 2) and 30Sep2026 (update 1), with Open MPI 4.1 and GCC 13.
The 2Aug2023 and 29Aug2024 releases log `Loading plugins for min styles not yet implemented` and then stop at `min_style rgmin` with `Illegal minimize style`, so the plugin's CMake refuses their sources.

```bash
cmake -S lammps-stable_30Sep2026_update1/cmake -B build-lammps \
  -D BUILD_MPI=on -D PKG_PLUGIN=on -D PKG_MANYBODY=on
cmake --build build-lammps -j

cmake -S rgmin/lammps -B build-rgmin-lammps \
  -D LAMMPS_SOURCE_DIR=$PWD/lammps-stable_30Sep2026_update1/src \
  -D RGMIN_LAMMPS_LMP=$PWD/build-lammps/lmp
cmake --build build-rgmin-lammps
```

The second build runs `cargo build --release --features capi` in the rgmin checkout and links the static library into `rgminplugin.so`, whose rgmin symbols stay private.
The plugin therefore needs no rgmin library at run time.
`-D RGMIN_LIBRARY=<path>` links a `librgmin.a` or `librgmin.so` that is already built with `capi` instead.
The plugin has to be compiled against the source tree, MPI library and `LAMMPS_SIZES` (`smallbig` by default) of the `lmp` that loads it.
`LAMMPS_MPI=OFF` builds against the serial `STUBS` library of a LAMMPS built without MPI.
The default `cargo build` of rgmin does not build or need any of this.

`plugin load` takes a path, or a file name that LAMMPS finds through `LAMMPS_PLUGIN_PATH`.
`cmake --install` puts the plugin under `lib/lammps/plugins` of the install prefix.


## Methods

`min_modify method` selects the rgmin session.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left"><code>method</code></th>
<th scope="col" class="org-left">rgmin session</th>
<th scope="col" class="org-left">nearest LAMMPS style</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>lbfgs</code> (default)</td>
<td class="org-left">L-BFGS without a line search, eOn's per-atom cap, memory 10</td>
<td class="org-left"><code>cg</code></td>
</tr>

<tr>
<td class="org-left"><code>fire</code></td>
<td class="org-left">FIRE of Bitzek et al. 2006</td>
<td class="org-left"><code>fire</code></td>
</tr>

<tr>
<td class="org-left"><code>fire2</code></td>
<td class="org-left">FIRE 2.0 with Guénolé et al. 2020 Table 1, the LAMMPS <code>fire</code> defaults</td>
<td class="org-left"><code>fire</code></td>
</tr>

<tr>
<td class="org-left"><code>sd</code></td>
<td class="org-left">steepest descent with energy acceptance</td>
<td class="org-left"><code>sd</code></td>
</tr>

<tr>
<td class="org-left"><code>bb</code></td>
<td class="org-left">Barzilai-Borwein</td>
<td class="org-left">none</td>
</tr>
</tbody>
</table>

`linesearch wolfe`, `trust`, `memory` and `accept` change the L-BFGS step.
The other rgmin methods are refused with an error.
`bfgs`, `sr1` and `sr2` store a dense `3N` by `3N` matrix.
`newton`, `rfo` and `dogleg` need a dense Hessian.
The nonlinear CG family, Adam and Newton-CG take session steps without a line search and stopped short of `ftol` on the test systems.
For example, at `ftol 1e-8` Hager-Zhang stopped at 10,000 iterations on a rattled 864-atom LJ crystal that L-BFGS relaxed in 57.


## Stopping criteria

`minimize etol ftol maxiter maxeval` keeps its LAMMPS meaning.

-   `ftol` stops when the force norm of `min_modify norm` is below `ftol`: the two-norm of the global `3N` force vector by default, the largest per-atom force for `norm max`, and the largest component for `norm inf`.
    rgmin computes it as its `TWO`, `MAX_ATOM` or `INF` stop norm of the gradient `-F`, which is the force after every fix has modified it.
    LAMMPS compares `|F|^2 < ftol^2` and rgmin `|F| < ftol`; the two agree except within rounding of the bound.
-   `etol` stops when `|E_k - E_{k-1}| < etol (|E_k| + |E_{k-1}| + eps) / 2` for the energies of two consecutive accepted steps, the test of `min_style cg`.
    The plugin passes `etol` to `rgmin_solver_set_etol` and ends the run with the LAMMPS stopping criterion `energy tolerance` when rgmin reports `RGMIN_STOP_ENERGY_TOLERANCE`.
    A step that rgmin refused leaves the iterate in place and does not take the test, and FIRE takes it only after its delay steps, as `min_style fire` does.
    The floor `eps` is `sqrt(DBL_EPSILON)`, 1.5e-8, on the total energy; LAMMPS adds 1e-8 to the energy per atom under `thermo_modify norm yes`.
    The two tests are the same relative test and differ only when the energy is as small as the floor.
    When `etol` and `ftol` are met by the same step, rgmin reports the force stop, where `min_style cg` reports `energy tolerance`.
    `etol 0` leaves the test off.
-   `maxiter` counts rgmin steps, one per LAMMPS iteration and timestep.
-   `maxeval` counts force evaluations, the `Force evaluations` line of the LAMMPS summary.
    A step that needs several evaluations can pass the limit by the evaluations of that step.

The energy that rgmin minimizes is the one LAMMPS minimizers see: the potential energy of `thermo_pe`, including fixes with `fix_modify energy yes`.
With `thermo_modify norm yes` the plugin hands rgmin the total energy, so energy and forces stay consistent. The relative `etol` test gives the same answer on the total and on the per-atom energy.


## Fixes

Fixes that change forces during a minimization act on the forces rgmin sees.
`fix setforce` atoms do not move, and a fix such as `fix spring/self` with `fix_modify energy yes` adds its energy and forces.
`python -m pytest lammps/tests` checks `setforce` and `spring/self` against `min_style cg`.

`fix box/relax` adds the box to the degrees of freedom of the minimization.
The plugin does not relax the box and stops with an error that names `fix box/relax`; use `min_style cg` for a cell relaxation.
Pair styles that add per-atom degrees of freedom, such as the electron radius of `eff`, are refused in the same way.


## Parallel runs

Every MPI rank holds the whole `3N` coordinate vector in atom-ID order and runs the same rgmin session on it.
An evaluation moves the atoms a rank owns to rgmin's coordinates, calls the minimizer's `energy_force` on all ranks, and sums each rank's owned forces into the `3N` gradient with one `MPI_Allreduce`.
Each entry of that sum has one owner and zeros elsewhere, so the result is exact and the same on every rank.
The energy is broadcast from rank 0.
The sessions are deterministic, so with identical inputs they take the same steps; the plugin compares the step energy across ranks after every step and stops with an error if they ever differ.

rgmin's solvers take a contiguous vector and compute their own dot products, and the C ABI has no hook for a distributed reduction.
Gathering costs one sum of `3N` doubles per force evaluation, against a force evaluation that already exchanges ghost atoms, and the session's own work is a few `O(mN)` vector operations per step.
That overhead shows on the 32k cell: at the same number of force calls, rgmin's `fire` took 2.4 s on 4 ranks against 1.6 s for LAMMPS's `fire`.
The `3N` copy on every rank and the full-length sum grow with N and not with N per rank, so they limit how far a run scales to more ranks.

Atoms move to the periodic image of rgmin's coordinates that is nearest to where they are, so LAMMPS keeps wrapping atoms and updating image flags across reneighboring.
Atom IDs must run from 1 to N; `reset_atoms id` renumbers them after atoms are deleted.

On the 500-atom Cu dumbbell-and-vacancy cell, 1, 2 and 4 ranks take the same number of steps and force calls.
Their energies agree to 1e-12 relative at every step and their force norms to 1e-9 eV/Å, the rounding of a force sum taken in a different order.


## Differences

-   `min_modify dmax` (default 0.1, distance units) is rgmin's per-atom step cap for every method.
    rgmin's own default cap is 0.2; `min_modify dmax 0.2` restores it.
-   `linesearch wolfe` with a positive `dmax` runs rgmin's geometry L-BFGS step, which bounds each trial by the cap.
    With `dmax 0` it is the plain strong Wolfe search, which ran into `maxeval` on the LJ cluster and the dumbbell cell at `ftol 1e-6`.
-   rgmin's FIRE integrators use unit masses and a step scale `istep` (default 0.1), not the atom masses and the LAMMPS `timestep` of `min_style fire`.
    `min_modify abcfire yes` selects ABC-FIRE on `fire2`.
    The other LAMMPS FIRE keywords must stay at their defaults, which are the FIRE 2.0 parameters the plugin uses; a changed value is an error.
-   `min_modify line` belongs to the LAMMPS line-search styles and is an error with `min_style rgmin`.
-   The first rgmin step uses the energy and forces that `Min::setup` computed, so it costs no force evaluation.
-   When rgmin's iterate is not the last point it evaluated, the plugin evaluates it before a thermo or dump step and counts that evaluation.
-   The summary line `Final line search alpha` is always 0.


## Benchmark

`python benches/lammps_minstyle.py` runs `min_style cg`, `sd`, `fire` and `quickmin` and `min_style rgmin` with `lbfgs`, `fire` and `fire2`, each at its defaults, on the test systems and on a 31,999-atom Cu vacancy cell with every atom rattled by up to 0.05 Å.
Every run is `minimize 0.0 1.0e-6 100000 1000000`.
Wall time is the LAMMPS loop time, the median of five repeats; iterations and force calls were the same on every repeat.
The 32k cell also ran on 4 MPI ranks.
The full table, with energies, final force norms and the minimum times, is `benches/lammps_minstyle/results.md`.

The table was measured before the 0.6.0 release, on commit 5a9f396 of `develop` with the plugin sources in the working tree and not yet committed.
At that commit `Cargo.toml` said 0.2.0 and the C ABI was 1.41.
The runs used LAMMPS 30Sep2026 (update 1), Open MPI 4.1.6 and GCC 13.3, on Linux 6.12 `x86_64` (Intel Xeon, 8 logical CPUs).
With rgmin 0.6.0 every row takes the same iterations and force calls, except `rgmin fire` and `rgmin fire2` on `lj177`, which take a few percent fewer iterations.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-right" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">case</th>
<th scope="col" class="org-left">min<sub>style</sub></th>
<th scope="col" class="org-left">1 rank: iterations/calls</th>
<th scope="col" class="org-right">1 rank: median s</th>
<th scope="col" class="org-left">4 ranks: iterations/calls</th>
<th scope="col" class="org-left">4 ranks: median s</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">lj177</td>
<td class="org-left"><code>cg</code></td>
<td class="org-left">81/162</td>
<td class="org-right">0.00614</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">lj177</td>
<td class="org-left"><code>sd</code></td>
<td class="org-left">559/1118</td>
<td class="org-right">0.03951</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">lj177</td>
<td class="org-left"><code>fire</code></td>
<td class="org-left">247/257</td>
<td class="org-right">0.009137</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">lj177</td>
<td class="org-left"><code>quickmin</code></td>
<td class="org-left">678/678</td>
<td class="org-right">0.02402</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">lj177</td>
<td class="org-left"><code>rgmin lbfgs</code></td>
<td class="org-left">89/90</td>
<td class="org-right">0.0046</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">lj177</td>
<td class="org-left"><code>rgmin fire</code></td>
<td class="org-left">237/237</td>
<td class="org-right">0.01121</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">lj177</td>
<td class="org-left"><code>rgmin fire2</code></td>
<td class="org-left">230/258</td>
<td class="org-right">0.01189</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-vac</td>
<td class="org-left"><code>cg</code></td>
<td class="org-left">32/64</td>
<td class="org-right">0.01321</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-vac</td>
<td class="org-left"><code>sd</code></td>
<td class="org-left">127/254</td>
<td class="org-right">0.0509</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-vac</td>
<td class="org-left"><code>fire</code></td>
<td class="org-left">158/163</td>
<td class="org-right">0.03314</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-vac</td>
<td class="org-left"><code>quickmin</code></td>
<td class="org-left">243/243</td>
<td class="org-right">0.04921</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-vac</td>
<td class="org-left"><code>rgmin lbfgs</code></td>
<td class="org-left">35/36</td>
<td class="org-right">0.008588</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-vac</td>
<td class="org-left"><code>rgmin fire</code></td>
<td class="org-left">154/154</td>
<td class="org-right">0.03396</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-vac</td>
<td class="org-left"><code>rgmin fire2</code></td>
<td class="org-left">115/120</td>
<td class="org-right">0.02632</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-sia-vac</td>
<td class="org-left"><code>cg</code></td>
<td class="org-left">66/129</td>
<td class="org-right">0.05803</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-sia-vac</td>
<td class="org-left"><code>sd</code></td>
<td class="org-left">305/604</td>
<td class="org-right">0.269</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-sia-vac</td>
<td class="org-left"><code>fire</code></td>
<td class="org-left">228/236</td>
<td class="org-right">0.1066</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-sia-vac</td>
<td class="org-left"><code>quickmin</code></td>
<td class="org-left">553/553</td>
<td class="org-right">0.2487</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-sia-vac</td>
<td class="org-left"><code>rgmin lbfgs</code></td>
<td class="org-left">64/65</td>
<td class="org-right">0.03302</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-sia-vac</td>
<td class="org-left"><code>rgmin fire</code></td>
<td class="org-left">175/175</td>
<td class="org-right">0.08474</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu-sia-vac</td>
<td class="org-left"><code>rgmin fire2</code></td>
<td class="org-left">204/215</td>
<td class="org-right">0.1033</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>

<tr>
<td class="org-left">cu32k-vac</td>
<td class="org-left"><code>cg</code></td>
<td class="org-left">121/243 (*)</td>
<td class="org-right">5.688</td>
<td class="org-left">130/260</td>
<td class="org-left">1.759</td>
</tr>

<tr>
<td class="org-left">cu32k-vac</td>
<td class="org-left"><code>sd</code></td>
<td class="org-left">859/1723 (*)</td>
<td class="org-right">39.03</td>
<td class="org-left">2449/4898</td>
<td class="org-left">30.16</td>
</tr>

<tr>
<td class="org-left">cu32k-vac</td>
<td class="org-left"><code>fire</code></td>
<td class="org-left">271/275</td>
<td class="org-right">6.264</td>
<td class="org-left">271/275</td>
<td class="org-left">1.628</td>
</tr>

<tr>
<td class="org-left">cu32k-vac</td>
<td class="org-left"><code>quickmin</code></td>
<td class="org-left">2680/2680</td>
<td class="org-right">60.65</td>
<td class="org-left">2680/2680</td>
<td class="org-left">16.04</td>
</tr>

<tr>
<td class="org-left">cu32k-vac</td>
<td class="org-left"><code>rgmin lbfgs</code></td>
<td class="org-left">145/146</td>
<td class="org-right">3.909</td>
<td class="org-left">145/146</td>
<td class="org-left">1.533</td>
</tr>

<tr>
<td class="org-left">cu32k-vac</td>
<td class="org-left"><code>rgmin fire</code></td>
<td class="org-left">270/270</td>
<td class="org-right">6.829</td>
<td class="org-left">270/270</td>
<td class="org-left">2.401</td>
</tr>

<tr>
<td class="org-left">cu32k-vac</td>
<td class="org-left"><code>rgmin fire2</code></td>
<td class="org-left">766/805</td>
<td class="org-right">20.09</td>
<td class="org-left">766/805</td>
<td class="org-left">6.962</td>
</tr>
</tbody>
</table>

(\*) The serial `cg` and `sd` runs on the 32k cell stopped on `linesearch alpha is zero` with force norms of 3.9e-6 and 2.3e-4, above `ftol`; every other run stopped on the force tolerance.
Both converged on 4 ranks, where the forces are summed in another order.
The runs that converged agree on the energy of each case to within 1e-13 relative.

`rgmin lbfgs` took the fewest force calls and the least time on every case, between half and three fifths of the calls of `cg`, which spends two calls on each iteration.
On 4 ranks it took the same 145 iterations and 146 force calls as on one rank.
rgmin's `fire` took no more force calls than LAMMPS's `fire` on any case, but more time on the 32k cell: 6.8 s against 6.3 s serially and 2.4 s against 1.6 s on 4 ranks.
With the same number of force calls, the extra time is the plugin's own work on each call: the gather and scatter between local atoms and the global vector, the sum of the full `3N` gradient across ranks, and the session's vector operations, which every rank repeats.
`fire2`, the Guénolé 2020 parameters with rgmin's FIRE 2.0 step, took about three times the iterations of LAMMPS's `fire` on the 32k cell.
