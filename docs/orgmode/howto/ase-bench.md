
# Table of Contents



The median −0.012693 reported earlier for ASE LBFGS on LJ13 is not a
minimum. That start was a corner of a cubic lattice at spacing 1.5,
rattled by `Normal(0, 0.05)`. On seed 0, ASE LBFGS with its default
`alpha=70` (initial inverse Hessian `I/70`, longest atom at most 0.2)
does this:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-right" />

<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-right">step</th>
<th scope="col" class="org-left">energy</th>
<th scope="col" class="org-right">max force</th>
<th scope="col" class="org-right">min pair</th>
<th scope="col" class="org-right">radius</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-right">0</td>
<td class="org-left">−7.044814</td>
<td class="org-right">2.270844</td>
<td class="org-right">1.3495</td>
<td class="org-right">2.5068</td>
</tr>

<tr>
<td class="org-right">1</td>
<td class="org-left">−4.571523</td>
<td class="org-right">1.417897</td>
<td class="org-right">1.5824</td>
<td class="org-right">2.6311</td>
</tr>

<tr>
<td class="org-right">31</td>
<td class="org-left">−0.023658</td>
<td class="org-right">0.009242</td>
<td class="org-right">3.0801</td>
<td class="org-right">6.4782</td>
</tr>
</tbody>
</table>

The first step still moves inward. After that the radius grows on
every step and the force falls because the pairs have gone. The same
`alpha=70` on a compact 13-atom start stays bound, and `alpha=10` on
the cubic start is worse (energy +2448 after 80 steps). The lattice,
not the drop-in and not `alpha` by itself, is what produced −0.01.

The runs below do not use that lattice. LJ13 starts from the Mackay
icosahedron and LJ38 from the Oh truncated octahedron in the Cambridge
Cluster Database
(`https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/13` and
`.../38`). The database energies are −44.326801 and −173.928427. With
ASE's Lennard-Jones calculator (`epsilon = 1`, `sigma = 1`, `rc = 10`,
potential shifted to zero at the cutoff) those geometries evaluate to
−44.326489 and −173.925615. Ten seeds add `Normal(0, 0.05)` from
`PCG64(20261113 + seed)`. A second set draws points uniformly in a
sphere of density 1 with no pair closer than 0.9. Cu64 is unchanged:
fcc `a=3.615` repeated `(4, 4, 4)`, rattled the same way.

A run is converged only when the maximum force is below 0.01 eV/Å and
the largest distance from the centroid stays under twice the start
value. The other count is dissociations. Medians below are over the
converged runs. `ase-lbfgs` uses `alpha=70`. `ase-lbfgs-a10` uses
`alpha=10`.

rgmin FIRE is Bitzek's velocity-Verlet step: the force is mixed after
the move. The parameters are ASE's defaults (`dt=0.1`, `dtmax=1`,
`Nmin=5`, `finc=1.1`, `fdec=0.5`, `astart=0.1`, `fa=0.99`) and the
longest atom moves at most 0.2. ASE FIRE mixes before the Euler kick
and caps the Euclidean length of the whole displacement.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">system</th>
<th scope="col" class="org-left">optimizer</th>
<th scope="col" class="org-left">converged</th>
<th scope="col" class="org-right">dissociated</th>
<th scope="col" class="org-right">force calls</th>
<th scope="col" class="org-left">energy</th>
<th scope="col" class="org-right">seconds</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">Cu64</td>
<td class="org-left">ase-bfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">22.5</td>
<td class="org-left">−0.315751</td>
<td class="org-right">0.1394</td>
</tr>

<tr>
<td class="org-left">Cu64</td>
<td class="org-left">ase-fire</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">47.5</td>
<td class="org-left">−0.315703</td>
<td class="org-right">0.1851</td>
</tr>

<tr>
<td class="org-left">Cu64</td>
<td class="org-left">ase-lbfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">22.5</td>
<td class="org-left">−0.315751</td>
<td class="org-right">0.0935</td>
</tr>

<tr>
<td class="org-left">Cu64</td>
<td class="org-left">ase-lbfgs-a10</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">9.0</td>
<td class="org-left">−0.315786</td>
<td class="org-right">0.0412</td>
</tr>

<tr>
<td class="org-left">Cu64</td>
<td class="org-left">rgmin-fire</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">44.5</td>
<td class="org-left">−0.315668</td>
<td class="org-right">0.1797</td>
</tr>

<tr>
<td class="org-left">Cu64</td>
<td class="org-left">rgmin-lbfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">11.0</td>
<td class="org-left">−0.315895</td>
<td class="org-right">0.0497</td>
</tr>

<tr>
<td class="org-left">LJ13</td>
<td class="org-left">ase-bfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">29.0</td>
<td class="org-left">−44.326489</td>
<td class="org-right">0.0224</td>
</tr>

<tr>
<td class="org-left">LJ13</td>
<td class="org-left">ase-fire</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">71.0</td>
<td class="org-left">−44.326487</td>
<td class="org-right">0.0387</td>
</tr>

<tr>
<td class="org-left">LJ13</td>
<td class="org-left">ase-lbfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">29.0</td>
<td class="org-left">−44.326489</td>
<td class="org-right">0.0205</td>
</tr>

<tr>
<td class="org-left">LJ13</td>
<td class="org-left">ase-lbfgs-a10</td>
<td class="org-left">9/10</td>
<td class="org-right">1</td>
<td class="org-right">161.0</td>
<td class="org-left">−39.624761</td>
<td class="org-right">0.1375</td>
</tr>

<tr>
<td class="org-left">LJ13</td>
<td class="org-left">rgmin-fire</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">69.5</td>
<td class="org-left">−44.326487</td>
<td class="org-right">0.0426</td>
</tr>

<tr>
<td class="org-left">LJ13</td>
<td class="org-left">rgmin-lbfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">19.0</td>
<td class="org-left">−44.326489</td>
<td class="org-right">0.0120</td>
</tr>

<tr>
<td class="org-left">LJ13-sphere</td>
<td class="org-left">ase-bfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">47.5</td>
<td class="org-left">−39.928329</td>
<td class="org-right">0.0364</td>
</tr>

<tr>
<td class="org-left">LJ13-sphere</td>
<td class="org-left">ase-fire</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">129.0</td>
<td class="org-left">−41.057068</td>
<td class="org-right">0.0709</td>
</tr>

<tr>
<td class="org-left">LJ13-sphere</td>
<td class="org-left">ase-lbfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">47.5</td>
<td class="org-left">−39.928329</td>
<td class="org-right">0.0349</td>
</tr>

<tr>
<td class="org-left">LJ13-sphere</td>
<td class="org-left">ase-lbfgs-a10</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">160.0</td>
<td class="org-left">−39.823170</td>
<td class="org-right">0.1403</td>
</tr>

<tr>
<td class="org-left">LJ13-sphere</td>
<td class="org-left">rgmin-fire</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">127.5</td>
<td class="org-left">−40.225870</td>
<td class="org-right">0.0801</td>
</tr>

<tr>
<td class="org-left">LJ13-sphere</td>
<td class="org-left">rgmin-lbfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">52.0</td>
<td class="org-left">−40.225874</td>
<td class="org-right">0.0345</td>
</tr>

<tr>
<td class="org-left">LJ38</td>
<td class="org-left">ase-bfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">91.0</td>
<td class="org-left">−173.925612</td>
<td class="org-right">0.1686</td>
</tr>

<tr>
<td class="org-left">LJ38</td>
<td class="org-left">ase-fire</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">81.0</td>
<td class="org-left">−173.925601</td>
<td class="org-right">0.0800</td>
</tr>

<tr>
<td class="org-left">LJ38</td>
<td class="org-left">ase-lbfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">91.0</td>
<td class="org-left">−173.925612</td>
<td class="org-right">0.1156</td>
</tr>

<tr>
<td class="org-left">LJ38</td>
<td class="org-left">ase-lbfgs-a10</td>
<td class="org-left">0/10</td>
<td class="org-right">1</td>
<td class="org-right">&#xa0;</td>
<td class="org-left">&#xa0;</td>
<td class="org-right">&#xa0;</td>
</tr>

<tr>
<td class="org-left">LJ38</td>
<td class="org-left">rgmin-fire</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">82.5</td>
<td class="org-left">−173.925604</td>
<td class="org-right">0.0875</td>
</tr>

<tr>
<td class="org-left">LJ38</td>
<td class="org-left">rgmin-lbfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">27.0</td>
<td class="org-left">−173.925606</td>
<td class="org-right">0.0308</td>
</tr>

<tr>
<td class="org-left">LJ38-sphere</td>
<td class="org-left">ase-bfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">179.0</td>
<td class="org-left">−167.460624</td>
<td class="org-right">0.3568</td>
</tr>

<tr>
<td class="org-left">LJ38-sphere</td>
<td class="org-left">ase-fire</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">515.0</td>
<td class="org-left">−168.195123</td>
<td class="org-right">0.5123</td>
</tr>

<tr>
<td class="org-left">LJ38-sphere</td>
<td class="org-left">ase-lbfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">188.0</td>
<td class="org-left">−167.137345</td>
<td class="org-right">0.2635</td>
</tr>

<tr>
<td class="org-left">LJ38-sphere</td>
<td class="org-left">ase-lbfgs-a10</td>
<td class="org-left">0/10</td>
<td class="org-right">2</td>
<td class="org-right">&#xa0;</td>
<td class="org-left">&#xa0;</td>
<td class="org-right">&#xa0;</td>
</tr>

<tr>
<td class="org-left">LJ38-sphere</td>
<td class="org-left">rgmin-fire</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">549.5</td>
<td class="org-left">−168.246137</td>
<td class="org-right">0.5867</td>
</tr>

<tr>
<td class="org-left">LJ38-sphere</td>
<td class="org-left">rgmin-lbfgs</td>
<td class="org-left">10/10</td>
<td class="org-right">0</td>
<td class="org-right">141.5</td>
<td class="org-left">−168.105354</td>
<td class="org-right">0.1689</td>
</tr>
</tbody>
</table>

On the rattled icosahedron and truncated octahedron, rgmin L-BFGS and
rgmin FIRE reach the cutoff-shifted global minimum on all ten seeds.
`alpha=10` does not: one LJ13 run dissociates, and none of the ten
LJ38 runs reach the force tolerance (one of them dissociates).
