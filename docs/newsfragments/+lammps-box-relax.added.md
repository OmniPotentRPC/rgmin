The LAMMPS plugin `min_style rgmin` relaxes the box with `fix box/relax` and other fixes that add global degrees of freedom.
The box coordinates join the atom coordinates in the rgmin vector, with the fix forces as their negative gradient.
The per-atom cap `dmax` bounds the atoms and the fix's `vmax` bounds each box coordinate, as in `min_style cg`.
Methods `lbfgs`, `sd` and `bb` support it; `fire` and `fire2` refuse it, as LAMMPS damped dynamics styles do, and so does `min_modify trust`.
