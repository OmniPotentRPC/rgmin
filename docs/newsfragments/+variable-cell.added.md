`relax_cell` minimizes a periodic cell and its atoms on the existing steppers, in the generalized solid-state NEB coordinate.
The strain is scaled by `V^(1/3) N^(1/6)`.
A frozen mask matches the fixed-cell minimizer.
The minimum reports the stress and the dipole `P = V sigma`.
A contiguous coordinate is read in place, and zero strain passes that slice to the potential.
