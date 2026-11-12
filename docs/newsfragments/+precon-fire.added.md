FIRE and FIRE 2.0 take an optional preconditioner.
The kick and the mix use `P^{-1} F`, the power check is `F · v`, and the displacement cap stays Cartesian, as in ASE `PreconFIRE` (Packwood et al., J. Chem. Phys. 144, 164109 (2016)).
`set_fire_precon(false)` keeps the Euclidean integrator.
`set_exp` still installs the exponential metric.
`rgmin_solver_set_fire_precon` is the C entry.
`abi_minor` 36.
