`min_style rgmin` now maps LAMMPS FIRE keyword overrides (`delaystep`,
`dtgrow`, `dtshrink`, `alpha0`, `alphashrink`, `tmax`) onto rgmin
through the C ABI `rgmin_solver_set_fire_config` instead of rejecting
all FIRE parameter edits, while `min_modify tmin`, `halfstepback`,
`initialdelay`, `vdfmax`, and non-`eulerimplicit` `integrator` values
fail with one-keyword errors that name the unsupported setting.
