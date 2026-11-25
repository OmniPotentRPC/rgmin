`min_style rgmin` now maps LAMMPS FIRE keyword overrides (`delaystep`,
`dtgrow`, `dtshrink`, `alpha0`, `alphashrink`, `tmax`) onto rgmin
through the C ABI `rgmin_solver_set_fire_config` instead of rejecting
all FIRE parameter edits, while `min_modify tmin`, `halfstepback`,
`initialdelay`, `vdfmax`, and non-`eulerimplicit` `integrator` values
fail with one-keyword errors that name the unsupported setting.
From ABI minor 62, `rgmin_solver_set_fire_config` refuses an `n_min`
above `PTRDIFF_MAX`, where a negative `int` converted to `size_t` lands.
