`Solver::set_fire_variant` selects the built-in FIRE schedule or
the Guénolé 2020 table on FIRE 2.0. A session that never calls
it keeps the integrator it was given. The call does not install
a displacement cap.
