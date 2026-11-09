`Solver::set_fire_variant` selects the built-in FIRE schedule or
Table 1 of Comput. Mater. Sci. 175 (2020) 109584,
DOI 10.1016/j.commatsci.2020.109584, on FIRE 2.0. A session that
never calls it keeps the integrator it was given. The call does not
install a displacement cap.
