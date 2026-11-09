`Solver::set_precon` installs one of the exponential, Lindh, or bonded
metrics. `set_exp`, `set_lindh`, and `set_ff` call it. A session that
never asks still keeps the steps it took before.
