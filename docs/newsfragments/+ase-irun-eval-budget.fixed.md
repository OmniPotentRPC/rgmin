`rgmin.ase` `irun` yields `False` when `max_evals` stops the solver. ASE's loop asks `converged`, which is `True` for that stop, and `run` already reported `False`.
