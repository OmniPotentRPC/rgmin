`rgmin.ase.Optimizer.irun` now honours `max_evals`. ASE's loop asks
`converged` to stop, which is true once a spent budget ends the run, so
`irun` yielded `True` where `run` returns `False`. It now yields `False`
while `stop` is `Stop.EVAL_BUDGET` and passes every other flag through.
