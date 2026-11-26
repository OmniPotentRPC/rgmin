`rgmin.ase` follows the ASE 3.29 optimizer contract for evaluation-log restart and the `FIRE` and `FIRE2` constructor keywords. `LBFGS` keeps rgmin's capped eOn step by default and offers ASE length control with `eon_cap=False`. `Newton`, `RFO`, and `Dogleg` require an explicit Hessian instead of synthesizing one with finite differences.
ABI minor 69 adds `rgmin_solver_set_fire_config` and `rgmin_solver_set_lbfgs_fixed_h0`.
`tests/python/test_ase_upstream.py` runs ASE's optimizer tests over every `rgmin.ase` class.
