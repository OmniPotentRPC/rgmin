Opt-in FIRE 2.0 parameter set and accelerated bias-corrected (ABC) FIRE on `fire2` sessions.
`Solver::set_fire_preset(FirePreset::Guenole2020)` takes Guenole et al. 2020 Table 1 and keeps `dt` when a shrink would pass `dt_min` (Algorithm 2).
`Solver::set_fire_abc` divides the mixed velocity by `1 - (1 - alpha)^(N+1)` (Echeverri Restrepo and Andric, Comput. Mater. Sci. 218, 111978 (2023)), as ASE `FIRE2(use_abc=True)` does.
Under ABC the cap bounds each displacement component.
C: `rgmin_fire_preset_t`, `rgmin_solver_set_fire_preset` and `rgmin_solver_set_fire_abc`; the C++ session and xts aliases follow.
`abi_minor` 30.
How-to `docs/orgmode/howto/fire.org` shows both.
`benches/stop_norm.rs` adds the two settings and a mean step count.
