Solver::set_fire_config now rejects out-of-range dt and dt_max values instead of silently skipping them, and the Python binding now raises ValueError for the same invalid inputs.
