`Solver::run` and `Solver::run_gauged` now check the coordinate and
objective dimensions against the session before a run starts. A
mismatch returns `Error::Dim` instead of reaching the projection onto
the bounds first.
