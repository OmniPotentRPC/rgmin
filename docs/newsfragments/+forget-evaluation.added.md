`Solver::forget_evaluation` drops the cached point and keeps
the method memory. The next step evaluates the current `x`.
A session that never calls it keeps the cache it already used.
