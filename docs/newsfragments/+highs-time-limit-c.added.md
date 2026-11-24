C: `rgmin_solver_set_highs_time_limit` sets the HiGHS time limit on a
session, where zero or a negative value clears it and NaN or an infinite
value is refused and leaves the limit as it was. `rgmin_solver_highs_timeouts`
reads the count of Hessian steps that fell back after a solve reached the
limit. A step that cannot fall back returns `RGMIN_INVALID_PARAMETER`, and
`rgmin_last_error` names the time limit, or the iteration limit when a
projection QP used up its budget. The C++ `Solver` wraps both as
`set_highs_time_limit` and `highs_timeouts`, and `xts.h` aliases them. ABI
minor 61.
