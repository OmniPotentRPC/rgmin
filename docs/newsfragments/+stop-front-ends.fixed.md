An L-BFGS run that spends its budget without moving `x` is now
`StopReason::LineSearchFailed` from `Solver::run` and
`Solver::run_gauged`, as `minimize_method` and `rgmin_minimize`
already reported it. Python `minimize` reports `Stop.LINE_SEARCH`
for the same run, where it reported `Stop.BUDGET`.
