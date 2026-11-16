An L-BFGS `Solver` step under `Accept::None` records the caller's move as a curvature pair before its one trial.
A non-finite trial now takes that pair back, so the memory is as it was when the step returns `Error::Oracle`.
Before, each retry from the moved point recorded the same pair again.
