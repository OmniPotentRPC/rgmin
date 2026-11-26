`error_on_unmet` now treats a `MachinePrecision` stop as unmet, as GROMACS
does when it converges to machine precision but not to the requested `Fmax`.
`StopReason::is_unmet`, `Solver::run`, the C `*_opts` one-shots
(`RGMIN_UNMET_STOP`), and Python `minimize` (`StopUnmetError`) agree.
