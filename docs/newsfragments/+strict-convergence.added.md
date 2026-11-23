`Control::with_require_convergence` opts into
`Error::NotConverged(Box<Report>)` when a finished run is still not
below `gtol`. The default stays `Ok`. `Report.converged` records the
result. The comparison is `grad_norm < gtol`, so a gradient exactly
at `gtol` is not converged. GPMin's standalone `run` uses that
comparison and returns through the same path as the other minimizers,
so `Report.converged` is set and `require_convergence` and
`error_on_unmet` apply. Its session step uses the same comparison
for a machine-precision stall.
A strict run stops early as `StepTolerance` only when an accepted step
leaves the point unchanged and the next iteration would repeat it. A
rejected trust region trial or a rejected adaptive step is not a stall,
so a run that converges by default converges identically when strict.
C ABI minor 55 and layout revision 6 add the control field, the report
flag, `RGMIN_NOT_CONVERGED`, and `rgmin_solver_set_require_convergence`.
A binary built against revision 5 keeps its 56-byte control and
report through `stop`; `rgmin.h` routes each entry to a `_sized`
symbol so a rebuilt caller has `require_convergence` read, and
`rgmin_control_init` fills either layout.
C++ throws `rgmin::NotConverged` and Python `minimize` raises
`rgmin.NotConverged`; both carry the report.
