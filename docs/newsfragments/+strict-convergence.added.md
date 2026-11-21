`Control::with_require_convergence` opts into
  `Error::NotConverged(Box<Report>)` when a finished run is still not
  below `gtol`. The default stays `Ok`. `Report.converged` records the
  result. The comparison is `grad_norm < gtol`, so a gradient exactly
  at `gtol` is not converged.
  A strict run stops early as `StepTolerance` only when an accepted step
  leaves the point unchanged and the next iteration would repeat it. A
  rejected trust region trial or a rejected adaptive step is not a stall,
  so a run that converges by default converges identically when strict.
  C ABI minor 53 and layout revision 6 add the control field, the report
  flag, `RGMIN_NOT_CONVERGED`, and `rgmin_solver_set_require_convergence`.
  C++ throws `rgmin::NotConverged` and Python `minimize` raises
  `rgmin.NotConverged`; both carry the report.
