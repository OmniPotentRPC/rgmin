`Control::stop_norm` selects the norm that a `Solver` session compares to `gtol`.
It defaults to `StopNorm::Two`, so a default `Control` stops where it did.
`Solver::set_stop_norm` still replaces it on a live session.
`StopNorm::Rms` is `||g||_2 / sqrt(n)`, the `rms` convergence metric of eOn.
`Report::stop_norm` names the norm of `Report::grad_norm`, and drivers that run no session report `StopNorm::Two`.
C: `RGMIN_STOP_NORM_RMS` is 3, `XTS_STOP_NORM_RMS` aliases it, and `abi_minor` is 32.
