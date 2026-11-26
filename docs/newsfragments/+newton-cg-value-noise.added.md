Newton-CG stops at the rounding floor of the objective value when the caller states that floor.
With `NoiseFloor::with_eps_f` set on `Control::noise_floor`, `minimize_newton_cg`, `minimize_newton_cg_precond` and the `newton-cg` and `newton-pcg` sessions stop with `Status::NoiseFloor` at the last accepted point when a trust region trial is refused while the model predicts a reduction of at most `c * eps_f`.
Without `eps_f` such a run still shrinks the radius to its floor and returns `Error::TrustCollapsed`.
The report keeps the gradient norm at that point, so `converged` is not set above `gtol` and `require_convergence` still returns `Error::NotConverged`.
`NoiseFloor::model_below_energy_noise` is the test.
The C ABI adds `noise_eps_f` to `rgmin_control_t` and `RGMIN_CONTROL_REVISION6_SIZE` at minor 68, and the layout revision becomes 7.
A positive `noise_eps_f` is `eps_f` with or without a positive `noise_eps`, and the stop is `RGMIN_STOP_NOISE_FLOOR` with `x` at the last accepted point.
A binary built against revision 6 keeps its 64-byte control: the `_sized` entries and `rgmin_control_init` accept that size and leave `noise_eps_f` at 0.
`rgmin_abi_compatible` returns 0 for a caller that reports layout revision 6, as for any layout revision other than this build's.
