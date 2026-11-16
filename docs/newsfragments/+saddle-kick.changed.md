Newton and dogleg leave a first-order saddle.
When the force is already under `gtol`, the report reads the analytic Hessian.
The floor is `max(|gtol|, 1e-6, 1e-8 ||H||_F)`.
A lowest eigenvalue below minus that floor steps along the mode, scaled to the per-atom cap, at most four times.
The dogleg radius restarts from the initial step.
A converged report costs one extra Hessian.
