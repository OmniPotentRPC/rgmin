Steihaug-Toint Newton-CG as `Method::NewtonCg` (`newton-cg`) and `Method::NewtonPcg` (`newton-pcg`).
`Method::from_name` parses the token.
The session uses Hessian-vector products only: analytic via `Solver::step_hvp` / `minimize_method_hvp`, otherwise one forward difference of the known gradient.
`control.maxmove` is the Euclidean trust-radius ceiling; a per-atom cap clips the accepted trial, and the boundary test uses the pre-clip length.
`Solver::gradient_calls` and `Solver::hvp_calls` count the two separately.
C: `RGMIN_NEWTON_CG` (21), `RGMIN_NEWTON_PCG` (22), `rgmin_method_from_name`, `rgmin_solver_step_hvp`, `rgmin_minimize_hvp`, `rgmin_solver_gradient_calls`, `rgmin_solver_hvp_calls`.
`abi_minor` 27.
Steihaug, Society for Industrial and Applied Mathematics (SIAM) Journal on Numerical Analysis 20 (1983), <https://doi.org/10.1137/0720042>.
Nocedal and Wright, *Numerical Optimization*, chapter 7.
