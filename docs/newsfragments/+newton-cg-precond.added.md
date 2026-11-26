Added `minimize_newton_cg_precond`, which runs the trust-region Newton-CG loop under a caller `Preconditioner`.
`minimize_newton_cg` is the same loop with `IdentityPrecond` and keeps its iterates.
The C ABI adds `rgmin_vec_precond_fn`, `rgmin_minimize_hvp_precond` and `rgmin_minimize_hvp_precond_sized` at minor 66 with the layout revision left at 6.
The entry takes a fused value-and-gradient callback, a Hessian action and an optional preconditioner inverse (null is the identity), one `user` pointer for all three, and returns the status of the first callback that fails.
`xts.h` aliases them as `xts_vec_precond_fn` and `xts_minimize_hvp_precond`.
