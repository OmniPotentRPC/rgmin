Typed lowest-mode waist: closed `EigensolverKind` (Lanczos=0 through EigenExa=13, `schema/eigen.capnp`) and `EigenParams` (`nev`, `krylov`, `max_iter`, `tol`).
Lanczos, Rayleigh-Ritz, Jacobi-Davidson, and LOBPCG run matrix-free.
Other named backends return `Error::EigenUnavailable` / `RGMIN_UNAVAILABLE`.
No string keys.
C: `rgmin_lowest_eigenpair` plus `rgmin_eigen_params_t`.
