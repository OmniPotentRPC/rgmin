`lowest_eigenpairs` is block LOBPCG for the k lowest eigenpairs of a symmetric Hessian product (Knyazev 2001).
The seed is an n-by-k block. Each pair carries its residual and whether that residual met the tolerance.
`k = 0`, `k` above the dimension, and a non-finite value or product return an error.
C: `rgmin_lowest_eigenpairs`, `rgmin_eigenpair_t`, and `rgmin_eigen_options_t`.
The C++ wrapper is `rgmin::lowest_eigenpairs`, and `xts_lowest_eigenpairs` aliases the C entry.
`rgmin_linesearch` and `rgmin_lowest_eigenpairs` are both ABI minor 37.
