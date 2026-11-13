`SuppliedSearch` and `rgmin_linesearch` take a caller-supplied energy and gradient.
The search does not call an objective. Its trials match `LineSearch::search`.
`rgmin_linesearch` and `rgmin_lowest_eigenpairs` are both ABI minor 37.
