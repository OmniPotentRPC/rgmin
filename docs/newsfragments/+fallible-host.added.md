`Solver::try_step_hvp` with the `TryHessianVector` trait, and
`try_lowest_eigenpairs` / `try_lowest_eigenpairs_precond` with the
`TryApplyHessian` trait, take host callbacks that can fail. On callbacks
that succeed they take the steps of `step_hvp` and `lowest_eigenpairs`
bit for bit. The first callback error ends the call as `Error::Host`,
nothing is evaluated after it, and `x` and the session stay as they
were before the step. A Newton-CG or Newton-PCG session with a chart or
distance constraints returns `Error::FallibleUnsupported`. Any other
method returns `Error::NeedHvp`. The C ABI and the Python binding are
unchanged.
