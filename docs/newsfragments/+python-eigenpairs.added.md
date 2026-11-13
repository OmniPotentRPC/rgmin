`rgmin.lowest_eigenpairs` is the Python binding of block LOBPCG.
It takes a Hessian-vector callable and a seed block and returns NumPy arrays.
A pair that misses the tolerance raises `RuntimeError`.
`examples/tutorial_lowest_eigenpairs.py` is run in CI.
