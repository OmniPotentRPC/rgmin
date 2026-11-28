

# A quadratic in Python

Install the package from PyPI, then minimize
$f(x) = \tfrac12 x \cdot x$.

```bash
python -m pip install rgmin
```

A checkout that can build the Rust static library installs the same
package with `python -m pip install .` from the repository root.

```python
import numpy as np
import rgmin

def f_and_grad(x):
    return 0.5 * float(x @ x), np.array(x, copy=True)

x0 = np.array([1.0, -2.0, 0.5], dtype=np.float64)
report, x = rgmin.minimize(f_and_grad, x0, method="lbfgs", gtol=1e-12)
assert report.grad_norm < 1e-8
```

`x0` has to be a contiguous float64 array. The same array is updated
in place, and `minimize` also returns it.

One iteration at a time uses a solver:

```python
config = rgmin.SolverConfig("lbfgs", memory=20)
solver = rgmin.Solver(x0.shape[0], config)
report = solver.step(f_and_grad, x0)
```

ASE is a separate install, `python -m pip install 'rgmin[ase]'`.
`import rgmin` does not import ASE.
