# Python

`Lbfgs.minimize` reads a C-contiguous NumPy vector without an extra
copy. The array handed to the callback is a view of the solver buffer
and is invalid after the callback returns. The result is a new NumPy
array. The quadratic `x^2 + y^2` from `(1, -2)` ends below `1e-8`.

```{literalinclude} ../examples/quadratic.py
:language: python
```

ASE:

`from rgmin.ase import LBFGS` then `LBFGS(atoms).run(fmax=0.05)`.

`LBFGS` is an `ase.optimize.Optimizer`. `maxstep` defaults to 0.2.
`run` is ASE's loop. Each `step` is one rgmin iteration, and ASE stops
when the largest force component is below `fmax`.
