# rgmin

Python bindings for [rgmin](https://github.com/OmniPotentRPC/rgmin).

`BFGS`, `LBFGS` and `FIRE` take the same arguments as the ASE optimizers
of those names: `atoms`, `trajectory`, `logfile`, `run(fmax, steps)`, and
`attach`. `LBFGS` shortens the step the way ASE does. Pass `eon_cap=True`
for the line-search-free cap that is the Rust default.

```python
from ase.build import molecule
from ase.calculators.emt import EMT
from rgmin import LBFGS

atoms = molecule("H2")
atoms.calc = EMT()
LBFGS(atoms, logfile=None).run(fmax=0.05)
```
