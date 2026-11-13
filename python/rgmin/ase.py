"""ASE optimizers.

`LBFGS` is an `ase.optimize.Optimizer`. `run(fmax=0.05)` is ASE's
loop: each `step` asks rgmin for one iteration, and ASE stops when
the largest force component is below `fmax`.
"""

from __future__ import annotations

import numpy as np
from ase.optimize.optimize import Optimizer

from rgmin._core import Lbfgs


class LBFGS(Optimizer):
    """Limited-memory BFGS with ASE's per-atom step cap.

    `maxstep` is the longest atom displacement, 0.2 by default, the
    same number ASE's own LBFGS uses. The curvature pairs stay on the
    optimizer across `step` calls.
    """

    defaults = {**Optimizer.defaults, "maxstep": 0.2}

    def __init__(
        self,
        atoms,
        restart=None,
        logfile="-",
        trajectory=None,
        maxstep=None,
        memory=8,
        **kwargs,
    ):
        """Keep one Lbfgs for the life of the optimizer."""
        super().__init__(
            atoms, restart=restart, logfile=logfile, trajectory=trajectory, **kwargs
        )
        if maxstep is None:
            maxstep = self.defaults["maxstep"]
        self.maxstep = maxstep
        self._lbfgs = Lbfgs(
            memory=memory, gtol=0.0, ftol=0.0, step="scaled", max_atom=float(maxstep), norm="infinity"
        )

    def step(self, f=None):
        """Move `atoms` by one iteration. `f` is accepted and ignored."""
        del f
        pos = np.ascontiguousarray(self.atoms.get_positions(), dtype=float).reshape(-1)
        saved = self.atoms.get_positions().copy()

        def fg(x):
            trial = np.asarray(x, dtype=float).reshape((-1, 3))
            self.atoms.set_positions(trial)
            try:
                energy = float(self.atoms.get_potential_energy())
                grad = -np.asarray(self.atoms.get_forces(), dtype=float).reshape(-1)
            finally:
                self.atoms.set_positions(saved)
            return energy, np.ascontiguousarray(grad, dtype=float)

        _value, new_x, _evals = self._lbfgs.minimize(pos, 1, fg)
        self.atoms.set_positions(np.asarray(new_x, dtype=float).reshape((-1, 3)))
