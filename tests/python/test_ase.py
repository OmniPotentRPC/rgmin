"""ASE LBFGS on an EMT hydrogen molecule."""

import numpy as np
from ase.build import molecule
from ase.calculators.emt import EMT

from rgmin.ase import LBFGS


def test_h2_relaxes_below_fmax():
    atoms = molecule("H2")
    atoms.positions[1, 0] = 1.3
    atoms.calc = EMT()
    opt = LBFGS(atoms, logfile=None)
    converged = opt.run(fmax=0.05, steps=80)
    forces = atoms.get_forces()
    assert np.abs(forces).max() < 0.05
    assert converged
    bond = np.linalg.norm(atoms.positions[1] - atoms.positions[0])
    assert 0.6 < bond < 1.0
