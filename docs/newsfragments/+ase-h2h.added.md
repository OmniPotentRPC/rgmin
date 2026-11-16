`benches/ase_h2h.py` compares ASE 3.29 LBFGS, FIRE and BFGSLineSearch with rgmin capped L-BFGS, Wolfe L-BFGS, the geometry step and FIRE 2.0.
Every arm uses fmax 0.01 and maxstep 0.2 on the same LJ7, LJ13, LJ38 and Morse Pt256 starts.
Pt256 is the periodic Morse cluster of 256 platinum atoms.
The table is `benches/ase_h2h/results.md`.
