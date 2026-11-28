# Distribution head-to-head

Fifty seeded starts for LJ13, LJ38 and LJ75. Morse Pt256 is one
structure. fmax 0.01. A start counts as converged when that force
is reached within 5000 steps. The median of an even sample is the
mean of the two central values. IQR is Q3 minus Q1 at indexes
`n/4` and `3n/4` of the sorted force calls.

rgmin-default is the line-search-free cap, memory 8, Nocedal-Wright
H0. rgmin-matched is that cap with memory 100 and H0 fixed at 1/70,
ASE LBFGS's defaults. rgmin-wolfe is the opt-in strong Wolfe search.
rgmin-fire2 uses timestep 0.1 and a Euclidean cap of 0.2.
ASE keeps its own memory, Hessian scale and FIRE parameters, with
maxstep 0.2. On LJ75, ASE LBFGS and the matched arm both spend the
5000-step budget and stop short of fmax 0.01. The Nocedal-Wright
scale on the default arm reaches that force. The opt-in Wolfe counts
match the before table: that search is unchanged.

## Before: Wolfe was the session default

Same starts and fmax, measured before the default changed.

| problem | conv | median calls | IQR | median energy |
|---|---:|---:|---:|---:|
| LJ13 | 50/50 | 320.5 | 329 | -30.645 |
| LJ38 | 50/50 | 826.0 | 320 | -164.187 |
| LJ75 | 50/50 | 1317.5 | 473 | -376.431 |

## After

| problem | arm | conv | median calls | IQR | median energy |
|---|---|---:|---:|---:|---:|
| LJ13 | ase-lbfgs | 50/50 | 77.5 | 29 | -39.705 |
| LJ13 | ase-fire | 50/50 | 168.0 | 67 | -39.688 |
| LJ13 | ase-bfgslinesearch | 50/50 | 111.0 | 25 | -39.750 |
| LJ13 | rgmin-default | 50/50 | 85.5 | 37 | -39.709 |
| LJ13 | rgmin-matched | 50/50 | 74.0 | 28 | -39.739 |
| LJ13 | rgmin-wolfe | 50/50 | 320.5 | 329 | -30.645 |
| LJ13 | rgmin-fire2 | 50/50 | 171.0 | 83 | -39.688 |
| LJ38 | ase-lbfgs | 50/50 | 235.5 | 86 | -165.407 |
| LJ38 | ase-fire | 50/50 | 496.5 | 307 | -164.758 |
| LJ38 | ase-bfgslinesearch | 50/50 | 318.0 | 73 | -165.513 |
| LJ38 | rgmin-default | 50/50 | 206.5 | 64 | -166.038 |
| LJ38 | rgmin-matched | 50/50 | 237.5 | 80 | -165.430 |
| LJ38 | rgmin-wolfe | 50/50 | 826.0 | 320 | -164.187 |
| LJ38 | rgmin-fire2 | 50/50 | 547.5 | 369 | -165.655 |
| LJ75 | ase-lbfgs | 0/50 | 5001.0 | 0 | -337.694 |
| LJ75 | ase-fire | 50/50 | 891.5 | 561 | -375.901 |
| LJ75 | ase-bfgslinesearch | 50/50 | 583.5 | 110 | -377.497 |
| LJ75 | rgmin-default | 50/50 | 329.5 | 144 | -376.354 |
| LJ75 | rgmin-matched | 0/50 | 5001.0 | 0 | -336.574 |
| LJ75 | rgmin-wolfe | 50/50 | 1317.5 | 473 | -376.431 |
| LJ75 | rgmin-fire2 | 50/50 | 1171.0 | 532 | -374.597 |
| MorsePt256 | ase-lbfgs | 1/1 | 20.0 | 0 | -1396.173 |
| MorsePt256 | ase-fire | 1/1 | 44.0 | 0 | -1396.172 |
| MorsePt256 | ase-bfgslinesearch | 1/1 | 11.0 | 0 | -1396.173 |
| MorsePt256 | rgmin-default | 1/1 | 11.0 | 0 | -1396.173 |
| MorsePt256 | rgmin-matched | 1/1 | 20.0 | 0 | -1396.173 |
| MorsePt256 | rgmin-wolfe | 1/1 | 37.0 | 0 | -1396.172 |
| MorsePt256 | rgmin-fire2 | 1/1 | 46.0 | 0 | -1396.172 |
