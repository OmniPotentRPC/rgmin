`Solver::set_cell_block(at, maxmove)` reads `[at, dim)` as cell coordinates, such as the box degrees of freedom of a variable-cell relaxation.
The per-atom cap groups only the atoms in triples and bounds each cell coordinate by `maxmove`.
`StopNorm::MaxAtom` counts each cell coordinate on its own.
C callers use `rgmin_solver_set_cell_block` at ABI minor 50, and `xts.h` spells it `xts_solver_set_cell_block`.
