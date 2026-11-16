`lammps/` builds `min_style rgmin`, a LAMMPS plugin (`plugin load`, LAMMPS 22Jul2025 or later) that runs the rgmin `lbfgs`, `fire`, `fire2`, `sd` and `bb` sessions through the C ABI.
`minimize etol ftol maxiter maxeval` keeps its meaning: `ftol` is the force norm of `min_modify norm` over the full force after fixes, and `etol` is the relative energy test of `min_style cg`.
`min_modify method`, `memory`, `trust`, `linesearch`, `accept` and `istep` select and tune the session, and `dmax` is the per-atom step cap.
`fix box/relax`, per-atom extra degrees of freedom and unknown keywords stop with an error.
Under MPI every rank runs the same session on the full coordinate vector, and 1, 2 and 4 ranks take the same steps.
`howto/lammps-migration` and `reference/lammps-min-style` document the plugin, and `benches/lammps_minstyle.py` compares it with `cg`, `sd`, `fire` and `quickmin`.
