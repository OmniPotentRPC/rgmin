GROMACS `integrator = l-bfgs` energy minimization is available as `Method::GromacsLbfgs`, with `GromacsLbfgsConfig` and `GromacsLbfgs`.
It includes the `do_lbfgs` correction recursion, line search, and memory reset, and it rebuilds the direction across rebase or external coordinate moves.
`nbfgscorr`, `emtol`, `emstep`, and `nsteps` map from the MDP options, and the stop is the maximum atomic force.
C ABI minor 67 adds `RGMIN_GROMACS_LBFGS` and `rgmin_solver_set_gromacs_lbfgs`.
Python exposes `GromacsLbfgs`.
