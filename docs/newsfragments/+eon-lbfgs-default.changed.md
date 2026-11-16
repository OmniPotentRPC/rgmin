The default L-BFGS step is the eOn client's line-search-free cap, including the pieces the previous cap left out: the finite-difference curvature probe and `distance_reset`.
The first step probes curvature over `EON_FINITE_DIFFERENCE` (0.01) and sets `H0 = 1/C` when that curvature is positive.
Later steps use `(s·y)/(y·y)` when the quotient is positive, and `EON_INVERSE_CURVATURE` (0.01) otherwise.
A proposal whose longest atom move is at least the cap clears the memory and steps along `H0` times the force.
A direction that is not descent does the same.
Negative curvature clears the memory and steps along the force at the cap.
The cap only shortens a step.
`LbfgsStep::Scaled` and `rgmin_solver_set_lbfgs_scaled` keep the previous rule, which shortens the two-loop step and retains the pairs.
`LbfgsStep::Wolfe` remains the explicit strong Wolfe search.
SciPy's CG coefficient `max(0, β_PR)` is `Conjugacy::PolakRibierePlus` (`prp`).
The eOn conjugate-gradient restart, `|g·g_old| >= 0.5 ||g_old||^2`, is `Restart::powell` and the method token `eon-cg`.
Hager-Zhang now applies the CG_DESCENT floor `β = max(β_N, -1 / (||d|| min(0.01, ||g_old||)))`.
FIRE 1.0 treats a zero power as downhill, as the eOn client does.
The comparison is `bench/results.md`, with the raw rows in `bench/raw/`.
`abi_minor` 35.
