A C session created with `istep = 0` sets its step scale to `min(1, 1/||g||)` at the first energy-tested steepest, BFGS, SR1, SR2, BB, or unpreconditioned CG step.
The scale outlasts that step and stays until `forget` or `rebase`.
Steepest descent and CG multiply every later gradient term by it, so the cap shortens all their steps.
BFGS, SR1, and SR2 use it while the matrix is the starting identity, and BB when it has no pair or no positive curvature.
SciPy BFGS starts its search at `min(1, 1.01/||g||)`, and L-BFGS-B at `1/||d||`.
The raw `-g` step rose on every energy halving, so from `(-1.2, 1.0)` on Rosenbrock 2D these sessions stayed at `f = 24.2`.
A positive `istep` and `Accept::None` are unchanged.
