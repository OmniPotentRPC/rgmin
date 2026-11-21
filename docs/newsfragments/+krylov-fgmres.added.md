- Added `rgmin::krylov`: restarted, right-preconditioned flexible GMRES
  (`fgmres`) with a caller-chosen product budget, an optional
  multi-cycle stagnation rule and a per-step observer, and the
  Eisenstat-Walker choice-2 forcing term (`EisenstatWalker`). Every
  cycle closes on a recomputed `b - A x`; only that residual converges
  a solve, is reported and restarts the next cycle.
