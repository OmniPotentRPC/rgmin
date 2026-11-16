`minimize_scg`, `minimize_scg_exact` and `rgmin_minimize_scg` bound every trial step by a positive finite `Control::maxmove`, as that field documents.
Before, scaled conjugate gradient (SCG) ignored it.
The bound applies at the first trial and at each retry after a non-finite value.
Zero, negative, infinite and NaN values leave SCG unbounded.
Runs whose bound never binds give the same reports as before.
The comparison ratio of a bounded step divides by the damped model's change at that step, as in Nocedal and Wright eq. 4.4.
A short step therefore no longer halves the damping on an agreement as low as 0.375.
The page `explanation/step-bounds` derives the ratio and sets out what can and cannot fix the size of a bound on log-hyperparameters.
