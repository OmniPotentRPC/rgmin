The cyclic Jacobi eigensolver no longer squares a rotation parameter `theta` above `1 / eps`, with eps the machine epsilon.
Above that bound, the solver takes the tangent `t = 1 / (2 theta)`, the limit of the full formula.
The Linear Algebra Package (LAPACK) takes the same branch in its one-sided Jacobi routines.
The transition-state Broyden-Fletcher-Goldfarb-Shanno (TS-BFGS) update and the rational function optimization (RFO) step call this solver.
An off-diagonal entry below about 3.7e-155 times its diagonal gap made `theta^2` overflow.
With overflow trapped, the host stopped with the floating-point exception signal (SIGFPE); with the trap masked, the solver skipped the rotation.
Every result that was finite before is unchanged.
