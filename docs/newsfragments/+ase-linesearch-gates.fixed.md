The ASE BFGS line search skips an inverse-Hessian pair when the step length is not positive or the absolute directional derivative did not fall, as ASE does, and it still declines a pair with nonpositive curvature.
BFGSLineSearch and LBFGSLineSearch leave the opening evaluation out of the force-call count, as ASE does, and a positional use_line_search raises TypeError.
A maxstep of zero stops that search before a non-finite trial. That stop is intentional: ASE scales the next zero increment by 0/0.
