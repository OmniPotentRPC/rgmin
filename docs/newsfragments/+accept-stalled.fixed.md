Session steepest descent, nonlinear CG, BB, BFGS, SR1, and SR2 return `Error::AcceptStalled` when `Accept::Energy` or `Accept::Nonmonotone` refuses the steepest restart that follows a refused step.
Each later call would retry that restart from the same point, at eleven oracle calls per step.
C sessions report `RGMIN_INTERNAL_ERROR`.
`Accept::None` is unchanged.
