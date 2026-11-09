`Solver::set_lindh` installs the Lindh model Hessian on an L-BFGS
session. The matrix stays off until the caller asks for it, so a
default session keeps the steps it took before.
