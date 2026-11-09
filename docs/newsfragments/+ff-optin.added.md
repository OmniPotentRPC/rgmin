`Solver::set_ff` installs the bonded metric on an L-BFGS session.
The matrix stays off until the caller asks for it, so a default
session keeps the steps it took before.
