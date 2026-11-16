Opt-in `Lbfgs::minimize_box` and `Lbfgs::minimize_box_watched` relax inside a box `lower <= x <= upper` with the generalized Cauchy point and the subspace minimization of L-BFGS-B (Byrd, Lu, Nocedal and Zhu 1995).
Each direction ends at the subspace minimizer from the Cauchy point, and the strong Wolfe search stops at the box.
`Lbfgs::minimize` keeps the eOn cap. `LbfgsStep::Wolfe` remains opt-in.
