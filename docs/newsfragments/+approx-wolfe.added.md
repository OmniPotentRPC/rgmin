Opt-in `Lbfgs::approx_wolfe` stops the strong Wolfe search of `Lbfgs::minimize_box`, and of `minimize` when `LbfgsStep::Wolfe` is selected, at the rounding floor of the value.
With a positive value the search accepts a trial at most `approx_wolfe` times `abs(f)` above the start when the trial meets the strong curvature test, the approximate Wolfe stop of Hager and Zhang 2005.
Otherwise it places that trial by the sign of its slope.
At the default of zero the search is strict. The eOn cap does not consult it.
On LJ75, with the Wolfe search, `approx_wolfe = 1e-12` ends all 400 relaxations on the `gtol` test, against 6 without it, and they need 278.6 evaluations per relaxation against 371.2.
A reused solver can then take a much too long first step after a large perturbation, so the test stays off by default.
