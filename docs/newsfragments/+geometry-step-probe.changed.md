Capped Cartesian L-BFGS now reserves its finite-difference curvature probe for nominal steps at least 64 times larger than the cap.
Ordinary starts use a `0.1 I` inverse scale and five-value nonmonotone acceptance.
An installed exponential preconditioner replaces that scalar seed, including those steepest fallbacks.
Uncapped L-BFGS keeps the two-loop step and the Wolfe search.
A HiGHS feasible set and a non-Euclidean manifold stay on that path.
