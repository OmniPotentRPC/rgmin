SR2 applies Powell's symmetrization with `c = y + Bs` and `δ = y - Bs`: `B += (δ c^T + c δ^T) / (c·s) - (δ·s) c c^T / (c·s)^2`.
With `a = s·y` and `b = s·Bs` both nonzero, that is the Broyden class (6.32) with `φ = a (a - b) / (a + b)^2`.
It is not Powell symmetric Broyden (`c = s`).
A symmetric `B` stays exactly symmetric in floating point.
