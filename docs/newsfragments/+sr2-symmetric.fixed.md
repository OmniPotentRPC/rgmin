SR2 is a symmetric secant update.
The `sr2.hpp` map `B += δ (y + Bs)^T / (δ·s)`, with `δ = y - Bs`, gives `B+ s - y = 2 (s·Bs) δ / (δ·s)`.
It broke the secant equation (Nocedal-Wright 6.6) and left `B` asymmetric.
Once `s·Bs` exceeded `s·y`, it doubled `B` along `s` on every step, so the steps halved and SR2 stalled away from a stationary point.
SR2 now applies Powell's symmetrization with `c = y + Bs`: `B += (δ c^T + c δ^T) / (c·s) - (δ·s) c c^T / (c·s)^2`.
With `a = s·y` and `b = s·Bs` both nonzero, that is the Broyden class (6.32) with `φ = a (a - b) / (a + b)^2`.
It is not Powell symmetric Broyden (`c = s`).
A symmetric `B` stays exactly symmetric in floating point.
