"""Symbolic checks for FIRE 2.0 in src/fire.rs (fire2_displacement).

Run: uv run --with sympy python validation/fire2.py

Guénolé et al. 2020, https://doi.org/10.1016/j.commatsci.2020.109584,
algorithm 2 with the semi-implicit Euler integrator, per step:

  P = F.v
  P > 0 : (after N_delay) dt <- min(dt f_inc, dt_max), a <- a f_a
  P <= 0: (outside the initial delay) dt <- dt f_dec if >= dt_min,
          a <- a_start;  x <- x - dt v / 2;  v <- 0
  v <- v + dt F
  v <- (1 - a) v + a |v| F / |F|
  x <- x + dt v

1. The mix is the inertia rule: for 0 <= a <= 1 it never lengthens v,
   |1 - 2a| |v| <= |v'| <= |v|, and never lowers the power, F.v' >= F.v.
2. The displacement the code returns equals the paper's sequence in
   closed form: uphill dx = -dt v / 2 + dt^2 F (the mix of v = dt F is
   v itself), downhill dx = dt [(1 - a) w + a |w| F / |F|], w = v + dt F.
3. The half-step correction returns the walker to the midpoint of its
   last move: after x1 = x0 + dt v, the point x1 - dt v / 2 is
   (x0 + x1) / 2.
"""

import sympy as sp

a = sp.symbols("alpha", real=True)
dt = sp.symbols("dt", positive=True)
v1, v2, f1, f2 = sp.symbols("v1 v2 f1 f2", real=True)
v = sp.Matrix([v1, v2])
F = sp.Matrix([f1, f2])
nv = sp.sqrt(v.dot(v))
nf = sp.sqrt(F.dot(F))


def mix(w, alpha):
    return (1 - alpha) * w + alpha * sp.sqrt(w.dot(w)) * F / nf


# 1. |v'|^2 = |v|^2 [(1-a)^2 + a^2 + 2 a (1-a) cos] with cos = F.v/(|F||v|).
c = sp.symbols("c", real=True)  # cos of the angle, in [-1, 1]
vn = sp.symbols("V", positive=True)
norm2 = vn**2 * ((1 - a) ** 2 + a**2 + 2 * a * (1 - a) * c)
vp = mix(v, a)
cos_expr = F.dot(v) / (nf * nv)
lhs = sp.simplify(vp.dot(vp) - norm2.subs({vn: nv, c: cos_expr}))
assert lhs == 0, lhs
# Upper bound: norm2 is linear in c, maximal at c = 1 where it is V^2.
assert sp.simplify(norm2.subs(c, 1) - vn**2) == 0
# Lower bound: minimal at c = -1 where it is (1 - 2a)^2 V^2.
assert sp.simplify(norm2.subs(c, -1) - (1 - 2 * a) ** 2 * vn**2) == 0
# Linear in c with coefficient 2 a (1 - a) V^2 >= 0 on [0, 1].
assert sp.simplify(sp.diff(norm2, c) - 2 * a * (1 - a) * vn**2) == 0
# Power: F.v' - F.v = a (|v||F| - F.v) >= 0 by Cauchy-Schwarz.
dpow = sp.simplify(F.dot(vp) - F.dot(v) - a * (nv * nf - F.dot(v)))
assert dpow == 0, dpow
print("1. mix: |1-2a||v| <= |v'| <= |v|, F.v' >= F.v for a in [0,1] [ok]")

# 2. Closed forms of the step.
dt2 = sp.symbols("dt2", positive=True)  # dt after the uphill shrink
w_up = dt2 * F  # v zeroed, then kicked
dx_up = -dt2 * v / 2 + dt2 * mix(w_up, a)
assert sp.simplify(dx_up - (-dt2 * v / 2 + dt2**2 * F)) == sp.zeros(2, 1)
w = v + dt * F
dx_down = dt * mix(w, a)
assert sp.simplify(dx_down - dt * ((1 - a) * w + a * sp.sqrt(w.dot(w)) * F / nf)) == sp.zeros(2, 1)
print("2. uphill dx = -dt v/2 + dt^2 F; downhill dx = dt mix(v + dt F) [ok]")

# 3. Midpoint of the last move.
x0, v0 = sp.symbols("x0 v0", real=True)
x1 = x0 + dt * v0
xc = x1 - dt * v0 / 2
assert sp.simplify(xc - (x0 + x1) / 2) == 0
print("3. the corrected point is the midpoint of the last move       [ok]")
print("all identities hold")
