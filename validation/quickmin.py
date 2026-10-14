"""Symbolic checks for QuickMin in src/quickmin.rs.

Run: uv run --with sympy python validation/quickmin.py

The step, with F = -g at the current point:

  v <- (v.F / F.F) F   if v.F >= 0,   else 0       (projection P)
  v <- v + dt F                                    (kick)
  x <- x + cap(dt v)                               (move)

1. P never adds kinetic energy and leaves no component along -F:
   |v|^2 - |P v|^2 = |v - P v|^2 >= 0 (Pythagoras, v - P v is orthogonal
   to F), the Lagrange identity |v|^2 |F|^2 - (v.F)^2 = sum of squares,
   and (P v).(-F) = -max(v.F, 0) <= 0. P keeps the power when it was
   nonnegative and is idempotent.
2. On E = lam x^2 / 2 with the projection inactive the step is the
   symplectic Euler map: det 1, trace 2 - h, h = dt^2 lam. Its roots lie
   on the unit circle exactly for 0 <= h <= 4: dt < 2 / sqrt(lam) is the
   stability limit of the unquenched dynamics.
3. From rest the move is dt^2 F. On a quadratic with curvature q <= lam
   |F|^2 along F the energy change -tau |F|^2 + tau^2 q / 2 (tau = dt^2)
   is negative for 0 < tau lam < 2; in one dimension the ratio
   E1 / E0 = (1 - h)^2, below 1 exactly for 0 < h < 2.
4. With a kept velocity v = c F (c >= 0) the step is steepest descent
   with tau = dt (c + dt). In one dimension the coefficient obeys
   c' = (c + dt) / (1 - tau lam) while the walker stays on the same
   side, so c grows and tau lam crosses 2: the eOn step can raise the
   energy. A concrete two-dimensional run exhibits the rise.
5. The backtracking variant restarts from the anchor at rest with
   dt <- f_dec dt. After k cuts tau lam = f_dec^(2k) dt0^2 lam, below 2
   once k >= log(dt0^2 lam / 2) / (2 log(1 / f_dec)); from then on every
   accepted anchor is lower than the one before (item 3).
6. The curvature cap dt <= c / sqrt(lam_hat), lam_hat = |y| / |s|, gives
   h = dt^2 lam <= c^2 lam / lam_hat. When lam_hat equals the curvature
   lam of the mode, c = sqrt(2) is the edge of item 3 (h = 2, the mode
   flips sign and keeps its energy) and c = 2 the edge of item 2; descent
   from rest needs c < sqrt(2). lam_hat >= the Rayleigh quotient along
   s, so the bound is never looser than the Rayleigh estimate gives.
"""

import math

import sympy as sp

# 1. Projection.
v = sp.Matrix(sp.symbols("v1:4", real=True))
F = sp.Matrix(sp.symbols("f1:4", real=True))
vf = v.dot(F)
ff = F.dot(F)
Pv = (vf / ff) * F
lagrange = sp.expand(v.dot(v) * ff - vf**2)
squares = sum(
    (v[i] * F[j] - v[j] * F[i]) ** 2 for i in range(3) for j in range(i + 1, 3)
)
assert sp.simplify(lagrange - sp.expand(squares)) == 0
# |v|^2 - |Pv|^2 = (|v|^2 |F|^2 - (v.F)^2) / |F|^2 >= 0.
assert sp.simplify(v.dot(v) - Pv.dot(Pv) - lagrange / ff) == 0
# Pythagoras: v - Pv is orthogonal to F.
assert sp.simplify((v - Pv).dot(F)) == 0
assert sp.simplify((v - Pv).dot(v - Pv) - (v.dot(v) - Pv.dot(Pv))) == 0
# The power is kept, P is idempotent, and the -F component is -v.F.
assert sp.simplify(Pv.dot(F) - vf) == 0
assert ((Pv.dot(F) / ff) * F - Pv).applyfunc(sp.simplify) == sp.zeros(3, 1)
assert sp.simplify(Pv.dot(-F) + vf) == 0
print("1. |Pv| <= |v|, Pv.F = max(v.F, 0) >= 0, (Pv).(-F) <= 0, P idempotent [ok]")

# 2. Stability of the unquenched step on E = lam x^2 / 2.
dt, lam = sp.symbols("dt lam", positive=True)
h, r = sp.symbols("h r", real=True)
M = sp.Matrix([[1 - dt**2 * lam, dt], [-dt * lam, 1]])  # (x, v) -> (x', v')
assert sp.simplify(M.det() - 1) == 0
assert sp.simplify(M.trace() - (2 - dt**2 * lam)) == 0
charpoly = M.charpoly(r).as_expr()
assert sp.simplify(charpoly - (r**2 - (2 - dt**2 * lam) * r + 1)) == 0
# Complex roots (|r| = 1) when the discriminant (2 - h)^2 - 4 <= 0.
disc = sp.factor((2 - h) ** 2 - 4)
assert sp.simplify(disc - h * (h - 4)) == 0
assert sp.solve_univariate_inequality(
    h * (h - 4) <= 0, h, relational=False
) == sp.Interval(0, 4)
# Outside, the larger root has modulus above 1.
root = ((2 - h) - sp.sqrt(disc)) / 2
assert abs(float(root.subs(h, 4.5))) > 1
print("2. symplectic Euler: r^2 - (2 - h) r + 1, stable iff 0 <= dt^2 lam <= 4 [ok]")

# 3. Descent from rest.
tau, n, q = sp.symbols("tau n q", positive=True)
dE = -tau * n + tau**2 * q / 2
# At q = lam n (worst case) dE = -tau n (1 - tau lam / 2).
assert sp.simplify(dE.subs(q, lam * n) + tau * n * (1 - tau * lam / 2)) == 0
x0 = sp.symbols("x0", positive=True)
x1 = x0 + dt**2 * (-lam * x0)
ratio = sp.simplify((lam * x1**2 / 2) / (lam * x0**2 / 2))
assert sp.simplify(ratio - (1 - dt**2 * lam) ** 2) == 0
assert sp.solve_univariate_inequality(
    (1 - h) ** 2 < 1, h, relational=False
) == sp.Interval.open(0, 2)
print("3. from rest: dE = -tau|F|^2 (1 - tau lam / 2) < 0 for tau lam < 2 [ok]")

# 4. The eOn step raises the energy on an anisotropic quadratic.
c = sp.symbols("c", nonnegative=True)
x = sp.symbols("x", positive=True)
Fx = -lam * x
vx = c * Fx
x_new = x + dt * (vx + dt * Fx)
tau_c = dt * (c + dt)
assert sp.simplify(x_new - (1 - tau_c * lam) * x) == 0
v_new = vx + dt * Fx
F_new = -lam * x_new
c_new = sp.simplify(v_new * F_new / F_new**2)
assert sp.simplify(c_new - (c + dt) / (1 - tau_c * lam)) == 0


def eon_quickmin(H, x, dt, steps):
    """Exact-arithmetic eOn QuickMin on E = x.Hx / 2; returns energies."""
    vel = sp.zeros(len(x), 1)
    out = [(x.T * H * x)[0] / 2]
    for _ in range(steps):
        f = -H * x
        p = vel.dot(f)
        vel = (p / f.dot(f)) * f if p >= 0 else sp.zeros(len(x), 1)
        vel = vel + dt * f
        x = x + dt * vel
        out.append((x.T * H * x)[0] / 2)
    return out


H = sp.diag(1, 100)
energies = eon_quickmin(H, sp.Matrix([1, sp.Rational(1, 100)]), sp.Rational(1, 8), 40)
rises = [k for k in range(1, len(energies)) if energies[k] > energies[k - 1]]
assert rises, "expected an energy rise in the eOn step"
print(
    f"4. eOn step: tau = dt (c + dt), c' = (c + dt) / (1 - tau lam); "
    f"energy rises at steps {rises[:3]} on diag(1, 100), dt = 1/8 [ok]"
)

# 5. Cuts to reach descent from rest.
f_dec = sp.Rational(1, 2)
dt0sq_lam = sp.symbols("D", positive=True)
k = sp.symbols("k", nonnegative=True)
cond = f_dec ** (2 * k) * dt0sq_lam
k_min = sp.log(dt0sq_lam / 2) / (2 * sp.log(1 / f_dec))
assert sp.simplify(cond.subs(k, k_min) - 2) == 0
for D in (3, 10, 100, 1e4):
    kk = math.ceil(math.log(D / 2) / (2 * math.log(2)))
    assert 0.25**kk * D < 2 + 1e-12
print("5. backtrack: k >= log(dt0^2 lam / 2) / (2 log(1/f_dec)) cuts restore descent [ok]")

# 6. Curvature cap.
cc, lam_hat = sp.symbols("c lam_hat", positive=True)
h_cap = (cc / sp.sqrt(lam_hat)) ** 2 * lam
assert sp.simplify(h_cap - cc**2 * lam / lam_hat) == 0
assert sp.simplify(h_cap.subs({cc: sp.sqrt(2), lam_hat: lam}) - 2) == 0
assert sp.simplify(h_cap.subs({cc: 2, lam_hat: lam}) - 4) == 0
# |y|/|s| >= s.Hs / s.s for symmetric positive definite H (2 x 2 check).
a, b, d, s1, s2 = sp.symbols("a b d s1 s2", real=True)
Hs = sp.Matrix([[a, b], [b, d]])
s = sp.Matrix([s1, s2])
y = Hs * s
gap = sp.expand(y.dot(y) * s.dot(s) - s.dot(y) ** 2)
assert sp.simplify(gap - sp.expand((y[0] * s[1] - y[1] * s[0]) ** 2)) == 0
print("6. curvature cap: dt^2 lam = c^2 lam / lam_hat, |y|/|s| >= Rayleigh [ok]")
print("all QuickMin checks passed")
