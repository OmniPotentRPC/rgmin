"""Symbolic checks for the L-BFGS map in src/lbfgs.rs.

Run: uv run --with sympy python validation/lbfgs_two_loop.py

1. The two-loop recursion with H0 = gamma I equals H_k g, where H_k is
   the BFGS inverse update (Nocedal-Wright 6.17) applied to the stored
   pairs oldest first. Checked for n = 3, m = 1, 2, 3 with exact
   rational pairs and a fully symbolic g.
2. gamma = s'y / y'y (Nocedal-Wright 7.20) is exact for a 1-D quadratic
   (gamma = 1 / h) and, for y = A s with A symmetric positive definite,
   lies in [1 / lambda_max, 1 / lambda_min].
3. Replaying the newest pair (the `extra_updates` path) is the identity
   on H: the BFGS update with (s, y) maps any H that already satisfies
   H y = s to itself, so a repeated pair changes nothing in exact
   arithmetic.
"""

import random

import sympy as sp

random.seed(7)
n = 3
g = sp.Matrix(sp.symbols("g0:3", real=True))
I = sp.eye(n)


def rand_spd():
    while True:
        L = sp.Matrix(n, n, lambda i, j: sp.Rational(random.randint(-4, 4), random.randint(1, 3)))
        A = L * L.T + sp.eye(n)
        if A.det() != 0:
            return A


def bfgs_inverse(H, s, y):
    rho = 1 / (y.T * s)[0]
    return (I - rho * s * y.T) * H * (I - rho * y * s.T) + rho * s * s.T


def two_loop(pairs, q, gamma):
    alphas = []
    for s, y in reversed(pairs):
        rho = 1 / (y.T * s)[0]
        a = rho * (s.T * q)[0]
        alphas.append(a)
        q = q - a * y
    r = gamma * q
    for (s, y), a in zip(pairs, reversed(alphas)):
        rho = 1 / (y.T * s)[0]
        b = rho * (y.T * r)[0]
        r = r + (a - b) * s
    return r


A = rand_spd()
for m in (1, 2, 3):
    pairs = []
    for _ in range(m):
        s = sp.Matrix([sp.Rational(random.randint(-5, 5), random.randint(1, 4)) for _ in range(n)])
        if all(v == 0 for v in s):
            s[0] = 1
        pairs.append((s, A * s))
    s_new, y_new = pairs[-1]
    gamma = (s_new.T * y_new)[0] / (y_new.T * y_new)[0]
    H = gamma * I
    for s, y in pairs:
        H = bfgs_inverse(H, s, y)
    lhs = two_loop(pairs, g, gamma)
    assert sp.simplify(lhs - H * g) == sp.zeros(n, 1), m
print("1. two-loop(g) == H_k g for m = 1, 2, 3 (n = 3, symbolic g)  [ok]")

h, x = sp.symbols("h s", positive=True)
assert sp.simplify((x * (h * x)) / ((h * x) ** 2) - 1 / h) == 0
# gamma = s'As / s'A^2 s = z'z / z'Az with z = A^(1/2) s: the inverse of a
# Rayleigh quotient of A, hence within [1/lambda_max, 1/lambda_min].
lams = [float(sp.re(sp.N(ev, 30))) for ev in A.eigenvals()]
lam_min, lam_max = min(lams), max(lams)
for _ in range(50):
    s = sp.Matrix([sp.Rational(random.randint(-9, 9), random.randint(1, 5)) for _ in range(n)])
    if all(v == 0 for v in s):
        continue
    y = A * s
    gam = (s.T * y)[0] / (y.T * y)[0]
    gv = float(gam)
    assert 1 / lam_max - 1e-12 <= gv <= 1 / lam_min + 1e-12, (gv, lam_min, lam_max)
print("2. gamma = 1/h in 1-D; 1/lambda_max <= gamma <= 1/lambda_min [ok]")

Hs = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"h{min(i, j)}{max(i, j)}", real=True))
s = sp.Matrix(sp.symbols("s0:3", real=True))
y = sp.Matrix(sp.symbols("y0:3", real=True))
H1 = bfgs_inverse(Hs, s, y)
# H1 satisfies the secant equation ...
assert sp.simplify(H1 * y - s) == sp.zeros(n, 1)
# ... and a second update with the same pair returns H1 itself.
H2 = bfgs_inverse(H1, s, y)
assert sp.simplify(H2 - H1) == sp.zeros(n, n)
print("3. a replayed newest pair leaves H unchanged                 [ok]")
print("all identities hold")
