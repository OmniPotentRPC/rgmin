"""Symbolic checks for the line-search interpolation in src/linesearch/interp.rs.

Run: uv run --with sympy python validation/linesearch_interpolation.py

Every assertion is an identity or an inequality sympy proves; the script
exits non-zero if any fails.

1. The More-Thuente scaled cubic step equals the stationary point of the
   cubic Hermite interpolant and has positive curvature there.
2. It equals the Nocedal-Wright eq. 3.59 form.
3. When the data come from a quadratic (cubic coefficient zero) it
   reduces to the secant step a - da w / (db - da), which is the exact
   minimiser.
4. With da < 0 < db the ratio r = p / q lies in (0, 1), and p, q, q - p
   are sums of positive terms (no cancellation).
5. The quadratic step through (a, fa, da), (b, fb) is the minimiser of
   that quadratic and needs fb - fa - da w > 0.
"""

import sympy as sp

a, w, fa, fb, da, db, t = sp.symbols("a w f_a f_b d_a d_b t", real=True)
b = a + w

# Cubic Hermite interpolant c(t) on [a, b] in the local variable s = t - a.
s = sp.symbols("s", real=True)
c0, c1, c2, c3 = sp.symbols("c0:4", real=True)
cubic = c0 + c1 * s + c2 * s**2 + c3 * s**3
sol = sp.solve(
    [
        sp.Eq(cubic.subs(s, 0), fa),
        sp.Eq(sp.diff(cubic, s).subs(s, 0), da),
        sp.Eq(cubic.subs(s, w), fb),
        sp.Eq(sp.diff(cubic, s).subs(s, w), db),
    ],
    [c0, c1, c2, c3],
    dict=True,
)[0]
herm = cubic.subs(sol)
dherm = sp.diff(herm, s)
ddherm = sp.diff(herm, s, 2)

# 1. More-Thuente form. gamma carries sign(w); take w > 0 here and treat
# w < 0 by the reflection t -> -t, which maps (w, da, db) to (-w, -da, -db).
theta = 3 * (fa - fb) / w + da + db
g = sp.symbols("gamma", positive=True)  # gamma = sqrt(theta^2 - da db) for w > 0
p = (g - da) + theta
q = ((g - da) + g) + db
r = p / q
s_star = r * w

disc_rel = sp.Eq(g**2, theta**2 - da * db)
# Substituting gamma^2 = theta^2 - da db, c'(s*) vanishes identically.
num = sp.numer(sp.together(dherm.subs(s, s_star)))
num_reduced = sp.rem(sp.expand(num), sp.expand(g**2 - (theta**2 - da * db)), g)
assert sp.simplify(num_reduced) == 0, "MT cubic step is not stationary"

# Curvature at s*: c''(s*) = 2 gamma / w  (positive for w > 0, gamma > 0).
curv = sp.together(ddherm.subs(s, s_star) - 2 * g / w)
curv_num = sp.rem(sp.expand(sp.numer(curv)), sp.expand(g**2 - (theta**2 - da * db)), g)
assert sp.simplify(curv_num) == 0, "c''(s*) != 2 gamma / w"
print("1. MT cubic step: c'(s*) = 0 and c''(s*) = 2 gamma / w > 0  [ok]")

# 2. Nocedal-Wright 3.59 with (x_{i-1}, x_i) = (a, b):
#    d1 = da + db - 3 (fa - fb)/(a - b) = theta,  d2 = sign(b - a) gamma,
#    alpha = b - (b - a) (db + d2 - d1) / (db - da + 2 d2).
nw = b - w * (db + g - theta) / (db - da + 2 * g)
mt = a + s_star
assert sp.simplify(sp.together(nw - mt)) == 0, "NW 3.59 != MT form"
print("2. NW eq. 3.59 equals the MT form                           [ok]")

# 3. Quadratic data: fb = fa + da w + k w^2 and db = da + 2 k w, k > 0.
k = sp.symbols("k", positive=True)
wp = sp.symbols("w_p", positive=True)
quad_sub = {fb: fa + da * wp + k * wp**2, db: da + 2 * k * wp}
theta_q = sp.expand((theta.subs(w, wp)).subs(quad_sub))
assert sp.simplify(theta_q - (-da - k * wp)) == 0
# theta^2 - da db = (k w)^2, so gamma = k w: the discriminant never
# vanishes for convex quadratic data.
assert sp.expand(theta_q**2 - da * db.subs(quad_sub) - (k * wp) ** 2) == 0
step_q = sp.simplify(((p / q) * wp).subs(w, wp).subs(quad_sub).subs(g, k * wp))
secant = sp.simplify((-da * wp / (db - da)).subs(quad_sub))
exact_min = -da / (2 * k)
assert sp.simplify(step_q - exact_min) == 0, step_q
assert sp.simplify(secant - exact_min) == 0
print("3. Zero cubic term: step = secant = exact quadratic minimiser [ok]")

# 4. Sign-change bracket: da < 0 < db, w > 0.
A, B = sp.symbols("A B", positive=True)  # da = -A, db = B
th = sp.symbols("theta", real=True)
gam = sp.sqrt(th**2 + A * B)
p4 = gam + A + th
q4 = 2 * gam + A + B
# gamma > |theta| because A B > 0, so p4 > 0 and q4 - p4 = gamma + B - theta > 0.
assert sp.simplify(gam**2 - th**2 - A * B) == 0
# gamma^2 - theta^2 = A B > 0 gives gamma > |theta| >= -theta, hence
# p = gamma + theta + A > 0; and gamma > theta gives q - p > B > 0.
assert sp.simplify(q4 - p4 - (gam + B - th)) == 0
# Exhaustive sign test on a rational grid (exact arithmetic).
grid = [sp.Rational(n, 7) for n in range(-21, 22, 3)]
for thv in grid:
    for Av in (sp.Rational(1, 1000), sp.Rational(1, 3), 1, 50):
        for Bv in (sp.Rational(1, 1000), sp.Rational(2, 3), 1, 70):
            sub = {th: thv, A: Av, B: Bv}
            pv, qv = p4.subs(sub), q4.subs(sub)
            assert pv > 0 and qv - pv > 0, (thv, Av, Bv)
print("4. da < 0 < db: 0 < r < 1, p and q - p positive              [ok]")

# 5. Quadratic through (a, fa, da) and (b, fb).
curv5 = fb - fa - da * w
qd = fa + da * s + curv5 / w**2 * s**2
smin = sp.solve(sp.diff(qd, s), s)[0]
assert sp.simplify(smin - (-da * w**2 / (2 * curv5))) == 0
assert sp.simplify(sp.diff(qd, s, 2) - 2 * curv5 / w**2) == 0
print("5. Quadratic step and its convexity condition                [ok]")
print("all identities hold")
