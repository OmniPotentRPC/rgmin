"""Exact quadratic identities and one binary64 line-search witness.

Endpoint derivatives do not prove descent for a general nonlinear function.
The energy window is a policy; this script does not bound arbitrary oracle error.
"""

from pathlib import Path
import subprocess
import sys

import sympy as sp

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions.")

alpha, curvature, slope, offset, c1 = sp.symbols(
    "alpha curvature slope offset c1", real=True
)
t = sp.symbols("t", real=True)
quadratic = offset + slope * t + curvature * t**2 / 2
start_slope = sp.diff(quadratic, t).subs(t, 0)
end_slope = sp.diff(quadratic, t).subs(t, alpha)
change = quadratic.subs(t, alpha) - quadratic.subs(t, 0)
assert sp.simplify(change - alpha * (start_slope + end_slope) / 2) == 0
assert sp.simplify(
    change - c1 * alpha * start_slope
    - alpha * (end_slope - (2 * c1 - 1) * start_slope) / 2
) == 0

# This cubic satisfies the endpoint slope tests and has positive exact change.
cubic = -3 * t**3 + 5 * t**2 - t
assert sp.diff(cubic, t).subs(t, 0) == -1
assert sp.diff(cubic, t).subs(t, 1) == 0
assert cubic.subs(t, 1) - cubic.subs(t, 0) == 1
# The secant product has the sign of the endpoint derivative difference.
assert sp.simplify(alpha * (end_slope - start_slope) - curvature * alpha**2) == 0

# A value window bounds true increase only with bounded endpoint errors.
f0, f1, e0, e1 = sp.symbols("f0 f1 e0 e1", real=True)
m0, m1 = f0 + e0, f1 + e1
assert sp.simplify((f1 - f0) - ((m1 - m0) + e0 - e1)) == 0
print("Exact quadratic identities, secant algebra, and nonlinear limitation verified.")

result = subprocess.run(
    ["sollya", str(Path(__file__).with_name("roundoff_wolfe.sollya"))],
    capture_output=True,
    text=True,
)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, end="", file=sys.stderr)
result.check_returncode()
expected = [
    "opening energy is six: true",
    "computed energy rises one ulp: true",
    "exact quadratic energy decreases: true",
    "energy change is within the relative window: true",
    "approximate Wolfe upper slope holds: true",
    "strong curvature holds: true",
    "cancelled objective is one hundred: true",
    "configured witness exact decrease: true",
    "configured witness exceeds four epsilons: true",
    "configured witness fits requested window: true",
    "configured witness satisfies both slope conditions: true",
]
if result.stdout.splitlines() != expected or result.stderr:
    raise SystemExit("The floating-point witness did not satisfy all predicates.")
