"""Exact reset algebra and bounded binary64 directional witnesses.

The real-arithmetic proofs assume a nonzero gradient and positive scale.
The floating-point examples do not bound arbitrary vector overflow or underflow.
"""
from pathlib import Path
import subprocess
import sys
import sympy as sp

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions.")

s = sp.Matrix([1, 0, 0])
y = sp.Matrix([1, 1, 0])
g = sp.Matrix([-4, 0, 0])
gamma = (s.dot(y)) / y.dot(y)
rho = 1 / s.dot(y)
identity = sp.eye(3)
h = (identity - rho*s*y.T) * (gamma*identity) * (identity-rho*y*s.T) + rho*s*s.T
assert -h*g == sp.Matrix([6, -2, 0])
assert -gamma*g == sp.Matrix([2, 0, 0])
assert (-gamma*g).dot(g) == -8
scale, norm_squared, cap = sp.symbols("scale norm_squared cap", positive=True)
assert (-scale*norm_squared).is_negative
assert (-cap*scale*norm_squared).is_negative
old_direction, gradient = sp.Integer(-2), sp.Integer(-1)
assert old_direction*gradient == 2
assert (-abs(sp.Integer(-2))*gradient)*gradient == -2
print("Finite-dimensional positive-scale descent and signed fallback counterexample verified.")
result = subprocess.run(["sollya", str(Path(__file__).with_name("neb_reset.sollya"))], capture_output=True, text=True)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, end="", file=sys.stderr)
result.check_returncode()
expected = [
    "signed fallback is uphill: true",
    "magnitude fallback has exact positive scale: true",
    "magnitude fallback is downhill: true",
    "positive cap preserves descent: true",
    "capped direction reaches the specified radius: true",
]
if result.stdout.splitlines() != expected or result.stderr:
    raise SystemExit("The reset witness did not satisfy every predicate.")
