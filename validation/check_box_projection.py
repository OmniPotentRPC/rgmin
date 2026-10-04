"""Exact coordinate-projection algebra and bounded floating witnesses.

The finite-dimensional descent inequality assumes every displacement interval
contains zero. The floating predicates cover the retained near-wall fixture;
they do not certify arbitrary floating-point sums or a HiGHS solution.
"""
from pathlib import Path
import subprocess
import sys
import sympy as sp

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions.")
g, lower, upper = sp.symbols("g lower upper", real=True)
assert sp.expand(g*lower+lower**2-lower*(g+lower)) == 0
assert sp.expand(g*upper+upper**2-upper*(g+upper)) == 0
assert sp.expand(g*(-g)+(-g)**2) == 0
pair_s = sp.Matrix([1, 2])
pair_y = sp.Matrix([1, 0])
identity = sp.eye(2)
v = identity-pair_s*pair_y.T/(pair_s.dot(pair_y))
inverse = v*v.T+pair_s*pair_s.T/(pair_s.dot(pair_y))
gradient = sp.Matrix([1, -1])
assert -inverse*gradient == sp.Matrix([1, 7])
assert gradient.dot(sp.Matrix([1, 0])) == 1
assert gradient.dot(sp.Matrix([-1, 0])) == -1
print("Exact projected-force identities and retained restart witness verified.")
result = subprocess.run(["sollya", str(Path(__file__).with_name("box_projection.sollya"))], capture_output=True, text=True)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, end="", file=sys.stderr)
result.check_returncode()
expected = [
    "free coordinate keeps its full step: true",
    "wall coordinate lands on the bound: true",
    "the projected near-wall step descends: true",
    "the projected restart reverses the bad slope: true",
]
if result.stdout.splitlines() != expected or result.stderr:
    raise SystemExit("The box-projection witness did not satisfy every predicate.")
