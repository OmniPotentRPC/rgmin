"""Exact Liu-Storey cancellation and its binary64 SCG witness."""

from pathlib import Path
import subprocess
import sys

import sympy as sp

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions.")

r, gx, gy = sp.symbols("r gx gy", real=True)
gradient = sp.Matrix([gx, gy])
previous_direction = -gradient
current_gradient = r * gradient
y = current_gradient - gradient
beta = -current_gradient.dot(y) / previous_direction.dot(gradient)
assert sp.simplify(beta - r * (r - 1)) == 0
for component in beta * previous_direction - current_gradient + r**2 * gradient:
    assert sp.simplify(component) == 0
print("The conjugate direction scales as the squared residual factor.")

result = subprocess.run(
    ["sollya", str(Path(__file__).with_name("scg_direction_restart.sollya"))],
    capture_output=True, text=True,
)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, end="", file=sys.stderr)
result.check_returncode()
expected = [
    "conjugate direction remains descending: true",
    "squared direction is below epsilon: true",
    "squared gradient is above epsilon: true",
]
if result.stdout.splitlines() != expected or result.stderr:
    raise SystemExit("The SCG direction witness did not satisfy all three predicates.")
