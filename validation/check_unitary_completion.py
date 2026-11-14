"""Exact complement algebra and a bounded binary64 projection witness.

The symbolic projector has exact arithmetic and nonzero norm. The examples
are not a bound on arbitrary dimensions, conditioning, overflow, or underflow.
"""
from pathlib import Path
import subprocess
import sys
import sympy as sp

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions.")
a, b, c, d = sp.symbols("a b c d", real=True)
q = sp.Matrix([a + sp.I*b, c + sp.I*d])
w = sp.Matrix([-c + sp.I*d, a - sp.I*b])
norm_squared = (q.conjugate().T*q)[0].expand()
assert sp.simplify((q.conjugate().T*w)[0]) == 0
assert sp.simplify((w.conjugate().T*w)[0] - norm_squared) == 0
p = sp.eye(2) - q*q.conjugate().T/norm_squared
assert all(sp.simplify(entry) == 0 for entry in p*q)
assert all(sp.simplify(entry) == 0 for entry in p*p-p)
assert sp.simplify(sp.trace(p)) == 1
assert sp.simplify(sum((p[:, j].conjugate().T*p[:, j])[0] for j in range(2))) == 1
print("Exact Hermitian projection, complement orthogonality, and positive residual sum verified.")
result = subprocess.run(["sollya", str(Path(__file__).with_name("unitary_completion.sollya"))], capture_output=True, text=True)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, end="", file=sys.stderr)
result.check_returncode()
expected = [
    "duplicate columns violate orthogonality: true",
    "coordinate residual has positive squared norm: true",
    "completed column is orthogonal in real part: true",
    "completed column is orthogonal in imaginary part: true",
    "completed column has unit norm within four epsilons: true",
]
if result.stdout.splitlines() != expected or result.stderr:
    raise SystemExit("The unitary completion witness did not satisfy every predicate.")
