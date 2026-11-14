"""Exact SPD metric algebra and scalar accepted-point witnesses.

The matrix identity is exact for nonsingular symmetric 2x2 X and symmetric
U. The scalar identity covers positive diagonal X when no ridge is needed.
These checks do not bound arbitrary-dimensional Cholesky roundoff.
"""
from pathlib import Path
import subprocess
import sys
import sympy as sp

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions.")
a,b,c,e,f,g,h,u,v,w=sp.symbols("a b c e f g h u v w",real=True)
x=sp.Matrix([[a,b],[b,c]])
egrad=sp.Matrix([[e,f],[g,h]])
eta=sp.Matrix([[u,v],[v,w]])
symmetric=(egrad+egrad.T)/2
rgrad=x*symmetric*x
metric=sp.trace(x.inv()*rgrad*x.inv()*eta)
derivative=sp.trace(egrad.T*eta)
assert sp.factor(metric-derivative)==0
q,t=sp.symbols("q t",real=True)
retract=lambda value,direction: value+direction+direction**2/(2*value)
y=retract(q,t)
assert sp.factor(y-((q+t)**2+q**2)/(2*q))==0
assert sp.factor(retract(q,y-q)-y-(y-q)**2/(2*q))==0
assert retract(sp.Integer(2),-sp.Rational(2,5))==sp.Rational(41,25)
assert retract(sp.Integer(2),sp.Rational(41,25)-2)==sp.Rational(4181,2500)
print("Exact metric-gradient and scalar retraction identities verified.")
result=subprocess.run(["sollya",str(Path(__file__).with_name("spd_retraction.sollya"))],capture_output=True,text=True)
print(result.stdout,end="")
if result.stderr:
    print(result.stderr,end="",file=sys.stderr)
result.check_returncode()
expected=[
    "single retraction matches the exact diagonal point: true",
    "chord retraction matches its distinct diagonal point: true",
    "the extra retraction changes the point beyond roundoff: true",
    "the exact positive scalar retraction stays positive: true",
]
if result.stdout.splitlines()!=expected or result.stderr:
    raise SystemExit("The SPD retraction witness did not satisfy every predicate.")
