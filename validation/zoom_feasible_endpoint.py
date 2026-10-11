"""Exact witness for finite-domain zoom exhaustion and its Armijo fallback."""

import sympy as sp


def main():
    low, high = sp.S.Zero, sp.S.One
    wall = sp.Rational(1, 10)
    c1, c2 = sp.Rational(1, 10000), sp.Rational(9, 10)
    for _ in range(6):
        trial = (low + high) / 2
        if trial > wall:
            high = trial
        else:
            assert -trial <= -c1 * trial
            assert not (sp.S.One <= c2)
            low = trial
    assert trial == sp.Rational(7, 64) > wall
    assert low == sp.Rational(3, 32) < wall
    assert -low < 0
    assert -low <= -c1 * low
    # Quadratic-exact central and forward derivative audits agree at a wall.
    x, h, a, b, c = sp.symbols("x h a b c", real=True)
    polynomial = a + b * x + c * x**2
    forward = (-3 * polynomial + 4 * polynomial.subs(x, x + h)
               - polynomial.subs(x, x + 2 * h)) / (2 * h)
    central = (polynomial.subs(x, x + h) - polynomial.subs(x, x - h)) / (2 * h)
    assert sp.simplify(forward - sp.diff(polynomial, x)) == 0
    assert sp.simplify(central - sp.diff(polynomial, x)) == 0
    print("Exact feasible zoom fallback and bounded derivative identities verified.")


if __name__ == "__main__":
    main()
