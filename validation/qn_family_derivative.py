"""Derivative of the signed QN family away from its denominator floor."""

import sympy as sp


def main():
    alpha, curvature, gradient, sign = sp.symbols(
        "alpha curvature gradient sign", real=True
    )
    denominator = curvature + alpha * sign
    step = -gradient / denominator
    expected = gradient * sign / denominator**2
    assert sp.simplify(sp.diff(step, alpha) - expected) == 0
    for mode_sign in (-1, 1):
        signed_step = step.subs({curvature: 2 * mode_sign, sign: mode_sign})
        assert sp.simplify(
            sp.diff(signed_step, alpha)
            - expected.subs({curvature: 2 * mode_sign, sign: mode_sign})
        ) == 0
    witness = expected.subs(
        {curvature: -2, gradient: 2, sign: -1, alpha: sp.Rational(1, 2)}
    )
    assert witness == -sp.Rational(8, 25)
    print("Signed QN derivatives and the flipped-mode witness verified.")


if __name__ == "__main__":
    main()
