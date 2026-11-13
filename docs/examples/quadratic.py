"""x^2 + y^2 from (1, -2). The value ends below 1e-8."""

import numpy as np

from rgmin import Lbfgs


def fg(x):
    """Value and gradient of x^2 + y^2."""
    x = np.asarray(x, dtype=float)
    return float(x @ x), 2.0 * x


def main():
    """Print the minimized value."""
    opt = Lbfgs()
    value, point, _evals = opt.minimize(np.array([1.0, -2.0]), 40, fg)
    if not value < 1e-8 or max(abs(point[0]), abs(point[1])) >= 1e-4:
        raise SystemExit(f"quadratic {value} {point}")
    print(f"quadratic {value:.3e}")


if __name__ == "__main__":
    main()
