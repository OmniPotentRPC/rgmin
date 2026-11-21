"""The line search takes the caller's energy and gradient."""

from rgmin import LineSearch


def quad(x):
    z = x[0] - 1.0
    return z * z, [2.0 * z]


def tiny_quad(x):
    z = x[0] - 5e-14
    return z * z, [2.0 * z]


def run(kind, istep, fg=quad, **kw):
    kw.setdefault("maxiter", 20)
    search = LineSearch(kind, [0.0], [1.0], istep=istep, **kw)
    f, g = fg([0.0])
    x, alpha, value, done = search.offer(f, g)
    for _ in range(10 * kw["maxiter"]):
        if done:
            return x, alpha, value
        f, g = fg(x)
        x, alpha, value, done = search.offer(f, g)
    raise AssertionError("search did not finish")


def test_wolfe_accepts_the_first_trial():
    x, alpha, value = run("wolfe", 0.25)
    assert abs(x[0] - 0.25) < 1e-12
    assert abs(alpha - 0.25) < 1e-12
    assert abs(value - quad([0.25])[0]) < 1e-12


def test_backtracking_reaches_the_minimum():
    x, _alpha, value = run("backtracking", 8.0)
    assert abs(x[0] - 1.0) < 1e-8
    assert value < 1e-14


def test_brent_resolves_a_minimum_below_the_old_floor():
    x, alpha, _value = run("brent", 1.0, tiny_quad, maxiter=128, tol=0.0)
    assert x[0] == alpha
    assert abs(alpha - 5e-14) < 1e-15
