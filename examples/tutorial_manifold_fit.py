"""Fit a point on St(6, 2) x SPD(3) to a target by gradient steps.

Continuous integration runs this file.
"""

import numpy as np

from rgmin import SPD, Product, Stiefel

rng = np.random.default_rng(0)


def frame(n, p):
    q, r = np.linalg.qr(rng.normal(size=(n, p)))
    return q * np.sign(np.diag(r))


def spd(p):
    a = rng.uniform(-1.0, 1.0, size=(p, p))
    return a @ a.T + 0.5 * np.eye(p)


manifold = Product(Stiefel(6, 2, "polar"), SPD(3, "affine_invariant"))
target = (frame(6, 2), spd(3))
x = (frame(6, 2), spd(3))


def cost_and_gradient(point):
    """f(U, S) = |U - U*|^2 / 2 + |S - S*|^2 / 2 and its Euclidean gradient."""
    diff = tuple(a - b for a, b in zip(point, target))
    return 0.5 * sum(np.sum(d * d) for d in diff), diff


start, _ = cost_and_gradient(x)
last = start
for _ in range(200):
    _, egrad = cost_and_gradient(x)
    rgrad, x = manifold.gradient_step(x, egrad, 0.1)
    cost, _ = cost_and_gradient(x)
    assert cost < last
    last = cost

u, s = x
assert np.max(np.abs(u.T @ u - np.eye(2))) < 1e-12
assert np.min(np.linalg.eigvalsh(s)) > 0.0
assert last < 1e-3 * start
print(f"cost {start:.3e} -> {last:.3e}, |grad| = {manifold.norm(x, rgrad):.3e}")
