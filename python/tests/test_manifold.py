"""Stiefel, SPD and product manifolds through the NumPy bindings."""

from pathlib import Path

import numpy as np
import pytest

from rgmin import SPD, Product, RigidQuotient, Stiefel

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "data" / "riemannian_manifold.txt"


def stiefel_point(rng, n, p):
    q, r = np.linalg.qr(rng.normal(size=(n, p)))
    return q * np.sign(np.diag(r))


def spd_point(rng, p):
    a = rng.uniform(-1.0, 1.0, size=(p, p))
    return a @ a.T + 0.5 * np.eye(p)


def sym(a):
    return 0.5 * (a + a.T)


def is_spd(s):
    if np.max(np.abs(s - s.T)) > 1e-12:
        return False
    try:
        np.linalg.cholesky(s)
    except np.linalg.LinAlgError:
        return False
    return True


def sym_func(s, f):
    w, q = np.linalg.eigh(s)
    return (q * f(w)) @ q.T


def read_fixture():
    rows = {}
    for line in FIXTURE.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        rows[key] = np.array([float(v) for v in values])
    return rows


def fixture_manifolds():
    st_qr = Stiefel(5, 2, "qr")
    ai = SPD(3, "affine_invariant")
    return {
        "stiefel_qr": st_qr,
        "stiefel_polar": Stiefel(5, 2, "polar"),
        "spd_ai": ai,
        "spd_le": SPD(3, "log_euclidean"),
        "product": Product(st_qr, ai),
    }


def unflatten(m, flat):
    parts, start = [], 0
    for rows, cols in m.shapes:
        parts.append(flat[start : start + rows * cols].reshape((rows, cols), order="F"))
        start += rows * cols
    return tuple(parts) if isinstance(m, Product) else parts[0]


def flatten(value):
    parts = value if isinstance(value, tuple) else (value,)
    return np.concatenate([np.asarray(p).ravel(order="F") for p in parts])


@pytest.mark.parametrize("name", ["stiefel_qr", "stiefel_polar", "spd_ai", "spd_le", "product"])
def test_results_match_the_rust_fixture(name):
    rows = read_fixture()
    m = fixture_manifolds()[name]
    x, y, v, g = (unflatten(m, rows[f"{name}.{k}"]) for k in "xyvg")
    rgrad, step_x = m.gradient_step(x, g, 0.1)
    got = {
        "project": m.project(x, g),
        "rgrad": m.egrad_to_rgrad(x, g),
        "retract": m.retract(x, v),
        "exp": m.exp(x, v),
        "transport": m.transport(x, y, v),
        "inner": np.array([m.inner(x, v, v)]),
        "step_rgrad": rgrad,
        "step_x": step_x,
    }
    if f"{name}.log" in rows:
        got["log"] = m.log(x, y)
    for key, value in got.items():
        want = rows[f"{name}.{key}"]
        scale = max(1.0, float(np.max(np.abs(want))))
        assert np.max(np.abs(flatten(value) - want)) <= 1e-13 * scale, key


@pytest.mark.parametrize("retraction", ["qr", "polar"])
def test_stiefel_retraction_stays_orthonormal(retraction):
    rng = np.random.default_rng(3)
    m = Stiefel(8, 3, retraction)
    for scale in (1e-3, 0.5, 3.0):
        x = stiefel_point(rng, 8, 3)
        v = scale * m.project(x, rng.normal(size=(8, 3)))
        y = m.retract(x, v)
        assert y.shape == (8, 3)
        assert np.max(np.abs(y.T @ y - np.eye(3))) < 1e-12


@pytest.mark.parametrize("metric", ["affine_invariant", "log_euclidean"])
def test_spd_retraction_stays_positive_definite(metric):
    rng = np.random.default_rng(5)
    m = SPD(4, metric)
    for scale in (1e-3, 0.5, 3.0):
        x = spd_point(rng, 4)
        y = m.retract(x, scale * sym(rng.normal(size=(4, 4))))
        assert is_spd(y)


def test_polar_retraction_agrees_with_the_exponential_to_second_order():
    rng = np.random.default_rng(7)
    m = Stiefel(6, 3, "polar")
    x = stiefel_point(rng, 6, 3)
    v = m.project(x, rng.normal(size=(6, 3)))
    v /= np.linalg.norm(v)

    def gap(t):
        return np.linalg.norm(m.retract(x, t * v) - m.exp(x, t * v))

    for t in (0.1, 0.05, 0.025):
        assert 7.0 < gap(t) / gap(t / 2) < 9.0


def test_affine_invariant_exp_matches_the_closed_form_and_log_inverts_it():
    rng = np.random.default_rng(9)
    m = SPD(3, "affine_invariant")
    x = spd_point(rng, 3)
    v = sym(rng.normal(size=(3, 3)))
    half = sym_func(x, np.sqrt)
    ihalf = sym_func(x, lambda w: 1.0 / np.sqrt(w))
    want = half @ sym_func(sym(ihalf @ v @ ihalf), np.exp) @ half
    y = m.exp(x, v)
    assert np.allclose(y, want, rtol=1e-12, atol=1e-12)
    assert np.allclose(m.log(x, y), v, atol=1e-10)
    z = spd_point(rng, 3)
    assert np.allclose(m.exp(x, m.log(x, z)), z, atol=1e-10)


@pytest.mark.parametrize("metric", ["affine_invariant", "log_euclidean"])
def test_spd_parallel_transport_is_tangent_and_isometric(metric):
    rng = np.random.default_rng(13)
    m = SPD(3, metric)
    x, y = spd_point(rng, 3), spd_point(rng, 3)
    u, w = sym(rng.normal(size=(3, 3))), sym(rng.normal(size=(3, 3)))
    tu, tw = m.transport(x, y, u), m.transport(x, y, w)
    assert np.array_equal(tu, tu.T)
    assert m.inner(y, tu, tw) == pytest.approx(m.inner(x, u, w), rel=1e-10)


def test_stiefel_transport_lands_in_the_arrival_tangent_space():
    rng = np.random.default_rng(15)
    m = Stiefel(5, 2)
    x, y = stiefel_point(rng, 5, 2), stiefel_point(rng, 5, 2)
    t = m.transport(x, y, m.project(x, rng.normal(size=(5, 2))))
    assert np.max(np.abs(y.T @ t + t.T @ y)) < 1e-13


def test_gradient_steps_decrease_a_convex_cost_monotonically():
    """The affine-invariant gradient X G X grows with the eigenvalues of X,
    so the fixed step 0.1 suits SPD points with eigenvalues of order one."""
    rng = np.random.default_rng(17)
    m = Product(Stiefel(6, 2, "polar"), SPD(3, "affine_invariant"))
    target = (stiefel_point(rng, 6, 2), spd_point(rng, 3))
    x = (stiefel_point(rng, 6, 2), spd_point(rng, 3))

    def cost(at):
        return 0.5 * sum(np.sum((a - b) ** 2) for a, b in zip(at, target))

    last = start = cost(x)
    for _ in range(200):
        egrad = tuple(a - b for a, b in zip(x, target))
        _, x = m.gradient_step(x, egrad, 0.1)
        assert cost(x) < last
        assert np.max(np.abs(x[0].T @ x[0] - np.eye(2))) < 1e-12
        assert is_spd(x[1])
        last = cost(x)
    assert last < 1e-3 * start


def test_bad_arguments_raise_value_error():
    with pytest.raises(ValueError):
        Stiefel(2, 3)
    with pytest.raises(ValueError):
        Stiefel(4, 2, "cayley")
    with pytest.raises(ValueError):
        SPD(2, "bures")
    m = SPD(2)
    with pytest.raises(ValueError, match="shape"):
        m.retract(np.eye(3), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="positive definite"):
        m.retract(np.diag([1.0, -1.0]), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="Stiefel"):
        Stiefel(3, 2).log(np.eye(3, 2), np.eye(3, 2))
    with pytest.raises(ValueError, match="sequence"):
        Product(Stiefel(3, 1), SPD(1)).project(np.zeros((3, 1)), np.zeros((3, 1)))


TRIANGLE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.8, 0.1]).reshape(-1, 1)


def rigid_motions(x):
    atoms = x.reshape(-1, 3)
    centred = atoms - atoms.mean(axis=0)
    motions = []
    for axis in range(3):
        t = np.zeros_like(atoms)
        t[:, axis] = 1.0
        motions.append(t.reshape(-1, 1))
    for axis in range(3):
        omega = np.zeros(3)
        omega[axis] = 1.0
        motions.append(np.cross(omega, centred).reshape(-1, 1))
    return motions


def test_rigid_quotient_removes_every_rigid_motion():
    m = RigidQuotient(3)
    assert m.size == 9
    assert m.shapes == [(9, 1)]
    for motion in rigid_motions(TRIANGLE):
        assert np.linalg.norm(m.project(TRIANGLE, motion)) < 1e-12


def test_rigid_quotient_projection_is_scale_invariant():
    m = RigidQuotient(3)
    for scale in (1e-12, 1e-6, 1e6, 1e12):
        x = TRIANGLE * scale
        for motion in rigid_motions(x):
            assert np.linalg.norm(m.project(x, motion)) <= 1e-9 * np.linalg.norm(motion)


def test_rigid_quotient_horizontal_space_has_dimension_three_n_minus_six():
    m = RigidQuotient(3)
    basis = []
    for i in range(9):
        e = np.zeros((9, 1))
        e[i] = 1.0
        w = m.project(TRIANGLE, e)
        for q in basis:
            w = w - q * np.vdot(q, w)
        if np.linalg.norm(w) > 1e-10:
            basis.append(w / np.linalg.norm(w))
    assert len(basis) == 3


def test_periodic_rigid_quotient_keeps_rotations():
    m = RigidQuotient(3, periodic=True)
    translation, rotation = rigid_motions(TRIANGLE)[0], rigid_motions(TRIANGLE)[3]
    assert np.linalg.norm(m.project(TRIANGLE, translation)) < 1e-12
    assert np.linalg.norm(m.project(TRIANGLE, rotation) - rotation) < 1e-12


def test_rigid_quotient_retraction_transport_and_log():
    m = RigidQuotient(3)
    v = m.project(TRIANGLE, np.arange(9.0).reshape(-1, 1) ** 2)
    assert np.allclose(m.retract(TRIANGLE, 0.1 * v), TRIANGLE + 0.1 * v)
    moved = m.retract(TRIANGLE, 0.1 * v)
    carried = m.transport(TRIANGLE, moved, v)
    assert np.allclose(carried, m.project(moved, v))
    assert np.linalg.norm(carried - m.project(moved, carried)) < 1e-12
    with pytest.raises(ValueError, match="rigid quotient"):
        m.log(TRIANGLE, moved)
    with pytest.raises(ValueError):
        RigidQuotient(0)
    with pytest.raises(ValueError, match="shape"):
        m.project(np.zeros((6, 1)), np.zeros((6, 1)))


def test_rigid_quotient_composes_in_a_product():
    m = Product(Stiefel(3, 1), RigidQuotient(3))
    assert m.shapes == [(3, 1), (9, 1)]
    x = (np.array([[0.0], [0.0], [1.0]]), TRIANGLE)
    v = (np.array([[0.3], [0.1], [0.2]]), rigid_motions(TRIANGLE)[4])
    out = m.project(x, v)
    assert np.linalg.norm(out[1]) < 1e-12
    assert abs(np.vdot(x[0], out[0])) < 1e-12
