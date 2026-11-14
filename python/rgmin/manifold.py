# MIT License
# Copyright 2026 Rohit Goswami
"""Stiefel, SPD and product manifolds on NumPy arrays.

A :class:`Stiefel` point is an ``(n, p)`` array with ``U.T @ U = I``. An
:class:`SPD` point is a ``(p, p)`` symmetric positive definite array. A
:class:`Product` point is a tuple with one array per factor, in factor
order. Tangent vectors and Euclidean gradients have the shape of the
point. Every method returns new arrays and raises ``ValueError`` on a
wrong shape, a non-finite entry, or an SPD point that is not symmetric
positive definite.

The arithmetic is the Rust :class:`rgmin::RiemannianManifold` behind the
C ABI ``rgmin_manifold_*`` functions. Arrays cross the boundary
column-major, so a C caller packing ``U.ravel(order="F")`` sees the same
numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import numpy.typing as npt

from rgmin._core import RiemannianManifold

__all__ = ["Manifold", "Product", "SPD", "Stiefel"]

Array = npt.NDArray[np.float64]
Point = TypeVar("Point", Array, tuple[Array, ...])
StiefelRetraction = Literal["qr", "polar"]
SpdMetric = Literal["affine_invariant", "log_euclidean"]


class Manifold(Generic[Point]):
    """Operations shared by :class:`Stiefel`, :class:`SPD` and :class:`Product`."""

    def __init__(self, core: RiemannianManifold) -> None:
        self._core = core
        self._shapes = [(rows, cols) for _, rows, cols in core.factors()]

    @property
    def shapes(self) -> list[tuple[int, int]]:
        """``(rows, columns)`` of each factor."""
        return list(self._shapes)

    @property
    def size(self) -> int:
        """Number of doubles in one packed point."""
        return len(self._core)

    def _pack(self, value: Any, name: str) -> list[float]:
        parts = self._parts(value, name)
        chunks = []
        for part, shape in zip(parts, self._shapes):
            array = np.asarray(part, dtype=np.float64)
            if array.shape != shape:
                raise ValueError(f"{name}: shape {array.shape} is not {shape}")
            chunks.append(array.ravel(order="F"))
        return np.concatenate(chunks).tolist()

    def _parts(self, value: Any, name: str) -> Sequence[Any]:
        return (value,)

    def _unpack(self, flat: list[float]) -> Any:
        data = np.asarray(flat, dtype=np.float64)
        rows, cols = self._shapes[0]
        return data.reshape((rows, cols), order="F")

    def project(self, x: Point, v: Point) -> Point:
        """Orthogonal projection of the ambient ``v`` onto the tangent space at ``x``.

        Stiefel: ``V - X sym(X^T V)``. SPD: ``sym(V)``.
        """
        return self._unpack(self._core.project(self._pack(x, "x"), self._pack(v, "v")))

    def egrad_to_rgrad(self, x: Point, egrad: Point) -> Point:
        """Riemannian gradient at ``x`` from the Euclidean gradient ``egrad``.

        Stiefel: the projection. SPD affine-invariant: ``X sym(G) X``.
        SPD log-Euclidean: ``Dlog(X)^-2 [sym(G)]``.
        """
        return self._unpack(
            self._core.egrad_to_rgrad(self._pack(x, "x"), self._pack(egrad, "egrad"))
        )

    def retract(self, x: Point, v: Point) -> Point:
        """Retraction of the step ``v`` at ``x``.

        Stiefel uses its QR or polar retraction. SPD uses the exponential
        of its metric.
        """
        return self._unpack(self._core.retract(self._pack(x, "x"), self._pack(v, "v")))

    def exp(self, x: Point, v: Point) -> Point:
        """Riemannian exponential at ``x``. ``v`` is projected to the tangent space first."""
        return self._unpack(self._core.exp(self._pack(x, "x"), self._pack(v, "v")))

    def log(self, x: Point, y: Point) -> Point:
        """Riemannian logarithm of ``y`` at ``x``. SPD factors only."""
        return self._unpack(self._core.log(self._pack(x, "x"), self._pack(y, "y")))

    def transport(self, x_from: Point, x_to: Point, v: Point) -> Point:
        """Vector transport of ``v`` from ``x_from`` to ``x_to``.

        Stiefel: projection at ``x_to``. SPD: parallel transport of the metric.
        """
        return self._unpack(
            self._core.transport(
                self._pack(x_from, "x_from"), self._pack(x_to, "x_to"), self._pack(v, "v")
            )
        )

    def inner(self, x: Point, u: Point, v: Point) -> float:
        """Riemannian inner product of the tangents ``u`` and ``v`` at ``x``."""
        return self._core.inner(self._pack(x, "x"), self._pack(u, "u"), self._pack(v, "v"))

    def norm(self, x: Point, v: Point) -> float:
        """Riemannian norm of the tangent ``v`` at ``x``."""
        return float(np.sqrt(self.inner(x, v, v)))

    def gradient_step(self, x: Point, egrad: Point, t: float) -> tuple[Point, Point]:
        """One Riemannian gradient step of size ``t``.

        Returns ``(rgrad, x_next)``: the Riemannian gradient of ``egrad`` at
        ``x`` and the retraction of ``-t * rgrad`` at ``x``.
        """
        rgrad, nxt = self._core.gradient_step(
            self._pack(x, "x"), self._pack(egrad, "egrad"), float(t)
        )
        return self._unpack(rgrad), self._unpack(nxt)


class Stiefel(Manifold[Array]):
    """Stiefel manifold St(n, p) of ``(n, p)`` frames with ``U.T @ U = I``.

    Embedded metric. ``retraction`` is ``"qr"`` (first order) or
    ``"polar"`` (second order). Transport is projection at the arrival
    point. There is no closed-form logarithm.
    """

    def __init__(self, n: int, p: int, retraction: StiefelRetraction = "qr") -> None:
        super().__init__(RiemannianManifold.stiefel(n, p, retraction))
        self.n = n
        self.p = p
        self.retraction = retraction


class SPD(Manifold[Array]):
    """Symmetric positive definite ``(p, p)`` matrices.

    ``metric`` is ``"affine_invariant"`` or ``"log_euclidean"``. The
    retraction is the exponential of the metric and transport is its
    parallel transport.
    """

    def __init__(self, p: int, metric: SpdMetric = "affine_invariant") -> None:
        super().__init__(RiemannianManifold.spd(p, metric))
        self.p = p
        self.metric = metric


class Product(Manifold[tuple[Array, ...]]):
    """Product of :class:`Stiefel` and :class:`SPD` factors.

    Points, tangents and gradients are tuples with one array per factor.
    The metric is the sum of the factor metrics. A nested product
    contributes its factors, so its point is one flat tuple.
    """

    def __init__(self, *factors: Stiefel | SPD | Product) -> None:
        super().__init__(RiemannianManifold.product([f._core for f in factors]))
        self.factors = factors

    def _parts(self, value: Any, name: str) -> Sequence[Any]:
        if not isinstance(value, (tuple, list)) or len(value) != len(self._shapes):
            raise ValueError(f"{name}: need a sequence of {len(self._shapes)} arrays")
        return value

    def _unpack(self, flat: list[float]) -> tuple[Array, ...]:
        data = np.asarray(flat, dtype=np.float64)
        parts = []
        start = 0
        for rows, cols in self._shapes:
            stop = start + rows * cols
            parts.append(data[start:stop].reshape((rows, cols), order="F"))
            start = stop
        return tuple(parts)
