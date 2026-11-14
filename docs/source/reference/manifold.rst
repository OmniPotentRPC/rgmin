Riemannian manifolds
--------------------

``RiemannianManifold`` is a Stiefel manifold ``St(n, p)``, a cone ``SPD(p)`` of symmetric positive definite matrices, or a product of them.
Rust, C, C++ and Python call the same code.
:doc:`../tutorials/manifold-fit` fits a point on ``St x SPD`` by gradient steps.

Packing
~~~~~~~

Every buffer is caller-owned, column-major and ``len`` doubles long.

.. table::

    +--------------+-----------------------------------------------------+-------------+
    | Factor       | Point                                               | ``len``     |
    +==============+=====================================================+=============+
    | ``St(n, p)`` | ``n x p`` frame, ``U^T U = I``, ``1 <= p <= n``     | ``n p``     |
    +--------------+-----------------------------------------------------+-------------+
    | ``SPD(p)``   | ``p x p`` symmetric positive definite               | ``p^2``     |
    +--------------+-----------------------------------------------------+-------------+
    | Product      | factor blocks in factor order                       | the sum     |
    +--------------+-----------------------------------------------------+-------------+

Tangent vectors and Euclidean gradients use the packing of the point.
A nested product contributes its factors.

Operations
~~~~~~~~~~

Stiefel uses the embedded metric.

``project``
    Stiefel: ``V - X sym(X^T V)``. SPD: ``sym(V)``.
``egrad_to_rgrad``
    Stiefel: the projection. SPD affine-invariant: ``X sym(G) X``. SPD log-Euclidean: ``Dlog(X)^{-2}[sym(G)]``.
``retract``
    Stiefel: the QR (first order) or polar (second order) factor of ``X + V``. SPD: ``exp`` of the metric.
``exp``
    Stiefel: Edelman, Arias and Smith (1998) eq. 2.41. SPD affine-invariant: ``X^{1/2} expm(X^{-1/2} V X^{-1/2}) X^{1/2}``. SPD log-Euclidean: ``expm(log X + Dlog(X)[V])``.
``log``
    SPD: the inverse of ``exp``. Stiefel: refused.
``transport``
    Stiefel: projection at the arrival point. SPD affine-invariant: ``E V E^T`` with ``E = (Y X^{-1})^{1/2}``. SPD log-Euclidean: ``Dlog(Y)^{-1}[Dlog(X)[V]]``.
``inner``
    Stiefel: ``tr(U^T V)``. SPD affine-invariant: ``tr(X^{-1} U X^{-1} V)``. SPD log-Euclidean: ``<Dlog(X)[U], Dlog(X)[V]>``.

``sym(A) = (A + A^T) / 2``.
``Dlog(X)`` is applied in the eigenbasis of ``X`` with the divided differences of ``log`` on its eigenvalues (Daleckii-Krein).
``exp`` projects ``v`` to the tangent space first.
The product metric is the sum of the factor metrics.
``gradient_step(x, egrad, t)`` returns ``rgrad = egrad_to_rgrad(x, egrad)`` and ``retract(x, -t rgrad)``.

The affine-invariant metric is Pennec, Fillard and Ayache (`2006 <https://doi.org/10.1007/s11263-005-3222-z>`_).
The log-Euclidean metric is Arsigny, Fillard, Pennec and Ayache (`2007 <https://doi.org/10.1137/050637996>`_).
The Stiefel geodesic is Edelman, Arias and Smith (`1998 <https://doi.org/10.1137/S0895479895290954>`_).

Refusals
~~~~~~~~

A wrong length, a non-finite entry, an SPD point that is not symmetric positive definite, a rank-deficient ``X + V`` under a Stiefel retraction, a non-finite step size, and ``log`` on a Stiefel factor are refused.
Rust returns ``Error::ManifoldArgument``, C returns ``RGMIN_INVALID_PARAMETER`` with ``rgmin_last_error``, C++ throws ``std::runtime_error``, and Python raises ``ValueError``.
C writes the outputs only on success, so an output may alias an input.

Rust
~~~~

.. code:: rust

   use rgmin::{RiemannianManifold, SpdMetric, StiefelRetraction};

   let m = RiemannianManifold::product([
       RiemannianManifold::stiefel(6, 2, StiefelRetraction::Polar)?,
       RiemannianManifold::spd(3, SpdMetric::AffineInvariant)?,
   ])?;
   m.gradient_step(&x, &egrad, 0.1, &mut rgrad, &mut x_next)?;

``RiemannianManifold`` implements the ``Manifold`` trait, so generic code such as ``rtr::truncated_cg`` accepts it.
``Solver`` selects its manifold with the closed ``ManifoldKind`` and does not take this type.

C
~

``rgmin_manifold_t`` is the session token of ``rgmin_solver_set_manifold``.
The handle here is ``rgmin_manifold_handle_t``. ABI minor 41.

``rgmin_manifold_stiefel(n, p, retraction)``
    ``RGMIN_STIEFEL_QR`` or ``RGMIN_STIEFEL_POLAR``. Null on a bad shape.
``rgmin_manifold_spd(p, metric)``
    ``RGMIN_SPD_AFFINE_INVARIANT`` or ``RGMIN_SPD_LOG_EUCLIDEAN``.
``rgmin_manifold_product(factors, count)``
    Copies the factors; the caller still frees them.
``rgmin_manifold_free(m)``, ``rgmin_manifold_len(m)``
    Null is a no-op; the length of a null handle is zero.
``rgmin_manifold_project(m, x, v, out, len)``, ``rgmin_manifold_egrad_to_rgrad(m, x, egrad, out, len)``, ``rgmin_manifold_retract(m, x, v, out, len)``, ``rgmin_manifold_exp(m, x, v, out, len)``, ``rgmin_manifold_log(m, x, y, out, len)``
    One output of ``len`` doubles.
``rgmin_manifold_transport(m, x_from, x_to, v, out, len)``
    Transport to ``x_to``.
``rgmin_manifold_inner(m, x, u, v, &value, len)``
    One double.
``rgmin_manifold_gradient_step(m, x, egrad, t, rgrad_out, x_out, len)``
    Two outputs that must not alias each other.

C++
~~~

``rgmin::Manifold`` owns the handle: ``Manifold::stiefel(n, p, r)``, ``Manifold::spd(p, metric)`` and ``Manifold::product(a, b, ...)``.
The members take buffers of ``len()`` doubles. ``xts::optimize::Manifold`` is the same class.

Python
~~~~~~

``rgmin.Stiefel(n, p, retraction="qr")``, ``rgmin.SPD(p, metric="affine_invariant")`` and ``rgmin.Product(*factors)`` take and return NumPy arrays.
A Stiefel point is an ``(n, p)`` array, an SPD point a ``(p, p)`` array, and a product point a tuple with one array per factor.
Each class has ``project``, ``egrad_to_rgrad``, ``retract``, ``exp``, ``log``, ``transport``, ``inner``, ``norm`` and ``gradient_step``, which returns ``(rgrad, x_next)``.
