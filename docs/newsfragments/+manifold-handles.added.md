`RiemannianManifold` is a Stiefel manifold `St(n, p)` with a QR or polar retraction, an SPD cone `SPD(p)` under the affine-invariant or log-Euclidean metric, or a product of them.
Each has projection, Riemannian gradient, retraction, exponential, SPD logarithm, transport, inner product and one gradient step on caller-owned column-major buffers.
C: `rgmin_manifold_handle_t` and the `rgmin_manifold_*` functions, ABI minor 41.
The C++ wrapper is `rgmin::Manifold`, and `xts_manifold_*` aliases the C entries.
Python: `rgmin.Stiefel`, `rgmin.SPD` and `rgmin.Product` on NumPy arrays.
`examples/tutorial_manifold_fit.py` is run in CI.
