Lowest eigenpairs
=================

``lowest_eigenpairs`` is block LOBPCG for the k lowest pairs of a
symmetric product (Knyazev 2001). The caller supplies ``H(x) v``.
The seed block has shape ``(n, k)``. Values come back ascending, and
the vectors are the columns of an ``(n, k)`` array. A pair that misses
the tolerance raises ``RuntimeError``.

The program is ``examples/tutorial_lowest_eigenpairs.py``. Continuous
integration runs it.

.. code:: python

   from rgmin import lowest_eigenpairs

   values, vectors = lowest_eigenpairs(hvp, seed, tol=1e-8, max_iter=20)
