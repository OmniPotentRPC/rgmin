========================================================
Randomness only in the clock: the Nystrom preconditioner
========================================================


.. contents::

The crate's one use of randomized numerical linear algebra is the
Nystrom preconditioner for the matrix-free Newton path. The design
rule it obeys is worth stating before the math: randomness may change
how fast an answer arrives, never which answer arrives. CG under any
symmetric positive definite preconditioner converges to the same
step; the preconditioner is a unit system for the iteration, not an
opinion about the solution. This page derives the sketch and pins
where each half of that claim is proved.

1 Why a preconditioner pays here
--------------------------------

CG's iteration count grows with the spread of the Hessian's spectrum.
Cluster and kernel Hessians typically carry a few stiff modes over a
soft bulk -- bond-stretch scales over collective, floppy ones. A
preconditioner that captures the stiff subspace and rescales it down
to the bulk flattens the effective spectrum, and the iteration count
drops to that of the easy part. The stiff subspace is low-rank, which
is exactly what a randomized sketch finds cheaply.

2 The sketch (Frangella, Tropp, Udell)
--------------------------------------

Draw a random test block Omega (n x k; the crate uses Rademacher
entries, +-1, drawn from a seeded generator so a rebuild at the same
point is reproducible) and pay ``k`` Hessian actions for Y = H Omega.
With the stabilizing shift nu, the Nystrom approximation is

H\ :sub:`nys`\ = Y\ :sub:`nu`\ (Omega\ :sup:`T`\ Y\ :sub:`nu`\)\ :sup:`-1`\ Y\ :sub:`nu`\ \ :sup:`T`\,   Y\ :sub:`nu`\ = Y + nu Omega,

a rank-k surrogate that agrees with ``H`` on the sketched subspace.
Factoring the small k x k core by a Jacobi eigensolve and thin-
factoring through the Gram matrix B\ :sup:`T`\ B yields H\ :sub:`nys`\ = U L U\ :sup:`T`\ with
orthonormal ``U`` and eigenvalue estimates ``L``. The preconditioner then
equalizes the captured modes down to the smallest kept eigenvalue mu:

P\ :sup:`-1`\ r = r + U ( diag(mu / (l\ :sub:`i`\ + mu)) - I ) U\ :sup:`T`\ r,

which damps each captured stiff mode by mu / (l\ :sub:`i`\ + mu) and passes the
orthogonal complement through untouched.

Two Lean facts in
`Spectral.lean <https://github.com/OmniPotentRPC/rgmin/blob/develop/proofs/lean/Rgmin/Spectral.lean>`_ pin the
construction's legitimacy:

- ``sumSq_nonneg`` (with ``sq_nonneg``): a Gram quadratic form is a sum of
  squares, so B B\ :sup:`T`\ cannot manufacture negative curvature; the sketch
  is positive semidefinite by construction, not by luck.

- ``shifted_weight_pos`` / ``equalized_weight_bounds``: with nonnegative
  kept eigenvalues and a positive shift, every weight the solve divides
  by is positive and every damping factor sits in (0, 1] -- no zero
  division, no sign flip, no amplification.

3 Exactness: the scale-invariance argument
------------------------------------------

The claim that CG's answer ignores the preconditioner has a rational
core one can state without any linear algebra: scaling M -> c M
(c > 0) scales z = M\ :sup:`-1`\ r by 1/c, and the iteration's outputs are
built from ratios in which the scale cancels. Concretely
(``Precond.lean``):

- ``step_scale_invariant``: alpha d = (rz / dHd) d is unchanged when
  rz -> rz/c, dHd -> dHd/c\ :sup:`2`\, d -> d/c.

- ``beta_scale_invariant``: beta = rz\ :sub:`next`\ / rz is a ratio of two
  quantities scaling identically.

- ``metric_update_scales``: the tracked M-norm products scale linearly,
  so the boundary comparison against a radius in the same metric is
  the same comparison.

The Rust test ``nystrom_flattens_a_decaying_spectrum`` then closes the
loop end to end: on a spectrum l\ :sub:`i`\ = 1e4 / i\ :sup:`2`\ it requires the
preconditioned step to equal the plain step (exactness) and the
sketch's ``k`` actions plus the preconditioned solve to spend fewer
total actions than the plain solve (the clock actually sped up).

4 Where randomization is refused
--------------------------------

The same discipline says where not to randomize. The ``par`` feature's
host threading was measured 2.6x slower than serial at n = 20000 and
now sits behind a length floor; stochastic-dynamics noise inside an
acceptance chain is priced exponentially by the consumer's own
theorems (anneal's ``shave_convicts``); and any preconditioner that is
not positive definite forfeits its metric, which ``steihaug_pcg``
detects (rz <= 0) and answers with the safe steepest boundary step
rather than trusting a broken clock.
