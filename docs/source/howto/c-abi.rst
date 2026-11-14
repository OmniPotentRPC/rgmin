

C and C++ ABI
-------------

Solvers live in Rust. Geometry hosts that call one outer iteration at
a time hold an opaque ``rgmin_solver_t``; scripts that want a one-shot
solve still call ``rgmin_minimize``. Every vector is a dlpk
``DLManagedTensorVersioned`` tensor.

Install
~~~~~~~

.. code:: bash

    cargo cinstall --prefix=/usr/local --features capi

``cargo cinstall`` comes from cargo-c (:doc:`install`).
That installs the hand-written ``include/rgmin.h``, the C++ wrappers
under ``include/rgmin/`` and ``include/xts/``, and ``dlpack/dlpack.h``.
The pkg-config file is ``rgmin``. A C or C++ file includes ``rgmin.h``
and links with ``pkg-config --cflags --libs rgmin``. The crate version
macros are ``RGMIN_H_MAJOR``, ``RGMIN_H_MINOR``, and ``RGMIN_H_PATCH``.
HiGHS ``HighsInt`` is not a type in this header. The constrained
setters take ``int32_t`` and the closed ``rgmin_highs_*`` enums.

Pass ``--features capi``. The C symbols are compiled only with that
feature. ``cbindgen`` output is not installed.

Session (eOn / ASE ``step``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c

    #include "rgmin.h"

    rgmin_control_t ctrl = { .maxiter = 200, .gtol = 1e-8, .istep = 1.0, .memory = 8 };
    rgmin_solver_t *s = rgmin_solver_create(RGMIN_LBFGS, &ctrl, n);
    rgmin_report_t report;
    while (!done) {
        rgmin_solver_set_maxmove(s, max_move);
        rgmin_solver_set_qn_step(s, RGMIN_QN_NEWTON); /* or RGMIN_QN_LBFGS + P as H0 */
        rgmin_solver_set_accept(s, RGMIN_ACCEPT_NONE);
        rgmin_solver_step_hess_fg(s, evalgrad, hess, user, x, &report);
    }
    rgmin_solver_free(s);

``rgmin_solver_forget`` drops pairs / conjugacy / moments. Newton and RFO
use ``rgmin_solver_step_hess_fg`` (fused energy and gradient, one host
potential call). Split ``eval`` / ``grad`` callbacks remain. ``accept``
is ``none`` (take the clipped step), ``energy``, or ``nonmonotone``.
Callbacks are arguments of each step, not stored on the handle.

``rgmin_solver_set_exp`` installs the exponential preconditioner.
``mu`` NaN estimates the energy scale on the next step.
``rgmin_solver_exp_mu`` returns that scale, or NaN when none is
installed. L-BFGS uses the factor as ``H0``. FIRE and nonlinear CG
use it as the metric. ``rgmin_solver_clear_exp`` drops it.
``abi_minor`` 28.

A HiGHS feasible set is ``rgmin_solver_set_highs`` plus
``rgmin_solver_set_box`` (NULL side unbounded), optional
``rgmin_solver_set_trust``, and ``rgmin_solver_add_equality``. Each
returns 0, or 1 when the build has no ``highs`` feature. Stamp
1.22.4.

One-shot
~~~~~~~~

.. code:: c

    #include "rgmin.h"

    rgmin_control_t ctrl = { .maxiter = 200, .gtol = 1e-8, .istep = 1.0, .memory = 8 };
    rgmin_report_t report;
    rgmin_status_t st = rgmin_minimize(eval, grad, user, x, &ctrl, RGMIN_LBFGS, &report);

SCG is the same one-shot shape. Conjugacy is a field on
``rgmin_scg_params_t``, not a different entry point. Stamp is 1.13.3
(``xts_abi_compatible`` still checks major and layout only).
``rgmin_lowest_eigenpair`` takes a typed ``rgmin_eigen_params_t``
(closed ``rgmin_eigen_kind_t``, no string). Unlinked kinds return
``RGMIN_UNAVAILABLE``. ``rgmin_lowest_eigenpairs`` is the block LOBPCG
solver for the k lowest pairs (Knyazev 2001). ``seed`` and ``modes_out``
are row-major n-by-k. ``rgmin_lowest_eigenpairs`` is ``abi_minor`` 39. The same call is
``rgmin::lowest_eigenpairs`` and ``xts_lowest_eigenpairs``.
``rgmin_manifold_handle_t`` is a Stiefel, SPD or product manifold with
column-major point buffers, ``abi_minor`` 41; see :doc:`../reference/manifold`.
``rgmin_lowest_eigenpairs_ex`` adds each residual and a converged flag, and an
optional preconditioner (null is the identity). It is ``abi_minor`` 42.
``rgmin_eigenpair_t`` is unchanged.

.. code:: c

    rgmin_eigenpair_t pairs[2];
    double values[2];
    rgmin_lowest_eigenpairs(hvp, user, x, seed, modes, values, NULL, NULL, pairs, 2);

.. code:: c

    rgmin_scg_params_t params = {
        .sigma0 = 1e-4, .lambda = 1.0, .lambda_limit = 1e60,
        .tol_sol = 1e-6, .tol_func = 1e-8,
        .conjugacy = RGMIN_CONJUGACY_LIU_STOREY, /* gpr_optim RgminScg.inl */
    };
    rgmin_minimize_scg(eval, grad, curv, user, x, &ctrl, &params, &report);
    /* params == NULL selects Netlab Polak-Ribiere + ScgParams::default */

A non-CPU tensor returns ``RGMIN_UNSUPPORTED_DEVICE``. The ABI does not
change when a CUDA path lands.

L-BFGS direction with a caller ``H0``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A host that keeps its own step and wants only the two-loop direction
holds an ``rgmin_lbfgs_pairs_t``. The handle stores ``(s, y)`` pairs, oldest
first, and never evaluates an objective.

.. code:: c

    rgmin_lbfgs_pairs_t *pairs = rgmin_lbfgs_pairs_create(n, 10);
    int32_t stored = 0;
    rgmin_lbfgs_pairs_push(pairs, s, y, n, &stored);          /* rho = 1 / (s . y) */
    rgmin_lbfgs_pairs_push_rho(pairs, s, y, rho, n, &stored); /* the caller's rho */
    rgmin_lbfgs_direction(pairs, grad, h0_diag, NULL, NULL, dir, n); /* dir = -H g */
    rgmin_lbfgs_direction(pairs, grad, NULL, apply_h0, user, dir, n);
    rgmin_lbfgs_pairs_free(pairs);

Pass the diagonal of ``H0`` or a callback, not both and not neither. A
diagonal whose entries are all ``gamma`` is ``H0 = gamma I``. An empty
memory writes ``-H0 g``. ``apply_h0(user, v, out, n)`` writes ``H0 v`` and
returns 0; any other value fails the call with
``RGMIN_INVALID_PARAMETER``. It runs once per direction, on the residual
of the backward loop that starts from ``grad``, so the direction is
``-H g`` when the map is linear. It must not call back into the handle.

``rgmin_lbfgs_pairs_push`` declines a pair whose ``s . y`` is not finite or
is at most ``1e-8 ||s|| ||y||``, the test ``Lbfgs::record`` applies. It
returns ``RGMIN_SUCCESS`` with ``*stored`` 0. ``rgmin_lbfgs_pairs_push_rho``
applies the same test and also declines a ``rho`` that is not finite and
positive, so a pair with non-positive curvature never enters the
two-loop. A length other than the handle's ``n`` is
``RGMIN_INVALID_PARAMETER``, and ``rgmin_last_error`` names it.
``capacity`` 0 keeps ten pairs; a full memory drops the oldest.

The direction is the Rust ``LbfgsPairs::direction`` and runs the
recursion of the session step. The pairs a capped L-BFGS session holds,
with a scalar ``H0`` equal to the session's, give that session's eOn step
bit for bit. ``rgmin_lbfgs_pairs_push`` and ``rgmin_lbfgs_direction`` are
``abi_minor`` 38; ``rgmin_lbfgs_pairs_push_rho`` is ``abi_minor`` 43, so check
``rgmin_abi_stamp().abi_minor >= 43`` before calling it. C++ wraps the
handle in ``rgmin::LbfgsPairs`` (``push``, ``push_rho``, ``direction_diagonal``,
``direction_apply``), ``xts.h`` aliases each entry as ``xts_lbfgs_*``, and
Python takes ``rgmin.LbfgsPairs.push(s, y, rho=None)``.

C++
~~~

.. code:: cpp

    #include <rgmin/optimize.hpp>

    rgmin::Control ctrl;
    auto report = rgmin::minimize_fn(eval, grad, user, x, ctrl,
                                     rgmin::Method::Lbfgs);

``<xts/optimize.hpp>`` keeps the same names under ``xts::optimize`` for old hosts.
xtensor callers include ``xts/xtensor.hpp``, which adapts ``xt::xarray`` to
the same ``rgmin_minimize`` entry. There is no solver in the header.

Manifold
~~~~~~~~

.. code:: c

    rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_SPHERE);

Tokens: ``RGMIN_MANIFOLD_EUCLIDEAN`` (default), ``RIGID_QUOTIENT``
(``R^{3N}/SE(3)``, Sella Cartesian), ``MW_RIGID`` (Page–McIver /
Sella IRC Eckart), ``SPHERE``, ``SO3`` (length 9), ``STIEFEL``
(``St(n,1)``), ``SE3`` (length 12), ``SYMMETRIC`` (n² row-major),
``MULTINOMIAL_DS`` (token 18, n² doubly-stochastic;
``rgmin_solver_set_multinomial_ds``), ``MULTINOMIAL_SYM`` (token 19,
n² symmetric doubly-stochastic; ``rgmin_solver_set_multinomial_sym``),
``SPHERE_COMPLEX`` (token 20, interleaved ``(re, im)``, length ``2 n``;
``rgmin_solver_set_sphere_complex``), ``POSITIVE`` (token 21, packed
length ``n``; ``rgmin_solver_set_positive``), ``CENTERED_MATRIX``
(token 22, packed ``m n`` row-major;
``rgmin_solver_set_centered_matrix``).
Tokens 7-10 stay reserved. See
:doc:`retract onto an embedded manifold <manifolds>`.
``rgmin_solver_set_masses`` supplies N atomic masses for ``MW_RIGID``.
