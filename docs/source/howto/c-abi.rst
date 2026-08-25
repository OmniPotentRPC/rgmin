

C and C++ ABI
-------------

Solvers live in Rust. Geometry hosts that call one outer iteration at
a time hold an opaque ``rgmin_solver_t``; scripts that want a one-shot
solve still call ``rgmin_minimize``. Every vector is a dlpk
``DLManagedTensorVersioned`` tensor.

Session (eOn / ASE ``step``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c

    #include "xts.h"

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

One-shot
~~~~~~~~

.. code:: c

    #include "xts.h"

    rgmin_control_t ctrl = { .maxiter = 200, .gtol = 1e-8, .istep = 1.0, .memory = 8 };
    rgmin_report_t report;
    rgmin_status_t st = rgmin_minimize(eval, grad, user, x, &ctrl, RGMIN_LBFGS, &report);

SCG is the same one-shot shape. Conjugacy is a field on
``rgmin_scg_params_t``, not a different entry point. Stamp is 1.13.3
(``xts_abi_compatible`` still checks major and layout only).
``rgmin_lowest_eigenpair`` takes a typed ``rgmin_eigen_params_t``
(closed ``rgmin_eigen_kind_t``, no string). Unlinked kinds return
``RGMIN_UNAVAILABLE``.

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

C++
~~~

.. code:: cpp

    #include <xts/optimize.hpp>

    xts::optimize::Control ctrl;
    auto report = xts::optimize::minimize(eval, grad, user, x, ctrl,
                                          xts::optimize::Method::Lbfgs);

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
length ``n``; ``rgmin_solver_set_positive``).
Tokens 7-10 stay reserved. See
`retract onto an embedded manifold <manifolds.rst>`_.
``rgmin_solver_set_masses`` supplies N atomic masses for ``MW_RIGID``.
