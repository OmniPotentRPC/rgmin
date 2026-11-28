Performance
===========

Generated from ``bench/raw``. Do not edit the table by hand.
``docs/render_performance.py`` rewrites this page.

The eOn L-BFGS arm and ``rgmin-eon`` spend the same evaluations
on Rosenbrock-2. The scaled arm is ASE's length control. On
Rosenbrock-10 it stops in the well near ``x0 = -1``; the cap does not.
That comparison is ``tests/scaled_rosenbrock10.rs``.

rgmin
-----

.. list-table::
   :header-rows: 1

   * - Problem
     - Arm
     - Evaluations
     - Gradient
   * - Rosenbrock2
     - rgmin-eon
     - 66
     - 4.95110615e-6
   * - Rosenbrock2
     - rgmin-eon-m20
     - 66
     - 3.22409888e-7
   * - Rosenbrock2
     - rgmin-scaled
     - 45
     - 1.27150291e-6
   * - Rosenbrock2
     - rgmin-wolfe
     - 176
     - 3.05191422e-6
   * - Rosenbrock2
     - rgmin-geometry
     - 101
     - 3.06789596e-6
   * - Rosenbrock2
     - rgmin-fire
     - 2714
     - 9.75280359e-6
   * - Rosenbrock2
     - rgmin-fire2
     - 2949
     - 9.87574091e-6
   * - Rosenbrock2
     - rgmin-cg
     - 87
     - 4.34764813e-6
   * - Rosenbrock10
     - rgmin-eon
     - 127
     - 8.55979902e-6
   * - Rosenbrock10
     - rgmin-eon-m20
     - 141
     - 3.66336737e-6
   * - Rosenbrock10
     - rgmin-scaled
     - 80
     - 8.61569782e-6
   * - Rosenbrock10
     - rgmin-wolfe
     - 310
     - 2.09115451e-6
   * - Rosenbrock10
     - rgmin-geometry
     - 172
     - 9.38133666e-6
   * - Rosenbrock10
     - rgmin-fire
     - 4183
     - 9.82543125e-6
   * - Rosenbrock10
     - rgmin-fire2
     - 4648
     - 9.32665962e-6
   * - Rosenbrock10
     - rgmin-cg
     - 883
     - 7.91021024e-6
   * - LJ7
     - rgmin-eon
     - 33
     - 4.78058700e-3
   * - LJ7
     - rgmin-eon-m20
     - 33
     - 6.38667112e-3
   * - LJ7
     - rgmin-scaled
     - 37
     - 9.85068118e-3
   * - LJ7
     - rgmin-wolfe
     - 116
     - 4.14548308e-3
   * - LJ7
     - rgmin-geometry
     - 29
     - 4.25146209e-3
   * - LJ7
     - rgmin-fire
     - 82
     - 8.33115971e-3
   * - LJ7
     - rgmin-fire2
     - 67
     - 9.30074397e-3
   * - LJ7
     - rgmin-cg
     - 37
     - 7.27390259e-3

eOn
---

.. list-table::
   :header-rows: 1

   * - Problem
     - Arm
     - Evaluations
     - Gradient
   * - Rosenbrock2
     - eon-lbfgs
     - 66
     - 3.224099e-07
   * - Rosenbrock10
     - eon-lbfgs
     - 149
     - 8.500893e-06
   * - Rosenbrock2
     - eon-fire
     - 2714
     - 9.752804e-06
   * - Rosenbrock10
     - eon-fire
     - 4183
     - 9.825431e-06
   * - Rosenbrock2
     - eon-cg
     - 87
     - 4.347648e-06
   * - Rosenbrock10
     - eon-cg
     - 883
     - 8.099467e-06
   * - LJ7
     - eon-lbfgs
     - 33
     - 6.386671e-03
   * - LJ7
     - eon-fire
     - 82
     - 8.331160e-03
   * - LJ7
     - eon-cg
     - 37
     - 7.273903e-03

ASE
---

.. list-table::
   :header-rows: 1

   * - Problem
     - Arm
     - Evaluations
     - Gradient
   * - LJ7
     - ase-lbfgs
     - 36
     - 4.49358312e-03
   * - LJ7
     - ase-fire
     - 78
     - 9.97915265e-03
   * - LJ7
     - ase-fire2
     - 47
     - 9.48931151e-03

SciPy
-----

.. list-table::
   :header-rows: 1

   * - Problem
     - Arm
     - Evaluations
     - Gradient
   * - Rosenbrock2
     - scipy-lbfgsb
     - 45
     - 3.24596871e-08
   * - Rosenbrock2
     - scipy-cg
     - 78
     - 5.49201702e-06
   * - Rosenbrock10
     - scipy-lbfgsb
     - 92
     - 1.46903255e-06
   * - Rosenbrock10
     - scipy-cg
     - 489
     - 5.57752427e-06
   * - LJ7
     - scipy-lbfgsb
     - 39
     - 6.98507624e-03

