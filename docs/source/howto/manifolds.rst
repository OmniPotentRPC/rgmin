

Retract onto an embedded manifold
---------------------------------

The default geometry is ambient Euclidean space. A session can
retract each trial point onto an embedded Riemannian manifold
through ``Solver::set_manifold`` / ``rgmin_solver_set_manifold``.

The contract is manopt\ :sub:`cpp`\'s ``AbstractManifold`` on a rank-1 f64
vector: ``project``, ``retract``, ``transport``. Citations:

- Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix Manifolds*
  (`10.1515/9781400830244 <https://doi.org/10.1515/9781400830244>`_).

- Boumal, *An Introduction to Optimization on Smooth Manifolds*
  (`10.1017/9781009166164 <https://doi.org/10.1017/9781009166164>`_).

- Boumal, Mishra, Absil, Sepulchre, Manopt
  (`10.5555/2627435.2638581 <https://doi.org/10.5555/2627435.2638581>`_).

- Huang, Absil, Gallivan, Hand, ROPTLIB
  (`10.1145/3218822 <https://doi.org/10.1145/3218822>`_).

Tokens
~~~~~~

.. table::

    +------------------------+---------------------------------------------+-------------------------------------------+
    | Token                  | Packing                                     | Retraction                                |
    +========================+=============================================+===========================================+
    | ``Euclidean``          | length ``n``                                | ``x + v``                                 |
    +------------------------+---------------------------------------------+-------------------------------------------+
    | ``RigidQuotient``      | 3N Cartesians, N >= 2                       | horizontal lift ``x + v``                 |
    +------------------------+---------------------------------------------+-------------------------------------------+
    | ``MwRigid``            | same; masses on the session                 | same; Eckart inner product                |
    +------------------------+---------------------------------------------+-------------------------------------------+
    | ``Sphere``             | unit vector, length ``n``                   | ``(x+v)/norm(x+v)``                       |
    +------------------------+---------------------------------------------+-------------------------------------------+
    | ``So3``                | row-major ``R``, length 9                   | QR with positive diagonal                 |
    +------------------------+---------------------------------------------+-------------------------------------------+
    | ``Stiefel``            | ``St(n,1)``: same as the sphere             | same as the sphere                        |
    +------------------------+---------------------------------------------+-------------------------------------------+
    | ``Se3``                | row-major ``R`` then ``t``, length 12       | SO(3) on the rotation, Euclidean on ``t`` |
    +------------------------+---------------------------------------------+-------------------------------------------+
    | ``Spd``                | row-major ``n x n`` SPD, length ``n^2``     | ``symm(X + U + (1/2) U X^{-1} U)``        |
    +------------------------+---------------------------------------------+-------------------------------------------+
    | ``Grassmann { n, p }`` | column-major ``n x p``, length ``n p``      | polar factor of ``X + U``                 |
    +------------------------+---------------------------------------------+-------------------------------------------+
    | ``Hyperbolic``         | Minkowski vector, length ``n`` (``n >= 2``) | hyperboloid exp (``cosh`` / ``sinh``)     |
    +------------------------+---------------------------------------------+-------------------------------------------+

An isolated molecule or cluster lives on ``RigidQuotient``
(``R^{3N}/SE(3)``): Sella Cartesian ``fix_translation`` plus
``fix_rotation`` (Hermes, Sarsfield, Zádor, JCTC 2022,
https://doi.org/10.1021/acs.jctc.2c00395). ``MwRigid`` is the same quotient
with the Page–McIver mass-weighted metric used by Sella IRC and
gpr\ :sub:`optim`\ ``IRCDriver`` (https://doi.org/10.1063/1.454172,
https://doi.org/10.1063/1.434152). Call ``set_masses`` with N atomic masses;
unit mass makes ``MwRigid`` identical to ``RigidQuotient``.

``Sphere``, ``So3``, ``Stiefel``, ``Se3``, ``Spd``, ``Grassmann``, and
``Hyperbolic`` are matrix-manifold embeddings. ``So3`` rejects any
length other than 9. ``Se3`` rejects any length other than 12. ``Spd``
rejects any length that is not a positive perfect square.
``Grassmann { n, p }`` rejects any length other than ``n p``.
``Hyperbolic`` is the Lorentz hyperboloid (manopt
``hyperbolicfactory``, :math:`m = 1`): length :math:`n` (:math:`n \ge 2`),
Minkowski square :math:`-1`. They do not pack or prefix-interpret a
3N cluster.

Euclidean is the default. Existing eOn / rgpot / eindir paths do
not change until a host calls the setter.

Rust
~~~~

.. code:: rust

    use ndarray::array;
    use rgmin::{Control, ManifoldKind, Method, Solver};

    let n = (3.0_f64).sqrt();
    let mut x = array![1.0 / n, 1.0 / n, 1.0 / n];
    let mut solver = Solver::new(Method::Steepest, Control::default(), 3);
    solver.set_manifold(ManifoldKind::Sphere);
    let _ = solver.step(&obj, &mut x).unwrap();

First-order steps retract the taken increment. Quasi-Newton
directions are projected into the tangent, then retracted by the
accept rule.

C
~

.. code:: c

    rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_RIGID_QUOTIENT);
    rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_MW_RIGID);
    rgmin_solver_set_masses(s, masses, n_atoms);
    rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_SPHERE);
    rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_SO3);
    rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_STIEFEL);
    rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_SE3);
    rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_SPD);
    rgmin_solver_set_grassmann(s, 5, 2);
    rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_HYPERBOLIC);
    rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_EUCLIDEAN);

Changing the manifold drops method memory (``forget``).

Packing notes
~~~~~~~~~~~~~

- ``RigidQuotient`` is 3N Cartesians. Isolated: kernel of three
  translations and three infinitesimal rotations
  (``R^{3N}/SE(3)``). Periodic: Sella ``proj_rot = false``, so only
  translations (``R^{3N}/T(3)``) via ``rgmin_solver_set_periodic``.
  TRICs are a different chart, not a different packing here.

- ``MwRigid`` is the same 3N packing. Masses are per atom (length
  N). The IRC path itself is steepest descent in that metric,
  not a separate token.

- ``So3`` is a 9-vector, row-major. The tangent projection returns
  the embedded vector ``R Omega``, not the skew factor alone.

- ``Se3`` is twelve numbers: the same 9-vector, then a translation.
  It is one rigid body, not N atoms.

- ``Stiefel`` is ``St(n,1)``. A frame with ``p > 1`` is not a length
  token: ``n p`` does not name ``p``.

- ``Spd`` is manopt ``sympositivedefinitefactory``: the affine-invariant
  SPD cone (Bhatia bi-invariant metric). The tangent is the
  symmetric matrices. Length must be a square; a 3N cluster is
  rejected.

- ``Grassmann { n, p }`` is manopt ``grassmannfactory``. The point is
  the column space of an orthonormal ``n x p`` frame, packed
  column-major (``X(:)``). Projection is the horizontal lift
  ``U - X (X^T U)``. Retraction is the polar factor of ``X + U``.
  A 3N cluster is not this packing.

- ``Hyperbolic`` is a length-:math:`n` Minkowski vector
  (:math:`-x_0^2 + ||x_{\mathrm{sp}}||^2 = -1`, :math:`n \ge 2`). Pack with
  ``pack_h`` / ``unpack_h`` (time-like then spatial). It is not the
  sphere and not a 3N cluster.

- ``set_project_rigid`` is the same horizontal projection as
  ``RigidQuotient`` and stays available on Euclidean.
