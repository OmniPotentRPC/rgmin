

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

    +-------------------+---------------------------------------+-------------------------------------------+
    | Token             | Packing                               | Retraction                                |
    +===================+=======================================+===========================================+
    | ``Euclidean``     | length ``n``                          | ``x + v``                                 |
    +-------------------+---------------------------------------+-------------------------------------------+
    | ``RigidQuotient`` | 3N Cartesians, N >= 2                 | horizontal lift ``x + v``                 |
    +-------------------+---------------------------------------+-------------------------------------------+
    | ``MwRigid``       | same; masses on the session           | same; Eckart inner product                |
    +-------------------+---------------------------------------+-------------------------------------------+
    | ``Sphere``        | unit vector, length ``n``             | ``(x+v)/norm(x+v)``                       |
    +-------------------+---------------------------------------+-------------------------------------------+
    | ``So3``           | row-major ``R``, length 9             | QR with positive diagonal                 |
    +-------------------+---------------------------------------+-------------------------------------------+
    | ``Stiefel``       | ``St(n,1)``: same as the sphere       | same as the sphere                        |
    +-------------------+---------------------------------------+-------------------------------------------+
    | ``Se3``           | row-major ``R`` then ``t``, length 12 | SO(3) on the rotation, Euclidean on ``t`` |
    +-------------------+---------------------------------------+-------------------------------------------+
    | ``ComplexCircle`` | interleaved ``(re,im)``, length 2n   | ``sign(z+v)`` per pair                    |
    +-------------------+---------------------------------------+-------------------------------------------+
    | ``Euc. complex``  | interleaved ``(re,im)``, length 2n   | ``x + v``                                 |
    +-------------------+---------------------------------------+-------------------------------------------+

An isolated molecule or cluster lives on ``RigidQuotient``
(``R^{3N}/SE(3)``): Sella Cartesian ``fix_translation`` plus
``fix_rotation`` (Hermes, Sarsfield, Zádor, JCTC 2022,
https://doi.org/10.1021/acs.jctc.2c00395). ``MwRigid`` is the same quotient
with the Page–McIver mass-weighted metric used by Sella IRC and
gpr\ :sub:`optim`\ ``IRCDriver`` (https://doi.org/10.1063/1.454172,
https://doi.org/10.1063/1.434152). Call ``set_masses`` with N atomic masses;
unit mass makes ``MwRigid`` identical to ``RigidQuotient``.

``Sphere``, ``So3``, ``Stiefel``, ``Se3``, ``ComplexCircle``, and
``EuclideanComplex`` are matrix-manifold embeddings. ``So3``
rejects any length other than 9. ``Se3`` rejects any length other
than 12. ``ComplexCircle { n }`` and ``EuclideanComplex { n }``
reject any length other than ``2 n``. They do not pack or
prefix-interpret a 3N cluster.

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
    rgmin_solver_set_complex_circle(s, 4);
    rgmin_solver_set_manifold(s, RGMIN_MANIFOLD_SYMMETRIC);
    rgmin_solver_set_euclidean_complex(s, 4);
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

- ``ComplexCircle { n }`` is manopt ``complexcirclefactory(n)``:
  interleaved ``(re, im)`` pairs, length ``2 n``. Each pair is
  independently unit-modulus. It is not the sphere :math:`S^{2n-1}`
  and not a 3N cluster.

- ``EuclideanComplex { n }`` is manopt ``euclideancomplexfactory``:
  interleaved ``(re, im)`` pairs, length ``2 n``. Projection and
  transport are the identity. Retraction is ``x + v``. It is not
  the sphere, not ``ComplexCircle``, and not a 3N cluster.

- ``Symmetric`` is manopt ``symmetricfactory``: an ``n x n`` real
  symmetric matrix packed row-major (``n^2``). Projection is
  symmetrization. Retraction is ``X + U``. Transport is the
  identity. It is not the SPD cone and not a 3N cluster.

- ``set_project_rigid`` is the same horizontal projection as
  ``RigidQuotient`` and stays available on Euclidean.
