

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

    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | Token                      | Packing                                         | Retraction                                |
    +============================+=================================================+===========================================+
    | ``Euclidean``              | length ``n``                                    | ``x + v``                                 |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``RigidQuotient``          | 3N Cartesians, N >= 2                           | horizontal lift ``x + v``                 |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``MwRigid``                | same; masses on the session                     | same; Eckart inner product                |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``Sphere``                 | unit vector, length ``n``                       | ``(x+v)/norm(x+v)``                       |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``So3``                    | row-major ``R``, length 9                       | QR with positive diagonal                 |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``Stiefel``                | ``St(n,1)``: same as the sphere                 | same as the sphere                        |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``StiefelP { n, p }``      | column-major ``n x p``, length ``n p``          | thin QR of ``X + U``                      |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``Se3``                    | row-major ``R`` then ``t``, length 12           | SO(3) on the rotation, Euclidean on ``t`` |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``Oblique { n, m }``       | column-major ``n x m``, length ``n m``          | column-wise ``(x+v)/norm``                |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``Multinomial``            | simplex, length ``n >= 2``                      | ``x odot exp(v oslash x)``, renormalize   |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``ComplexCircle { n }``    | interleaved ``(re, im)``, length ``2 n``        | ``sign(z+v)`` per pair                    |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``Symmetric``              | row-major ``n x n``, length ``n^2``             | ``X + U`` then ``multisym``               |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``SkewSymmetric``          | row-major ``n x n``, length ``n^2``, ``n >= 2`` | ``X + U`` then ``multiskew``              |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``EuclideanComplex { n }`` | interleaved ``(re, im)``, length ``2 n``        | ``x + v``                                 |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``Constant { n }``         | singleton of length ``n``                       | the fixed point                           |
    +----------------------------+-------------------------------------------------+-------------------------------------------+
    | ``Positive { n }``         | positive orthant, length ``n``                  | ``x odot exp(v oslash x)``                |
    +----------------------------+-------------------------------------------------+-------------------------------------------+

An isolated molecule or cluster lives on ``RigidQuotient``
(``R^{3N}/SE(3)``): Sella Cartesian ``fix_translation`` plus
``fix_rotation`` (Hermes, Sarsfield, Zádor, JCTC 2022,
https://doi.org/10.1021/acs.jctc.2c00395). ``MwRigid`` is the same quotient
with the Page–McIver mass-weighted metric used by Sella IRC and
gpr\ :sub:`optim`\ ``IRCDriver`` (https://doi.org/10.1063/1.454172,
https://doi.org/10.1063/1.434152). Call ``set_masses`` with N atomic masses;
unit mass makes ``MwRigid`` identical to ``RigidQuotient``.

``Sphere``, ``So3``, ``Stiefel``, ``Se3``, and ``ComplexCircle`` are
matrix-manifold embeddings. ``So3`` rejects any length other than 9.
``Se3`` rejects any length other than 12. ``ComplexCircle { n }``
rejects any length other than ``2 n``. They do not pack or
prefix-interpret a 3N cluster.

Euclidean is the default. Existing eOn / rgpot / eindir paths do
not change until a host calls the setter.

Rust
~~~~

.. code:: rust

    use eindir_core::objectives::Rosenbrock;
    use ndarray::array;
    use rgmin::{Control, ManifoldKind, Method, Solver};

    let obj = Rosenbrock::<3>::new();
    let n = (3.0_f64).sqrt();
    let mut x = array![1.0 / n, 1.0 / n, 1.0 / n];
    let mut solver = Solver::new(Method::Steepest, Control::default(), 3);
    solver.set_manifold(ManifoldKind::Sphere);
    let _ = solver.step(&obj, &mut x).unwrap();
    assert!((x.dot(&x) - 1.0).abs() < 1e-12);

On a curved kind a step projects its increment onto the tangent space.
The step is capped there and then retracted.
The objective sees only retracted points.
A trial that the box would move is refused before any oracle call.
Here ``P`` is the projection of the kind onto the tangent space.
Newton-CG and Newton-PCG solve the model ``P H P + (I - P)``.
Dogleg solves the model ``Z^T H Z`` with the gradient ``Z^T g``, for an orthonormal basis ``Z`` of the tangent space, and maps its point back through ``Z``.
Its Newton and Cauchy points are then tangent, and the trust radius bounds the step that is retracted.
On the sphere dogleg writes the same model in ambient form, ``P^T H P + (I - P)^T (I - P)`` with ``P = I - x x^T``.
Where ``P`` is an orthogonal projection, the dogleg model equals ``P H P + (I - P)`` on the tangent space, so dogleg and Newton-CG/PCG solve one model.
That is so on every curved kind but the three multinomial kinds.
Their Fisher projection is oblique, and there the two models differ.
A ``fire2`` session applies ``P`` to its velocity.
That projection is at each new point.
A refused trial under ``Method::Fire`` returns ``Error::OffManifold``.
So does a refused L-BFGS trial under ``Accept::None``.
Under ``Accept::Energy`` and ``Accept::Nonmonotone``, L-BFGS on the sphere and on Stiefel searches on the set.
On the other curved kinds the L-BFGS line search runs in ambient space.
Its Wolfe probes evaluate ``x + a d`` off the set.
The step is retracted once, after the search.
Flat kinds keep the old order of retraction, cap and box clip.

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
    rgmin_solver_set_multinomial_ds(s, 3);
    rgmin_solver_set_multinomial_sym(s, 3);
    rgmin_solver_set_sphere_complex(s, 3);
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

- ``Symmetric`` is manopt ``symmetricfactory``: an ``n x n`` real
  symmetric matrix packed row-major (``n^2``). Projection is
  symmetrization. Retraction is ``X + U``. Transport is the
  identity. It is not the SPD cone and not a 3N cluster.

- ``MultinomialDoublyStochastic { n }`` is manopt
  ``multinomialdoublystochasticfactory``: an ``n x n`` positive
  doubly-stochastic matrix packed row-major (``n^2``, ``n >= 2``).
  Projection is Fisher-orthogonal onto ``V 1 = V^T 1 = 0``.
  Retraction is ``X ⊙ exp(V ⊘ X)`` then Sinkhorn. Transport is
  projection at arrival. It is not the simplex, not the sphere,
  and not a 3N cluster. C token 18
  (``rgmin_solver_set_multinomial_ds``). Tokens 7-10 stay reserved.

- ``MultinomialSymmetric { n }`` is manopt
  ``multinomialsymmetricfactory``: an ``n x n`` positive symmetric
  doubly-stochastic matrix packed row-major (``n^2``, ``n >= 2``).
  Projection is Fisher-orthogonal onto symmetric ``V 1 = 0``.
  Retraction is ``X ⊙ exp(V ⊘ X)`` then Sinkhorn then symmetrize.
  Transport is projection at arrival. It is not the simplex, not
  the unsigned Birkhoff polytope, not the sphere, and not a 3N
  cluster. C token 19
  (``rgmin_solver_set_multinomial_sym``). Tokens 7-10 stay reserved.

- ``SphereComplex { n }`` is manopt ``spherecomplexfactory``: the
  unit sphere in ``C^n``, interleaved ``(re, im)``, length ``2 n``.
  Projection is ``u - Re(x^* u) x``. Retraction is
  ``(x+v)/||x+v||``. Transport is projection at arrival. It is
  not the real sphere token, not ``(S^1)^n``, and not a 3N
  cluster. C token 20 (``rgmin_solver_set_sphere_complex``).
  Tokens 7-10 stay reserved.

- ``Positive { n }`` is manopt ``positivefactory``: the open
  positive orthant of packed length ``n``. Projection is the
  identity. Retraction is ``x ⊙ exp(v ⊘ x)``. Transport is
  the identity. It is not the sphere, not the simplex, and
  not a 3N cluster. C token 21 (``rgmin_solver_set_positive``).
  Tokens 7-10 stay reserved.

- ``CenteredMatrix { m, n, rows }`` is manopt
  ``centeredmatrixfactory``: an ``m x n`` matrix packed
  row-major (``m n``) with ``X 1 = 0`` (``rows = false``,
  default) or ``1^T X = 0`` (``rows = true``). Projection is the centering
  operator. Retraction is ``X + U`` then center. Transport is
  the identity. It is not the sphere and not a 3N cluster. C
  token 22 (``rgmin_solver_set_centered_matrix``). Tokens 7-10
  stay reserved.

- ``set_project_rigid`` is the same horizontal projection as
  ``RigidQuotient`` and stays available on Euclidean.
