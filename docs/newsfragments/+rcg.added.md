Added Riemannian conjugate gradients as `minimize_rcg` and `Method::Rcg` (token `rcg`).
The retraction, tangent projection and vector transport are the ones the manifold, the level-set hook and the holonomic bond set already provide.
The coefficient is Fletcher-Reeves, PRP+ or Hager-Zhang.
The transport is scaled when it lengthens a vector, or the identity in the Euclidean case.
A direction that loses descent is replaced by the negative Riemannian gradient.
In a session the method honours `maxmove` and the per-atom cap, composes a distance constraint set with the configured manifold, takes its gradient, tangent projection and transport from the session, including rigid quotients, products with rigid blocks and periodic sessions, keeps every trial in the objective's box, and clears its direction when a level-set retraction moves the point.
A distance constraint set with masses steps along the cut gradient raised by the inverse masses, and its stop and report read the cut gradient.
A start with no retraction in the box, or one that putting on its bonds moves out of the box, returns `Error::OffManifold` before the objective reads a point.
Each Wolfe slope is the Euclidean gradient against the velocity of the retraction curve, the derivative of the value along it, so the strong Wolfe tests hold on the Poincare ball and the hyperboloid as well.
A step that stalls on the energy noise floor keeps its start and reports `NoiseFloor`.
`RcgOptions::validate` refuses a non-finite or negative tolerance and a non-finite or non-positive initial step with `Error::InvalidControl`, and a projection or retraction that fails, or a retraction slope that cannot be resolved, is returned as a typed error.
The Python session accepts `rcg`, and `rgmin.optimize.RiemannianCG` runs it from ASE with the default per-atom cap off unless `maxstep` is set.
`Manifold::to_tangent` projects a vector that is not a gradient, such as the velocity of a curve, and is the identity on the Poincare ball, whose `project` converts a gradient.
The C ABI adds `RGMIN_RCG`, `rgmin_rcg_options_t`, `rgmin_rcg_options_default` and `rgmin_minimize_rcg` at minor 58 with the layout revision left at 6, `xts.h` aliases them as `XTS_RCG`, `xts_rcg_options_t`, `xts_rcg_options_default` and `xts_minimize_rcg`, and the C++ and nanobind method enums list `Rcg`.
