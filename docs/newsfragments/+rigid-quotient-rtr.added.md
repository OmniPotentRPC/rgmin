Added `RiemannianManifold::rigid_quotient` for the quotient `R^{3N}/SE(3)`, and `R^{3N}/T(3)` when the cell is periodic. Projection removes translations, and rotations as well on the isolated quotient. The retraction is the horizontal lift `x + v`.

LOBPCG now honors the same tangent projector Lanczos already applies. `lowest_eigenpairs_projected` returns the block pairs on that subspace, matching a dense eigenproblem with the rigid modes removed.

Exposed the quotient, one Steihaug-Toint truncated CG step (`rgmin_rtr_tcg_solve`), the Absil-Baker-Gallivan radius update (`rgmin_rtr_radius_update`), and the projected eigenpair entries (`rgmin_lowest_eigenpair_on_manifold`, `rgmin_lowest_eigenpairs_projected`) at ABI minor 56. `rgmin_control_t` is unchanged.
