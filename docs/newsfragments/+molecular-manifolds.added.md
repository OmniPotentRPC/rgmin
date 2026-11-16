Molecular manifolds: `rigid_quotient` is Sella Cartesian `R^{3N}/SE(3)` (`fix_translation` + `fix_rotation`); `mw_rigid` is the Page-McIver / Sella IRC Eckart metric on the same quotient (`rgmin_solver_set_masses`).
SO(3) is length 9 and SE(3) is length 12; a 3N cluster is rejected rather than packed or prefix-interpreted.
Periodic bulk sets `rgmin_solver_set_periodic`: Sella `proj_rot = false`, quotient `R^{3N}/T(3)`.
Isolated stays `R^{3N}/SE(3)`.
`abi_minor` 10.
