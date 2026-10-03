# Formal contracts in Lean 4 and Mathlib

Machine-checked statements of what the solvers' algorithms guarantee,
each tied to the Rust function it constrains and to the precondition
the Rust code must establish for the guarantee to hold. The proofs are
over the reals (exact arithmetic); the Rust tests hold the
floating-point implementation to them at fixed tolerances.

`proofs/lean/` holds the scalar algebra of the line search, the trust
region and the Nystrom sketch. `formal/` holds the vector-space
contracts: the Armijo existence argument, BFGS and L-BFGS positive
definiteness, the per-atom clamp, and FIRE mixing.

## Check

```sh
cd formal
./check.sh            # lake exe cache get, lake build, sorry/axiom audit
SKIP_CACHE=1 ./check.sh   # when the Mathlib oleans are already present
```

`check.sh` fails when the build fails, when a source contains `sorry`,
`admit` or an `axiom` declaration, when the build log reports a sorry,
or when `#print axioms` of any theorem names an axiom beyond
`propext`, `Classical.choice` and `Quot.sound`. The toolchain is pinned
in `lean-toolchain` (`leanprover/lean4:v4.34.0-rc2`) and Mathlib in
`lakefile.toml` and `lake-manifest.json`
(`f2916a54665af851fc9a4da901cfc242c47a8922`), the same pins as
`proofs/lean/`, so one Mathlib cache serves both. Install the toolchain
with [elan](https://github.com/leanprover/elan).

## Theorem map

| Theorem (`RgminContracts.*`) | Source | What the Rust code must guarantee |
|---|---|---|
| `armijo_eventually`, `armijo_exists_step`, `armijo_exists_step_gradient` | `src/linesearch/conditions.rs::armijo` | A descent direction, `g.d < 0`, and `c < 1`. Then every step in some `(0, eps)` passes. `LineSearch::search_from` searches an uphill `d` as `-d` and refuses `g.d = 0`. |
| `backtracking_terminates` | `src/linesearch/known.rs::backtrack`, `LineSearch::Backtracking` | `alpha0 > 0` and `beta` in `(0, 1)`. Then some `alpha0 beta^k` passes; only the iteration cap or the step floor can stop the loop first. |
| `bfgsProd_inner_self`, `bfgsProd_posDef` | `src/qn.rs::bfgs_inverse_update` | `s.y > 0` for each update. The code skips `y.s <= 1e-12`. |
| `bfgsProd_eq_expanded`, `bfgsProd_isSymm` | `src/qn.rs::bfgs_inverse_update` | `H` symmetric on entry. The expanded entry loop then equals the product form, and the result stays symmetric. |
| `twoLoop_eq_lbfgsInv` | `src/lbfgs.rs::Lbfgs::direction_with_precon` | None. The two loops compute the nested BFGS product. |
| `lbfgsInv_posDef`, `lbfgs_descent` | `src/lbfgs.rs::Lbfgs::direction_with_precon` | Every stored pair has `s.y > 0` (`push_pair` rejects `s.y <= 1e-8 |s| |y|`). `H0` must be positive definite. |
| `gamma_pos`, `smul_id_posDef`, `lbfgs_gamma_descent` | `src/lbfgs.rs::Lbfgs::scale_gamma` | The newest pair has `s.y > 0`, so `gamma = s.y / y.y > 0`. With no pairs, `gamma = 1`. `extra_updates` is ignored: a replayed newest pair leaves the map unchanged (`validation/lbfgs_two_loop.py`). |
| `clampScale_pos`, `clampScale_le_one`, `clamp_inactive` | `src/step.rs::scale_step_atom`, `scale_step` | `cap > 0`. The factor is then in `(0, 1]` and equals `1` when no block exceeds the cap. |
| `clamp_block_le`, `clamp_block_eq_cap` | `src/step.rs::scale_step_atom` | `M` is the largest block norm. Every block then ends within `cap`, and the largest lands on it. |
| `clamp_direction`, `clamp_descent` | `src/step.rs::scale_step_atom` | `cap > 0`. Each block keeps its unit vector, and a descent direction stays one. |
| `clampScale_neg` | `src/step.rs::scale_step` | Shows the precondition matters. A negative cap would reverse the step; the code treats `cap <= 0` as no cap. |
| `fireMix_norm_le`, `fireMix_norm_ge` | `src/fire.rs::mix_velocity` | `a` in `[0, 1]` and `F /= 0`. Then `|1 - 2a| |v| <= |v'| <= |v|`. |
| `fireMix_power_ge` | `src/fire.rs::mix_velocity` | `a >= 0`. Mixing never lowers `F.v`. |
| `fireMix_not_norm_preserving`, `fireMix_norm_gt_of_two` | `src/fire.rs::mix_velocity` | Shows the limits: the mix does not preserve `|v|`, and `a = 2` breaks the bound. |
| `fire_rescale_consistent` | `src/fire.rs::fire_rescale_velocity` | The move taken is a nonnegative multiple of the proposed one. Then `dt v'` equals the move taken. |

## Where the Rust code establishes the preconditions

* `Control::maxmove` is a public `Option<f64>`. `scale_step`,
  `scale_step_atom` and `cap_alpha` (`src/step.rs`) treat a cap that is
  not positive and finite as no cap, so `clampScale_neg` cannot occur;
  `Solver::set_maxmove` and `set_atom_maxmove` store only positive caps.
* `FireState::alpha` is public; `mix_velocity` uses it clamped to
  `[0, 1]` (NaN as 0), the range `fireMix_norm_le` needs.
* `direction_with_precon` solves `P q = r` by Cholesky (`qn.rs::solve_spd`),
  which fails on a matrix that is not symmetric positive definite; the
  two-loop then falls back to `gamma I` and counts the event
  (`Lbfgs::precon_fallbacks`), so `lbfgs_descent` holds.
* `LineSearch::search_from` bounds its trials by `alpha_max` from the
  cap, so the accepted value and gradient are the oracle's at the
  returned point (`step.rs` test
  `the_accepted_value_is_the_oracle_value_at_the_capped_point`).
