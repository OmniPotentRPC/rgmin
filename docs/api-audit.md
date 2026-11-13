# API audit

Commits on `develop` from 2026-09-13 through 2026-11-13. The C ABI that
TheochemUI/gpridge pins is `fb3a852` on `integration-gpridge-current-deps`,
through `subprojects/rgmin.wrap`. New C symbols are additive. Enum values
and struct layouts are unchanged. `abi_minor` is 36.

| item | problem | fix | done |
| --- | --- | --- | --- |
| `Lbfgs::minimize_box` | `ftol` stopped `minimize` and was ignored by the box driver, so a small relative decrease ran on to the iteration cap | the box loop uses the same stall test | yes |
| closure gradient | a gradient shorter than `x` panicked in the dot product | the answer is refused and the call returns `+inf` | yes |
| box bounds | a short bound, a NaN bound, or a lower entry above its upper entry panicked or clipped to a point outside the caller's box | those bounds return `+inf` and do not call the objective | yes |
| `lowest_eigenpair` | an empty seed made `lowest_mode` return `Err`, and the wrapper panicked | the wrapper returns that `Result` | yes |
| L-BFGS pair assembly | `expect` on the saved gradient aborted the process if the session lost it | the step returns `Error::Oracle` | yes |
| C++ `Method` | `RGMIN_EON_CG` and `RGMIN_POLAK_RIBIERE_PLUS` had no enumerators, so `method_from_name` produced a value the enum did not name | `EonCg` and `PolakRibierePlus` | yes |
| error string | the header had `rgmin_last_error` and no second name for the borrowed detail | `rgmin_last_error_message` returns the same pointer; the caller does not free it | yes |
| const queries | dimension and step count were not readable from a const session | `rgmin_solver_dim` and `rgmin_solver_steps` | yes |
| export set | the shared library exports Rust symbols beside the C ABI, and ld rejects a second version script next to rustc's anonymous map | `scripts/abi_check.sh` diffs the `rgmin_*` names and runs `abidiff` on a copy whose other dynamic symbols are localized | yes |
| property tests | the optimizers had example starts and no random convex or Rosenbrock cases | `tests/optimizer_properties.rs` | yes |
| Python | `develop` had no bindings | PyO3 `Lbfgs`, stubs, and `rgmin.ase.LBFGS` | yes |
| machine paths | tracked files named a temporary directory and an asv output directory under the benchflow tree | those instructions use a project-relative path, and the benchmark campaigns read an environment variable | yes |
| `CITATION.cff` | the file had no version while `Cargo.toml` says 0.2.0 | `version: 0.2.0` | yes |
| C acceptance default | `rgmin_solver_create` selects energy acceptance and `Solver::new` does not | left as it is; the guide states both, and changing the C default would change hosts that already call `rgmin_solver_create` | documented |
| installed header | cbindgen emits internal HiGHS types and is not what cargo cinstall ships | `include/rgmin.h` stays the installed header; `cbindgen.toml` records why generation is off | documented |
