- `rgmin::minimize_eindir` ignored `Control::error_on_unmet`, so a spent
  budget came back as a report. It now throws, as `minimize_fn` and
  `minimize_hess_fn` do. The C ABI gains `rgmin_minimize_eindir_opts`,
  the eindir one-shot with the `error_on_unmet` and `backward_polish`
  flags, at ABI minor 50. `rgmin_minimize_eindir` is unchanged.
