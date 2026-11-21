- The C ABI no longer folds `StopReason::Iterating` into
  `RGMIN_STOP_BUDGET` for a finished run. `Iterating` is a step in
  progress, so it has no `rgmin_stop_t` of its own and no code is
  renumbered. A session step that has not stopped still reports
  `RGMIN_STOP_BUDGET`, as documented. The one-shots now return
  `RGMIN_INTERNAL_ERROR` and write nothing if a finished run ever
  carried `Iterating`, which no driver does today.
