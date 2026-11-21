- The unmet stop now carries its report. C++ `minimize_fn`,
  `minimize_hess_fn` and `minimize_eindir` throw `rgmin::StopUnmetError`
  with `report()`, and Python `minimize` raises `rgmin.StopUnmetError`
  with `.report`. Both derive from the exception they threw before,
  `std::runtime_error` and `RuntimeError`, and keep the message.
