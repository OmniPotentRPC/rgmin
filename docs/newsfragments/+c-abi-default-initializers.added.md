`rgmin_control_default`, `rgmin_exp_default`, `rgmin_scg_params_default`, `rgmin_scg_options_default`, `rgmin_eigen_params_default` and `rgmin_eigen_options_default` return the Rust defaults of the C records a caller fills in, bit for bit, at ABI minor 46. `xts.h` spells them `xts_*_default`.
A C caller must start each record from its initializer, and `rgmin_precon_t` from `rgmin_precon_init`, then assign only the fields it changes. A field added later then holds its default rather than stack bytes or zero.
The C++ wrappers, the LAMMPS plugin, the asv helper, the C tests and the C ABI docs start from them.
In `rgmin_minimize_scg` and `rgmin_minimize_scg_with_options`, a null params or options pointer stands for the default record, which holds the values a null pointer selected before.
`RGMIN_C_TEST_RUNNER` and `RGMIN_LAMMPS_RUNNER` put a runner such as valgrind in front of the C test consumers and the plugin tests' `lmp`, and CI runs both under valgrind.
