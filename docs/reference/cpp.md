# C++

`include/rgmin/optimize.hpp` is header-only. `rgmin::Method` names
every `rgmin_method_t` enumerator, including `EonCg` and
`PolakRibierePlus`. `method_from_name` throws `std::invalid_argument`
and reads `rgmin_last_error`.

The quadratic `x^2 + y^2` from `(1, -2)` ends below `1e-8`, as in
`docs/examples/quadratic.c`.
