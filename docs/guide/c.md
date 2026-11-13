# C

`include/rgmin.h` is the stable header. `rgmin_solver_t` is opaque.
`rgmin_solver_create` allocates it and `rgmin_solver_free` releases it.
A null solver is a no-op for `rgmin_solver_free`.

`rgmin_last_error` and `rgmin_last_error_message` return the same
borrowed string. It stays valid until the next rgmin call on this
thread. The caller does not free it.

`rgmin_solver_dim` and `rgmin_solver_steps` take a const session.
A null session returns zero.

The quadratic `x^2 + y^2` from `(1, -2)` ends below `1e-8`.

```{literalinclude} ../examples/quadratic.c
:language: c
```

`scripts/abi_check.sh` diffs the dynamic `rgmin_*` names against
`abi/rgmin-symbols.txt` and runs `abidiff` against `abi/rgmin.xml`.
The comparison library has every other dynamic symbol localized,
because rustc's anonymous version script cannot be combined with a
second map.

A C session is created with energy acceptance. A Rust `Solver::new`
starts with acceptance off. The two agree after `rgmin_solver_set_accept`.
