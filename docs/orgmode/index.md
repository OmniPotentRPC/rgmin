

# Overview

`rgmin` finds a local minimum of a smooth function from its value and gradient.
A host program owns the loop, and a `Solver` session takes one step each time the host calls it.
The step can stay on an embedded manifold (`Euclidean`, `MwRigid`, `Sphere`, and the rest).
A session can also keep atom pairs at fixed distances, with the SHAKE, LINCS and SETTLE solvers of molecular dynamics.
It is the local-minimization crate of the OmniPotentRPC family, beside [rgpot](https://github.com/OmniPotentRPC/rgpot) and [rgsaddle](https://github.com/OmniPotentRPC/rgsaddle).

The solvers are written only in Rust.
C and C++ call them through one small C application binary interface (ABI), `include/rgmin.h` (`rgmin_solver_t` / `rgmin_minimize`).
Vectors cross it as DLPack `DLManagedTensorVersioned` tensors from the **dlpk** crate.
`include/rgmin/optimize.hpp` is the C++ wrapper and `include/xts/xtensor.hpp` adapts `xt::xarray`.
`include/xts.h` and `include/xts/optimize.hpp` keep the old `xts` names for existing hosts.
There is no second solver implementation in C++.
draws these layers.

The production unconstrained local method is limited-memory BFGS
without a line search: the eOn cap of 0.2, a curvature probe on the
first step, and a memory reset when the proposal is at least that
long. `LbfgsStep::Scaled` keeps a shortened pair.
`LbfgsStep::Wolfe` opts into the strong Wolfe search (Nocedal and
Wright algorithms 3.5&#x2013;3.6). Hopping chains keep that solver's pair
history on `Lbfgs`; anneal's `WarmLbfgs` is a thin handle around it.

Saddle, band, and IRC sessions live in `rgsaddle`. They consume this
crate's `Solver`; they do not grow a second optimizer.

```bash
cargo add rgmin@0.8 ndarray@0.17
```

That installs the 0.8 release on Rust 1.88 or newer.
covers the features, the `develop` branch, the C library from cargo-c or Meson, the Python packages and the LAMMPS plugin.


# Contact

Ask questions and report bugs on the [issue tracker](https://github.com/OmniPotentRPC/rgmin/issues).
Report a security problem privately by email to the maintainer, Rohit Goswami, at [rgoswami@ieee.org](mailto:rgoswami@ieee.org), as [SECURITY.md](https://github.com/OmniPotentRPC/rgmin/blob/main/SECURITY.md) describes.
The same address takes reports under the [code of conduct](https://github.com/OmniPotentRPC/rgmin/blob/main/CODE_OF_CONDUCT.md).


# Citation

If you use rgmin in published work, cite it with `CITATION.cff` and add the version or commit you ran.
gives a BibTeX entry and shows where to find the version.


# License

MIT.
