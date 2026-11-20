Added GPMin, the Gaussian-process-accelerated local geometry optimizer of Garijo del Río, Mortensen, and Jacobsen (Phys. Rev. B 100, 104103, 2019), as a general system-agnostic Rust stepper (`rgmin::GpMin`, `Method::GpMin`) with a matching `rgmin.optimize.GPMin` ASE drop-in and how-to documentation.
`GpMinConfig` and `SquaredExponentialKernel` are non-exhaustive and are built from `Default` with `with_*` setters.
The kernel default is ASE `GPMin` with `update_hyperparams` false: weight 1 and scale 0.4.
