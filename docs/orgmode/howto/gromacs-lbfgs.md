
# Table of Contents

1.  [Run GROMACS L-BFGS minimization](#org7a5d4ac)
    1.  [Python and MDP keys](#orgbe0b808)
    2.  [Rust, C, and C++](#org8e012fd)



<a id="org7a5d4ac"></a>

# Run GROMACS L-BFGS minimization

Use `gromacs-lbfgs` to carry a GROMACS `integrator = l-bfgs`
energy-minimization block into rgmin. The defaults are `emtol = 10.0`
kJ mol<sup>-1</sup> nm<sup>-1</sup>, `emstep = 0.01` nm, `nsteps = -1`, `nbfgscorr = 10`,
and `constraints = none`.


<a id="orgbe0b808"></a>

## Python and MDP keys

```python
from rgmin.optimize import GromacsLbfgs

opt = GromacsLbfgs(
    atoms,
    emtol=10.0,
    emstep=0.01,
    nsteps=500,
    nbfgscorr=10,
)
opt.run()
```

For the functional API, map the MDP keys first:

```python
options = rgmin.gromacs_mdp_options(
    {
        "integrator": "l-bfgs",
        "emtol": 10.0,
        "emstep": 0.01,
        "nsteps": 500,
        "nbfgscorr": 10,
        "constraints": "none",
    }
)
method = options.pop("method")
report, x = rgmin.minimize(value_and_gradient, x0, method, **options)
```

Unknown keys raise `ValueError` and name the key. A non-`none`
`constraints` value is refused, as it is by GROMACS L-BFGS.


<a id="org8e012fd"></a>

## Rust, C, and C++

The Rust core is unit-agnostic:

```rust
let control = rgmin::Control::default()
    .with_gtol(force_tolerance)
    .with_istep(emstep)
    .with_maxiter(nsteps);
let mut solver =
    rgmin::Solver::new(rgmin::Method::GromacsLbfgs, control, x.len());
solver.set_gromacs_lbfgs_config(
    rgmin::GromacsLbfgsConfig::default().with_nbfgscorr(10),
)?;
let report = solver.run(&objective, &mut x)?;
```

The C selector is `RGMIN_GROMACS_LBFGS` and
`rgmin_solver_set_gromacs_lbfgs` sets the correction count. C++ uses
`rgmin::Method::GromacsLbfgs`, `Solver::set_gromacs_lbfgs`, and
`SolverConfig::with_gromacs_lbfgs`. The `nbfgscorr` runtime setting
selects the same count.

The exact line search, restart, stopping rules, and documented
differences from GROMACS 2023.3 are in
.
