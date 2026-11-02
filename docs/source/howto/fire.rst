

Run fire2 with the published parameters
---------------------------------------

A ``fire2`` session runs the Fast Inertial Relaxation Engine (FIRE)
2.0 on the crate's own parameters.
``set_fire_preset`` selects the parameters of
Guenole et al. :cite:`guenoleAssessmentOptimizationFast2020` Table 1
instead.
It returns an error on any method other than ``fire2``.
The values are listed in :doc:`../reference/sota` under
FIRE.

Rust
~~~~

.. code:: rust

    use ndarray::{Array1, ArrayView1, array};
    use rgmin::{Control, FireKind, FirePreset, Method, Oracle, Solver};

    let control = Control {
        maxiter: 5000,
        gtol: 1e-5,
        istep: 1e-3,
        maxmove: Some(0.2),
    };
    let mut solver = Solver::new(Method::Fire { kind: FireKind::V2 }, control, 2);
    solver.set_fire_preset(FirePreset::Guenole2020).unwrap();
    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        (x.dot(&x), Array1::from_iter(x.iter().map(|v| 2.0 * v)))
    });
    let mut x = array![0.5, -0.25];
    let report = solver.step(&obj, &mut x).unwrap();

C
~

.. code:: c

    #include "rgmin.h"

    rgmin_control_t ctrl = { .maxiter = 5000, .gtol = 1e-5, .istep = 1e-3, .maxmove = 0.2 };
    rgmin_solver_t *s = rgmin_solver_create(RGMIN_FIRE2, &ctrl, n);
    if (rgmin_solver_set_fire_preset(s, RGMIN_FIRE_PRESET_GUENOLE2020) != 0) {
        /* rgmin_last_error() says why */
    }

The setter needs ABI minor 30.
``xts.h`` spells it ``xts_solver_set_fire_preset``.
The C++ wrapper has ``set_fire_preset``.

What the preset does
~~~~~~~~~~~~~~~~~~~~

``set_fire_preset`` restarts the integrator: the time step goes back
to ``istep``, the velocity to zero and the counters to zero.
Call it before the first step.
The preset scales its time-step bounds by ``istep``, so
``dt_max = 10 istep`` and ``dt_min = 0.02 istep``.
A shrink that would take the time step below ``dt_min`` leaves it
where it is, as in Algorithm 2 of the paper.
The default parameters clamp it to ``dt_min`` instead.

The default session is unchanged until the setter is called.
