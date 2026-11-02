

Run fire2 with the published parameters or bias correction
----------------------------------------------------------

A ``fire2`` session runs the Fast Inertial Relaxation Engine (FIRE)
2.0 on the crate's own parameters.
Two opt-in settings change that.
``set_fire_preset`` selects the parameters of
Guenole et al. :cite:`guenoleAssessmentOptimizationFast2020` Table 1.
``set_fire_abc`` turns on the accelerated bias-corrected (ABC) FIRE
update :cite:`echeverrirestrepoABCFIREAcceleratedBiascorrected2023`.
Both return an error on any method other than ``fire2``.
The values and the update are listed under FIRE on the reference
page :doc:`../reference/sota`.

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
    solver.set_fire_abc(true).unwrap();
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
    if (rgmin_solver_set_fire_preset(s, RGMIN_FIRE_PRESET_GUENOLE2020) != 0
        || rgmin_solver_set_fire_abc(s, 1) != 0) {
        /* rgmin_last_error() says why */
    }

Both setters need ABI minor 30.
``xts.h`` spells them ``xts_solver_set_fire_preset`` and
``xts_solver_set_fire_abc``.
The C++ wrapper has ``set_fire_preset`` and ``set_fire_abc``.

What each setting does
~~~~~~~~~~~~~~~~~~~~~~

``set_fire_preset`` restarts the integrator: the time step goes back
to ``istep``, the velocity to zero and the counters to zero.
Call it before the first step.
The preset scales its time-step bounds by ``istep``, so
``dt_max = 10 istep`` and ``dt_min = 0.02 istep``.
A shrink that would take the time step below ``dt_min`` leaves it
where it is, as in Algorithm 2 of the paper.
The default parameters clamp it to ``dt_min`` instead.

``set_fire_abc`` keeps its value across ``set_fire_preset``.
With ABC on, each displacement component is bounded by the
per-atom cap, or by ``maxmove`` without one, and the velocity is
clipped to the same bound divided by the time step.
That follows ASE ``FIRE2(use_abc=True)``.
Without ABC, the cap scales the whole step, as before.
With neither cap set, ABC has no clip.

The default session is unchanged by either setter until it is
called.
