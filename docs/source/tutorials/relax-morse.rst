

Relax a platinum crystal with L-BFGS and FIRE
---------------------------------------------

This tutorial relaxes 256 platinum atoms in a periodic box with three methods from ``rgmin``: limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) :cite:`liuLimitedMemoryBfgs1989`, the fast inertial relaxation engine (FIRE) :cite:`bitzekStructuralRelaxationMade2006`, and FIRE 2.0 :cite:`guenoleAssessmentOptimizationFast2020`.
At the end you have the relaxed energy from each method and a plot of how many force calls each one needed.

What you will need
~~~~~~~~~~~~~~~~~~

- A checkout of ``rgmin`` and Rust 1.88 or newer.

- Python 3 with matplotlib, only to redraw the figure.

The system
~~~~~~~~~~

The start is ``tests/data/morse_pos_0000.con``: 256 Pt atoms in a cubic cell of side 16.387907 Å.
The atoms interact through a Morse pair potential with :math:`D_0 = 0.7102` eV, :math:`\alpha = 1.6047` Å\ :sup:`-1`\, and :math:`r_0 = 2.897` Å, cut at 7.0 Å and shifted to zero there, the same potential the Morse tests use.
At the start the energy is -1392.989740 eV and the largest force on any atom is 1.556715 eV/Å.

Run the example
~~~~~~~~~~~~~~~

.. code:: bash

    cargo run --release --example relax_morse > trace.csv

The program prints one line per method when it stops:

::

    lbfgs: 9 steps, 11 force calls, E = -1396.173031 eV, fmax = 0.005466 eV/Å
    fire: 50 steps, 51 force calls, E = -1396.172302 eV, fmax = 0.009318 eV/Å
    fire2: 42 steps, 46 force calls, E = -1396.172349 eV, fmax = 0.009145 eV/Å

``trace.csv`` holds one row per step: method, step, force calls so far, energy, and the largest per-atom force.
Row 0 is the start.

How the loop works
~~~~~~~~~~~~~~~~~~

.. figure:: ../../figures/relax-morse/session_loop.png

    The host owns the loop, and one step may ask for several force calls.

The example never hands the whole job to the library.
It builds a ``Solver`` session, then calls ``step`` until the force is small enough, the way a geometry host such as eOn drives an optimizer.
This is the core of ``examples/relax_morse.rs`` with the printing left out:

.. code:: rust

    let mut solver = Solver::new(method, control, x0.len());
    solver.set_atom_maxmove(0.2);
    solver.set_stop_norm(StopNorm::MaxAtom);

    let mut x = x0.clone();
    while steps < BUDGET && last >= FMAX {
        let rep = solver.step(&obj, &mut x).expect("step");
        steps += 1;
        last = rep.grad_norm;
    }

``obj`` is an ``Oracle`` around a closure that returns the energy and its gradient and adds one to a counter on every call.
``set_atom_maxmove`` caps how far any atom moves in one step, here 0.2 Å.
``StopNorm::MaxAtom`` makes ``grad_norm`` the largest per-atom force, the same measure as ``fmax`` in the Atomic Simulation Environment (ASE), so the stop test is 0.01 eV/Å on every atom.
FIRE runs through the same loop.
Only the ``Method`` and ``istep`` change; for FIRE, ``istep`` is the initial time step.

Read the result
~~~~~~~~~~~~~~~

.. figure:: ../../figures/relax-morse/convergence.png

    Largest per-atom force against force calls on the Pt256 Morse start.

All three methods reach the same minimum: the final energies agree to within 0.000729 eV.
L-BFGS gets there in 11 force calls: its first step used three and every later step one.
FIRE used one call per step plus one for the start, 51 in all.
FIRE 2.0 took fewer steps, 42, and 46 calls.

To redraw the figure from a fresh run:

.. code:: bash

    python3 docs/figures/relax-morse/plot_convergence.py

Try next
~~~~~~~~

- Change ``FMAX`` in ``examples/relax_morse.rs`` and see how the gap between L-BFGS and FIRE moves.

- Point ``read_con`` at ``tests/data/morse_pos_0008.con``, the second fixture.

- Read :doc:`hold a stateful solver <../howto/session>` for the setters a host can call, and :doc:`the secant family <../explanation/quasi-newton>` for why a capped L-BFGS step behaves as it does.

If you use ``rgmin`` in published work, cite it with the repository's ``CITATION.cff``.
:doc:`../howto/cite` shows how to add the version you ran.
