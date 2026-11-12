# Variable-cell minimization against ASE FrechetCellFilter

ASE 3.29.0 `FrechetCellFilter` with FIRE (`dt = 0.1`) and BFGS (`maxstep = 0.2`),
and rgmin `relax_cell` with capped L-BFGS (memory 8, longest atom move 0.2)
and FIRE 2.0 (`istep = 0.1`).
Five serial repeats. The seconds column is the median wall time of the whole
run, optimizer and potential together. ASE is Python and rgmin is Rust.
Force calls do not change across repeats (`calls_stable true` on every arm).

The potential is nearest-neighbour Lennard-Jones, `epsilon = 1`, `sigma = 1`,
cutoff 1.4, minimum image, no shift. The same three starts are built in
`examples/bench_cell.rs` and `benches/variable_cell/ase_frechet.py`.
Initial energies match to the printed digit:

| system | start energy |
|---|---:|
| bulk4 | -22.862272158828 |
| shear4 | -23.761965007023 |
| vacancy31 | -172.762472447582 |

rgmin stops when the largest atomic block of the generalized gradient is
below `1e-3`, or at 300 steps. ASE stops at `fmax = 1e-3` or 300 steps.
rgmin force calls include the stress sample taken at the accepted geometry.
`converged` in the log is `max_force < 1e-3` and `max_stress < 1e-4`, which
is tighter than either stop, so an arm can sit on the known minimum
(bulk and shear energy `-24`, vacancy energy `-180`) with that flag clear.

| system | arm | calls | median seconds | max force | max \|stress\| | energy | converged |
|---|---|---:|---:|---:|---:|---:|---|
| bulk4 | rgmin-lbfgs | 17 | 4.158200e-5 | 8.366277e-4 | 9.070645e-5 | -24.00000000 | true |
| bulk4 | rgmin-fire2 | 41 | 8.947100e-5 | 4.862140e-4 | 3.495316e-4 | -23.99999998 | false |
| bulk4 | ase-fire | 75 | 9.134666e-1 | 1.618029e-4 | 2.854806e-4 | -23.99999999 | false |
| bulk4 | ase-bfgs | 21 | 1.522214e-1 | 1.131568e-4 | 2.709305e-5 | -24.00000000 | true |
| shear4 | rgmin-lbfgs | 12 | 2.640800e-5 | 1.363692e-4 | 2.744673e-4 | -24.00000000 | false |
| shear4 | rgmin-fire2 | 98 | 2.104560e-4 | 3.572323e-6 | 4.857333e-4 | -23.99999997 | false |
| shear4 | ase-fire | 99 | 6.929864e-1 | 6.046267e-4 | 9.877169e-4 | -23.99999992 | false |
| shear4 | ase-bfgs | 16 | 1.118764e-1 | 1.956619e-4 | 9.048195e-4 | -23.99999993 | false |
| vacancy31 | rgmin-lbfgs | 10 | 6.583090e-4 | 3.369937e-4 | 5.298456e-5 | -180.00000000 | true |
| vacancy31 | rgmin-fire2 | 50 | 3.317502e-3 | 5.973306e-4 | 7.596853e-5 | -179.99999999 | true |
| vacancy31 | ase-fire | 62 | 6.289845e+0 | 8.838533e-4 | 7.392287e-4 | -179.99999980 | false |
| vacancy31 | ase-bfgs | 12 | 1.831269e+0 | 5.202464e-4 | 9.841275e-6 | -179.99999999 | true |

Raw rows, rgmin:

```
bulk4 lbfgs 17 4.158200e-5 8.366277e-4 9.070645e-5 -24.00000000 true calls_stable true
bulk4 fire2 41 8.947100e-5 4.862140e-4 3.495316e-4 -23.99999998 false calls_stable true
shear4 lbfgs 12 2.640800e-5 1.363692e-4 2.744673e-4 -24.00000000 false calls_stable true
shear4 fire2 98 2.104560e-4 3.572323e-6 4.857333e-4 -23.99999997 false calls_stable true
vacancy31 lbfgs 10 6.583090e-4 3.369937e-4 5.298456e-5 -180.00000000 true calls_stable true
vacancy31 fire2 50 3.317502e-3 5.973306e-4 7.596853e-5 -179.99999999 true calls_stable true
```

Raw rows, ASE:

```
bulk4 ase-fire 75 9.134666e-01 1.618029e-04 2.854806e-04 -23.99999999 0 calls_stable True
bulk4 ase-bfgs 21 1.522214e-01 1.131568e-04 2.709305e-05 -24.00000000 1 calls_stable True
shear4 ase-fire 99 6.929864e-01 6.046267e-04 9.877169e-04 -23.99999992 0 calls_stable True
shear4 ase-bfgs 16 1.118764e-01 1.956619e-04 9.048195e-04 -23.99999993 0 calls_stable True
vacancy31 ase-fire 62 6.289845e+00 8.838533e-04 7.392287e-04 -179.99999980 0 calls_stable True
vacancy31 ase-bfgs 12 1.831269e+00 5.202464e-04 9.841275e-06 -179.99999999 1 calls_stable True
```
