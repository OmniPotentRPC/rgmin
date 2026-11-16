A steady `Solver::step` reuses its coordinate and gradient buffers, so it allocates nothing extra.
This holds for L-BFGS with `Accept::Energy` and for FIRE 2.0.
The first step at a length still builds the scratch.
Later steps do not.
The flat FIRE velocity update runs as one pass.
The mix and the displacement run as one pass each.
Guenole FIRE 2.0 does the same with or without the per-atom cap, and with or without ABC.
