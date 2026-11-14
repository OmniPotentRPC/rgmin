A steady Solver step for L-BFGS and FIRE 2.0 allocates nothing extra.
The first step at a length still builds the scratch.
The flat FIRE velocity update runs as one pass.
The mix and the displacement do the same.
