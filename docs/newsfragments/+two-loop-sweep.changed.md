Guenole FIRE 2.0 no longer allocates on a steady step.
The recycle pool keeps two buffers of one length for each stored pair.
A dot of at least 65536 values is a fixed-chunk pairwise sum.
That sum is the same on every thread count.
Each two-loop direction is one sweep.
The final negation is folded into the last update.
