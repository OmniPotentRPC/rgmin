The 3×3 SPD block inverse in the geometry preconditioner now floors the leading 2×2 minor at eps*s^2 and the determinant at eps*s^3, where s is the largest absolute entry of the block, instead of an absolute threshold. Exact scale invariance of acceptance decisions and preconditioned actions holds for power-of-two scales, and exactly singular minors are still rejected.

`tests/precon_scaling.rs` now defines its near-zero pair-distance floor from machine epsilon (`sqrt(f64::EPSILON)`) with an explicit rationale comment, replacing a bare literal tolerance.
