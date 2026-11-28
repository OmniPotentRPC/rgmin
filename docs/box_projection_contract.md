
# Table of Contents

1.  [Descent fallback](#orgbe0b808)
2.  [Checks and scope](#org8e012fd)

The additive `Lbfgs::highs_step_boxed` method intersects coordinate bounds
with the uniform bounds and trust radius supplied through `HighsStep`.
It preserves the existing `highs_step` method and its scalar record fields.
The starting point must lie inside the coordinate box.


<a id="orgbe0b808"></a>

# Descent fallback

Let each displacement interval be `[l_i,u_i]`, with `l_i <` 0 <= u<sub>i</sub>=.
The projected force is `p_i = clip(-g_i,l_i,u_i)`.
The three cases of clipping give

    g_i p_i <= -p_i^2.

Summing over coordinates gives `g^T p <` -||p||<sub>2</sub><sup>2</sup>= in exact arithmetic.
A nonzero projected force therefore descends.
The implementation uses this fallback when component projection removes
all descent from the model direction.

The proof applies to the branch with coordinate and trust bounds.
A nonzero equality right-hand side can exclude the zero displacement.
The equality branch uses the existing HiGHS projection and checks its
bounded result against each equality, using an absolute row tolerance of `1e-7`.


<a id="org8e012fd"></a>

# Checks and scope

`proofs/lean/Rgmin/BoxProjection.lean` states the scalar bounds, scalar
descent inequality, and its finite-dimensional sum.
`validation/check_box_projection.py` checks the exact algebra and the
retained two-coordinate history that needs a projected-force restart.
Its Sollya driver checks explicit binary64 near-wall values.

The proof does not bound floating-point accumulation for arbitrary inputs.
It does not certify an inexact QP solve or a molecular search trajectory.
The retained Rust fixtures check coordinate widths, free motion near a
wall, one-sided bounds, equality intersections, and invalid input handling.
