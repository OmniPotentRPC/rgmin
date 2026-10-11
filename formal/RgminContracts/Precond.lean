import Mathlib

/- Citation: Nocedal and Wright, Numerical Optimization, doi:10.1007/978-0-387-40065-5. Steihaug, SIAM J. Numer. Anal. 20, 626 (1983), doi:10.1137/0720042. -/

/-! # Preconditioning changes the clock, never the answer

`src/hvp.rs::steihaug_pcg` accepts any symmetric positive definite
preconditioner, including the randomized Nystrom sketch. The design
claim printed on the tin is that randomness lives only in the
preconditioner: the CG iterates themselves are exact.

The core of that claim is scale invariance. Scaling the preconditioner
`M -> c * M` (any `c > 0`) scales the preconditioned residual
`z = M^{-1} r` by `1/c`, and everything the iteration builds from it
-- the step `alpha * d`, the conjugacy weight `beta` -- comes out
unchanged. A preconditioner is a unit system, not an opinion about the
answer: if the mere scale of `M` could move an iterate, the
"exactness" story would already be false at the first step.

The identities are stated on the scalar recurrences the code runs,
with the scaling applied to each quantity exactly as `M -> c M`
induces: `z, d -> z/c, d/c`, `rz -> rz/c`, `dHd -> dHd/c^2`.
-/

namespace RgminContracts

/-- **The CG step ignores the preconditioner's scale.** With
`alpha = rz / dHd` and the scaled quantities `rz/c`, `dHd/c^2`,
`d/c`, the step `alpha' * d'` equals `alpha * d`, coordinate by
coordinate. -/
theorem step_scale_invariant (rz dHd d c : ℝ)
    (hc : c ≠ 0) (hd : dHd ≠ 0) :
    ((rz / c) / (dHd / (c * c))) * (d / c) = (rz / dHd) * d := by
  field_simp

/-- **The conjugacy weight ignores the preconditioner's scale.**
`beta = rz_next / rz` is a ratio of two quantities that scale the same
way, so the direction update `d <- -z + beta d` keeps its shape. -/
theorem beta_scale_invariant (rzNext rz c : ℝ)
    (hc : c ≠ 0) (hrz : rz ≠ 0) :
    (rzNext / c) / (rz / c) = rzNext / rz := by
  field_simp

/-- **The tracked metric norm scales linearly, so the boundary test is
scale-consistent.** `steihaug_pcg` compares `p_Mp` against `r^2` in
the `M`-metric; under `M -> c M` every tracked product picks up one
factor of `c`, so the comparison against a radius measured in the same
metric is the same comparison. Stated on the update recurrence. -/
theorem metric_update_scales (pMp pMd dMd alpha c : ℝ) :
    (c * pMp) + 2 * alpha * (c * pMd) + alpha ^ 2 * (c * dMd)
      = c * (pMp + 2 * alpha * pMd + alpha ^ 2 * dMd) := by
  ring

end RgminContracts
