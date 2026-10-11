import Mathlib

/- Citation: Steihaug, SIAM J. Numer. Anal. 20, 626 (1983), doi:10.1137/0720042. Nocedal and Wright, Numerical Optimization, doi:10.1007/978-0-387-40065-5. -/

/-! # The Steihaug boundary step

`src/hvp.rs::steihaug_cg` follows a conjugate-gradient direction until
one of three things happens: the residual is small, the curvature along
the direction fails to be positive, or the iterate would leave the
trust region. In the last two cases the code walks to the boundary
along the current direction, with the step length `tau` from
`boundary_tau`: the positive root of the quadratic
`dd * tau^2 + 2 * zd * tau + (zz - r^2) = 0`.

Two facts carry the whole construction: the root lands exactly on the
sphere, and it never walks backwards. The negative-curvature case adds
the third: pushing further along a non-positive-curvature descent
direction only lowers the model, so stopping at the boundary is the
right amount of greed.
-/

namespace RgminContracts

/-- The discriminant `boundary_tau` clamps to nonnegative before the
square root. -/
def disc (zz zd dd r : ℝ) : ℝ := zd ^ 2 + dd * (r ^ 2 - zz)

/-- The boundary step length: the positive root of the crossing
quadratic, exactly as `hvp.rs::boundary_tau` computes it. -/
noncomputable def boundaryTau (zz zd dd r : ℝ) : ℝ :=
  (-zd + Real.sqrt (disc zz zd dd r)) / dd

/-- **The boundary root lands on the sphere.** With a nonnegative
discriminant and a nonzero direction mass, the returned `tau`
satisfies the crossing quadratic exactly:
`||z + tau d||^2 = r^2` in whichever inner product the caller tracked
`zz`, `zd`, `dd` in -- Euclidean or preconditioned. -/
theorem boundary_on_sphere (zz zd dd r : ℝ)
    (hdd : dd ≠ 0) (hdisc : 0 ≤ disc zz zd dd r) :
    zz + 2 * boundaryTau zz zd dd r * zd
      + (boundaryTau zz zd dd r) ^ 2 * dd = r ^ 2 := by
  have hs : Real.sqrt (disc zz zd dd r) ^ 2 = disc zz zd dd r :=
    Real.sq_sqrt hdisc
  unfold boundaryTau
  field_simp
  unfold disc at hs ⊢
  nlinarith [hs]

/-- **The boundary step never walks backwards.** Inside the region
(`zz ≤ r^2`) with positive direction mass, the discriminant dominates
`zd^2`, so its square root is at least `|zd|` and the numerator
`-zd + sqrt(disc)` cannot be negative. A boundary exit always extends
the current step. -/
theorem boundary_tau_nonneg (zz zd dd r : ℝ)
    (hdd : 0 < dd) (hin : zz ≤ r ^ 2) :
    0 ≤ boundaryTau zz zd dd r := by
  have hslack : 0 ≤ dd * (r ^ 2 - zz) :=
    mul_nonneg hdd.le (by linarith)
  have hdom : zd ^ 2 ≤ disc zz zd dd r := by
    unfold disc; linarith
  have habs : |zd| ≤ Real.sqrt (disc zz zd dd r) := by
    have := Real.sqrt_le_sqrt hdom
    simpa [Real.sqrt_sq_eq_abs] using this
  have hzd : zd ≤ Real.sqrt (disc zz zd dd r) :=
    le_trans (le_abs_self zd) habs
  exact div_nonneg (by linarith) hdd.le

/-- Model value along a ray: `m(tau) = gd * tau + dHd * tau^2 / 2`. -/
noncomputable def rayModel (gd dHd tau : ℝ) : ℝ :=
  gd * tau + dHd * tau ^ 2 / 2

/-- **Under non-positive curvature, further is never worse.** Along a
descent direction (`gd ≤ 0`) with `dHd ≤ 0`, the ray model is
monotonically nonincreasing in the step, so the largest feasible step
-- the trust boundary -- minimizes the model on the ray. This is why
`steihaug_cg` treats negative curvature as licence to walk straight to
the sphere. -/
theorem ray_further_is_lower (gd dHd t1 t2 : ℝ)
    (hgd : gd ≤ 0) (hc : dHd ≤ 0) (h0 : 0 ≤ t1) (h12 : t1 ≤ t2) :
    rayModel gd dHd t2 ≤ rayModel gd dHd t1 := by
  unfold rayModel
  have hlin : gd * t2 ≤ gd * t1 := mul_le_mul_of_nonpos_left h12 hgd
  have hsq : t1 ^ 2 ≤ t2 ^ 2 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr h12) (by linarith : (0:ℝ) ≤ t1 + t2)]
  have hquad : dHd * t2 ^ 2 ≤ dHd * t1 ^ 2 :=
    mul_le_mul_of_nonpos_left hsq hc
  linarith

end RgminContracts
