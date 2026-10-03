import Mathlib

/-! The rational QN family away from its denominator floor. -/

namespace Rgmin

/-- A flipped spectral mode changes the sign of the alpha derivative. -/
theorem signed_qn_hasDerivAt (curvature gradient sign alpha : ℝ)
    (hne : curvature + alpha * sign ≠ 0) :
    HasDerivAt (fun a : ℝ => -gradient / (curvature + a * sign))
      (gradient * sign / (curvature + alpha * sign) ^ 2) alpha := by
  have hd : HasDerivAt (fun a : ℝ => curvature + a * sign) sign alpha := by
    simpa using ((hasDerivAt_id alpha).mul_const sign).const_add curvature
  convert (hasDerivAt_const alpha (-gradient)).div hd hne using 1 <;> ring

end Rgmin
