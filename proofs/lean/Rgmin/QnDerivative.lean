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
  simpa only [zero_mul, zero_sub, neg_mul, neg_neg] using
    (hasDerivAt_const alpha (-gradient)).fun_div hd hne

end Rgmin
