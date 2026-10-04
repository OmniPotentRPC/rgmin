import Mathlib

namespace Rgmin

/-- Scalar projection of the force into a displacement interval. -/
noncomputable def projectedForce (g lower upper : ℝ) : ℝ :=
  if -g < lower then lower else if upper < -g then upper else -g

theorem projected_force_in_box (g lower upper : ℝ)
    (hl : lower ≤ 0) (hu : 0 ≤ upper) :
    lower ≤ projectedForce g lower upper ∧ projectedForce g lower upper ≤ upper := by
  unfold projectedForce
  split_ifs with hlow hupp
  · exact ⟨le_rfl, le_trans hl hu⟩
  · exact ⟨le_trans hl hu, le_rfl⟩
  · constructor <;> linarith

theorem projected_force_descent (g lower upper : ℝ)
    (hl : lower ≤ 0) (hu : 0 ≤ upper) :
    g * projectedForce g lower upper ≤ -(projectedForce g lower upper) ^ 2 := by
  unfold projectedForce
  split_ifs with hlow hupp
  · have hsum : 0 ≤ g + lower := by linarith
    have hproduct := mul_nonneg hsum (neg_nonneg.mpr hl)
    nlinarith
  · have hsum : 0 ≤ -g - upper := by linarith
    have hproduct := mul_nonneg hsum hu
    nlinarith
  · nlinarith [sq_nonneg g]

theorem projected_force_sum_descent {ι : Type*} [Fintype ι]
    (g lower upper : ι → ℝ) (hl : ∀ i, lower i ≤ 0) (hu : ∀ i, 0 ≤ upper i) :
    (∑ i, g i * projectedForce (g i) (lower i) (upper i)) ≤
      -(∑ i, (projectedForce (g i) (lower i) (upper i)) ^ 2) := by
  have hsum : (∑ i, g i * projectedForce (g i) (lower i) (upper i)) ≤
      ∑ i, -((projectedForce (g i) (lower i) (upper i)) ^ 2) :=
    Finset.sum_le_sum fun i _ => projected_force_descent (g i) (lower i) (upper i) (hl i) (hu i)
  simpa only [Finset.sum_neg_distrib] using hsum

end Rgmin
