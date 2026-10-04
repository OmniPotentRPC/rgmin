import Mathlib

open scoped BigOperators

namespace Rgmin

/-- A nonzero scalar curvature scale has a positive magnitude. -/
theorem reset_scale_pos (h0 : ℝ) (hne : h0 ≠ 0) : 0 < |h0| :=
  abs_pos.mpr hne

/-- The reset follows the negative gradient in any finite dimension. -/
theorem force_reset_inner_product {n : ℕ} (scale : ℝ) (g : Fin n → ℝ) :
    (∑ i, (-scale * g i) * g i) = -scale * ∑ i, (g i) ^ 2 := by
  simp only [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Positive scale gives strict descent when the gradient norm is nonzero. -/
theorem force_reset_descent {n : ℕ} (scale : ℝ) (g : Fin n → ℝ)
    (hs : 0 < scale) (hg : 0 < ∑ i, (g i) ^ 2) :
    (∑ i, (-scale * g i) * g i) < 0 := by
  rw [force_reset_inner_product]
  have hprod := mul_pos hs hg
  nlinarith

/-- Multiplication by a positive cap factor preserves the directional sign. -/
theorem capped_force_reset_descent (slope factor : ℝ)
    (hd : slope < 0) (hf : 0 < factor) : factor * slope < 0 :=
  mul_neg_of_pos_of_neg hf hd

end Rgmin
