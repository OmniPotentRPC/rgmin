import Mathlib

namespace Rgmin

/-- Scalar restriction of the second-order SPD retraction. -/
def spdScalarRetract (x u : ℝ) : ℝ := x + u + u ^ 2 / (2 * x)

theorem spd_scalar_square_form (x u : ℝ) (hx : x ≠ 0) :
    spdScalarRetract x u = ((x + u) ^ 2 + x ^ 2) / (2 * x) := by
  unfold spdScalarRetract
  field_simp [hx]
  <;> ring

theorem spd_scalar_retract_positive (x u : ℝ) (hx : 0 < x) :
    0 < spdScalarRetract x u := by
  rw [spd_scalar_square_form x u (ne_of_gt hx)]
  apply div_pos
  · have hxx : 0 < x ^ 2 := pow_pos hx 2
    nlinarith [sq_nonneg (x + u)]
  · positivity

/-- Retraction of the accepted chord changes the returned point. -/
theorem spd_retract_chord_gap (x u : ℝ) :
    spdScalarRetract x (spdScalarRetract x u - x) - spdScalarRetract x u =
      (spdScalarRetract x u - x) ^ 2 / (2 * x) := by
  unfold spdScalarRetract
  ring

theorem spd_retained_first_step_witness :
    spdScalarRetract 2 (-2 / 5) = (41 / 25 : ℝ) ∧
      spdScalarRetract 2 (spdScalarRetract 2 (-2 / 5) - 2) = (4181 / 2500 : ℝ) := by
  norm_num [spdScalarRetract]

end Rgmin
