import Mathlib

/-! # Approximate Wolfe conditions on quadratic restrictions

The slope inequality used by `roundoff_strong_wolfe` implies Armijo for
an exact quadratic restriction. Strong curvature also implies strict decrease
for that restriction when the opening slope is negative.

These are real-arithmetic statements. They do not bound an oracle's rounding
error or prove descent for a general nonlinear restriction.
-/

namespace RgminContracts

def lineQuadratic (c b a t : ℝ) : ℝ := c + b * t + a * t ^ 2 / 2

/-- The exact change equals the step times the mean endpoint slope. -/
theorem quadratic_change (c b a α : ℝ) :
    lineQuadratic c b a α - lineQuadratic c b a 0 =
      α * (b + (b + a * α)) / 2 := by
  unfold lineQuadratic
  ring

/-- The approximate Wolfe upper slope bound is Armijo for a quadratic. -/
theorem quadratic_armijo_of_slope {c b a α c₁ : ℝ} (hα : 0 < α)
    (hslope : b + a * α ≤ (2 * c₁ - 1) * b) :
    lineQuadratic c b a α ≤ lineQuadratic c b a 0 + c₁ * α * b := by
  have hscaled := mul_le_mul_of_nonneg_left hslope (le_of_lt hα)
  unfold lineQuadratic
  nlinarith

/-- A positive Armijo coefficient and negative opening slope give decrease. -/
theorem quadratic_decrease_of_slope {c b a α c₁ : ℝ}
    (hα : 0 < α) (hb : b < 0) (hc₁ : 0 < c₁)
    (hslope : b + a * α ≤ (2 * c₁ - 1) * b) :
    lineQuadratic c b a α < lineQuadratic c b a 0 := by
  have harmijo := quadratic_armijo_of_slope (c := c) hα hslope
  have hnegative : c₁ * α * b < 0 :=
    mul_neg_of_pos_of_neg (mul_pos hc₁ hα) hb
  linarith

/-- Strong curvature gives exact quadratic decrease for a descent direction. -/
theorem quadratic_decrease_of_strong_curvature {c b a α c₂ : ℝ}
    (hα : 0 < α) (hb : b < 0) (hc₂ : c₂ < 1)
    (hcurvature : |b + a * α| ≤ c₂ * |b|) :
    lineQuadratic c b a α < lineQuadratic c b a 0 := by
  have hslope : b + a * α ≤ -(c₂ * b) := by
    calc
      b + a * α ≤ |b + a * α| := le_abs_self _
      _ ≤ c₂ * |b| := hcurvature
      _ = -(c₂ * b) := by rw [abs_of_neg hb]; ring
  have hfactor : (1 - c₂) * b < 0 :=
    mul_neg_of_pos_of_neg (sub_pos.mpr hc₂) hb
  have hsum : b + (b + a * α) < 0 := by nlinarith
  have hchange : α * (b + (b + a * α)) / 2 < 0 :=
    div_neg_of_neg_of_pos (mul_neg_of_pos_of_neg hα hsum) (by norm_num)
  have hid := quadratic_change c b a α
  linarith

end RgminContracts
