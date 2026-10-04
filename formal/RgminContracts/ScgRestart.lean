import Mathlib

/-! # SCG direction cancellation

On an isotropic quadratic, a damped step leaves a gradient `r * g`.
The Liu-Storey recurrence can then produce the smaller direction `-r^2 * g`.
A direction-norm threshold alone cannot certify a gradient-norm threshold.
-/

namespace RgminContracts

/-- The Liu-Storey coefficient on this quadratic scales the direction twice. -/
theorem liu_storey_scaled_direction (r g : ℝ) :
    (r * (r - 1)) * (-g) - r * g = -(r ^ 2) * g := by ring

/-- An exact witness separates the two squared-norm thresholds. -/
theorem small_direction_with_nonstationary_gradient :
    (1 / 100000 : ℝ) ^ 4 * 25 < 1 / 2 ^ 52 ∧
      1 / 2 ^ 52 < (1 / 100000 : ℝ) ^ 2 * 25 := by norm_num

/-- The restarted direction is strictly descending for a nonzero gradient. -/
theorem gradient_restart_descent {gx gy : ℝ} (h : 0 < gx ^ 2 + gy ^ 2) :
    (-gx) * gx + (-gy) * gy < 0 := by nlinarith

end RgminContracts
