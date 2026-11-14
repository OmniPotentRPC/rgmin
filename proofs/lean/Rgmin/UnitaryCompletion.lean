import Mathlib

open scoped BigOperators

namespace Rgmin

/-- The complex vectors (a+ib,c+id) and (-c+id,a-ib) are orthogonal. -/
theorem unitary_complement_real (a b c d : ℝ) :
    a * (-c) + b * d + c * a + d * (-b) = 0 := by ring

theorem unitary_complement_imag (a b c d : ℝ) :
    a * d - b * (-c) + c * (-b) - d * a = 0 := by ring

/-- The complement is nonzero whenever the source vector is nonzero. -/
theorem unitary_complement_norm_pos (a b c d : ℝ)
    (h : 0 < a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2) :
    0 < (-c) ^ 2 + d ^ 2 + a ^ 2 + (-b) ^ 2 := by nlinarith

/-- A positive sum of residual squared norms contains a resolved direction. -/
theorem positive_residual_exists {n : ℕ} (residual : Fin n → ℝ)
    (h : 0 < ∑ i, residual i) : ∃ i, 0 < residual i := by
  by_contra hnone
  push_neg at hnone
  have hsum : (∑ i, residual i) ≤ 0 := Finset.sum_nonpos (fun i _ => hnone i)
  linarith

end Rgmin
