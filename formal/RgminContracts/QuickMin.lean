import Mathlib
import RgminContracts.Clamp

/-! # QuickMin velocity projection

Contract for `src/quickmin.rs::project_on_force` and the step
`quickmin_step` builds from it.

`project_on_force` sets `v <- (v.F / |F|^2) F` when `v.F >= 0` and
`v <- 0` otherwise. It is the orthogonal projection of `v` on the closed
ray `{c F : c >= 0}`. The projected velocity is a nonnegative multiple
of `F` (`qmProject_parallel`), so its power is nonnegative and it has no
component along `-F` (`qmProject_power_nonneg`); the power is kept when
it was nonnegative (`qmProject_power_eq`); and it is never longer than
`v` (`qmProject_norm_le`), so the projection never adds kinetic energy.
The cell block and the per-atom blocks apply the same map to their own
coordinates, so each block satisfies these statements.

The step then kicks, `v' = P v + dt F`, and the session clamps the move
`dt v'` with `clampScale` (`RgminContracts.Clamp`). The clamped move is
a positive multiple of `F` (`qm_step_parallel`), so it is a descent
direction (`qm_step_descent`) and points along `F / |F|`
(`qm_step_direction`).

From rest the move is `dt^2 F`. On a quadratic whose curvature along `F`
is at most `lam` (`F.HF <= lam |F|^2`) the energy change
`-tau |F|^2 + tau^2 F.HF / 2` with `tau = dt^2` is negative whenever
`0 < tau lam < 2` (`qm_rest_step_decrease`): a time step under
`sqrt(2 / lam)` lowers the energy from rest, the bound the backtracking
variant restores by cutting `dt`.
-/

open RealInnerProductSpace

namespace RgminContracts

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The QuickMin projection of `v` on the force `F`. -/
noncomputable def qmProject (v F : E) : E :=
  if 0 ≤ ⟪v, F⟫ then (⟪v, F⟫ / ‖F‖ ^ 2) • F else 0

/-- The coefficient of `F` in `qmProject v F`. -/
noncomputable def qmCoeff (v F : E) : ℝ :=
  if 0 ≤ ⟪v, F⟫ then ⟪v, F⟫ / ‖F‖ ^ 2 else 0

theorem qmCoeff_nonneg (v F : E) : 0 ≤ qmCoeff v F := by
  unfold qmCoeff
  split_ifs with h
  · exact div_nonneg h (sq_nonneg _)
  · exact le_rfl

theorem qmProject_eq_coeff (v F : E) : qmProject v F = qmCoeff v F • F := by
  unfold qmProject qmCoeff
  split_ifs <;> simp

/-- **The projected velocity is parallel to the force**: a nonnegative
multiple of `F`. -/
theorem qmProject_parallel (v F : E) : ∃ c : ℝ, 0 ≤ c ∧ qmProject v F = c • F :=
  ⟨qmCoeff v F, qmCoeff_nonneg v F, qmProject_eq_coeff v F⟩

/-- **The projected power is nonnegative**: `v'.F >= 0`, so `v'` has no
component along `-F`. -/
theorem qmProject_power_nonneg (v F : E) : 0 ≤ ⟪qmProject v F, F⟫ := by
  rw [qmProject_eq_coeff, real_inner_smul_left, real_inner_self_eq_norm_sq]
  exact mul_nonneg (qmCoeff_nonneg v F) (sq_nonneg _)

/-- The component of the projected velocity along `-F` is nonpositive. -/
theorem qmProject_against_force_nonpos (v F : E) : ⟪qmProject v F, -F⟫ ≤ 0 := by
  rw [inner_neg_right]
  exact neg_nonpos.mpr (qmProject_power_nonneg v F)

/-- **A velocity along the force keeps its power.** -/
theorem qmProject_power_eq {v F : E} (hF : F ≠ 0) (h : 0 ≤ ⟪v, F⟫) :
    ⟪qmProject v F, F⟫ = ⟪v, F⟫ := by
  have hn : ‖F‖ ^ 2 ≠ 0 := pow_ne_zero 2 (norm_ne_zero_iff.mpr hF)
  unfold qmProject
  rw [if_pos h, real_inner_smul_left, real_inner_self_eq_norm_sq]
  exact div_mul_cancel₀ _ hn

/-- **The projection never lengthens the velocity**: `|v'| <= |v|`. -/
theorem qmProject_norm_le (v F : E) : ‖qmProject v F‖ ≤ ‖v‖ := by
  unfold qmProject
  split_ifs with h
  · by_cases hF : F = 0
    · simp [hF]
    · have hFn : 0 < ‖F‖ := norm_pos_iff.mpr hF
      rw [norm_smul, Real.norm_eq_abs, abs_div, abs_of_nonneg h,
        abs_of_nonneg (sq_nonneg _)]
      have cs : ⟪v, F⟫ ≤ ‖v‖ * ‖F‖ := real_inner_le_norm v F
      calc ⟪v, F⟫ / ‖F‖ ^ 2 * ‖F‖ = ⟪v, F⟫ / ‖F‖ := by
            rw [sq, div_mul_eq_mul_div, mul_div_mul_right _ _ hFn.ne']
        _ ≤ ‖v‖ * ‖F‖ / ‖F‖ := by gcongr
        _ = ‖v‖ := by rw [mul_div_assoc, div_self hFn.ne', mul_one]
  · simp

/-- **The kinetic energy does not grow**: `|v'|^2 <= |v|^2`. -/
theorem qmProject_kinetic_le (v F : E) : ‖qmProject v F‖ ^ 2 ≤ ‖v‖ ^ 2 := by
  have := qmProject_norm_le v F
  gcongr

/-- The clamped QuickMin move from velocity `v` under force `F`. -/
noncomputable def qmMove (cap M dt : ℝ) (v F : E) : E :=
  clampScale cap M • (dt • (qmProject v F + dt • F))

/-- **The clamped move is a positive multiple of the force.** -/
theorem qm_step_parallel {cap M dt : ℝ} (hcap : 0 < cap) (hdt : 0 < dt) (v F : E) :
    ∃ c : ℝ, 0 < c ∧ qmMove cap M dt v F = c • F := by
  refine ⟨clampScale cap M * (dt * (qmCoeff v F + dt)), ?_, ?_⟩
  · have := qmCoeff_nonneg v F
    exact mul_pos (clampScale_pos hcap) (mul_pos hdt (by linarith))
  · unfold qmMove
    rw [qmProject_eq_coeff, ← add_smul, smul_smul, smul_smul]
    congr 1
    ring

/-- **The clamped move is a descent direction** for the gradient
`g = -F` whenever the force is nonzero. -/
theorem qm_step_descent {cap M dt : ℝ} (hcap : 0 < cap) (hdt : 0 < dt) (v : E) {F : E}
    (hF : F ≠ 0) : ⟪-F, qmMove cap M dt v F⟫ < 0 := by
  obtain ⟨c, hc, h⟩ := qm_step_parallel (M := M) hcap hdt v F
  rw [h, real_inner_smul_right, inner_neg_left, real_inner_self_eq_norm_sq]
  have : 0 < ‖F‖ ^ 2 := pow_pos (norm_pos_iff.mpr hF) 2
  nlinarith

/-- **The clamped move points along the force**: its unit vector is
`F / |F|` (both sides are zero when `F = 0`). -/
theorem qm_step_direction {cap M dt : ℝ} (hcap : 0 < cap) (hdt : 0 < dt) (v F : E) :
    ‖qmMove cap M dt v F‖⁻¹ • qmMove cap M dt v F = ‖F‖⁻¹ • F := by
  obtain ⟨c, hc, h⟩ := qm_step_parallel (M := M) hcap hdt v F
  rw [h, norm_smul, Real.norm_of_nonneg hc.le, smul_smul]
  congr 1
  rw [mul_inv_rev, mul_assoc, inv_mul_cancel₀ hc.ne', mul_one]

/-- **A step from rest lowers a quadratic energy when `tau lam < 2`.**
With `n = |F|^2 > 0`, curvature `q = F.HF <= lam n` along the force and
`tau = dt^2 > 0`, the change `-tau n + tau^2 q / 2` is negative. -/
theorem qm_rest_step_decrease {n q lam tau : ℝ} (hn : 0 < n) (hq : q ≤ lam * n)
    (htau : 0 < tau) (hstab : tau * lam < 2) : -tau * n + tau ^ 2 * q / 2 < 0 := by
  have h1 : tau ^ 2 * q ≤ tau ^ 2 * (lam * n) :=
    mul_le_mul_of_nonneg_left hq (sq_nonneg tau)
  have h2 : tau ^ 2 * (lam * n) = tau * n * (tau * lam) := by ring
  have h3 : tau * n * (tau * lam) < tau * n * 2 :=
    mul_lt_mul_of_pos_left hstab (mul_pos htau hn)
  nlinarith

end RgminContracts
