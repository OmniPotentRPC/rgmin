import Mathlib

/-! # FIRE velocity mixing

Contract for `src/fire.rs::mix_velocity` and
`src/fire.rs::fire_rescale_velocity`.

`mix_velocity` sets `v <- (1 - a) v + a |v| F / |F|` (skipped when
`|F| = 0`). The mix does **not** preserve `|v|` in general
(`fireMix_not_norm_preserving`): the triangle inequality gives only the
bound `|v'| <= |v|` for `a` in `[0, 1]` (`fireMix_norm_le`), with the
lower bound `|1 - 2a| |v| <= |v'|` (`fireMix_norm_ge`). Outside `[0, 1]`
the bound fails (`fireMix_norm_gt_of_two`). The code keeps `a` in
`[0, 1]` only by its defaults (`alpha_start = 0.1`, `f_alpha = 0.99`);
`FireState::alpha`, `alpha_start` and `f_alpha` are public fields with
no range check.

The mix never lowers the power `F.v` (`fireMix_power_ge`), which is the
quantity `adapt` tests.
-/

open RealInnerProductSpace

namespace RgminContracts

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The FIRE mixing step. -/
noncomputable def fireMix (a : ℝ) (v F : E) : E := (1 - a) • v + (a * ‖v‖ / ‖F‖) • F

/-- **Mixing never speeds the system up.** For `a` in `[0, 1]` and a
nonzero force, `|v'| <= |v|`. -/
theorem fireMix_norm_le {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (v : E) {F : E} (hF : F ≠ 0) :
    ‖fireMix a v F‖ ≤ ‖v‖ := by
  have hFn : ‖F‖ ≠ 0 := norm_ne_zero_iff.mpr hF
  unfold fireMix
  calc ‖(1 - a) • v + (a * ‖v‖ / ‖F‖) • F‖
      ≤ ‖(1 - a) • v‖ + ‖(a * ‖v‖ / ‖F‖) • F‖ := norm_add_le _ _
    _ = (1 - a) * ‖v‖ + a * ‖v‖ := by
        rw [norm_smul, norm_smul, Real.norm_of_nonneg (by linarith),
          Real.norm_of_nonneg (by positivity)]
        field_simp
    _ = ‖v‖ := by ring

/-- **The reverse triangle bound.** `|1 - 2a| |v| <= |v'|`: the mix
can stop the system only when `a = 1/2` and `F` points against `v`. -/
theorem fireMix_norm_ge {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (v : E) {F : E} (hF : F ≠ 0) :
    |1 - 2 * a| * ‖v‖ ≤ ‖fireMix a v F‖ := by
  have hFn : ‖F‖ ≠ 0 := norm_ne_zero_iff.mpr hF
  unfold fireMix
  have h := abs_norm_sub_norm_le ((1 - a) • v) (-((a * ‖v‖ / ‖F‖) • F))
  rw [norm_neg, sub_neg_eq_add, norm_smul, norm_smul, Real.norm_of_nonneg (by linarith),
    Real.norm_of_nonneg (by positivity)] at h
  have e : a * ‖v‖ / ‖F‖ * ‖F‖ = a * ‖v‖ := by field_simp
  rw [e] at h
  have e2 : (1 - a) * ‖v‖ - a * ‖v‖ = (1 - 2 * a) * ‖v‖ := by ring
  rw [e2, abs_mul, abs_of_nonneg (norm_nonneg v)] at h
  exact h

/-- **Mixing never lowers the power.** For `a >= 0`,
`F.v' >= F.v` (Cauchy-Schwarz). -/
theorem fireMix_power_ge {a : ℝ} (ha0 : 0 ≤ a) (v : E) {F : E} (hF : F ≠ 0) :
    ⟪F, v⟫ ≤ ⟪F, fireMix a v F⟫ := by
  have hFn : ‖F‖ ≠ 0 := norm_ne_zero_iff.mpr hF
  unfold fireMix
  rw [inner_add_right, real_inner_smul_right, real_inner_smul_right,
    real_inner_self_eq_norm_sq]
  have e : a * ‖v‖ / ‖F‖ * ‖F‖ ^ 2 = a * (‖v‖ * ‖F‖) := by field_simp
  rw [e]
  have cs : ⟪F, v⟫ ≤ ‖v‖ * ‖F‖ := by
    have := real_inner_le_norm F v
    linarith [mul_comm ‖F‖ ‖v‖]
  nlinarith

/-- **The mix is not norm preserving.** In one dimension, `v = 1`,
`F = -1`, `a = 1/2` gives `v' = 0`. -/
theorem fireMix_not_norm_preserving :
    ¬ ∀ (a : ℝ), 0 ≤ a → a ≤ 1 → ∀ v F : ℝ, F ≠ 0 → ‖fireMix a v F‖ = ‖v‖ := by
  intro h
  have := h (1 / 2) (by norm_num) (by norm_num) 1 (-1) (by norm_num)
  norm_num [fireMix] at this

/-- **Outside `[0, 1]` the bound fails.** `a = 2`, `v = 1`, `F = -1`
gives `|v'| = 3`. -/
theorem fireMix_norm_gt_of_two : ‖fireMix (2 : ℝ) (1 : ℝ) (-1)‖ = 3 := by
  norm_num [fireMix]

/-- **The velocity rescale matches the clamped move.** If the proposed
displacement is `dt v` and the move taken is `c` times it with
`0 <= c <= 1` (the max-move clamp), `fire_rescale_velocity` multiplies
`v` by `|taken| / |proposed| = c`, so `dt v'` equals the move taken. A
bounds clip that is not a scaling breaks the premise. -/
theorem fire_rescale_consistent {dt c : ℝ} (hdt : 0 < dt) (hc0 : 0 ≤ c) {v : E} (hv : v ≠ 0) :
    dt • ((‖c • (dt • v)‖ / ‖dt • v‖) • v) = c • (dt • v) := by
  have hp : ‖dt • v‖ ≠ 0 := by
    rw [norm_smul]
    exact mul_ne_zero (by rw [Real.norm_of_nonneg hdt.le]; exact hdt.ne')
      (norm_ne_zero_iff.mpr hv)
  rw [norm_smul (c : ℝ), Real.norm_of_nonneg hc0, mul_div_assoc, div_self hp, mul_one,
    smul_smul, smul_smul, mul_comm]

end RgminContracts
