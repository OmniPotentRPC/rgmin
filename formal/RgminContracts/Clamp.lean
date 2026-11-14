import Mathlib

/-! # The per-atom max-move clamp

Contract for `src/step.rs::scale_step_atom` (eOn `maxAtomMotionAppliedV`)
and its Euclidean sibling `src/step.rs::scale_step`.

`scale_step_atom` splits the displacement `trial - origin` into blocks
(consecutive xyz triples, the trailing remainder as one more block),
takes the largest block norm `M`, and when `M > cap` and `M > 0` scales
the whole displacement by `cap / M`. Here a displacement is a family
`d : ι → E` of blocks in a normed space (`E = EuclideanSpace ℝ (Fin 3)`
for atoms; a short trailing block embeds isometrically by zero padding).
`scale_step` is the one-block case `ι = Unit`.

The scale factor is `clampScale cap M`. The code must guarantee
`cap > 0`: `Solver::set_atom_maxmove` stores only a positive cap, but
`Control::maxmove` is a public `Option<f64>` with no check, and
`clampScale_neg` shows a negative cap reverses the step.
-/

namespace RgminContracts

/-- The factor `scale_step_atom` applies: `cap / M` when the largest
block exceeds the cap, else `1`. -/
noncomputable def clampScale (cap M : ℝ) : ℝ := if cap < M ∧ 0 < M then cap / M else 1

theorem clampScale_pos {cap M : ℝ} (hcap : 0 < cap) : 0 < clampScale cap M := by
  unfold clampScale
  split_ifs with h
  · exact div_pos hcap h.2
  · exact one_pos

/-- The clamp never lengthens a step. -/
theorem clampScale_le_one {cap M : ℝ} : clampScale cap M ≤ 1 := by
  unfold clampScale
  split_ifs with h
  · exact (div_le_one h.2).mpr h.1.le
  · exact le_rfl

variable {ι E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **Every block lands inside the cap.** If `M` bounds every block
norm (the code's `M` is the maximum), each scaled block has norm at
most `cap`. -/
theorem clamp_block_le {cap M : ℝ} (hcap : 0 < cap) (d : ι → E)
    (hM : ∀ i, ‖d i‖ ≤ M) (i : ι) : ‖clampScale cap M • d i‖ ≤ cap := by
  rw [norm_smul, Real.norm_of_nonneg (clampScale_pos hcap).le]
  unfold clampScale
  split_ifs with h
  · calc cap / M * ‖d i‖ ≤ cap / M * M :=
          mul_le_mul_of_nonneg_left (hM i) (div_pos hcap h.2).le
      _ = cap := div_mul_cancel₀ cap h.2.ne'
  · rw [one_mul]
    by_cases hM0 : 0 < M
    · have : M ≤ cap := by
        by_contra hc
        exact h ⟨lt_of_not_ge hc, hM0⟩
      exact (hM i).trans this
    · exact ((hM i).trans (not_lt.mp hM0)).trans hcap.le

/-- **The largest block lands on the cap.** When the block attaining
`M` exceeds the cap, the clamp puts it exactly on the cap sphere. -/
theorem clamp_block_eq_cap {cap : ℝ} (hcap : 0 < cap) (d : ι → E) (j : ι)
    (hj : cap < ‖d j‖) : ‖clampScale cap ‖d j‖ • d j‖ = cap := by
  have hM : 0 < ‖d j‖ := hcap.trans hj
  rw [norm_smul, Real.norm_of_nonneg (clampScale_pos hcap).le]
  unfold clampScale
  split_ifs with h
  · exact div_mul_cancel₀ cap hM.ne'
  · exact absurd ⟨hj, hM⟩ h

/-- **The clamp preserves every block's direction.** Scaling by a
positive factor leaves each nonzero block's unit vector unchanged. -/
theorem clamp_direction {cap M : ℝ} (hcap : 0 < cap) (x : E) :
    ‖clampScale cap M • x‖⁻¹ • (clampScale cap M • x) = ‖x‖⁻¹ • x := by
  have hs := clampScale_pos (M := M) hcap
  rw [norm_smul, Real.norm_of_nonneg hs.le, smul_smul]
  by_cases hx : x = 0
  · simp [hx]
  · congr 1
    have : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
    field_simp

/-- **The clamp keeps a descent direction a descent direction.** -/
theorem clamp_descent {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    {cap M : ℝ} (hcap : 0 < cap) {g d : F} (hd : inner ℝ g d < 0) :
    inner ℝ g (clampScale cap M • d) < 0 := by
  rw [real_inner_smul_right]
  exact mul_neg_of_pos_of_neg (clampScale_pos hcap) hd

/-- **A clamp that does not fire is the identity.** -/
theorem clamp_inactive {cap M : ℝ} (h : M ≤ cap) : clampScale cap M = 1 := by
  unfold clampScale
  split_ifs with h'
  · exact absurd h'.1 (not_lt.mpr h)
  · rfl

/-- **A negative cap reverses the step.** `Control::maxmove = Some(c)`
with `c < 0` reaches `scale_step` unchecked; any nonzero step then
gets a negative factor. -/
theorem clampScale_neg {cap M : ℝ} (hcap : cap < 0) (hM : 0 < M) : clampScale cap M < 0 := by
  unfold clampScale
  split_ifs with h
  · exact div_neg_of_neg_of_pos hcap hM
  · exact absurd ⟨hcap.trans hM, hM⟩ h

end RgminContracts
