import Mathlib

/-! # BFGS keeps the inverse Hessian positive definite; L-BFGS descends

Contracts for `src/qn.rs::bfgs_inverse_update` and the L-BFGS two-loop
recursion `src/lbfgs.rs::Lbfgs::direction_with_precon`.

Operators are functions on a real inner product space `E` (for the
code, `E = EuclideanSpace ℝ (Fin n)`, the flat coordinate vector).

* `bfgsProd H s y` is Nocedal-Wright 6.17 in product form,
  `H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T`, `rho = 1/(s.y)`.
* `bfgsExpanded H s y` is the expanded entry formula the Rust loop
  writes, `H - rho s (Hy)^T - rho (Hy) s^T + rho^2 (y.Hy) s s^T + rho s s^T`.
  The two agree when `H` is symmetric and linear (`bfgsProd_eq_expanded`).
* `twoLoop H0 pairs` is the two-loop recursion with the pairs listed
  newest first; `lbfgsInv H0 pairs` is the nested product of BFGS
  updates on `H0`; they agree (`twoLoop_eq_lbfgsInv`).

The precondition every theorem needs is positive curvature `0 < s.y`
for every update. `bfgs_inverse_update` skips a pair with
`y.s <= 1e-12`; `Lbfgs::push_pair` skips a pair with
`s.y <= 1e-8 |s| |y|` (or a non-finite `s.y`), which also rules out
`s = 0` or `y = 0`. The scaling `gamma = s.y / y.y` of the newest stored
pair is then positive (`gamma_pos`).
-/

open RealInnerProductSpace

namespace RgminContracts

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Positive definiteness of an operator, as a quadratic-form statement. -/
def PosDef (H : E → E) : Prop := ∀ z, z ≠ 0 → 0 < ⟪H z, z⟫

/-- Symmetry of an operator with respect to the inner product. -/
def IsSymm (H : E → E) : Prop := ∀ a b, ⟪H a, b⟫ = ⟪a, H b⟫

/-- Inverse BFGS update, product form (Nocedal-Wright 6.17). -/
noncomputable def bfgsProd (H : E → E) (s y : E) (z : E) : E :=
  H (z - (⟪s, y⟫⁻¹ * ⟪s, z⟫) • y)
    - (⟪s, y⟫⁻¹ * ⟪y, H (z - (⟪s, y⟫⁻¹ * ⟪s, z⟫) • y)⟫) • s
    + (⟪s, y⟫⁻¹ * ⟪s, z⟫) • s

/-- Inverse BFGS update in the expanded form `bfgs_inverse_update`
accumulates entry by entry. -/
noncomputable def bfgsExpanded (H : E → E) (s y : E) (z : E) : E :=
  H z - (⟪s, y⟫⁻¹ * ⟪H y, z⟫) • s - (⟪s, y⟫⁻¹ * ⟪s, z⟫) • H y
    + (⟪s, y⟫⁻¹ ^ 2 * ⟪y, H y⟫ * ⟪s, z⟫) • s + (⟪s, y⟫⁻¹ * ⟪s, z⟫) • s

/-- **The quadratic form of the update.** With `w = z - rho (s.z) y`,
`<H+ z, z> = <H w, w> + rho (s.z)^2`. No symmetry or linearity of `H`
is used. -/
theorem bfgsProd_inner_self (H : E → E) (s y z : E) :
    ⟪bfgsProd H s y z, z⟫ =
      ⟪H (z - (⟪s, y⟫⁻¹ * ⟪s, z⟫) • y), z - (⟪s, y⟫⁻¹ * ⟪s, z⟫) • y⟫
        + ⟪s, y⟫⁻¹ * ⟪s, z⟫ ^ 2 := by
  unfold bfgsProd
  set a := H (z - (⟪s, y⟫⁻¹ * ⟪s, z⟫) • y)
  simp only [inner_add_left, inner_sub_left, real_inner_smul_left, inner_sub_right,
    real_inner_smul_right, real_inner_comm a y]
  ring

/-- **BFGS preserves positive definiteness** (`bfgs_inverse_update`).
If `H` is positive definite and `0 < s.y`, the update is positive
definite. -/
theorem bfgsProd_posDef {H : E → E} (hH : PosDef H) {s y : E}
    (hsy : 0 < ⟪s, y⟫) : PosDef (bfgsProd H s y) := by
  intro z hz
  rw [bfgsProd_inner_self]
  set w := z - (⟪s, y⟫⁻¹ * ⟪s, z⟫) • y with hw
  have hρ : 0 < ⟪s, y⟫⁻¹ := inv_pos.mpr hsy
  by_cases h : ⟪s, z⟫ = 0
  · have hwz : w = z := by simp [hw, h]
    rw [hwz, h]
    have := hH z hz
    simpa using this
  · have h1 : 0 ≤ ⟪H w, w⟫ := by
      by_cases hw0 : w = 0
      · rw [hw0, inner_zero_right]
      · exact le_of_lt (hH w hw0)
    have h2 : 0 < ⟪s, z⟫ ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 h))
    have h3 : 0 < ⟪s, y⟫⁻¹ * ⟪s, z⟫ ^ 2 := mul_pos hρ h2
    linarith

/-- **The Rust entry formula is the product form** for a symmetric
linear `H`, so the positive-definiteness theorem covers the matrix the
code builds. -/
theorem bfgsProd_eq_expanded (H : E →ₗ[ℝ] E) (hS : IsSymm H) (s y z : E) :
    bfgsProd H s y z = bfgsExpanded H s y z := by
  unfold bfgsProd bfgsExpanded
  simp only [map_sub, map_smul, inner_sub_right, real_inner_smul_right]
  rw [← hS y z]
  module

/-- **BFGS preserves symmetry** for a symmetric linear `H`. -/
theorem bfgsProd_isSymm (H : E →ₗ[ℝ] E) (hS : IsSymm H) (s y : E) :
    IsSymm (bfgsProd H s y) := by
  intro a b
  rw [bfgsProd_eq_expanded H hS, bfgsProd_eq_expanded H hS]
  unfold bfgsExpanded
  simp only [inner_add_left, inner_sub_left, real_inner_smul_left, inner_add_right,
    inner_sub_right, real_inner_smul_right]
  rw [hS a b, real_inner_comm s a, real_inner_comm (H y) a]
  ring

/-- Nested BFGS updates on `H0`, pairs listed newest first:
`lbfgsInv H0 [p_m, ..., p_1] = U_m (... U_1 (H0))`. -/
noncomputable def lbfgsInv (H0 : E → E) : List (E × E) → E → E
  | [] => H0
  | p :: ps => bfgsProd (lbfgsInv H0 ps) p.1 p.2

/-- The two-loop recursion, pairs newest first. The first loop runs
newest to oldest (`a = rho s.q`, `q <- q - a y`), the middle applies
`H0`, the second runs oldest to newest (`b = rho y.r`,
`r <- r + (a - b) s`), as in `direction_with_precon`. Its result is
`H g`; the code returns `-H g`. -/
noncomputable def twoLoop (H0 : E → E) : List (E × E) → E → E
  | [], q => H0 q
  | p :: ps, q =>
    twoLoop H0 ps (q - (⟪p.1, p.2⟫⁻¹ * ⟪p.1, q⟫) • p.2)
      + (⟪p.1, p.2⟫⁻¹ * ⟪p.1, q⟫
          - ⟪p.1, p.2⟫⁻¹ * ⟪p.2, twoLoop H0 ps (q - (⟪p.1, p.2⟫⁻¹ * ⟪p.1, q⟫) • p.2)⟫) • p.1

/-- **The two loops compute the nested BFGS product.** -/
theorem twoLoop_eq_lbfgsInv (H0 : E → E) (ps : List (E × E)) (q : E) :
    twoLoop H0 ps q = lbfgsInv H0 ps q := by
  induction ps generalizing q with
  | nil => rfl
  | cons p ps ih =>
    simp only [twoLoop, lbfgsInv, bfgsProd, ih]
    module

/-- **Positive curvature on every pair keeps L-BFGS positive definite.** -/
theorem lbfgsInv_posDef {H0 : E → E} (hH0 : PosDef H0) :
    ∀ ps : List (E × E), (∀ p ∈ ps, 0 < ⟪p.1, p.2⟫) → PosDef (lbfgsInv H0 ps)
  | [], _ => hH0
  | p :: ps, h =>
    bfgsProd_posDef (lbfgsInv_posDef hH0 ps (fun q hq => h q (List.mem_cons_of_mem p hq)))
      (h p List.mem_cons_self)

/-- **The L-BFGS direction is a descent direction.** If `H0` is
positive definite and every stored pair has `0 < s.y`, then for a
nonzero gradient `g` the direction `d = -twoLoop H0 pairs g` that
`direction_with_precon` returns satisfies `<g, d> < 0`. -/
theorem lbfgs_descent {H0 : E → E} (hH0 : PosDef H0) (ps : List (E × E))
    (hps : ∀ p ∈ ps, 0 < ⟪p.1, p.2⟫) {g : E} (hg : g ≠ 0) :
    ⟪g, -twoLoop H0 ps g⟫ < 0 := by
  rw [twoLoop_eq_lbfgsInv, inner_neg_right, real_inner_comm]
  have := lbfgsInv_posDef hH0 ps hps g hg
  linarith

/-- The scaling `gamma = s.y / y.y` of `scale_gamma` is positive when
the newest pair has positive curvature. -/
theorem gamma_pos {s y : E} (hsy : 0 < ⟪s, y⟫) : 0 < ⟪s, y⟫ / ⟪y, y⟫ := by
  have hy : y ≠ 0 := by
    rintro rfl
    simp at hsy
  have : 0 < ⟪y, y⟫ := real_inner_self_pos.mpr hy
  exact div_pos hsy this

/-- A positive multiple of the identity is positive definite: the
middle step `q <- gamma q` of the two-loop, and `gamma = 1` when no pair
is stored. -/
theorem smul_id_posDef {γ : ℝ} (hγ : 0 < γ) : PosDef (fun z : E => γ • z) := by
  intro z hz
  rw [real_inner_smul_left]
  exact mul_pos hγ (real_inner_self_pos.mpr hz)

/-- **The unpreconditioned L-BFGS direction descends.** With
`H0 = gamma I`, `gamma = s.y / y.y` from the newest pair, and every
stored pair (including the `extra_updates` repeats of the newest) of
positive curvature, `<g, -H g> < 0` for `g /= 0`. -/
theorem lbfgs_gamma_descent (s y : E) (ps : List (E × E))
    (hps : ∀ p ∈ ps, 0 < ⟪p.1, p.2⟫) (hnew : 0 < ⟪s, y⟫) {g : E} (hg : g ≠ 0) :
    ⟪g, -twoLoop (fun z : E => (⟪s, y⟫ / ⟪y, y⟫) • z) ps g⟫ < 0 :=
  lbfgs_descent (smul_id_posDef (gamma_pos hnew)) ps hps hg

end RgminContracts
