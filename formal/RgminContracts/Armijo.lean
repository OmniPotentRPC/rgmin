import Mathlib

/-! # Armijo sufficient decrease admits a positive step

Contract for `src/linesearch/conditions.rs::armijo` and the backtracking
loop behind `LineSearch::Backtracking`.

`armijo(phi, phi0, alpha, dphi0, c)` tests
`phi(alpha) <= phi(0) + c * alpha * phi'(0)`, where `phi(t) = f(x + t d)`
and `phi'(0)` is the slope `g . d` that the backtracking arm computes from
the gradient at the start. Backtracking tries `alpha0 * beta^k` for
`k = 0, 1, ...` and stops at the first trial that passes.

What the code needs from the mathematics: if `d` is a descent direction
(`phi'(0) < 0`) and `c < 1`, then every small enough positive step passes
the test, so the geometric sequence enters the accepting interval after
finitely many shrinks. Only differentiability of `f` at `x` is used; `C^1`
is more than enough. `c > 0` is not needed for existence; it is what makes
an accepted step strictly decrease the value (`proofs/lean/Rgmin/Zoom.lean`,
`armijo_strict`).
-/

open Filter Topology Set

namespace RgminContracts

/-- **Armijo holds on an initial interval.** A line function with a
negative slope `d0` at `0` satisfies the strict Armijo inequality for
every step in some `(0, eps)`, for any `c < 1`. -/
theorem armijo_eventually {φ : ℝ → ℝ} {d0 c : ℝ} (hφ : HasDerivAt φ d0 0)
    (hd : d0 < 0) (hc : c < 1) :
    ∃ ε > 0, ∀ t ∈ Ioo 0 ε, φ t < φ 0 + c * t * d0 := by
  have hlim := hφ.tendsto_slope_zero_right
  have hlt : d0 < c * d0 := by nlinarith
  have hev : ∀ᶠ t in 𝓝[>] (0 : ℝ), t⁻¹ • (φ (0 + t) - φ 0) < c * d0 :=
    hlim.eventually (eventually_lt_nhds hlt)
  rcases mem_nhdsGT_iff_exists_Ioo_subset.mp hev with ⟨ε, hε, hsub⟩
  refine ⟨ε, hε, fun t ht => ?_⟩
  have h1 : t⁻¹ * (φ t - φ 0) < c * d0 := by
    have := hsub ht
    simp only [Set.mem_ofPred_eq, zero_add, smul_eq_mul] at this
    exact this
  have ht0 : 0 < t := ht.1
  have key : t * (t⁻¹ * (φ t - φ 0)) = φ t - φ 0 := by
    field_simp
  nlinarith [mul_lt_mul_of_pos_left h1 ht0]

/-- **A descent direction admits an Armijo step (Frechet form).** For
`f` differentiable at `x` with derivative `f'` and `f' d < 0` (a descent
direction), every step in some `(0, eps)` passes the Armijo test along
`x + t d` with any `c < 1`. -/
theorem armijo_exists_step {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {f : E → ℝ} {f' : E →L[ℝ] ℝ} {x d : E} {c : ℝ}
    (hf : HasFDerivAt f f' x) (hd : f' d < 0) (hc : c < 1) :
    ∃ ε > 0, ∀ t ∈ Ioo 0 ε, f (x + t • d) < f x + c * t * f' d := by
  have hline : HasDerivAt (fun t : ℝ => x + t • d) d 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const d).const_add x
  have hf0 : HasFDerivAt f f' (x + (0 : ℝ) • d) := by simpa using hf
  have hφ : HasDerivAt (fun t : ℝ => f (x + t • d)) (f' d) 0 :=
    hf0.comp_hasDerivAt (0 : ℝ) hline
  obtain ⟨ε, hε, h⟩ := armijo_eventually hφ hd hc
  refine ⟨ε, hε, fun t ht => ?_⟩
  simpa using h t ht

/-- **Gradient form, the slope the backtracking arm computes.** With the
gradient `g` at `x` and `<g, d> < 0`, small positive steps pass
`f(x + t d) < f(x) + c t <g, d>`. -/
theorem armijo_exists_step_gradient {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [CompleteSpace E]
    {f : E → ℝ} {g x d : E} {c : ℝ}
    (hf : HasGradientAt f g x) (hd : inner ℝ g d < 0) (hc : c < 1) :
    ∃ ε > 0, ∀ t ∈ Ioo 0 ε, f (x + t • d) < f x + c * t * inner ℝ g d := by
  have hF := hf.hasFDerivAt
  have hd' : (InnerProductSpace.toDual ℝ E g) d < 0 := by
    simpa [InnerProductSpace.toDual_apply_apply] using hd
  obtain ⟨ε, hε, h⟩ := armijo_exists_step hF hd' hc
  exact ⟨ε, hε, fun t ht => by simpa [InnerProductSpace.toDual_apply_apply] using h t ht⟩

/-- **Backtracking terminates.** From any opening step `alpha0 > 0` and
any shrink factor `beta` in `(0, 1)`, some trial `alpha0 * beta^k` passes
the Armijo test. The backtracking arm finds it unless its iteration cap
(`maxiter`) or its step floor stops the shrinking first. -/
theorem backtracking_terminates {φ : ℝ → ℝ} {d0 c α0 β : ℝ}
    (hφ : HasDerivAt φ d0 0) (hd : d0 < 0) (hc : c < 1)
    (hα0 : 0 < α0) (hβ0 : 0 < β) (hβ1 : β < 1) :
    ∃ k : ℕ, φ (α0 * β ^ k) < φ 0 + c * (α0 * β ^ k) * d0 := by
  obtain ⟨ε, hε, h⟩ := armijo_eventually hφ hd hc
  obtain ⟨k, hk⟩ := exists_pow_lt_of_lt_one (div_pos hε hα0) hβ1
  refine ⟨k, h _ ⟨by positivity, ?_⟩⟩
  have := (lt_div_iff₀ hα0).mp hk
  linarith [mul_comm (β ^ k) α0]

end RgminContracts
