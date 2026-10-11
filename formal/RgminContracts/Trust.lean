import Mathlib

/- Citation: Conn, Gould, and Toint, Trust Region Methods, doi:10.1137/1.9780898719857. Nocedal and Wright, Numerical Optimization, doi:10.1007/978-0-387-40065-5. -/

/-! # The trust radius cannot lie

`src/trust.rs::update_radius` and the outer loop of
`src/hvp.rs::minimize_newton_cg` manage one number, the trust radius,
and the solver's honesty rests on three properties of that number: it
stays positive, it never exceeds the cap the caller set, and repeated
model failures drive it to the floor geometrically -- at which point
the code reports `TrustCollapsed` instead of pretending convergence.
-/

namespace RgminContracts

/-- The shrink branch of `update_radius`: a quarter of the radius,
floored. -/
noncomputable def shrink (r floor : ℝ) : ℝ := max (r / 4) floor

/-- The growth branch: doubled, capped. -/
noncomputable def grow (r rmax : ℝ) : ℝ := min (2 * r) (max rmax r)

/-- Shrinking keeps the radius at or above the floor. -/
theorem shrink_ge_floor (r floor : ℝ) : floor ≤ shrink r floor :=
  le_max_right _ _

/-- Shrinking a positive radius keeps it positive. -/
theorem shrink_pos (r floor : ℝ) (hf : 0 < floor) : 0 < shrink r floor :=
  lt_of_lt_of_le hf (shrink_ge_floor r floor)

/-- **Shrinking genuinely shrinks** once the radius sits above four
floors: the branch strictly decreases the radius, so a run of bad
ratios cannot orbit. -/
theorem shrink_strict (r floor : ℝ) (hf : 0 < floor) (h : 4 * floor < r) :
    shrink r floor < r := by
  unfold shrink
  exact max_lt (by linarith) (by linarith)

/-- Growth never exceeds the larger of the cap and where it started:
the radius cannot spring past `rmax` from below. -/
theorem grow_le_cap (r rmax : ℝ) : grow r rmax ≤ max rmax r :=
  min_le_right _ _

/-- Growth never loses ground when the radius is nonnegative. -/
theorem grow_ge (r rmax : ℝ) (hr : 0 ≤ r) (hcap : r ≤ rmax) :
    r ≤ grow r rmax := by
  unfold grow
  exact le_min (by linarith) (le_max_of_le_left hcap)

/-- **Model failure is priced geometrically.** Ignoring the floor,
`j` consecutive rejections leave `r / 4^j`: the envelope reaches any
positive floor in finitely many failures, so the outer loop must
either accept a step or report the collapse. -/
theorem collapse_envelope (r : ℝ) (hr : 0 ≤ r) (j : ℕ) :
    r / 4 ^ j ≤ r :=
  div_le_self hr (one_le_pow₀ (by norm_num))

/-- The collapse envelope goes to zero: no positive floor survives an
unbounded run of rejections. -/
theorem collapse_tendsto (r : ℝ) :
    Filter.Tendsto (fun j => r / 4 ^ j) Filter.atTop (nhds 0) := by
  have h : Filter.Tendsto (fun j => (4 : ℝ)⁻¹ ^ j)
      Filter.atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  have : (fun j => r / 4 ^ j) = fun j => (4 : ℝ)⁻¹ ^ j * r := by
    funext j
    rw [inv_pow]
    field_simp
  rw [this]
  simpa using h.mul_const r

/-- **A rejection is a rejection.** `reduction_ratio` with a positive
predicted reduction and a non-positive actual reduction is
non-positive, so the acceptance test `rho > eta` (with `eta > 0`)
cannot accept a step that did not lower the objective. -/
theorem rejection_propagates (ared pred eta : ℝ)
    (hp : 0 < pred) (ha : ared ≤ 0) (he : 0 < eta) :
    ared / pred < eta := by
  have hle : ared / pred ≤ 0 := div_nonpos_of_nonpos_of_nonneg ha hp.le
  linarith

end RgminContracts
