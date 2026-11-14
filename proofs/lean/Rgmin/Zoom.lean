import Mathlib

/-! # The zoom bracket

`src/linesearch/known.rs::zoom_trial` refines a bracket `[lo, hi]`
and proposes a cubic-Hermite trial inside it (Nocedal-Wright
eq. 3.59). The cubic can propose anything, so the code clamps every
trial into the middle 80 percent of the bracket: no closer to either
end than a tenth of the width. That guard is the entire termination
story: whichever end the trial replaces, the bracket keeps at least a
tenth of its width away, so the width falls geometrically and the
search cannot stall on a degenerate cubic.
-/

namespace Rgmin

/-- One guarded zoom trial: `t` clamped into the middle 80 percent of
the bracket `[lo, hi]`. -/
noncomputable def guarded (lo hi t : ℝ) : ℝ :=
  max (lo + (hi - lo) / 10) (min t (hi - (hi - lo) / 10))

/-- The guard really lands in the interior: at least a tenth of the
width from the low end. -/
theorem guarded_above (lo hi t : ℝ) :
    lo + (hi - lo) / 10 ≤ guarded lo hi t :=
  le_max_left _ _

/-- ... and at least a tenth of the width from the high end. -/
theorem guarded_below (lo hi t : ℝ) (h : lo ≤ hi) :
    guarded lo hi t ≤ hi - (hi - lo) / 10 :=
  max_le (by linarith) (min_le_right _ _)

/-- **The bracket shrinks by at least ten percent per zoom step.**
Whichever end the guarded trial replaces, the surviving interval is at
most nine tenths of the old width. This is the geometric decay that
makes the evaluation budget of `wolfe_search` finite; the cubic only
decides how much faster than the guarantee it goes. -/
theorem zoom_shrinks (lo hi t : ℝ) (h : lo ≤ hi) :
    max (hi - guarded lo hi t) (guarded lo hi t - lo)
      ≤ (9 / 10) * (hi - lo) := by
  have ha := guarded_above lo hi t
  have hb := guarded_below lo hi t h
  exact max_le (by linarith) (by linarith)

/-- Width after `k` guaranteed shrinks: the geometric envelope in
closed form. -/
theorem width_envelope (w : ℝ) (hw : 0 ≤ w) (k : ℕ) :
    (9 / 10 : ℝ) ^ k * w ≤ w := by
  have h1 : (9 / 10 : ℝ) ^ k ≤ 1 :=
    pow_le_one₀ (by norm_num) (by norm_num)
  calc (9 / 10 : ℝ) ^ k * w ≤ 1 * w :=
        mul_le_mul_of_nonneg_right h1 hw
    _ = w := one_mul w

/-- The envelope is a width: never negative. -/
theorem width_envelope_nonneg (w : ℝ) (hw : 0 ≤ w) (k : ℕ) :
    0 ≤ (9 / 10 : ℝ) ^ k * w :=
  mul_nonneg (pow_nonneg (by norm_num) k) hw

/-- **The envelope actually decays to zero**: for any positive
tolerance there is a zoom count after which the guaranteed width is
below it. Termination is not merely monotone, it is achieved. -/
theorem width_envelope_tendsto (w : ℝ) :
    Filter.Tendsto (fun k => (9 / 10 : ℝ) ^ k * w)
      Filter.atTop (nhds 0) := by
  have h : Filter.Tendsto (fun k => (9 / 10 : ℝ) ^ k)
      Filter.atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  simpa using h.mul_const w

/-- **Sufficient decrease is a real decrease.** The Armijo condition
`phi(a) ≤ phi(0) + c1 * a * slope` with a genuine descent slope and a
positive step strictly improves on the starting value; the accepted
point of every Wolfe search in the crate sits strictly below where the
search began. -/
theorem armijo_strict (phi0 phia c1 a slope : ℝ)
    (hc : 0 < c1) (ha : 0 < a) (hs : slope < 0)
    (h : phia ≤ phi0 + c1 * a * slope) : phia < phi0 := by
  have h1 : 0 < c1 * a := mul_pos hc ha
  have h2 : c1 * a * slope < 0 := mul_neg_of_pos_of_neg h1 hs
  linarith

end Rgmin
