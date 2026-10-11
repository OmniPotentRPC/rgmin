import Mathlib

/- Citation: Frangella, Tropp, and Udell, SIAM J. Matrix Anal. Appl. 44 (2023), doi:10.1137/21M1466244. -/

/-! # The sketch cannot manufacture negative curvature

`src/hvp.rs::NystromPrecond` builds its low-rank factor as `B` with
`A_nys = B * B^T`, then shifts by the smallest kept eigenvalue. The
preconditioner is only legitimate if that construction is positive
semidefinite: an indefinite `M` would corrupt the CG inner products
and the "randomness only in the sketch" claim with them.

`B B^T` is a Gram matrix, and Gram matrices are positive semidefinite
-- here as the actual matrix theorem, not a scalar stand-in. The
weight lemmas then pin the solve: with a nonnegative captured
spectrum and a positive shift, every divisor is positive and every
damping factor sits in `(0, 1]`.
-/

namespace RgminContracts

open Matrix

/-- **The Nystrom factor is positive semidefinite, whatever the random
draw produced.** `B * Bᴴ` is a Gram matrix; over the reals the
conjugate transpose is the transpose, so this is exactly the
`A_nys = B B^T` the code assembles. No sketch, however unlucky, can
hand CG an indefinite preconditioner. -/
theorem sketch_posSemidef {n k : ℕ} (B : Matrix (Fin n) (Fin k) ℝ) :
    (B * Bᴴ).PosSemidef :=
  posSemidef_self_mul_conjTranspose B

/-- The quadratic form of the sketch at any vector is a sum of squares
in disguise: nonnegative pointwise, which is the statement
`sketch_posSemidef` unfolds to and the form the CG inner-product
argument uses. -/
theorem sketch_form_nonneg {n k : ℕ} (B : Matrix (Fin n) (Fin k) ℝ)
    (t : Fin n → ℝ) :
    0 ≤ t ⬝ᵥ (B * Bᴴ) *ᵥ t := by
  simpa using (sketch_posSemidef B).dotProduct_mulVec_nonneg t

/-- **The shifted weights are positive**: the `(lam_i + mu)` divisors
in `NystromPrecond::solve` never vanish when the kept eigenvalues are
nonnegative and the shift is positive, so the solve divides by
nothing it should not. -/
theorem shifted_weight_pos (lam mu : ℝ) (hl : 0 ≤ lam) (hm : 0 < mu) :
    0 < lam + mu := by linarith

/-- **The equalized weights damp and never amplify.** `solve` scales
captured mode `i` by `mu / (lam_i + mu)`; with `lam_i ≥ 0` and
`mu > 0` the weight sits in `(0, 1]`, so the preconditioner's action
on the sketched subspace is a contraction and no component ever flips
sign. -/
theorem equalized_weight_bounds (lam mu : ℝ) (hl : 0 ≤ lam) (hm : 0 < mu) :
    0 < mu / (lam + mu) ∧ mu / (lam + mu) ≤ 1 := by
  have hpos := shifted_weight_pos lam mu hl hm
  constructor
  · exact div_pos hm hpos
  · rw [div_le_one hpos]
    linarith

/-- A sum of squares over any finite index set is nonnegative: the
scalar heart of `nrm2` being a norm and of every Gram argument
above. -/
theorem sumSq_nonneg {n : ℕ} (x : Fin n → ℝ) :
    0 ≤ ∑ i, x i ^ 2 :=
  Finset.sum_nonneg fun i _ => sq_nonneg (x i)

end RgminContracts
