import RgminContracts.BoxProjection
import RgminContracts.NebReset
import RgminContracts.Precond
import RgminContracts.QnDerivative
import RgminContracts.SpdRetraction
import RgminContracts.Spectral
import RgminContracts.Steihaug
import RgminContracts.Trust
import RgminContracts.UnitaryCompletion
import RgminContracts.Zoom

/- The proofs moved out of `proofs/lean` keep their declaration names.
`Rgmin.t` is `RgminContracts.t`. -/

namespace Rgmin

export RgminContracts (
  projectedForce
  projected_force_in_box
  projected_force_descent
  projected_force_sum_descent
  reset_scale_pos
  force_reset_inner_product
  force_reset_descent
  capped_force_reset_descent
  step_scale_invariant
  beta_scale_invariant
  metric_update_scales
  signed_qn_hasDerivAt
  spdScalarRetract
  spd_scalar_square_form
  spd_scalar_retract_positive
  spd_retract_chord_gap
  spd_retained_first_step_witness
  sketch_posSemidef
  sketch_form_nonneg
  shifted_weight_pos
  equalized_weight_bounds
  sumSq_nonneg
  disc
  boundaryTau
  boundary_on_sphere
  boundary_tau_nonneg
  rayModel
  ray_further_is_lower
  shrink
  grow
  shrink_ge_floor
  shrink_pos
  shrink_strict
  grow_le_cap
  grow_ge
  collapse_envelope
  collapse_tendsto
  rejection_propagates
  unitary_complement_real
  unitary_complement_imag
  unitary_complement_norm_pos
  positive_residual_exists
  guarded
  guarded_above
  guarded_below
  zoom_shrinks
  width_envelope
  width_envelope_nonneg
  width_envelope_tendsto
  armijo_strict
)

end Rgmin
