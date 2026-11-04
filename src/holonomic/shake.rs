//! Gauss-Seidel SHAKE with mass weighting.
//!
//! Each constraint in turn moves its two atoms along its anchor bond by
//! the Newton step of its own length equation, the inverse masses
//! splitting the move (Ryckaert, Ciccotti and Berendsen 1977, eq. 5.6
//! without the `g^2` term; GROMACS `mdlib/shake.cpp`, `cshake`). A
//! constraint within its tolerance is skipped, and a block stops when
//! every member is within its tolerance.

use ndarray::Array1;

use super::{Block, ConstraintAlgorithm, ConstraintError, Plan, Tangent, U, bond, dot3};
use crate::error::Result;

impl Tangent<'_> {
    /// Iterations allowed to bring a block from `excess` (the largest
    /// residual over its tolerance) to one.
    ///
    /// Near the set the sweep is linear Gauss-Seidel on the scaled
    /// coupling `S` with contraction `rho` in the `S` norm. Turning the
    /// largest relative residual into that norm and back costs
    /// `sqrt(b kappa) c_max / c_min`, with `kappa` the condition number of
    /// `S` and `c_k = d_k^2 / sqrt(diag(J W J^T))_k` the scale between a
    /// relative residual and its `S` residual. A lone bond has `rho = 0`
    /// and SHAKE is Newton on its multiplier, which doubles the correct
    /// bits per sweep; the Newton term covers that phase.
    fn shake_budget(&self, block: &Block, plan: &Plan, excess: f64) -> usize {
        let set = self.set;
        let b = block.members.len();
        let c: Vec<f64> = block
            .members
            .iter()
            .zip(&plan.scale)
            .map(|(&k, s)| set.lengths[k] * set.lengths[k] / s)
            .collect();
        let spread = c.iter().fold(0.0_f64, |m, &v| m.max(v))
            / c.iter().fold(f64::INFINITY, |m, &v| m.min(v));
        let conversion = (b as f64 * plan.lambda_max / plan.lambda_min).sqrt() * spread;
        let rho = plan.rho_gauss_seidel();
        let linear = if rho > 0.0 {
            ((excess.ln() + conversion.ln()) / -rho.ln())
                .ceil()
                .max(0.0) as usize
        } else {
            0
        };
        let newton = excess.log2().max(1.0).log2().ceil() as usize + 1;
        linear + newton
    }

    /// SHAKE one block of `x` from this anchor. Returns the sweeps. An
    /// error can leave the block partly corrected, so `x` is the copy that
    /// [`Tangent::constrain`] publishes only on success.
    pub(super) fn shake(&self, block: &Block, plan: &Plan, x: &mut Array1<f64>) -> Result<usize> {
        let set = self.set;
        let (excess, _) = self.block_excess(block, x.view());
        if excess.is_nan() {
            return Err(ConstraintError::NonFinite.into());
        }
        if excess <= 1.0 {
            return Ok(0);
        }
        let budget = self.shake_budget(block, plan, excess);
        for sweep in 1..=budget {
            for &k in &block.members {
                let p = set.pairs[k];
                let r = bond(x.view(), p);
                let d2 = set.lengths[k] * set.lengths[k];
                let r2 = dot3(&r, &r);
                if ((d2 - r2) / (2.0 * d2)).abs() <= set.tol(x.view(), k) {
                    continue;
                }
                let b = &self.bonds[k];
                let rb = dot3(&r, b);
                // A move along b changes |r| only through r . b; at a
                // rounding-level angle from a right angle it cannot.
                if rb.is_nan() || rb <= U * (r2 * dot3(b, b)).sqrt() {
                    return Err(ConstraintError::Reversed {
                        algorithm: ConstraintAlgorithm::Shake,
                        pair: p,
                    }
                    .into());
                }
                let (wi, wj) = (set.inverse_mass(p.0), set.inverse_mass(p.1));
                let g = (d2 - r2) / (2.0 * (wi + wj) * rb);
                for c in 0..3 {
                    x[3 * p.0 + c] += wi * g * b[c];
                    x[3 * p.1 + c] -= wj * g * b[c];
                }
            }
            let (e, k) = self.block_excess(block, x.view());
            if e.is_nan() {
                return Err(ConstraintError::NonFinite.into());
            }
            if e <= 1.0 {
                return Ok(sweep);
            }
            if sweep == budget {
                return Err(self.not_converged(ConstraintAlgorithm::Shake, x.view(), k, sweep));
            }
        }
        let (_, k) = self.block_excess(block, x.view());
        Err(self.not_converged(ConstraintAlgorithm::Shake, x.view(), k, budget))
    }
}
