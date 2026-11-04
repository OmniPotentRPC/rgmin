//! LINCS with mass weighting.
//!
//! The corrections run along the unit anchor directions `n_k`. One
//! projection sets every `n_k . r_k` to a target by solving the scaled
//! coupling `S = I - A` with the truncated expansion
//! `(I - A)^-1 ~ sum_{p <= order} A^p` (Hess et al. 1997, eqs. 18-20).
//! The first projection lowers `n_k . r_k` by `n_k . r_k - d_k`; each
//! correction then lowers it by `d_k - sqrt(2 d_k^2 - l_k^2)`, which
//! removes the lengthening a rotation leaves and has `l_k = d_k` as its
//! fixed point (eq. 21; GROMACS `mdlib/lincs.cpp`, `calc_dist_iter`).
//! Corrections repeat until the tolerance, so the expansion error is
//! refined away with the rotation instead of being left as GROMACS
//! leaves it after a fixed `lincs-iter`.

use ndarray::Array1;

use super::{Block, ConstraintAlgorithm, ConstraintError, Plan, Tangent, U, bond, dot3};
use crate::error::Result;

/// Expansion terms that bring the truncation bound
/// `rho^(n + 1) / (1 - rho)` of the Neumann series to `target`.
pub(super) fn derived_order(rho: f64, target: f64) -> usize {
    if rho <= 0.0 {
        return 0;
    }
    let terms = ((target * (1.0 - rho)).ln() / rho.ln()).ceil();
    (terms.max(1.0) as usize) - 1
}

impl Tangent<'_> {
    /// LINCS one block of `x` from this anchor. Returns the projections.
    /// An error can leave the block partly corrected, so `x` is the copy
    /// that [`Tangent::constrain`] publishes only on success.
    pub(super) fn lincs(&self, block: &Block, plan: &Plan, x: &mut Array1<f64>) -> Result<usize> {
        let set = self.set;
        let m = &block.members;
        let b = m.len();
        let rho = plan.rho_lincs();
        let order = set
            .lincs_order
            .unwrap_or_else(|| derived_order(rho, set.tolerance.max(U)));
        let dirs: Vec<[f64; 3]> = m
            .iter()
            .map(|&k| {
                let r = self.bonds[k];
                let n = dot3(&r, &r).sqrt();
                r.map(|c| c / n)
            })
            .collect();
        let weights: Vec<(f64, f64)> = m
            .iter()
            .map(|&k| {
                (
                    set.inverse_mass(set.pairs[k].0),
                    set.inverse_mass(set.pairs[k].1),
                )
            })
            .collect();
        let sw: Vec<f64> = weights.iter().map(|(wi, wj)| (wi + wj).sqrt()).collect();
        let reversed = |k: usize| ConstraintError::Reversed {
            algorithm: ConstraintAlgorithm::Lincs,
            pair: set.pairs[k],
        };
        let project = |x: &mut Array1<f64>, lower: &[f64]| {
            let rhs: Vec<f64> = (0..b).map(|r| lower[r] / sw[r]).collect();
            let mut sum = rhs.clone();
            let mut term = rhs;
            for _ in 0..order {
                let next: Vec<f64> = (0..b)
                    .map(|r| {
                        let row = &plan.s[r * b..(r + 1) * b];
                        let s: f64 = (0..b).filter(|&c| c != r).map(|c| row[c] * term[c]).sum();
                        -s
                    })
                    .collect();
                sum.iter_mut().zip(&next).for_each(|(s, t)| *s += t);
                term = next;
            }
            for r in 0..b {
                let (i, j) = set.pairs[m[r]];
                let mu = sum[r] / sw[r];
                let (wi, wj) = weights[r];
                for c in 0..3 {
                    x[3 * i + c] -= wi * mu * dirs[r][c];
                    x[3 * j + c] += wj * mu * dirs[r][c];
                }
            }
        };
        for (r, &k) in m.iter().enumerate() {
            if dot3(&dirs[r], &bond(x.view(), set.pairs[k])) <= 0.0 {
                return Err(reversed(k).into());
            }
        }
        let (excess, _) = self.block_excess(block, x.view());
        if excess.is_nan() {
            return Err(ConstraintError::NonFinite.into());
        }
        if excess <= 1.0 {
            return Ok(0);
        }
        let lengths: Vec<f64> = m.iter().map(|&k| set.lengths[k]).collect();
        let first: Vec<f64> = (0..b)
            .map(|r| dot3(&dirs[r], &bond(x.view(), set.pairs[m[r]])) - lengths[r])
            .collect();
        project(x, &first);
        // A correction is credited with one bit from the rotation and
        // (order + 1) log2(1 / rho) bits from the expansion; the smaller
        // of the two paces the budget.
        let per = if rho > 0.0 {
            ((order + 1) as f64 * (1.0 / rho).log2()).min(1.0)
        } else {
            1.0
        };
        let (excess, _) = self.block_excess(block, x.view());
        let budget = (excess.max(1.0).log2() / per).ceil() as usize + 1;
        let mut projections = 1;
        loop {
            let (e, k) = self.block_excess(block, x.view());
            if e.is_nan() {
                return Err(ConstraintError::NonFinite.into());
            }
            if e <= 1.0 {
                return Ok(projections);
            }
            if projections > budget {
                return Err(self.not_converged(
                    ConstraintAlgorithm::Lincs,
                    x.view(),
                    k,
                    projections,
                ));
            }
            let mut lower = Vec::with_capacity(b);
            for (r, &k) in m.iter().enumerate() {
                let r_k = bond(x.view(), set.pairs[k]);
                let p2 = 2.0 * lengths[r] * lengths[r] - dot3(&r_k, &r_k);
                if p2 <= 0.0 || dot3(&dirs[r], &r_k) <= 0.0 {
                    return Err(reversed(k).into());
                }
                lower.push(lengths[r] - p2.sqrt());
            }
            project(x, &lower);
            projections += 1;
        }
    }
}
