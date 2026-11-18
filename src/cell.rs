//! Cell coordinates: a tail of the session vector that is not atoms.
//!
//! A variable-cell relaxation appends the cell degrees of freedom to the
//! `3N` atom coordinates. [`crate::Solver::set_cell_block`] marks
//! `[at, n)` as that tail. While such a session steps, the per-atom
//! measures in [`crate::step`] take `(x, y, z)` triples over `[0, at)`
//! only and count each cell coordinate on its own, scaled by
//! [`CellBlock::weight`], so the per-atom cap bounds a cell coordinate
//! by the cell cap. The layout lives in a thread-local slot that a
//! session fills for the length of its own step and empties around
//! every call into the objective, so a nested session or a free
//! function called from an oracle never sees it.

use std::cell::Cell;

use eindir_core::{Bounds, DifferentiableObjective, FPair, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::hvp::HessianVector;
use crate::newton::HessianObjective;

/// Cell tail of a session vector, as the per-atom measures read it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CellBlock {
    /// First cell coordinate. A multiple of 3.
    pub at: usize,
    /// Factor on a cell displacement before it is compared with a
    /// per-atom length: the per-atom cap over the cell cap. 1 compares
    /// a cell coordinate with the per-atom cap itself, 0 leaves it free.
    pub weight: f64,
}

impl CellBlock {
    /// The weight that bounds a cell coordinate by `cell_cap` where an
    /// atom is bounded by `atom_cap`. A positive finite `cell_cap` is the
    /// cell cap, `+inf` leaves the cell free, and anything else is the
    /// per-atom cap.
    pub(crate) fn weight_for(atom_cap: f64, cell_cap: f64) -> f64 {
        if cell_cap == f64::INFINITY {
            0.0
        } else if crate::step::binding(cell_cap) && crate::step::binding(atom_cap) {
            atom_cap / cell_cap
        } else {
            1.0
        }
    }

    /// Largest per-atom length of `d(0..n)`: triples over the atoms and
    /// the weighted magnitude of each cell coordinate.
    pub(crate) fn max_motion(self, n: usize, d: impl Fn(usize) -> f64) -> f64 {
        let at = self.at.min(n);
        let mut m = 0.0_f64;
        let mut i = 0;
        while i + 3 <= at {
            let (x, y, z) = (d(i), d(i + 1), d(i + 2));
            m = m.max((x * x + y * y + z * z).sqrt());
            i += 3;
        }
        if self.weight > 0.0 {
            for k in at..n {
                m = m.max(self.weight * d(k).abs());
            }
        }
        m
    }
}

thread_local! {
    static ACTIVE: Cell<Option<CellBlock>> = const { Cell::new(None) };
}

/// The cell block in force on this thread, for a vector of length `n`.
/// A block that starts past `n` does not describe that vector.
pub(crate) fn active(n: usize) -> Option<CellBlock> {
    ACTIVE.with(Cell::get).filter(|b| b.at <= n)
}

/// Restores the previous slot when dropped.
pub(crate) struct Scope {
    prev: Option<CellBlock>,
}

impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| slot.set(self.prev));
    }
}

/// Put `block` in force until the returned guard drops.
pub(crate) fn enter(block: Option<CellBlock>) -> Scope {
    Scope {
        prev: ACTIVE.with(|slot| slot.replace(block)),
    }
}

/// `obj` with the cell block out of force during each of its calls.
pub(crate) struct Unscoped<'a, O: ?Sized>(pub &'a O);

impl<O: Objective<f64> + ?Sized> Objective<f64> for Unscoped<'_, O> {
    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.0.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let _out = enter(None);
        self.0.eval(x)
    }

    fn eval_batch(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let _out = enter(None);
        self.0.eval_batch(x)
    }

    fn global_min(&self) -> Option<&FPair<f64>> {
        self.0.global_min()
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> Gradient<f64> for Unscoped<'_, O> {
    fn dim(&self) -> usize {
        Objective::dim(self.0)
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let _out = enter(None);
        Gradient::grad(self.0, x)
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> DifferentiableObjective<f64> for Unscoped<'_, O> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let _out = enter(None);
        self.0.value_and_gradient(x)
    }
}

impl<O: HessianObjective + ?Sized> HessianObjective for Unscoped<'_, O> {
    fn hessian(&self, x: ArrayView1<f64>) -> Array2<f64> {
        let _out = enter(None);
        self.0.hessian(x)
    }
}

impl<O: HessianVector + ?Sized> HessianVector for Unscoped<'_, O> {
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let _out = enter(None);
        self.0.hessian_vector(x, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scope_restores_the_slot_it_found() {
        let outer = CellBlock { at: 3, weight: 2.0 };
        let inner = CellBlock { at: 6, weight: 1.0 };
        assert_eq!(active(9), None);
        {
            let _a = enter(Some(outer));
            assert_eq!(active(9), Some(outer));
            {
                let _b = enter(Some(inner));
                assert_eq!(active(9), Some(inner));
                let _c = enter(None);
                assert_eq!(active(9), None);
            }
            assert_eq!(active(9), Some(outer));
            assert_eq!(active(2), None, "a block past the end does not apply");
        }
        assert_eq!(active(9), None);
    }

    #[test]
    fn each_cell_coordinate_counts_on_its_own() {
        let d = [3.0, 4.0, 0.0, 0.5, 0.5, 0.5];
        let block = CellBlock { at: 3, weight: 1.0 };
        assert_eq!(block.max_motion(6, |k| d[k]), 5.0);
        let block = CellBlock {
            at: 3,
            weight: 20.0,
        };
        assert_eq!(block.max_motion(6, |k| d[k]), 10.0);
        let block = CellBlock { at: 3, weight: 0.0 };
        assert_eq!(block.max_motion(6, |k| d[k]), 5.0);
    }

    #[test]
    fn the_weight_is_the_atom_cap_over_the_cell_cap() {
        assert_eq!(CellBlock::weight_for(0.1, 0.001), 0.1 / 0.001);
        assert_eq!(CellBlock::weight_for(0.1, f64::INFINITY), 0.0);
        assert_eq!(CellBlock::weight_for(0.1, 0.0), 1.0);
        assert_eq!(CellBlock::weight_for(0.1, -1.0), 1.0);
        assert_eq!(CellBlock::weight_for(0.1, f64::NAN), 1.0);
    }
}
