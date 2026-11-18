//! Opt-in extended layout `x = [positions (3 N), cell strain (9)]`.
//!
//! The solver steps on the extended vector like on any other. What the
//! layout changes is what the solver is told about it: rigid-body removal
//! is off, because translating the cell block as three more atoms is
//! meaningless, and the exponential preconditioner carries the `μ_c I`
//! block on the cell coordinates.

use ndarray::{Array1, ArrayView1};

use super::Solver;
use crate::cell::{Cell, CellEvaluation, CellFilter};
use crate::error::{Error, Result};
use crate::manifold::ManifoldKind;
use crate::precon::Slot;
use crate::report::Report;

/// The filter and the iterate the session keeps between steps.
#[derive(Clone, Debug)]
pub(super) struct CellSession {
    pub(super) filter: CellFilter,
    /// Extended iterate after the last step. Re-deriving it from the
    /// Cartesian answer would perturb its last bits and miss the cached
    /// evaluation, so the session keeps it.
    x: Option<Array1<f64>>,
    /// The Cartesian answer written for `x`.
    seen: Option<(Array1<f64>, Cell)>,
}

fn same_bits(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(p, q)| p.to_bits() == q.to_bits())
}

fn same_cell_bits(a: &Cell, b: &Cell) -> bool {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .all(|(p, q)| p.to_bits() == q.to_bits())
}

impl Solver {
    /// Run this session on the extended layout of `filter`.
    ///
    /// The session dimension must be [`CellFilter::dim`]. Set the filter
    /// before the preconditioner, on a session that has not stepped: the
    /// method memory is dropped. Euclidean sessions without distance or
    /// level-set constraints only. Rigid-body removal is off while the
    /// filter is installed.
    ///
    /// # Errors
    /// [`Error::Dim`] when the dimensions differ, [`Error::Cell`] for a
    /// manifold, a constraint or a preconditioner already installed.
    pub fn set_cell_filter(&mut self, filter: CellFilter) -> Result<()> {
        if filter.dim() != self.dim {
            return Err(Error::Dim {
                got: filter.dim(),
                dim: self.dim,
            });
        }
        if self.manifold != ManifoldKind::Euclidean || self.chart.is_some() {
            return Err(Error::Cell {
                what: "the cell layout is Euclidean",
            });
        }
        if self.constraints.is_some() || self.level_set.is_some() {
            return Err(Error::Cell {
                what: "the cell layout takes no distance or level-set constraint",
            });
        }
        if self.precon.is_some() {
            return Err(Error::Cell {
                what: "install the cell filter before the preconditioner",
            });
        }
        self.cell = Some(CellSession {
            filter,
            x: None,
            seen: None,
        });
        self.forget();
        Ok(())
    }

    /// Back to the plain layout. A preconditioner installed for the cell
    /// layout is dropped with it.
    pub fn clear_cell_filter(&mut self) {
        if self.cell.take().is_some() {
            self.clear_precon();
            self.forget();
        }
    }

    /// The installed cell filter.
    #[must_use]
    pub fn cell_filter(&self) -> Option<&CellFilter> {
        self.cell.as_ref().map(|c| &c.filter)
    }

    /// `μ_c` of the installed exponential preconditioner, after estimation.
    #[must_use]
    pub fn exp_mu_c(&self) -> Option<f64> {
        self.precon.as_ref().and_then(crate::precon::Slot::mu_c)
    }

    /// Fix the scale `μ_c` of the preconditioner's cell block, or `None`
    /// to estimate it with `μ` on the next step.
    ///
    /// # Errors
    /// [`Error::Cell`] without a cell filter or an exponential
    /// preconditioner, and for a `μ_c` that is not positive and finite.
    pub fn set_exp_mu_c(&mut self, mu_c: Option<f64>) -> Result<()> {
        if self.cell.is_none() {
            return Err(Error::Cell {
                what: "mu_c belongs to the cell layout",
            });
        }
        if mu_c.is_some_and(|v| !(v.is_finite() && v > 0.0)) {
            return Err(Error::Cell {
                what: "mu_c is not positive",
            });
        }
        match self.precon.take() {
            Some(Slot::Exp(mut state)) => {
                state.set_mu_c(mu_c);
                self.install_slot(Slot::Exp(state));
                Ok(())
            }
            other => {
                self.precon = other;
                Err(Error::Cell {
                    what: "no exponential preconditioner is installed",
                })
            }
        }
    }

    /// One outer iteration on Cartesian `positions` and `cell`, both
    /// overwritten with the accepted point.
    ///
    /// `eval` answers for positions and a cell with the energy, the forces
    /// and the stress. The returned [`Report`] describes the extended
    /// vector: its value is the enthalpy, its gradient the filter's, and its
    /// norm is what [`crate::Control::gtol`] is tested against.
    ///
    /// The session keeps its own extended iterate. When `positions` and
    /// `cell` are the bits it wrote last time it continues from that
    /// iterate and the cached evaluation; any other values restart the
    /// packing from them.
    ///
    /// # Errors
    /// [`Error::Cell`] without a filter, [`Error::Dim`] on a wrong length,
    /// and whatever [`Self::step`] reports.
    pub fn step_cell<F>(
        &mut self,
        eval: &F,
        positions: &mut Array1<f64>,
        cell: &mut Cell,
    ) -> Result<Report>
    where
        F: Fn(ArrayView1<f64>, &Cell) -> CellEvaluation + Send + Sync,
    {
        let Some(session) = &self.cell else {
            return Err(Error::Cell {
                what: "no cell filter is installed",
            });
        };
        let filter = session.filter.clone();
        let mut x = match (&session.x, &session.seen) {
            (Some(x), Some((p, c))) if same_bits(p, positions) && same_cell_bits(c, cell) => {
                x.clone()
            }
            _ => filter.pack(positions.view(), cell)?,
        };
        let objective = filter.objective(eval);
        let report = self.step(&objective, &mut x)?;
        let (p, c) = filter.unpack(x.view())?;
        positions.assign(&p);
        *cell = c;
        if let Some(session) = &mut self.cell {
            session.x = Some(x);
            session.seen = Some((p, c));
        }
        Ok(report)
    }
}
