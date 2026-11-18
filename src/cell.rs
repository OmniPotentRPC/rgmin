//! Variable-cell relaxation: the extended layout `x = [positions (3 N),
//! cell strain (9)]` and the filter that builds it.
//!
//! The filter is ASE 3.29.0 `FrechetCellFilter`, after Tadmor, Smith,
//! Bernstein and Kaxiras, Phys. Rev. B 59, 235 (1999) and the JuLIP
//! `ExpCell` of C. Ortner. The degrees of freedom are the atomic
//! positions in the original cell, `s = F^-1 r`, and `exp_cell_factor`
//! times the matrix logarithm of the deformation gradient
//! `F = (orig_cell^-1 cell)^T`. Cells are 3 by 3 with the lattice
//! vectors as rows, the ASE convention. The cell block holds the 9
//! entries of that matrix in C order.
//!
//! The oracle supplies the energy, the forces `-dE/dr` and the stress
//! `sigma = V^-1 dE/d(strain)` (the ASE sign: tensile positive, eV per
//! cubic angstrom). The gradient is the one ASE forms from them,
//! including the scalar pressure, the strain mask, the hydrostatic and
//! the constant-volume options.
//!
//! One difference from ASE is the origin of the cell block. ASE 3.29.0
//! takes the logarithm of `N F`, so its block carries the constant
//! `exp_cell_factor ln N` on the diagonal. Optimizers see only
//! differences of `x`, so the trajectories agree; this layout drops the
//! constant and the block is `exp_cell_factor log F` exactly.

mod matrix;

#[cfg(test)]
mod gold;

use ndarray::{Array1, ArrayView1};

use crate::Oracle;
use crate::error::{Error, Result};
use matrix::{Mat, det3, eye, inverse, logm, mul, transpose};

/// A cell: three lattice vectors as rows, in angstrom.
pub type Cell = [[f64; 3]; 3];

/// Cell degrees of freedom appended to the positions.
pub const CELL_DOF: usize = 9;

/// Options of the cell filter.
///
/// Non-exhaustive: start from [`Self::default`] and chain the `with_*`
/// setters. The default is `FrechetCellFilter(atoms)`: every strain
/// component relaxes, the factor is the atom count, and the pressure
/// is zero.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct CellFilterConfig {
    /// Scale of the cell block against the positions
    /// (`exp_cell_factor`). `None` is the atom count.
    pub exp_cell_factor: Option<f64>,
    /// Applied scalar pressure `p`, eV per cubic angstrom. The
    /// objective is the enthalpy `E + p V`.
    pub pressure: f64,
    /// Which entries of the deformation relax. A `false` entry keeps
    /// that component of the cell fixed.
    pub mask: [[bool; 3]; 3],
    /// Only isotropic strain: the virial is replaced by its trace over
    /// three on the diagonal.
    pub hydrostatic: bool,
    /// Project the trace out of the cell gradient. This breaks the
    /// gradient-energy consistency, so use it with a method that takes
    /// no line search on the enthalpy.
    pub constant_volume: bool,
}

impl Default for CellFilterConfig {
    fn default() -> Self {
        Self {
            exp_cell_factor: None,
            pressure: 0.0,
            mask: [[true; 3]; 3],
            hydrostatic: false,
            constant_volume: false,
        }
    }
}

with_setters!(CellFilterConfig {
    with_exp_cell_factor => exp_cell_factor: opt f64,
    with_pressure => pressure: val f64,
    with_mask => mask: val [[bool; 3]; 3],
    with_hydrostatic => hydrostatic: val bool,
    with_constant_volume => constant_volume: val bool,
});

impl CellFilterConfig {
    /// Strain mask in Voigt order `[xx, yy, zz, yz, xz, xy]`, ASE's
    /// six entry `mask`.
    #[must_use]
    pub fn with_voigt_mask(mut self, voigt: [bool; 6]) -> Self {
        self.mask = [
            [voigt[0], voigt[5], voigt[4]],
            [voigt[5], voigt[1], voigt[3]],
            [voigt[4], voigt[3], voigt[2]],
        ];
        self
    }
}

/// Energy, forces and stress of one cell and positions.
///
/// `forces` is `-dE/dr`, length `3 N`. `stress` is `V^-1 dE/d(strain)`
/// as a symmetric 3 by 3 matrix in eV per cubic angstrom.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct CellEvaluation {
    /// Potential energy.
    pub energy: f64,
    /// Atomic forces `-dE/dr`, length `3 N`, C order.
    pub forces: Array1<f64>,
    /// Stress in the ASE sign convention.
    pub stress: Cell,
}

impl CellEvaluation {
    /// Bundle an oracle answer.
    #[must_use]
    pub fn new(energy: f64, forces: Array1<f64>, stress: Cell) -> Self {
        Self {
            energy,
            forces,
            stress,
        }
    }
}

/// The extended coordinates of one system: its original cell, atom
/// count and [`CellFilterConfig`].
#[derive(Clone, Debug)]
pub struct CellFilter {
    config: CellFilterConfig,
    orig_cell: Cell,
    natoms: usize,
    dim: usize,
    factor: f64,
}

fn finite3(m: &Cell) -> bool {
    m.iter().all(|row| row.iter().all(|v| v.is_finite()))
}

impl CellFilter {
    /// Filter for `natoms` atoms whose reference cell is `orig_cell`.
    ///
    /// # Errors
    /// [`Error::Cell`] for no atoms, an atom count whose extended
    /// length `3 N + 9` overflows `usize`, a singular or non-finite
    /// cell, or a factor or pressure that is not finite and positive or
    /// finite.
    pub fn new(config: CellFilterConfig, orig_cell: Cell, natoms: usize) -> Result<Self> {
        if natoms == 0 {
            return Err(Error::Cell { what: "no atoms" });
        }
        let Some(dim) = natoms.checked_mul(3).and_then(|n| n.checked_add(CELL_DOF)) else {
            return Err(Error::Cell {
                what: "3 * natoms + 9 overflows usize",
            });
        };
        if !finite3(&orig_cell) || det3(&orig_cell).abs() <= 0.0 {
            return Err(Error::Cell {
                what: "the original cell is singular or not finite",
            });
        }
        if !config.pressure.is_finite() {
            return Err(Error::Cell {
                what: "the pressure is not finite",
            });
        }
        let factor = match config.exp_cell_factor {
            Some(f) => f,
            None => natoms as f64,
        };
        if !(factor.is_finite() && factor > 0.0) {
            return Err(Error::Cell {
                what: "exp_cell_factor is not finite and positive",
            });
        }
        Ok(Self {
            config,
            orig_cell,
            natoms,
            dim,
            factor,
        })
    }

    /// Atom count `N`.
    #[must_use]
    pub fn natoms(&self) -> usize {
        self.natoms
    }

    /// Length of the extended vector, `3 N + 9`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The resolved `exp_cell_factor`.
    #[must_use]
    pub fn exp_cell_factor(&self) -> f64 {
        self.factor
    }

    /// Reference cell the deformation is measured from.
    #[must_use]
    pub fn orig_cell(&self) -> &Cell {
        &self.orig_cell
    }

    /// The options in force.
    #[must_use]
    pub fn config(&self) -> &CellFilterConfig {
        &self.config
    }

    /// Cell volume `|det|`.
    #[must_use]
    pub fn volume(cell: &Cell) -> f64 {
        det3(cell).abs()
    }

    /// Enthalpy `E + p V` of an energy at `cell`.
    #[must_use]
    pub fn enthalpy(&self, energy: f64, cell: &Cell) -> f64 {
        energy + self.config.pressure * Self::volume(cell)
    }

    /// Deformation gradient `F = (orig_cell^-1 cell)^T`.
    ///
    /// # Errors
    /// [`Error::Cell`] when `cell` is not finite.
    pub fn deformation(&self, cell: &Cell) -> Result<Cell> {
        if !finite3(cell) {
            return Err(Error::Cell {
                what: "the cell is not finite",
            });
        }
        let Some(inv_orig) = inverse(&self.orig_cell) else {
            return Err(Error::Cell {
                what: "the original cell is singular",
            });
        };
        Ok(transpose(&mul(&inv_orig, cell)))
    }

    fn check_positions(&self, len: usize) -> Result<()> {
        if len != 3 * self.natoms {
            return Err(Error::Dim {
                got: len,
                dim: 3 * self.natoms,
            });
        }
        Ok(())
    }

    /// The extended vector of Cartesian `positions` in `cell`.
    ///
    /// # Errors
    /// [`Error::Dim`] on a wrong length, [`Error::Cell`] when the
    /// deformation is singular or has no real logarithm.
    pub fn pack(&self, positions: ArrayView1<f64>, cell: &Cell) -> Result<Array1<f64>> {
        self.check_positions(positions.len())?;
        let f = self.deformation(cell)?;
        let Some(f_inv) = inverse(&f) else {
            return Err(Error::Cell {
                what: "the deformation is singular",
            });
        };
        let log = logm(&f)?;
        let mut x = Array1::zeros(self.dim());
        for a in 0..self.natoms {
            for i in 0..3 {
                let mut s = 0.0;
                for j in 0..3 {
                    s += f_inv[i][j] * positions[3 * a + j];
                }
                x[3 * a + i] = s;
            }
        }
        let base = 3 * self.natoms;
        for i in 0..3 {
            for j in 0..3 {
                x[base + 3 * i + j] = self.factor * log[i][j];
            }
        }
        Ok(x)
    }

    /// Cartesian positions and cell of an extended vector.
    ///
    /// The masked components of the strain are ignored, as in ASE.
    ///
    /// # Errors
    /// [`Error::Dim`] on a wrong length, [`Error::Cell`] when the cell
    /// block is not finite.
    pub fn unpack(&self, x: ArrayView1<f64>) -> Result<(Array1<f64>, Cell)> {
        if x.len() != self.dim() {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim(),
            });
        }
        let base = 3 * self.natoms;
        let mut log = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                log[i][j] = x[base + 3 * i + j] / self.factor;
            }
        }
        let f = matrix::expm(&log)?;
        let mut shift: Mat<3> = eye::<3>();
        for i in 0..3 {
            for j in 0..3 {
                let strain = f[j][i] - if i == j { 1.0 } else { 0.0 };
                if self.config.mask[i][j] {
                    shift[i][j] += strain;
                }
            }
        }
        let cell = mul(&self.orig_cell, &shift);
        let mut positions = Array1::zeros(base);
        for a in 0..self.natoms {
            for j in 0..3 {
                let mut s = 0.0;
                for i in 0..3 {
                    s += x[3 * a + i] * shift[i][j];
                }
                positions[3 * a + j] = s;
            }
        }
        Ok((positions, cell))
    }

    /// Gradient of the enthalpy over the extended vector.
    ///
    /// `forces` is `-dE/dr` at the Cartesian positions of `cell`, length
    /// `3 N`. The first `3 N` entries are `-F^T f` per atom, and the cell
    /// block is `-(1 / exp_cell_factor)` times the derivative of the
    /// enthalpy with respect to `log F`, formed from the stress as
    /// `FrechetCellFilter.get_forces` does.
    ///
    /// # Errors
    /// [`Error::Dim`] on a wrong length, [`Error::Cell`] on a singular
    /// cell or a stress that is not finite.
    pub fn gradient(
        &self,
        forces: ArrayView1<f64>,
        stress: &Cell,
        cell: &Cell,
    ) -> Result<Array1<f64>> {
        self.check_positions(forces.len())?;
        if !finite3(stress) {
            return Err(Error::Cell {
                what: "the stress is not finite",
            });
        }
        let f = self.deformation(cell)?;
        let log = logm(&f)?;
        let volume = Self::volume(cell);
        let mut virial = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let p = if i == j { self.config.pressure } else { 0.0 };
                virial[i][j] = -volume * (stress[i][j] + p);
            }
        }
        if self.config.hydrostatic {
            let third = (virial[0][0] + virial[1][1] + virial[2][2]) / 3.0;
            virial = [[0.0; 3]; 3];
            for i in 0..3 {
                virial[i][i] = third;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                if !self.config.mask[i][j] {
                    virial[i][j] = 0.0;
                }
            }
        }
        let Some(f_t_inv) = inverse(&transpose(&f)) else {
            return Err(Error::Cell {
                what: "the deformation is singular",
            });
        };
        let ucf = mul(&virial, &f_t_inv);
        let mut cell_force = matrix::expm_frechet(&transpose(&log), &ucf)?;
        if self.config.constant_volume {
            let third = (cell_force[0][0] + cell_force[1][1] + cell_force[2][2]) / 3.0;
            for i in 0..3 {
                cell_force[i][i] -= third;
            }
        }
        let mut g = Array1::zeros(self.dim());
        for a in 0..self.natoms {
            for j in 0..3 {
                let mut s = 0.0;
                for i in 0..3 {
                    s += forces[3 * a + i] * f[i][j];
                }
                g[3 * a + j] = -s;
            }
        }
        let base = 3 * self.natoms;
        for i in 0..3 {
            for j in 0..3 {
                g[base + 3 * i + j] = -cell_force[i][j] / self.factor;
            }
        }
        Ok(g)
    }

    /// Objective over the extended vector, built from an oracle that
    /// answers for Cartesian positions and a cell.
    ///
    /// The value is the enthalpy and the gradient is [`Self::gradient`]. An
    /// oracle answer the filter cannot use (a wrong length, a
    /// non-finite energy or stress, a cell with no real logarithm) is a
    /// non-finite value and gradient, which the solver reports as
    /// [`Error::Oracle`].
    pub fn objective<F>(
        &self,
        eval: F,
    ) -> Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync>
    where
        F: Fn(ArrayView1<f64>, &Cell) -> CellEvaluation + Send + Sync,
    {
        let filter = self.clone();
        let dim = self.dim();
        Oracle::unbounded(dim, move |x| {
            let answer = filter.unpack(x).and_then(|(positions, cell)| {
                let evaluation = eval(positions.view(), &cell);
                filter.value_and_gradient(&evaluation, &cell)
            });
            match answer {
                Ok(pair) => pair,
                Err(_) => (f64::INFINITY, Array1::from_elem(dim, f64::NAN)),
            }
        })
    }

    /// Enthalpy and extended gradient of an [`CellEvaluation`] at the
    /// cell it was taken in.
    ///
    /// # Errors
    /// As [`Self::gradient`], and [`Error::Oracle`] for a non-finite energy.
    pub fn value_and_gradient(
        &self,
        evaluation: &CellEvaluation,
        cell: &Cell,
    ) -> Result<(f64, Array1<f64>)> {
        if !evaluation.energy.is_finite() {
            return Err(Error::Oracle {
                what: "energy is not finite",
            });
        }
        let g = self.gradient(evaluation.forces.view(), &evaluation.stress, cell)?;
        Ok((self.enthalpy(evaluation.energy, cell), g))
    }
}
