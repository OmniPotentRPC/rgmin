//! Positions and the cell as one Euclidean objective.
//!
//! [`CellFilter`] is opt-in. L-BFGS, FIRE, FIRE 2.0, and conjugate
//! gradients minimize it through the same
//! [`eindir_core::DifferentiableObjective`] handle they already use, and
//! no existing constructor changes its default.
//!
//! The map follows ASE 3.23.0 `ase.filters.UnitCellFilter` and
//! `FrechetCellFilter`
//! (<https://gitlab.com/ase/ase/-/blob/3.23.0/ase/filters.py>).
//! Lattice vectors are rows of the cell. The deformation gradient `F`
//! is the matrix with `cell = orig_cell F^T`, so each lattice vector is
//! `F` times the reference lattice vector, and a Cartesian position is
//! `r_i = F q_i`. The optimizer stores the reference positions `q_i`
//! and a `3 x 3` cell block:
//!
//! - [`CellCoordinates::Deformation`] stores `cell_factor * F`, ASE's
//!   `UnitCellFilter`. `cell_factor` defaults to the number of atoms.
//! - [`CellCoordinates::Frechet`] stores
//!   `exp_cell_factor * log(cell_factor * F)`, the principal logarithm
//!   of that block. `exp_cell_factor` defaults to the same atom count.
//!   ASE 3.29.0 `CellUtility.get_positions_frechet` agrees when
//!   `cell_factor` is the atom count.
//!
//! The value is the enthalpy `E + P V` with `V = det(cell)`. The
//! gradient is the derivative of that enthalpy with respect to the
//! stored vector. `P` is an external scalar pressure in energy per
//! volume. The stress the wrapped objective returns is the Cauchy
//! stress ASE exposes as `get_stress(voigt=False)`: at a stationary
//! cell under pressure `P` that stress is `-P` on the diagonal.
//!
//! [`CellFilter::with_mask`] fixes Voigt components of `F` at the
//! identity deformation, and [`CellFilter::with_hydrostatic_strain`]
//! replaces `F` by `(tr F / 3) I` before the mask. The gradient is the
//! derivative through that projection.

use std::sync::atomic::{AtomicU64, Ordering};

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use thiserror::Error;

/// Energy, Cartesian forces, and Cauchy stress at one geometry.
///
/// `forces` has length `3 * natoms`, atom `i` in entries `3i..3i+3`.
/// A force is the negative gradient of the energy with respect to the
/// Cartesian position. `stress` is row-major, in energy per volume, and
/// uses the sign of ASE `get_stress(voigt=False)`.
#[derive(Clone, Debug, PartialEq)]
pub struct CellSample {
    /// Potential energy, without the pressure term.
    pub energy: f64,
    /// Cartesian forces, row-major.
    pub forces: Vec<f64>,
    /// Cauchy stress, row-major `3 x 3`.
    pub stress: [f64; 9],
}

/// How the last nine generalized coordinates represent the cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CellCoordinates {
    /// ASE `UnitCellFilter`: the cell block is `cell_factor * F`.
    Deformation,
    /// ASE `FrechetCellFilter`: the cell block is
    /// `exp_cell_factor * log(cell_factor * F)`.
    Frechet,
}

/// A cell filter could not be built or a geometry could not be read.
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum CellFilterError {
    /// `natoms` was zero.
    #[error("a cell filter needs at least one atom")]
    Empty,
    /// The reference cell was non-finite or had no inverse, or the
    /// current cell has no inverse.
    #[error("the cell is singular or non-finite")]
    SingularCell,
    /// The reference cell was left-handed or flat.
    #[error("the reference cell must have a positive determinant")]
    Orientation,
    /// `cell_factor` or `exp_cell_factor` was not finite and positive.
    #[error("{name} must be finite and positive, got {value}")]
    Factor {
        /// Which scale was refused.
        name: &'static str,
        /// The refused value.
        value: f64,
    },
    /// The external pressure was not finite.
    #[error("pressure must be finite, got {value}")]
    Pressure {
        /// The refused pressure.
        value: f64,
    },
    /// A position or coordinate vector had the wrong length.
    #[error("length {got} != {dim}")]
    Dim {
        /// The length that was supplied.
        got: usize,
        /// The length the filter required.
        dim: usize,
    },
    /// A position, cell, or coordinate entry was NaN or infinite.
    #[error("a position or cell entry is non-finite")]
    NonFinite,
    /// The generalized vector does not decode to a cell of positive volume.
    ///
    /// A left-handed deformation takes this path. The objective value and
    /// gradient are NaN for the same vector.
    #[error("the generalized vector does not decode")]
    Decode,
    /// `log(cell_factor * F)` is not a real logarithm.
    ///
    /// A reflected cell takes this path. A half-turn has a real logarithm,
    /// and [`CellFilter::coordinates`] accepts a cell that
    /// [`CellFilter::geometry`] produced from that deformation.
    #[error("the deformation gradient has no principal logarithm")]
    Logarithm,
}

/// Objective whose variables are reference positions plus a cell block.
///
/// `evaluate` receives Cartesian positions and the row-major cell, after
/// a mask or hydrostatic projection. It returns the energy, the
/// Cartesian forces, and the Cauchy stress of that geometry.
pub struct CellFilter<F> {
    evaluate: F,
    natoms: usize,
    kind: CellCoordinates,
    orig: Mat3,
    cell_factor: f64,
    exp_cell_factor: f64,
    pressure: f64,
    /// `1` where that entry of `F` is free, `0` where it stays at `I`.
    mask: [[f64; 3]; 3],
    masked: bool,
    hydrostatic: bool,
    bounds: Bounds<f64>,
    evaluations: AtomicU64,
}

impl<F> CellFilter<F>
where
    F: Fn(&[f64], &[f64; 9]) -> CellSample + Send + Sync,
{
    /// Frechet cell coordinates. Both scale factors start at `natoms`.
    ///
    /// `orig_cell` is row-major and its rows are the reference lattice
    /// vectors. The determinant must be positive.
    pub fn frechet(
        natoms: usize,
        orig_cell: [f64; 9],
        evaluate: F,
    ) -> Result<Self, CellFilterError> {
        Self::build(natoms, orig_cell, CellCoordinates::Frechet, evaluate)
    }

    /// Deformation-gradient coordinates. `cell_factor` starts at `natoms`.
    pub fn deformation(
        natoms: usize,
        orig_cell: [f64; 9],
        evaluate: F,
    ) -> Result<Self, CellFilterError> {
        Self::build(natoms, orig_cell, CellCoordinates::Deformation, evaluate)
    }

    fn build(
        natoms: usize,
        orig_cell: [f64; 9],
        kind: CellCoordinates,
        evaluate: F,
    ) -> Result<Self, CellFilterError> {
        if natoms == 0 {
            return Err(CellFilterError::Empty);
        }
        let orig = Mat3::from_array(orig_cell);
        if !orig.is_finite() {
            return Err(CellFilterError::SingularCell);
        }
        let det = orig.det();
        if !det.is_finite() || det == 0.0 || orig.inverse().is_none() {
            return Err(CellFilterError::SingularCell);
        }
        if det < 0.0 {
            return Err(CellFilterError::Orientation);
        }
        let dim = 3 * (natoms + 3);
        let scale = natoms as f64;
        Ok(Self {
            evaluate,
            natoms,
            kind,
            orig,
            cell_factor: scale,
            exp_cell_factor: scale,
            pressure: 0.0,
            mask: [[1.0; 3]; 3],
            masked: false,
            hydrostatic: false,
            bounds: Bounds::new(
                Array1::from_elem(dim, -1.0e20),
                Array1::from_elem(dim, 1.0e20),
                0.0,
            ),
            evaluations: AtomicU64::new(0),
        })
    }

    /// Replace the factor that multiplies `F` in both coordinate maps.
    ///
    /// ASE's default is the number of atoms. The value must be finite
    /// and positive. On the Frechet map the stored block is
    /// `exp_cell_factor * log(cell_factor * F)`.
    pub fn with_cell_factor(mut self, cell_factor: f64) -> Result<Self, CellFilterError> {
        positive_factor("cell_factor", cell_factor)?;
        self.cell_factor = cell_factor;
        Ok(self)
    }

    /// Replace the factor in front of the matrix logarithm.
    ///
    /// Ignored by [`CellCoordinates::Deformation`]. The value must be
    /// finite and positive. ASE's default is the number of atoms.
    pub fn with_exp_cell_factor(mut self, exp_cell_factor: f64) -> Result<Self, CellFilterError> {
        positive_factor("exp_cell_factor", exp_cell_factor)?;
        self.exp_cell_factor = exp_cell_factor;
        Ok(self)
    }

    /// External pressure `P` in the enthalpy `E + P V`.
    ///
    /// The same `P` is added to the diagonal of the Cauchy stress before
    /// the virial `W = -V (stress + P I)` is formed. A stationary cell
    /// therefore has `stress = -P I`. `P` is in energy per volume and
    /// must be finite. The default is zero.
    pub fn with_scalar_pressure(mut self, pressure: f64) -> Result<Self, CellFilterError> {
        if !pressure.is_finite() {
            return Err(CellFilterError::Pressure { value: pressure });
        }
        self.pressure = pressure;
        Ok(self)
    }

    /// Fix Voigt components of the deformation gradient at the identity.
    ///
    /// The six flags are `xx, yy, zz, yz, xz, xy`, the order of ASE's
    /// strain mask. `false` holds that entry of `F` at the identity.
    /// The cell is `orig_cell F^T`, so a free entry of `F` can move a
    /// masked Cartesian entry when the reference cell is skewed.
    /// `true` leaves the entry free. The default is all free.
    #[must_use]
    pub fn with_mask(mut self, mask: [bool; 6]) -> Self {
        let v = mask.map(|free| if free { 1.0 } else { 0.0 });
        self.mask = [[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]];
        self.masked = mask.iter().any(|free| !free);
        self
    }

    /// Keep the cell a scalar multiple of the reference cell.
    ///
    /// The proposed deformation is replaced by `(tr F / 3) I` before the
    /// mask is applied. Off-diagonal entries of `F` stay at zero and the
    /// diagonal entries of `F` stay equal. The mask can still pin a
    /// diagonal entry.
    #[must_use]
    pub fn with_hydrostatic_strain(mut self, enabled: bool) -> Self {
        self.hydrostatic = enabled;
        self
    }

    /// Number of atoms the wrapped objective was built for.
    #[must_use]
    pub fn natoms(&self) -> usize {
        self.natoms
    }

    /// Which cell block this filter stores.
    #[must_use]
    pub fn coordinates_kind(&self) -> CellCoordinates {
        self.kind
    }

    /// Factor multiplying `F` inside the cell block.
    #[must_use]
    pub fn cell_factor(&self) -> f64 {
        self.cell_factor
    }

    /// Factor multiplying the matrix logarithm. Deformation coordinates
    /// do not read it.
    #[must_use]
    pub fn exp_cell_factor(&self) -> f64 {
        self.exp_cell_factor
    }

    /// External pressure in the enthalpy.
    #[must_use]
    pub fn scalar_pressure(&self) -> f64 {
        self.pressure
    }

    /// `true` when only a hydrostatic deformation is admitted.
    #[must_use]
    pub fn hydrostatic_strain(&self) -> bool {
        self.hydrostatic
    }

    /// Voigt mask, `xx, yy, zz, yz, xz, xy`. `true` means free.
    #[must_use]
    pub fn mask(&self) -> [bool; 6] {
        let m = &self.mask;
        [
            m[0][0] > 0.5,
            m[1][1] > 0.5,
            m[2][2] > 0.5,
            m[1][2] > 0.5,
            m[0][2] > 0.5,
            m[0][1] > 0.5,
        ]
    }

    /// Reference cell, row-major.
    #[must_use]
    pub fn reference_cell(&self) -> [f64; 9] {
        self.orig.to_array()
    }

    /// Calls made to the objective since the last reset.
    ///
    /// One call of the value, the gradient, or the fused pair counts once.
    #[must_use]
    pub fn evaluations(&self) -> u64 {
        self.evaluations.load(Ordering::Relaxed)
    }

    /// Set the evaluation counter back to zero.
    pub fn reset_evaluations(&self) {
        self.evaluations.store(0, Ordering::Relaxed);
    }

    /// Generalized coordinates of a Cartesian geometry.
    ///
    /// `positions` has length `3 * natoms`. `cell` is row-major. The
    /// result has length `3 * (natoms + 3)`: reference positions `q_i`,
    /// then the cell block in row-major order. A mask does not alter
    /// this vector. The objective applies the mask when it reads the
    /// vector back.
    pub fn coordinates(
        &self,
        positions: &[f64],
        cell: &[f64; 9],
    ) -> Result<Array1<f64>, CellFilterError> {
        if positions.len() != 3 * self.natoms {
            return Err(CellFilterError::Dim {
                got: positions.len(),
                dim: 3 * self.natoms,
            });
        }
        if !finite_slice(positions) {
            return Err(CellFilterError::NonFinite);
        }
        let cell_m = Mat3::from_array(*cell);
        if !cell_m.is_finite() {
            return Err(CellFilterError::NonFinite);
        }
        let deform =
            deformation_gradient(self.orig, cell_m).ok_or(CellFilterError::SingularCell)?;
        let q = reference_positions(positions, deform).ok_or(CellFilterError::SingularCell)?;
        let block = self.cell_block(deform)?;
        let mut out = Vec::with_capacity(3 * (self.natoms + 3));
        out.extend_from_slice(&q);
        for row in 0..3 {
            for col in 0..3 {
                out.push(block.m[row][col]);
            }
        }
        Ok(Array1::from(out))
    }

    /// Cartesian positions and row-major cell of a generalized vector.
    ///
    /// Hydrostatic strain and the mask are applied to `F`. A masked
    /// entry of `F` stays at the identity. A Cartesian entry of a skewed
    /// cell can still move. A vector of the wrong length is
    /// [`CellFilterError::Dim`]. A vector that does not become a cell of
    /// positive volume is [`CellFilterError::Decode`].
    pub fn geometry(&self, x: ArrayView1<f64>) -> Result<(Vec<f64>, [f64; 9]), CellFilterError> {
        let dim = 3 * (self.natoms + 3);
        if x.len() != dim {
            return Err(CellFilterError::Dim { got: x.len(), dim });
        }
        let decoded = self.decode(x)?;
        Ok((decoded.positions, decoded.cell.to_array()))
    }

    fn cell_block(&self, deform: Mat3) -> Result<Mat3, CellFilterError> {
        let scaled = deform.scale(self.cell_factor);
        match self.kind {
            CellCoordinates::Deformation => Ok(scaled),
            CellCoordinates::Frechet => logm(scaled)
                .map(|lg| lg.scale(self.exp_cell_factor))
                .ok_or(CellFilterError::Logarithm),
        }
    }

    fn decode(&self, x: ArrayView1<f64>) -> Result<Decoded, CellFilterError> {
        let dim = 3 * (self.natoms + 3);
        if x.len() != dim {
            return Err(CellFilterError::Dim { got: x.len(), dim });
        }
        // Logical order, so a strided view is readable.
        let coords = x.to_vec();
        if !finite_slice(&coords) {
            return Err(CellFilterError::NonFinite);
        }
        let q = coords[..3 * self.natoms].to_vec();
        let mut block = Mat3::zero();
        let base = 3 * self.natoms;
        for row in 0..3 {
            for col in 0..3 {
                block.m[row][col] = coords[base + 3 * row + col];
            }
        }
        let (proposed, exponent) = match self.kind {
            CellCoordinates::Deformation => (block.scale(1.0 / self.cell_factor), Mat3::zero()),
            CellCoordinates::Frechet => {
                let exponent = block.scale(1.0 / self.exp_cell_factor);
                let grown = expm3(exponent);
                (grown.scale(1.0 / self.cell_factor), exponent)
            }
        };
        if !proposed.is_finite() {
            return Err(CellFilterError::Decode);
        }
        let used = self.project(proposed);
        if !used.is_finite() || used.det() <= 0.0 {
            return Err(CellFilterError::Decode);
        }
        let positions = cartesian_positions(&q, used);
        let cell = self.orig.mul(used.transpose());
        if cell.det() <= 0.0 || !cell.is_finite() || !finite_slice(&positions) {
            return Err(CellFilterError::Decode);
        }
        Ok(Decoded {
            positions,
            cell,
            used,
            exponent,
        })
    }

    fn project(&self, proposed: Mat3) -> Mat3 {
        let mut out = proposed;
        if self.hydrostatic {
            let lambda = (proposed.m[0][0] + proposed.m[1][1] + proposed.m[2][2]) / 3.0;
            out = Mat3::eye().scale(lambda);
        }
        if self.masked {
            for row in 0..3 {
                for col in 0..3 {
                    let weight = self.mask[row][col];
                    let reference = if row == col { 1.0 } else { 0.0 };
                    out.m[row][col] = (1.0 - weight) * reference + weight * out.m[row][col];
                }
            }
        }
        out
    }

    /// Pull a force on the projected `F` back to the proposed `F`.
    fn pull_force(&self, mut force: Mat3) -> Mat3 {
        if self.masked {
            for row in 0..3 {
                for col in 0..3 {
                    force.m[row][col] *= self.mask[row][col];
                }
            }
        }
        if self.hydrostatic {
            let trace = force.m[0][0] + force.m[1][1] + force.m[2][2];
            force = Mat3::eye().scale(trace / 3.0);
        }
        force
    }

    fn sample(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        let dim = 3 * (self.natoms + 3);
        let decoded = match self.decode(x) {
            Ok(decoded) => decoded,
            Err(_) => return (f64::NAN, Array1::from_elem(dim, f64::NAN)),
        };
        let sample = (self.evaluate)(&decoded.positions, &decoded.cell.to_array());
        let Some(gradient) = self.gradient_of(&decoded, &sample) else {
            return (f64::NAN, Array1::from_elem(dim, f64::NAN));
        };
        (sample.energy + self.pressure * decoded.cell.det(), gradient)
    }

    fn gradient_of(&self, decoded: &Decoded, sample: &CellSample) -> Option<Array1<f64>> {
        if sample.forces.len() != 3 * self.natoms
            || !sample.energy.is_finite()
            || !finite_slice(&sample.forces)
            || !finite_slice(&sample.stress)
        {
            return None;
        }
        let volume = decoded.cell.det();
        let mut stress = Mat3::from_array(sample.stress);
        for diag in 0..3 {
            stress.m[diag][diag] += self.pressure;
        }
        let virial = stress.scale(-volume);
        let force_on_used = virial.mul(decoded.used.transpose().inverse()?);
        let force_on_proposed = self.pull_force(force_on_used);
        let mut gradient = vec![0.0; 3 * (self.natoms + 3)];
        for atom in 0..self.natoms {
            for axis in 0..3 {
                let mut pulled = 0.0;
                for source in 0..3 {
                    pulled += sample.forces[3 * atom + source] * decoded.used.m[source][axis];
                }
                gradient[3 * atom + axis] = -pulled;
            }
        }
        let cell_force = match self.kind {
            CellCoordinates::Deformation => force_on_proposed.scale(1.0 / self.cell_factor),
            CellCoordinates::Frechet => frechet_force(
                decoded.exponent,
                force_on_proposed,
                self.cell_factor,
                self.exp_cell_factor,
            ),
        };
        let base = 3 * self.natoms;
        for row in 0..3 {
            for col in 0..3 {
                gradient[base + 3 * row + col] = -cell_force.m[row][col];
            }
        }
        if !finite_slice(&gradient) {
            return None;
        }
        Some(Array1::from(gradient))
    }
}

fn positive_factor(name: &'static str, value: f64) -> Result<(), CellFilterError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(CellFilterError::Factor { name, value })
    }
}

fn finite_slice(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

struct Decoded {
    positions: Vec<f64>,
    cell: Mat3,
    used: Mat3,
    exponent: Mat3,
}

impl<F> Objective<f64> for CellFilter<F>
where
    F: Fn(&[f64], &[f64; 9]) -> CellSample + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.sample(x).0
    }
}

impl<F> Gradient<f64> for CellFilter<F>
where
    F: Fn(&[f64], &[f64; 9]) -> CellSample + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.sample(x).1
    }
}

impl<F> DifferentiableObjective<f64> for CellFilter<F>
where
    F: Fn(&[f64], &[f64; 9]) -> CellSample + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.sample(x)
    }
}

fn deformation_gradient(orig: Mat3, cell: Mat3) -> Option<Mat3> {
    // `solve(orig, cell)` is `F^T` in the ASE convention.
    let solved = orig.inverse()?.mul(cell);
    Some(solved.transpose())
}

fn reference_positions(cartesian: &[f64], deform: Mat3) -> Option<Vec<f64>> {
    let inverse = deform.inverse()?;
    Some(cartesian_positions(cartesian, inverse))
}

fn cartesian_positions(reference: &[f64], deform: Mat3) -> Vec<f64> {
    let mut out = vec![0.0; reference.len()];
    let natoms = reference.len() / 3;
    for atom in 0..natoms {
        for axis in 0..3 {
            let mut acc = 0.0;
            for source in 0..3 {
                acc += reference[3 * atom + source] * deform.m[axis][source];
            }
            out[3 * atom + axis] = acc;
        }
    }
    out
}

fn frechet_force(exponent: Mat3, force_on_f: Mat3, cell_factor: f64, exp_cell_factor: f64) -> Mat3 {
    let mut out = Mat3::zero();
    for row in 0..3 {
        for col in 0..3 {
            let mut direction = Mat3::zero();
            direction.m[row][col] = 1.0;
            let derivative = expm_frechet(exponent, direction);
            let mut paired = 0.0;
            for i in 0..3 {
                for j in 0..3 {
                    paired += derivative.m[i][j] * force_on_f.m[i][j];
                }
            }
            out.m[row][col] = paired / cell_factor / exp_cell_factor;
        }
    }
    out
}

#[derive(Clone, Copy)]
struct Mat3 {
    m: [[f64; 3]; 3],
}

impl Mat3 {
    fn zero() -> Self {
        Self { m: [[0.0; 3]; 3] }
    }

    fn eye() -> Self {
        let mut out = Self::zero();
        out.m[0][0] = 1.0;
        out.m[1][1] = 1.0;
        out.m[2][2] = 1.0;
        out
    }

    fn from_array(values: [f64; 9]) -> Self {
        Self {
            m: [
                [values[0], values[1], values[2]],
                [values[3], values[4], values[5]],
                [values[6], values[7], values[8]],
            ],
        }
    }

    fn to_array(self) -> [f64; 9] {
        let m = self.m;
        [
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        ]
    }

    fn scale(self, factor: f64) -> Self {
        let mut out = self;
        for row in &mut out.m {
            for value in row {
                *value *= factor;
            }
        }
        out
    }

    fn add(self, other: Self) -> Self {
        let mut out = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                out.m[i][j] = self.m[i][j] + other.m[i][j];
            }
        }
        out
    }

    fn sub(self, other: Self) -> Self {
        self.add(other.scale(-1.0))
    }

    fn mul(self, other: Self) -> Self {
        let mut out = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                out.m[i][j] = self.m[i][0] * other.m[0][j]
                    + self.m[i][1] * other.m[1][j]
                    + self.m[i][2] * other.m[2][j];
            }
        }
        out
    }

    fn transpose(self) -> Self {
        let mut out = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                out.m[i][j] = self.m[j][i];
            }
        }
        out
    }

    fn det(self) -> f64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    fn inverse(self) -> Option<Self> {
        let det = self.det();
        let scale = self.frobenius();
        if !det.is_finite() || det.abs() <= 1.0e-14 * scale.powi(3).max(1.0e-300) {
            return None;
        }
        let inv_det = 1.0 / det;
        let m = &self.m;
        let out = Self {
            m: [
                [
                    (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                    (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                    (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
                ],
                [
                    (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                    (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                    (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
                ],
                [
                    (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                    (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                    (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
                ],
            ],
        };
        Some(out)
    }

    fn frobenius(self) -> f64 {
        let mut sum = 0.0;
        for row in self.m {
            for value in row {
                sum += value * value;
            }
        }
        libm::sqrt(sum)
    }

    fn is_finite(self) -> bool {
        self.m.iter().flatten().all(|value| value.is_finite())
    }
}

fn expm3(matrix: Mat3) -> Mat3 {
    Mat3 { m: expm(matrix.m) }
}

fn expm<const N: usize>(matrix: [[f64; N]; N]) -> [[f64; N]; N] {
    let mut scaled = matrix;
    let mut norm = inf_norm(&scaled);
    if !norm.is_finite() {
        return [[f64::NAN; N]; N];
    }
    let mut squares = 0u32;
    while norm > 0.25 && squares < 64 {
        scaled = scale_n(&scaled, 0.5);
        norm *= 0.5;
        squares += 1;
    }
    let mut term = eye::<N>();
    let mut sum = eye::<N>();
    for k in 1u32..=40 {
        term = mul_n(&term, &scaled);
        term = scale_n(&term, 1.0 / f64::from(k));
        sum = add_n(&sum, &term);
    }
    for _ in 0..squares {
        sum = mul_n(&sum, &sum);
    }
    sum
}

fn expm_frechet(exponent: Mat3, direction: Mat3) -> Mat3 {
    let mut block = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            block[i][j] = exponent.m[i][j];
            block[i + 3][j + 3] = exponent.m[i][j];
            block[i][j + 3] = direction.m[i][j];
        }
    }
    let grown = expm(block);
    let mut out = Mat3::zero();
    for i in 0..3 {
        for j in 0..3 {
            out.m[i][j] = grown[i][j + 3];
        }
    }
    out
}

fn eye<const N: usize>() -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for i in 0..N {
        out[i][i] = 1.0;
    }
    out
}

fn scale_n<const N: usize>(matrix: &[[f64; N]; N], factor: f64) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            out[i][j] = matrix[i][j] * factor;
        }
    }
    out
}

fn add_n<const N: usize>(left: &[[f64; N]; N], right: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            out[i][j] = left[i][j] + right[i][j];
        }
    }
    out
}

fn mul_n<const N: usize>(left: &[[f64; N]; N], right: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += left[i][k] * right[k][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

fn inf_norm<const N: usize>(matrix: &[[f64; N]; N]) -> f64 {
    let mut max = 0.0;
    for row in matrix {
        let mut sum = 0.0;
        for value in row {
            sum += value.abs();
        }
        if sum > max {
            max = sum;
        }
    }
    max
}

fn logm(matrix: Mat3) -> Option<Mat3> {
    if !matrix.is_finite() {
        return None;
    }
    if let Some(scalar) = near_scalar(matrix) {
        if scalar <= 0.0 {
            return None;
        }
        return Some(Mat3::eye().scale(libm::log(scalar)));
    }
    if is_symmetric(matrix)
        && let Some(logged) = logm_jacobi(matrix)
    {
        return Some(logged);
    }
    logm_series(matrix)
}

fn near_scalar(matrix: Mat3) -> Option<f64> {
    let mean = (matrix.m[0][0] + matrix.m[1][1] + matrix.m[2][2]) / 3.0;
    let mut error = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let target = if i == j { mean } else { 0.0 };
            let delta = matrix.m[i][j] - target;
            error += delta * delta;
        }
    }
    if libm::sqrt(error) <= 1.0e-12 * (mean.abs() + 1.0) {
        Some(mean)
    } else {
        None
    }
}

fn is_symmetric(matrix: Mat3) -> bool {
    let skew = matrix.sub(matrix.transpose()).frobenius();
    skew <= 1.0e-12 * matrix.frobenius().max(1.0)
}

fn logm_jacobi(matrix: Mat3) -> Option<Mat3> {
    let (values, vectors) = jacobi(matrix);
    if values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    if values.iter().all(|value| *value > 0.0) {
        let mut weighted = Mat3::zero();
        for col in 0..3 {
            let logarithm = libm::log(values[col]);
            for row in 0..3 {
                weighted.m[col][row] = logarithm * vectors.m[row][col];
            }
        }
        return accept_logarithm(matrix, vectors.mul(weighted));
    }
    // A half-turn has a repeated negative eigenvalue. Its real logarithm
    // is a rotation by pi in that plane, not the diagonal of the logs.
    let logged = real_log_half_turn(values, vectors)?;
    accept_logarithm(matrix, logged)
}

/// Real logarithm when two eigenvalues are equal and negative and the
/// third is positive. `geometry` produces this branch from `skew(pi)`.
fn real_log_half_turn(values: [f64; 3], vectors: Mat3) -> Option<Mat3> {
    let mut negative = [0usize; 2];
    let mut n_neg = 0usize;
    let mut positive = None;
    for index in 0..3 {
        let value = values[index];
        if value > 0.0 {
            if positive.is_some() {
                return None;
            }
            positive = Some(index);
        } else if value < 0.0 {
            if n_neg == 2 {
                return None;
            }
            negative[n_neg] = index;
            n_neg += 1;
        } else {
            return None;
        }
    }
    if n_neg != 2 {
        return None;
    }
    let i = negative[0];
    let j = negative[1];
    let p = positive?;
    let scale = values[i].abs().max(values[j].abs()).max(1.0);
    if (values[i] - values[j]).abs() > 1.0e-8 * scale {
        return None;
    }
    let magnitude = -0.5 * (values[i] + values[j]);
    if magnitude <= 0.0 {
        return None;
    }
    let log_mag = libm::log(magnitude);
    let log_pos = libm::log(values[p]);
    if !log_mag.is_finite() || !log_pos.is_finite() {
        return None;
    }
    let mut ell = Mat3::zero();
    ell.m[i][i] = log_mag;
    ell.m[j][j] = log_mag;
    ell.m[p][p] = log_pos;
    let pi = std::f64::consts::PI;
    // Both signs of pi exponentiate to the same half-turn.
    ell.m[i][j] = -pi;
    ell.m[j][i] = pi;
    Some(vectors.mul(ell.mul(vectors.transpose())))
}

fn logm_series(matrix: Mat3) -> Option<Mat3> {
    let det = matrix.det();
    if !det.is_finite() || det <= 0.0 {
        return None;
    }
    let mu = libm::cbrt(det);
    if !mu.is_finite() || mu <= 0.0 {
        return None;
    }
    let mut reduced = matrix.scale(1.0 / mu);
    let mut roots = 0u32;
    for _ in 0..40 {
        if reduced.sub(Mat3::eye()).frobenius() <= 0.25 {
            break;
        }
        reduced = sqrtm(reduced)?;
        roots += 1;
    }
    if reduced.sub(Mat3::eye()).frobenius() > 0.5 {
        return None;
    }
    let series = log_series(reduced.sub(Mat3::eye()));
    let logged = series
        .scale(libm::ldexp(1.0, roots as i32))
        .add(Mat3::eye().scale(libm::log(mu)));
    accept_logarithm(matrix, logged)
}

fn accept_logarithm(matrix: Mat3, logged: Mat3) -> Option<Mat3> {
    if !logged.is_finite() {
        return None;
    }
    let back = expm3(logged);
    let error = back.sub(matrix).frobenius();
    if error <= 1.0e-8 * matrix.frobenius().max(1.0) {
        Some(logged)
    } else {
        None
    }
}

fn log_series(increment: Mat3) -> Mat3 {
    let mut power = increment;
    let mut sum = increment;
    for k in 2u32..=48 {
        power = power.mul(increment);
        let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
        let term = power.scale(sign / f64::from(k));
        sum = sum.add(term);
        if term.frobenius() <= 1.0e-18 * sum.frobenius().max(1.0) {
            break;
        }
    }
    sum
}

fn sqrtm(matrix: Mat3) -> Option<Mat3> {
    if let Some(scalar) = near_scalar(matrix) {
        if scalar < 0.0 {
            return None;
        }
        return Some(Mat3::eye().scale(libm::sqrt(scalar)));
    }
    let mut y = matrix;
    let mut z = Mat3::eye();
    for _ in 0..80 {
        let y_next = y.add(z.inverse()?).scale(0.5);
        let z_next = z.add(y.inverse()?).scale(0.5);
        let delta = y_next.sub(y).frobenius();
        y = y_next;
        z = z_next;
        if delta <= 1.0e-14 * y.frobenius().max(1.0) {
            let residual = y.mul(y).sub(matrix).frobenius();
            if residual <= 1.0e-8 * matrix.frobenius().max(1.0) {
                return Some(y);
            }
        }
    }
    None
}

/// Symmetric eigendecomposition. Columns of the returned matrix are the
/// eigenvectors, in the same order as the values.
fn jacobi(matrix: Mat3) -> ([f64; 3], Mat3) {
    let mut diagonal = matrix;
    let mut vectors = Mat3::eye();
    for _ in 0..32 {
        let mut largest = 0.0_f64;
        for p in 0..3 {
            for q in (p + 1)..3 {
                largest = largest.max(diagonal.m[p][q].abs());
            }
        }
        let scale = diagonal.m[0][0].abs() + diagonal.m[1][1].abs() + diagonal.m[2][2].abs() + 1.0;
        if largest <= 1.0e-15 * scale {
            break;
        }
        for p in 0..3 {
            for q in (p + 1)..3 {
                let off = diagonal.m[p][q];
                if off.abs() <= 1.0e-15 * scale {
                    continue;
                }
                let app = diagonal.m[p][p];
                let aqq = diagonal.m[q][q];
                let tau = (aqq - app) / (2.0 * off);
                let root = libm::sqrt(1.0 + tau * tau);
                let t = if tau >= 0.0 {
                    1.0 / (tau + root)
                } else {
                    -1.0 / (-tau + root)
                };
                let c = 1.0 / libm::sqrt(1.0 + t * t);
                let s = t * c;
                diagonal.m[p][p] = app - t * off;
                diagonal.m[q][q] = aqq + t * off;
                diagonal.m[p][q] = 0.0;
                diagonal.m[q][p] = 0.0;
                for k in 0..3 {
                    if k == p || k == q {
                        continue;
                    }
                    let aik = diagonal.m[k][p];
                    let akq = diagonal.m[k][q];
                    let next_p = c * aik - s * akq;
                    let next_q = s * aik + c * akq;
                    diagonal.m[k][p] = next_p;
                    diagonal.m[p][k] = next_p;
                    diagonal.m[k][q] = next_q;
                    diagonal.m[q][k] = next_q;
                }
                for k in 0..3 {
                    let vip = vectors.m[k][p];
                    let viq = vectors.m[k][q];
                    vectors.m[k][p] = c * vip - s * viq;
                    vectors.m[k][q] = s * vip + c * viq;
                }
            }
        }
    }
    (
        [diagonal.m[0][0], diagonal.m[1][1], diagonal.m[2][2]],
        vectors,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_zero(positions: &[f64], _cell: &[f64; 9]) -> CellSample {
        CellSample {
            energy: 0.0,
            forces: vec![0.0; positions.len()],
            stress: [0.0; 9],
        }
    }

    #[test]
    fn the_identity_deformation_is_the_ase_scaled_logarithm() {
        let cell = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let positions = [0.2, -0.1, 0.4, 1.0, 1.5, 2.0, 0.0, 0.3, 0.6, -0.2, 0.4, 0.8];
        let filter = CellFilter::frechet(4, cell, sample_zero).expect("filter");
        let x = filter.coordinates(&positions, &cell).expect("coordinates");
        assert_eq!(x.len(), 21);
        for (got, want) in x.iter().take(12).zip(positions) {
            assert!((got - want).abs() < 1.0e-12);
        }
        let expect = 4.0 * libm::log(4.0);
        for row in 0..3 {
            for col in 0..3 {
                let want = if row == col { expect } else { 0.0 };
                assert!((x[12 + 3 * row + col] - want).abs() < 1.0e-10);
            }
        }
        let (back_pos, back_cell) = filter.geometry(x.view()).expect("geometry");
        for (got, want) in back_pos.iter().zip(positions) {
            assert!((got - want).abs() < 1.0e-10);
        }
        for (got, want) in back_cell.iter().zip(cell) {
            assert!((got - want).abs() < 1.0e-10);
        }
    }

    #[test]
    fn a_scaled_cell_round_trips_through_both_maps() {
        let orig = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let cell = [2.2, 0.05, 0.0, 0.05, 1.9, -0.02, 0.0, -0.02, 2.1];
        let positions = [0.1, 0.2, 0.3, 1.0, 1.1, 0.4, 0.5, 1.4, 1.6, 1.2, 0.3, 0.7];
        for kind in [CellCoordinates::Frechet, CellCoordinates::Deformation] {
            let filter = match kind {
                CellCoordinates::Frechet => CellFilter::frechet(4, orig, sample_zero),
                CellCoordinates::Deformation => CellFilter::deformation(4, orig, sample_zero),
            }
            .expect("filter");
            let x = filter.coordinates(&positions, &cell).expect("coordinates");
            let (back_pos, back_cell) = filter.geometry(x.view()).expect("geometry");
            for (got, want) in back_pos.iter().zip(positions) {
                assert!(
                    (got - want).abs() < 1.0e-8,
                    "{kind:?} position {got} {want}"
                );
            }
            for (got, want) in back_cell.iter().zip(cell) {
                assert!((got - want).abs() < 1.0e-8, "{kind:?} cell {got} {want}");
            }
        }
    }

    #[test]
    fn expm_and_logm_invert_a_small_deformation() {
        let matrix = Mat3 {
            m: [
                [0.12, -0.03, 0.01],
                [0.02, -0.08, 0.04],
                [-0.01, 0.05, 0.07],
            ],
        };
        let grown = expm3(matrix);
        let logged = logm(grown).expect("log");
        assert!(logged.sub(matrix).frobenius() < 1.0e-10);

        let symmetric = Mat3 {
            m: [[1.2, 0.05, 0.0], [0.05, 0.9, -0.02], [0.0, -0.02, 1.1]],
        };
        let logged = logm(symmetric).expect("symmetric log");
        assert!(expm3(logged).sub(symmetric).frobenius() < 1.0e-9);
    }

    #[test]
    fn the_frechet_derivative_matches_a_difference_of_expm() {
        let exponent = Mat3 {
            m: [[0.2, -0.05, 0.01], [0.04, -0.1, 0.02], [0.0, 0.03, 0.08]],
        };
        let mut direction = Mat3::zero();
        direction.m[0][1] = 1.0;
        let analytic = expm_frechet(exponent, direction);
        let step = 1.0e-7;
        let plus = expm3(exponent.add(direction.scale(step)));
        let minus = expm3(exponent.add(direction.scale(-step)));
        let numeric = plus.sub(minus).scale(1.0 / (2.0 * step));
        assert!(analytic.sub(numeric).frobenius() < 1.0e-8);
    }

    #[test]
    fn a_mask_holds_the_fixed_deformation_component() {
        let cell = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let filter = CellFilter::frechet(1, cell, sample_zero)
            .expect("filter")
            .with_mask([false, true, true, false, false, false]);
        let mut x = filter
            .coordinates(&[0.0, 0.0, 0.0], &cell)
            .expect("coordinates");
        // The xx entry of the cell block. A free relaxation would move it.
        x[3] += 0.4;
        let (_positions, relaxed) = filter.geometry(x.view()).expect("geometry");
        assert!((relaxed[0] - 2.0).abs() < 1.0e-12);
        assert!(relaxed[1].abs() < 1.0e-12);
    }

    #[test]
    fn a_masked_entry_of_f_stays_at_the_identity_on_a_skewed_cell() {
        // Rows [2, 0.5, 0], [0.1, 3, 0], [0, 0.2, 4]. cell = orig F^T, so
        // cell_xx = 2 F_xx + 0.5 F_xy. Holding F_xx does not hold cell_xx.
        let orig = [2.0, 0.5, 0.0, 0.1, 3.0, 0.0, 0.0, 0.2, 4.0];
        let filter = CellFilter::deformation(1, orig, sample_zero)
            .expect("filter")
            .with_cell_factor(1.0)
            .expect("factor")
            .with_mask([false, true, true, true, true, true]);
        let mut stored = filter
            .coordinates(&[0.2, -0.1, 0.4], &orig)
            .expect("coordinates");
        stored[3] = 1.4;
        stored[4] = 0.3;
        let (_positions, cell) = filter.geometry(stored.view()).expect("geometry");
        assert!(
            (cell[0] - 2.15).abs() < 1.0e-12,
            "cell_xx {} should move with F_xy",
            cell[0]
        );
        let solved = Mat3::from_array(orig)
            .inverse()
            .expect("orig")
            .mul(Mat3::from_array(cell));
        let deform = solved.transpose();
        assert!(
            (deform.m[0][0] - 1.0).abs() < 1.0e-12,
            "F_xx {}",
            deform.m[0][0]
        );
        assert!(
            (deform.m[0][1] - 0.3).abs() < 1.0e-12,
            "F_xy {}",
            deform.m[0][1]
        );
    }

    #[test]
    fn frechet_coordinates_of_a_reflected_cell_have_no_logarithm() {
        let orig = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let reflected = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, -4.0];
        let filter = CellFilter::frechet(1, orig, sample_zero).expect("filter");
        let err = filter
            .coordinates(&[0.1, 0.2, 0.3], &reflected)
            .expect_err("reflected");
        assert_eq!(err, CellFilterError::Logarithm);
        assert_eq!(
            err.to_string(),
            "the deformation gradient has no principal logarithm"
        );
    }

    #[test]
    fn a_singular_current_cell_is_not_a_missing_logarithm() {
        let orig = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let singular = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for kind in [CellCoordinates::Deformation, CellCoordinates::Frechet] {
            let filter = match kind {
                CellCoordinates::Deformation => CellFilter::deformation(1, orig, sample_zero),
                CellCoordinates::Frechet => CellFilter::frechet(1, orig, sample_zero),
            }
            .expect("filter");
            let err = filter
                .coordinates(&[0.0, 0.0, 0.0], &singular)
                .expect_err("singular");
            assert_eq!(err, CellFilterError::SingularCell, "{kind:?}");
        }
    }

    #[test]
    fn a_left_handed_deformation_does_not_decode() {
        let orig = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let reflected = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, -4.0];
        let filter = CellFilter::deformation(1, orig, sample_zero).expect("filter");
        let stored = filter
            .coordinates(&[0.1, 0.2, 0.3], &reflected)
            .expect("deformation coordinates exist");
        let err = filter.geometry(stored.view()).expect_err("decode");
        assert_eq!(err, CellFilterError::Decode);
        let (value, gradient) = filter.value_and_gradient(stored.view());
        assert!(value.is_nan());
        assert_eq!(gradient.len(), 12);
        assert!(gradient.iter().all(|entry| entry.is_nan()));
    }

    #[test]
    fn geometry_rejects_a_short_vector_as_dim() {
        let cell = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let filter = CellFilter::frechet(1, cell, sample_zero).expect("filter");
        let err = filter.geometry(Array1::zeros(3).view()).expect_err("short");
        assert_eq!(err, CellFilterError::Dim { got: 3, dim: 12 });
    }

    #[test]
    fn a_strided_coordinate_view_decodes() {
        let cell = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let positions = [0.2, -0.1, 0.4];
        let filter = CellFilter::frechet(1, cell, sample_zero).expect("filter");
        let stored = filter.coordinates(&positions, &cell).expect("coordinates");
        let mut paired = Array1::zeros(stored.len() * 2);
        for (index, value) in stored.iter().enumerate() {
            paired[2 * index] = *value;
        }
        let strided = paired.slice(ndarray::s![..; 2]);
        assert!(strided.as_slice().is_none());
        let (back_pos, back_cell) = filter.geometry(strided).expect("strided");
        for (got, want) in back_pos.iter().zip(positions) {
            assert!((got - want).abs() < 1.0e-10);
        }
        for (got, want) in back_cell.iter().zip(cell) {
            assert!((got - want).abs() < 1.0e-10);
        }
        let (value, gradient) = filter.value_and_gradient(strided);
        assert!(value.is_finite());
        assert!(gradient.iter().all(|entry| entry.is_finite()));
    }

    #[test]
    fn a_half_turn_round_trips_through_frechet_coordinates() {
        let orig = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let filter = CellFilter::frechet(1, orig, sample_zero)
            .expect("filter")
            .with_cell_factor(1.0)
            .expect("cell factor")
            .with_exp_cell_factor(1.0)
            .expect("exp factor");
        let pi = std::f64::consts::PI;
        let mut stored = filter
            .coordinates(&[0.2, -0.1, 0.4], &orig)
            .expect("coordinates");
        let skew = [0.0, -pi, 0.0, pi, 0.0, 0.0, 0.0, 0.0, 0.0];
        for (index, value) in skew.into_iter().enumerate() {
            stored[3 + index] = value;
        }
        let (positions, produced) = filter.geometry(stored.view()).expect("geometry");
        assert!((produced[0] + 1.0).abs() < 1.0e-8, "xx {}", produced[0]);
        assert!((produced[4] + 1.0).abs() < 1.0e-8, "yy {}", produced[4]);
        assert!((produced[8] - 1.0).abs() < 1.0e-8, "zz {}", produced[8]);
        let again = filter
            .coordinates(&positions, &produced)
            .expect("half turn has a real logarithm");
        let (back_pos, back_cell) = filter.geometry(again.view()).expect("back");
        for (got, want) in back_pos.iter().zip(&positions) {
            assert!((got - want).abs() < 1.0e-8);
        }
        for (got, want) in back_cell.iter().zip(produced) {
            assert!((got - want).abs() < 1.0e-8, "cell {got} {want}");
        }

        // Half-turn about (1, 1, 0), so the eigenplane is not xy.
        let tilted = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0];
        let positions = [0.2, -0.1, 0.3];
        let stored = filter
            .coordinates(&positions, &tilted)
            .expect("tilted half turn");
        let (back_pos, back_cell) = filter.geometry(stored.view()).expect("tilted geometry");
        for (got, want) in back_pos.iter().zip(positions) {
            assert!((got - want).abs() < 1.0e-8, "position {got} {want}");
        }
        for (got, want) in back_cell.iter().zip(tilted) {
            assert!((got - want).abs() < 1.0e-8, "tilted cell {got} {want}");
        }
    }
}
