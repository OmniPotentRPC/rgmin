//! Variable-cell relaxation on the solid-state NEB coordinate.
//!
//! The chart is the generalized coordinate of the generalized solid-state
//! nudged elastic band (Sheppard, Xiao, and Henkelman, J. Chem. Phys. 136,
//! 074103 (2012), <https://doi.org/10.1063/1.3684549>). Atomic degrees of
//! freedom are Cartesian positions in the reference cell. The cell degree
//! of freedom is the deformation \(F = I + \varepsilon\) from that cell,
//! scaled by the Jacobian
//!
//! \[ J = V_{\mathrm{ref}}^{1/3} N^{1/6}. \]
//!
//! \(J\) has the dimension of length, so a FIRE step and the capped L-BFGS
//! atom cap bound an atom and the cell in the same unit. At the reference
//! cell the generalized force on a scaled strain component is
//! \(-(V/J)(\sigma - \sigma_{\mathrm{applied}})\), with
//! \(\sigma_{ij} = (1/V) \partial E/\partial \delta_{ij}\) the derivative of
//! the energy under an incremental strain at fixed fractional coordinates.
//! Away from the reference the same enthalpy is differentiated through
//! \(F\), which is the chain rule of the ASE `UnitCellFilter` deformation
//! gradient (Tadmor, Smith, Bernstein, and Kaxiras, Phys. Rev. B 59, 235
//! (1999), <https://doi.org/10.1103/PhysRevB.59.235>). A frozen mask drops
//! the cell coordinates and calls the fixed-cell stepper on the Cartesian
//! positions.
//!
//! ASE's `FrechetCellFilter` (ASE 3.23) uses \(U = \log F\) and the Fréchet
//! derivative of the matrix exponential (Al-Mohy and Higham, SIAM J. Matrix
//! Anal. Appl. 30, 1639 (2009), <https://doi.org/10.1137/080716426>). That
//! gradient is exact for large strain and keeps the cell in \(\mathrm{GL}^+(3)\).
//! It is a different chart from the solid-state dimer (Xiao, Sheppard, Rogal,
//! and Henkelman, J. Chem. Phys. 140, 174104 (2014),
//! <https://doi.org/10.1063/1.4873437>), which is the saddle half of this
//! minimizer. Defect relaxations sit near the reference cell, where the two
//! forces agree. This module keeps the dimer chart.
//!
//! Lattice vectors are rows of the 3×3 cell, ASE's layout. A column of
//! Cartesian coordinates transforms as \(r = F q\).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::lbfgs::LbfgsStep;
use crate::method::Method;
use crate::session::Solver;

/// Energy, Cartesian forces, and the strain derivative of one geometry.
///
/// `forces` is \(F = -\partial E/\partial R\) at fixed cell, length `3N`,
/// xyz contiguous.
/// `virial` is row-major \(W_{ij} = \partial E/\partial \delta_{ij}\) at
/// fixed fractional coordinates, where the cell rows transform so that a
/// column lattice vector \(a\) becomes \((I+\delta)a\) and a column
/// position becomes \((I+\delta)r\).
/// The stress is \(\sigma = W / V\). The elastic dipole reported by
/// [`CellMinimum::dipole`] is \(P = W = V\sigma\).
pub trait PeriodicPotential {
    /// Evaluate `positions` (length `3N`) in `cell` (row-major, rows are
    /// the lattice vectors).
    fn evaluate(&mut self, positions: &[f64], cell: &[f64; 9]) -> Result<CellSample>;
}

/// One potential sample.
#[derive(Clone, Debug)]
pub struct CellSample {
    /// Potential energy.
    pub energy: f64,
    /// Cartesian forces, \(F = -\partial E/\partial R\).
    pub forces: Vec<f64>,
    /// Row-major virial \(W = \partial E/\partial \delta = V\sigma\).
    pub virial: [f64; 9],
}

impl CellSample {
    /// \(\sigma = W / V\). `None` when `volume` is not a positive finite
    /// number.
    #[must_use]
    pub fn stress(&self, volume: f64) -> Option<[f64; 9]> {
        if !volume.is_finite() || volume <= 0.0 {
            return None;
        }
        let mut stress = self.virial;
        for value in &mut stress {
            *value /= volume;
        }
        Some(stress)
    }
}

/// Which strain components the cell stepper may change.
///
/// [`StrainMask::Frozen`] stores no cell coordinate. The relaxation is the
/// fixed-cell stepper on the Cartesian positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrainMask {
    /// All nine components of \(\varepsilon\).
    Full,
    /// One isotropic strain, \(\varepsilon = s I\).
    Hydrostatic,
    /// Row-major flags for \(\varepsilon\). `true` is free.
    Epsilon([bool; 9]),
    /// Row-major flags for the cell matrix. `true` means that entry may
    /// differ from the reference. The reference cell has to be diagonal,
    /// which an orthorhombic box is: \(H_{ij}\) then tracks \(\varepsilon_{ji}\).
    Lattice([bool; 9]),
    /// No cell coordinate.
    Frozen,
}

/// How [`relax_cell`] drives the chart.
#[derive(Clone, Debug)]
pub struct CellOptions {
    /// Cell coordinates. [`StrainMask::Full`] if unset through [`Default`].
    pub mask: StrainMask,
    /// External pressure added to the enthalpy as \(P V\). Positive \(P\)
    /// compresses. Equilibrium at the reference cell is \(\sigma = -P I\)
    /// when [`Self::external_stress`] is zero.
    pub pressure: f64,
    /// Dead-load stress in the enthalpy, \(-V_{\mathrm{ref}} \sigma_{\mathrm{ext}}:\varepsilon\).
    /// Row-major. Together with [`Self::pressure`] the applied stress in
    /// the reference-cell force is \(\sigma_{\mathrm{ext}} - P I\).
    pub external_stress: [f64; 9],
    /// Stepper. Any [`Method`] a Euclidean [`Solver`] can run, including
    /// [`Method::Fire`] and capped L-BFGS.
    pub method: Method,
    /// Iteration limit, gradient tolerance, and the FIRE time step.
    pub control: Control,
    /// L-BFGS length rule. `None` keeps the session default, the eOn cap.
    pub lbfgs_step: Option<LbfgsStep>,
}

impl Default for CellOptions {
    fn default() -> Self {
        Self {
            mask: StrainMask::Full,
            pressure: 0.0,
            external_stress: [0.0; 9],
            method: Method::lbfgs(),
            control: Control::default(),
            lbfgs_step: None,
        }
    }
}

/// Fixed-cell minimum. The same record [`StrainMask::Frozen`] returns
/// inside [`CellMinimum`], apart from the cell sample taken at the end.
#[derive(Clone, Debug)]
pub struct FixedMinimum {
    /// Potential energy.
    pub energy: f64,
    /// Cartesian positions.
    pub positions: Vec<f64>,
    /// Accepted stepper iterations.
    pub steps: usize,
    /// Potential evaluations during the solve.
    pub evaluations: usize,
    /// Gradient norm at the accepted point, in [`Control::stop_norm`].
    pub grad_norm: f64,
}

/// Variable-cell minimum, including the stress at the accepted geometry.
#[derive(Clone, Debug)]
pub struct CellMinimum {
    /// Potential energy at the accepted geometry.
    pub energy: f64,
    /// Enthalpy the stepper minimized. A frozen cell reports the potential
    /// energy here as well: \(PV\) does not enter a fixed-cell solve.
    pub enthalpy: f64,
    /// Cartesian positions.
    pub positions: Vec<f64>,
    /// Cell at the minimum, rows are lattice vectors.
    pub cell: [f64; 9],
    /// Strain from the reference cell, \(F = I + \varepsilon\).
    pub strain: [f64; 9],
    /// \(\sigma = W/V\) at the minimum, row-major.
    pub stress: [f64; 9],
    /// Virial \(W = V\sigma\).
    pub virial: [f64; 9],
    /// Cell volume.
    pub volume: f64,
    /// Jacobian of the chart, from the reference volume.
    pub jacobian: f64,
    /// Accepted stepper iterations.
    pub steps: usize,
    /// Potential evaluations, including the sample that fills [`Self::stress`].
    pub evaluations: usize,
    /// Gradient norm at the accepted point, in [`Control::stop_norm`].
    pub grad_norm: f64,
}

impl CellMinimum {
    /// Elastic dipole \(P = V\sigma = W\).
    ///
    /// This is \(\partial E/\partial \delta\) at fixed fractional coordinates.
    /// The dipole whose interaction with a strain is \(-P:\varepsilon\) is
    /// the negative of this tensor.
    #[must_use]
    pub fn dipole(&self) -> [f64; 9] {
        self.virial
    }

    /// \(-\mathrm{tr}(\sigma)/3\). Positive when the derivative stress is
    /// compressive.
    #[must_use]
    pub fn pressure(&self) -> f64 {
        -(self.stress[0] + self.stress[4] + self.stress[8]) / 3.0
    }
}

/// Cartesian positions, cell, and strain of one chart point.
#[derive(Clone, Debug)]
pub struct Geometry {
    /// Cartesian positions \(r = F q\).
    pub positions: Vec<f64>,
    /// Cell \(H = H_0 F^{\mathsf T}\).
    pub cell: [f64; 9],
    /// \(\varepsilon = F - I\).
    pub strain: [f64; 9],
    /// \(\det H\).
    pub volume: f64,
}

/// Value, gradient, and potential sample of one chart point.
#[derive(Clone, Debug)]
pub struct Probe {
    /// Enthalpy \(\Phi\).
    pub enthalpy: f64,
    /// Potential energy.
    pub energy: f64,
    /// \(\partial \Phi/\partial y\), atoms first, then the free scaled strains.
    pub gradient: Vec<f64>,
    /// Potential sample at the Cartesian geometry.
    pub sample: CellSample,
    /// Geometry of this chart point.
    pub geometry: Geometry,
}

impl Probe {
    /// Generalized cell force, the negative of the cell block of
    /// [`Probe::gradient`]. At the reference cell this is
    /// \(-(V/J)(\sigma - \sigma_{\mathrm{applied}})\).
    #[must_use]
    pub fn cell_force(&self) -> Vec<f64> {
        let n3 = self.geometry.positions.len();
        self.gradient[n3..]
            .iter()
            .map(|component| -component)
            .collect()
    }
}

/// Reference cell, Jacobian, and the free strain components.
#[derive(Clone, Debug)]
pub struct Chart {
    reference: [f64; 9],
    volume0: f64,
    jacobian: f64,
    n_atoms: usize,
    pressure: f64,
    external_stress: [f64; 9],
    kind: Free,
}

#[derive(Clone, Debug)]
enum Free {
    Frozen,
    Hydrostatic,
    /// Row-major indices into \(\varepsilon\) that the chart stores.
    Components(Vec<usize>),
}

impl Chart {
    /// Chart for `n_atoms` in `reference`. `reference` is the zero of
    /// \(\varepsilon\), and \(J\) is fixed from its volume.
    pub fn new(n_atoms: usize, reference: &[f64; 9], options: &CellOptions) -> Result<Self> {
        if n_atoms == 0 {
            return Err(Error::Cell {
                what: "the cell has no atoms",
            });
        }
        if !reference.iter().all(|value| value.is_finite()) {
            return Err(Error::Cell {
                what: "the reference cell is not finite",
            });
        }
        if !options.pressure.is_finite() || !options.external_stress.iter().all(|v| v.is_finite()) {
            return Err(Error::Cell {
                what: "the external load is not finite",
            });
        }
        let volume0 = det(reference);
        if !volume0.is_finite() || volume0 <= 0.0 {
            return Err(Error::Cell {
                what: "the reference cell is not right-handed",
            });
        }
        let jacobian = cell_jacobian(volume0, n_atoms);
        if !jacobian.is_finite() || jacobian <= 0.0 {
            return Err(Error::Cell {
                what: "the Jacobian is not positive",
            });
        }
        let kind = free_kind(reference, options.mask)?;
        Ok(Self {
            reference: *reference,
            volume0,
            jacobian,
            n_atoms,
            pressure: options.pressure,
            external_stress: options.external_stress,
            kind,
        })
    }

    /// Number of atoms.
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Length of the generalized coordinate.
    #[must_use]
    pub fn dimension(&self) -> usize {
        3 * self.n_atoms + self.n_cell()
    }

    /// Jacobian \(J = V_{\mathrm{ref}}^{1/3} N^{1/6}\).
    #[must_use]
    pub fn jacobian(&self) -> f64 {
        self.jacobian
    }

    /// Reference volume.
    #[must_use]
    pub fn reference_volume(&self) -> f64 {
        self.volume0
    }

    /// `true` when the chart stores no cell coordinate.
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        matches!(self.kind, Free::Frozen)
    }

    fn n_cell(&self) -> usize {
        match &self.kind {
            Free::Frozen => 0,
            Free::Hydrostatic => 1,
            Free::Components(index) => index.len(),
        }
    }

    /// Generalized coordinate of `positions` at zero strain. `positions`
    /// are the Cartesian coordinates, and at zero strain they are \(q\).
    pub fn origin(&self, positions: &[f64]) -> Result<Vec<f64>> {
        self.require_positions(positions)?;
        let mut y = vec![0.0; self.dimension()];
        y[..positions.len()].copy_from_slice(positions);
        Ok(y)
    }

    /// Enthalpy and its gradient at `y`.
    pub fn evaluate<P: PeriodicPotential + ?Sized>(
        &self,
        potential: &mut P,
        y: &[f64],
    ) -> Result<Probe> {
        if y.len() != self.dimension() {
            return Err(Error::Cell {
                what: "the coordinate length does not match the chart",
            });
        }
        let laid = self.layout(y)?;
        let mut scratch = Vec::new();
        let positions = self.cartesian(y, &laid.deformation, &mut scratch);
        let sample = potential.evaluate(positions, &laid.cell)?;
        self.read_sample(&sample)?;
        let gradient = self.gradient_of(&laid, &sample);
        let enthalpy = self.enthalpy(sample.energy, laid.volume, &laid.strain);
        let owned = positions.to_vec();
        Ok(Probe {
            enthalpy,
            energy: sample.energy,
            gradient,
            sample,
            geometry: Geometry {
                positions: owned,
                cell: laid.cell,
                strain: laid.strain,
                volume: laid.volume,
            },
        })
    }

    fn require_positions(&self, positions: &[f64]) -> Result<()> {
        if positions.len() != 3 * self.n_atoms {
            return Err(Error::Cell {
                what: "positions must have length 3N",
            });
        }
        if positions.iter().any(|value| !value.is_finite()) {
            return Err(Error::Cell {
                what: "positions are not finite",
            });
        }
        Ok(())
    }

    fn layout(&self, y: &[f64]) -> Result<Layout> {
        let strain = self.strain_of(y);
        let mut deformation = strain;
        deformation[0] += 1.0;
        deformation[4] += 1.0;
        deformation[8] += 1.0;
        let det_f = det(&deformation);
        if !det_f.is_finite() || det_f <= 0.0 {
            return Err(Error::Cell {
                what: "the deformation is not right-handed",
            });
        }
        let cell = mul_right_transpose(&self.reference, &deformation);
        let volume = det(&cell);
        if !volume.is_finite() || volume <= 0.0 {
            return Err(Error::Cell {
                what: "the deformation is not right-handed",
            });
        }
        Ok(Layout {
            strain,
            deformation,
            cell,
            volume,
        })
    }

    /// Cartesian positions for `y`. Zero strain returns the coordinate
    /// prefix, which is already those positions.
    fn cartesian<'a>(
        &self,
        y: &'a [f64],
        deformation: &[f64; 9],
        scratch: &'a mut Vec<f64>,
    ) -> &'a [f64] {
        let n3 = 3 * self.n_atoms;
        if is_identity(deformation) {
            return &y[..n3];
        }
        if scratch.len() != n3 {
            scratch.resize(n3, 0.0);
        }
        map_atoms(deformation, &y[..n3], scratch);
        scratch
    }

    fn read_sample(&self, sample: &CellSample) -> Result<()> {
        if sample.forces.len() != 3 * self.n_atoms {
            return Err(Error::Cell {
                what: "potential forces must have length 3N",
            });
        }
        if !sample.energy.is_finite()
            || sample.forces.iter().any(|value| !value.is_finite())
            || sample.virial.iter().any(|value| !value.is_finite())
        {
            return Err(Error::Oracle {
                what: "non-finite value or gradient",
            });
        }
        Ok(())
    }

    fn strain_of(&self, y: &[f64]) -> [f64; 9] {
        let mut strain = [0.0; 9];
        let n3 = 3 * self.n_atoms;
        match &self.kind {
            Free::Frozen => {}
            Free::Hydrostatic => {
                let scale = y[n3] / self.jacobian;
                strain[0] = scale;
                strain[4] = scale;
                strain[8] = scale;
            }
            Free::Components(index) => {
                for (slot, &component) in index.iter().enumerate() {
                    strain[component] = y[n3 + slot] / self.jacobian;
                }
            }
        }
        strain
    }

    fn enthalpy(&self, energy: f64, volume: f64, strain: &[f64; 9]) -> f64 {
        if self.is_frozen() {
            return energy;
        }
        let load: f64 = strain
            .iter()
            .zip(self.external_stress)
            .map(|(strain_ij, stress_ij)| strain_ij * stress_ij)
            .sum();
        energy + self.pressure * volume - self.volume0 * load
    }

    fn gradient_of(&self, laid: &Layout, sample: &CellSample) -> Vec<f64> {
        let n3 = 3 * self.n_atoms;
        let mut gradient = vec![0.0; self.dimension()];
        if is_identity(&laid.deformation) {
            for (slot, force) in sample.forces.iter().enumerate() {
                gradient[slot] = -force;
            }
            if self.is_frozen() {
                return gradient;
            }
        } else {
            let Some(inverse) = inv(&laid.deformation) else {
                return vec![f64::NAN; self.dimension()];
            };
            neg_transpose_times(&laid.deformation, &sample.forces, &mut gradient[..n3]);
            self.cell_block(&mut gradient, laid, sample, &inverse);
            return gradient;
        }
        let inverse = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        self.cell_block(&mut gradient, laid, sample, &inverse);
        gradient
    }

    fn cell_block(
        &self,
        gradient: &mut [f64],
        laid: &Layout,
        sample: &CellSample,
        inverse: &[f64; 9],
    ) {
        let n3 = 3 * self.n_atoms;
        let mut load = sample.virial;
        let pressure_volume = self.pressure * laid.volume;
        load[0] += pressure_volume;
        load[4] += pressure_volume;
        load[8] += pressure_volume;
        let mut cell_gradient = mul_right_inv_transpose(&load, inverse);
        for (component, derivative) in cell_gradient.iter_mut().enumerate() {
            *derivative -= self.volume0 * self.external_stress[component];
        }
        match &self.kind {
            Free::Frozen => {}
            Free::Hydrostatic => {
                gradient[n3] =
                    (cell_gradient[0] + cell_gradient[4] + cell_gradient[8]) / self.jacobian;
            }
            Free::Components(index) => {
                for (slot, &component) in index.iter().enumerate() {
                    gradient[n3 + slot] = cell_gradient[component] / self.jacobian;
                }
            }
        }
    }
}

struct Layout {
    strain: [f64; 9],
    deformation: [f64; 9],
    cell: [f64; 9],
    volume: f64,
}

fn is_identity(deformation: &[f64; 9]) -> bool {
    deformation[0] == 1.0
        && deformation[4] == 1.0
        && deformation[8] == 1.0
        && deformation[1] == 0.0
        && deformation[2] == 0.0
        && deformation[3] == 0.0
        && deformation[5] == 0.0
        && deformation[6] == 0.0
        && deformation[7] == 0.0
}

/// \(J = V^{1/3} N^{1/6}\).
#[must_use]
pub fn cell_jacobian(volume: f64, n_atoms: usize) -> f64 {
    volume.abs().cbrt() * (n_atoms as f64).powf(1.0 / 6.0)
}

/// Determinant of a row-major 3×3 matrix. Positive for a right-handed cell.
#[must_use]
pub fn cell_volume(cell: &[f64; 9]) -> f64 {
    det(cell)
}

/// Applied stress \(\sigma_{\mathrm{ext}} - P I\) that the reference-cell
/// force subtracts from \(\sigma\).
#[must_use]
pub fn applied_stress(pressure: f64, external_stress: &[f64; 9]) -> [f64; 9] {
    let mut stress = *external_stress;
    stress[0] -= pressure;
    stress[4] -= pressure;
    stress[8] -= pressure;
    stress
}

/// Relax `positions` in a fixed `cell`.
///
/// [`relax_cell`] with [`StrainMask::Frozen`] calls this and then samples
/// the stress. The positions, energy, step count, and gradient norm match.
pub fn relax_fixed<P: PeriodicPotential + Send + ?Sized>(
    potential: &mut P,
    positions: &[f64],
    cell: &[f64; 9],
    options: &CellOptions,
) -> Result<FixedMinimum> {
    let mut frozen = options.clone();
    frozen.mask = StrainMask::Frozen;
    frozen.pressure = 0.0;
    frozen.external_stress = [0.0; 9];
    let chart = Chart::new(positions.len() / 3, cell, &frozen)?;
    if positions.len() != 3 * chart.n_atoms() {
        return Err(Error::Cell {
            what: "positions must have length 3N",
        });
    }
    let y0 = chart.origin(positions)?;
    let minimum = solve(potential, &chart, y0, options)?;
    Ok(FixedMinimum {
        energy: minimum.energy,
        positions: minimum.positions,
        steps: minimum.steps,
        evaluations: minimum.evaluations,
        grad_norm: minimum.grad_norm,
    })
}

/// Relax atoms and, unless the mask is frozen, the cell.
///
/// The stepper is [`CellOptions::method`] on the generalized coordinate.
/// FIRE, capped L-BFGS, and the other Euclidean methods all go through
/// [`Solver`]. A frozen mask calls [`relax_fixed`] and does not change
/// the cell.
pub fn relax_cell<P: PeriodicPotential + Send + ?Sized>(
    potential: &mut P,
    positions: &[f64],
    cell: &[f64; 9],
    options: &CellOptions,
) -> Result<CellMinimum> {
    let n_atoms = positions.len() / 3;
    let chart = Chart::new(n_atoms, cell, options)?;
    let y0 = chart.origin(positions)?;
    solve(potential, &chart, y0, options)
}

fn solve<P: PeriodicPotential + Send + ?Sized>(
    potential: &mut P,
    chart: &Chart,
    y0: Vec<f64>,
    options: &CellOptions,
) -> Result<CellMinimum> {
    let objective = CellObjective {
        inner: Mutex::new(Inner {
            potential,
            fault: None,
            cartesian: Vec::new(),
        }),
        chart,
        evaluations: AtomicUsize::new(0),
        bounds: wide_bounds(chart.dimension()),
    };
    let mut y = Array1::from_vec(y0);
    let mut solver = Solver::new(options.method.clone(), options.control.clone(), y.len());
    if let Some(step) = options.lbfgs_step {
        solver.set_lbfgs_step(step)?;
    }
    let mut steps = 0usize;
    let mut grad_norm = f64::INFINITY;
    for _ in 0..options.control.maxiter {
        let report = solver.step(&objective, &mut y)?;
        note_fault(&objective)?;
        steps = report.steps;
        grad_norm = report.grad_norm;
        if grad_norm < options.control.gtol {
            break;
        }
    }
    let coords = y.as_slice().ok_or(Error::Cell {
        what: "the coordinate length does not match the chart",
    })?;
    let probe = {
        let mut inner = objective.lock();
        chart.evaluate(inner.potential, coords)?
    };
    if steps == 0 {
        grad_norm = options
            .control
            .stop_norm
            .of(ArrayView1::from(&probe.gradient));
    }
    Ok(CellMinimum {
        energy: probe.energy,
        enthalpy: probe.enthalpy,
        positions: probe.geometry.positions,
        cell: probe.geometry.cell,
        strain: probe.geometry.strain,
        stress: probe
            .sample
            .stress(probe.geometry.volume)
            .ok_or(Error::Cell {
                what: "the deformation is not right-handed",
            })?,
        virial: probe.sample.virial,
        volume: probe.geometry.volume,
        jacobian: chart.jacobian(),
        steps,
        evaluations: objective.evaluations.load(Ordering::Relaxed) + 1,
        grad_norm,
    })
}

fn note_fault<P: PeriodicPotential + Send + ?Sized>(
    objective: &CellObjective<'_, P>,
) -> Result<()> {
    match objective.lock().fault.take() {
        Some(what) => Err(Error::Cell { what }),
        None => Ok(()),
    }
}

struct Inner<'a, P: ?Sized> {
    potential: &'a mut P,
    fault: Option<&'static str>,
    /// Cartesian image of a nonzero strain. Zero strain does not use it.
    cartesian: Vec<f64>,
}

struct CellObjective<'a, P: ?Sized> {
    inner: Mutex<Inner<'a, P>>,
    chart: &'a Chart,
    evaluations: AtomicUsize,
    bounds: Bounds<f64>,
}

impl<'a, P: ?Sized> CellObjective<'a, P> {
    fn lock(&self) -> MutexGuard<'_, Inner<'a, P>> {
        self.inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl<P: PeriodicPotential + Send + ?Sized> CellObjective<'_, P> {
    fn answer(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        if let Some(coords) = x.as_slice() {
            return self.answer_slice(coords);
        }
        let owned: Vec<f64> = x.iter().copied().collect();
        self.answer_slice(&owned)
    }

    fn answer_slice(&self, coords: &[f64]) -> (f64, Array1<f64>) {
        let chart = self.chart;
        let mut inner = self.lock();
        let laid = match chart.layout(coords) {
            Ok(laid) => laid,
            Err(Error::Cell { what }) if is_refused(what) => {
                return (f64::INFINITY, Array1::zeros(coords.len()));
            }
            Err(Error::Cell { what }) => {
                inner.fault = Some(what);
                return (f64::INFINITY, Array1::zeros(coords.len()));
            }
            Err(_) => {
                inner.fault = Some("potential evaluation failed");
                return (f64::INFINITY, Array1::zeros(coords.len()));
            }
        };
        let sample = {
            let Inner {
                potential,
                cartesian,
                fault: _,
            } = &mut *inner;
            let positions = chart.cartesian(coords, &laid.deformation, cartesian);
            potential.evaluate(positions, &laid.cell)
        };
        let sample = match sample {
            Ok(sample) => sample,
            Err(Error::Cell { what }) => {
                inner.fault = Some(what);
                return (f64::INFINITY, Array1::zeros(coords.len()));
            }
            Err(Error::Oracle { .. }) => {
                return (f64::INFINITY, Array1::from_elem(coords.len(), f64::NAN));
            }
            Err(_) => {
                inner.fault = Some("potential evaluation failed");
                return (f64::INFINITY, Array1::zeros(coords.len()));
            }
        };
        if let Err(err) = chart.read_sample(&sample) {
            return match err {
                Error::Cell { what } => {
                    inner.fault = Some(what);
                    (f64::INFINITY, Array1::zeros(coords.len()))
                }
                Error::Oracle { .. } => (f64::INFINITY, Array1::from_elem(coords.len(), f64::NAN)),
                _ => {
                    inner.fault = Some("potential evaluation failed");
                    (f64::INFINITY, Array1::zeros(coords.len()))
                }
            };
        }
        let gradient = chart.gradient_of(&laid, &sample);
        (
            chart.enthalpy(sample.energy, laid.volume, &laid.strain),
            Array1::from_vec(gradient),
        )
    }
}

impl<P: PeriodicPotential + Send + ?Sized> Objective<f64> for CellObjective<'_, P> {
    fn dim(&self) -> usize {
        self.chart.dimension()
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.answer(x).0
    }
}

impl<P: PeriodicPotential + Send + ?Sized> Gradient<f64> for CellObjective<'_, P> {
    fn dim(&self) -> usize {
        self.chart.dimension()
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.answer(x).1
    }
}

impl<P: PeriodicPotential + Send + ?Sized> DifferentiableObjective<f64> for CellObjective<'_, P> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.answer(x)
    }
}

fn wide_bounds(dim: usize) -> Bounds<f64> {
    const LO: f64 = -1e12;
    const HI: f64 = 1e12;
    Bounds::new(Array1::from_elem(dim, LO), Array1::from_elem(dim, HI), 0.0)
}

fn is_refused(what: &str) -> bool {
    what == "the deformation is not right-handed"
}

fn free_kind(reference: &[f64; 9], mask: StrainMask) -> Result<Free> {
    let flags = match mask {
        StrainMask::Frozen => return Ok(Free::Frozen),
        StrainMask::Hydrostatic => return Ok(Free::Hydrostatic),
        StrainMask::Full => [true; 9],
        StrainMask::Epsilon(flags) => flags,
        StrainMask::Lattice(flags) => {
            if !is_diagonal(reference) {
                return Err(Error::Cell {
                    what: "a lattice-component mask needs a diagonal reference cell",
                });
            }
            let mut epsilon = [false; 9];
            for row in 0..3 {
                for col in 0..3 {
                    epsilon[col * 3 + row] = flags[row * 3 + col];
                }
            }
            epsilon
        }
    };
    let index: Vec<usize> = flags
        .iter()
        .enumerate()
        .filter_map(|(component, free)| free.then_some(component))
        .collect();
    if index.is_empty() {
        Ok(Free::Frozen)
    } else {
        Ok(Free::Components(index))
    }
}

fn is_diagonal(cell: &[f64; 9]) -> bool {
    cell[1] == 0.0
        && cell[2] == 0.0
        && cell[3] == 0.0
        && cell[5] == 0.0
        && cell[6] == 0.0
        && cell[7] == 0.0
}

fn det(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

fn inv(m: &[f64; 9]) -> Option<[f64; 9]> {
    let d = det(m);
    if !d.is_finite() || d.abs() <= f64::MIN_POSITIVE {
        return None;
    }
    let mut c = [0.0; 9];
    c[0] = m[4] * m[8] - m[5] * m[7];
    c[1] = m[2] * m[7] - m[1] * m[8];
    c[2] = m[1] * m[5] - m[2] * m[4];
    c[3] = m[5] * m[6] - m[3] * m[8];
    c[4] = m[0] * m[8] - m[2] * m[6];
    c[5] = m[2] * m[3] - m[0] * m[5];
    c[6] = m[3] * m[7] - m[4] * m[6];
    c[7] = m[1] * m[6] - m[0] * m[7];
    c[8] = m[0] * m[4] - m[1] * m[3];
    for value in &mut c {
        *value /= d;
    }
    Some(c)
}

fn map_atoms(deformation: &[f64; 9], q: &[f64], out: &mut [f64]) {
    for atom in (0..q.len()).step_by(3) {
        let x = q[atom];
        let y = q[atom + 1];
        let z = q[atom + 2];
        out[atom] = deformation[0] * x + deformation[1] * y + deformation[2] * z;
        out[atom + 1] = deformation[3] * x + deformation[4] * y + deformation[5] * z;
        out[atom + 2] = deformation[6] * x + deformation[7] * y + deformation[8] * z;
    }
}

fn neg_transpose_times(deformation: &[f64; 9], force: &[f64], out: &mut [f64]) {
    for atom in (0..force.len()).step_by(3) {
        for row in 0..3 {
            let mut acc = 0.0;
            for col in 0..3 {
                acc += deformation[col * 3 + row] * force[atom + col];
            }
            out[atom + row] = -acc;
        }
    }
}

/// \(H = H_0 F^{\mathsf T}\).
fn mul_right_transpose(left: &[f64; 9], right: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += left[row * 3 + k] * right[col * 3 + k];
            }
            out[row * 3 + col] = acc;
        }
    }
    out
}

/// \(M F^{-\mathsf T}\), with `inverse` holding \(F^{-1}\).
fn mul_right_inv_transpose(left: &[f64; 9], inverse: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += left[row * 3 + k] * inverse[col * 3 + k];
            }
            out[row * 3 + col] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{cell_jacobian, det, inv};

    #[test]
    fn jacobian_scales_volume_and_atom_count() {
        let jacobian = cell_jacobian(8.0, 64);
        assert!((jacobian - 4.0).abs() < 1e-12);
    }

    #[test]
    fn inverse_inverts_a_known_matrix() {
        let matrix = [1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0];
        assert!((det(&matrix) - 1.0).abs() < 1e-12);
        let inverse = inv(&matrix).unwrap();
        let mut product = [0.0; 9];
        for row in 0..3 {
            for col in 0..3 {
                product[row * 3 + col] = (0..3)
                    .map(|k| matrix[row * 3 + k] * inverse[k * 3 + col])
                    .sum();
            }
        }
        for (index, value) in product.iter().enumerate() {
            let expect = if index == 0 || index == 4 || index == 8 {
                1.0
            } else {
                0.0
            };
            assert!((value - expect).abs() < 1e-12);
        }
    }
}
