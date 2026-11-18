//! Constrained Langevin and RATTLE on an embedded manifold.
//!
//! The integrator asks the caller for forces and uses only the manifold's
//! projection, second-order retraction, and vector transport, together
//! with the mass-metric projector. It does not own a potential.
//!
//! One BAOAB step (Leimkuhler and Matthews, Appl. Math. Res. Express 2013,
//! <https://doi.org/10.1093/amrx/abs010>, and the geodesic form in Proc. R.
//! Soc. A 472, 20160138, 2016, <https://doi.org/10.1098/rspa.2016.0138>) is
//!
//! 1. **B** half kick by the force, then the mass-metric projection of the
//!    velocity (RATTLE; Andersen, J. Comput. Phys. 52, 24, 1983,
//!    <https://doi.org/10.1016/0021-9991(83)90014-1>).
//! 2. **A** half drift. Equal positive masses use the embedding geodesic,
//!    which is then the kinetic geodesic. Any other diagonal mass, on a
//!    manifold that publishes a holonomic chart, takes one RATTLE step
//!    along \(M^{-1}\nabla c\) (the \(A_g\) map with one substep). A flat
//!    constraint takes the straight step and restores the point.
//! 3. **O** Ornstein–Uhlenbeck kick with the noise projected onto the
//!    tangent space.
//! 4. **A** and **B** again. The second kick uses the force at the new point.
//!
//! On a flat affine constraint the drift is `x + Δt v` followed by the
//! mass-metric restoration, which is the hyperplane BAOAB step. The
//! invariant of the constrained Langevin equation is the Boltzmann density
//! on the manifold times the Gaussian of the tangent kinetic energy
//! (Lelièvre, Rousset and Stoltz, Math. Comp. 81, 2012,
//! <https://doi.org/10.1090/S0025-5718-2012-02594-4>). BAOAB is built so
//! that a flat or exactly geodesic drift preserves that measure for a
//! quadratic energy; the tests check the marginals on the sphere and on a
//! product of circles.
//!
//! Specular planar walls are an optional extra half-space constraint. On a
//! flat manifold the reflection is the mass-metric law used for the
//! hyperplane: the wall normal is reduced to the tangent space, the
//! position and velocity reverse along `M^{-1}` of that direction, and the
//! point is restored to the manifold. A curved manifold recomputes that
//! tangent at the current point and retracts the reflected step.

use ndarray::Array1;

use crate::error::Error;
use crate::manifold::Manifold;
use crate::manifold::mass::{self, MassMetric};

mod rng;

use rng::SplitMix64;

/// Failure of a geometry query or of the force callback.
#[derive(Debug)]
#[non_exhaustive]
pub enum DynamicsError<E> {
    /// The manifold or the mass metric refused the step.
    Geometry(Error),
    /// The force callback failed.
    Force(E),
}

impl<E: std::fmt::Display> std::fmt::Display for DynamicsError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Geometry(err) => write!(f, "{err}"),
            Self::Force(err) => write!(f, "{err}"),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for DynamicsError<E> {}

fn geometry<E>(err: Error) -> DynamicsError<E> {
    DynamicsError::Geometry(err)
}

/// Point and velocity of one constrained trajectory.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PhasePoint {
    /// Ambient coordinates.
    pub position: Array1<f64>,
    /// Ambient velocity. The mass-metric projector puts it in the tangent space.
    pub velocity: Array1<f64>,
}

impl Default for PhasePoint {
    fn default() -> Self {
        Self {
            position: Array1::zeros(0),
            velocity: Array1::zeros(0),
        }
    }
}

impl PhasePoint {
    /// Set [`Self::position`].
    #[must_use]
    pub fn with_position(mut self, position: Array1<f64>) -> Self {
        self.position = position;
        self
    }

    /// Set [`Self::velocity`].
    #[must_use]
    pub fn with_velocity(mut self, velocity: Array1<f64>) -> Self {
        self.velocity = velocity;
        self
    }
}

/// Controls for constrained BAOAB. Energies and `thermal` share a unit;
/// `kinetic_scale` converts `½ v^T M v` into that unit.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct LangevinConfig {
    /// Thermal energy `kT`.
    pub thermal: f64,
    /// Friction coefficient. Zero leaves the kinetic energy unchanged by O.
    pub friction: f64,
    /// Time step.
    pub time_step: f64,
    /// Seed of the SplitMix64 / Box–Muller stream.
    pub seed: u64,
    /// `kinetic_energy = ½ v^T M v / kinetic_scale`.
    pub kinetic_scale: f64,
}

impl Default for LangevinConfig {
    fn default() -> Self {
        Self {
            thermal: 1.0,
            friction: 1.0,
            time_step: 0.01,
            seed: 1,
            kinetic_scale: 1.0,
        }
    }
}

impl LangevinConfig {
    /// Set [`Self::thermal`].
    #[must_use]
    pub fn with_thermal(mut self, thermal: f64) -> Self {
        self.thermal = thermal;
        self
    }

    /// Set [`Self::friction`].
    #[must_use]
    pub fn with_friction(mut self, friction: f64) -> Self {
        self.friction = friction;
        self
    }

    /// Set [`Self::time_step`].
    #[must_use]
    pub fn with_time_step(mut self, time_step: f64) -> Self {
        self.time_step = time_step;
        self
    }

    /// Set [`Self::seed`].
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set [`Self::kinetic_scale`].
    #[must_use]
    pub fn with_kinetic_scale(mut self, kinetic_scale: f64) -> Self {
        self.kinetic_scale = kinetic_scale;
        self
    }
}

/// Controls for constrained velocity Verlet (RATTLE).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct NveConfig {
    /// Time step.
    pub time_step: f64,
    /// `kinetic_energy = ½ v^T M v / kinetic_scale`.
    pub kinetic_scale: f64,
}

impl Default for NveConfig {
    fn default() -> Self {
        Self {
            time_step: 0.01,
            kinetic_scale: 1.0,
        }
    }
}

impl NveConfig {
    /// Set [`Self::time_step`].
    #[must_use]
    pub fn with_time_step(mut self, time_step: f64) -> Self {
        self.time_step = time_step;
        self
    }

    /// Set [`Self::kinetic_scale`].
    #[must_use]
    pub fn with_kinetic_scale(mut self, kinetic_scale: f64) -> Self {
        self.kinetic_scale = kinetic_scale;
        self
    }
}

/// One specular half-space. The allowed side is `normal · (x - point) <= 0`
/// after `normal` is made a Euclidean unit vector.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PlanarWall {
    /// A point of the wall.
    pub point: Array1<f64>,
    /// Outward normal, stored as given and normalized at reflection time.
    pub normal: Array1<f64>,
}

impl Default for PlanarWall {
    fn default() -> Self {
        Self {
            point: Array1::zeros(1),
            normal: Array1::from_elem(1, 1.0),
        }
    }
}

impl PlanarWall {
    /// `point` and `normal` share a finite nonzero direction.
    pub fn new(point: Array1<f64>, normal: Array1<f64>) -> Result<Self, Error> {
        if point.is_empty() || point.len() != normal.len() {
            return Err(Error::MassMetric {
                what: "a wall point and normal need one positive length",
            });
        }
        if !point.iter().chain(normal.iter()).all(|v| v.is_finite()) {
            return Err(Error::MassMetric {
                what: "a wall point and normal must be finite",
            });
        }
        let norm = normal.iter().map(|v| v * v).sum::<f64>();
        if !(norm.is_finite() && norm > 0.0) {
            return Err(Error::MassMetric {
                what: "a wall normal must be nonzero",
            });
        }
        Ok(Self { point, normal })
    }

    /// Set [`Self::point`].
    #[must_use]
    pub fn with_point(mut self, point: Array1<f64>) -> Self {
        self.point = point;
        self
    }

    /// Set [`Self::normal`].
    #[must_use]
    pub fn with_normal(mut self, normal: Array1<f64>) -> Self {
        self.normal = normal;
        self
    }
}

/// What one integration call left behind.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct DynamicsReport {
    /// Potential returned with the last force.
    pub potential: f64,
    /// Tangent kinetic energy `½ v^T M v / kinetic_scale`.
    pub kinetic: f64,
    /// Lagrange multipliers of the last force, one per normal.
    pub multipliers: Array1<f64>,
    /// Specular wall hits.
    pub wall_hits: u64,
    /// Positive-mass coordinates minus the number of normals.
    pub degrees_of_freedom: usize,
}

impl Default for DynamicsReport {
    fn default() -> Self {
        Self {
            potential: 0.0,
            kinetic: 0.0,
            multipliers: Array1::zeros(0),
            wall_hits: 0,
            degrees_of_freedom: 0,
        }
    }
}

impl DynamicsReport {
    /// Set [`Self::potential`].
    #[must_use]
    pub fn with_potential(mut self, potential: f64) -> Self {
        self.potential = potential;
        self
    }

    /// Set [`Self::kinetic`].
    #[must_use]
    pub fn with_kinetic(mut self, kinetic: f64) -> Self {
        self.kinetic = kinetic;
        self
    }

    /// Set [`Self::multipliers`].
    #[must_use]
    pub fn with_multipliers(mut self, multipliers: Array1<f64>) -> Self {
        self.multipliers = multipliers;
        self
    }

    /// Set [`Self::wall_hits`].
    #[must_use]
    pub fn with_wall_hits(mut self, wall_hits: u64) -> Self {
        self.wall_hits = wall_hits;
        self
    }

    /// Set [`Self::degrees_of_freedom`].
    #[must_use]
    pub fn with_degrees_of_freedom(mut self, degrees_of_freedom: usize) -> Self {
        self.degrees_of_freedom = degrees_of_freedom;
        self
    }
}

const MAX_REFLECTION_PASSES: usize = 8;
const START_CONSTRAINT_TOLERANCE_FACTOR: f64 = 1_024.0;

fn check_langevin<E>(config: &LangevinConfig) -> Result<(), DynamicsError<E>> {
    if !(config.thermal.is_finite() && config.thermal > 0.0) {
        return Err(geometry(Error::MassMetric {
            what: "Langevin kT must be finite and positive",
        }));
    }
    if !(config.friction.is_finite() && config.friction >= 0.0) {
        return Err(geometry(Error::MassMetric {
            what: "Langevin friction must be finite and non-negative",
        }));
    }
    if !(config.time_step.is_finite()
        && config.time_step > 0.0
        && config.kinetic_scale.is_finite()
        && config.kinetic_scale > 0.0)
    {
        return Err(geometry(Error::MassMetric {
            what: "Langevin time step and kinetic scale must be finite and positive",
        }));
    }
    Ok(())
}

fn check_nve<E>(config: &NveConfig) -> Result<(), DynamicsError<E>> {
    if !(config.time_step.is_finite()
        && config.time_step > 0.0
        && config.kinetic_scale.is_finite()
        && config.kinetic_scale > 0.0)
    {
        return Err(geometry(Error::MassMetric {
            what: "RATTLE time step and kinetic scale must be finite and positive",
        }));
    }
    Ok(())
}

fn kinetic(metric: &MassMetric, velocity: &Array1<f64>, scale: f64) -> f64 {
    let mut sum = 0.0;
    for (vi, mi) in velocity.iter().zip(metric.masses().iter()) {
        sum += mi * vi * vi;
    }
    0.5 * sum / scale
}

fn kick(velocity: &mut Array1<f64>, force: &[f64], masses: &[f64], half: f64, scale: f64) {
    for i in 0..velocity.len() {
        if masses[i] > 0.0 {
            velocity[i] += force[i] * (half * scale / masses[i]);
        }
    }
}

fn positive_multiple_of_identity(metric: &MassMetric) -> bool {
    let mut found = None;
    for mass in metric.masses() {
        if *mass > 0.0 {
            match found {
                None => found = Some(*mass),
                Some(value) if value == *mass => {}
                Some(_) => return false,
            }
        } else {
            return false;
        }
    }
    found.is_some()
}

fn check_phase<M, E>(
    manifold: &M,
    metric: &MassMetric,
    position: &Array1<f64>,
    velocity: &Array1<f64>,
) -> Result<(), DynamicsError<E>>
where
    M: Manifold + ?Sized,
{
    for (mass, speed) in metric.masses().iter().zip(velocity.iter()) {
        if *mass == 0.0 && *speed != 0.0 {
            return Err(geometry(Error::Dynamics {
                what: "a zero-mass coordinate must have zero velocity",
            }));
        }
    }
    if !manifold.drift_is_straight()
        && let Some((values, _)) = manifold.holonomic_chart(position)
        && values.iter().any(|value| {
            !value.is_finite()
                || value.abs() > START_CONSTRAINT_TOLERANCE_FACTOR * chart_tolerance(position)
        })
    {
        return Err(geometry(Error::Dynamics {
            what: "the initial point is off the manifold",
        }));
    }
    Ok(())
}

fn embedding_drift<M, E>(
    manifold: &M,
    metric: &MassMetric,
    position: &Array1<f64>,
    velocity: &Array1<f64>,
    step: f64,
) -> Result<(Array1<f64>, Array1<f64>), DynamicsError<E>>
where
    M: Manifold + ?Sized,
{
    let mut tangent = velocity.clone();
    for value in tangent.iter_mut() {
        *value *= step;
    }
    let tangent = manifold.project(position, &tangent);
    let next = manifold.retract_second_order(position, &tangent);
    let carried = manifold.transport_second_order(position, &tangent, &next, velocity);
    let carried = manifold
        .project_mass_velocity(metric, &next, &carried)
        .map_err(geometry)?;
    Ok((next, carried))
}

fn chart_tolerance(position: &Array1<f64>) -> f64 {
    let mut scale = 1.0;
    for value in position {
        if value.abs() > scale {
            scale = value.abs();
        }
    }
    (position.len() as f64) * f64::EPSILON * scale * scale.max(1.0)
}

type KineticStep<E> = Result<Option<(Array1<f64>, Array1<f64>)>, DynamicsError<E>>;

/// One oblique RATTLE step, the paper's \(A_g\) map at \(K_r = 1\).
///
/// `None` means the manifold did not publish a chart. The correction
/// direction \(M^{-1}\nabla c\) is frozen at the start of the step, and
/// Newton drives the chart to its solver tolerance.
fn kinetic_rattle<M, E>(
    manifold: &M,
    metric: &MassMetric,
    position: &Array1<f64>,
    velocity: &Array1<f64>,
    step: f64,
) -> KineticStep<E>
where
    M: Manifold + ?Sized,
{
    let Some((start_values, start_grad)) = manifold.holonomic_chart(position) else {
        return Ok(None);
    };
    let codim = start_grad.ncols();
    if codim == 0 || start_values.len() != codim || start_grad.nrows() != position.len() {
        return Ok(None);
    }
    let masses = metric.masses().as_slice().ok_or_else(|| {
        geometry(Error::MassMetric {
            what: "the mass diagonal is not contiguous",
        })
    })?;
    let n = position.len();
    let mut lambda = vec![0.0; codim];
    let mut position_next = position.clone();
    for i in 0..n {
        position_next[i] += velocity[i] * step;
    }
    let tolerance = chart_tolerance(position);
    let mut satisfied = false;
    for _ in 0..MAX_REFLECTION_PASSES {
        let Some((values, gradient)) = manifold.holonomic_chart(&position_next) else {
            return Err(geometry(Error::MassMetric {
                what: "the holonomic chart disappeared during RATTLE",
            }));
        };
        if values.len() != codim || gradient.ncols() != codim || gradient.nrows() != n {
            return Err(geometry(Error::MassMetric {
                what: "the holonomic chart changed rank during RATTLE",
            }));
        }
        satisfied = values
            .iter()
            .all(|value| value.is_finite() && value.abs() <= tolerance);
        if satisfied {
            break;
        }
        let mut gram = vec![0.0; codim * codim];
        let mut rhs = vec![0.0; codim];
        for a in 0..codim {
            rhs[a] = -values[a];
            for b in 0..codim {
                let mut entry = 0.0;
                for i in 0..n {
                    if masses[i] > 0.0 {
                        entry += gradient[(i, a)] * step * start_grad[(i, b)] / masses[i];
                    }
                }
                gram[a * codim + b] = entry;
            }
        }
        mass::solve_linear(&mut gram, &mut rhs, codim).map_err(geometry)?;
        for (slot, delta) in lambda.iter_mut().zip(rhs.iter()) {
            *slot += *delta;
        }
        if lambda.iter().any(|value| !value.is_finite()) {
            return Err(geometry(Error::MassMetric {
                what: "the RATTLE multiplier is not finite",
            }));
        }
        for i in 0..n {
            let mut corr = 0.0;
            if masses[i] > 0.0 {
                for b in 0..codim {
                    corr += start_grad[(i, b)] * lambda[b];
                }
                corr /= masses[i];
            }
            position_next[i] = position[i] + velocity[i] * step + step * corr;
        }
    }
    if !satisfied && let Some((values, _)) = manifold.holonomic_chart(&position_next) {
        satisfied = values.len() == codim
            && values
                .iter()
                .all(|value| value.is_finite() && value.abs() <= tolerance);
    }
    if !satisfied {
        return Err(geometry(Error::MassMetric {
            what: "the RATTLE step missed the constraint",
        }));
    }
    let mut carried = velocity.clone();
    for i in 0..n {
        carried[i] = (position_next[i] - position[i]) / step;
    }
    let carried = manifold
        .project_mass_velocity(metric, &position_next, &carried)
        .map_err(geometry)?;
    Ok(Some((position_next, carried)))
}

fn drift<M, E>(
    manifold: &M,
    metric: &MassMetric,
    position: &Array1<f64>,
    velocity: &Array1<f64>,
    step: f64,
) -> Result<(Array1<f64>, Array1<f64>), DynamicsError<E>>
where
    M: Manifold + ?Sized,
{
    if manifold.drift_is_straight() {
        let mut next = position.clone();
        for i in 0..next.len() {
            next[i] += velocity[i] * step;
        }
        let next = manifold.snap_mass(metric, &next).map_err(geometry)?;
        Ok((next, velocity.clone()))
    } else if positive_multiple_of_identity(metric) {
        embedding_drift(manifold, metric, position, velocity, step)
    } else if let Some(stepped) = kinetic_rattle(manifold, metric, position, velocity, step)? {
        Ok(stepped)
    } else {
        embedding_drift(manifold, metric, position, velocity, step)
    }
}

struct PreparedWall {
    normal: Vec<f64>,
    point: Vec<f64>,
    tangent_m_inv: Vec<f64>,
    scale: f64,
}

fn prepare_walls<E>(
    normals: &ndarray::Array2<f64>,
    metric: &MassMetric,
    walls: &[PlanarWall],
) -> Result<Vec<PreparedWall>, DynamicsError<E>> {
    if walls.is_empty() {
        return Ok(Vec::new());
    }
    let n = metric.len();
    let masses = metric.masses().as_slice().ok_or_else(|| {
        geometry(Error::MassMetric {
            what: "the mass diagonal is not contiguous",
        })
    })?;
    let codim = normals.ncols();
    if codim == 0 {
        return walls
            .iter()
            .map(|wall| prepare_free_wall::<E>(wall, masses))
            .collect();
    }
    if codim != 1 {
        return Err(geometry(Error::MassMetric {
            what: "specular walls are implemented for one constraint normal",
        }));
    }
    let normal = normals.column(0);
    let mut alpha = 0.0;
    for i in 0..n {
        if masses[i] > 0.0 {
            alpha += normal[i] * normal[i] / masses[i];
        }
    }
    if !(alpha.is_finite() && alpha > 0.0) {
        return Err(geometry(Error::MassMetric {
            what: "the mass-metric gram matrix is singular",
        }));
    }
    let mut prepared = Vec::with_capacity(walls.len());
    for wall in walls {
        if wall.point.len() != n || wall.normal.len() != n {
            return Err(geometry(Error::MassMetric {
                what: "a wall and the masses differ in length",
            }));
        }
        let mut u_sq = 0.0;
        for value in &wall.normal {
            u_sq += value * value;
        }
        let u_norm = u_sq.sqrt();
        if !(u_norm.is_finite() && u_norm > 0.0) {
            return Err(geometry(Error::MassMetric {
                what: "a wall normal must be nonzero",
            }));
        }
        let inv_u = 1.0 / u_norm;
        let mut unit = vec![0.0; n];
        for i in 0..n {
            unit[i] = wall.normal[i] * inv_u;
        }
        let mut n_minv_u = 0.0;
        for i in 0..n {
            if masses[i] > 0.0 {
                n_minv_u += normal[i] * unit[i] / masses[i];
            }
        }
        let coeff = n_minv_u / alpha;
        let mut u_tan = vec![0.0; n];
        let mut scale_sq = 0.0;
        for i in 0..n {
            u_tan[i] = unit[i] - normal[i] * coeff;
            if masses[i] > 0.0 {
                scale_sq += u_tan[i] * u_tan[i] / masses[i];
            }
        }
        let scale = scale_sq.sqrt();
        let mut tangent_m_inv = vec![0.0; n];
        if scale > 0.0 {
            for i in 0..n {
                if masses[i] > 0.0 {
                    tangent_m_inv[i] = u_tan[i] * (1.0 / (scale * masses[i]));
                }
            }
        }
        prepared.push(PreparedWall {
            normal: unit,
            point: wall.point.to_vec(),
            tangent_m_inv,
            scale,
        });
    }
    Ok(prepared)
}

fn prepare_free_wall<E>(
    wall: &PlanarWall,
    masses: &[f64],
) -> Result<PreparedWall, DynamicsError<E>> {
    let n = masses.len();
    if wall.point.len() != n || wall.normal.len() != n {
        return Err(geometry(Error::MassMetric {
            what: "a wall and the masses differ in length",
        }));
    }
    let mut u_sq = 0.0;
    for value in &wall.normal {
        u_sq += value * value;
    }
    let u_norm = u_sq.sqrt();
    if !(u_norm.is_finite() && u_norm > 0.0) {
        return Err(geometry(Error::MassMetric {
            what: "a wall normal must be nonzero",
        }));
    }
    let mut unit = vec![0.0; n];
    let mut scale_sq = 0.0;
    for i in 0..n {
        unit[i] = wall.normal[i] * (1.0 / u_norm);
        if masses[i] > 0.0 {
            scale_sq += unit[i] * unit[i] / masses[i];
        }
    }
    let scale = scale_sq.sqrt();
    let mut tangent_m_inv = vec![0.0; n];
    if scale > 0.0 {
        for i in 0..n {
            if masses[i] > 0.0 {
                tangent_m_inv[i] = unit[i] * (1.0 / (scale * masses[i]));
            }
        }
    }
    Ok(PreparedWall {
        normal: unit,
        point: wall.point.to_vec(),
        tangent_m_inv,
        scale,
    })
}

fn reflect_curved<M, E>(
    manifold: &M,
    metric: &MassMetric,
    walls: &[PlanarWall],
    position: &mut Array1<f64>,
    velocity: &mut Array1<f64>,
) -> Result<u64, DynamicsError<E>>
where
    M: Manifold + ?Sized,
{
    let masses = metric
        .masses()
        .as_slice()
        .ok_or_else(|| {
            geometry(Error::MassMetric {
                what: "the mass diagonal is not contiguous",
            })
        })?
        .to_vec();
    let mut hits = 0u64;
    for pass in 0..=MAX_REFLECTION_PASSES {
        let normals = match manifold.euclidean_normals(position) {
            Some(basis) => basis,
            None => mass::normal_frame(manifold, position).map_err(geometry)?,
        };
        let prepared = prepare_walls(&normals, metric, walls)?;
        let mut any = false;
        for wall in &prepared {
            if wall.scale <= 0.0 {
                continue;
            }
            let mut raw = 0.0;
            for i in 0..position.len() {
                raw += wall.normal[i] * (position[i] - wall.point[i]);
            }
            let excess = raw / wall.scale;
            if excess > 0.0 {
                if pass == MAX_REFLECTION_PASSES {
                    return Err(geometry(Error::Dynamics {
                        what: "curved wall reflections did not converge",
                    }));
                }
                let mut v_dot = 0.0;
                for i in 0..velocity.len() {
                    if masses[i] > 0.0 {
                        v_dot += wall.tangent_m_inv[i] * masses[i] * velocity[i];
                    }
                }
                let mut step = Array1::zeros(position.len());
                let mut reflected = velocity.clone();
                for i in 0..position.len() {
                    let delta = wall.tangent_m_inv[i] * (2.0 * excess);
                    step[i] = -delta;
                    reflected[i] -= wall.tangent_m_inv[i] * (2.0 * v_dot);
                }
                let next = manifold.retract_second_order(position, &step);
                let carried = manifold.transport_second_order(position, &step, &next, &reflected);
                *position = next;
                *velocity = manifold
                    .project_mass_velocity(metric, position, &carried)
                    .map_err(geometry)?;
                any = true;
                hits += 1;
            }
        }
        if !any {
            break;
        }
    }
    Ok(hits)
}

fn reflect_prepared<M, E>(
    manifold: &M,
    metric: &MassMetric,
    prepared: &[PreparedWall],
    walls: &[PlanarWall],
    position: &mut Array1<f64>,
    velocity: &mut Array1<f64>,
) -> Result<u64, DynamicsError<E>>
where
    M: Manifold + ?Sized,
{
    if prepared.is_empty() {
        return Ok(0);
    }
    if !manifold.drift_is_straight() {
        return reflect_curved(manifold, metric, walls, position, velocity);
    }
    let masses = metric.masses().as_slice().ok_or_else(|| {
        geometry(Error::MassMetric {
            what: "the mass diagonal is not contiguous",
        })
    })?;
    let mut hits = 0u64;
    for pass in 0..=MAX_REFLECTION_PASSES {
        let mut any = false;
        for wall in prepared {
            if wall.scale <= 0.0 {
                continue;
            }
            let mut raw = 0.0;
            for i in 0..position.len() {
                raw += wall.normal[i] * (position[i] - wall.point[i]);
            }
            let excess = raw / wall.scale;
            if excess > 0.0 {
                if pass == MAX_REFLECTION_PASSES {
                    return Err(geometry(Error::Dynamics {
                        what: "flat wall reflections did not converge",
                    }));
                }
                let mut v_dot = 0.0;
                for i in 0..velocity.len() {
                    if masses[i] > 0.0 {
                        v_dot += wall.tangent_m_inv[i] * masses[i] * velocity[i];
                    }
                }
                for i in 0..position.len() {
                    position[i] -= wall.tangent_m_inv[i] * (2.0 * excess);
                    velocity[i] -= wall.tangent_m_inv[i] * (2.0 * v_dot);
                }
                *position = manifold.snap_mass(metric, position).map_err(geometry)?;
                *velocity = manifold
                    .project_mass_velocity(metric, position, velocity)
                    .map_err(geometry)?;
                any = true;
                hits += 1;
            }
        }
        if !any {
            break;
        }
    }
    Ok(hits)
}

fn pull_force<E, F>(
    force: &mut F,
    position: &Array1<f64>,
) -> Result<(Vec<f64>, f64), DynamicsError<E>>
where
    F: FnMut(&[f64]) -> Result<(Vec<f64>, f64), E>,
{
    let coords = position.as_slice().ok_or_else(|| {
        geometry::<E>(Error::MassMetric {
            what: "the position is not contiguous",
        })
    })?;
    let (values, potential) = force(coords).map_err(DynamicsError::Force)?;
    if values.len() != position.len()
        || !values.iter().all(|v| v.is_finite())
        || !potential.is_finite()
    {
        return Err(geometry(Error::Oracle {
            what: "non-finite force",
        }));
    }
    Ok((values, potential))
}

fn finish<M, E>(
    manifold: &M,
    metric: &MassMetric,
    position: &Array1<f64>,
    velocity: &Array1<f64>,
    potential: f64,
    hits: u64,
    scale: f64,
    force: &[f64],
) -> Result<DynamicsReport, DynamicsError<E>>
where
    M: Manifold + ?Sized,
{
    let reaction = mass::constraint_reaction(
        manifold,
        metric,
        position,
        &Array1::from_vec(force.to_vec()),
    )
    .map_err(geometry)?;
    let degrees = mass::tangent_dimension(manifold, metric, position).map_err(geometry)?;
    Ok(DynamicsReport {
        potential,
        kinetic: kinetic(metric, velocity, scale),
        multipliers: reaction.multipliers,
        wall_hits: hits,
        degrees_of_freedom: degrees,
    })
}

/// Constrained BAOAB. `force` returns the ambient force and the potential
/// at a position. On a flat manifold `point` is restored before the first
/// step; on a curved manifold it must already lie on the manifold. Each
/// force kick is projected before the following drift.
pub fn baoab<M, E, F>(
    manifold: &M,
    metric: &MassMetric,
    point: &mut PhasePoint,
    config: &LangevinConfig,
    walls: &[PlanarWall],
    steps: usize,
    mut force: F,
) -> Result<DynamicsReport, DynamicsError<E>>
where
    M: Manifold + ?Sized,
    F: FnMut(&[f64]) -> Result<(Vec<f64>, f64), E>,
{
    check_langevin(config)?;
    let n = metric.len();
    if point.position.len() != n || point.velocity.len() != n {
        return Err(geometry(Error::MassMetric {
            what: "phase point and masses differ in length",
        }));
    }
    if steps == 0 {
        return Ok(DynamicsReport::default());
    }
    check_phase(manifold, metric, &point.position, &point.velocity)?;
    let masses = metric
        .masses()
        .as_slice()
        .ok_or_else(|| {
            geometry(Error::MassMetric {
                what: "the mass diagonal is not contiguous",
            })
        })?
        .to_vec();
    if manifold.drift_is_straight() {
        point.position = manifold
            .snap_mass(metric, &point.position)
            .map_err(geometry)?;
    }
    let normals = match manifold.euclidean_normals(&point.position) {
        Some(basis) => basis,
        None => mass::normal_frame(manifold, &point.position).map_err(geometry)?,
    };
    let prepared = prepare_walls(&normals, metric, walls)?;
    let half = 0.5 * config.time_step;
    let decay = (-config.friction * config.time_step).exp();
    let noise_amp = (config.thermal
        * config.kinetic_scale
        * (1.0 - (-2.0 * config.friction * config.time_step).exp()))
    .sqrt();
    let mut rng = SplitMix64::new(config.seed);
    let mut hits = 0u64;
    let (mut force_now, mut potential) = pull_force(&mut force, &point.position)?;

    for _ in 0..steps {
        kick(
            &mut point.velocity,
            &force_now,
            &masses,
            half,
            config.kinetic_scale,
        );
        point.velocity = manifold
            .project_mass_velocity(metric, &point.position, &point.velocity)
            .map_err(geometry)?;
        let (next, carried) = drift(manifold, metric, &point.position, &point.velocity, half)?;
        point.position = next;
        point.velocity = carried;
        hits += reflect_prepared(
            manifold,
            metric,
            &prepared,
            walls,
            &mut point.position,
            &mut point.velocity,
        )?;

        for i in 0..n {
            let sample = rng.next_normal();
            if masses[i] > 0.0 {
                let sigma = noise_amp / masses[i].sqrt();
                point.velocity[i] = point.velocity[i] * decay + sigma * sample;
            }
        }
        point.velocity = manifold
            .project_mass_velocity(metric, &point.position, &point.velocity)
            .map_err(geometry)?;

        let (next, carried) = drift(manifold, metric, &point.position, &point.velocity, half)?;
        point.position = next;
        point.velocity = carried;
        hits += reflect_prepared(
            manifold,
            metric,
            &prepared,
            walls,
            &mut point.position,
            &mut point.velocity,
        )?;

        let pulled = pull_force(&mut force, &point.position)?;
        force_now = pulled.0;
        potential = pulled.1;
        kick(
            &mut point.velocity,
            &force_now,
            &masses,
            half,
            config.kinetic_scale,
        );
        point.velocity = manifold
            .project_mass_velocity(metric, &point.position, &point.velocity)
            .map_err(geometry)?;
    }
    finish(
        manifold,
        metric,
        &point.position,
        &point.velocity,
        potential,
        hits,
        config.kinetic_scale,
        &force_now,
    )
}

/// Constrained velocity Verlet. On a Euclidean manifold this is the
/// unconstrained velocity Verlet step. On a constraint it is RATTLE.
pub fn rattle<M, E, F>(
    manifold: &M,
    metric: &MassMetric,
    point: &mut PhasePoint,
    config: &NveConfig,
    steps: usize,
    mut force: F,
) -> Result<DynamicsReport, DynamicsError<E>>
where
    M: Manifold + ?Sized,
    F: FnMut(&[f64]) -> Result<(Vec<f64>, f64), E>,
{
    check_nve(config)?;
    let n = metric.len();
    if point.position.len() != n || point.velocity.len() != n {
        return Err(geometry(Error::MassMetric {
            what: "phase point and masses differ in length",
        }));
    }
    if steps == 0 {
        return Ok(DynamicsReport::default());
    }
    check_phase(manifold, metric, &point.position, &point.velocity)?;
    let masses = metric
        .masses()
        .as_slice()
        .ok_or_else(|| {
            geometry(Error::MassMetric {
                what: "the mass diagonal is not contiguous",
            })
        })?
        .to_vec();
    if manifold.drift_is_straight() {
        point.position = manifold
            .snap_mass(metric, &point.position)
            .map_err(geometry)?;
    }
    let half = 0.5 * config.time_step;
    let (mut force_now, mut potential) = pull_force(&mut force, &point.position)?;
    for _ in 0..steps {
        kick(
            &mut point.velocity,
            &force_now,
            &masses,
            half,
            config.kinetic_scale,
        );
        point.velocity = manifold
            .project_mass_velocity(metric, &point.position, &point.velocity)
            .map_err(geometry)?;
        let (next, carried) = drift(
            manifold,
            metric,
            &point.position,
            &point.velocity,
            config.time_step,
        )?;
        point.position = next;
        point.velocity = carried;
        let pulled = pull_force(&mut force, &point.position)?;
        force_now = pulled.0;
        potential = pulled.1;
        kick(
            &mut point.velocity,
            &force_now,
            &masses,
            half,
            config.kinetic_scale,
        );
        point.velocity = manifold
            .project_mass_velocity(metric, &point.position, &point.velocity)
            .map_err(geometry)?;
    }
    finish(
        manifold,
        metric,
        &point.position,
        &point.velocity,
        potential,
        0,
        config.kinetic_scale,
        &force_now,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{AffineHyperplane, Euclidean, Sphere};
    use ndarray::array;

    /// Independent transcription of the hyperplane BAOAB arithmetic.
    fn reference(
        point: &[f64],
        normal_in: &[f64],
        mass: &[f64],
        x0: &[f64],
        v0: &[f64],
        force_at: impl Fn(&[f64]) -> Vec<f64>,
        wall: Option<(&[f64], &[f64])>,
        dt: f64,
        friction: f64,
        k_t: f64,
        seed: u64,
        steps: usize,
    ) -> (Vec<f64>, Vec<f64>, u64) {
        let n = x0.len();
        let nnorm = normal_in.iter().map(|v| v * v).sum::<f64>().sqrt();
        let inv_norm = 1.0 / nnorm;
        let normal: Vec<f64> = normal_in.iter().map(|v| v * inv_norm).collect();
        let mut alpha = 0.0;
        for i in 0..n {
            if mass[i] > 0.0 {
                alpha += normal[i] * normal[i] / mass[i];
            }
        }
        let mut plane_w = vec![0.0; n];
        for i in 0..n {
            if mass[i] > 0.0 {
                plane_w[i] = normal[i] * (1.0 / (alpha * mass[i]));
            }
        }
        let prepared = wall.map(|(wp, wn)| {
            let un = wn.iter().map(|v| v * v).sum::<f64>().sqrt();
            let inv_u = 1.0 / un;
            let unit: Vec<f64> = wn.iter().map(|v| v * inv_u).collect();
            let mut n_minv_u = 0.0;
            for i in 0..n {
                if mass[i] > 0.0 {
                    n_minv_u += normal[i] * unit[i] / mass[i];
                }
            }
            let coeff = n_minv_u / alpha;
            let u_tan: Vec<f64> = (0..n).map(|i| unit[i] - normal[i] * coeff).collect();
            let mut scale_sq = 0.0;
            for i in 0..n {
                if mass[i] > 0.0 {
                    scale_sq += u_tan[i] * u_tan[i] / mass[i];
                }
            }
            let scale = scale_sq.sqrt();
            let mut tangent = vec![0.0; n];
            if scale > 0.0 {
                for i in 0..n {
                    if mass[i] > 0.0 {
                        tangent[i] = u_tan[i] * (1.0 / (scale * mass[i]));
                    }
                }
            }
            (wp.to_vec(), unit, tangent, scale)
        });
        let restore = |x: &mut [f64]| {
            let mut nx = 0.0;
            for i in 0..n {
                nx += normal[i] * (x[i] - point[i]);
            }
            for i in 0..n {
                x[i] -= plane_w[i] * nx;
            }
        };
        let project = |v: &mut [f64]| {
            let mut nv = 0.0;
            for i in 0..n {
                nv += normal[i] * v[i];
            }
            for i in 0..n {
                v[i] -= plane_w[i] * nv;
            }
        };
        let mut x = x0.to_vec();
        let mut v = v0.to_vec();
        restore(&mut x);
        project(&mut v);
        let half = 0.5 * dt;
        let decay = (-friction * dt).exp();
        let noise_amp = (k_t * (1.0 - (-2.0 * friction * dt).exp())).sqrt();
        let mut rng = rng::SplitMix64::new(seed);
        let mut hits = 0u64;
        let mut f = force_at(&x);
        for _ in 0..steps {
            for i in 0..n {
                if mass[i] > 0.0 {
                    v[i] += f[i] * (half / mass[i]);
                }
            }
            project(&mut v);
            for i in 0..n {
                x[i] += v[i] * half;
            }
            restore(&mut x);
            if let Some((wp, unit, tangent, scale)) = &prepared
                && *scale > 0.0
            {
                let mut raw = 0.0;
                for i in 0..n {
                    raw += unit[i] * (x[i] - wp[i]);
                }
                let excess = raw / scale;
                if excess > 0.0 {
                    let mut v_dot = 0.0;
                    for i in 0..n {
                        if mass[i] > 0.0 {
                            v_dot += tangent[i] * mass[i] * v[i];
                        }
                    }
                    for i in 0..n {
                        x[i] -= tangent[i] * (2.0 * excess);
                        v[i] -= tangent[i] * (2.0 * v_dot);
                    }
                    restore(&mut x);
                    project(&mut v);
                    hits += 1;
                }
            }
            for i in 0..n {
                let sample = rng.next_normal();
                if mass[i] > 0.0 {
                    let sigma = noise_amp / mass[i].sqrt();
                    v[i] = v[i] * decay + sigma * sample;
                }
            }
            project(&mut v);
            for i in 0..n {
                x[i] += v[i] * half;
            }
            restore(&mut x);
            if let Some((wp, unit, tangent, scale)) = &prepared
                && *scale > 0.0
            {
                let mut raw = 0.0;
                for i in 0..n {
                    raw += unit[i] * (x[i] - wp[i]);
                }
                let excess = raw / scale;
                if excess > 0.0 {
                    let mut v_dot = 0.0;
                    for i in 0..n {
                        if mass[i] > 0.0 {
                            v_dot += tangent[i] * mass[i] * v[i];
                        }
                    }
                    for i in 0..n {
                        x[i] -= tangent[i] * (2.0 * excess);
                        v[i] -= tangent[i] * (2.0 * v_dot);
                    }
                    restore(&mut x);
                    project(&mut v);
                    hits += 1;
                }
            }
            f = force_at(&x);
            for i in 0..n {
                if mass[i] > 0.0 {
                    v[i] += f[i] * (half / mass[i]);
                }
            }
            project(&mut v);
        }
        (x, v, hits)
    }

    #[test]
    fn flat_hyperplane_matches_the_rattle_baoab_bits() {
        let point = array![0.2, -0.4, 0.1, 0.0, 0.5, -0.2];
        let normal = array![1.0, 0.0, 0.0, -1.0, 0.0, 0.0];
        let plane = AffineHyperplane::new(point.clone(), normal.clone()).unwrap();
        let metric = MassMetric::per_atom(&[4.0, 39.9]).unwrap();
        let mut phase = PhasePoint::default()
            .with_position(array![0.3, 0.1, -0.2, 0.05, 0.4, 0.2])
            .with_velocity(array![0.2, -0.1, 0.05, 0.0, 0.3, -0.4]);
        let wall = PlanarWall::new(
            array![0.0, 0.8, 0.0, 0.0, 0.8, 0.0],
            array![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let config = LangevinConfig::default()
            .with_thermal(0.03)
            .with_friction(0.7)
            .with_time_step(0.02)
            .with_seed(11)
            .with_kinetic_scale(1.0);
        let force = |x: &[f64]| {
            let mut f = vec![0.0; x.len()];
            f[1] = -0.4 * x[1];
            f[4] = -0.2 * x[4];
            Ok::<_, &'static str>((f, 0.2 * x[1] * x[1] + 0.1 * x[4] * x[4]))
        };
        let report = baoab(&plane, &metric, &mut phase, &config, &[wall], 5, force).expect("baoab");
        let (rx, rv, rh) = reference(
            point.as_slice().unwrap(),
            normal.as_slice().unwrap(),
            metric.masses().as_slice().unwrap(),
            &[0.3, 0.1, -0.2, 0.05, 0.4, 0.2],
            &[0.2, -0.1, 0.05, 0.0, 0.3, -0.4],
            |x| {
                let mut f = vec![0.0; x.len()];
                f[1] = -0.4 * x[1];
                f[4] = -0.2 * x[4];
                f
            },
            Some((
                &[0.0, 0.8, 0.0, 0.0, 0.8, 0.0],
                &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            )),
            0.02,
            0.7,
            0.03,
            11,
            5,
        );
        for (got, want) in phase.position.iter().zip(&rx) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
        for (got, want) in phase.velocity.iter().zip(&rv) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
        assert_eq!(report.wall_hits, rh);
        let distance = plane.signed_distance(&phase.position).unwrap();
        let magnitude = phase
            .position
            .iter()
            .fold(0.0_f64, |acc, value| acc + value.abs());
        let roundoff = magnitude * f64::EPSILON * phase.position.len() as f64;
        assert!(distance.abs() <= roundoff);
    }

    #[test]
    fn euclidean_rattle_is_velocity_verlet() {
        let metric = MassMetric::uniform(1, 2.0).unwrap();
        let mut phase = PhasePoint::default()
            .with_position(array![0.0])
            .with_velocity(array![1.0]);
        let config = NveConfig::default()
            .with_time_step(0.1)
            .with_kinetic_scale(1.0);
        rattle(&Euclidean, &metric, &mut phase, &config, 1, |x| {
            Ok::<_, &'static str>((vec![-x[0]], 0.5 * x[0] * x[0]))
        })
        .unwrap();
        // v += dt/2 * F/m, x += dt * v, F = -x, v += dt/2 * F/m.
        let half = 0.05_f64;
        let mut v = 1.0 + half * (0.0 / 2.0);
        let x = 0.0 + 0.1 * v;
        v += half * (-x / 2.0);
        assert_eq!(phase.position[0].to_bits(), x.to_bits());
        assert_eq!(phase.velocity[0].to_bits(), v.to_bits());
    }

    #[test]
    fn sphere_step_stays_on_the_sphere() {
        let metric = MassMetric::uniform(3, 1.0).unwrap();
        let mut phase = PhasePoint::default()
            .with_position(array![0.0, 0.0, 1.0])
            .with_velocity(array![0.2, -0.1, 0.0]);
        let config = LangevinConfig::default()
            .with_thermal(0.2)
            .with_friction(0.5)
            .with_time_step(0.05)
            .with_seed(3);
        baoab(&Sphere, &metric, &mut phase, &config, &[], 4, |x| {
            Ok::<_, &'static str>((vec![-x[0], 0.0, 0.0], x[0]))
        })
        .unwrap();
        let norm = phase.position.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() <= 32.0 * f64::EPSILON);
        let radial = phase
            .position
            .iter()
            .zip(phase.velocity.iter())
            .map(|(q, u)| q * u)
            .sum::<f64>();
        assert!(radial.abs() <= 32.0 * f64::EPSILON);
    }
}
