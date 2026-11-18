//! Persistent solver session. One [`Solver::step`] is one outer iteration.

mod chart;
mod constrained;
pub(crate) mod drive;

use std::collections::VecDeque;
use std::sync::Arc;

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::accept::{Accept, Acceptance, Trial, accept_step, retract_within_cap, single_trial};
use crate::adam::{adam_direction, scale_to_unit_atom};
use crate::bb::bb_direction;
use crate::control::{Control, StopNorm};
use crate::diis::{DiisState, gdiis_point};
use crate::error::{Error, Result};
use crate::fire::{
    Fire2Extras, FireConfig, FireKind, FireMix, FirePreparation, FirePreset, FireState, FireTrace,
    FireVariant, fire_abc_clipped, fire_after_v1_metric, fire_clip_components,
    fire_displacement_into, fire_move_clipped, fire_prepare_v2, fire_rescale_velocity,
    fire2_displacement_metric, guenole2020,
};
use crate::holonomic::DistanceConstraints;
use crate::hvp::{
    FiniteDifference, HessianVector, HostAction, HostOracle, NewtonCgRequest, StepAction,
    StepOracle, TryHessianVector, newton_cg_step, sym_eig_jacobi,
};
use crate::lbfgs::{GradNorm, Lbfgs, LbfgsStep};
use crate::level_set::{LevelSetConstraint, project_tangent, retract_to_level_set_fallible};
use crate::lindh::LindhSpec;
use crate::linesearch::{LineSearch, LineSearchOptions};
use crate::lowest_mode::ApplyHessian;
use crate::manifold::{Manifold, ManifoldKind};
use crate::method::Method;
use crate::newton::{HessianObjective, NewtonKind, rfo_direction, shifted_newton};
use crate::nlcg::{Conjugacy, ConjugacyContext, Restart};
use crate::precon::{
    ExpParams, FfParams, FfTerms, Ghost, LindhParams, Neighbours, PreconSpec, Slot,
};
use crate::pso::{Particle, RNG_SEED, random_velocity, update_swarm};
use crate::qn::{
    bfgs_inverse_update, sr_trust_direction, sr_trust_limits, sr1_hessian_update,
    sr2_hessian_update,
};
use crate::qn_step::QnStep;
use crate::quickmin::{
    QuickMinOptions, QuickMinSession, QuickMinState, quickmin_rescale_velocity,
    quickmin_session_step,
};
use crate::report::{Report, Status};
use crate::rigid::{RigidMode, RigidWorkspace, project_out_rot_trans};
use crate::step::{
    Taken, apply_caps, cap_alpha, cap_displacement_atom, l2, max_atom_displacement,
    nlcg_opening_alpha, qn_istep, scale_displacement_to_atom_cap, take_step_with_options,
};
use crate::trust::{
    accept_ratio, burdakov_radius, dogleg_direction, predicted_reduction, reduction_ratio,
    update_radius,
};

/// The session's line search until [`Solver::set_linesearch`]: strong
/// Wolfe with `c1 = 1e-4`, `c2 = 0.9`. On the atomistic benchmark it
/// takes the quasi-Newton unit step at about 1.1 oracle calls per
/// iteration where Brent's exact search spends about 25.
const SESSION_LINESEARCH: LineSearch = LineSearch::Wolfe {
    c1: 1e-4,
    c2: 0.9,
    maxiter: 20,
};

/// Why a capped trial did not move.
enum CappedMiss {
    /// The quasi-Newton trial was non-finite. Try the force once.
    Retry,
    /// Stop and restore the pairs.
    Stop(Error),
    /// The level-set tangent of the capped step is zero. Keep the pairs
    /// and the current point; it is already on the set.
    LevelStay,
}

/// Smallest initial trust radius. The constrained stepper reads this.
const RADIUS_START: f64 = 1e-8;
/// Largest trust-radius growth, in units of the current radius, with no `maxmove`.
const RADIUS_GROWTH: f64 = 8.0;
/// Saddle-escape kicks from one stationary point.
const SADDLE_KICKS: u8 = 4;

/// `P^{-1} v` when a geometry metric is installed, otherwise `v`.
///
/// A failed solve is an error. Substituting `v` would mix `P^{-1} g` and
/// `g` inside one CG recurrence or one FIRE mix.
fn metric_vector(precon: &Option<Slot>, v: &Array1<f64>) -> Result<Array1<f64>> {
    match precon {
        None => Ok(v.clone()),
        Some(state) => {
            let p = state.metric().ok_or(Error::Precon {
                what: "metric is not assembled",
            })?;
            p.solve(v.view()).ok_or(Error::Precon {
                what: "linear solve failed",
            })
        }
    }
}

/// Remove rigid motion at the anchor stored on `ws`.
#[allow(clippy::too_many_arguments)]
fn eckart(
    ws: &mut RigidWorkspace,
    mode: RigidMode,
    masses: Option<&[f64]>,
    frozen: Option<&[u8]>,
    weighted: bool,
    metric: bool,
    pbc: [bool; 3],
    broken_translation: [bool; 3],
    broken_rotation: [bool; 3],
    v: &mut Array1<f64>,
) {
    if mode == RigidMode::Off {
        return;
    }
    let Some(vs) = v.as_slice_mut() else {
        return;
    };
    ws.project_at_anchor(
        vs,
        masses,
        frozen,
        weighted,
        metric,
        mode,
        pbc,
        broken_translation,
        broken_rotation,
    );
}

/// What a first-order step removes from its force, the metric image of
/// the force and its velocity: the rigid motion at the stored anchor.
struct RigidRemoval<'a> {
    mode: RigidMode,
    masses: Option<&'a [f64]>,
    frozen: Option<&'a [u8]>,
    weighted: bool,
    force_metric: bool,
    velocity_metric: bool,
    pbc: [bool; 3],
    broken_translation: [bool; 3],
    broken_rotation: [bool; 3],
}

impl RigidRemoval<'_> {
    fn project(&self, ws: &mut RigidWorkspace, metric: bool, v: &mut Array1<f64>) {
        eckart(
            ws,
            self.mode,
            self.masses,
            self.frozen,
            self.weighted,
            metric,
            self.pbc,
            self.broken_translation,
            self.broken_rotation,
            v,
        );
    }

    /// `v` is a force, or its image under the preconditioner.
    fn force(&self, ws: &mut RigidWorkspace, v: &mut Array1<f64>) {
        self.project(ws, self.force_metric, v);
    }

    /// `v` is a velocity.
    fn velocity(&self, ws: &mut RigidWorkspace, v: &mut Array1<f64>) {
        self.project(ws, self.velocity_metric, v);
    }
}

/// Puts a taken [`RigidWorkspace`] back if a stepper returns early.
struct WsHold<'a> {
    slot: &'a mut RigidWorkspace,
    ws: RigidWorkspace,
}

impl<'a> WsHold<'a> {
    fn take(slot: &'a mut RigidWorkspace) -> Self {
        let mut hold = WsHold {
            slot,
            ws: RigidWorkspace::new(),
        };
        std::mem::swap(hold.slot, &mut hold.ws);
        hold
    }
}

impl Drop for WsHold<'_> {
    fn drop(&mut self) {
        std::mem::swap(self.slot, &mut self.ws);
    }
}

/// Steepest direction in the metric, `-z`, with the `Accept::Step` scale.
fn nlcg_steepest(z: &Array1<f64>, accept: Accept, istep: f64) -> Array1<f64> {
    if accept == Accept::Step {
        z.mapv(|v| -v * istep)
    } else {
        z.mapv(|v| -v)
    }
}

/// P-norms for the FIRE mix: `sqrt(v^T P v)` and `sqrt(F · P^{-1} F)`.
fn fire_mix(
    precon: &Option<Slot>,
    vel: &Array1<f64>,
    raw: &Array1<f64>,
    steepest: &Array1<f64>,
) -> Option<FireMix> {
    let p = precon.as_ref().and_then(Slot::metric)?;
    let pv = p.apply(vel.view());
    let vel_n = vel.dot(&pv).max(0.0).sqrt();
    let dir_n = raw.dot(steepest).max(0.0).sqrt();
    Some(FireMix {
        vel: vel_n,
        direction: dir_n,
    })
}

/// Dogleg model on a curved kind. The sphere form keeps its work vectors
/// `H x` and `H^T x`, and other kinds keep the tangent basis `Z`.
#[derive(Default)]
struct TangentModel {
    model: Array2<f64>,
    hx: Array1<f64>,
    htx: Array1<f64>,
    basis: Array2<f64>,
}

/// Long-lived solver. Algorithm memory lives here; `x` stays with the caller.
pub struct Solver {
    dim: usize,
    control: Control,
    linesearch: LineSearch,
    line_options: LineSearchOptions,
    istep: f64,
    steps: usize,
    qn_step: QnStep,
    accept: Accept,
    e_hist: VecDeque<f64>,
    /// Per-atom cap in force. `None` is no per-atom cap.
    atom_maxmove: Option<f64>,
    /// `set_atom_maxmove` has been called. The default cap is then left alone.
    atom_maxmove_set: bool,
    project_rigid: bool,
    /// `Auto` until a host declares a molecule. See [`Self::removal_mode`].
    rigid_mode: RigidMode,
    /// Periodic axes, masses, a freeze, a field, or an explicit mode.
    rigid_declared: bool,
    /// `set_project_rigid(false)` / `set_rigid_mode(Off)` on a Euclidean session.
    rigid_forced_off: bool,
    /// The rigid settings before periodic axes declared the molecule.
    /// Clearing the axes puts them back; any other declaration drops them.
    rigid_before_pbc: Option<(RigidMode, bool, bool, bool)>,
    /// Periodic axes. Any true axis drops rotations under Auto.
    pbc: [bool; 3],
    /// Translations the host says are not symmetries.
    broken_translation: [bool; 3],
    /// Rotations the host says are not symmetries.
    broken_rotation: [bool; 3],
    /// Length `3N`. Nonzero freezes that coordinate.
    frozen: Option<Vec<u8>>,
    /// Cartesian mass inner product. Not the `MwRigid` chart.
    mass_metric: bool,
    /// `x` stores `q_i = sqrt(m_i) x_i`.
    mass_weighted_coords: bool,
    rigid_ws: RigidWorkspace,
    /// Sella `proj_rot`: false under PBC. Rotation is not a symmetry of the cell.
    periodic: bool,
    manifold: ManifoldKind,
    /// Per-atom masses for [`ManifoldKind::MwRigid`]. Length N, not 3N.
    masses: Option<Array1<f64>>,
    /// Optional Gr(n,p) shape for the Grassmann selector.
    factor_shape: Option<(usize, usize)>,
    /// Scales `d` of the metric chart `z = d * x` of a weighted product.
    chart: Option<Arc<Array1<f64>>>,
    /// Natural point last written back from the chart, and its chart point.
    chart_seen: Option<(Array1<f64>, Array1<f64>)>,
    #[cfg(feature = "highs")]
    highs: bool,
    #[cfg(feature = "highs")]
    highs_options: crate::HighsOptions,
    #[cfg(feature = "highs")]
    box_lo: Option<Vec<f64>>,
    #[cfg(feature = "highs")]
    box_hi: Option<Vec<f64>>,
    #[cfg(feature = "highs")]
    highs_trust: Option<f64>,
    #[cfg(feature = "highs")]
    equalities: Vec<(Vec<(usize, f64)>, f64)>,
    last_pos: Option<Array1<f64>>,
    last_value: f64,
    last_grad: Array1<f64>,
    /// Unlimited L-BFGS plateau. Unused by every other method, and unused
    /// when the iteration budget is positive.
    lbfgs_plateau: crate::lbfgs_plateau::RepeatedAcceptedStep,
    /// Objective gradients. A finite-difference product is an
    /// [`Self::hvp_calls`] entry, not a gradient.
    gradient_calls: usize,
    /// Analytic or finite-difference Hessian-vector products.
    hvp_calls: usize,
    /// Negative-curvature kicks already taken in this solve.
    saddle_kicks: u8,
    /// Whether geometry L-BFGS needs a finite-difference stiffness probe.
    geometry_lbfgs_stiff: Option<bool>,
    /// Capped Cartesian L-BFGS takes the geometry step. `None` leaves
    /// a Wolfe session with an explicit per-atom cap on it.
    geometry_step: Option<bool>,
    /// Geometry metric. `None` leaves every method unchanged.
    precon: Option<Slot>,
    /// Bumped when `P` is installed, cleared, or rebuilt.
    metric_epoch: u64,
    /// Times NLCG replaced a non-descent direction with steepest descent.
    nlcg_descent_fallbacks: u64,
    /// First-step scale used by the constrained stepper. Off unless the C API asks.
    unit_first_step: bool,
    /// The constrained acceptance test refused the previous step.
    refused: bool,
    /// Dogleg model on a curved kind, rewritten in place at each step.
    tangent: TangentModel,
    /// Norm compared with [`Control::gtol`]. Starts as [`Control::stop_norm`].
    stop_norm: StopNorm,
    /// Holonomic distance constraints. `None` is the unconstrained step.
    constraints: Option<Arc<DistanceConstraints>>,
    /// Optional level-set constraint \(g(x) = c\).
    level_set: Option<LevelSetConstraint>,
    /// [`Self::run`] skips the coordinate copy on intermediate reports.
    brief: bool,
    /// Optional trace of the eOn fill step.
    eon_log: Option<Vec<drive::EonFeat>>,
    /// Whether per-step FIRE tracing is active.
    fire_trace_on: bool,
    /// Recorded per-step trace of a FIRE session.
    fire_trace: Vec<FireTrace>,
    /// Ratio test on a capped two-loop fill. Off by default.
    trust: bool,
    trust_delta: f64,
    trust_max: f64,
    /// Starting Burdakov radius. Zero leaves that radius off.
    lbfgs_burdakov_init: f64,
    /// Current Burdakov radius.
    lbfgs_burdakov: f64,
    scratch: Array1<f64>,
    scratch_b: Array1<f64>,
    saved_x: Array1<f64>,
    /// `Some` when the exponential preconditioner is the installed metric.
    exp: Option<()>,
    lindh_spec: Option<LindhSpec>,
    lindh_mu: Option<f64>,
    /// SR1/SR2 trust radius. The matrix itself lives on [`Inner`].
    sr_radius: f64,
    /// Published FIRE 2.0 schedule. `None` is the rgmin integrator.
    fire2: Option<Fire2Extras>,
    /// Quick-min, ODE12r, or DIIS. [`Inner`] stays a method the constraint
    /// stepper already matches; this overlay runs instead of that arm.
    overlay: Overlay,
    /// Objective bounds noted at the start of a step, for the box norm.
    bound_low: Option<Array1<f64>>,
    bound_high: Option<Array1<f64>>,
    inner: Inner,
}

/// Methods added after the constraint stepper's match was written.
enum Overlay {
    None,
    QuickMin(QuickMinState, QuickMinSession),
    Ode12r(crate::ode12r::Ode12rState),
    Diis(DiisState),
}

#[allow(clippy::large_enum_variant)]
enum Inner {
    Lbfgs(Lbfgs),
    Nlcg {
        conjugacy: Conjugacy,
        restart: Restart,
        dir: Array1<f64>,
        g_old: Array1<f64>,
        /// `P^{-1} g` at the previous accepted point. Equals `g_old` when
        /// no geometry metric is installed.
        z_old: Array1<f64>,
        d_old: Array1<f64>,
        initialized: bool,
        /// `metric_epoch` at the moment `z_old` and `dir` were formed.
        z_epoch: u64,
        /// Accepted line-search step `α_{k-1}`. Zero before the first move.
        prev_alpha: f64,
        /// `g_{k-1} · d_{k-1}` for Nocedal-Wright equation 3.60.
        prev_gd: f64,
    },
    /// `fresh`: no secant pair has reached the matrix since construction,
    /// `forget`, or a refused constrained step.
    Bfgs {
        h: Array2<f64>,
        fresh: bool,
    },
    Sr1 {
        h: Array2<f64>,
        fresh: bool,
    },
    Sr2 {
        b: Array2<f64>,
        fresh: bool,
    },
    Adam {
        m: Array1<f64>,
        v: Array1<f64>,
        b1p: f64,
        b2p: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    },
    Steepest,
    Pso {
        n_particles: usize,
        inertia: f64,
        c1: f64,
        c2: f64,
        swarm: Option<PsoState>,
    },
    Newton {
        kind: NewtonKind,
    },
    Fire(FireState),
    Bb {
        prev_s: Option<Array1<f64>>,
        prev_y: Option<Array1<f64>>,
    },
    Dogleg {
        radius: f64,
    },
    NewtonCg {
        radius: f64,
    },
    NewtonPcg {
        radius: f64,
    },
}

struct PsoState {
    swarm: Vec<Particle>,
    gbest_position: Array1<f64>,
    gbest_value: f64,
    rng: StdRng,
}

/// Projects the rigid modes out of `dir` at `x`. True when more than
/// rounding left the direction.
fn strip_rigid(dir: &mut Array1<f64>, x: &Array1<f64>) -> bool {
    let before = dir.clone();
    project_out_rot_trans(dir, x.view());
    l2(&(&before - &*dir)) > 1e-12 * l2(&before)
}

impl Solver {
    /// Fresh session for `method` in dimension `dim`.
    pub fn new(method: Method, control: Control, dim: usize) -> Self {
        let istep = control.istep;
        let stop_norm = control.stop_norm;
        let inner = Inner::from_method(&method, dim, istep);
        let mut solver = Self {
            dim,
            control,
            linesearch: SESSION_LINESEARCH,
            line_options: LineSearchOptions::default(),
            istep,
            steps: 0,
            qn_step: QnStep::TwoLoop,
            accept: Accept::None,
            e_hist: VecDeque::new(),
            atom_maxmove: None,
            atom_maxmove_set: false,
            project_rigid: false,
            rigid_mode: RigidMode::Auto,
            rigid_declared: false,
            rigid_forced_off: false,
            rigid_before_pbc: None,
            pbc: [false; 3],
            broken_translation: [false; 3],
            broken_rotation: [false; 3],
            frozen: None,
            mass_metric: false,
            mass_weighted_coords: false,
            rigid_ws: RigidWorkspace::new(),
            periodic: false,
            manifold: ManifoldKind::Euclidean,
            masses: None,
            factor_shape: None,
            chart: None,
            chart_seen: None,
            #[cfg(feature = "highs")]
            highs: false,
            #[cfg(feature = "highs")]
            highs_options: crate::HighsOptions::default(),
            #[cfg(feature = "highs")]
            box_lo: None,
            #[cfg(feature = "highs")]
            box_hi: None,
            #[cfg(feature = "highs")]
            highs_trust: None,
            #[cfg(feature = "highs")]
            equalities: Vec::new(),
            last_pos: None,
            last_value: 0.0,
            last_grad: Array1::zeros(dim),
            lbfgs_plateau: crate::lbfgs_plateau::RepeatedAcceptedStep::new(),
            gradient_calls: 0,
            hvp_calls: 0,
            saddle_kicks: 0,
            geometry_lbfgs_stiff: None,
            geometry_step: None,
            precon: None,
            metric_epoch: 0,
            nlcg_descent_fallbacks: 0,
            unit_first_step: false,
            refused: false,
            tangent: TangentModel::default(),
            stop_norm,
            constraints: None,
            level_set: None,
            brief: false,
            eon_log: None,
            fire_trace_on: false,
            fire_trace: Vec::new(),
            trust: false,
            trust_delta: 0.0,
            trust_max: 0.0,
            lbfgs_burdakov_init: 0.0,
            lbfgs_burdakov: 0.0,
            scratch: Array1::zeros(dim),
            scratch_b: Array1::zeros(dim),
            saved_x: Array1::zeros(dim),
            exp: None,
            lindh_spec: None,
            lindh_mu: None,
            sr_radius: 0.0,
            fire2: None,
            overlay: Overlay::None,
            bound_low: None,
            bound_high: None,
            inner,
        };
        solver.overlay = match &method {
            Method::QuickMin => {
                Overlay::QuickMin(QuickMinState::new(dim, istep), QuickMinSession::new(istep))
            }
            Method::Ode12r => Overlay::Ode12r(crate::ode12r::Ode12rState::new(istep)),
            Method::Diis { memory } => Overlay::Diis(DiisState::new(*memory)),
            _ => Overlay::None,
        };
        solver.apply_default_atom_cap();
        solver
    }

    /// [`Method::installs_default_atom_cap`] for the arm this session runs.
    fn installs_default_atom_cap(&self) -> bool {
        match &self.inner {
            Inner::Bfgs { .. } => Method::Bfgs.installs_default_atom_cap(),
            Inner::Dogleg { .. } => Method::Dogleg.installs_default_atom_cap(),
            Inner::Fire(state) => Method::Fire { kind: state.kind }.installs_default_atom_cap(),
            // The remaining arms install the cap. A method added to the
            // exception list in [`Method::installs_default_atom_cap`]
            // needs an arm here.
            _ => true,
        }
    }

    /// Install [`crate::DEFAULT_ATOM_MAXMOVE`] on a Cartesian session whose
    /// method has no step control of its own, when the caller has set
    /// neither a per-atom cap nor a positive Euclidean cap.
    ///
    /// BFGS, dogleg, and FIRE 2.0 leave the default off. Dogleg would
    /// otherwise clip the trust step and feed that shorter length to
    /// the radius update, so a step on the trust boundary looks
    /// interior and the radius stays put.
    fn apply_default_atom_cap(&mut self) {
        if self.atom_maxmove_set {
            return;
        }
        // A negative Euclidean cap is the caller's request for no cap at
        // all, including this default. Zero was stored as `None` and keeps it.
        self.atom_maxmove = if matches!(self.control.maxmove, Some(cap) if cap < 0.0) {
            None
        } else if self.installs_default_atom_cap() && self.manifold.retract_is_translation() {
            crate::step::implicit_atom_cap(&self.control)
        } else {
            None
        };
        if self.steps == 0 {
            self.reset_sr_radius();
        }
    }

    /// Family used by [`crate::select_precon`] when the caller asks for auto.
    ///
    /// FIRE is the identity. Nonlinear CG is the metric rule. Every
    /// other method is the quasi-Newton rule, which installs `P^{-1}`
    /// as `H0` when the geometry asks for a matrix.
    #[must_use]
    pub fn optimizer_family(&self) -> crate::OptimizerFamily {
        if matches!(self.overlay, Overlay::Ode12r(_)) || matches!(self.inner, Inner::Fire(_)) {
            crate::OptimizerFamily::Fire
        } else if matches!(self.inner, Inner::Nlcg { .. }) {
            crate::OptimizerFamily::Cg
        } else {
            crate::OptimizerFamily::QuasiNewton
        }
    }

    /// Install one geometry preconditioner.
    ///
    /// L-BFGS takes \(H_0 = P^{-1}\). FIRE and nonlinear CG take `P`
    /// as the metric. An index outside `0..dim/3` fails here.
    /// [`ExpParams::mu`] `None`, and the same field on Lindh and FF,
    /// defers the energy scale to the next [`Self::step`].
    pub fn set_precon(&mut self, spec: PreconSpec<'_>) -> Result<()> {
        let slot = match spec {
            PreconSpec::Exp {
                positions,
                neighbours,
                params,
            } => {
                self.require_precon_dim(positions.len())?;
                Slot::Exp(crate::precon::ExpState::new(positions, neighbours, params)?)
            }
            PreconSpec::Lindh {
                positions,
                z,
                ghosts,
                params,
            } => {
                self.require_precon_dim(positions.len())?;
                Slot::Lindh(crate::precon::LindhState::new(
                    positions, z, ghosts, params,
                )?)
            }
            PreconSpec::Ff {
                positions,
                terms,
                params,
            } => {
                self.require_precon_dim(positions.len())?;
                Slot::Ff(crate::precon::FfState::new(positions, terms, params)?)
            }
        };
        self.install_slot(slot);
        Ok(())
    }

    fn require_precon_dim(&self, got: usize) -> Result<()> {
        if got != self.dim {
            return Err(Error::Dim { got, dim: self.dim });
        }
        Ok(())
    }

    fn install_slot(&mut self, slot: Slot) {
        let is_exp = matches!(&slot, Slot::Exp(_));
        let matrix = slot.metric().cloned();
        self.precon = Some(slot);
        self.exp = if is_exp { Some(()) } else { None };
        self.metric_epoch = self.metric_epoch.wrapping_add(1);
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.set_h0(matrix);
        }
    }

    /// Install the exponential preconditioner at `positions`.
    ///
    /// `positions` has length [`Self::dim`] and is `3 N` coordinates in
    /// C order. See [`Self::set_precon`].
    pub fn set_exp(
        &mut self,
        positions: ArrayView1<f64>,
        neighbours: &Neighbours,
        params: &ExpParams,
    ) -> Result<()> {
        self.set_precon(PreconSpec::Exp {
            positions,
            neighbours,
            params,
        })
    }

    /// Install the Lindh model Hessian at `positions`.
    ///
    /// `z` has one atomic number per free atom. `ghosts` are frozen
    /// atoms. The matrix is held until the largest atomic move exceeds
    /// half the nearest-neighbour distance, the same rule as
    /// [`Self::set_exp`].
    pub fn set_lindh(
        &mut self,
        positions: ArrayView1<f64>,
        z: &[i32],
        ghosts: &[Ghost],
        params: &LindhParams,
    ) -> Result<()> {
        self.set_precon(PreconSpec::Lindh {
            positions,
            z,
            ghosts,
            params,
        })
    }

    /// Install the bonded metric at `positions`.
    ///
    /// The terms are harmonic bonds and angles, Morse bonds, and
    /// harmonic torsions. Unit force constants are a geometric metric.
    /// The hold rule is the one used by [`Self::set_exp`].
    pub fn set_ff(
        &mut self,
        positions: ArrayView1<f64>,
        terms: FfTerms<'_>,
        params: &FfParams,
    ) -> Result<()> {
        self.set_precon(PreconSpec::Ff {
            positions,
            terms,
            params,
        })
    }

    /// Install a neighbour list whose geometry arrives on the next step.
    #[cfg(feature = "capi")]
    pub(crate) fn set_exp_deferred(&mut self, state: crate::precon::ExpState) {
        self.install_slot(Slot::Exp(state));
    }

    /// Install a Lindh list whose geometry arrives on the next step.
    #[cfg(feature = "capi")]
    pub(crate) fn set_lindh_deferred(&mut self, state: crate::precon::LindhState) {
        self.install_slot(Slot::Lindh(state));
    }

    /// Install an FF list whose geometry arrives on the next step.
    #[cfg(feature = "capi")]
    pub(crate) fn set_ff_deferred(&mut self, state: crate::precon::FfState) {
        self.install_slot(Slot::Ff(state));
    }

    /// Drop the installed geometry preconditioner.
    pub fn clear_precon(&mut self) {
        self.precon = None;
        self.exp = None;
        self.metric_epoch = self.metric_epoch.wrapping_add(1);
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.set_h0(None);
        }
    }

    /// Times nonlinear CG replaced a non-descent direction with `-P^{-1} g`
    /// (or `-g` when no metric is installed).
    #[must_use]
    pub fn nlcg_descent_fallbacks(&self) -> u64 {
        self.nlcg_descent_fallbacks
    }

    /// Energy scale of the installed geometry preconditioner, after
    /// estimation. `None` when no preconditioner is installed or the
    /// scale has not been estimated yet.
    #[must_use]
    pub fn precon_scale(&self) -> Option<f64> {
        self.precon.as_ref().and_then(Slot::scale)
    }

    fn refresh_exp<O>(&mut self, obj: &O, x: &Array1<f64>, grad: &Array1<f64>) -> Result<()>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if self.precon.is_none() {
            return Ok(());
        }
        let grad_fn = |p: ArrayView1<f64>| obj.grad(p);
        let mut rebuilt = false;
        if let Some(state) = &mut self.precon {
            rebuilt = state.refresh(x.view(), grad, grad_fn)?;
        }
        if rebuilt {
            self.metric_epoch = self.metric_epoch.wrapping_add(1);
        }
        let matrix = self.precon.as_ref().and_then(|s| s.metric().cloned());
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.set_h0(matrix);
        }
        Ok(())
    }

    /// eOn `lbfgs_step`. Newton / RFO need a Hessian on [`Self::step_hess`].
    pub fn set_qn_step(&mut self, step: QnStep) {
        self.qn_step = step;
    }

    /// eOn `lbfgs_accept`. Default is take the clipped step.
    pub fn set_accept(&mut self, accept: Accept) {
        self.accept = accept;
    }

    /// Line search for the line-searched arms on the next [`Self::step`].
    /// A Euclidean session reads it on steepest descent, NLCG, BFGS, Adam,
    /// DIIS, and L-BFGS after [`LbfgsStep::Wolfe`], unless the acceptance
    /// is [`Accept::Step`]. The sphere and Stiefel L-BFGS arm reads it
    /// under [`Accept::Energy`] or [`Accept::Nonmonotone`], and so does the
    /// constrained L-BFGS search. The default eOn L-BFGS step and the
    /// other methods ignore it. Default is strong Wolfe (`c1 = 1e-4`,
    /// `c2 = 0.9`, 20 trials), not [`LineSearch::default`] (Brent).
    pub fn set_linesearch(&mut self, linesearch: LineSearch) {
        self.linesearch = linesearch;
    }

    /// Set the relative value window of the session's Euclidean line searches.
    /// It replaces the default four-epsilon window under every [`Accept`]
    /// policy. Manifold searches keep their own acceptance, and the
    /// [`Accept::Energy`] / [`Accept::Nonmonotone`] backtracking of a
    /// step that is not line-searched keeps its absolute `1e-8` rise.
    /// Invalid values leave the session unchanged. Zero permits no objective rise.
    pub fn set_objective_roundoff(&mut self, relative: f64) -> bool {
        let Some(options) = LineSearchOptions::with_objective_roundoff(relative) else {
            return false;
        };
        self.line_options = options;
        true
    }

    /// The configured line search.
    pub fn linesearch(&self) -> LineSearch {
        self.linesearch
    }

    /// Dimension of the session's coordinate vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Gradient evaluations charged to this session.
    ///
    /// A forward-difference Hessian-vector product costs one gradient
    /// inside the product and is counted by [`Self::hvp_calls`], not
    /// here. Both counters accumulate for the life of the session.
    /// [`Self::forget`] does not clear them.
    pub fn gradient_calls(&self) -> usize {
        self.gradient_calls
    }

    /// Hessian-vector products charged to this session.
    ///
    /// An analytic product is one call. A finite-difference product is
    /// one extra gradient and is still one product here, so a harness
    /// can charge `gradient_calls + hvp_calls` force evaluations on
    /// this path and `n` force evaluations per dense Hessian elsewhere.
    pub fn hvp_calls(&self) -> usize {
        self.hvp_calls
    }

    /// Euclidean cap applied on the next [`Self::step`].
    ///
    /// A positive finite value is the step limit and the per-atom default
    /// stays off, unless [`Self::set_atom_maxmove`] was called. Zero and
    /// NaN keep the defaults. A negative or infinite value turns every cap
    /// off, the per-atom default included.
    pub fn set_maxmove(&mut self, maxmove: f64) {
        // A negative or infinite value is stored so that the per-atom
        // default stays off; every length read takes only a positive
        // finite cap. NaN is not a length and keeps the defaults.
        self.control.maxmove = if maxmove == 0.0 || maxmove.is_nan() {
            None
        } else {
            Some(maxmove)
        };
        self.apply_default_atom_cap();
    }

    /// eOn `maxAtomMotionAppliedV`. Preferred over the Euclidean cap.
    ///
    /// A fresh Cartesian session holds [`crate::DEFAULT_ATOM_MAXMOVE`]
    /// when `Method::installs_default_atom_cap` is set. BFGS, dogleg,
    /// and FIRE 2.0 start without that cap. A positive `maxmove`
    /// replaces the value in force. A non-positive or non-finite value
    /// clears the cap.
    pub fn set_atom_maxmove(&mut self, maxmove: f64) {
        self.atom_maxmove_set = true;
        self.atom_maxmove = Some(maxmove).filter(|&cap| crate::step::binding(cap));
        if let Inner::Lbfgs(solver) = &mut self.inner {
            let cap = self.atom_maxmove.unwrap_or(crate::lbfgs::EON_MAX_MOVE);
            match &mut solver.step {
                LbfgsStep::Capped { max_atom } | LbfgsStep::Scaled { max_atom } => {
                    *max_atom = cap;
                }
                LbfgsStep::Wolfe => {}
            }
        }
        if self.steps == 0 {
            self.reset_sr_radius();
        }
    }

    /// Capped Cartesian L-BFGS takes the geometry step: eOn's L-BFGS
    /// with a finite-difference or `sy/yy` initial inverse curvature, a
    /// reset on negative curvature, on a step past the per-atom cap or
    /// against the force, and an energy test with at most ten halvings.
    /// Off, the per-atom cap only clips the two-loop trial and the NEB
    /// guards of [`Self::set_lbfgs_neb_guards`] apply. Needs [`Self::set_atom_maxmove`].
    /// Unset, a Wolfe session with an explicit per-atom cap takes this
    /// step; off keeps its strong Wolfe search under the cap.
    /// [`Self::set_lbfgs_step`] unsets it.
    pub fn set_geometry_step(&mut self, on: bool) {
        self.geometry_step = Some(on);
    }

    /// eOn `lbfgs_project_rigid`.
    ///
    /// Nonzero selects [`RigidMode::Auto`] and declares a molecule, so a
    /// host that also set the axis mask drops rotations on a cell. Zero
    /// forces the Euclidean path off. A quotient manifold still projects.
    pub fn set_project_rigid(&mut self, enabled: bool) {
        self.set_rigid_mode(if enabled {
            RigidMode::Auto
        } else {
            RigidMode::Off
        });
    }

    /// `Off`, `Auto`, `Translation`, or `TranslationRotation`.
    ///
    /// `Auto` declares a molecule: translation and rotation when no axis is
    /// periodic, translation on the periodic axes otherwise. Frozen
    /// coordinates keep the combinations that do not move them. Broken
    /// axes are a host mask, not a field vector.
    /// A bare Euclidean session stays off until something declares it.
    pub fn set_rigid_mode(&mut self, mode: RigidMode) {
        self.rigid_before_pbc = None;
        self.rigid_mode = mode;
        self.project_rigid = mode != RigidMode::Off;
        if mode == RigidMode::Off {
            self.rigid_forced_off = true;
            self.rigid_declared = false;
        } else {
            self.rigid_forced_off = false;
            self.rigid_declared = true;
        }
    }

    /// Cartesian mass inner product `sum m_i u_i · v_i`.
    ///
    /// Turns the mass-weighted chart off. A raw force should leave this false.
    pub fn set_rigid_mass_metric(&mut self, enabled: bool) {
        self.mass_metric = enabled;
        if enabled {
            self.mass_weighted_coords = false;
        }
    }

    /// Coordinates are `q_i = sqrt(m_i) x_i`. The metric is Euclidean.
    ///
    /// Turns the Cartesian mass metric off. The two switches are not both on.
    pub fn set_rigid_mass_weighted(&mut self, enabled: bool) {
        self.mass_weighted_coords = enabled;
        if enabled {
            self.mass_metric = false;
            self.declare_molecule();
        }
    }

    /// Periodic axes. Any true axis drops rotations and keeps only those
    /// translations. This replaces the single bit from [`Self::set_periodic`].
    /// Clearing every axis undoes a declaration the axes alone made.
    pub fn set_pbc_axes(&mut self, x: bool, y: bool, z: bool) {
        self.pbc = [x, y, z];
        self.periodic = x || y || z;
        if self.periodic {
            let before = self.rigid_before_pbc.or_else(|| {
                (!self.rigid_declared).then_some((
                    self.rigid_mode,
                    self.project_rigid,
                    self.rigid_declared,
                    self.rigid_forced_off,
                ))
            });
            self.declare_molecule();
            self.rigid_before_pbc = before;
        } else if let Some((mode, project, declared, forced_off)) = self.rigid_before_pbc.take() {
            self.rigid_mode = mode;
            self.project_rigid = project;
            self.rigid_declared = declared;
            self.rigid_forced_off = forced_off;
        }
    }

    /// Length `3N`. A nonzero byte freezes that coordinate. The basis is
    /// the rigid combinations that are already zero there. Empty clears
    /// the mask.
    pub fn set_frozen(&mut self, frozen: &[u8]) {
        if frozen.is_empty() {
            self.frozen = None;
            return;
        }
        self.frozen = Some(frozen.to_vec());
        self.declare_molecule();
    }

    /// Axes that are not symmetries. The library does not infer them
    /// from a force or a field. Any set bit declares a molecule.
    pub fn set_broken_axes(
        &mut self,
        trans_x: bool,
        trans_y: bool,
        trans_z: bool,
        rot_x: bool,
        rot_y: bool,
        rot_z: bool,
    ) {
        self.broken_translation = [trans_x, trans_y, trans_z];
        self.broken_rotation = [rot_x, rot_y, rot_z];
        if trans_x || trans_y || trans_z || rot_x || rot_y || rot_z {
            self.declare_molecule();
        }
    }

    fn declare_molecule(&mut self) {
        self.rigid_before_pbc = None;
        self.rigid_declared = true;
        self.rigid_forced_off = false;
        if self.rigid_mode == RigidMode::Off {
            self.rigid_mode = RigidMode::Auto;
        }
        self.project_rigid = true;
    }

    /// Periodic cell. Sella leaves `proj_rot` off; the quotient is \(R^{3N}/T(3)\).
    ///
    /// Nonzero is [`Self::set_pbc_axes`]`(true, true, true)` and declares
    /// the molecule. Zero clears the three axes and undoes that
    /// declaration unless something else declared the molecule since.
    pub fn set_periodic(&mut self, periodic: bool) {
        self.set_pbc_axes(periodic, periodic, periodic);
    }

    /// Embedded manifold for project / retract / transport.
    ///
    /// A [`ManifoldKind::Product`] with a weight other than one steps in
    /// its metric chart, [`crate::manifold::ProductManifold::chart`]. Points,
    /// gradients and reports stay in the coordinates of the factors; the
    /// chart only changes the inner product every method measures in to the
    /// weighted product metric. Settings that read raw coordinates as atoms
    /// or boxes (the exponential preconditioner, the Lindh model, rigid
    /// projection of the whole vector, the HiGHS box) refuse such a product
    /// with [`Error::ProductChart`]. Steps refuse a product with a
    /// [`crate::manifold::TangentMetric`] with
    /// [`Error::ProductTangentMetric`]; [`crate::rtr::minimize_rtr`] runs
    /// in its metric.
    pub fn set_manifold(&mut self, kind: ManifoldKind) {
        let kind = match (kind, self.factor_shape) {
            (ManifoldKind::Grassmann, Some((n, p))) => ManifoldKind::GrassmannP { n, p },
            (other, _) => other,
        };
        let (kind, chart) = chart::chart_form(kind);
        if kind != self.manifold || chart != self.chart {
            self.forget();
            self.tangent = TangentModel::default();
            self.chart_seen = None;
        }
        self.manifold = kind;
        self.chart = chart;
        self.apply_default_atom_cap();
    }

    /// Weighted product of `factors`, \(g = \sum_i w_i g_i\).
    /// See [`crate::manifold::ProductManifold::new`] and
    /// [`Self::set_manifold`].
    pub fn set_product<I>(&mut self, factors: I) -> Result<()>
    where
        I: IntoIterator<Item = crate::manifold::ProductFactor>,
    {
        self.set_manifold(ManifoldKind::product(factors)?);
        Ok(())
    }

    /// `count` copies of `factor`, each of packed length `block`, with
    /// unit weights. manopt `powermanifold`. Returns `false`, keeping the
    /// manifold, when `factor` does not fit `block`. MwRigid copies read
    /// [`Self::set_masses`] as one molecule reused on every copy or as
    /// every copy's masses in order.
    pub fn set_power(&mut self, factor: ManifoldKind, block: usize, count: usize) -> bool {
        match ManifoldKind::power(factor, block, count) {
            Ok(kind) => {
                self.set_manifold(kind);
                true
            }
            Err(_) => false,
        }
    }

    /// Stiefel \(\mathrm{St}(n,p)\). `p = 1` is the sphere packing.
    /// `p > 1` is column-major, length `n*p`.
    pub fn set_stiefel(&mut self, n: usize, p: usize) {
        self.set_manifold(ManifoldKind::stiefel(n, p));
    }

    /// Real Grassmann `Gr(n,p)`, packed column-major.
    pub fn set_grassmann(&mut self, n: usize, p: usize) {
        self.set_manifold(ManifoldKind::GrassmannP { n, p });
    }

    /// Oblique \(\mathrm{OB}(n,m)\): product of `m` unit spheres in `R^n`.
    /// Packed column-major, length `n*m`.
    pub fn set_oblique(&mut self, n: usize, m: usize) {
        self.set_manifold(ManifoldKind::Oblique { n, m });
    }

    /// Product of `n` unit-modulus complex numbers. Packed length `2 n`.
    pub fn set_complex_circle(&mut self, n: usize) {
        self.set_manifold(ManifoldKind::ComplexCircle { n });
    }

    /// Complex Euclidean \(\mathbb{C}^n\). Packed interleaved, length `2 n`.
    pub fn set_euclidean_complex(&mut self, n: usize) {
        self.set_manifold(ManifoldKind::EuclideanComplex { n });
    }

    /// Singleton of packed length `n`. manopt `constantfactory`.
    pub fn set_constant(&mut self, n: usize) {
        self.set_manifold(ManifoldKind::Constant { n });
    }

    /// Doubly-stochastic n-by-n, packed length `n^2`.
    /// manopt `multinomialdoublystochasticfactory`.
    pub fn set_multinomial_ds(&mut self, n: usize) {
        self.set_manifold(ManifoldKind::MultinomialDoublyStochastic { n });
    }

    /// Symmetric doubly-stochastic n-by-n, packed length `n^2`.
    /// manopt `multinomialsymmetricfactory`.
    pub fn set_multinomial_sym(&mut self, n: usize) {
        self.set_manifold(ManifoldKind::MultinomialSymmetric { n });
    }

    /// Complex unit sphere in \(\mathbb{C}^n\). Packed interleaved,
    /// length `2 n`. manopt `spherecomplexfactory`.
    pub fn set_sphere_complex(&mut self, n: usize) {
        self.set_manifold(ManifoldKind::SphereComplex { n });
    }

    /// Positive orthant of packed length `n`. manopt `positivefactory`.
    pub fn set_positive(&mut self, n: usize) {
        self.set_manifold(ManifoldKind::Positive { n });
    }

    /// Centered `m x n` matrices, packed row-major `m n`.
    /// manopt `centeredmatrixfactory`. `rows = false` centers
    /// columns (`X 1 = 0`).
    pub fn set_centered_matrix(&mut self, m: usize, n: usize, rows: bool) {
        self.set_manifold(ManifoldKind::CenteredMatrix { m, n, rows });
    }

    /// Set Gr(n,p); a zero dimension clears the optional factor shape.
    pub fn set_factor_shape(&mut self, n: usize, p: usize) {
        let next = if n == 0 || p == 0 { None } else { Some((n, p)) };
        if next != self.factor_shape {
            self.forget();
        }
        self.factor_shape = next;
        if matches!(
            self.manifold,
            ManifoldKind::Grassmann | ManifoldKind::GrassmannP { .. }
        ) {
            self.manifold = match next {
                Some((n, p)) => ManifoldKind::GrassmannP { n, p },
                None => ManifoldKind::Grassmann,
            };
        }
    }

    /// Complex unitary U(n), packed interleaved row-major.
    pub fn set_unitary(&mut self, n: usize) {
        self.set_manifold(ManifoldKind::unitary(n));
    }

    /// Per-atom masses for [`ManifoldKind::MwRigid`] (Page–McIver / Eckart).
    /// Empty clears them (unit mass). A product reads them per MwRigid
    /// factor, as in [`Self::set_power`].
    pub fn set_masses(&mut self, masses: Array1<f64>) {
        if masses.is_empty() {
            self.masses = None;
        } else {
            self.masses = Some(masses);
        }
    }

    /// Al-Baali extra-updates on the newest L-BFGS pair. No effect: a
    /// replayed newest pair leaves the BFGS map unchanged (see
    /// [`Lbfgs::extra_updates`]).
    pub fn set_extra_updates(&mut self, extra: usize) {
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.extra_updates = extra;
        }
    }

    /// Li-Fukushima cautious pair filter. `eps <= 0` disables it.
    pub fn set_cautious(&mut self, eps: f64, alpha: f64) {
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.cautious_eps = eps;
            solver.cautious_alpha = alpha;
        }
    }

    /// Enable the opt-in direct NEB guards and empty-memory scale 0.01.
    /// Other methods ignore this setting. Disabling restores scale one.
    pub fn set_lbfgs_neb_guards(&mut self, enabled: bool) {
        if let Inner::Lbfgs(solver) = &mut self.inner {
            if solver.curvature_reset != enabled {
                solver.forget();
            }
            solver.distance_reset = enabled;
            solver.angle_reset = enabled;
            solver.curvature_reset = enabled;
            solver.inverse_curvature = if enabled { 0.01 } else { 1.0 };
        }
    }

    /// HiGHS feasible-set step. No-op unless this build has `highs`.
    pub fn set_highs(&mut self, enabled: bool) {
        #[cfg(not(feature = "highs"))]
        let _ = enabled;
        #[cfg(feature = "highs")]
        {
            self.highs = enabled;
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.highs_options = self.highs_options;
                solver.highs = if enabled {
                    Some(crate::HighsStep {
                        trust: self
                            .highs_trust
                            .or(self.atom_maxmove)
                            .or(self.control.maxmove_cap()),
                        lo: None,
                        hi: None,
                        equalities: self.equalities.clone(),
                        center_axes: None,
                    })
                } else {
                    None
                };
            }
        }
    }

    /// Select a HiGHS engine. Returns false when the feature is absent.
    pub fn set_highs_solver(&mut self, solver: crate::HighsSolverKind) -> bool {
        #[cfg(feature = "highs")]
        {
            self.highs_options.solver = solver;
            self.set_highs(self.highs);
            true
        }
        #[cfg(not(feature = "highs"))]
        {
            let _ = solver;
            false
        }
    }

    /// Select a HiGHS crossover policy. Returns false when the feature is absent.
    pub fn set_highs_crossover(&mut self, crossover: crate::HighsCrossover) -> bool {
        #[cfg(feature = "highs")]
        {
            self.highs_options.crossover = crossover;
            self.set_highs(self.highs);
            true
        }
        #[cfg(not(feature = "highs"))]
        {
            let _ = crossover;
            false
        }
    }

    /// Set or clear a host callback. Returns false when HiGHS is absent.
    pub fn set_highs_callback(&mut self, callback: Option<crate::HighsCallback>) -> bool {
        #[cfg(feature = "highs")]
        {
            self.highs_options.callback = callback;
            self.set_highs(self.highs);
            true
        }
        #[cfg(not(feature = "highs"))]
        {
            let _ = callback;
            false
        }
    }

    /// Per-coordinate box. Empty or missing sides are unbounded; length one broadcasts.
    /// Bounds are enforced when HiGHS constraints are enabled.
    pub fn set_box(&mut self, lo: Option<Vec<f64>>, hi: Option<Vec<f64>>) -> bool {
        #[cfg(not(feature = "highs"))]
        {
            let _ = (lo, hi);
            false
        }
        #[cfg(feature = "highs")]
        {
            self.box_lo = lo;
            self.box_hi = hi;
            true
        }
    }

    /// L-infinity trust radius on the constrained model step.
    /// Non-positive or non-finite radii disable the radius.
    pub fn set_trust(&mut self, radius: f64) -> bool {
        #[cfg(not(feature = "highs"))]
        {
            let _ = radius;
            false
        }
        #[cfg(feature = "highs")]
        {
            self.highs_trust = (radius > 0.0 && radius.is_finite()).then_some(radius);
            self.set_highs(self.highs);
            true
        }
    }

    /// Alias for [`Self::set_trust`], with the same radius and feature contract.
    pub fn set_highs_trust(&mut self, radius: f64) -> bool {
        self.set_trust(radius)
    }

    /// Append a sparse linear equality on the model and accepted displacement.
    /// A line search or retraction that violates a row returns `Error::Highs`,
    /// restores the starting point, and clears method memory. The residual
    /// tolerance is 1e-7 in row units plus floating-point summation error.
    /// Reject non-finite coefficients and indices outside the session dimension.
    pub fn add_equality(&mut self, coefficients: Vec<(usize, f64)>, rhs: f64) -> bool {
        #[cfg(not(feature = "highs"))]
        {
            let _ = (coefficients, rhs);
            false
        }
        #[cfg(feature = "highs")]
        {
            if !rhs.is_finite()
                || coefficients
                    .iter()
                    .any(|(k, a)| *k >= self.dim || !a.is_finite())
            {
                return false;
            }
            self.equalities.push((coefficients, rhs));
            self.set_highs(self.highs);
            true
        }
    }

    /// Remove every model-step equality.
    pub fn clear_equalities(&mut self) -> bool {
        #[cfg(not(feature = "highs"))]
        {
            false
        }
        #[cfg(feature = "highs")]
        {
            self.equalities.clear();
            self.set_highs(self.highs);
            true
        }
    }

    fn has_equalities(&self) -> bool {
        #[cfg(feature = "highs")]
        {
            !self.equalities.is_empty()
        }
        #[cfg(not(feature = "highs"))]
        {
            false
        }
    }

    fn has_coordinate_box(&self) -> bool {
        #[cfg(feature = "highs")]
        {
            self.highs && (self.box_lo.is_some() || self.box_hi.is_some())
        }
        #[cfg(not(feature = "highs"))]
        {
            false
        }
    }

    fn note_bounds(&mut self, bounds: &eindir_core::Bounds<f64>) {
        if self.box_stationarity() {
            self.bound_low = Some(bounds.low.clone());
            self.bound_high = Some(bounds.high.clone());
        }
    }

    fn box_stationarity(&self) -> bool {
        self.has_coordinate_box()
            && self.manifold == ManifoldKind::Euclidean
            && self.removal_mode() == RigidMode::Off
            && !self.has_equalities()
    }

    /// Stationarity of `grad` at `x`, in [`Self::stop_norm`].
    ///
    /// A Euclidean session with a coordinate box and no other constraint
    /// measures the projected gradient `grad.clamp(x - hi, x - lo)`.
    fn stationarity_norm(&self, x: &Array1<f64>, grad: &Array1<f64>) -> f64 {
        self.stationarity_norm_in(x, grad, self.bound_low.as_ref(), self.bound_high.as_ref())
    }

    fn stationarity_norm_in(
        &self,
        x: &Array1<f64>,
        grad: &Array1<f64>,
        bound_low: Option<&Array1<f64>>,
        bound_high: Option<&Array1<f64>>,
    ) -> f64 {
        if self.box_stationarity()
            && let (Some(lo), Some(hi)) = (bound_low, bound_high)
        {
            let mut projected = Array1::zeros(x.len());
            for k in 0..x.len() {
                let lo_k = lo.get(k).copied().unwrap_or(f64::NEG_INFINITY);
                let hi_k = hi.get(k).copied().unwrap_or(f64::INFINITY);
                if !(x[k].is_finite() && grad[k].is_finite() && lo_k <= x[k] && x[k] <= hi_k) {
                    return f64::INFINITY;
                }
                // Distance clipping retains gradients smaller than an ulp of x.
                projected[k] = grad[k].clamp(x[k] - hi_k, x[k] - lo_k);
            }
            return self.stop_norm.of(projected.view());
        }
        self.stop_norm.of(grad.view())
    }

    fn noted_free_len(&self) -> usize {
        if self.box_stationarity()
            && let (Some(lo), Some(hi)) = (&self.bound_low, &self.bound_high)
        {
            let n = self.dim.min(lo.len()).min(hi.len());
            return (0..n).filter(|&k| hi[k] > lo[k]).count();
        }
        self.dim
    }

    fn finish(
        &self,
        value: f64,
        x: &Array1<f64>,
        grad: &Array1<f64>,
        grad_norm: f64,
        status: crate::report::Status,
    ) -> Report {
        Report {
            value,
            coords: if self.brief {
                Array1::zeros(0)
            } else {
                x.clone()
            },
            steps: self.steps,
            grad_norm,
            stop_norm: self.stop_norm,
            grad: Some(grad.clone()),
            status,
            reason: None,
        }
    }

    fn report(&self, value: f64, x: &Array1<f64>, grad: &Array1<f64>) -> Report {
        let grad_norm = self.stationarity_norm(x, grad);
        let status = self.control.classify(grad_norm, self.noted_free_len());
        self.finish(value, x, grad, grad_norm, status)
    }

    /// Coordinates the stationarity measure runs over.
    ///
    /// A coordinate box drops entries whose bounds have no width. Every
    /// other path uses the length of the horizontal or projected gradient,
    /// which is the session dimension.
    pub(crate) fn free_len(&self, bounds: &eindir_core::Bounds<f64>) -> usize {
        if self.has_coordinate_box()
            && self.manifold == ManifoldKind::Euclidean
            && self.removal_mode() == RigidMode::Off
            && !self.has_equalities()
        {
            let n = self.dim.min(bounds.low.len()).min(bounds.high.len());
            return (0..n).filter(|&k| bounds.high[k] > bounds.low[k]).count();
        }
        self.dim
    }

    /// Active Eckart mode. Quotient manifolds project even when the
    /// Euclidean flag is off. A bare Euclidean session does not.
    fn removal_mode(&self) -> RigidMode {
        let quotient = matches!(
            self.manifold,
            ManifoldKind::RigidQuotient | ManifoldKind::MwRigid
        );
        if quotient {
            if self.rigid_mode == RigidMode::Off {
                RigidMode::Auto
            } else {
                self.rigid_mode
            }
        } else if self.rigid_forced_off || !self.rigid_declared {
            RigidMode::Off
        } else {
            self.rigid_mode
        }
    }

    /// `MwRigid` stores `q = sqrt(m) x`. That chart uses the Euclidean product.
    fn chart_weighted(&self) -> bool {
        self.mass_weighted_coords || matches!(self.manifold, ManifoldKind::MwRigid)
    }

    /// Cartesian force. The mass metric is not this switch.
    fn force_metric(&self) -> bool {
        self.mass_metric && !self.chart_weighted()
    }

    /// FIRE velocity. Momentum is `m v` on raw Cartesians.
    fn velocity_metric(&self) -> bool {
        !self.chart_weighted() && (self.mass_metric || self.masses.is_some())
    }

    fn anchor_and_project_into(&self, ws: &mut RigidWorkspace, x: &[f64], v: &mut [f64]) {
        let mode = self.removal_mode();
        if mode == RigidMode::Off || v.len() != x.len() {
            return;
        }
        let metric = self.force_metric();
        let weighted = self.chart_weighted();
        let pbc = self.pbc;
        let broken_t = self.broken_translation;
        let broken_r = self.broken_rotation;
        ws.set_anchor(x);
        let masses = self.masses.as_ref().and_then(|m| m.as_slice());
        let frozen = self.frozen.as_deref();
        ws.project_at_anchor(
            v, masses, frozen, weighted, metric, mode, pbc, broken_t, broken_r,
        );
    }

    fn kill_at_into(&self, ws: &mut RigidWorkspace, x: &Array1<f64>, v: &mut Array1<f64>) {
        let Some(xs) = x.as_slice() else {
            return;
        };
        let Some(vs) = v.as_slice_mut() else {
            return;
        };
        self.anchor_and_project_into(ws, xs, vs);
    }

    /// Remove the rigid motion of `v` at the geometry `x`.
    fn kill_at(&mut self, x: &Array1<f64>, v: &mut Array1<f64>) {
        let mut ws = std::mem::take(&mut self.rigid_ws);
        self.kill_at_into(&mut ws, x, v);
        self.rigid_ws = ws;
    }

    fn check_manifold(&self, n: usize) -> Result<()> {
        if self.manifold.required_dim(n).is_err() {
            return Err(Error::ManifoldDim {
                kind: self.manifold.as_str(),
                got: n,
            });
        }
        self.refuse_tangent_metric()
    }

    fn refuse_tangent_metric(&self) -> Result<()> {
        match &self.manifold {
            ManifoldKind::Product(product) => product
                .tangent_metric_factor()
                .map_or(Ok(()), |index| Err(Error::ProductTangentMetric { index })),
            _ => Ok(()),
        }
    }

    /// Tangent projection. Periodic cells drop rotation (Sella `proj_rot`).
    fn project_vec(&mut self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut w = match &self.manifold {
            ManifoldKind::MwRigid | ManifoldKind::RigidQuotient => v.clone(),
            ManifoldKind::Product(product) if product.has_rigid() => {
                let mut w = product.project(x, v);
                self.project_rigid_blocks(product, x, v, &mut w);
                w
            }
            other => other.project(x, v),
        };
        self.kill_at(x, &mut w);
        w
    }

    /// Dogleg model on a curved kind, for the tangent projection `P` at `x`.
    ///
    /// On the sphere `P = I - x x^T` and `(I - P)^T (I - P) = (x.x) x x^T`,
    /// so the model `P^T H P + (I - P)^T (I - P)` is
    /// `H - x (H^T x)^T - (H x) x^T + (x^T H x + x.x) x x^T`.
    /// That form costs O(n^2) and writes into `tangent` without building
    /// `P` or allocating. On the unit sphere `P` is the orthogonal
    /// projection and the second term is the identity on `span{x}`, so a
    /// tangent gradient gives tangent Newton and Cauchy points.
    ///
    /// The projection of another kind can be oblique, as the Fisher
    /// projection of the multinomial kinds is. The Newton point of the
    /// ambient form then leaves the tangent space. Those kinds keep an
    /// orthonormal basis `Z` of the tangent space and the model `Z^T H Z`,
    /// which stays symmetric for the LDLT solve.
    fn fill_tangent_model(&mut self, x: &Array1<f64>, hess: &Array2<f64>) {
        let n = x.len();
        if self.manifold != ManifoldKind::Sphere {
            let z = self.tangent_basis(x);
            self.tangent.model = z.t().dot(hess).dot(&z);
            self.tangent.basis = z;
            return;
        }
        let tm = &mut self.tangent;
        // Another kind leaves `Z^T H Z` in `model` and no work vectors.
        if tm.model.dim() != (n, n) || tm.hx.len() != n {
            tm.model = Array2::zeros((n, n));
            tm.hx = Array1::zeros(n);
            tm.htx = Array1::zeros(n);
        }
        let mut scale = 0.0;
        for i in 0..n {
            let (mut a, mut b) = (0.0, 0.0);
            for j in 0..n {
                a += hess[[i, j]] * x[j];
                b += hess[[j, i]] * x[j];
            }
            tm.hx[i] = a;
            tm.htx[i] = b;
            scale += x[i] * (a + x[i]);
        }
        for i in 0..n {
            for j in 0..n {
                tm.model[[i, j]] =
                    hess[[i, j]] - x[i] * tm.htx[j] - tm.hx[i] * x[j] + scale * x[i] * x[j];
            }
        }
    }

    /// The columns `P e_j` of the tangent projection at `x`.
    fn projection_columns(&self, x: &Array1<f64>) -> Array2<f64> {
        let n = x.len();
        let mut p = Array2::<f64>::zeros((n, n));
        let mut e = Array1::<f64>::zeros(n);
        for j in 0..n {
            e[j] = 1.0;
            p.column_mut(j).assign(&self.manifold.project(x, &e));
            e[j] = 0.0;
        }
        p
    }

    /// Orthonormal basis of the range of the tangent projection at `x`, by
    /// Gram-Schmidt with column pivoting on its columns. A projection has
    /// no nonzero singular value below one, so the basis stops at the first
    /// remainder no longer than `RANK` times the longest column.
    fn tangent_basis(&self, x: &Array1<f64>) -> Array2<f64> {
        const RANK: f64 = 1e-8;
        let mut cols = self.projection_columns(x);
        let n = x.len();
        let longest_col = (0..n)
            .map(|j| cols.column(j).dot(&cols.column(j)).sqrt())
            .fold(0.0, f64::max);
        let floor = RANK * longest_col;
        let mut left: Vec<usize> = (0..n).collect();
        let mut basis: Vec<Array1<f64>> = Vec::with_capacity(n);
        while !left.is_empty() {
            let (mut pos, mut longest) = (0, -1.0);
            for (i, &j) in left.iter().enumerate() {
                let len = cols.column(j).dot(&cols.column(j)).sqrt();
                if len > longest {
                    (pos, longest) = (i, len);
                }
            }
            if longest <= floor {
                break;
            }
            let mut q = cols.column(left.remove(pos)).to_owned();
            for b in &basis {
                let c = b.dot(&q);
                q.scaled_add(-c, b);
            }
            let len = l2(&q);
            q.mapv_inplace(|v| v / len);
            for &j in &left {
                let mut col = cols.column_mut(j);
                let c = q.dot(&col);
                col.scaled_add(-c, &q);
            }
            basis.push(q);
        }
        let mut z = Array2::<f64>::zeros((n, basis.len()));
        for (k, b) in basis.iter().enumerate() {
            z.column_mut(k).assign(b);
        }
        z
    }

    /// Dogleg point at `x` within `radius`. A curved kind other than the
    /// sphere runs the dogleg on `Z^T H Z` and `Z^T g` and maps the point
    /// back with `Z`, so the Newton and Cauchy points are tangent and the
    /// radius bounds `|Z u| = |u|`.
    fn dogleg_point(
        &mut self,
        x: &Array1<f64>,
        hess: &Array2<f64>,
        grad: &Array1<f64>,
        radius: f64,
    ) -> Array1<f64> {
        if self.manifold.is_flat() {
            return dogleg_direction(hess, grad, radius);
        }
        self.fill_tangent_model(x, hess);
        let tm = &self.tangent;
        if self.manifold == ManifoldKind::Sphere {
            return dogleg_direction(&tm.model, grad, radius);
        }
        let u = dogleg_direction(&tm.model, &tm.basis.t().dot(grad), radius);
        tm.basis.dot(&u)
    }

    /// Predicted reduction of the taken step `p` on the model of
    /// [`Self::dogleg_point`]. With `Q = Z Z^T` the basis model is
    /// `Q H Q + (I - Q)` in ambient form, so the normal part of `p` enters
    /// with the identity, as on the sphere.
    fn dogleg_pred(&self, hess: &Array2<f64>, grad: &Array1<f64>, p: &Array1<f64>) -> f64 {
        if self.manifold.is_flat() {
            return predicted_reduction(hess, grad, p);
        }
        let tm = &self.tangent;
        if self.manifold == ManifoldKind::Sphere {
            return predicted_reduction(&tm.model, grad, p);
        }
        let u = tm.basis.t().dot(p);
        let normal = p.dot(p) - u.dot(&u);
        -grad.dot(p) - 0.5 * (u.dot(&tm.model.dot(&u)) + normal)
    }

    fn horizontal_grad_into(
        &self,
        ws: &mut RigidWorkspace,
        x: &Array1<f64>,
        grad: &Array1<f64>,
    ) -> Array1<f64> {
        let mut g = match &self.manifold {
            ManifoldKind::MwRigid | ManifoldKind::RigidQuotient => grad.clone(),
            ManifoldKind::Product(product) if product.has_rigid() => {
                let mut g = product.egrad2rgrad(x, grad);
                self.project_rigid_blocks(product, x, grad, &mut g);
                g
            }
            other => other.egrad2rgrad(x, grad),
        };
        self.kill_at_into(ws, x, &mut g);
        g
    }

    fn horizontal_grad(&mut self, x: &Array1<f64>, grad: &Array1<f64>) -> Array1<f64> {
        let mut ws = std::mem::take(&mut self.rigid_ws);
        let g = self.horizontal_grad_into(&mut ws, x, grad);
        self.rigid_ws = ws;
        g
    }

    /// Vector transport. Quotient manifolds project at the arrival point.
    fn transport_vec(
        &mut self,
        x_from: &Array1<f64>,
        x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        match &self.manifold {
            ManifoldKind::RigidQuotient | ManifoldKind::MwRigid => self.project_vec(x_to, v),
            ManifoldKind::Product(product) if product.has_rigid() => {
                let mut w = product.transport(x_from, x_to, v);
                self.project_rigid_blocks(product, x_to, v, &mut w);
                w
            }
            other => other.transport(x_from, x_to, v),
        }
    }

    fn project_level_set_vec(&self, grad: &Array1<f64>, v: &Array1<f64>) -> Result<Array1<f64>> {
        if self.level_set.is_none() {
            return Ok(v.clone());
        }
        project_tangent(grad.view(), v)
    }

    fn project_level_set_pair(
        &self,
        grad: &Array1<f64>,
        s: &mut Array1<f64>,
        y: &mut Array1<f64>,
    ) -> Result<()> {
        if self.level_set.is_none() {
            return Ok(());
        }
        *s = project_tangent(grad.view(), s)?;
        *y = project_tangent(grad.view(), y)?;
        Ok(())
    }

    fn retract_level_set_here<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        value: &mut f64,
        grad: &mut Array1<f64>,
    ) -> Result<()>
    where
        O: StepOracle + ?Sized,
    {
        let Some(cfg) = self.level_set else {
            return Ok(());
        };
        let retracted =
            retract_to_level_set_fallible(cfg, x, |point| obj.step_value_and_gradient(point))?;
        *x = retracted.point;
        *value = retracted.value;
        *grad = retracted.gradient;
        Ok(())
    }

    fn retract_level_set_start<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<()>
    where
        O: StepOracle + ?Sized,
    {
        if self.level_set.is_none() {
            return Ok(());
        }
        let had = x.clone();
        let mut value = 0.0;
        let mut grad = Array1::zeros(x.len());
        self.retract_level_set_here(obj, x, &mut value, &mut grad)?;
        if x.iter()
            .zip(had.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            self.last_pos = None;
            self.remember(x, value, &grad);
        }
        Ok(())
    }

    /// Pull `x` onto the level set after a step that did not, and rebuild
    /// the report from that point. A no-op when no level set is set, and
    /// when the step already landed inside the tolerance, so an unset
    /// session keeps the bytes of the path that produced `report`.
    fn finish_level_set<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        report: &mut Report,
    ) -> Result<()>
    where
        O: StepOracle + ?Sized,
    {
        if self.level_set.is_none() {
            return Ok(());
        }
        let before = x.clone();
        let mut value = 0.0;
        let mut grad = Array1::zeros(x.len());
        self.retract_level_set_here(obj, x, &mut value, &mut grad)?;
        if x.iter()
            .zip(before.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Ok(());
        }
        grad = self.horizontal_grad(x, &grad);
        self.remember(x, value, &grad);
        let grad_norm = self.stationarity_norm(x, &grad);
        let status = self.control.classify(grad_norm, self.noted_free_len());
        report.value = value;
        if !self.brief {
            report.coords = x.clone();
        }
        report.grad_norm = grad_norm;
        report.grad = Some(grad);
        report.status = status;
        Ok(())
    }

    /// Report the current point when the tangent step vanished.
    fn level_set_stay<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: StepOracle + ?Sized,
    {
        let (value, raw) = obj.step_value_and_gradient(x.view())?;
        if !crate::step::finite(value, raw.view()) {
            return Err(Error::LevelSet {
                what: "non-finite value or gradient during level-set retraction",
            });
        }
        let grad = self.horizontal_grad(x, &raw);
        self.remember(x, value, &grad);
        self.steps += 1;
        Ok(self.report(value, x, &grad))
    }

    /// Project `step` onto the level-set tangent. `None` is a zero tangent
    /// step. The input is returned unchanged when no level set is set.
    fn tangent_displacement(
        &self,
        grad: &Array1<f64>,
        step: Array1<f64>,
    ) -> Result<Option<Array1<f64>>> {
        if self.level_set.is_none() {
            return Ok(Some(step));
        }
        let step = project_tangent(grad.view(), &step)?;
        let motion = max_atom_displacement(&step);
        if !(motion.is_finite() && motion > 0.0) {
            return Ok(None);
        }
        Ok(Some(step))
    }

    /// Riemannian L-BFGS pair at `x`: `s = T(x - old)`, `y = g - T(g_old)`.
    fn lbfgs_sy(
        &mut self,
        old: &Array1<f64>,
        x: &Array1<f64>,
        gold: &Array1<f64>,
        grad: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut s = self.transport_vec(old, x, &(x - old));
        let mut y = grad - &self.transport_vec(old, x, gold);
        self.kill_at(x, &mut s);
        self.kill_at(x, &mut y);
        (s, y)
    }

    /// True when `x` is the iterate this session last evaluated, up to
    /// a host's round trip of the coordinates.
    ///
    /// The test is relative, `|a - b| <= 4 eps max(|a|, |b|) + 1e-15`: a
    /// host that copies positions through its own storage (eOn's
    /// `Matter`, rgsaddle's band resync) can return a coordinate of
    /// 30 Angstrom a few ulps (3.6e-15 each) away, which an absolute
    /// 1e-15 test reads as a new geometry and pays a force call for.
    fn same_last_x(&self, x: &Array1<f64>) -> bool {
        match &self.last_pos {
            Some(p) if p.len() == x.len() => p
                .iter()
                .zip(x.iter())
                .all(|(a, b)| (*a - *b).abs() <= 4.0 * f64::EPSILON * a.abs().max(b.abs()) + 1e-15),
            _ => false,
        }
    }

    fn remember(&mut self, x: &Array1<f64>, value: f64, grad: &Array1<f64>) {
        self.last_pos = Some(x.clone());
        self.last_value = value;
        self.last_grad = grad.clone();
    }

    /// FIRE schedule for a FIRE session; a no-op on any other method.
    ///
    /// [`FireVariant::Guenole2020`] replaces the session's FIRE state with
    /// the published FIRE 2.0 ([`crate::fire::fire2_displacement`]) and
    /// Table 1 of Comput. Mater. Sci. 175 (2020) 109584,
    /// DOI 10.1016/j.commatsci.2020.109584, from `dt = Control::istep`;
    /// [`FireVariant::Rgmin`] restores [`FireState::new`] for the kind the
    /// session was built with. Either way the velocity starts at zero.
    ///
    /// On the rgpot benchmark (10 starts, per-atom cap 0.2) the published
    /// schedule halves the force calls of the Pt7 island and the EAM Al
    /// slab and costs 1.5 times as many on LJ38 from random packings,
    /// which is why it is a choice and not the default.
    pub fn set_fire_variant(&mut self, variant: FireVariant) {
        let dim = self.dim;
        let dt0 = self.control.istep;
        let fire = matches!(self.inner, Inner::Fire(_));
        let mut produced = None;
        if let Inner::Fire(state) = &mut self.inner {
            match variant {
                FireVariant::Guenole2020 => {
                    // Table 1 parameters on the state, and the published
                    // integrator (vdfmax) beside it.
                    state.set_preset(FirePreset::Guenole2020);
                    let (_, extras) = guenole2020(dim, state.dt);
                    produced = Some(Some(extras));
                }
                _ => {
                    let kind = state.kind;
                    *state = FireState::new(kind, dim, dt0);
                    produced = Some(None);
                }
            }
        }
        if let Some(extras) = produced {
            self.fire2 = extras;
            // FIRE 2.0 leaves the implicit cap off. Switching the schedule
            // updates that choice unless the caller already set a cap.
            if fire {
                self.apply_default_atom_cap();
            }
        }
    }

    /// Consecutive non-positive-power limit for the published FIRE 2.0
    /// schedule (`vdfmax`). The step that makes the count exceed
    /// `limit` returns [`Error::FireStalled`] and does not move.
    /// A positive power step, and [`Self::forget`], clear the count.
    /// A session that is not running that schedule ignores the call.
    /// The published schedule starts at 2000. `usize::MAX` does not stop.
    pub fn set_fire_vdfmax(&mut self, limit: usize) {
        if let Some(ext) = &mut self.fire2 {
            ext.n_neg_max = limit;
        }
    }

    /// Split quick-min into an atomic block `[0, at)` and a cell block
    /// `[at, n)`. Each block takes the reference projection on its own
    /// force, so a cell velocity that opposes the stress is zeroed even
    /// when the atoms keep the total power positive.
    pub fn set_quickmin_cell(&mut self, at: usize) -> Result<()> {
        if at == 0 || at >= self.dim {
            return Err(Error::QuickMin {
                what: "cell block must start after the atomic coordinates and before the end",
            });
        }
        match &mut self.overlay {
            Overlay::QuickMin(state, _) => {
                state.cell_at = Some(at);
                Ok(())
            }
            _ => Err(Error::QuickMin {
                what: "cell block requires a quick-min session",
            }),
        }
    }

    /// Turn the FIRE time-step controller on or off. The reference
    /// update is a fixed `dt` ([`crate::quickmin::quickmin_baseline`]).
    /// A session that is not quick-min ignores the call.
    pub fn set_quickmin_adapt(&mut self, enabled: bool) {
        if let Overlay::QuickMin(state, session) = &mut self.overlay {
            state.adapt_dt = enabled;
            session.options.adapt_dt = enabled;
        }
    }

    /// Quick-min options ([`QuickMinOptions`]); a no-op on any other
    /// method. The velocity and `dt` restart from [`Control::istep`];
    /// the cell split stays.
    pub fn set_quickmin_options(&mut self, options: QuickMinOptions) {
        if let Overlay::QuickMin(state, session) = &mut self.overlay {
            session.configure(state, options);
        }
    }

    /// Drop the cached evaluation and keep the method's memory.
    ///
    /// The next [`Self::step`] calls the oracle at its `x` even when `x`
    /// is the iterate this session last evaluated. A host whose oracle
    /// changes between steps calls this after the change: a min-mode
    /// walker returning `g - 2 (g . tau) tau` after refreshing `tau`, or
    /// a band after arming its climbing image. Without it the next step
    /// starts from the gradient of the old objective at the same `x`.
    ///
    /// FIRE velocity, BB and NLCG history, the dense quasi-Newton
    /// matrices and the L-BFGS pairs survive. Each stored pair is a
    /// secant pair of one objective: its `s` and both gradients of its
    /// `y` come from the same step, so the same `tau`, and this method
    /// keeps that true for the next pair by re-evaluating the start under
    /// the new oracle. What a pair no longer matches after the change is
    /// the current curvature: with `R = I - 2 tau tau'` the effective
    /// Hessian is `R H R`, which moves by `O(theta) ||H||` when `tau`
    /// rotates by `theta`. Small rotations between steps keep the memory
    /// useful; after a large one call [`Self::forget`].
    pub fn forget_evaluation(&mut self) {
        self.geometry_lbfgs_stiff = None;
        self.last_pos = None;
        if let Overlay::QuickMin(_, session) = &mut self.overlay {
            session.forget_last();
        }
    }

    /// Record `s = x+ - x`, `y = g+ - g` from the caller's previous outer.
    ///
    /// A dimer (or any one-oracle-per-outer walker) cannot answer the
    /// second eval [`Self::step`] would take at the trial point. The
    /// pair that belongs in L-BFGS memory is the one between successive
    /// outers, not a frozen-gradient inner trial (that writes `y = 0`
    /// and wrecks the two-loop).
    ///
    /// On a weighted product `s` is a displacement and `y` a difference of
    /// Euclidean gradients in the factors' coordinates. They enter the
    /// chart as `d * s` and `y / d`. A product with a tangent metric keeps
    /// no pairs.
    pub fn push_pair(&mut self, s: ArrayView1<f64>, y: ArrayView1<f64>) -> bool {
        if s.len() != self.dim || y.len() != self.dim || self.refuse_tangent_metric().is_err() {
            return false;
        }
        let charted = self
            .chart
            .as_ref()
            .filter(|d| d.len() == s.len())
            .map(|d| (&s * &**d, &y / &**d));
        let (s, y) = match &charted {
            Some((s, y)) => (s.view(), y.view()),
            None => (s, y),
        };
        if self.constraints.is_some() {
            let Ok((s, y)) = self.constrained_pair(s, y) else {
                return false;
            };
            return match &mut self.inner {
                Inner::Lbfgs(solver) => solver.push(s, y),
                _ => false,
            };
        }
        match &mut self.inner {
            Inner::Lbfgs(solver) => {
                solver.push(s.to_owned(), y.to_owned());
                true
            }
            _ => false,
        }
    }

    /// Number of accepted L-BFGS curvature pairs retained by the session.
    pub fn pair_count(&self) -> usize {
        match &self.inner {
            Inner::Lbfgs(solver) => solver.len(),
            _ => 0,
        }
    }

    /// Two-loop direction `d = -H g` with no evaluation and no push.
    ///
    /// On a weighted product `g` is a Euclidean gradient in the factors'
    /// coordinates and the direction is a tangent vector there.
    pub fn search_direction(&self, g: ArrayView1<f64>) -> Result<Array1<f64>> {
        if g.len() != self.dim {
            return Err(Error::Dim {
                dim: self.dim,
                got: g.len(),
            });
        }
        self.refuse_tangent_metric()?;
        if let Some(d) = self.chart.as_ref().filter(|d| d.len() == g.len()) {
            let dir = self.search_direction_plain((&g / &**d).view())?;
            return Ok(dir / &**d);
        }
        self.search_direction_plain(g)
    }

    fn search_direction_plain(&self, g: ArrayView1<f64>) -> Result<Array1<f64>> {
        if self.constraints.is_some() {
            return self.constrained_direction(g);
        }
        let dir = match &self.inner {
            // Two-loop only. search_direction would try HiGHS from
            // whatever x the caller did not pass; the dimer waist is
            // the Nocedal pair book, not a boxed QP.
            Inner::Lbfgs(solver) => solver.direction(g),
            _ => g.mapv(|v| -v),
        };
        if dir.iter().any(|v| !v.is_finite()) {
            return Err(Error::Oracle {
                what: "non-finite search direction",
            });
        }
        Ok(dir)
    }

    /// Keep method memory and discard the cached point and acceptance window.
    /// The next step evaluates the current objective at the supplied point
    /// with the initial step scale and the retained curvature information.
    pub fn rebase(&mut self) {
        self.istep = self.control.istep;
        self.e_hist.clear();
        self.last_pos = None;
        self.lbfgs_burdakov = self.lbfgs_burdakov_init;
        self.lbfgs_plateau = crate::lbfgs_plateau::RepeatedAcceptedStep::new();
        if let Overlay::QuickMin(_, session) = &mut self.overlay {
            session.forget_last();
        }
    }

    /// Drop method memory. The next step is a cold start from the current `x`.
    pub fn forget(&mut self) {
        self.saddle_kicks = 0;
        self.geometry_lbfgs_stiff = None;
        self.istep = self.control.istep;
        self.e_hist.clear();
        self.last_pos = None;
        self.lbfgs_burdakov = self.lbfgs_burdakov_init;
        self.lbfgs_plateau = crate::lbfgs_plateau::RepeatedAcceptedStep::new();
        let sr_radius = self.sr_opening_radius();
        match &mut self.inner {
            Inner::Lbfgs(solver) => solver.forget(),
            Inner::Nlcg {
                initialized,
                prev_alpha,
                prev_gd,
                ..
            } => {
                *initialized = false;
                *prev_alpha = 0.0;
                *prev_gd = 0.0;
            }
            Inner::Bfgs { h, fresh } => {
                *h = Array2::<f64>::eye(self.dim);
                *fresh = true;
            }
            Inner::Sr1 { h, fresh } => {
                *h = Array2::<f64>::eye(self.dim);
                *fresh = true;
            }
            Inner::Sr2 { b, fresh } => {
                *b = Array2::<f64>::eye(self.dim);
                *fresh = true;
            }
            Inner::Adam {
                m,
                v,
                b1p,
                b2p,
                beta1,
                beta2,
                ..
            } => {
                m.fill(0.0);
                v.fill(0.0);
                *b1p = *beta1;
                *b2p = *beta2;
            }
            Inner::Steepest => {}
            Inner::Pso { swarm, .. } => *swarm = None,
            Inner::Newton { .. } => {}
            Inner::Fire(state) => state.reset(),
            Inner::Bb { prev_s, prev_y } => {
                *prev_s = None;
                *prev_y = None;
            }
            Inner::Dogleg { radius } | Inner::NewtonCg { radius } | Inner::NewtonPcg { radius } => {
                *radius = self.control.istep.max(RADIUS_START);
            }
        }
        if matches!(self.inner, Inner::Sr1 { .. } | Inner::Sr2 { .. }) {
            self.sr_radius = sr_radius;
        }
        if let Some(ext) = &mut self.fire2 {
            ext.steps = 0;
            ext.n_neg = 0;
        }
        self.reset_overlay();
    }

    fn step_with_equalities<O, F>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        step: F,
    ) -> Result<Report>
    where
        O: Objective<f64> + ?Sized,
        F: FnOnce(&mut Self, &O, &mut Array1<f64>) -> Result<Report>,
    {
        #[cfg(feature = "highs")]
        let checkpoint = if self.highs && !self.equalities.is_empty() {
            if x.len() != self.dim || Objective::dim(obj) != self.dim {
                return Err(Error::Dim {
                    got: x.len(),
                    dim: self.dim,
                });
            }
            Some((obj.bounds().clip(x.view()), self.steps))
        } else {
            None
        };
        let result = step(self, obj, x);
        #[cfg(feature = "highs")]
        if let Some((start, steps)) = checkpoint
            && result.is_ok()
        {
            for (row, (coefficients, rhs)) in self.equalities.iter().enumerate() {
                let mut residual = -*rhs;
                let mut magnitude = rhs.abs();
                for &(k, a) in coefficients {
                    residual += a * (x[k] - start[k]);
                    magnitude += a.abs() * (x[k].abs() + start[k].abs());
                }
                // The QP tolerance is in row units. The gamma bound also
                // covers subtraction at a large coordinate origin.
                let roundoff = (2.0 * coefficients.len() as f64 + 4.0) * f64::EPSILON;
                let tolerance = crate::lbfgs_qp::EQUALITY_FEASIBILITY_TOLERANCE
                    + roundoff / (1.0 - roundoff) * magnitude;
                if !residual.is_finite() || !tolerance.is_finite() || residual.abs() > tolerance {
                    *x = start;
                    self.steps = steps;
                    self.forget();
                    return Err(Error::Highs(format!(
                        "accepted displacement violates equality {row}: residual {residual} exceeds {tolerance}"
                    )));
                }
            }
        }
        result
    }

    /// The window states how accurate the objective's values are, so it
    /// holds under every acceptance policy, as the default four-epsilon
    /// window it replaces does.
    fn search_options(&self) -> LineSearchOptions {
        if self.manifold == ManifoldKind::Euclidean {
            self.line_options
        } else {
            LineSearchOptions::default()
        }
    }

    /// A zero iteration budget is the SCG reading. The plateau itself is
    /// `lbfgs_plateau`: this only asks it, and only for L-BFGS. A positive
    /// budget does not ask, so its trajectory is unchanged.
    fn note_repeated_lbfgs_step(
        &mut self,
        before: &Array1<f64>,
        x: &Array1<f64>,
        report: &mut Report,
    ) {
        if self.control.maxiter != 0 || !matches!(self.inner, Inner::Lbfgs(_)) {
            return;
        }
        if report.status != Status::Iterating {
            return;
        }
        let moved = before.len() == x.len()
            && before
                .iter()
                .zip(x.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits());
        if self.lbfgs_plateau.observe(
            self.dim,
            moved,
            report.value,
            report.grad_norm,
            self.control.gtol,
            self.search_options().objective_window(),
        ) {
            report.status = Status::StepTolerance;
        }
    }

    /// One outer iteration. `x` is the iterate and is overwritten in place.
    ///
    /// With `control.maxiter == 0` an L-BFGS session reports
    /// [`Status::StepTolerance`] once `max(n, 2)` consecutive accepted steps
    /// move `x` while `||g||` stays at or above `gtol`, the value stays
    /// inside the objective roundoff window
    /// ([`Self::set_objective_roundoff`], else four epsilon relative) and
    /// `||g||` inside `sqrt(eps)` relative, both of the first of those
    /// steps. The step that reports it is taken. A caller's own loop reads
    /// it as the end of the fit: the objective cannot resolve the progress
    /// the steps make. A positive `maxiter` never reports it here, and
    /// [`Self::run`] keeps reading a zero `maxiter` as no steps.
    pub fn step<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if let Some(d) = self.chart.clone() {
            return self.in_chart(&d, obj, x, |s, obj, z| s.step_plain(obj, z));
        }
        self.step_plain(obj, x)
    }

    fn step_plain<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        self.retract_level_set_start(obj, x)?;
        let before = x.clone();
        let mut report = self.step_once(obj, x)?;
        self.finish_level_set(obj, x, &mut report)?;
        self.note_repeated_lbfgs_step(&before, x, &mut report);
        Ok(report)
    }

    /// One iteration, without the unlimited-budget plateau check.
    fn step_once<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if self.constraints.is_some() {
            return self.constrained_step(x, |s, x| s.step_constrained(obj, x));
        }
        if matches!(self.inner, Inner::NewtonCg { .. } | Inner::NewtonPcg { .. }) {
            return self.step_newton_cg(obj, x, &FiniteDifference);
        }
        if matches!(self.inner, Inner::Newton { .. } | Inner::Dogleg { .. }) {
            return Err(Error::NeedHessian);
        }
        #[cfg(feature = "highs")]
        {
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.coordinate_box = None;
            }
            if self.has_coordinate_box() {
                let bounded = crate::box_objective::BoxObjective::new(
                    obj,
                    self.box_lo.clone(),
                    self.box_hi.clone(),
                )?;
                if bounded.clip_start(x)? {
                    self.last_pos = None;
                }
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    // `StepOracle::bounds` is also in scope on every
                    // differentiable objective. This is the box itself.
                    solver.coordinate_box = Some(eindir_core::Objective::bounds(&bounded).clone());
                }
                return self.step_with_equalities(&bounded, x, Self::step_first_order);
            }
        }
        self.step_with_equalities(obj, x, Self::step_first_order)
    }

    /// One Newton / RFO iteration. Hessian is rebuilt at the current `x`.
    pub fn step_hess<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: HessianObjective + ?Sized,
    {
        if let Some(d) = self.chart.clone() {
            return self.in_chart(&d, obj, x, |s, obj, z| s.step_hess_plain(obj, z));
        }
        self.step_hess_plain(obj, x)
    }

    fn step_hess_plain<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: HessianObjective + ?Sized,
    {
        self.retract_level_set_start(obj, x)?;
        let mut report = if matches!(self.inner, Inner::NewtonCg { .. } | Inner::NewtonPcg { .. })
            && self.level_set.is_some()
        {
            // This verb has an analytic dense Hessian. Use it for the
            // constrained product instead of estimating one from gradients.
            let hess = obj.hessian(x.view());
            let action = |_: ArrayView1<f64>, v: ArrayView1<f64>| hess.dot(&v);
            if self.constraints.is_some() {
                self.constrained_step(x, |s, x| s.step_constrained_newton_cg(obj, x, &action))?
            } else {
                self.step_newton_cg(obj, x, &action)?
            }
        } else if self.constraints.is_some() {
            self.constrained_step(x, |s, x| s.step_constrained_hess(obj, x))?
        } else if matches!(self.inner, Inner::NewtonCg { .. } | Inner::NewtonPcg { .. }) {
            // Preserve the unconstrained path when no level set is active.
            self.step_once(obj, x)?
        } else {
            #[cfg(feature = "highs")]
            {
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.coordinate_box = None;
                }
                if self.has_coordinate_box() {
                    let bounded = crate::box_objective::BoxObjective::new(
                        obj,
                        self.box_lo.clone(),
                        self.box_hi.clone(),
                    )?;
                    if bounded.clip_start(x)? {
                        self.last_pos = None;
                    }
                    if let Inner::Lbfgs(solver) = &mut self.inner {
                        solver.coordinate_box =
                            Some(eindir_core::Objective::bounds(&bounded).clone());
                    }
                    self.step_with_equalities(&bounded, x, Self::step_hess_inner)?
                } else {
                    self.step_with_equalities(obj, x, Self::step_hess_inner)?
                }
            }
            #[cfg(not(feature = "highs"))]
            {
                self.step_with_equalities(obj, x, Self::step_hess_inner)?
            }
        };
        self.finish_level_set(obj, x, &mut report)?;
        Ok(report)
    }

    /// One Steihaug-Toint iteration. `obj` supplies analytic products;
    /// this verb does not build a dense Hessian.
    ///
    /// [`Self::step`] estimates products from gradient calls.
    /// [`Self::step_hess`] does the same unless a level set is active, where
    /// it multiplies the supplied dense Hessian.
    pub fn step_hvp<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: HessianVector + ?Sized,
    {
        if let Some(d) = self.chart.clone() {
            return self.in_chart(&d, obj, x, |s, obj, z| s.step_hvp_plain(obj, z));
        }
        self.step_hvp_plain(obj, x)
    }

    fn step_hvp_plain<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: HessianVector + ?Sized,
    {
        if !matches!(self.inner, Inner::NewtonCg { .. } | Inner::NewtonPcg { .. }) {
            return Err(Error::NeedHvp);
        }
        self.retract_level_set_start(obj, x)?;
        let action = |xv: ArrayView1<f64>, v: ArrayView1<f64>| obj.hessian_vector(xv, v);
        let mut report = if self.constraints.is_some() {
            self.constrained_step(x, |s, x| s.step_constrained_newton_cg(obj, x, &action))?
        } else {
            self.step_newton_cg(obj, x, &action)?
        };
        self.finish_level_set(obj, x, &mut report)?;
        Ok(report)
    }

    /// [`Self::step_hvp`] on an objective whose callbacks can fail.
    ///
    /// The step is the one [`Self::step_hvp`] takes on the same values,
    /// gradients and products. The first callback error ends the step
    /// as [`Error::Host`], evaluates nothing after it, and leaves `x`
    /// and the session (counters, bounds, trust radius and cached point)
    /// as they were before the step. A trial value the host returns as
    /// non-finite is refused by the ratio test, as on [`Self::step_hvp`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::step_hvp`], [`Error::Host`], and
    /// [`Error::FallibleUnsupported`] when this session is Newton-CG or
    /// Newton-PCG and a chart or distance constraints are set. Any other
    /// method returns [`Error::NeedHvp`] before those checks.
    pub fn try_step_hvp<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: TryHessianVector + ?Sized,
    {
        // The same refusal `step_hvp` makes, before a chart or a
        // constraint set, which this entry does not carry.
        if !matches!(self.inner, Inner::NewtonCg { .. } | Inner::NewtonPcg { .. }) {
            return Err(Error::NeedHvp);
        }
        if self.chart.is_some() {
            return Err(Error::FallibleUnsupported { what: "a chart" });
        }
        if self.constraints.is_some() {
            return Err(Error::FallibleUnsupported {
                what: "distance constraints",
            });
        }
        let saved_steps = self.steps;
        let saved_gradient_calls = self.gradient_calls;
        let saved_hvp_calls = self.hvp_calls;
        let saved_last_pos = self.last_pos.clone();
        let saved_last_value = self.last_value;
        let saved_last_grad = self.last_grad.clone();
        let saved_bound_low = self.bound_low.clone();
        let saved_bound_high = self.bound_high.clone();
        let saved_radius = match &self.inner {
            Inner::NewtonCg { radius } | Inner::NewtonPcg { radius } => Some(*radius),
            _ => None,
        };
        let saved_x = x.clone();
        let oracle = HostOracle(obj);
        let result = (|| {
            self.retract_level_set_start(&oracle, x)?;
            let mut report = self.step_newton_cg(&oracle, x, &HostAction(obj))?;
            self.finish_level_set(&oracle, x, &mut report)?;
            Ok(report)
        })();
        match result {
            Err(err @ Error::Host(_)) => {
                self.steps = saved_steps;
                self.gradient_calls = saved_gradient_calls;
                self.hvp_calls = saved_hvp_calls;
                self.last_pos = saved_last_pos;
                self.last_value = saved_last_value;
                self.last_grad = saved_last_grad;
                self.bound_low = saved_bound_low;
                self.bound_high = saved_bound_high;
                if let Some(radius) = saved_radius {
                    match &mut self.inner {
                        Inner::NewtonCg { radius: r } | Inner::NewtonPcg { radius: r } => {
                            *r = radius;
                        }
                        _ => {}
                    }
                }
                *x = saved_x;
                Err(err)
            }
            other => other,
        }
    }

    fn step_newton_cg<O, A>(&mut self, obj: &O, x: &mut Array1<f64>, action: &A) -> Result<Report>
    where
        O: StepOracle + ?Sized,
        A: StepAction,
    {
        if x.len() != self.dim || self.dim != obj.dim() {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim,
            });
        }
        self.check_manifold(x.len())?;
        let (radius, precondition) = match &self.inner {
            Inner::NewtonCg { radius } => (*radius, false),
            Inner::NewtonPcg { radius } => (*radius, true),
            _ => return Err(Error::NeedHvp),
        };
        let bounds = obj.bounds();
        let cached = self.same_last_x(x);
        let pos = if cached {
            x.clone()
        } else {
            bounds.clip(x.view())
        };
        let (value, raw_grad, opening_gradients) = if cached {
            (self.last_value, self.last_grad.clone(), 0)
        } else {
            let (value, raw_grad) = obj.step_value_and_gradient(pos.view())?;
            (value, raw_grad, 1)
        };
        let project_rigid = self.removal_mode() != RigidMode::Off;
        let mut ws = if project_rigid {
            self.rigid_ws.clone()
        } else {
            RigidWorkspace::new()
        };
        // CG sees the horizontal gradient. The forward difference
        // needs the raw gradient, which is what `remember` stores on
        // this path.
        let grad = self.horizontal_grad_into(&mut ws, &pos, &raw_grad);
        let gnorm = self.stationarity_norm_in(&pos, &grad, Some(&bounds.low), Some(&bounds.high));
        if !gnorm.is_finite() {
            self.note_bounds(bounds);
            if !cached {
                *x = pos;
                self.gradient_calls += opening_gradients;
            }
            if project_rigid {
                self.rigid_ws = ws;
            }
            return Err(Error::TrustCollapsed { steps: self.steps });
        }
        if gnorm < self.control.gtol {
            self.note_bounds(bounds);
            if !cached {
                *x = pos;
                self.gradient_calls += opening_gradients;
            }
            if project_rigid {
                self.rigid_ws = ws;
            }
            self.remember(x, value, &raw_grad);
            return Ok(self.finish(
                value,
                x,
                &grad,
                gnorm,
                self.control.classify(gnorm, self.free_len(bounds)),
            ));
        }
        // Same ceiling as dogleg: Euclidean `maxmove` when one is set,
        // otherwise eight times the current radius, and never below it.
        let rmax = self
            .control
            .maxmove_cap()
            .unwrap_or(radius * 8.0)
            .max(radius);
        let taken = newton_cg_step(
            obj,
            &pos,
            value,
            &grad,
            &raw_grad,
            &NewtonCgRequest {
                radius,
                rmax,
                atom_cap: self.atom_maxmove,
                euclid_cap: self.control.maxmove_cap(),
                project_rigid,
                precondition,
                seed: self.steps as u64,
                manifold: (!self.manifold.is_flat()).then(|| self.manifold.clone()),
            },
            action,
        )?;
        self.note_bounds(bounds);
        if !cached {
            *x = pos;
        }
        if project_rigid {
            self.rigid_ws = ws;
        }
        self.gradient_calls += opening_gradients + taken.trial_gradients;
        self.hvp_calls += taken.hvps;
        if let Inner::NewtonCg { radius } | Inner::NewtonPcg { radius } = &mut self.inner {
            *radius = taken.radius;
        }
        if taken.collapsed {
            // The free-function loop reports the attempt and keeps the
            // previous point when the radius hits the floor.
            self.remember(x, value, &raw_grad);
            self.steps += 1;
            return Err(Error::TrustCollapsed { steps: self.steps });
        }
        let (raw_out, report_grad) = if taken.moved {
            *x = taken.pos;
            let horizontal = self.horizontal_grad(x, &taken.grad);
            (taken.grad, horizontal)
        } else {
            (raw_grad, grad)
        };
        self.remember(x, taken.value, &raw_out);
        self.steps += 1;
        let gnorm = self.stationarity_norm(x, &report_grad);
        Ok(self.finish(
            taken.value,
            x,
            &report_grad,
            gnorm,
            self.control.classify(gnorm, self.free_len(bounds)),
        ))
    }

    /// `P^{-1} force` when an exponential preconditioner is installed.
    /// Otherwise `h0 * force`, the scalar geometry seed.
    fn geometry_seed_force(&self, force: &Array1<f64>, h0: f64) -> Array1<f64> {
        if let Inner::Lbfgs(solver) = &self.inner
            && let Some(pre) = solver.exp_apply(force.view())
        {
            return pre;
        }
        h0 * force
    }

    /// `scale * P^{-1} force` when an exponential preconditioner is installed.
    /// Otherwise `scale * force`.
    fn geometry_scaled_force(&self, force: &Array1<f64>, scale: f64) -> Array1<f64> {
        if let Inner::Lbfgs(solver) = &self.inner
            && let Some(pre) = solver.exp_apply(force.view())
        {
            return scale * &pre;
        }
        scale * force
    }

    /// Adaptive Burdakov trust radius for session L-BFGS, starting at
    /// `radius`. A non-positive or non-finite `radius` turns it off,
    /// the default.
    ///
    /// The radius follows Algorithm 1 of Burdakov, Gong, Zikrin, and
    /// Yuan, Math. Program. Comput. 9, 101-134 (2017),
    /// <https://doi.org/10.1007/s12532-016-0109-7>, with their Section 9
    /// constants. The step is a ray truncation along the two-loop
    /// direction, the third L-BFGS variant in their comparison. It does
    /// not solve their trust-region subproblem.
    /// [`Self::set_lbfgs_trust`] is a different opt-in: it scales a long
    /// capped step and scores the Newton model. The two can both be
    /// stored. When this radius is on, the Euclidean L-BFGS step uses
    /// this ray and leaves the capped step and that Newton-model radius
    /// alone.
    ///
    /// With [`Self::set_project_rigid`] the rigid modes leave the
    /// direction first. The ray stops at the Euclidean radius, then at
    /// the per-atom or Euclidean cap. `rho` is the actual over the
    /// predicted energy drop. `rho = 1` when the energy changes by at
    /// most `1e-11 |f|`, so a rise that small is accepted. `rho < 0`
    /// refuses the step: `x` and the pairs stay. `rho < 0.25` sets the
    /// radius to `min(r/4, |s|/2)`. `rho >= 0.75` with `|s| >= 0.8 r`
    /// doubles it. A trial moved by the box or by the rigid projection
    /// is predicted from `-g^T s`. Whenever the radius falls below
    /// `1e-15`, the step keeps the previous point and returns
    /// [`Error::TrustCollapsed`]. When the slope `g^T d` is still NaN
    /// or infinite after the pairs are dropped, the step returns
    /// [`Error::Oracle`]. When it is finite and not negative, the step
    /// returns [`Error::NoDescent`]. Neither moves.
    ///
    /// On Euclidean L-BFGS without HiGHS this replaces [`Self::set_accept`]
    /// and the geometry step of [`Self::set_atom_maxmove`]. Other methods
    /// and manifolds, and the HiGHS step, ignore the radius. `forget` and
    /// `rebase` restore `radius`.
    pub fn set_lbfgs_trust_radius(&mut self, radius: f64) {
        let on = radius > 0.0 && radius.is_finite();
        self.lbfgs_burdakov_init = if on { radius } else { 0.0 };
        self.lbfgs_burdakov = self.lbfgs_burdakov_init;
    }

    /// Current Burdakov radius, or `None` when it is off. A session
    /// whose method, manifold or HiGHS step ignores the radius still
    /// reports the stored value.
    pub fn lbfgs_trust_radius(&self) -> Option<f64> {
        (self.lbfgs_burdakov_init > 0.0).then_some(self.lbfgs_burdakov)
    }

    fn use_lbfgs_burdakov(&self) -> bool {
        // NaN is not `<= 0`, so a non-finite stored radius stays on.
        let on = matches!(
            self.lbfgs_burdakov_init.partial_cmp(&0.0),
            Some(std::cmp::Ordering::Greater) | None
        ) && matches!(self.inner, Inner::Lbfgs(_))
            && matches!(self.manifold, ManifoldKind::Euclidean);
        #[cfg(feature = "highs")]
        let on = on && !self.highs;
        on
    }

    /// One step under [`Self::set_lbfgs_trust_radius`]. One oracle call.
    fn lbfgs_burdakov_step<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        value: f64,
        grad: Array1<f64>,
    ) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let rigid = self.project_rigid;
        let Inner::Lbfgs(solver) = &mut self.inner else {
            unreachable!("use_lbfgs_burdakov requires L-BFGS")
        };
        let mut dir = solver.direction(grad.view());
        let mut projected = rigid && strip_rigid(&mut dir, x);
        let mut gd = grad.dot(&dir);
        if !gd.is_finite() || gd >= 0.0 {
            solver.forget();
            dir = solver.direction(grad.view());
            projected = rigid && strip_rigid(&mut dir, x);
            gd = grad.dot(&dir);
        }
        if !gd.is_finite() {
            return Err(Error::Oracle {
                what: "non-finite search direction",
            });
        }
        if gd >= 0.0 {
            return Err(Error::NoDescent { steps: self.steps });
        }
        let mut tau = (self.lbfgs_burdakov / l2(&dir)).min(1.0);
        if let Some(cap) = self.atom_maxmove {
            tau = tau.min(cap / max_atom_displacement(&dir));
        } else if let Some(cap) = self.control.maxmove_cap() {
            tau = tau.min(cap / l2(&dir));
        }
        let raw = &*x + &(tau * &dir);
        let trial = obj.bounds().clip(raw.view());
        let s = &trial - &*x;
        // q(tau d) = tau g^T d + tau^2/2 d^T B d, and d^T B d = -g^T d
        // only for d = -H g.
        let pred = if trial == raw && !projected {
            -gd * (tau - 0.5 * tau * tau)
        } else {
            -grad.dot(&s)
        };
        let (tval, tgrad) = obj.value_and_gradient(trial.view());
        let ared = value - tval;
        let rho = if !tval.is_finite() || tgrad.iter().any(|g| !g.is_finite()) {
            f64::NEG_INFINITY
        } else if ared.abs() <= 1e-11 * value.abs() {
            1.0
        } else if pred > 0.0 {
            ared / pred
        } else {
            f64::NEG_INFINITY
        };
        self.lbfgs_burdakov = burdakov_radius(self.lbfgs_burdakov, rho, l2(&s));
        self.steps += 1;
        if self.lbfgs_burdakov < 1e-15 {
            self.remember(x, value, &grad);
            return Err(Error::TrustCollapsed { steps: self.steps });
        }
        if rho < 0.0 {
            self.remember(x, value, &grad);
            return Ok(self.report(value, x, &grad));
        }
        *x = trial;
        let new_grad = self.horizontal_grad(x, &tgrad);
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.push_pair(s, &new_grad - &grad, Some(l2(&new_grad)));
        }
        self.remember(x, tval, &new_grad);
        Ok(self.report(tval, x, &new_grad))
    }

    /// Default Euclidean L-BFGS: the eOn cap, no line search.
    /// [`Self::set_geometry_step`] keeps the geometry algorithm.
    fn use_eon_lbfgs(&self) -> bool {
        // Accept::Step is the one-oracle two-loop clip. The default
        // Accept::None session takes the eOn cap.
        if self.geometry_step == Some(true)
            || self.accept == Accept::Step
            || !matches!(self.lbfgs_step(), LbfgsStep::Capped { .. })
        {
            return false;
        }
        if !matches!(self.manifold, ManifoldKind::Euclidean) {
            return false;
        }
        if !matches!(self.inner, Inner::Lbfgs(_)) || self.constraints.is_some() {
            return false;
        }
        if self.use_lbfgs_burdakov() {
            return false;
        }
        #[cfg(feature = "highs")]
        if self.highs {
            return false;
        }
        true
    }

    fn shape_step(&mut self, x: &Array1<f64>, mut direction: Array1<f64>) -> Array1<f64> {
        if !self.project_rigid && matches!(self.manifold, ManifoldKind::Euclidean) {
            return direction;
        }
        if self.project_rigid {
            project_out_rot_trans(&mut direction, x.view());
        }
        self.project_vec(x, &direction)
    }

    /// Trial point of a capped displacement. `steepest` reports that the
    /// direction had no quasi-Newton length, so a non-finite answer is
    /// not retried.
    fn capped_trial<O>(
        &self,
        obj: &O,
        x: &Array1<f64>,
        step: &Array1<f64>,
        steepest: bool,
    ) -> Result<(Array1<f64>, f64, Array1<f64>)>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let motion = max_atom_displacement(step);
        if !motion.is_finite() || motion <= 0.0 {
            return Err(Error::LbfgsCap {
                what: "the capped step is zero",
            });
        }
        let trial = crate::step::clip_owned(obj.bounds(), x + step);
        let taken = max_atom_displacement(&(&trial - x));
        if !taken.is_finite() || taken <= 0.0 {
            return Err(Error::OffManifold {
                kind: self.manifold.as_str(),
            });
        }
        let (value, grad) = obj.value_and_gradient(trial.view());
        if !crate::step::finite(value, grad.view()) {
            return Err(if steepest {
                crate::step::non_finite()
            } else {
                Error::Oracle {
                    what: "non-finite quasi-Newton trial",
                }
            });
        }
        Ok((trial, value, grad))
    }

    /// Curvature probe of the first eOn step, clipped to the box.
    /// The quotient uses the displacement that remains inside the box.
    fn eon_fd_seed<O>(&self, obj: &O, x: &Array1<f64>, g: &Array1<f64>) -> crate::lbfgs::EonSeed
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        use crate::lbfgs::{EON_FINITE_DIFFERENCE, EON_INVERSE_CURVATURE, EonSeed};
        let force = g.mapv(|v| -v);
        let norm = l2(&force);
        if !(norm.is_finite() && norm > 0.0) {
            return EonSeed::Scale(EON_INVERSE_CURVATURE);
        }
        let u = &force / norm;
        let raw = x + &(EON_FINITE_DIFFERENCE * &u);
        let probe = crate::step::clip_owned(obj.bounds(), raw);
        let along = (&probe - x).dot(&u);
        if !(along.is_finite() && along > 0.0) {
            return EonSeed::Scale(EON_INVERSE_CURVATURE);
        }
        let (value, grad) = obj.value_and_gradient(probe.view());
        if !crate::step::finite(value, grad.view()) {
            return EonSeed::Steepest;
        }
        let curvature = (&grad - g).dot(&u) / along;
        if curvature.is_finite() && curvature > 0.0 {
            EonSeed::Scale(1.0 / curvature)
        } else {
            EonSeed::Steepest
        }
    }

    /// One eOn L-BFGS step. A caller's move since the last step is a
    /// curvature pair. A non-finite quasi-Newton trial is retried once
    /// along `H0` times the force. A second non-finite trial restores the
    /// pairs, the caller's included, and returns [`Error::Oracle`].
    fn eon_lbfgs_step<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        grad: Array1<f64>,
    ) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        use crate::lbfgs::{EON_INVERSE_CURVATURE, EonSeed};
        if self.lindh_spec.is_some() {
            let xs: Vec<f64> = x.iter().copied().collect();
            let gs: Vec<f64> = grad.iter().copied().collect();
            self.ensure_lindh(&xs, &gs, |p| obj.grad(ArrayView1::from(p)));
        }
        let Some(cap) = self.lbfgs_step().capped_limit() else {
            return Err(Error::LbfgsCap {
                what: "the cap is not positive",
            });
        };
        let mut txn = match &mut self.inner {
            Inner::Lbfgs(solver) => solver.begin_txn(),
            _ => {
                return Err(Error::LbfgsCap {
                    what: "requires an L-BFGS session",
                });
            }
        };
        if !self.same_last_x(x)
            && let Some(previous) = self.last_pos.as_ref().filter(|p| p.len() == x.len())
        {
            let prev = previous.as_slice().unwrap_or(&[]);
            let xs = x.as_slice().unwrap_or(&[]);
            let gold = self.last_grad.as_slice().unwrap_or(&[]);
            let gnew = grad.as_slice().unwrap_or(&[]);
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.store_pos_diff(prev, xs, gold, gnew, &mut txn);
            }
        }
        let wants_fd = match &self.inner {
            Inner::Lbfgs(solver) => solver.eon_wants_fd(),
            _ => false,
        };
        let seed = if wants_fd {
            self.eon_fd_seed(obj, x, &grad)
        } else {
            match &self.inner {
                Inner::Lbfgs(solver) => solver.eon_seed(),
                _ => EonSeed::Scale(EON_INVERSE_CURVATURE),
            }
        };
        let h0 = match seed {
            EonSeed::Scale(h) => h,
            EonSeed::Steepest => EON_INVERSE_CURVATURE,
        };
        let proposed = match self.eon_propose(obj, x, &grad, cap, seed, &mut txn) {
            Ok(point) => point,
            Err(CappedMiss::Retry) => {
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.forget_txn(&mut txn);
                }
                let force = grad.mapv(|v| -v);
                let step = self.shape_step(x, cap_displacement_atom(&(h0 * &force), cap));
                let step = match self.tangent_displacement(&grad, step) {
                    Ok(Some(step)) => step,
                    Ok(None) => {
                        if let Inner::Lbfgs(solver) = &mut self.inner {
                            solver.rollback_txn(txn);
                        }
                        return self.level_set_stay(obj, x);
                    }
                    Err(err) => {
                        if let Inner::Lbfgs(solver) = &mut self.inner {
                            solver.rollback_txn(txn);
                        }
                        return Err(err);
                    }
                };
                match self.capped_trial(obj, x, &step, true) {
                    Ok(point) => point,
                    Err(err) => {
                        if let Inner::Lbfgs(solver) = &mut self.inner {
                            solver.rollback_txn(txn);
                        }
                        return Err(err);
                    }
                }
            }
            Err(CappedMiss::LevelStay) => {
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.rollback_txn(txn);
                }
                return self.level_set_stay(obj, x);
            }
            Err(CappedMiss::Stop(err)) => {
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.rollback_txn(txn);
                }
                return Err(err);
            }
        };
        let (trial, mut tval, mut tgrad) = proposed;
        let old = x.clone();
        let gold = grad;
        *x = trial;
        if self.level_set.is_some()
            && let Err(err) = self.retract_level_set_here(obj, x, &mut tval, &mut tgrad)
        {
            *x = old;
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.rollback_txn(txn);
            }
            return Err(err);
        }
        let grad = self.horizontal_grad(x, &tgrad);
        let stayed = self.level_set.is_some()
            && x.iter()
                .zip(old.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if stayed {
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.rollback_txn(txn);
            }
            self.remember(x, tval, &grad);
            self.steps += 1;
            return Ok(self.report(tval, x, &grad));
        }
        let y_prev = gold.as_slice().unwrap_or(&[]);
        let y_new = grad.as_slice().unwrap_or(&[]);
        let prev = old.as_slice().unwrap_or(&[]);
        let xs = x.as_slice().unwrap_or(&[]);
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.store_pos_diff(prev, xs, y_prev, y_new, &mut txn);
            solver.commit_txn(txn);
        }
        self.remember(x, tval, &grad);
        self.steps += 1;
        Ok(self.report(tval, x, &grad))
    }

    /// Euclidean eOn proposal and its trial.
    fn eon_propose<O>(
        &mut self,
        obj: &O,
        x: &Array1<f64>,
        grad: &Array1<f64>,
        cap: f64,
        seed: crate::lbfgs::EonSeed,
        txn: &mut crate::lbfgs::Txn,
    ) -> std::result::Result<(Array1<f64>, f64, Array1<f64>), CappedMiss>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let take = match &mut self.inner {
            Inner::Lbfgs(solver) => solver.eon_step_from_seed(grad.view(), cap, seed, txn),
            _ => None,
        };
        let Some(take) = take else {
            return Err(CappedMiss::Stop(Error::LbfgsCap {
                what: "the capped step is zero",
            }));
        };
        let step = self.shape_step(x, take.step);
        let step = match self.tangent_displacement(grad, step) {
            Ok(Some(step)) => step,
            Ok(None) => return Err(CappedMiss::LevelStay),
            Err(err) => return Err(CappedMiss::Stop(err)),
        };
        let result = self.capped_trial(obj, x, &step, take.fallback);
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.recycle(step);
        }
        match result {
            Ok(point) => Ok(point),
            Err(Error::Oracle {
                what: "non-finite quasi-Newton trial",
            }) => Err(CappedMiss::Retry),
            Err(err) => Err(CappedMiss::Stop(err)),
        }
    }

    /// The geometry step is the explicit flag, or a Wolfe session that
    /// has a per-atom cap. The default eOn cap is not this path.
    fn use_geometry_lbfgs(&self) -> bool {
        // An explicit per-atom cap on a Wolfe session is the geometry
        // step unless the caller turned it off. Accept::Step keeps the
        // one-oracle two-loop. The implicit default cap does not select
        // this path.
        let wolfe_cap = matches!(self.lbfgs_step(), LbfgsStep::Wolfe)
            && self.geometry_step.is_none()
            && self.atom_maxmove_set
            && self.atom_maxmove.is_some()
            && self.accept != Accept::Step;
        if self.geometry_step != Some(true) && !wolfe_cap {
            return false;
        }
        if self.atom_maxmove.is_none() || !matches!(self.manifold, ManifoldKind::Euclidean) {
            return false;
        }
        if !matches!(self.inner, Inner::Lbfgs(_)) {
            return false;
        }
        #[cfg(feature = "highs")]
        if self.highs {
            return false;
        }
        true
    }

    fn apply_geometry_curvature(
        &mut self,
        sy: f64,
        yy: f64,
        h0: &mut f64,
        force: &Array1<f64>,
        cap: f64,
        forced: &mut Option<Array1<f64>>,
    ) {
        let curvature = if sy.is_finite() && yy.is_finite() && sy != 0.0 {
            yy / sy
        } else {
            -1.0
        };
        if curvature < 0.0 {
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.forget();
            }
            *forced = Some(cap_displacement_atom(
                &self.geometry_scaled_force(force, 1000.0),
                cap,
            ));
        } else if sy > 0.0 && yy > 0.0 {
            *h0 = sy / yy;
        }
    }

    /// Capped Cartesian L-BFGS step: severely clipped starts estimate inverse
    /// curvature by finite difference; ordinary starts use `0.1 I`. Later
    /// iterations use `sy/yy`. The finite-difference probe is clipped to the
    /// objective bounds, and the quotient uses the displacement that remains
    /// inside the box. Curvature pairs are reset when curvature is negative,
    /// the uncapped step exceeds the per-atom cap, or the step opposes the
    /// force. Trials use monotone acceptance after a stiffness probe and a
    /// five-value nonmonotone window otherwise, with at most ten halvings
    /// before a short steepest step.
    /// An installed exponential preconditioner replaces that scalar seed:
    /// the two-loop applies `P^{-1}`, and a reset that would have stepped
    /// along `h0` times the force steps along `P^{-1}` times the force.
    /// Those steepest fallbacks use `P^{-1}` as well. A failed solve falls
    /// back to the scalar seed. The cap still limits the displacement.
    fn geometry_lbfgs_step<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        value: f64,
        mut grad: Array1<f64>,
    ) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let cap = self.atom_maxmove.unwrap_or(0.2);
        let nominal_motion = max_atom_displacement(&(self.istep * grad.mapv(|g| -g)));
        let stiff = *self
            .geometry_lbfgs_stiff
            .get_or_insert(nominal_motion >= 64.0 * cap);
        let mut h0 = if stiff { 0.01 } else { 0.1 };
        let mut forced: Option<Array1<f64>> = None;
        let force = grad.mapv(|g| -g);
        let cached = self.same_last_x(x);
        if !cached {
            if let Some(previous) = self.last_pos.clone() {
                let s = &*x - &previous;
                let y = &grad - &self.last_grad;
                let sy = s.dot(&y);
                let yy = y.dot(&y);
                self.apply_geometry_curvature(sy, yy, &mut h0, &force, cap, &mut forced);
                if forced.is_none()
                    && let Inner::Lbfgs(solver) = &mut self.inner
                {
                    solver.push_geometry(s, y);
                }
            }
        } else {
            let (have, sy, yy) = match &self.inner {
                Inner::Lbfgs(solver) => (solver.have_curvature, solver.last_sy, solver.last_yy),
                _ => (false, 0.0, 0.0),
            };
            if have {
                self.apply_geometry_curvature(sy, yy, &mut h0, &force, cap, &mut forced);
            }
        }
        if stiff && forced.is_none() && self.steps == 0 {
            let fnorm = l2(&force);
            if fnorm > 0.0 {
                let unit = &force / fnorm;
                // 0.01 is the eOn probe length. Clip before the evaluation:
                // the raw point can lie outside the objective box.
                let raw = &*x + &(0.01 * &unit);
                let probe = obj.bounds().clip(raw.view());
                let along = (&probe - &*x).dot(&unit);
                let curvature = if along > 0.0 && along.is_finite() {
                    let (_efd, gfd) = obj.value_and_gradient(probe.view());
                    let delta = &gfd - &grad;
                    delta.dot(&unit) / along
                } else {
                    -1.0
                };
                if curvature > 0.0 && curvature.is_finite() {
                    h0 = 1.0 / curvature;
                } else if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.forget();
                    forced = Some(cap_displacement_atom(
                        &self.geometry_scaled_force(&force, 1000.0),
                        cap,
                    ));
                }
            }
        }
        let direction = if let Some(dir) = forced {
            dir
        } else {
            let mut direction = match &self.inner {
                Inner::Lbfgs(solver) => solver.direction_h0(grad.view(), h0),
                _ => h0 * &force,
            };
            direction = self.project_vec(x, &direction);
            let motion = max_atom_displacement(&direction);
            if motion >= cap {
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.forget();
                }
                cap_displacement_atom(&self.geometry_seed_force(&force, h0), cap)
            } else {
                let dn = l2(&direction);
                let fnorm = l2(&force);
                if dn > 0.0 && fnorm > 0.0 {
                    let cosine = (direction.dot(&force) / (dn * fnorm)).clamp(-1.0, 1.0);
                    if cosine < 0.0 {
                        if let Inner::Lbfgs(solver) = &mut self.inner {
                            solver.forget();
                        }
                        direction =
                            cap_displacement_atom(&self.geometry_seed_force(&force, h0), cap);
                    }
                }
                cap_displacement_atom(&direction, cap)
            }
        };
        let reference = if stiff {
            value
        } else {
            self.e_hist
                .iter()
                .copied()
                .fold(value, |largest, candidate| largest.max(candidate))
        };
        let mut alpha = 1.0;
        let mut accepted = false;
        let mut trial = x.clone();
        let mut tval = value;
        let mut tgrad = grad.clone();
        for _ in 0..10 {
            trial = obj.bounds().clip((&*x + alpha * &direction).view());
            let ev = obj.value_and_gradient(trial.view());
            tval = ev.0;
            tgrad = ev.1;
            if tval.is_finite() && tval - reference <= 1.0e-8 {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.forget();
            }
            let sd = cap_displacement_atom(&self.geometry_scaled_force(&force, 0.1), cap);
            trial = obj.bounds().clip((&*x + &sd).view());
            let ev = obj.value_and_gradient(trial.view());
            tval = ev.0;
            tgrad = ev.1;
        }
        if !stiff {
            self.e_hist.push_back(tval);
            if self.e_hist.len() > 5 {
                self.e_hist.pop_front();
            }
        }
        let old = x.clone();
        let gold = grad.clone();
        *x = trial;
        grad = self.horizontal_grad(x, &tgrad);
        let s = &*x - &old;
        let y = &grad - &gold;
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.note_curvature(&s, &y);
            if solver.have_curvature {
                let sy = solver.last_sy;
                let yy = solver.last_yy;
                let curvature = if sy != 0.0 { yy / sy } else { -1.0 };
                if curvature >= 0.0 {
                    solver.push_geometry(s, y);
                }
            }
        }
        self.remember(x, tval, &grad);
        self.steps += 1;
        let gnorm = self.stationarity_norm(x, &grad);
        Ok(self.finish(
            tval,
            x,
            &grad,
            gnorm,
            self.control.classify(gnorm, self.free_len(obj.bounds())),
        ))
    }

    fn saddle_escape_enabled(&self) -> bool {
        #[cfg(feature = "highs")]
        if self.highs {
            return false;
        }
        matches!(self.inner, Inner::Newton { .. } | Inner::Dogleg { .. })
    }

    /// Lowest eigenpair of `hess` when it lies below the stationarity tolerance.
    fn unstable_mode(&self, hess: &ndarray::Array2<f64>) -> Option<Array1<f64>> {
        let n = hess.nrows();
        if n == 0 || hess.ncols() != n || hess.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut fro2 = 0.0;
        for v in hess.iter() {
            fro2 += v * v;
        }
        let fro = fro2.sqrt();
        let tol = self.control.gtol.abs().max(1.0e-6).max(1.0e-8 * fro);
        let (value, vector) = if n <= 160 {
            let (evals, evecs) = sym_eig_jacobi(hess.clone());
            let mut k = 0usize;
            let mut best = f64::INFINITY;
            for (i, lam) in evals.iter().enumerate() {
                if *lam < best {
                    best = *lam;
                    k = i;
                }
            }
            if !best.is_finite() {
                return None;
            }
            (best, evecs.column(k).to_owned())
        } else {
            struct Mat<'a>(&'a ndarray::Array2<f64>);
            impl ApplyHessian for Mat<'_> {
                fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
                    self.0.dot(&v)
                }
            }
            let mut seed = Array1::zeros(n);
            for i in 0..n {
                seed[i] = ((i % 11) as f64) - 5.0;
            }
            // `lowest_mode` returns a result. A failed Lanczos run is not a
            // direction this kick can take.
            let Ok(found) = crate::lowest_mode(
                &Mat(hess),
                seed.view(),
                seed.view(),
                &crate::EigenParams {
                    kind: crate::EigensolverKind::Lanczos,
                    krylov: 48.min(n),
                    ..crate::EigenParams::default()
                },
            ) else {
                return None;
            };
            (found.value, found.vector)
        };
        if value < -tol { Some(vector) } else { None }
    }

    /// Move off a converged saddle along the lowest Hessian mode, scaled to
    /// the per-atom cap. Analytic Hessian only; at most four kicks per solve.
    fn leave_saddle<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        value: &mut f64,
        grad: &mut Array1<f64>,
    ) where
        O: HessianObjective + ?Sized,
    {
        loop {
            let gnorm = self.stationarity_norm(x, grad);
            if gnorm > self.control.gtol || self.saddle_kicks >= 4 {
                return;
            }
            let hess = obj.hessian(x.view());
            let Some(mode) = self.unstable_mode(&hess) else {
                return;
            };
            let cap = self
                .atom_maxmove
                .unwrap_or_else(|| self.control.istep.max(1.0e-8));
            let step = scale_displacement_to_atom_cap(&mode, cap);
            let trial = obj.bounds().clip((&*x + &step).view());
            let (v, g) = obj.value_and_gradient(trial.view());
            if !v.is_finite() || g.iter().any(|c| !c.is_finite()) {
                return;
            }
            *x = trial;
            *value = v;
            *grad = self.horizontal_grad(x, &g);
            self.saddle_kicks = self.saddle_kicks.saturating_add(1);
            if let Inner::Dogleg { radius } = &mut self.inner {
                *radius = self.control.istep.max(1.0e-8);
            }
            self.remember(x, *value, grad);
        }
    }

    fn report_or_escape<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        mut value: f64,
        mut grad: Array1<f64>,
    ) -> Result<Report>
    where
        O: HessianObjective + ?Sized,
    {
        if self.saddle_escape_enabled() && self.stationarity_norm(x, &grad) <= self.control.gtol {
            self.leave_saddle(obj, x, &mut value, &mut grad);
        }
        let gnorm = self.stationarity_norm(x, &grad);
        Ok(self.finish(
            value,
            x,
            &grad,
            gnorm,
            self.control.classify(gnorm, self.free_len(obj.bounds())),
        ))
    }

    fn step_hess_inner<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: HessianObjective + ?Sized,
    {
        if x.len() != self.dim || self.dim != Objective::dim(obj) {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim,
            });
        }
        self.check_manifold(x.len())?;
        self.note_bounds(obj.bounds());
        let newton_kind = match &self.inner {
            Inner::Newton { kind } => Some(*kind),
            Inner::Lbfgs(_) => match self.qn_step {
                QnStep::Newton => Some(NewtonKind::Shifted),
                QnStep::Rfo => Some(NewtonKind::Rfo),
                QnStep::TwoLoop => None,
            },
            Inner::Dogleg { .. } => None,
            // A missed early return must not fall through to `obj.hessian`.
            Inner::NewtonCg { .. } | Inner::NewtonPcg { .. } => {
                return self.step_newton_cg(obj, x, &FiniteDifference);
            }
            _ => return self.step_first_order(obj, x),
        };
        let cached = self.same_last_x(x);
        if !cached {
            *x = obj.bounds().clip(x.view());
        }
        let (mut value, mut grad) = if cached {
            (self.last_value, self.last_grad.clone())
        } else {
            obj.value_and_gradient(x.view())
        };
        grad = self.horizontal_grad(x, &grad);
        let gnorm = self.stationarity_norm(x, &grad);
        if self
            .control
            .halt(gnorm, self.free_len(obj.bounds()))
            .is_some()
        {
            return self.report_or_escape(obj, x, value, grad);
        }
        let hess = obj.hessian(x.view());
        if matches!(self.inner, Inner::Dogleg { .. }) {
            return self.step_dogleg(obj, x, value, grad, &hess);
        }
        #[cfg(feature = "highs")]
        if self.highs {
            let center: Option<(usize, usize)> = None;
            let feasible = if self.has_coordinate_box()
                || self.highs_trust.is_some()
                || !self.equalities.is_empty()
            {
                let mut model_hess = hess.clone();
                if newton_kind == Some(NewtonKind::Rfo) {
                    let (_, shift) = crate::sella_step::rfo_step_and_shift(&hess, &grad, 0, 1.0);
                    for k in 0..self.dim {
                        model_hess[(k, k)] -= shift;
                    }
                }
                Ok(crate::lbfgs_qp::highs_feasible_step_boxed(
                    None,
                    Some(&model_hess),
                    &grad,
                    self.atom_maxmove,
                    self.highs_trust.or(self.control.maxmove_cap()),
                    center,
                    self.has_coordinate_box().then(|| (x.view(), obj.bounds())),
                    &self.equalities,
                    &self.highs_options,
                )?)
            } else {
                crate::lbfgs_qp::highs_feasible_step_with_options(
                    None,
                    Some(&hess),
                    &grad,
                    self.atom_maxmove,
                    self.control.maxmove_cap(),
                    center,
                    &self.highs_options,
                )
            };
            if let Ok(dir) = feasible {
                let old = x.clone();
                let gold = grad.clone();
                let allow_gradient_fallback = !self.has_coordinate_box()
                    && self.highs_trust.is_none()
                    && self.equalities.is_empty();
                let (npos, nval, ngrad, moved) = crate::accept::accept_step_with_fallback(
                    obj,
                    x,
                    value,
                    &gold,
                    &dir,
                    &self.control,
                    self.accept,
                    &mut self.e_hist,
                    None,
                    &self.manifold,
                    allow_gradient_fallback,
                );
                if moved {
                    *x = npos;
                    value = nval;
                    grad = ngrad;
                }
                grad = self.horizontal_grad(x, &grad);
                self.remember(x, value, &grad);
                let pair = if x.iter().zip(old.iter()).any(|(a, b)| a != b) {
                    let (s, y) = self.lbfgs_sy(&old, x, &gold, &grad);
                    Some((s, y, l2(&grad)))
                } else {
                    None
                };
                if let (Inner::Lbfgs(solver), Some((s, y, gn))) = (&mut self.inner, pair) {
                    solver.push_pair(s, y, Some(gn));
                }
                self.steps += 1;
                return self.report_or_escape(obj, x, value, grad);
            }
        }
        let mut dir = if let Some(kind) = newton_kind {
            match kind {
                NewtonKind::Shifted => shifted_newton(&hess, &grad),
                NewtonKind::Rfo => rfo_direction(&hess, &grad),
            }
        } else if let Inner::Lbfgs(solver) = &self.inner {
            solver.direction_with_precon(grad.view(), Some(&hess))
        } else {
            return self.step_first_order(obj, x);
        };
        dir = self.project_vec(x, &dir);
        let old = x.clone();
        let gold = grad.clone();
        let (npos, nval, ngrad, moved) = accept_step(
            obj,
            x,
            value,
            &gold,
            &dir,
            Acceptance {
                control: &self.control,
                accept: self.accept,
                e_hist: &mut self.e_hist,
                atom_maxmove: self.atom_maxmove,
                manifold: &self.manifold,
            },
        );
        if moved {
            *x = npos;
            value = nval;
            grad = ngrad;
        }
        grad = self.horizontal_grad(x, &grad);
        self.remember(x, value, &grad);
        let pair = if x.iter().zip(old.iter()).any(|(a, b)| a != b) {
            let (s, y) = self.lbfgs_sy(&old, x, &gold, &grad);
            Some((s, y, l2(&grad)))
        } else {
            None
        };
        if let (Inner::Lbfgs(solver), Some((s, y, gn))) = (&mut self.inner, pair) {
            solver.push_pair(s, y, Some(gn));
        }
        self.steps += 1;
        self.report_or_escape(obj, x, value, grad)
    }

    fn step_dogleg<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        value: f64,
        grad: Array1<f64>,
        hess: &ndarray::Array2<f64>,
    ) -> Result<Report>
    where
        O: HessianObjective + ?Sized,
    {
        let radius = match &self.inner {
            Inner::Dogleg { radius } => *radius,
            _ => self.control.istep.max(1e-8),
        };
        // Nocedal and Wright, algorithm 4.1: grow Δ only when the trust
        // step meets the boundary, and only up to a Euclidean Δ_max.
        // A per-atom cap is a different norm. Using it as Δ_max (the
        // 0.4.1 default of 0.2, and every tuned Morse row) lets Δ climb
        // from the initial radius to that cap and then stop, so a
        // 768-coordinate step stays inside a Euclidean ball of radius
        // 0.2. Feeding the clipped length to the boundary test does the
        // same once the cap binds: the trust step is on the sphere, but
        // the shorter trial looks interior, so Δ does not grow.
        // Δ_max is the Euclidean `maxmove` when one is set, otherwise
        // eight times the current radius. The cap still limits the trial.
        let rmax = self
            .control
            .maxmove_cap()
            .unwrap_or(radius * 8.0)
            .max(radius);
        // A curved kind steps in the tangent space. The dogleg point is
        // tangent, the trial is the retracted tangent step, and the ratio
        // compares the energy there with the tangent model.
        let curved = !self.manifold.is_flat();
        let mut dir = self.dogleg_point(x, hess, &grad, radius);
        self.kill_at(x, &mut dir);
        if curved {
            self.manifold.project_in_place(x, &mut dir);
        }
        let trust_norm = l2(&dir);
        let (trial, p) = if curved {
            let fitted = retract_within_cap(
                obj,
                x,
                &dir,
                self.atom_maxmove,
                self.control.maxmove_cap(),
                &self.manifold,
            );
            let Some(pair) = fitted else {
                // Refused before any oracle: a failed trust step.
                if let Inner::Dogleg { radius } = &mut self.inner {
                    *radius = update_radius(*radius, -1.0, trust_norm, rmax);
                }
                self.remember(x, value, &grad);
                self.steps += 1;
                return self.report_or_escape(obj, x, value, grad);
            };
            pair
        } else {
            let mut trial = &*x + &dir;
            apply_caps(x, &mut trial, self.atom_maxmove, self.control.maxmove_cap());
            trial = obj.bounds().clip(trial.view());
            let p = &trial - &*x;
            (trial, p)
        };
        let taken = l2(&p);
        let (ft, gt) = obj.value_and_gradient(trial.view());
        let pred = self.dogleg_pred(hess, &grad, &p);
        let rho = reduction_ratio(value - ft, pred);
        // No displacement cap: the boundary length is the step after the
        // box clip, as before. With a cap, the dogleg length is the one
        // algorithm 4.1 tests against Δ.
        let boundary_norm = if self.atom_maxmove.is_some() || self.control.maxmove_cap().is_some() {
            trust_norm
        } else {
            taken
        };
        if let Inner::Dogleg { radius } = &mut self.inner {
            *radius = update_radius(*radius, rho, boundary_norm, rmax);
        }
        if accept_ratio(rho) {
            *x = trial;
            self.remember(x, ft, &gt);
            self.steps += 1;
            self.report_or_escape(obj, x, ft, gt)
        } else {
            self.remember(x, value, &grad);
            self.steps += 1;
            self.report_or_escape(obj, x, value, grad)
        }
    }

    fn step_first_order<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if x.len() != self.dim || self.dim != Objective::dim(obj) {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim,
            });
        }
        self.check_manifold(x.len())?;
        self.note_bounds(obj.bounds());
        let cached = self.same_last_x(x);
        if !cached {
            if x.iter().any(|v| !v.is_finite()) {
                return Err(crate::step::non_finite());
            }
            *x = obj.bounds().clip(x.view());
        }

        if let Inner::Pso { .. } = &self.inner {
            return self.step_pso(obj, x);
        }

        let (mut value, mut grad) = if cached {
            (self.last_value, self.last_grad.clone())
        } else {
            obj.value_and_gradient(x.view())
        };
        grad = self.horizontal_grad(x, &grad);
        if !value.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            return Err(Error::Oracle {
                what: "non-finite value or gradient",
            });
        }
        let gnorm = self.stationarity_norm(x, &grad);
        let eon = self.use_eon_lbfgs();
        // A caller that moved the iterate between steps measured the
        // gradient at both ends, so the displacement is a secant of the
        // retained inverse Hessian. The eOn step records it in the
        // transaction a refused trial rolls back.
        if !cached
            && !eon
            && self.manifold.retract_is_translation()
            && let Some(previous) = self.last_pos.clone().filter(|p| p.len() == x.len())
        {
            let last_grad = self.last_grad.clone();
            let (s, y) = self.lbfgs_sy(&previous, x, &last_grad, &grad);
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.push_pair(s, y, Some(gnorm));
            }
        }
        if let Some(status) = self.control.halt(gnorm, self.free_len(obj.bounds())) {
            return Ok(self.finish(value, x, &grad, gnorm, status));
        }
        self.refresh_exp(obj, x, &grad)?;

        if self.use_lbfgs_burdakov() {
            return self.lbfgs_burdakov_step(obj, x, value, grad);
        }
        if eon {
            return self.eon_lbfgs_step(obj, x, grad);
        }
        if self.use_geometry_lbfgs() {
            return self.geometry_lbfgs_step(obj, x, value, grad);
        }
        if self.use_scaled_lbfgs() {
            return self.scaled_lbfgs_step(obj, x, value, grad);
        }

        if matches!(self.manifold, ManifoldKind::Sphere | ManifoldKind::Stiefel)
            && matches!(self.inner, Inner::Lbfgs(_))
            && matches!(self.accept, Accept::Energy | Accept::Nonmonotone)
        {
            if !cached && let Some(previous) = self.last_pos.clone() {
                let last_grad = self.last_grad.clone();
                let (s, y) = self.lbfgs_sy(&previous, x, &last_grad, &grad);
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.transport(|v| crate::manifold::Sphere.transport(&previous, x, v));
                    solver.push_pair(s, y, Some(l2(&grad)));
                }
            }
            if let Inner::Lbfgs(solver) = &mut self.inner {
                let (point, f, g) = crate::sphere_lbfgs::step(
                    obj,
                    x,
                    value,
                    &grad,
                    solver,
                    self.linesearch,
                    &self.control,
                    self.atom_maxmove,
                )?;
                *x = point;
                value = f;
                grad = g;
            }
            self.remember(x, value, &grad);
            self.steps += 1;
            return Ok(self.finish(
                value,
                x,
                &grad,
                self.stationarity_norm(x, &grad),
                self.control.classify(
                    self.stationarity_norm(x, &grad),
                    self.free_len(obj.bounds()),
                ),
            ));
        }

        let raw_gradient = match self.inner {
            Inner::Steepest
            | Inner::Bfgs { .. }
            | Inner::Sr1 { .. }
            | Inner::Sr2 { .. }
            | Inner::Bb { .. } => true,
            Inner::Nlcg { .. } => self.precon.is_none(),
            _ => false,
        };
        if self.unit_first_step
            && raw_gradient
            && self.accept != Accept::None
            && self.last_pos.is_none()
        {
            self.istep = self.control.istep.min(1.0 / l2(&grad));
        }

        let line_options = self.search_options();
        let start = x.clone();
        let gold = grad.clone();
        // Accept::Step: the two-loop direction goes through accept_step
        // like the BB arm and the Hessian path in step_hess, one oracle
        // call and the clipped step with no energy test. The oracle value
        // of a projected NEB force is not the potential of the gradient it
        // returns, so a decrease test there says nothing. Every other
        // Accept keeps the line-searched step_objective path.
        let guarded_direction = if self.accept == Accept::Step {
            match &mut self.inner {
                Inner::Lbfgs(solver)
                    if solver.distance_reset || solver.angle_reset || solver.curvature_reset =>
                {
                    Some(solver.guarded_direction(grad.view(), self.atom_maxmove))
                }
                _ => None,
            }
        } else {
            None
        };
        #[cfg(feature = "highs")]
        let constrained_direction =
            if self.highs && (self.highs_trust.is_some() || !self.equalities.is_empty()) {
                if let Inner::Lbfgs(solver) = &self.inner {
                    let direction = guarded_direction
                        .clone()
                        .unwrap_or_else(|| solver.direction(grad.view()));
                    let center: Option<(usize, usize)> = None;
                    Some(crate::lbfgs_qp::highs_feasible_step_boxed(
                        Some(&direction),
                        None,
                        &grad,
                        self.atom_maxmove,
                        self.highs_trust.or(self.control.maxmove_cap()),
                        center,
                        self.has_coordinate_box().then(|| (x.view(), obj.bounds())),
                        &self.equalities,
                        &self.highs_options,
                    )?)
                } else {
                    None
                }
            } else {
                None
            };
        #[cfg(not(feature = "highs"))]
        let constrained_direction: Option<Array1<f64>> = None;
        let lbfgs_direct = match (&self.inner, self.accept) {
            (Inner::Lbfgs(solver), Accept::Step) => {
                let dir = constrained_direction.clone().unwrap_or_else(|| {
                    guarded_direction
                        .clone()
                        .unwrap_or_else(|| solver.direction(grad.view()))
                });
                #[cfg(feature = "highs")]
                let dir = if let Some(bounds) = &solver.coordinate_box {
                    // The model QP already enforces its box and equalities.
                    if constrained_direction.is_some() {
                        dir
                    } else {
                        crate::box_objective::project_direction(bounds, x.view(), grad.view(), dir)
                    }
                } else {
                    dir
                };
                Some(dir)
            }
            _ => None,
        };
        // On a curved kind the line search would probe the chord, off the
        // set. Accept::None takes one retracted trial of the two-loop step.
        // So does a product with rigid factors: its horizontal gradient is
        // the mass-metric projection, whose slope along a step is not the
        // directional derivative the line search measures at its trials.
        let rigid_product =
            matches!(&self.manifold, ManifoldKind::Product(product) if product.has_rigid());
        let lbfgs_single = lbfgs_direct.is_none()
            && self.accept == Accept::None
            && (!self.manifold.is_flat() || rigid_product)
            && matches!(self.inner, Inner::Lbfgs(_));
        let lbfgs_line_searched =
            lbfgs_direct.is_none() && !lbfgs_single && matches!(self.inner, Inner::Lbfgs(_));
        let on_set = lbfgs_single;
        if lbfgs_single {
            let mut dir = match &self.inner {
                Inner::Lbfgs(solver) => solver.search_direction(x.view(), grad.view())?,
                _ => unreachable!(),
            };
            dir = self.project_level_set_vec(&grad, &dir)?;
            match single_trial(
                obj,
                x,
                &dir,
                &self.control,
                &mut self.e_hist,
                self.atom_maxmove,
                &self.manifold,
            ) {
                Trial::Accepted(npos, nval, ngrad) => {
                    *x = npos;
                    value = nval;
                    grad = ngrad;
                }
                Trial::OffManifold => {
                    return Err(Error::OffManifold {
                        kind: self.manifold.as_str(),
                    });
                }
                Trial::NonFinite => return Err(crate::step::non_finite()),
            }
        }
        if let Some(mut dir) = lbfgs_direct {
            dir = self.project_vec(x, &dir);
            dir = self.project_level_set_vec(&grad, &dir)?;
            let (npos, nval, ngrad, moved) = accept_step(
                obj,
                x,
                value,
                &gold,
                &dir,
                Acceptance {
                    control: &self.control,
                    accept: self.accept,
                    e_hist: &mut self.e_hist,
                    atom_maxmove: self.atom_maxmove,
                    manifold: &self.manifold,
                },
            );
            if moved {
                *x = npos;
                value = nval;
                grad = ngrad;
            }
        }
        #[cfg(feature = "highs")]
        let boxed = self.has_coordinate_box();
        let mut energy_stall = false;
        let mut sr_secant: Option<(Array1<f64>, Array1<f64>)> = None;
        let diis_alpha = self.istep;
        let diis_atom_cap = self.atom_maxmove;
        let diis_euclidean_cap = self.control.maxmove_cap();
        if matches!(&self.inner, Inner::Sr1 { .. } | Inner::Sr2 { .. }) {
            self.sr_trust_attempt(obj, x, &mut value, &mut grad, &start, &gold, &mut sr_secant);
        }
        // The rigid projector for the trial directions: the workspace is
        // taken out of the session for the step so that the closure and
        // the stepper borrow disjoint fields, and returns when the hold
        // drops.
        if !matches!(self.overlay, Overlay::None) {
            self.overlay_apply(
                obj,
                x,
                &mut value,
                &mut grad,
                &mut energy_stall,
                line_options,
                diis_alpha,
                diis_atom_cap,
                diis_euclidean_cap,
            )?;
        } else {
            let mut fire2_local = self.fire2.clone();
            let mut stalled = false;
            let rigid = RigidRemoval {
                mode: self.removal_mode(),
                masses: self.masses.as_ref().and_then(|m| m.as_slice()),
                frozen: self.frozen.as_deref(),
                weighted: self.chart_weighted(),
                force_metric: self.force_metric(),
                velocity_metric: self.velocity_metric(),
                pbc: self.pbc,
                broken_translation: self.broken_translation,
                broken_rotation: self.broken_rotation,
            };
            let level_set_on = self.level_set.is_some();
            let mut hold = WsHold::take(&mut self.rigid_ws);
            if rigid.mode != RigidMode::Off
                && let Some(xs) = x.as_slice()
            {
                hold.ws.set_anchor(xs);
            }
            match &mut self.inner {
                Inner::Lbfgs(_) if !lbfgs_line_searched => {}
                Inner::Lbfgs(solver) => {
                    let mut project = |d: &mut Array1<f64>| {
                        rigid.force(&mut hold.ws, d);
                    };
                    energy_stall = solver.step_objective_with_direction(
                        obj,
                        x,
                        &mut value,
                        &mut grad,
                        &mut self.istep,
                        self.linesearch,
                        &self.control,
                        self.atom_maxmove,
                        constrained_direction,
                        false,
                        line_options,
                        &mut project,
                    );
                }
                Inner::Steepest => {
                    let dir = grad.mapv(|g| -g);
                    let t = take_session_step(
                        obj,
                        x,
                        value,
                        &grad,
                        dir.view(),
                        self.istep,
                        self.linesearch,
                        &self.control,
                        self.atom_maxmove,
                        self.accept,
                        &mut self.e_hist,
                        &self.manifold,
                        self.level_set,
                        self.istep,
                        line_options,
                    )?;
                    energy_stall = t.energy_noise;
                    *x = t.x;
                    value = t.f;
                    grad = t.g;
                    if self.accept != Accept::Step {
                        self.istep = qn_istep(&self.control);
                    }
                }
                Inner::Nlcg {
                    conjugacy,
                    restart,
                    dir,
                    g_old,
                    z_old,
                    d_old,
                    initialized,
                    z_epoch,
                    prev_alpha,
                    prev_gd,
                } => {
                    let z_now = metric_vector(&self.precon, &grad)?;
                    if !*initialized || *z_epoch != self.metric_epoch {
                        // A new or replaced P makes z_old belong to the old
                        // matrix. This iteration is steepest descent in the
                        // current metric, so the next β is formed with one P.
                        if *initialized {
                            *prev_alpha = 0.0;
                            *prev_gd = 0.0;
                        }
                        *dir = nlcg_steepest(&z_now, self.accept, self.istep);
                        *g_old = grad.clone();
                        *z_old = z_now;
                        *d_old = dir.clone();
                        *z_epoch = self.metric_epoch;
                        *initialized = true;
                    }
                    let gd_used = crate::vecops::dot(grad.view(), dir.view());
                    let cap = cap_alpha(dir.view(), &self.control, self.atom_maxmove);
                    let first = if self.precon.is_some() {
                        1.0
                    } else {
                        self.control.istep
                    };
                    let alpha0 = if self.accept == Accept::Step {
                        self.istep
                    } else {
                        nlcg_opening_alpha(*prev_alpha, *prev_gd, gd_used, cap, first)
                    };
                    let t = take_session_step(
                        obj,
                        x,
                        value,
                        &grad,
                        dir.view(),
                        alpha0,
                        self.linesearch,
                        &self.control,
                        self.atom_maxmove,
                        self.accept,
                        &mut self.e_hist,
                        &self.manifold,
                        self.level_set,
                        1.0,
                        line_options,
                    )?;
                    energy_stall = t.energy_noise;
                    *x = t.x;
                    value = t.f;
                    grad = t.g;
                    if t.moved || self.accept != Accept::Step {
                        let z_new = metric_vector(&self.precon, &grad)?;
                        let ctx = ConjugacyContext {
                            current_gradient: grad.view(),
                            previous_gradient: g_old.view(),
                            previous_direction: d_old.view(),
                        };
                        let mut beta = if *z_epoch != self.metric_epoch {
                            0.0
                        } else {
                            conjugacy.beta_metric(&ctx, z_new.view(), z_old.view())
                        };
                        if restart.should_restart_metric(
                            grad.view(),
                            z_new.view(),
                            g_old.view(),
                            z_old.view(),
                        ) {
                            beta = 0.0;
                        }
                        *dir = if self.accept == Accept::Step {
                            Array1::from_iter(
                                z_new
                                    .iter()
                                    .zip(d_old.iter())
                                    .map(|(z, d)| -z * self.istep + beta * d),
                            )
                        } else {
                            Array1::from_iter(
                                z_new.iter().zip(d_old.iter()).map(|(z, d)| -z + beta * d),
                            )
                        };
                        // A non-descent direction is not a valid conjugate step.
                        let gd = crate::vecops::dot(grad.view(), dir.view());
                        if gd.partial_cmp(&0.0) != Some(std::cmp::Ordering::Less) {
                            self.nlcg_descent_fallbacks += 1;
                            *dir = nlcg_steepest(&z_new, self.accept, self.istep);
                        }
                        g_old.assign(&grad);
                        z_old.assign(&z_new);
                        d_old.assign(dir);
                        *z_epoch = self.metric_epoch;
                    }
                    if self.accept != Accept::Step {
                        *prev_alpha = t.alpha;
                        *prev_gd = gd_used;
                    }
                }
                Inner::Bfgs { h, .. } => {
                    let direction = -h.dot(&grad);
                    let old = x.clone();
                    let gold = grad.clone();
                    let t = take_session_step(
                        obj,
                        x,
                        value,
                        &grad,
                        direction.view(),
                        self.istep,
                        self.linesearch,
                        &self.control,
                        self.atom_maxmove,
                        self.accept,
                        &mut self.e_hist,
                        &self.manifold,
                        self.level_set,
                        1.0,
                        line_options,
                    )?;
                    energy_stall = t.energy_noise;
                    let moved = t.moved;
                    *x = t.x;
                    value = t.f;
                    grad = t.g;
                    if moved {
                        bfgs_inverse_update(h, &(&*x - &old), &(&grad - &gold));
                    }
                    if self.accept != Accept::Step {
                        self.istep = qn_istep(&self.control);
                    }
                }
                Inner::Sr1 { .. } | Inner::Sr2 { .. } => {}
                Inner::Adam {
                    m,
                    v,
                    b1p,
                    b2p,
                    beta1,
                    beta2,
                    eps,
                } => {
                    let m_old = m.clone();
                    let v_old = v.clone();
                    let raw = adam_direction(m, v, &grad, *beta1, *beta2, (*b1p, *b2p), *eps);
                    let dir = scale_to_unit_atom(&raw);
                    let t = take_session_step(
                        obj,
                        x,
                        value,
                        &grad,
                        dir.view(),
                        self.istep,
                        self.linesearch,
                        &self.control,
                        self.atom_maxmove,
                        self.accept,
                        &mut self.e_hist,
                        &self.manifold,
                        self.level_set,
                        self.istep,
                        line_options,
                    )?;
                    energy_stall = t.energy_noise;
                    if t.moved {
                        *x = t.x;
                        value = t.f;
                        grad = t.g;
                        *b1p *= *beta1;
                        *b2p *= *beta2;
                        if self.accept != Accept::Step && t.alpha.is_finite() && t.alpha > 0.0 {
                            self.istep = t.alpha;
                        }
                    } else {
                        m.assign(&m_old);
                        v.assign(&v_old);
                    }
                }
                Inner::Fire(state) => {
                    // A refused trial returns with the integrator as it was,
                    // so a retry probes the same point.
                    let saved = state.checkpoint();
                    let manifold = self.manifold.clone();
                    let curved = !manifold.is_flat();
                    let n_pos_saved = state.n_pos;
                    let dt_saved = state.dt;
                    let dt_max_saved = state.dt_max;
                    let alpha_saved = state.alpha;
                    let vel_saved = state.vel.clone();
                    // The force, its metric image and the velocity leave the
                    // rigid subspace at the start geometry; the velocity in the
                    // momentum metric, the force in the force metric.
                    let mut raw = grad.mapv(|g| -g);
                    rigid.force(&mut hold.ws, &mut raw);
                    if level_set_on {
                        raw = project_tangent(grad.view(), &raw)?;
                    }
                    #[allow(unused_mut)]
                    let mut steepest = metric_vector(&self.precon, &raw)?;
                    rigid.force(&mut hold.ws, &mut steepest);
                    if level_set_on {
                        steepest = project_tangent(grad.view(), &steepest)?;
                    }
                    rigid.velocity(&mut hold.ws, &mut state.vel);
                    if level_set_on {
                        state.vel = project_tangent(grad.view(), &state.vel)?;
                    }
                    #[cfg(feature = "highs")]
                    if boxed {
                        crate::box_objective::project_tangent(obj.bounds(), x, &mut raw);
                        crate::box_objective::project_tangent(obj.bounds(), x, &mut steepest);
                        crate::box_objective::project_tangent(obj.bounds(), x, &mut state.vel);
                    }
                    let mut power_trace = crate::vecops::dot(raw.view(), state.vel.view());
                    let had_vel = state.vel.iter().any(|v| *v != 0.0);
                    let mut hsb_trace = false;
                    let mix = fire_mix(&self.precon, &state.vel, &raw, &steepest);
                    let displacement = match fire2_local.as_mut() {
                        Some(ext) => {
                            hsb_trace = power_trace <= 0.0 && had_vel;
                            fire2_displacement_metric(state, ext, &raw, &steepest, mix)
                                .map_err(|_| ())
                        }
                        None => match fire_prepare_v2(state, &raw) {
                            FirePreparation::Stalled => Err(()),
                            FirePreparation::Rollback(correction) => {
                                hsb_trace = true;
                                if curved {
                                    let back = manifold.project(x, &correction);
                                    let Some((point, _)) =
                                        retract_within_cap(obj, x, &back, None, None, &manifold)
                                    else {
                                        state.restore(saved);
                                        *x = start;
                                        return Err(Error::OffManifold {
                                            kind: manifold.as_str(),
                                        });
                                    };
                                    *x = point;
                                } else {
                                    *x = &*x + &correction;
                                }
                                let ev = obj.value_and_gradient(x.view());
                                if !crate::step::finite(ev.0, ev.1.view()) {
                                    state.restore(saved);
                                    *x = start;
                                    return Err(crate::step::non_finite());
                                }
                                value = ev.0;
                                grad = ev.1;
                                raw = grad.mapv(|g| -g);
                                let mut dx = Array1::zeros(x.len());
                                fire_displacement_into(state, &raw, &mut dx);
                                Ok(dx)
                            }
                            FirePreparation::Ready => {
                                let mut dx = Array1::zeros(x.len());
                                fire_displacement_into(state, &raw, &mut dx);
                                Ok(dx)
                            }
                        },
                    };
                    let mut dx = match displacement {
                        Ok(dx) => dx,
                        Err(()) => {
                            // The published integrator increments the negative-power
                            // count and then returns before it changes dt, alpha,
                            // or velocity. Undo the rigid projection of velocity
                            // and the n_pos clear so this stopping step leaves
                            // the integrator as it was, apart from that count.
                            state.n_pos = n_pos_saved;
                            state.dt = dt_saved;
                            state.dt_max = dt_max_saved;
                            state.alpha = alpha_saved;
                            state.vel.assign(&vel_saved);
                            stalled = true;
                            Array1::zeros(state.vel.len())
                        }
                    };
                    let mut clip_trace = false;
                    if !stalled
                        && state.abc
                        && let Some(cap) = self.atom_maxmove.or(self.control.maxmove_cap())
                    {
                        clip_trace = fire_abc_clipped(state, cap);
                        dx = fire_clip_components(state, cap);
                    }
                    if !stalled {
                        rigid.force(&mut hold.ws, &mut dx);
                        let trial = if curved {
                            // The chord leaves the set; the trial is the
                            // retraction of the tangent step, inside the cap.
                            let (atom_cap, cap) = if state.abc {
                                (None, None)
                            } else {
                                (self.atom_maxmove, self.control.maxmove_cap())
                            };
                            let tangent = manifold.project(x, &dx);
                            if tangent.iter().any(|v| !v.is_finite()) {
                                state.restore(saved);
                                *x = start;
                                return Err(crate::step::non_finite());
                            }
                            let Some((trial, used)) =
                                retract_within_cap(obj, x, &tangent, atom_cap, cap, &manifold)
                            else {
                                state.restore(saved);
                                *x = start;
                                return Err(Error::OffManifold {
                                    kind: manifold.as_str(),
                                });
                            };
                            if fire_move_clipped(&used, &tangent) {
                                clip_trace = true;
                            }
                            if matches!(state.kind, FireKind::V2) && state.dt > 0.0 {
                                let mut secant = (&trial - &*x) / state.dt;
                                manifold.project_in_place(&trial, &mut secant);
                                state.vel = secant;
                            }
                            trial
                        } else {
                            let mut trial = x.clone();
                            trial += &dx;
                            let plain = state.preset() == FirePreset::Default
                                && !state.abc
                                && fire2_local.is_none();
                            let unclipped = (!plain || self.fire_trace_on).then(|| trial.clone());
                            if !state.abc {
                                apply_caps(
                                    x,
                                    &mut trial,
                                    self.atom_maxmove,
                                    self.control.maxmove_cap(),
                                );
                            }
                            let bounds = obj.bounds();
                            if (bounds.low.iter().any(|v| v.is_finite())
                                || bounds.high.iter().any(|v| v.is_finite()))
                                && trial.iter().all(|v| v.is_finite())
                            {
                                trial = bounds.clip(trial.view());
                            }
                            let shortened = unclipped
                                .as_ref()
                                .is_some_and(|free| fire_move_clipped(&trial, free));
                            if shortened {
                                clip_trace = true;
                            }
                            if unclipped
                                .as_ref()
                                .is_some_and(|free| free.iter().any(|v| !v.is_finite()))
                                || trial.iter().any(|v| !v.is_finite())
                            {
                                state.restore(saved);
                                *x = start;
                                return Err(crate::step::non_finite());
                            }
                            // The clamp and clip shorten the move; the velocity has to
                            // describe the move taken, or the power adapt reads an
                            // integral of moves that never happened. An unshortened
                            // move leaves the velocity alone: `trial - x` is not
                            // always bit-identical to `dx`. A move rounding swallows
                            // whole is shortened to nothing.
                            let swallowed = trial == *x && dx.iter().any(|v| *v != 0.0);
                            if swallowed
                                || ((plain || shortened)
                                    && !state.abc
                                    && (self.atom_maxmove.is_some()
                                        || self.control.maxmove_cap().is_some()))
                            {
                                let taken = &trial - &*x;
                                fire_rescale_velocity(state, &dx, &taken);
                            }
                            trial
                        };
                        let mut trial = trial;
                        let mut ev = obj.value_and_gradient(trial.view());
                        // A flat trial with no answer retreats toward the start,
                        // inside the clip box, before the step is refused.
                        let mut halvings = 0;
                        while !curved
                            && !crate::step::finite(ev.0, ev.1.view())
                            && halvings < crate::accept::HALVINGS
                        {
                            let before = &trial - &*x;
                            let after = before.mapv(|v| 0.5 * v);
                            trial = &*x + &after;
                            fire_rescale_velocity(state, &before, &after);
                            halvings += 1;
                            ev = obj.value_and_gradient(trial.view());
                        }
                        if !crate::step::finite(ev.0, ev.1.view()) {
                            state.restore(saved);
                            *x = start;
                            return Err(crate::step::non_finite());
                        }
                        *x = trial;
                        if rigid.mode != RigidMode::Off
                            && let Some(xs) = x.as_slice()
                        {
                            hold.ws.set_anchor(xs);
                        }
                        value = ev.0;
                        grad = ev.1;
                        let mut raw_new = grad.mapv(|g| -g);
                        rigid.force(&mut hold.ws, &mut raw_new);
                        #[allow(unused_mut)]
                        let mut steep_new = metric_vector(&self.precon, &raw_new)?;
                        rigid.force(&mut hold.ws, &mut steep_new);
                        rigid.velocity(&mut hold.ws, &mut state.vel);
                        #[cfg(feature = "highs")]
                        if boxed {
                            crate::box_objective::project_tangent(obj.bounds(), x, &mut raw_new);
                            crate::box_objective::project_tangent(obj.bounds(), x, &mut steep_new);
                            crate::box_objective::project_tangent(obj.bounds(), x, &mut state.vel);
                        }
                        if matches!(state.kind, FireKind::V1) {
                            power_trace = crate::vecops::dot(raw_new.view(), state.vel.view());
                        }
                        let mix_new = fire_mix(&self.precon, &state.vel, &raw_new, &steep_new);
                        fire_after_v1_metric(state, &raw_new, &steep_new, mix_new);
                        if self.fire_trace_on {
                            let (prev_hsb, prev_clip) = match self.fire_trace.last() {
                                Some(last) => (last.half_step_backs, last.clipped_moves),
                                None => (0, 0),
                            };
                            self.fire_trace.push(FireTrace {
                                dt: state.dt,
                                dt_max: state.dt_max,
                                alpha: state.alpha,
                                power: power_trace,
                                n_pos: state.n_pos,
                                half_step_backs: prev_hsb + usize::from(hsb_trace),
                                clipped_moves: prev_clip + usize::from(clip_trace),
                            });
                        }
                    }
                }
                Inner::Bb { prev_s, prev_y } => {
                    let mut dir = bb_direction(prev_s.as_ref(), prev_y.as_ref(), &grad, self.istep);
                    if level_set_on {
                        dir = project_tangent(grad.view(), &dir)?;
                    }
                    let old = x.clone();
                    let gold = grad.clone();
                    let (npos, nval, ngrad, moved) = accept_step(
                        obj,
                        x,
                        value,
                        &gold,
                        &dir,
                        Acceptance {
                            control: &self.control,
                            accept: self.accept,
                            e_hist: &mut self.e_hist,
                            atom_maxmove: self.atom_maxmove,
                            manifold: &self.manifold,
                        },
                    );
                    *x = npos;
                    value = nval;
                    grad = ngrad;
                    if moved {
                        *prev_s = Some(&*x - &old);
                        *prev_y = Some(&grad - &gold);
                    } else if self.accept != Accept::None {
                        // The refused spectral length would be retried from
                        // the same point on every later call.
                        *prev_s = None;
                        *prev_y = None;
                    }
                }
                Inner::Pso { .. }
                | Inner::Newton { .. }
                | Inner::Dogleg { .. }
                | Inner::NewtonCg { .. }
                | Inner::NewtonPcg { .. } => unreachable!(),
            }

            // A translation retract maps the accepted point to itself; the
            // round trip start + (x - start) only adds rounding (one ulp of
            // a 30 Angstrom coordinate is 3.6e-15) and would buy a second
            // oracle call at the same geometry.
            // BB and the direct first-order routes evaluate the retracted point.
            // A nonlinear retraction applied to its chord is a different point.
            // The stepper is done with the workspace; the hold puts it back.
            drop(hold);
            self.fire2 = fire2_local;
            if stalled {
                return Err(Error::FireStalled { steps: self.steps });
            }
        }
        let curved_trial = !self.manifold.is_flat()
            && matches!(
                &self.inner,
                Inner::Steepest
                    | Inner::Nlcg { .. }
                    | Inner::Bfgs { .. }
                    | Inner::Adam { .. }
                    | Inner::Fire(_)
            );
        let already_retracted = matches!(self.overlay, Overlay::None)
            && (matches!(&self.inner, Inner::Bb { .. })
                || on_set
                || curved_trial
                || (self.accept == Accept::Step
                    && matches!(
                        &self.inner,
                        Inner::Lbfgs(_)
                            | Inner::Steepest
                            | Inner::Nlcg { .. }
                            | Inner::Bfgs { .. }
                            | Inner::Sr1 { .. }
                            | Inner::Sr2 { .. }
                            | Inner::Adam { .. }
                    )));
        let y = if already_retracted || self.manifold.retract_is_translation() {
            None
        } else {
            Some(self.manifold.retract(&start, &(&*x - &start)))
        };
        if let Some(y) = y.filter(|y| y.iter().zip(x.iter()).any(|(a, b)| (*a - *b).abs() > 1e-15))
        {
            *x = y;
            let ev = obj.value_and_gradient(x.view());
            value = ev.0;
            grad = ev.1;
        }
        self.retract_level_set_here(obj, x, &mut value, &mut grad)?;
        let level_grad = grad.clone();
        grad = self.horizontal_grad(x, &grad);
        if self.level_set.is_some()
            && let Inner::Fire(state) = &mut self.inner
        {
            state.vel = project_tangent(level_grad.view(), &state.vel)?;
        }
        if let Some((old_x, old_g)) = sr_secant
            && x.iter().zip(old_x.iter()).any(|(a, b)| a != b)
        {
            let s = &*x - &old_x;
            let y = &grad - &old_g;
            match &mut self.inner {
                Inner::Sr1 { h, .. } => sr1_hessian_update(h, &s, &y),
                Inner::Sr2 { b, .. } => sr2_hessian_update(b, &s, &y),
                _ => {}
            }
        }

        let pair = if matches!(self.inner, Inner::Lbfgs(_))
            && x.iter().zip(start.iter()).any(|(a, b)| a != b)
        {
            let (mut s, mut y) = self.lbfgs_sy(&start, x, &gold, &grad);
            self.project_level_set_pair(&level_grad, &mut s, &mut y)?;
            Some((s, y, l2(&grad)))
        } else {
            None
        };
        if let (Inner::Lbfgs(solver), Some((s, y, gn))) = (&mut self.inner, pair) {
            // Only the final transported pair can enter the history. A
            // refused pair leaves every retained pair in place.
            let kept = solver.push_pair(s, y, Some(gn));
            if !kept && self.level_set.is_some() && self.accept == Accept::Step {
                solver.forget();
            }
        }

        self.remember(x, value, &grad);
        self.steps += 1;
        let grad_norm = self.stationarity_norm(x, &grad);
        let mut status = self
            .control
            .classify(grad_norm, self.free_len(obj.bounds()));
        if energy_stall && status == Status::Iterating {
            status = Status::NoiseFloor;
        }
        Ok(self.finish(value, x, &grad, grad_norm, status))
    }

    fn step_pso<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let bounds = obj.bounds();
        let (n_particles, _inertia, _c1, _c2) = match &self.inner {
            Inner::Pso {
                n_particles,
                inertia,
                c1,
                c2,
                ..
            } => (*n_particles, *inertia, *c1, *c2),
            _ => unreachable!(),
        };
        if match &self.inner {
            Inner::Pso { swarm, .. } => swarm.is_none(),
            _ => true,
        } {
            let mut rng = StdRng::seed_from_u64(RNG_SEED);
            let start_val = obj.eval(x.view());
            let mut swarm = Vec::with_capacity(n_particles);
            swarm.push(Particle {
                velocity: random_velocity(bounds, &mut rng),
                best_position: x.clone(),
                best_value: start_val,
                position: x.clone(),
            });
            let mut gbest_position = swarm[0].best_position.clone();
            let mut gbest_value = start_val;
            for _ in 1..n_particles {
                let position = bounds.mkpoint(&mut rng);
                let value = obj.eval(position.view());
                let particle = Particle {
                    velocity: random_velocity(bounds, &mut rng),
                    best_position: position.clone(),
                    best_value: value,
                    position,
                };
                if particle.best_value < gbest_value {
                    gbest_position = particle.best_position.clone();
                    gbest_value = particle.best_value;
                }
                swarm.push(particle);
            }
            if let Inner::Pso { swarm: slot, .. } = &mut self.inner {
                *slot = Some(PsoState {
                    swarm,
                    gbest_position,
                    gbest_value,
                    rng,
                });
            }
        }
        if let Inner::Pso {
            swarm: Some(state),
            inertia,
            c1,
            c2,
            ..
        } = &mut self.inner
        {
            update_swarm(
                obj,
                bounds,
                &mut state.swarm,
                &mut state.gbest_position,
                &mut state.gbest_value,
                (*inertia, *c1, *c2),
                &mut state.rng,
            );
            *x = state.gbest_position.clone();
            let (value, grad) = obj.value_and_gradient(x.view());
            self.steps += 1;
            return Ok(self.finish(
                value,
                x,
                &grad,
                self.stationarity_norm(x, &grad),
                self.control.classify(
                    self.stationarity_norm(x, &grad),
                    self.free_len(obj.bounds()),
                ),
            ));
        }
        unreachable!("PSO slot populated above")
    }

    fn sr_opening_radius(&self) -> f64 {
        sr_trust_limits(
            self.control.istep,
            self.dim,
            self.atom_maxmove,
            self.control.maxmove_cap(),
        )
        .0
    }

    fn reset_sr_radius(&mut self) {
        if matches!(self.inner, Inner::Sr1 { .. } | Inner::Sr2 { .. }) {
            self.sr_radius = self.sr_opening_radius();
        }
    }

    fn clip_trial(&self, origin: &Array1<f64>, trial: &mut Array1<f64>) {
        apply_caps(origin, trial, self.atom_maxmove, self.control.maxmove_cap());
    }

    /// One SR1/SR2 trust-region trial. The Hessian update waits until the
    /// accepted point has been retracted.
    fn sr_trust_attempt<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        value: &mut f64,
        grad: &mut Array1<f64>,
        start: &Array1<f64>,
        gold: &Array1<f64>,
        secant: &mut Option<(Array1<f64>, Array1<f64>)>,
    ) where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let radius = self.sr_radius;
        let model = match &self.inner {
            Inner::Sr1 { h, .. } => h.clone(),
            Inner::Sr2 { b, .. } => b.clone(),
            _ => return,
        };
        let (_, rmax) = sr_trust_limits(
            self.control.istep,
            self.dim,
            self.atom_maxmove,
            self.control.maxmove_cap(),
        );
        let mut dir = sr_trust_direction(&model, grad, radius);
        if self.project_rigid {
            project_out_rot_trans(&mut dir, x.view());
        }
        dir = self.project_vec(x, &dir);
        let mut trial = &*x + &dir;
        self.clip_trial(x, &mut trial);
        trial = obj.bounds().clip(trial.view());
        let p = &trial - &*x;
        let pnorm = l2(&p);
        let mut rho = -1.0;
        let mut finite = false;
        let mut ft = *value;
        let mut gt = grad.clone();
        if pnorm.is_finite() && pnorm > 1e-16 {
            let ev = obj.value_and_gradient(trial.view());
            ft = ev.0;
            gt = ev.1;
            finite = ft.is_finite() && gt.iter().all(|v| v.is_finite());
            if finite {
                rho = reduction_ratio(*value - ft, predicted_reduction(&model, grad, &p));
            }
        }
        let new_radius = update_radius(radius, rho, pnorm, rmax);
        self.sr_radius = new_radius;
        let take = finite && (self.accept == Accept::Step || accept_ratio(rho));
        if take {
            *secant = Some((start.clone(), gold.clone()));
            *x = trial;
            *value = ft;
            *grad = gt;
        }
    }
    fn reset_overlay(&mut self) {
        match &mut self.overlay {
            Overlay::QuickMin(state, session) => {
                state.reset();
                session.forget_last();
            }
            Overlay::Ode12r(state) => state.reset(),
            Overlay::Diis(state) => state.clear(),
            Overlay::None => {}
        }
    }

    /// Outer steps completed on this session.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Norm that every stop test and [`Report::grad_norm`] apply to the gradient.
    pub fn set_stop_norm(&mut self, norm: StopNorm) {
        self.stop_norm = norm;
    }

    /// Opt in to [`crate::Error::StopUnmet`] from [`Self::run`].
    ///
    /// The default is off. One [`Self::step`] is unchanged: a step that
    /// is still iterating is not an unmet minimization.
    pub fn set_error_on_unmet(&mut self, on: bool) {
        self.control.error_on_unmet = on;
    }

    /// Record the backward-error polish flag on this session's control.
    ///
    /// [`Self::run`] does not form a Hessian. [`crate::minimize_newton`],
    /// [`crate::minimize_method_hess`], [`crate::minimize_method_hvp`],
    /// [`crate::polish_newton`] and [`crate::polish_hvp`] read the flag.
    pub fn set_backward_polish(&mut self, on: bool) {
        self.control.backward_polish = on;
    }

    /// Norm selected by [`Self::set_stop_norm`].
    pub fn stop_norm(&self) -> StopNorm {
        self.stop_norm
    }

    /// Parameter set of a FIRE 2.0 session. The integrator restarts from
    /// [`Control::istep`]. [`FirePreset::Guenole2020`] also installs the
    /// published schedule, so a non-positive power run stops at `vdfmax`.
    pub fn set_fire_preset(&mut self, preset: FirePreset) -> Result<()> {
        let dt = match &mut self.inner {
            Inner::Fire(state) if state.kind == FireKind::V2 => {
                state.set_preset(preset);
                state.dt
            }
            _ => return Err(Error::NeedFire2),
        };
        // The preset keeps the prepare/into integrator. The published
        // Guénolé integrator, including its vdfmax stop, is
        // [`Self::set_fire_variant`].
        let _ = dt;
        self.fire2 = None;
        self.apply_default_atom_cap();
        Ok(())
    }

    /// Integrator parameters of a FIRE session. The velocity is cleared.
    pub fn set_fire_config(&mut self, config: FireConfig) -> Result<()> {
        match &mut self.inner {
            Inner::Fire(state) => {
                state.apply_config(&config);
                if config.dt.is_finite() && config.dt > 0.0 {
                    self.control.istep = config.dt;
                    self.istep = config.dt;
                }
                Ok(())
            }
            _ => Err(Error::NeedFire),
        }
    }

    /// Opt into capping `dt_max` after one downhill step followed by an uphill one.
    pub fn set_fire_cap_two_step_overshoot(&mut self, on: bool) -> Result<()> {
        match &mut self.inner {
            Inner::Fire(state) if state.kind == FireKind::V2 => {
                state.set_cap_two_step_overshoot(on);
                Ok(())
            }
            _ => Err(Error::NeedFire2),
        }
    }

    /// ABC-FIRE on a FIRE 2.0 session. The fill path applies the bias.
    /// [`Solver::step`] on the published schedule keeps [`fire2_displacement`](crate::fire::fire2_displacement).
    pub fn set_fire_abc(&mut self, on: bool) -> Result<()> {
        match &mut self.inner {
            Inner::Fire(state) if state.kind == FireKind::V2 => {
                state.abc = on;
                Ok(())
            }
            _ => Err(Error::NeedFire2),
        }
    }

    /// Start recording per-step [`FireTrace`] entries on a FIRE session.
    pub fn start_fire_trace(&mut self) {
        self.fire_trace.clear();
        self.fire_trace_on = true;
    }

    /// Take the recorded [`FireTrace`] entries, if tracing was started.
    pub fn take_fire_trace(&mut self) -> Vec<FireTrace> {
        self.fire_trace_on = false;
        std::mem::take(&mut self.fire_trace)
    }

    /// Step-length rule for an L-BFGS session.
    ///
    /// [`LbfgsStep::Capped`] at [`crate::EON_MAX_MOVE`] is the default:
    /// the eOn step, no line search. The first step probes curvature.
    /// A proposal at least as long as the cap clears the memory and
    /// steps along `H0` times the force. [`LbfgsStep::Scaled`] shortens
    /// the two-loop step and keeps the pairs. [`LbfgsStep::Wolfe`] opts
    /// into the strong Wolfe search.
    pub fn set_lbfgs_step(&mut self, step: LbfgsStep) -> Result<()> {
        let step = match step {
            LbfgsStep::Capped { max_atom } | LbfgsStep::Scaled { max_atom }
                if !max_atom.is_finite() =>
            {
                return Err(Error::LbfgsCap {
                    what: "max_atom is not finite",
                });
            }
            LbfgsStep::Capped { max_atom } if max_atom <= 0.0 => LbfgsStep::Capped {
                max_atom: crate::lbfgs::EON_MAX_MOVE,
            },
            LbfgsStep::Scaled { max_atom } if max_atom <= 0.0 => LbfgsStep::Scaled {
                max_atom: crate::lbfgs::EON_MAX_MOVE,
            },
            other => other,
        };
        if !matches!(self.inner, Inner::Lbfgs(_)) {
            return Err(Error::LbfgsCap {
                what: "requires an L-BFGS session",
            });
        }
        if matches!(step, LbfgsStep::Capped { .. } | LbfgsStep::Scaled { .. }) {
            if !matches!(self.manifold, ManifoldKind::Euclidean) {
                return Err(Error::LbfgsCap {
                    what: "requires Euclidean coordinates",
                });
            }
            if self.constraints.is_some() {
                return Err(Error::LbfgsCap {
                    what: "distance constraints are set",
                });
            }
            #[cfg(feature = "highs")]
            if self.highs {
                return Err(Error::LbfgsCap {
                    what: "HiGHS projection is set",
                });
            }
        }
        let cap = step.capped_limit();
        match &mut self.inner {
            Inner::Lbfgs(solver) => solver.step = step,
            _ => {
                return Err(Error::LbfgsCap {
                    what: "requires an L-BFGS session",
                });
            }
        }
        // Selecting a step rule clears the geometry choice.
        // [`Self::set_geometry_step`] is that algorithm, and a Wolfe
        // session with a per-atom cap also takes it unless set off.
        self.geometry_step = None;
        if let Some(cap) = cap {
            self.set_atom_maxmove(cap);
        }
        Ok(())
    }

    /// Opt into the eOn geometry step at `max_atom`.
    pub fn set_lbfgs_capped(&mut self, max_atom: f64) -> Result<()> {
        self.set_lbfgs_step(LbfgsStep::Capped { max_atom })
    }

    /// Step-length rule stored on an L-BFGS session.
    /// Other methods report [`LbfgsStep::Wolfe`].
    pub fn lbfgs_step(&self) -> LbfgsStep {
        match &self.inner {
            Inner::Lbfgs(solver) => solver.step,
            _ => LbfgsStep::Wolfe,
        }
    }

    /// ASE's fixed inverse-Hessian scale. `None` keeps the Nocedal-Wright scale.
    pub fn set_lbfgs_fixed_h0(&mut self, h0: Option<f64>) -> Result<()> {
        match &mut self.inner {
            Inner::Lbfgs(solver) => {
                solver.fixed_h0 = h0.filter(|v| v.is_finite() && *v > 0.0);
                Ok(())
            }
            _ => Err(Error::LbfgsCap {
                what: "requires an L-BFGS session",
            }),
        }
    }

    /// Measure the L-BFGS cap by the Euclidean length of the step.
    pub fn set_lbfgs_euclidean_cap(&mut self, on: bool) -> Result<()> {
        match &mut self.inner {
            Inner::Lbfgs(solver) => {
                solver.euclidean_cap = on;
                Ok(())
            }
            _ => Err(Error::LbfgsCap {
                what: "requires an L-BFGS session",
            }),
        }
    }

    /// Trust-region radius on the capped fill step. Off by default.
    pub fn set_lbfgs_trust(&mut self, on: bool) -> Result<()> {
        if !matches!(self.inner, Inner::Lbfgs(_)) {
            return Err(Error::LbfgsCap {
                what: "requires an L-BFGS session",
            });
        }
        let cap = self
            .lbfgs_step()
            .capped_limit()
            .unwrap_or(crate::lbfgs::EON_MAX_MOVE);
        self.trust = on;
        self.trust_delta = cap;
        self.trust_max = cap;
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.keep_long = on;
        }
        Ok(())
    }

    /// Keep a long downhill quasi-Newton step and scale it per atom.
    pub fn set_lbfgs_model_atom(&mut self, on: bool) -> Result<()> {
        match &mut self.inner {
            Inner::Lbfgs(solver) => {
                solver.model_atom = on;
                Ok(())
            }
            _ => Err(Error::LbfgsCap {
                what: "requires an L-BFGS session",
            }),
        }
    }

    /// Compact trust-region step inside the per-atom cap.
    pub fn set_lbfgs_compact_tr(&mut self, on: bool) -> Result<()> {
        match &mut self.inner {
            Inner::Lbfgs(solver) => {
                solver.compact_tr = on;
                Ok(())
            }
            _ => Err(Error::LbfgsCap {
                what: "requires an L-BFGS session",
            }),
        }
    }

    /// Drop a pair with `s·y <= 0` and keep the older pairs.
    pub fn set_lbfgs_skip_bad(&mut self, on: bool) -> Result<()> {
        match &mut self.inner {
            Inner::Lbfgs(solver) => {
                solver.skip_bad = on;
                Ok(())
            }
            _ => Err(Error::LbfgsCap {
                what: "requires an L-BFGS session",
            }),
        }
    }

    /// Lindh model used by the eOn fill path. Distinct from [`Self::set_lindh`],
    /// which installs the geometry preconditioner.
    pub fn set_lindh_model(&mut self, spec: Option<LindhSpec>) -> Result<()> {
        if !matches!(self.inner, Inner::Lbfgs(_)) {
            return Err(Error::LbfgsCap {
                what: "requires an L-BFGS session",
            });
        }
        if let Some(spec) = &spec {
            if !(spec.c_stab.is_finite() && spec.c_stab > 0.0) {
                return Err(Error::Precon {
                    what: "c_stab is not positive",
                });
            }
            if !spec.z.is_empty() && spec.z.len() * 3 != self.dim {
                return Err(Error::Precon {
                    what: "atomic numbers do not match the dimension",
                });
            }
        }
        self.lindh_spec = spec;
        self.lindh_mu = None;
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.set_lindh_factor(None);
        }
        Ok(())
    }

    fn ensure_lindh<G>(&mut self, x: &[f64], g: &[f64], mut grad_at: G)
    where
        G: FnMut(&[f64]) -> Array1<f64>,
    {
        if self.lindh_spec.is_none() {
            return;
        }
        if self.lindh_mu.is_none() {
            self.lindh_mu = Some(self.probe_lindh_mu(x, g, &mut grad_at));
        }
        let mu = self.lindh_mu.unwrap_or(1.0);
        let built = self
            .lindh_spec
            .as_ref()
            .and_then(|spec| crate::lindh::factor(x, &spec.z, spec.c_stab, spec.cell, mu));
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.set_lindh_factor(built);
        }
    }

    fn probe_lindh_mu<G>(&self, x: &[f64], g: &[f64], grad_at: &mut G) -> f64
    where
        G: FnMut(&[f64]) -> Array1<f64>,
    {
        let mut n2 = 0.0;
        for v in g {
            n2 += *v * *v;
        }
        let norm = n2.sqrt();
        if !(norm.is_finite() && norm > 0.0 && g.len() == x.len()) {
            return 1.0;
        }
        let fd = crate::lbfgs::EON_FINITE_DIFFERENCE;
        let mut probe = x.to_vec();
        let mut u = vec![0.0; g.len()];
        for i in 0..g.len() {
            u[i] = -g[i] / norm;
            probe[i] = x[i] + fd * u[i];
        }
        let gp = grad_at(&probe);
        if gp.len() != g.len() {
            return 1.0;
        }
        let mut curv = 0.0;
        for i in 0..g.len() {
            curv += (gp[i] - g[i]) * u[i];
        }
        curv /= fd;
        let Some(spec) = self.lindh_spec.as_ref() else {
            return 1.0;
        };
        let quad = crate::lindh::shape_quad(x, &spec.z, spec.c_stab, spec.cell, &u);
        if curv.is_finite() && curv > 0.0 && quad > 1e-18 {
            curv / quad
        } else {
            1.0
        }
    }

    /// Drop the exponential preconditioner. Other geometry metrics stay.
    pub fn clear_exp(&mut self) {
        if self.exp.is_some() {
            self.clear_precon();
        }
    }

    /// Energy scale of the installed exponential preconditioner.
    #[must_use]
    pub fn exp_mu(&self) -> Option<f64> {
        if self.exp.is_none() {
            None
        } else {
            self.precon_scale()
        }
    }

    /// Keep every pair of `set` at its length. The method memory restarts.
    pub fn set_distance_constraints(&mut self, set: DistanceConstraints) -> Result<()> {
        self.refuse_constraint_settings(&set)?;
        self.constraints = Some(Arc::new(set));
        self.forget();
        Ok(())
    }

    /// Drop the constraint set. The method memory restarts.
    pub fn clear_distance_constraints(&mut self) {
        if self.constraints.take().is_some() {
            self.forget();
        }
    }

    /// The constraint set, if any.
    pub fn distance_constraints(&self) -> Option<&DistanceConstraints> {
        self.constraints.as_deref()
    }

    /// Keep accepted points on the level set `value(x) = target`.
    ///
    /// Trial directions, including the capped eOn L-BFGS displacement, and
    /// the FIRE velocity are projected onto the tangent of that set. Each
    /// accepted point is retracted by Newton along the objective gradient.
    /// Retraction stops at a tolerance in the units of the objective, or
    /// returns [`Error::LevelSetStalled`] when the residual does not contract.
    pub fn set_level_set(&mut self, level_set: LevelSetConstraint) -> Result<()> {
        self.level_set = Some(level_set.validate()?);
        Ok(())
    }

    /// Drop the level-set constraint.
    pub fn clear_level_set(&mut self) {
        self.level_set = None;
    }

    /// Active level-set constraint, if any.
    pub fn level_set(&self) -> Option<LevelSetConstraint> {
        self.level_set
    }

    /// Open the first raw-gradient step at `min(istep, 1 / |g|)`. The C API
    /// sets this when the caller leaves `istep` at zero.
    #[cfg(feature = "capi")]
    pub(crate) fn set_unit_first_step(&mut self) {
        self.unit_first_step = true;
    }

    /// Judge `x` with no step: [`Status::Tolerance`] or
    /// [`Status::NoiseFloor`] when the gradient evaluated there meets that
    /// stop, else [`Status::Budget`]. `x` is clipped to the objective's
    /// bounds, as a step clips it, and the method memory is untouched. The
    /// stationarity measure is the one a step opens with; a distance or
    /// equality constraint set does not project the gradient here, so
    /// such a session reads the larger, unprojected norm.
    pub(crate) fn assess<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if x.len() != self.dim || self.dim != Objective::dim(obj) {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim,
            });
        }
        self.check_manifold(x.len())?;
        self.note_bounds(obj.bounds());
        let (value, raw_grad) = if self.same_last_x(x) {
            (self.last_value, self.last_grad.clone())
        } else {
            if x.iter().any(|v| !v.is_finite()) {
                return Err(crate::step::non_finite());
            }
            *x = obj.bounds().clip(x.view());
            self.gradient_calls += 1;
            obj.value_and_gradient(x.view())
        };
        let grad = self.horizontal_grad(x, &raw_grad);
        if !value.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            return Err(Error::Oracle {
                what: "non-finite value or gradient",
            });
        }
        let grad_norm = self.stationarity_norm(x, &grad);
        let status = self
            .control
            .finish_status(grad_norm, self.free_len(obj.bounds()), true);
        Ok(self.finish(value, x, &grad, grad_norm, status))
    }

    /// Minimize until a step reports a stop or the iteration budget is spent.
    ///
    /// The run ends on the first step whose status is not
    /// [`Status::Iterating`] and returns that status, so an energy-noise
    /// [`Status::NoiseFloor`] is kept rather than stepped past. A spent
    /// budget is [`Status::Budget`]. A zero `maxiter` takes no step: the
    /// report is `Self::assess` of `x`, so it says [`Status::Tolerance`]
    /// only when the gradient evaluated there is below `gtol`.
    pub fn run<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if let Some(d) = self.chart.clone() {
            return self.in_chart(&d, obj, x, |s, obj, z| s.run_plain(obj, z));
        }
        self.run_plain(obj, x)
    }

    /// [`Self::run`] with a gauge move between steps.
    ///
    /// `gauge(x)` replaces the iterate with another representation of the
    /// same point: a map that leaves the objective unchanged, such as
    /// re-centring the pivot of a hyperplane on the plane. It acts on the
    /// caller's coordinates before every step but the first, so the
    /// returned report describes the point left in `x`. Every report
    /// carries its coordinates and gradient. The run ends as [`Self::run`]
    /// does, on the first step whose status is not [`Status::Iterating`],
    /// and a spent budget is [`Status::Budget`].
    ///
    /// A move is not a step: the value and gradient before it belong to
    /// the old representation, where a plane rotates about the old pivot,
    /// so their secant with the moved point holds no curvature. A gauge
    /// that changes `x` therefore rebases the session, as
    /// [`Self::rebase`]: the next step evaluates the objective afresh at
    /// the moved point and the retained curvature preconditions it. A
    /// gauge that leaves `x` as it was is no move.
    pub fn run_gauged<O, G>(&mut self, obj: &O, x: &mut Array1<f64>, mut gauge: G) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
        G: FnMut(&mut Array1<f64>),
    {
        if self.control.maxiter == 0 {
            return self.run(obj, x);
        }
        let tol = self.control.gtol;
        let mut report = self.step(obj, x)?;
        for _ in 1..self.control.maxiter {
            if report.grad_norm < tol || report.status != Status::Iterating {
                break;
            }
            let before = x.clone();
            gauge(x);
            if *x != before {
                self.rebase();
            }
            report = self.step(obj, x)?;
        }
        report.status = match report.status {
            Status::Iterating if report.grad_norm < tol => Status::Tolerance,
            Status::Iterating => Status::Budget,
            other => other,
        };
        self.control.deliver(report)
    }

    fn run_plain<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let budget = self.control.maxiter;
        if budget == 0 {
            let report = self.assess(obj, x)?;
            return self.control.deliver(report);
        }
        self.brief = true;
        let tol = self.control.gtol;
        let mut last = None;
        let mut stop: Result<()> = Ok(());
        for _ in 0..budget {
            match self.step_plain(obj, x) {
                Ok(rep) => {
                    let status = match rep.status {
                        Status::Iterating if rep.grad_norm < tol => Status::Tolerance,
                        other => other,
                    };
                    last = Some((rep.value, rep.grad_norm, status));
                    if status != Status::Iterating {
                        break;
                    }
                }
                Err(err) => {
                    stop = Err(err);
                    break;
                }
            }
        }
        self.brief = false;
        stop?;
        let Some((value, grad_norm, status)) = last else {
            unreachable!("a positive budget takes a step or returns its error");
        };
        let status = match status {
            Status::Iterating => Status::Budget,
            other => other,
        };
        let g = self.reported_grad(x);
        self.control
            .deliver(self.finish(value, x, &g, grad_norm, status))
    }

    /// The gradient a step report carries, for the report at the end of a
    /// run. The unconstrained Newton-CG path remembers the raw gradient,
    /// which its difference products need; its reports are horizontal.
    fn reported_grad(&mut self, x: &Array1<f64>) -> Array1<f64> {
        let raw = self.constraints.is_none()
            && matches!(self.inner, Inner::NewtonCg { .. } | Inner::NewtonPcg { .. });
        if raw && self.last_grad.len() == x.len() {
            let g = self.last_grad.clone();
            self.horizontal_grad(x, &g)
        } else {
            self.last_grad.clone()
        }
    }

    /// Euclidean FIRE with the default preset and no preconditioner.
    fn fire_is_flat(&self) -> bool {
        let Inner::Fire(state) = &self.inner else {
            return false;
        };
        matches!(self.manifold, ManifoldKind::Euclidean)
            && !self.project_rigid
            && self.exp.is_none()
            && self.fire2.is_none()
            && !self.fire_trace_on
            && !state.abc
            && state.preset() == FirePreset::Default
    }

    fn use_scaled_lbfgs(&self) -> bool {
        if !matches!(self.lbfgs_step(), LbfgsStep::Scaled { .. }) {
            return false;
        }
        if self.geometry_step == Some(true) || !matches!(self.manifold, ManifoldKind::Euclidean) {
            return false;
        }
        if !matches!(self.inner, Inner::Lbfgs(_)) || self.constraints.is_some() {
            return false;
        }
        #[cfg(feature = "highs")]
        if self.highs {
            return false;
        }
        true
    }

    /// ASE length control: shorten the two-loop step, keep the pairs.
    /// A non-descent direction clears the memory and steps along the force.
    /// One non-finite trial is retried; a second restores the pairs.
    fn scaled_lbfgs_step<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        _value: f64,
        grad: Array1<f64>,
    ) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let cap = match self.lbfgs_step() {
            LbfgsStep::Scaled { max_atom } if max_atom > 0.0 => max_atom,
            _ => {
                return Err(Error::LbfgsCap {
                    what: "the cap is not positive",
                });
            }
        };
        let saved = match &self.inner {
            Inner::Lbfgs(solver) => solver.curvature(),
            _ => {
                return Err(Error::LbfgsCap {
                    what: "requires an L-BFGS session",
                });
            }
        };
        if !self.same_last_x(x)
            && let Some(previous) = self.last_pos.clone()
        {
            let s = &*x - &previous;
            let y = &grad - &self.last_grad;
            if let Inner::Lbfgs(solver) = &mut self.inner {
                let _ = solver.push(s, y);
            }
        }
        let proposed = match self.scaled_trial(obj, x, &grad, cap) {
            Ok(point) => point,
            Err(Error::Oracle { .. }) => {
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.forget();
                }
                match self.scaled_trial(obj, x, &grad, cap) {
                    Ok(point) => point,
                    Err(_) => {
                        if let Inner::Lbfgs(solver) = &mut self.inner {
                            solver.restore(saved);
                        }
                        return Err(Error::Oracle {
                            what: "non-finite value or gradient",
                        });
                    }
                }
            }
            Err(err) => {
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.restore(saved);
                }
                return Err(err);
            }
        };
        let (trial, tval, tgrad) = proposed;
        let old = x.clone();
        let gold = grad;
        *x = trial;
        let grad = self.horizontal_grad(x, &tgrad);
        let s = &*x - &old;
        let y = &grad - &gold;
        if let Inner::Lbfgs(solver) = &mut self.inner {
            let _ = solver.push(s, y);
        }
        self.remember(x, tval, &grad);
        self.steps += 1;
        Ok(self.report(tval, x, &grad))
    }

    fn scaled_trial<O>(
        &mut self,
        obj: &O,
        x: &Array1<f64>,
        grad: &Array1<f64>,
        cap: f64,
    ) -> Result<(Array1<f64>, f64, Array1<f64>)>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let mut dir = match &self.inner {
            Inner::Lbfgs(solver) => solver.direction(grad.view()),
            _ => grad.mapv(|g| -g),
        };
        let slope = dir.dot(grad);
        if !(slope.is_finite() && slope < 0.0) {
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.forget();
            }
            dir = grad.mapv(|g| -g);
        }
        dir = self.project_vec(x, &dir);
        let mut trial = x + &dir;
        crate::step::scale_step_atom(x, &mut trial, cap);
        trial = obj.bounds().clip(trial.view());
        let (value, next) = obj.value_and_gradient(trial.view());
        if !value.is_finite() || next.iter().any(|v| !v.is_finite()) {
            return Err(Error::Oracle {
                what: "non-finite value or gradient",
            });
        }
        Ok((trial, value, next))
    }

    fn overlay_apply<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        value_out: &mut f64,
        grad_out: &mut Array1<f64>,
        energy_out: &mut bool,
        line_options: LineSearchOptions,
        diis_alpha: f64,
        diis_atom_cap: Option<f64>,
        diis_euclidean_cap: Option<f64>,
    ) -> Result<()>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let mut value = *value_out;
        let mut grad = grad_out.clone();
        let mut energy_stall = *energy_out;
        #[cfg(feature = "highs")]
        let boxed = self.has_coordinate_box();
        let mut overlay = std::mem::replace(&mut self.overlay, Overlay::None);
        match &mut overlay {
            Overlay::QuickMin(state, session) => {
                #[allow(unused_mut)]
                let mut force = grad.mapv(|g| -g);
                #[cfg(feature = "highs")]
                if boxed {
                    crate::box_objective::project_tangent(obj.bounds(), x, &mut force);
                    crate::box_objective::project_tangent(obj.bounds(), x, &mut state.vel);
                }
                let dx = quickmin_session_step(state, session, x, &force);
                let mut trial = &*x + &dx;
                apply_caps(x, &mut trial, self.atom_maxmove, self.control.maxmove_cap());
                trial = obj.bounds().clip(trial.view());
                if session.options.rescale_on_clamp {
                    quickmin_rescale_velocity(&mut state.vel, &dx, &(&trial - &*x));
                }
                *x = trial;
                let ev = obj.value_and_gradient(x.view());
                value = ev.0;
                grad = ev.1;
                #[cfg(feature = "highs")]
                if boxed {
                    crate::box_objective::project_tangent(obj.bounds(), x, &mut state.vel);
                }
            }
            Overlay::Ode12r(state) => {
                // Forward Euler on P^{-1} F with the ODE12r acceptance test.
                // A rejected trial keeps x, value and grad; the next call
                // reads them from the cache and retries with a shorter h.
                #[allow(unused_mut)]
                let mut force = grad.mapv(|g| -g);
                #[allow(unused_mut)]
                let mut steepest = metric_vector(&self.precon, &force)?;
                #[cfg(feature = "highs")]
                if boxed {
                    crate::box_objective::project_tangent(obj.bounds(), x, &mut force);
                    crate::box_objective::project_tangent(obj.bounds(), x, &mut steepest);
                }
                let h = state.open();
                let dx = steepest.mapv(|v| h * v);
                let proposed = &*x + &dx;
                let mut trial = proposed.clone();
                apply_caps(x, &mut trial, self.atom_maxmove, self.control.maxmove_cap());
                trial = obj.bounds().clip(trial.view());
                // Only a move the cap or the box changed is shorter than h.
                // The length ratio of an unchanged move rounds, and one ulp
                // under h would trip the capped-move floor below.
                let h_taken = if trial == proposed {
                    h
                } else {
                    crate::ode12r::taken_step(h, &dx, &(&trial - &*x))
                };
                let (trial_value, trial_grad) = obj.value_and_gradient(trial.view());
                #[allow(unused_mut)]
                let mut force_new = trial_grad.mapv(|g| -g);
                #[allow(unused_mut)]
                let mut steepest_new = metric_vector(&self.precon, &force_new)?;
                #[cfg(feature = "highs")]
                if boxed {
                    crate::box_objective::project_tangent(obj.bounds(), &trial, &mut force_new);
                    crate::box_objective::project_tangent(obj.bounds(), &trial, &mut steepest_new);
                }
                let verdict = state.judge(&force, &steepest, &force_new, &steepest_new, h_taken);
                // An accepted move that the cap shortened is a length the
                // surface took; the next trial starts no shorter, so the
                // error estimate of a cap-bound move cannot shrink the
                // step below the cap it is already held at.
                if verdict.accepted && h_taken < h {
                    state.h = state.h.max(h_taken);
                }
                if verdict.accepted {
                    *x = trial;
                    value = trial_value;
                    grad = trial_grad;
                }
            }
            Overlay::Diis(state) => {
                let alpha = diis_alpha;
                let cap = diis_atom_cap.or(diis_euclidean_cap);
                let gnorm_now = l2(&grad);
                state.push(x.clone(), grad.clone());
                let image = gdiis_point(state.positions(), state.gradients(), alpha, cap);
                let built = state.gradients().len() >= 2;
                let gdiis_dir = image.and_then(|y| {
                    let d = &y - &*x;
                    (d.dot(&grad) < 0.0 && l2(&d) > 1e-16).then_some(d)
                });
                let tried_gdiis = gdiis_dir.is_some();
                let (dir, open, direct) = if let Some(d) = gdiis_dir {
                    (d, 1.0, 1.0)
                } else {
                    if built {
                        state.clear();
                    }
                    (grad.mapv(|g| -g), self.istep, self.istep)
                };
                let t = take_session_step(
                    obj,
                    x,
                    value,
                    &grad,
                    dir.view(),
                    open,
                    self.linesearch,
                    &self.control,
                    self.atom_maxmove,
                    self.accept,
                    &mut self.e_hist,
                    &self.manifold,
                    self.level_set,
                    direct,
                    line_options,
                )?;
                energy_stall = t.energy_noise;
                let force_fell = l2(&t.g) < gnorm_now;
                if tried_gdiis && !(t.moved && force_fell) {
                    state.clear();
                }
                *x = t.x;
                value = t.f;
                grad = t.g;
                if self.accept != Accept::Step {
                    self.istep = qn_istep(&self.control);
                }
            }

            Overlay::None => {}
        }
        self.overlay = overlay;
        *value_out = value;
        *grad_out = grad;
        *energy_out = energy_stall;
        Ok(())
    }
}

// The explicit step policy uses the method's direction and scale once.
// Other policies retain the configured line search and its initial step
// on a flat kind. A curved kind takes the retracted trial under the
// policy: the line search would evaluate the chord, off the set.
#[allow(clippy::too_many_arguments)]
fn take_session_step<O>(
    obj: &O,
    pos: &Array1<f64>,
    value: f64,
    grad: &Array1<f64>,
    dir: ArrayView1<'_, f64>,
    istep: f64,
    linesearch: LineSearch,
    control: &Control,
    atom_maxmove: Option<f64>,
    accept: Accept,
    e_hist: &mut VecDeque<f64>,
    manifold: &ManifoldKind,
    level_set: Option<LevelSetConstraint>,
    direct_scale: f64,
    options: LineSearchOptions,
) -> Result<Taken>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    if accept != Accept::Step && manifold.is_flat() && level_set.is_none() {
        return Ok(take_step_with_options(
            obj,
            pos,
            value,
            grad,
            dir,
            istep,
            linesearch,
            control,
            atom_maxmove,
            options,
        ));
    }
    if accept != Accept::Step && manifold.is_flat() {
        let direction = project_tangent(grad.view(), &dir.to_owned())?;
        return Ok(take_step_with_options(
            obj,
            pos,
            value,
            grad,
            direction.view(),
            istep,
            linesearch,
            control,
            atom_maxmove,
            options,
        ));
    }
    let mut direction = if direct_scale == 1.0 {
        dir.to_owned()
    } else {
        &dir * direct_scale
    };
    if level_set.is_some() {
        direction = project_tangent(grad.view(), &direction)?;
    }
    let (x, f, g, moved) = accept_step(
        obj,
        pos,
        value,
        grad,
        &direction,
        Acceptance {
            control,
            accept,
            e_hist,
            atom_maxmove,
            manifold,
        },
    );
    Ok(Taken {
        x,
        f,
        g,
        alpha: if moved { direct_scale } else { 0.0 },
        moved,
        energy_noise: false,
    })
}

impl Inner {
    fn from_method(method: &Method, dim: usize, istep: f64) -> Self {
        match method {
            Method::Lbfgs { memory } => {
                let mut solver = Lbfgs::with_capacity(*memory);
                solver.norm = GradNorm::Euclidean;
                Inner::Lbfgs(solver)
            }
            Method::Nlcg { conjugacy, restart } => Inner::Nlcg {
                conjugacy: conjugacy.clone(),
                restart: *restart,
                dir: Array1::zeros(dim),
                g_old: Array1::zeros(dim),
                z_old: Array1::zeros(dim),
                d_old: Array1::zeros(dim),
                initialized: false,
                z_epoch: 0,
                prev_alpha: 0.0,
                prev_gd: 0.0,
            },
            Method::Bfgs => Inner::Bfgs {
                h: Array2::<f64>::eye(dim),
                fresh: true,
            },
            Method::Sr1 => Inner::Sr1 {
                h: Array2::<f64>::eye(dim),
                fresh: true,
            },
            Method::Sr2 => Inner::Sr2 {
                b: Array2::<f64>::eye(dim),
                fresh: true,
            },
            Method::Adam { beta1, beta2, eps } => Inner::Adam {
                m: Array1::zeros(dim),
                v: Array1::zeros(dim),
                b1p: *beta1,
                b2p: *beta2,
                beta1: *beta1,
                beta2: *beta2,
                eps: *eps,
            },
            Method::Steepest => Inner::Steepest,
            Method::Pso {
                n_particles,
                inertia,
                c1,
                c2,
            } => Inner::Pso {
                n_particles: (*n_particles).max(1),
                inertia: *inertia,
                c1: *c1,
                c2: *c2,
                swarm: None,
            },
            Method::Newton { kind } => Inner::Newton { kind: *kind },
            Method::Fire { kind } => Inner::Fire(FireState::new(*kind, dim, istep)),
            // The overlay on Solver carries the real state. Inner stays a
            // variant the constraint stepper matches.
            Method::QuickMin | Method::Ode12r | Method::Diis { .. } => Inner::Steepest,
            Method::Bb => Inner::Bb {
                prev_s: None,
                prev_y: None,
            },
            Method::Dogleg => Inner::Dogleg {
                radius: istep.max(RADIUS_START),
            },
            Method::NewtonCg => Inner::NewtonCg {
                radius: istep.max(RADIUS_START),
            },
            Method::NewtonPcg => Inner::NewtonPcg {
                radius: istep.max(RADIUS_START),
            },
        }
    }
}

impl Solver {
    /// Set the gradient tolerance of the session and its internal solver.
    pub fn with_gtol(mut self, gtol: f64) -> Self {
        self.control.gtol = gtol;
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.gtol = gtol;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::FireKind;
    use crate::oracle::Oracle;
    use ndarray::{ArrayView1, array};

    /// A steep well under a tiny Euclidean cap: every FIRE step is
    /// clamped, and the velocity has to describe the clamped move.
    fn steep() -> Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
        Oracle::unbounded(2, |x: ArrayView1<f64>| {
            (
                500.0 * x.iter().map(|v| v * v).sum::<f64>(),
                x.mapv(|v| 1000.0 * v),
            )
        })
    }

    fn fire_solver(kind: FireKind) -> Solver {
        Solver::new(
            Method::Fire { kind },
            Control {
                maxiter: 10,
                gtol: 1e-8,
                istep: 0.1,
                maxmove: Some(0.01),
                ftol_rel: None,
                noise_floor: None,
                error_on_unmet: false,
                backward_polish: false,
                stop_norm: StopNorm::Two,
            },
            2,
        )
    }

    fn fire_state(solver: &Solver) -> &FireState {
        match &solver.inner {
            Inner::Fire(state) => state,
            _ => unreachable!("FIRE session"),
        }
    }

    fn fire_state_mut(solver: &mut Solver) -> &mut FireState {
        match &mut solver.inner {
            Inner::Fire(state) => state,
            _ => unreachable!("FIRE session"),
        }
    }

    /// `f = |x|^2 / 2` whose oracle flips the sign of the gradient's
    /// first component when `flip` is set: an objective that changes
    /// between steps, like a min-mode effective gradient after `tau`
    /// moves.
    struct Flipping {
        flip: std::sync::atomic::AtomicBool,
        calls: std::sync::atomic::AtomicUsize,
        bounds: eindir_core::Bounds<f64>,
    }

    impl Objective<f64> for Flipping {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &eindir_core::Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.value_and_gradient(x).0
        }
    }

    impl eindir_core::Gradient<f64> for Flipping {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            self.value_and_gradient(x).1
        }
    }

    impl DifferentiableObjective<f64> for Flipping {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            use std::sync::atomic::Ordering;
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut g = x.to_owned();
            if self.flip.load(Ordering::Relaxed) {
                g[0] = -g[0];
            }
            (0.5 * x.dot(&x), g)
        }
    }

    #[test]
    fn forget_evaluation_re_evaluates_and_keeps_memory() {
        use std::sync::atomic::Ordering;
        let obj = Flipping {
            flip: false.into(),
            calls: 0.into(),
            bounds: eindir_core::Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0),
        };
        for method in [
            Method::Lbfgs { memory: 5 },
            Method::Fire { kind: FireKind::V2 },
        ] {
            let mut solver = Solver::new(
                method.clone(),
                Control {
                    maxiter: 10,
                    gtol: 1e-12,
                    istep: 0.1,
                    maxmove: Some(0.05),
                    ftol_rel: None,
                    noise_floor: None,
                    error_on_unmet: false,
                    backward_polish: false,
                    stop_norm: StopNorm::Two,
                },
                2,
            );
            solver.set_accept(Accept::Step);
            obj.flip.store(false, Ordering::Relaxed);
            let mut x = array![1.0, 0.5];
            solver.step(&obj, &mut x).unwrap();
            solver.step(&obj, &mut x).unwrap();
            let memory = |s: &Solver| match &s.inner {
                Inner::Lbfgs(l) => l.len() as f64,
                Inner::Fire(f) => l2(&f.vel),
                _ => unreachable!(),
            };
            let kept = memory(&solver);
            assert!(kept > 0.0, "{method:?} built no memory");
            // The oracle changes; the cached gradient at x is stale.
            obj.flip.store(true, Ordering::Relaxed);
            solver.forget_evaluation();
            assert_eq!(memory(&solver), kept, "{method:?} lost its memory");
            let before = obj.calls.load(Ordering::Relaxed);
            let start = x.clone();
            solver.step(&obj, &mut x).unwrap();
            // One call at the unchanged x (the re-evaluation) and one at
            // the step's trial under Accept::Step. FIRE 2.0 adds a call
            // when the half-step-back correction is nonzero, because that
            // correction is scored before the forward step.
            let want = if matches!(method, Method::Fire { kind: FireKind::V2 }) {
                3
            } else {
                2
            };
            assert_eq!(
                obj.calls.load(Ordering::Relaxed) - before,
                want,
                "{method:?}"
            );
            // Without forget_evaluation the cached point is reused.
            let before = obj.calls.load(Ordering::Relaxed);
            solver.step(&obj, &mut x).unwrap();
            assert_eq!(obj.calls.load(Ordering::Relaxed) - before, 1, "{method:?}");
            assert!(x != start);
        }
    }

    #[test]
    fn fire_v2_velocity_times_dt_is_the_clamped_move() {
        let obj = steep();
        let mut solver = fire_solver(FireKind::V2);
        solver.start_fire_trace();
        let mut x = array![1.0, 1.0];
        for _ in 0..5 {
            let before = x.clone();
            solver.step(&obj, &mut x).unwrap();
            let dx = &x - &before;
            let moved = l2(&dx);
            assert!((moved - 0.01).abs() <= 1e-12, "clamp did not bind: {moved}");
            // FIRE 2.0 adapts dt, then steps dx = vel * dt; nothing
            // touches vel after the step, so vel * dt is the move taken.
            let state = fire_state(&solver);
            for i in 0..2 {
                assert!(
                    (state.vel[i] * state.dt - dx[i]).abs() <= 1e-12,
                    "vel {} dt {} dx {}",
                    state.vel[i],
                    state.dt,
                    dx[i]
                );
            }
        }
        let trace = solver.take_fire_trace();
        assert_eq!(trace.len(), 5);
        assert_eq!(trace.last().unwrap().clipped_moves, 5);
    }

    #[test]
    fn fire_v2_stall_restores_dt_max() {
        let obj = steep();
        let mut solver = fire_solver(FireKind::V2);
        solver.set_fire_variant(FireVariant::Guenole2020);
        solver
            .set_fire_config(FireConfig::ase().with_dt_max(4.0).with_n_min(0))
            .unwrap();
        solver.set_fire_cap_two_step_overshoot(true).unwrap();
        solver.set_fire_vdfmax(1);
        if let Some(ext) = &mut solver.fire2 {
            ext.steps = ext.n_delay;
        }
        let mut x = array![1.0, 1.0];
        {
            let state = fire_state_mut(&mut solver);
            state.n_pos = 1;
            state.vel = array![1.0, 1.0];
        }
        solver.step(&obj, &mut x).unwrap();
        {
            let state = fire_state_mut(&mut solver);
            state.n_pos = 1;
            state.vel = array![1.0, 1.0];
        }
        let err = solver.step(&obj, &mut x).unwrap_err();
        assert!(matches!(err, Error::FireStalled { .. }));
        assert_eq!(fire_state(&solver).dt_max, 0.1);
        assert_eq!(fire_state(&solver).n_pos, 1);
    }

    #[test]
    fn fire_nan_trial_or_input_is_refused_without_recording_a_clip() {
        let obj = steep();
        let mut solver = fire_solver(FireKind::V2);
        solver.start_fire_trace();
        if let Inner::Fire(state) = &mut solver.inner {
            state.vel = array![f64::NAN, 0.2];
        }
        let mut x = array![0.5, 0.5];
        let err = solver.step(&obj, &mut x).unwrap_err();
        assert!(matches!(err, Error::Oracle { .. }), "{err:?}");
        assert_eq!(x, array![0.5, 0.5]);
        assert!(solver.take_fire_trace().is_empty());

        solver.start_fire_trace();
        let mut nan_start = array![f64::NAN, 0.5];
        let err = solver.step(&obj, &mut nan_start).unwrap_err();
        assert!(matches!(err, Error::Oracle { .. }), "{err:?}");
        assert!(nan_start[0].is_nan());
        assert!(solver.take_fire_trace().is_empty());
    }

    #[test]
    fn fire_v1_velocity_is_bounded_by_the_clamped_move() {
        let obj = steep();
        let mut solver = fire_solver(FireKind::V1);
        let mut x = array![1.0, 1.0];
        for _ in 0..5 {
            let dt = fire_state(&solver).dt;
            let before = x.clone();
            solver.step(&obj, &mut x).unwrap();
            let moved = l2(&(&x - &before));
            assert!((moved - 0.01).abs() <= 1e-12, "clamp did not bind: {moved}");
            // FIRE 1.0 steps dx = vel * dt, then mixes: the mix keeps
            // |vel| at most what it was, and a reset zeroes it. Either way
            // |vel| cannot exceed the clamped move over the dt used.
            let vnorm = l2(&fire_state(&solver).vel);
            assert!(
                vnorm <= moved / dt + 1e-12,
                "vel {vnorm} exceeds {} for a move of {moved} over dt {dt}",
                moved / dt
            );
        }
    }
}

#[cfg(test)]
mod retained_neb_guard_tests {
    use super::*;
    use eindir_core::{Bounds, Gradient};
    use ndarray::{ArrayView1, array};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Scripted {
        grads: Vec<Array1<f64>>,
        n: AtomicUsize,
    }

    impl Objective<f64> for Scripted {
        fn dim(&self) -> usize {
            3
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(array![-10.0, -10.0, -10.0], array![10.0, 10.0, 10.0], 0.0)
            })
        }
        fn eval(&self, _x: ArrayView1<f64>) -> f64 {
            0.0
        }
    }

    impl Gradient<f64> for Scripted {
        fn dim(&self) -> usize {
            3
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            self.value_and_gradient(x).1
        }
    }

    impl DifferentiableObjective<f64> for Scripted {
        fn value_and_gradient(&self, _x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            let i = self.n.load(Ordering::Relaxed).min(self.grads.len() - 1);
            let g = self.grads[i].clone();
            self.n.fetch_add(1, Ordering::Relaxed);
            (0.0, g)
        }
    }

    #[test]
    fn refused_step_keeps_the_previous_lbfgs_pair() {
        let obj = Scripted {
            grads: vec![
                array![-1.0, 0.0, 0.0],
                array![-0.5, 0.0, 0.0],
                array![-1.0, 0.0, 0.0],
            ],
            n: AtomicUsize::new(0),
        };
        let mut x = array![0.0, 0.0, 0.0];
        let mut solver = Solver::new(
            Method::lbfgs(),
            Control {
                maxiter: 2,
                gtol: 0.0,
                istep: 1.0,
                maxmove: None,
                ftol_rel: None,
                noise_floor: None,
                error_on_unmet: false,
                backward_polish: false,
                stop_norm: StopNorm::Two,
            },
            3,
        );
        solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
        solver.set_accept(Accept::Step);
        solver.set_atom_maxmove(0.2);
        solver.step(&obj, &mut x).unwrap();
        solver.step(&obj, &mut x).unwrap();
        let pairs = match &solver.inner {
            Inner::Lbfgs(lbfgs) => lbfgs.len(),
            _ => panic!("expected L-BFGS"),
        };
        assert_eq!(pairs, 1);
        assert!((x[0] - 0.4).abs() < 1e-12, "x0 {}", x[0]);
    }

    fn guard_script() -> Scripted {
        Scripted {
            grads: vec![
                array![-100.0, 0.0, 0.0],
                array![-99.0, 1.0, 0.0],
                array![-98.0, 1.0, 0.0],
            ],
            n: AtomicUsize::new(0),
        }
    }

    fn guard_solver(guards: bool) -> Solver {
        let mut solver = Solver::new(
            Method::lbfgs(),
            Control {
                maxiter: 2,
                gtol: 0.0,
                istep: 1.0,
                maxmove: None,
                ftol_rel: None,
                noise_floor: None,
                error_on_unmet: false,
                backward_polish: false,
                stop_norm: StopNorm::Two,
            },
            3,
        );
        solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
        solver.set_accept(Accept::Step);
        solver.set_atom_maxmove(0.2);
        solver.set_lbfgs_neb_guards(guards);
        solver
    }

    #[test]
    fn distance_reset_second_step_follows_h0_times_the_force() {
        let obj = guard_script();
        let mut x = array![0.0, 0.0, 0.0];
        let mut solver = guard_solver(true);
        solver.step(&obj, &mut x).unwrap();
        solver.step(&obj, &mut x).unwrap();
        assert!((x[0] - 0.39998979774019283).abs() < 1e-9, "x0 {}", x[0]);
        assert!((x[1] - (-0.002020098967072655)).abs() < 1e-9, "x1 {}", x[1]);
        assert!(x[2].abs() < 1e-12, "x2 {}", x[2]);
        let pairs = match &solver.inner {
            Inner::Lbfgs(lbfgs) => lbfgs.len(),
            _ => panic!("expected L-BFGS"),
        };
        assert_eq!(pairs, 1);
    }

    #[test]
    fn without_the_guard_the_two_loop_step_is_kept() {
        let obj = guard_script();
        let mut x = array![0.0, 0.0, 0.0];
        let mut solver = guard_solver(false);
        solver.step(&obj, &mut x).unwrap();
        solver.step(&obj, &mut x).unwrap();
        assert!((x[0] - 0.3896090211663347).abs() < 1e-8, "x0 {}", x[0]);
        assert!((x[1] - (-0.06362718831085057)).abs() < 1e-8, "x1 {}", x[1]);
    }

    /// Ill-conditioned diagonal quadratic. `P` is the Hessian, so
    /// `P^{-1} g` is the Newton step and CG finishes in at most `n` steps.
    #[test]
    fn preconditioned_cg_on_a_diagonal_quadratic_finishes_within_n_steps() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let lam = [1.0e-4, 1.0e-2, 1.0, 1.0e2, 1.0e4, 1.0e6];
        let n = lam.len();
        let calls = AtomicUsize::new(0);
        struct Quad<'a> {
            lam: &'a [f64],
            calls: &'a AtomicUsize,
            bounds: eindir_core::Bounds<f64>,
        }
        impl Objective<f64> for Quad<'_> {
            fn dim(&self) -> usize {
                self.lam.len()
            }
            fn bounds(&self) -> &eindir_core::Bounds<f64> {
                &self.bounds
            }
            fn eval(&self, x: ArrayView1<f64>) -> f64 {
                self.value_and_gradient(x).0
            }
        }
        impl eindir_core::Gradient<f64> for Quad<'_> {
            fn dim(&self) -> usize {
                self.lam.len()
            }
            fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
                self.value_and_gradient(x).1
            }
        }
        impl DifferentiableObjective<f64> for Quad<'_> {
            fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
                self.calls.fetch_add(1, Ordering::Relaxed);
                let mut f = 0.0;
                let mut g = Array1::zeros(self.lam.len());
                for i in 0..self.lam.len() {
                    f += 0.5 * self.lam[i] * x[i] * x[i];
                    g[i] = self.lam[i] * x[i];
                }
                (f, g)
            }
        }
        let obj = Quad {
            lam: &lam,
            calls: &calls,
            bounds: eindir_core::Bounds::new(
                Array1::from_elem(n, -1e6),
                Array1::from_elem(n, 1e6),
                0.0,
            ),
        };
        let mut x = Array1::ones(n);
        let mut solver = Solver::new(
            Method::Nlcg {
                conjugacy: Conjugacy::PolakRibierePlus,
                restart: Restart::njws(),
            },
            Control {
                maxiter: 1,
                gtol: 1e-10,
                istep: 0.01,
                maxmove: None,
                ftol_rel: None,
                noise_floor: None,
                error_on_unmet: false,
                backward_polish: false,
                stop_norm: StopNorm::Two,
            },
            n,
        );
        // P^{-1} g is a Newton step longer than the default per-atom cap.
        // Finite termination needs that step.
        solver.set_atom_maxmove(0.0);
        solver.set_linesearch(LineSearch::Wolfe {
            c1: 1e-4,
            c2: 0.1,
            maxiter: 20,
        });
        solver.precon = Some(Slot::Exp(crate::precon::ExpState::holding_diagonal(
            &lam,
            x.view(),
        )));
        solver.metric_epoch = 1;
        let mut steps = 0usize;
        let mut used = 0usize;
        let mut gnorm = f64::INFINITY;
        for _ in 0..n + 2 {
            let before = calls.load(Ordering::Relaxed);
            let rep = solver.step(&obj, &mut x).unwrap();
            used += calls.load(Ordering::Relaxed) - before;
            steps += 1;
            gnorm = rep.grad_norm;
            if gnorm < 1e-8 {
                break;
            }
        }
        assert!(gnorm < 1e-8, "not converged, |g| {gnorm}, x {x:?}");
        assert!(steps <= n, "steps {steps} exceed n {n}");
        assert!(
            steps > 0 && used <= steps * 2,
            "calls {used} over {steps} steps"
        );
    }
}

#[cfg(test)]
mod fallible_newton_cg_tests {
    use super::*;
    use eindir_core::{Bounds, Gradient};
    use ndarray::{ArrayView1, array};
    use std::cell::Cell;
    use std::fmt;

    const DIM: usize = 4;

    struct Valleys {
        bounds: Bounds<f64>,
    }

    impl Valleys {
        fn wide() -> Self {
            Self {
                bounds: Bounds::new(
                    Array1::from_elem(DIM, -1024.0),
                    Array1::from_elem(DIM, 1024.0),
                    0.0,
                ),
            }
        }
    }

    impl Objective<f64> for Valleys {
        fn dim(&self) -> usize {
            DIM
        }

        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }

        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            (0..DIM / 2)
                .map(|k| {
                    let (a, b) = (x[2 * k], x[2 * k + 1]);
                    (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
                })
                .sum()
        }
    }

    impl Gradient<f64> for Valleys {
        fn dim(&self) -> usize {
            DIM
        }

        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let mut g = Array1::zeros(DIM);
            for k in 0..DIM / 2 {
                let (a, b) = (x[2 * k], x[2 * k + 1]);
                g[2 * k] = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
                g[2 * k + 1] = 200.0 * (b - a * a);
            }
            g
        }
    }

    impl DifferentiableObjective<f64> for Valleys {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    impl HessianVector for Valleys {
        fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
            let mut hv = Array1::zeros(DIM);
            for k in 0..DIM / 2 {
                let (a, b) = (x[2 * k], x[2 * k + 1]);
                let haa = 2.0 - 400.0 * (b - a * a) + 800.0 * a * a;
                let hab = -400.0 * a;
                hv[2 * k] = haa * v[2 * k] + hab * v[2 * k + 1];
                hv[2 * k + 1] = hab * v[2 * k] + 200.0 * v[2 * k + 1];
            }
            hv
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct HostFailure(usize);

    impl fmt::Display for HostFailure {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "host call {} failed", self.0)
        }
    }

    impl std::error::Error for HostFailure {}

    struct FallibleValleys {
        calls: Cell<usize>,
        fail_at: Option<usize>,
        bounds: Bounds<f64>,
        allow_hvp: bool,
    }

    impl FallibleValleys {
        fn analytic(fail_at: Option<usize>, bounds: Bounds<f64>) -> Self {
            Self {
                calls: Cell::new(0),
                fail_at,
                bounds,
                allow_hvp: true,
            }
        }

        fn gradient_only(fail_at: Option<usize>, bounds: Bounds<f64>) -> Self {
            Self {
                calls: Cell::new(0),
                fail_at,
                bounds,
                allow_hvp: false,
            }
        }

        fn tick(&self) -> std::result::Result<(), HostFailure> {
            let n = self.calls.get() + 1;
            self.calls.set(n);
            if self.fail_at == Some(n) {
                return Err(HostFailure(n));
            }
            Ok(())
        }
    }

    impl TryHessianVector for FallibleValleys {
        type Error = HostFailure;

        fn dim(&self) -> usize {
            DIM
        }

        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }

        fn try_value_and_gradient(
            &self,
            x: ArrayView1<f64>,
        ) -> std::result::Result<(f64, Array1<f64>), HostFailure> {
            self.tick()?;
            Ok(Valleys::wide().value_and_gradient(x))
        }

        fn try_hessian_vector(
            &self,
            x: ArrayView1<f64>,
            v: ArrayView1<f64>,
        ) -> std::result::Result<Array1<f64>, HostFailure> {
            assert!(
                self.allow_hvp,
                "forward-difference path must not call try_hessian_vector"
            );
            self.tick()?;
            Ok(Valleys::wide().hessian_vector(x, v))
        }
    }

    fn vec_bits(x: &Array1<f64>) -> Vec<u64> {
        x.iter().map(|v| v.to_bits()).collect()
    }

    #[derive(Debug, PartialEq, Eq)]
    struct SessionSnapshot {
        x: Vec<u64>,
        steps: usize,
        gradient_calls: usize,
        hvp_calls: usize,
        last_pos: Option<Vec<u64>>,
        last_value: u64,
        last_grad: Vec<u64>,
        bound_low: Option<Vec<u64>>,
        bound_high: Option<Vec<u64>>,
        radius: u64,
        stop_norm: StopNorm,
        rigid_rank: usize,
    }

    impl SessionSnapshot {
        fn capture(solver: &Solver, x: &Array1<f64>) -> Self {
            let radius = match &solver.inner {
                Inner::NewtonCg { radius } | Inner::NewtonPcg { radius } => radius.to_bits(),
                _ => 0,
            };
            Self {
                x: vec_bits(x),
                steps: solver.steps,
                gradient_calls: solver.gradient_calls,
                hvp_calls: solver.hvp_calls,
                last_pos: solver.last_pos.as_ref().map(vec_bits),
                last_value: solver.last_value.to_bits(),
                last_grad: vec_bits(&solver.last_grad),
                bound_low: solver.bound_low.as_ref().map(vec_bits),
                bound_high: solver.bound_high.as_ref().map(vec_bits),
                radius,
                stop_norm: solver.stop_norm,
                rigid_rank: solver.rigid_ws.rank(),
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct ReportSnapshot {
        value: u64,
        coords: Vec<u64>,
        steps: usize,
        grad_norm: u64,
        stop_norm: StopNorm,
        grad: Option<Vec<u64>>,
        status: Status,
    }

    impl ReportSnapshot {
        fn of(report: &Report) -> Self {
            Self {
                value: report.value.to_bits(),
                coords: vec_bits(&report.coords),
                steps: report.steps,
                grad_norm: report.grad_norm.to_bits(),
                stop_norm: report.stop_norm,
                grad: report.grad.as_ref().map(vec_bits),
                status: report.status,
            }
        }
    }

    fn wide_box() -> Bounds<f64> {
        Bounds::new(
            Array1::from_elem(DIM, -1024.0),
            Array1::from_elem(DIM, 1024.0),
            0.0,
        )
    }

    fn unit_box() -> Bounds<f64> {
        Bounds::new(
            Array1::from_elem(DIM, -1.0),
            Array1::from_elem(DIM, 1.0),
            0.0,
        )
    }

    fn configured_solver(method: Method) -> Solver {
        let mut solver = Solver::new(method, Control::default(), DIM);
        solver.set_highs(true);
        solver.set_box(Some(vec![-1024.0; DIM]), Some(vec![1024.0; DIM]));
        solver
    }

    fn expect_host_call(err: &Error) -> usize {
        let Error::Host(inner) = err else {
            unreachable!("expected Error::Host, got {err:?}");
        };
        let Some(failure) = inner.downcast_ref::<HostFailure>() else {
            unreachable!("expected HostFailure, got {inner:?}");
        };
        failure.0
    }

    #[test]
    fn try_step_hvp_host_error_preserves_every_session_field() {
        let outside = array![-1.2, 1.4, 0.3, -1.7];
        for method in [Method::NewtonCg, Method::NewtonPcg] {
            let mut probe = configured_solver(method.clone());
            let probe_host = FallibleValleys::analytic(None, unit_box());
            let mut x_probe = outside.clone();
            let ref_first = probe
                .try_step_hvp(&probe_host, &mut x_probe)
                .expect("reference step");
            let first_calls = probe_host.calls.get();
            assert!(first_calls >= 3);

            for fail_at in 1..=first_calls {
                let mut solver = configured_solver(method.clone());
                let mut x = outside.clone();
                let before = SessionSnapshot::capture(&solver, &x);
                let bad = FallibleValleys::analytic(Some(fail_at), unit_box());
                let err = solver
                    .try_step_hvp(&bad, &mut x)
                    .expect_err("host failed on cold step");
                assert_eq!(expect_host_call(&err), fail_at);
                assert_eq!(bad.calls.get(), fail_at);
                assert_eq!(SessionSnapshot::capture(&solver, &x), before);

                let good = FallibleValleys::analytic(None, unit_box());
                let retry = solver
                    .try_step_hvp(&good, &mut x)
                    .expect("retry after cold failure");
                assert_eq!(ReportSnapshot::of(&retry), ReportSnapshot::of(&ref_first));
                assert_eq!(
                    SessionSnapshot::capture(&solver, &x),
                    SessionSnapshot::capture(&probe, &x_probe)
                );
            }

            // Warm session: step 1 succeeds on wide_box, step 2 fails on unit_box
            // from an unclipped point, then retries at step 1's point on wide_box.
            let mut fresh = configured_solver(method.clone());
            let mut x_fresh = array![-1.2, 1.0, 0.3, -0.7];
            fresh
                .try_step_hvp(&FallibleValleys::analytic(None, wide_box()), &mut x_fresh)
                .expect("warm fresh step 1");
            let warm_count_host = FallibleValleys::analytic(None, wide_box());
            let ref_second = fresh
                .try_step_hvp(&warm_count_host, &mut x_fresh)
                .expect("warm fresh step 2");

            let mut moved_probe = configured_solver(method.clone());
            let mut x_moved_probe = array![-1.2, 1.0, 0.3, -0.7];
            moved_probe
                .try_step_hvp(
                    &FallibleValleys::analytic(None, wide_box()),
                    &mut x_moved_probe,
                )
                .expect("moved probe step 1");
            let moved_counter = FallibleValleys::analytic(None, unit_box());
            let mut x_moved = outside.clone();
            moved_probe
                .try_step_hvp(&moved_counter, &mut x_moved)
                .expect("moved probe step 2");
            let warm_calls = moved_counter.calls.get();

            for fail_at in 1..=warm_calls {
                let mut solver = configured_solver(method.clone());
                let mut x1 = array![-1.2, 1.0, 0.3, -0.7];
                solver
                    .try_step_hvp(&FallibleValleys::analytic(None, wide_box()), &mut x1)
                    .expect("warm step 1");
                let mut x_try = outside.clone();
                let before = SessionSnapshot::capture(&solver, &x_try);
                let bad = FallibleValleys::analytic(Some(fail_at), unit_box());
                let err = solver
                    .try_step_hvp(&bad, &mut x_try)
                    .expect_err("host failed on warm step");
                assert_eq!(expect_host_call(&err), fail_at);
                assert_eq!(bad.calls.get(), fail_at);
                assert_eq!(SessionSnapshot::capture(&solver, &x_try), before);

                let good = FallibleValleys::analytic(None, wide_box());
                let retry = solver
                    .try_step_hvp(&good, &mut x1)
                    .expect("warm retry at cached point");
                assert_eq!(good.calls.get(), warm_count_host.calls.get());
                assert_eq!(ReportSnapshot::of(&retry), ReportSnapshot::of(&ref_second));
                assert_eq!(
                    SessionSnapshot::capture(&solver, &x1),
                    SessionSnapshot::capture(&fresh, &x_fresh)
                );
            }
        }
    }

    #[test]
    fn forward_difference_hvp_host_error_preserves_session_and_retries_cleanly() {
        let start = array![-1.2, 1.0, 0.3, -0.7];
        let outside = array![-1.2, 1.4, 0.3, -1.7];
        let valleys = Valleys::wide();
        for method in [Method::NewtonCg, Method::NewtonPcg] {
            // Succeeding callbacks match Solver::step bit for bit.
            let mut plain = configured_solver(method.clone());
            let mut fallible = configured_solver(method.clone());
            let mut x_plain = start.clone();
            let mut x_fallible = start.clone();
            let ok_host = FallibleValleys::gradient_only(None, wide_box());
            for _ in 0..8 {
                let rep_plain = plain.step(&valleys, &mut x_plain).expect("plain step");
                let rep_fallible = fallible
                    .step_newton_cg(&HostOracle(&ok_host), &mut x_fallible, &FiniteDifference)
                    .expect("fallible fd step");
                assert_eq!(
                    ReportSnapshot::of(&rep_plain),
                    ReportSnapshot::of(&rep_fallible)
                );
                assert_eq!(
                    SessionSnapshot::capture(&plain, &x_plain),
                    SessionSnapshot::capture(&fallible, &x_fallible)
                );
            }
            assert_eq!(
                ok_host.calls.get(),
                fallible.gradient_calls() + fallible.hvp_calls()
            );

            // Cold failure on every call (opening gradient, forward-difference
            // Hessian-vector products in sketch and CG, and trial evaluation).
            let mut ref_solver = configured_solver(method.clone());
            let ref_host = FallibleValleys::gradient_only(None, unit_box());
            let mut x_ref = outside.clone();
            let ref_report = ref_solver
                .step_newton_cg(&HostOracle(&ref_host), &mut x_ref, &FiniteDifference)
                .expect("reference fd step");
            let per_step = ref_host.calls.get();
            assert!(per_step >= 3);

            for fail_at in 1..=per_step {
                let mut solver = configured_solver(method.clone());
                let mut x = outside.clone();
                let before = SessionSnapshot::capture(&solver, &x);
                let bad = FallibleValleys::gradient_only(Some(fail_at), unit_box());
                let err = solver
                    .step_newton_cg(&HostOracle(&bad), &mut x, &FiniteDifference)
                    .expect_err("fd host failed");
                assert_eq!(expect_host_call(&err), fail_at);
                assert_eq!(bad.calls.get(), fail_at);
                assert_eq!(SessionSnapshot::capture(&solver, &x), before);

                let good = FallibleValleys::gradient_only(None, unit_box());
                let retry = solver
                    .step_newton_cg(&HostOracle(&good), &mut x, &FiniteDifference)
                    .expect("fd retry");
                assert_eq!(ReportSnapshot::of(&retry), ReportSnapshot::of(&ref_report));
                assert_eq!(
                    SessionSnapshot::capture(&solver, &x),
                    SessionSnapshot::capture(&ref_solver, &x_ref)
                );
            }

            // Warm failure on the forward-difference product path (cached opening
            // gradient, so call 1 is already a forward-difference HVP probe).
            let mut fresh = configured_solver(method.clone());
            let mut x_fresh = start.clone();
            fresh
                .step_newton_cg(
                    &HostOracle(&FallibleValleys::gradient_only(None, wide_box())),
                    &mut x_fresh,
                    &FiniteDifference,
                )
                .expect("warm fd step 1");
            let warm_ref_host = FallibleValleys::gradient_only(None, wide_box());
            let ref_warm = fresh
                .step_newton_cg(&HostOracle(&warm_ref_host), &mut x_fresh, &FiniteDifference)
                .expect("warm fd step 2");
            let warm_calls = warm_ref_host.calls.get();
            assert!(warm_calls >= 2);

            for fail_at in 1..=warm_calls {
                let mut solver = configured_solver(method.clone());
                let mut x = start.clone();
                solver
                    .step_newton_cg(
                        &HostOracle(&FallibleValleys::gradient_only(None, wide_box())),
                        &mut x,
                        &FiniteDifference,
                    )
                    .expect("warm step 1");
                let before = SessionSnapshot::capture(&solver, &x);
                let bad = FallibleValleys::gradient_only(Some(fail_at), wide_box());
                let err = solver
                    .step_newton_cg(&HostOracle(&bad), &mut x, &FiniteDifference)
                    .expect_err("warm fd host failed");
                assert_eq!(expect_host_call(&err), fail_at);
                assert_eq!(bad.calls.get(), fail_at);
                assert_eq!(SessionSnapshot::capture(&solver, &x), before);

                let good = FallibleValleys::gradient_only(None, wide_box());
                let retry = solver
                    .step_newton_cg(&HostOracle(&good), &mut x, &FiniteDifference)
                    .expect("warm fd retry");
                assert_eq!(good.calls.get(), warm_calls);
                assert_eq!(ReportSnapshot::of(&retry), ReportSnapshot::of(&ref_warm));
                assert_eq!(
                    SessionSnapshot::capture(&solver, &x),
                    SessionSnapshot::capture(&fresh, &x_fresh)
                );
            }
        }
    }
}
