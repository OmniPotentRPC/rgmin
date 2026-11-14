//! Persistent solver session. One [`Solver::step`] is one outer iteration.

mod constrained;
pub(crate) mod drive;

use std::collections::VecDeque;
use std::sync::Arc;

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[cfg(feature = "highs")]
use crate::accept::accept_step_with_fallback;
use crate::accept::{Accept, Acceptance, Trial, accept_step, retract_within_cap, single_trial};
use crate::adam::adam_direction;
use crate::bb::bb_direction;
use crate::control::{Control, StopNorm};
use crate::error::{Error, Result};
use crate::fire::{
    FireConfig, FireKind, FireMix, FirePreparation, FirePreset, FireState, Prepared, fire_after_v1,
    fire_after_v1_metric, fire_clip_components, fire_clip_velocity, fire_displacement,
    fire_displacement_into, fire_displacement_metric, fire_prepare_v2, fire_prepare_v2_into,
    project_box_tangent,
};
use crate::holonomic::DistanceConstraints;
use crate::hvp::{
    FiniteDifference, HessianVector, NewtonCgRequest, OptionalAction, newton_cg_step,
    sym_eig_jacobi,
};
use crate::lbfgs::{GradNorm, Lbfgs, LbfgsStep};
use crate::lindh::LindhSpec;
use crate::linesearch::LineSearch;
use crate::lowest_mode::{ApplyHessian, lowest_eigenpair};
use crate::manifold::{Manifold, ManifoldKind};
use crate::method::Method;
use crate::newton::{HessianObjective, NewtonKind, rfo_direction, shifted_newton};
use crate::nlcg::{Conjugacy, ConjugacyContext, Restart};
use crate::precon::{ExpParams, ExpState, Neighbours};
use crate::pso::{Particle, RNG_SEED, random_velocity, update_swarm};
use crate::qn::{
    bfgs_inverse_update, secant_scale, solve_dense, sr1_inverse_update, sr2_hessian_update,
};
use crate::qn_step::QnStep;
use crate::report::Report;
use crate::rigid::{project_horizontal, project_out_rot_trans};
use crate::step::{
    cap_displacement_atom, clip_mut, clip_owned, finite, l2, max_atom_displacement,
    max_atom_motion, non_finite, scale_displacement_to_atom_cap, scale_step, scale_step_atom,
};
use crate::trust::{
    accept_ratio, burdakov_radius, dogleg_direction, predicted_reduction, reduction_ratio,
    update_radius,
};

/// Projects the rigid modes out of `dir` at `x`. True when more than
/// rounding left the direction.
fn strip_rigid(dir: &mut Array1<f64>, x: &Array1<f64>) -> bool {
    let before = dir.clone();
    project_out_rot_trans(dir, x.view());
    l2(&(&before - &*dir)) > 1e-12 * l2(&before)
}

/// The smallest initial trust radius, taken when `istep` is smaller.
const RADIUS_START: f64 = 1e-8;
/// The largest trust radius, in units of the current one, when no
/// `maxmove` caps it.
const RADIUS_GROWTH: f64 = 8.0;
/// Saddle-escape kicks from one stationary point.
const SADDLE_KICKS: u8 = 4;

/// A length-n vector from the recycle pool, filled with `src`.
fn copied(src: &Array1<f64>) -> Array1<f64> {
    let mut dst = crate::workspace::take_array(src.len());
    if let (Some(d), Some(s)) = (dst.as_slice_mut(), src.as_slice()) {
        d.copy_from_slice(s);
    } else {
        dst.assign(src);
    }
    dst
}

/// Why a capped trial did not move.
enum CappedMiss {
    /// The quasi-Newton trial was non-finite. Try the force once.
    Retry,
    /// Stop and restore the pairs.
    Stop(Error),
}

/// `P^{-1} v` when an exponential metric is installed, otherwise `v`.
fn metric_vector(exp: &Option<ExpState>, v: &Array1<f64>) -> Array1<f64> {
    match exp.as_ref().and_then(ExpState::matrix) {
        Some(p) => p.solve(v.view()).unwrap_or_else(|| v.clone()),
        None => v.clone(),
    }
}

/// P-norms for the FIRE mix: `sqrt(v^T P v)` and `sqrt(F · P^{-1} F)`.
fn fire_mix(
    exp: &Option<ExpState>,
    vel: &Array1<f64>,
    raw: &Array1<f64>,
    steepest: &Array1<f64>,
) -> Option<FireMix> {
    let p = exp.as_ref().and_then(ExpState::matrix)?;
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

/// Scratch for one Euclidean eOn or FIRE step. Sized in [`Solver::new`].
struct StepScratch {
    /// Gradient at the point the step starts from.
    gold: Array1<f64>,
    /// Gradient at the trial.
    grad: Array1<f64>,
    /// Accepted trial coordinates.
    trial: Array1<f64>,
    /// Point the FIRE displacement is measured from.
    fwd: Array1<f64>,
    /// Uncapped FIRE trial, kept to see whether a cap moved it.
    free: Array1<f64>,
    /// FIRE rollback, or a scratch displacement.
    corr: Array1<f64>,
}

impl StepScratch {
    fn new(n: usize) -> Self {
        Self {
            gold: Array1::zeros(n),
            grad: Array1::zeros(n),
            trial: Array1::zeros(n),
            fwd: Array1::zeros(n),
            free: Array1::zeros(n),
            corr: Array1::zeros(n),
        }
    }
}

/// Long-lived solver. Algorithm memory lives here; `x` stays with the caller.
pub struct Solver {
    dim: usize,
    control: Control,
    linesearch: LineSearch,
    istep: f64,
    /// Scale raw gradient steps to unit length at the first step after
    /// construction, `forget`, or `rebase`. The C default.
    unit_first_step: bool,
    steps: usize,
    qn_step: QnStep,
    accept: Accept,
    e_hist: VecDeque<f64>,
    /// The acceptance test refused the last first-order step.
    refused: bool,
    /// Dogleg model on a curved kind, rewritten in place at each step.
    tangent: TangentModel,
    atom_maxmove: Option<f64>,
    /// Whether geometry L-BFGS needs a finite-difference stiffness probe.
    geometry_lbfgs_stiff: Option<bool>,
    project_rigid: bool,
    /// Sella `proj_rot`: false under PBC. Rotation is not a symmetry of the cell.
    periodic: bool,
    manifold: ManifoldKind,
    /// Per-atom masses for [`ManifoldKind::MwRigid`]. Length N, not 3N.
    masses: Option<Array1<f64>>,
    #[cfg(feature = "highs")]
    highs: bool,
    #[cfg(feature = "highs")]
    box_lo: Option<Vec<f64>>,
    #[cfg(feature = "highs")]
    box_hi: Option<Vec<f64>>,
    #[cfg(feature = "highs")]
    highs_trust: Option<f64>,
    #[cfg(feature = "highs")]
    equalities: Vec<(Vec<(usize, f64)>, f64)>,
    #[cfg(feature = "highs")]
    highs_solver: crate::HighsSolverKind,
    #[cfg(feature = "highs")]
    highs_crossover: crate::HighsCrossover,
    #[cfg(feature = "highs")]
    highs_callback: Option<crate::HighsCCallback>,
    #[cfg(feature = "highs")]
    highs_callback_user: *mut std::os::raw::c_void,
    last_pos: Option<Array1<f64>>,
    last_value: f64,
    last_grad: Array1<f64>,
    /// When set, [`Self::report`] skips the coordinate and gradient copies.
    brief: bool,
    /// Optional trace of the eOn fill step, for the cap and reset study.
    eon_log: Option<Vec<crate::session::drive::EonFeat>>,
    /// Ratio test on the capped two-loop step. Off by default.
    trust: bool,
    /// Starting Burdakov radius. Zero leaves that radius off.
    lbfgs_burdakov_init: f64,
    /// Current Burdakov radius.
    lbfgs_burdakov: f64,
    /// Current trust radius, in the same length as the cap.
    trust_delta: f64,
    /// Largest trust radius. The cap selected at construction.
    trust_max: f64,
    /// Reused buffers for the Euclidean FIRE and L-BFGS steps.
    scratch: Array1<f64>,
    scratch_b: Array1<f64>,
    saved_x: Array1<f64>,
    /// Trial, gradient and FIRE buffers for a steady [`Self::step`].
    reuse: StepScratch,
    /// Objective gradients. A finite-difference product is an
    /// [`Self::hvp_calls`] entry, not a gradient.
    gradient_calls: usize,
    /// Packwood exponential metric. `None` leaves every method unchanged.
    exp: Option<ExpState>,
    /// Lindh model Hessian. `None` leaves the scalar initial scale.
    lindh_spec: Option<LindhSpec>,
    /// Rayleigh scale of the Lindh matrix, from one force probe.
    lindh_mu: Option<f64>,
    /// Analytic or finite-difference Hessian-vector products.
    hvp_calls: usize,
    /// Negative-curvature kicks already taken in this solve.
    saddle_kicks: u8,
    stop_norm: StopNorm,
    /// Holonomic distance constraints. With a set, `last_grad` is the
    /// raw gradient, whose multipliers the Hessian steps need.
    constraints: Option<Arc<DistanceConstraints>>,
    inner: Inner,
}

enum Inner {
    Lbfgs(Lbfgs),
    Nlcg {
        conjugacy: Conjugacy,
        restart: Restart,
        dir: Array1<f64>,
        g_old: Array1<f64>,
        /// `P^{-1} g` at the previous accepted point. Equals `g_old` when
        /// no exponential metric is installed.
        z_old: Array1<f64>,
        d_old: Array1<f64>,
        initialized: bool,
    },
    /// `fresh`: no secant pair has reached the matrix since construction,
    /// `forget`, or a refused energy step.
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

impl Solver {
    /// Fresh session for `method` in dimension `dim`.
    pub fn new(method: Method, control: Control, dim: usize) -> Self {
        let istep = control.istep;
        let stop_norm = control.stop_norm;
        let inner = Inner::from_method(&method, dim, istep);
        let linesearch = if matches!(&inner, Inner::Lbfgs(_)) {
            LineSearch::Wolfe {
                c1: 1e-4,
                c2: 0.9,
                maxiter: 40,
            }
        } else {
            LineSearch::default()
        };
        Self {
            dim,
            control,
            linesearch,
            istep,
            unit_first_step: false,
            steps: 0,
            qn_step: QnStep::TwoLoop,
            accept: Accept::None,
            e_hist: VecDeque::new(),
            refused: false,
            tangent: TangentModel::default(),
            atom_maxmove: None,
            geometry_lbfgs_stiff: None,
            project_rigid: false,
            periodic: false,
            manifold: ManifoldKind::Euclidean,
            masses: None,
            #[cfg(feature = "highs")]
            highs: false,
            #[cfg(feature = "highs")]
            box_lo: None,
            #[cfg(feature = "highs")]
            box_hi: None,
            #[cfg(feature = "highs")]
            highs_trust: None,
            #[cfg(feature = "highs")]
            equalities: Vec::new(),
            #[cfg(feature = "highs")]
            highs_solver: crate::HighsSolverKind::Ipm,
            #[cfg(feature = "highs")]
            highs_crossover: crate::HighsCrossover::Off,
            #[cfg(feature = "highs")]
            highs_callback: None,
            #[cfg(feature = "highs")]
            highs_callback_user: std::ptr::null_mut(),
            last_pos: None,
            last_value: 0.0,
            last_grad: Array1::zeros(dim),
            brief: false,
            eon_log: None,
            trust: false,
            lbfgs_burdakov_init: 0.0,
            lbfgs_burdakov: 0.0,
            trust_delta: 0.0,
            trust_max: 0.0,
            scratch: Array1::zeros(dim),
            scratch_b: Array1::zeros(dim),
            saved_x: Array1::zeros(dim),
            reuse: StepScratch::new(dim),
            gradient_calls: 0,
            hvp_calls: 0,
            exp: None,
            lindh_spec: None,
            lindh_mu: None,
            saddle_kicks: 0,
            stop_norm,
            constraints: None,
            inner,
        }
    }

    /// Install the exponential preconditioner at `positions`.
    ///
    /// `positions` has length [`Self::dim`] and is `3 N` coordinates in
    /// C order. [`ExpParams::mu`] `None` defers the energy scale to the
    /// next [`Self::step`], which estimates it from the objective.
    /// L-BFGS then takes \(H_0 = P^{-1}\), including the capped geometry
    /// step selected by [`Self::set_atom_maxmove`]. FIRE and nonlinear CG
    /// take `P` as the metric. A dense pair or Lindh matrix supplied to the
    /// two-loop still wins over this factor.
    pub fn set_exp(
        &mut self,
        positions: ArrayView1<f64>,
        neighbours: &Neighbours,
        params: &ExpParams,
    ) -> Result<()> {
        if positions.len() != self.dim {
            return Err(Error::Dim {
                got: positions.len(),
                dim: self.dim,
            });
        }
        let state = ExpState::new(positions, neighbours, params)?;
        let matrix = state.matrix().cloned();
        self.exp = Some(state);
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.set_h0(matrix);
        }
        Ok(())
    }

    /// Install a neighbour list whose geometry arrives on the next step.
    #[cfg(feature = "capi")]
    pub(crate) fn set_exp_deferred(&mut self, state: ExpState) {
        self.exp = Some(state);
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.set_h0(None);
        }
    }

    /// Drop the exponential preconditioner.
    pub fn clear_exp(&mut self) {
        self.exp = None;
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.set_h0(None);
        }
    }

    /// Energy scale of the installed preconditioner, after estimation.
    #[must_use]
    pub fn exp_mu(&self) -> Option<f64> {
        self.exp.as_ref().and_then(ExpState::mu)
    }

    fn refresh_exp<O>(&mut self, obj: &O, x: &Array1<f64>, grad: &Array1<f64>) -> Result<()>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if self.exp.is_none() {
            return Ok(());
        }
        let grad_fn = |p: ArrayView1<f64>| obj.grad(p);
        if let Some(state) = &mut self.exp {
            state.refresh(x.view(), grad, grad_fn)?;
        }
        let matrix = self.exp.as_ref().and_then(|s| s.matrix().cloned());
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

    /// Step-length rule for an L-BFGS session.
    ///
    /// [`LbfgsStep::Capped`] at [`crate::EON_MAX_MOVE`] is the default:
    /// the eOn step, no line search. The first step probes curvature.
    /// A proposal at least as long as the cap clears the memory and
    /// steps along `H0` times the force. [`LbfgsStep::Scaled`] shortens
    /// the two-loop step and keeps the pairs. [`LbfgsStep::Wolfe`] opts
    /// into the strong Wolfe search.
    /// A non-positive `max_atom` restores [`crate::EON_MAX_MOVE`] on
    /// that same rule. [`Self::set_atom_maxmove`] replaces the cap
    /// while the step is line-search-free. A non-finite `max_atom`, a
    /// session that is not L-BFGS, a non-Euclidean manifold, a
    /// distance-constraint set, or a HiGHS feasible set is an error
    /// when selecting a cap.
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
        match &mut self.inner {
            Inner::Lbfgs(solver) => solver.step = step,
            _ => {
                return Err(Error::LbfgsCap {
                    what: "requires an L-BFGS session",
                });
            }
        }
        Ok(())
    }

    /// Step-length rule stored on an L-BFGS session.
    /// Other methods report [`LbfgsStep::Wolfe`].
    pub fn lbfgs_step(&self) -> LbfgsStep {
        match &self.inner {
            Inner::Lbfgs(solver) => solver.step,
            _ => LbfgsStep::Wolfe,
        }
    }

    /// Under `Accept::Energy` or `Accept::Nonmonotone`, set the step
    /// scale of steepest descent, BFGS, SR1, SR2, BB, and unpreconditioned CG
    /// to `min(istep, 1 / ||g||)` at the first step after construction,
    /// `forget`, or `rebase`. The scale stays until the next `forget` or
    /// `rebase`. SciPy BFGS and L-BFGS-B begin at about unit length.
    #[cfg(feature = "capi")]
    pub(crate) fn set_unit_first_step(&mut self) {
        self.unit_first_step = true;
    }

    /// Norm that every stop test and [`Report::grad_norm`] apply to the
    /// gradient. The session starts with [`Control::stop_norm`].
    pub fn set_stop_norm(&mut self, norm: StopNorm) {
        self.stop_norm = norm;
    }

    /// Norm selected by [`Self::set_stop_norm`].
    pub fn stop_norm(&self) -> StopNorm {
        self.stop_norm
    }

    /// Parameter set of a FIRE 2.0 session. The integrator restarts
    /// from [`Control::istep`]. [`FirePreset::Default`] is what a new
    /// session runs. Any other method returns [`Error::NeedFire2`].
    pub fn set_fire_preset(&mut self, preset: FirePreset) -> Result<()> {
        match &mut self.inner {
            Inner::Fire(state) if state.kind == FireKind::V2 => {
                state.set_preset(preset);
                Ok(())
            }
            _ => Err(Error::NeedFire2),
        }
    }

    /// Integrator parameters of a FIRE or FIRE 2.0 session.
    ///
    /// The velocity is cleared. Any other method returns
    /// [`Error::NeedFire`]. [`FireConfig::ase`] is the ASE `FIRE`
    /// constructor; a new session uses [`FireState::new`], whose
    /// `dt_max` is `2.5` times the start step rather than ASE's `1`.
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

    /// ABC-FIRE on a FIRE 2.0 session, off by default. The per-atom cap,
    /// else [`Control::maxmove`], then bounds each displacement component
    /// in place of the norm cap. Any other method returns
    /// [`Error::NeedFire2`].
    pub fn set_fire_abc(&mut self, on: bool) -> Result<()> {
        match &mut self.inner {
            Inner::Fire(state) if state.kind == FireKind::V2 => {
                state.abc = on;
                Ok(())
            }
            _ => Err(Error::NeedFire2),
        }
    }

    /// Line search stored on the session. First-order `step` uses
    /// [`Accept`]; Riemannian solvers that need a Wolfe/Armijo search
    /// read this.
    pub fn set_linesearch(&mut self, linesearch: LineSearch) {
        self.linesearch = linesearch;
    }

    /// Current line-search policy.
    pub fn linesearch(&self) -> LineSearch {
        self.linesearch
    }

    /// Packed length of `x` this session was created with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Length-n vectors the session is holding for scratch and history.
    pub fn retained_vectors(&self) -> usize {
        let mut n = 0;
        for buf in [
            &self.scratch,
            &self.scratch_b,
            &self.saved_x,
            &self.reuse.gold,
            &self.reuse.grad,
            &self.reuse.trial,
            &self.reuse.fwd,
            &self.reuse.free,
            &self.reuse.corr,
        ] {
            if !buf.is_empty() {
                n += 1;
            }
        }
        if let Inner::Lbfgs(solver) = &self.inner {
            n += solver.vector_slots();
        }
        n
    }

    /// Outer steps completed on this session.
    pub fn steps(&self) -> usize {
        self.steps
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
    pub fn set_maxmove(&mut self, maxmove: f64) {
        self.control.maxmove = if maxmove > 0.0 { Some(maxmove) } else { None };
        #[cfg(feature = "highs")]
        self.apply_highs_opts();
    }

    /// eOn `maxAtomMotionAppliedV`. Preferred over the Euclidean cap.
    /// While the L-BFGS step is the default cap, this replaces that cap.
    /// After [`Self::set_lbfgs_step`]`(Wolfe)`, and without a distance
    /// constraint set, Euclidean L-BFGS takes the geometry step and
    /// ignores [`Self::set_accept`]. With a set it keeps the two-loop
    /// step under the acceptance mode, and the cap scales each trial
    /// before it returns to the set.
    pub fn set_atom_maxmove(&mut self, maxmove: f64) {
        self.atom_maxmove = if maxmove > 0.0 { Some(maxmove) } else { None };
        if let Inner::Lbfgs(solver) = &mut self.inner {
            let cap = self.atom_maxmove.unwrap_or(crate::lbfgs::EON_MAX_MOVE);
            match &mut solver.step {
                LbfgsStep::Capped { max_atom } | LbfgsStep::Scaled { max_atom } => {
                    *max_atom = cap;
                }
                LbfgsStep::Wolfe => {}
            }
        }
        #[cfg(feature = "highs")]
        self.apply_highs_opts();
    }

    /// ASE's fixed inverse-Hessian scale. `None` keeps the Nocedal-Wright
    /// scale. A non-positive value clears it. Only an L-BFGS session.
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
    /// The default is the longest per-atom displacement. Only an
    /// L-BFGS session.
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

    /// Trust-region radius on the capped two-loop step.
    ///
    /// A long direction is scaled to the radius and the pairs are kept.
    /// The radius starts at the cap. A ratio below `0.1` rejects the
    /// trial and shrinks the radius. A ratio above `0.75` on a step
    /// that meets the radius grows it, up to the cap. Off by default.
    /// The predicted reduction is the Newton model `−½ g·s`
    /// (Nocedal and Wright, algorithm 4.1,
    /// <https://doi.org/10.1007/978-0-387-40065-5>).
    /// This is the radius update, not the compact subproblem solve of
    /// Burdakov, Gong, Yuan and Zikrin
    /// (<https://doi.org/10.1007/s12532-016-0109-7>)
    /// or Brust, Erway and Marcia
    /// (<https://doi.org/10.1007/s10589-016-9868-3>).
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

    /// Keep a long downhill quasi-Newton step and scale it per atom.
    ///
    /// The eOn cap replaces that step with `H0` times the force and
    /// clears the pairs. This clears the pairs too, but the step stays
    /// the model direction, so the Euclidean length may exceed the cap
    /// while no atom moves further than the cap. That is the L-BFGS
    /// form of the per-atom-only CG step. Off by default.
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

    /// Compact Byrd–Nocedal–Schnabel step inside the per-atom cap.
    ///
    /// A two-loop step inside the cap is unchanged. A longer step is
    /// the trust-region solution `(B + λ I) s = −g` with `λ` chosen so
    /// the longest atom sits on the cap. The pairs are kept. Off by
    /// default. The radius update in [`Self::set_lbfgs_trust`] is a
    /// different rule.
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

    /// Skip a curvature pair with `s·y ≤ 0` instead of clearing memory
    /// on the next step. Nocedal and Wright skip that update. The eOn
    /// client stores it and resets. Off by default.
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

    /// Lindh model Hessian as the initial inverse. `None` removes it.
    ///
    /// The factor is rebuilt from the current geometry. The first call
    /// spends one gradient to set the scale `μ`. A non-positive ridge,
    /// a session that is not L-BFGS, or atomic numbers whose length is
    /// neither 0 nor `N` is an error.
    pub fn set_lindh(&mut self, spec: Option<LindhSpec>) -> Result<()> {
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

    /// Rebuild the Lindh factor. One gradient probe the first time.
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
        let fd = crate::EON_FINITE_DIFFERENCE;
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

    /// eOn `lbfgs_project_rigid`. Isolated clusters only.
    pub fn set_project_rigid(&mut self, enabled: bool) {
        self.project_rigid = enabled;
        #[cfg(feature = "highs")]
        self.apply_highs_opts();
    }

    /// Periodic cell. Sella leaves `proj_rot` off; the quotient is \(R^{3N}/T(3)\).
    pub fn set_periodic(&mut self, periodic: bool) {
        self.periodic = periodic;
    }

    /// Keep every pair of `set` at its length on every later step, with
    /// the set's masses as the metric of every method. The memory of the
    /// method restarts. A step under a setting the set cannot honour
    /// returns [`crate::ConstraintError::Unsupported`]: a manifold other
    /// than Euclidean coordinates, the exponential preconditioner, the
    /// HiGHS step, a periodic cell, particle swarm, or rigid projection
    /// at an active coordinate bound. A trial that takes a constrained
    /// atom to its coordinate bound is [`crate::ConstraintError::Bound`].
    pub fn set_distance_constraints(&mut self, set: DistanceConstraints) -> Result<()> {
        self.refuse_constraint_settings(&set)?;
        self.constraints = Some(Arc::new(set));
        self.forget();
        Ok(())
    }

    /// Drop the constraint set. The memory of the method restarts.
    pub fn clear_distance_constraints(&mut self) {
        if self.constraints.take().is_some() {
            self.forget();
        }
    }

    /// The constraint set, if any.
    pub fn distance_constraints(&self) -> Option<&DistanceConstraints> {
        self.constraints.as_deref()
    }

    /// Embedded manifold for project / retract / transport.
    pub fn set_manifold(&mut self, kind: ManifoldKind) {
        if kind != self.manifold {
            self.forget();
            self.tangent = TangentModel::default();
        }
        self.manifold = kind;
    }

    /// Oblique \(\mathrm{OB}(n,m)\): product of `m` unit spheres in `R^n`.
    /// Packed column-major, length `n*m`.
    pub fn set_oblique(&mut self, n: usize, m: usize) {
        self.set_manifold(ManifoldKind::Oblique { n, m });
    }

    /// Stiefel \(\mathrm{St}(n,p)\). `p = 1` is the sphere packing.
    /// `p > 1` is column-major, length `n*p`.
    pub fn set_stiefel(&mut self, n: usize, p: usize) {
        self.set_manifold(ManifoldKind::stiefel(n, p));
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

    /// Per-atom masses for [`ManifoldKind::MwRigid`] (Page–McIver / Eckart).
    /// Empty clears them (unit mass).
    pub fn set_masses(&mut self, masses: Array1<f64>) {
        if masses.is_empty() {
            self.masses = None;
        } else {
            self.masses = Some(masses);
        }
    }

    /// Al-Baali extra-updates on the newest L-BFGS pair.
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

    /// HiGHS feasible-set step. No-op unless this build has `highs`.
    pub fn set_highs(&mut self, enabled: bool) {
        #[cfg(not(feature = "highs"))]
        let _ = enabled;
        #[cfg(feature = "highs")]
        {
            self.highs = enabled;
            self.apply_highs_opts();
        }
    }

    /// Per-coordinate box on `x + p`. `None` on a side is unbounded.
    /// Returns `true` when this build has `highs`.
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
            self.apply_highs_opts();
            true
        }
    }

    /// L_inf trust radius on the HiGHS step. Non-positive clears it.
    /// Returns `true` when this build has `highs`.
    pub fn set_trust(&mut self, radius: f64) -> bool {
        #[cfg(not(feature = "highs"))]
        {
            let _ = radius;
            false
        }
        #[cfg(feature = "highs")]
        {
            self.highs_trust = if radius > 0.0 { Some(radius) } else { None };
            self.apply_highs_opts();
            true
        }
    }

    /// Append one linear equality `a · p = rhs`. Returns `true` with `highs`.
    pub fn add_equality(&mut self, coeffs: Vec<(usize, f64)>, rhs: f64) -> bool {
        #[cfg(not(feature = "highs"))]
        {
            let _ = (coeffs, rhs);
            false
        }
        #[cfg(feature = "highs")]
        {
            self.equalities.push((coeffs, rhs));
            self.apply_highs_opts();
            true
        }
    }

    /// Drop every stored HiGHS equality. Returns `true` with `highs`.
    pub fn clear_equalities(&mut self) -> bool {
        #[cfg(not(feature = "highs"))]
        {
            false
        }
        #[cfg(feature = "highs")]
        {
            self.equalities.clear();
            self.apply_highs_opts();
            true
        }
    }

    /// HiGHS `solver` token. Returns `true` with `highs`.
    pub fn set_highs_solver(&mut self, kind: crate::HighsSolverKind) -> bool {
        #[cfg(not(feature = "highs"))]
        {
            let _ = kind;
            false
        }
        #[cfg(feature = "highs")]
        {
            self.highs_solver = kind;
            self.apply_highs_opts();
            true
        }
    }

    /// HiGHS user callback for logging and IPM interrupt. Returns
    /// `true` with `highs`.
    pub fn set_highs_callback(
        &mut self,
        cb: Option<crate::HighsCCallback>,
        user: *mut std::os::raw::c_void,
    ) -> bool {
        #[cfg(not(feature = "highs"))]
        {
            let _ = (cb, user);
            false
        }
        #[cfg(feature = "highs")]
        {
            self.highs_callback = cb;
            self.highs_callback_user = user;
            self.apply_highs_opts();
            true
        }
    }

    /// HiGHS `run_crossover`. Returns `true` with `highs`.
    pub fn set_highs_crossover(&mut self, kind: crate::HighsCrossover) -> bool {
        #[cfg(not(feature = "highs"))]
        {
            let _ = kind;
            false
        }
        #[cfg(feature = "highs")]
        {
            self.highs_crossover = kind;
            self.apply_highs_opts();
            true
        }
    }

    #[cfg(feature = "highs")]
    fn apply_highs_opts(&mut self) {
        if !self.highs {
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.highs = None;
            }
            return;
        }
        let step = crate::HighsStep {
            trust: self
                .highs_trust
                .or(self.atom_maxmove)
                .or(self.control.maxmove),
            lo: self.box_lo.clone(),
            hi: self.box_hi.clone(),
            equalities: self.equalities.clone(),
            center_axes: if self.project_rigid && self.dim.is_multiple_of(3) {
                Some((self.dim / 3, 3))
            } else {
                None
            },
            solver: self.highs_solver,
            crossover: self.highs_crossover,
            callback: self.highs_callback,
            callback_user: self.highs_callback_user,
        };
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.highs = Some(step);
        }
    }

    fn check_manifold(&self, n: usize) -> Result<()> {
        if self.manifold.required_dim(n).is_ok() {
            return Ok(());
        }
        Err(Error::ManifoldDim {
            kind: self.manifold.as_str(),
            got: n,
        })
    }

    /// Tangent projection. Periodic cells drop rotation (Sella `proj_rot`).
    fn project_vec(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self.manifold {
            ManifoldKind::MwRigid => {
                let mut w = v.clone();
                let masses = self.masses.as_ref().and_then(|m| m.as_slice());
                project_horizontal(&mut w, x.view(), masses, !self.periodic);
                w
            }
            ManifoldKind::RigidQuotient => {
                let mut w = v.clone();
                project_horizontal(&mut w, x.view(), None, !self.periodic);
                w
            }
            other => other.project(x, v),
        }
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
            p.column_mut(j).assign(&self.project_vec(x, &e));
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

    /// Column expansion `P^T H P + (I - P)^T (I - P)` that the sphere form
    /// is checked against.
    #[cfg(test)]
    fn tangent_hessian(&self, x: &Array1<f64>, hess: &Array2<f64>) -> Array2<f64> {
        let n = x.len();
        let p = self.projection_columns(x);
        let normal = Array2::<f64>::eye(n) - &p;
        p.t().dot(hess).dot(&p) + normal.t().dot(&normal)
    }

    /// Euclidean sessions keep the owned gradient. Other kinds project it.
    fn horizontal_owned(&self, x: &Array1<f64>, grad: Array1<f64>) -> Array1<f64> {
        if !self.project_rigid && matches!(self.manifold, ManifoldKind::Euclidean) {
            return grad;
        }
        self.horizontal_grad(x, &grad)
    }

    fn horizontal_grad(&self, x: &Array1<f64>, grad: &Array1<f64>) -> Array1<f64> {
        let mut g = match self.manifold {
            ManifoldKind::MwRigid | ManifoldKind::RigidQuotient => self.project_vec(x, grad),
            other => other.egrad2rgrad(x, grad),
        };
        if self.project_rigid
            && !matches!(
                self.manifold,
                ManifoldKind::RigidQuotient | ManifoldKind::MwRigid
            )
        {
            project_out_rot_trans(&mut g, x.view());
        }
        g
    }

    /// Vector transport. Quotient manifolds project at the arrival point.
    fn transport_vec(
        &self,
        x_from: &Array1<f64>,
        x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        match self.manifold {
            ManifoldKind::RigidQuotient | ManifoldKind::MwRigid => self.project_vec(x_to, v),
            other => other.transport(x_from, x_to, v),
        }
    }

    /// Riemannian L-BFGS pair at `x`: `s = T(x - old)`, `y = g - T(g_old)`.
    fn lbfgs_sy(
        &self,
        old: &Array1<f64>,
        x: &Array1<f64>,
        gold: &Array1<f64>,
        grad: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        // Euclidean transport is a copy, and with rigid motion left in
        // nothing is removed afterwards. `s = x - old` and `y = g - g_old`
        // are those copies.
        if !self.project_rigid && self.manifold == ManifoldKind::Euclidean {
            return (x - old, grad - gold);
        }
        let s = self.transport_vec(old, x, &(x - old));
        let y = grad - &self.transport_vec(old, x, gold);
        (s, y)
    }

    fn same_last_x(&self, x: &Array1<f64>) -> bool {
        match &self.last_pos {
            Some(p) if p.len() == x.len() => p
                .iter()
                .zip(x.iter())
                .all(|(a, b)| (*a - *b).abs() <= 1e-15),
            _ => false,
        }
    }

    fn remember(&mut self, x: &Array1<f64>, value: f64, grad: &Array1<f64>) {
        match &mut self.last_pos {
            Some(pos) if pos.len() == x.len() => pos.assign(x),
            _ => self.last_pos = Some(x.to_owned()),
        }
        self.last_value = value;
        crate::workspace::assign_fit(&mut self.last_grad, grad.view());
    }

    /// Store `x` and `value`. The gradient is already in [`Self::last_grad`].
    fn remember_pos(&mut self, x: &Array1<f64>, value: f64) {
        match &mut self.last_pos {
            Some(pos) if pos.len() == x.len() => pos.assign(x),
            _ => self.last_pos = Some(x.to_owned()),
        }
        self.last_value = value;
    }

    fn stationarity_norm(&self, x: &Array1<f64>, grad: &Array1<f64>) -> f64 {
        #[cfg(feature = "highs")]
        if self.highs
            && (self.box_lo.is_some() || self.box_hi.is_some())
            && matches!(self.manifold, ManifoldKind::Euclidean)
            && !self.project_rigid
            && self.equalities.is_empty()
        {
            use crate::lbfgs_qp::side_at;
            let mut norm = 0.0_f64;
            let mut projected = Array1::zeros(x.len());
            for k in 0..x.len() {
                let lo = side_at(self.box_lo.as_deref(), k).unwrap_or(f64::NEG_INFINITY);
                let hi = side_at(self.box_hi.as_deref(), k).unwrap_or(f64::INFINITY);
                if !(x[k].is_finite() && grad[k].is_finite() && lo <= x[k] && x[k] <= hi) {
                    return f64::INFINITY;
                }
                // x - clip(x-g, lo, hi) = clip(g, x-hi, x-lo).
                // Distance clipping preserves gradients smaller than an ulp of x.
                projected[k] = grad[k].clamp(x[k] - hi, x[k] - lo);
                norm = norm.hypot(projected[k]);
            }
            return match self.stop_norm {
                StopNorm::Two => norm,
                other => other.of(projected.view()),
            };
        }
        let _ = x;
        self.stop_norm.of(grad.view())
    }

    fn report(&self, value: f64, x: &Array1<f64>, grad: &Array1<f64>) -> Report {
        let grad_norm = self.stationarity_norm(x, grad);
        if self.brief {
            return Report {
                value,
                coords: Array1::zeros(0),
                steps: self.steps,
                grad_norm,
                stop_norm: self.stop_norm,
                grad: None,
            };
        }
        Report {
            value,
            coords: copied(x),
            steps: self.steps,
            grad_norm,
            stop_norm: self.stop_norm,
            grad: Some(copied(grad)),
        }
    }

    /// Minimize until the gradient test, the iteration budget, or a stop.
    ///
    /// Intermediate reports do not clone the coordinates or the gradient.
    /// The returned report does.
    pub fn run<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        self.brief = true;
        let tol = self.control.gtol;
        let budget = self.control.maxiter;
        let mut stop: Result<()> = Ok(());
        for _ in 0..budget {
            match self.step(obj, x) {
                Ok(rep) if rep.grad_norm < tol => {
                    let value = rep.value;
                    self.brief = false;
                    let g = self.last_grad.clone();
                    return Ok(self.report(value, x, &g));
                }
                Ok(_) => {}
                Err(err) => {
                    stop = Err(err);
                    break;
                }
            }
        }
        self.brief = false;
        match stop {
            Err(err) => Err(err),
            Ok(()) => {
                let value = self.last_value;
                let g = self.last_grad.clone();
                Ok(self.report(value, x, &g))
            }
        }
    }

    /// Record `s = x+ - x`, `y = g+ - g` from the caller's previous outer.
    ///
    /// A dimer (or any one-oracle-per-outer walker) cannot answer the
    /// second eval [`Self::step`] would take at the trial point. The
    /// pair that belongs in L-BFGS memory is the one between successive
    /// outers, not a frozen-gradient inner trial (that writes `y = 0`
    /// and wrecks the two-loop).
    pub fn push_pair(&mut self, s: ArrayView1<f64>, y: ArrayView1<f64>) -> bool {
        if s.len() != self.dim || y.len() != self.dim {
            return false;
        }
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
    pub fn search_direction(&self, g: ArrayView1<f64>) -> Result<Array1<f64>> {
        if g.len() != self.dim {
            return Err(Error::Dim {
                dim: self.dim,
                got: g.len(),
            });
        }
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

    /// Drop method memory. The next step is a cold start from the current `x`.
    /// Keep the method memory, drop the point. The next step evaluates
    /// the objective afresh at the current `x` with the initial step
    /// scale and an empty acceptance window, and the retained curvature
    /// pairs (or dense approximation) precondition it. This is the
    /// restart for a sequence of related objectives, such as a
    /// marginal-likelihood fit refreshed as observations arrive: the
    /// previous fit's optimum is the start, its curvature still applies,
    /// but its value and gradient there belong to the previous objective.
    pub fn rebase(&mut self) {
        self.istep = self.control.istep;
        self.lbfgs_burdakov = self.lbfgs_burdakov_init;
        self.e_hist.clear();
        self.last_pos = None;
        self.geometry_lbfgs_stiff = None;
    }

    /// Drop method memory. The next step is a cold start from the current `x`.
    pub fn forget(&mut self) {
        self.istep = self.control.istep;
        self.lbfgs_burdakov = self.lbfgs_burdakov_init;
        self.e_hist.clear();
        self.last_pos = None;
        self.geometry_lbfgs_stiff = None;
        self.saddle_kicks = 0;
        match &mut self.inner {
            Inner::Lbfgs(solver) => solver.forget(),
            Inner::Nlcg { initialized, .. } => *initialized = false,
            Inner::Bfgs { h, fresh } | Inner::Sr1 { h, fresh } | Inner::Sr2 { b: h, fresh } => {
                *h = Array2::<f64>::eye(self.dim);
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
    }

    /// One outer iteration. `x` is the iterate and is overwritten in place.
    /// A first-order step whose start, or whose FIRE, geometry fallback or
    /// L-BFGS [`Accept::None`] trial, answers with a non-finite value or
    /// gradient returns [`Error::Oracle`] with `x` and the solver as they
    /// were, so a retry probes the same trial. Under
    /// [`Self::set_lbfgs_trust_radius`] the step scores a non-finite
    /// trial as `rho < 0` instead. The radius shrinks, `x` and the
    /// pairs stay, and the step returns a report at `x`.
    pub fn step<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
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
        if self.highs && (self.box_lo.is_some() || self.box_hi.is_some()) {
            let bounded = crate::box_objective::BoxObjective::new(
                obj,
                self.box_lo.clone(),
                self.box_hi.clone(),
            )?;
            if bounded.clip_start(x)? {
                self.last_pos = None;
            }
            return self.step_first_order(&bounded, x);
        }
        self.step_first_order(obj, x)
    }

    /// One Newton / RFO iteration. Hessian is rebuilt at the current `x`.
    pub fn step_hess<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: HessianObjective + ?Sized,
    {
        if self.constraints.is_some() {
            return self.constrained_step(x, |s, x| s.step_constrained_hess(obj, x));
        }
        // Newton-CG never forms `obj.hessian`, even when the caller
        // handed a dense oracle to this verb. The product is a forward
        // difference of gradients.
        if matches!(self.inner, Inner::NewtonCg { .. } | Inner::NewtonPcg { .. }) {
            return self.step_newton_cg(obj, x, &FiniteDifference);
        }
        #[cfg(feature = "highs")]
        if self.highs && (self.box_lo.is_some() || self.box_hi.is_some()) {
            let bounded = crate::box_objective::BoxObjective::new(
                obj,
                self.box_lo.clone(),
                self.box_hi.clone(),
            )?;
            if bounded.clip_start(x)? {
                self.last_pos = None;
            }
            return self.step_hess_inner(&bounded, x);
        }
        self.step_hess_inner(obj, x)
    }

    /// One Steihaug-Toint iteration. `obj` supplies analytic products;
    /// this verb does not build a dense Hessian.
    ///
    /// [`Self::step`] and [`Self::step_hess`] on the same methods use a
    /// forward difference (one gradient per product) instead.
    pub fn step_hvp<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: HessianVector + ?Sized,
    {
        if !matches!(self.inner, Inner::NewtonCg { .. } | Inner::NewtonPcg { .. }) {
            return Err(Error::NeedHvp);
        }
        let action = |xv: ArrayView1<f64>, v: ArrayView1<f64>| obj.hessian_vector(xv, v);
        if self.constraints.is_some() {
            return self.constrained_step(x, |s, x| s.step_constrained_newton_cg(obj, x, &action));
        }
        self.step_newton_cg(obj, x, &action)
    }

    fn step_newton_cg<O, A>(&mut self, obj: &O, x: &mut Array1<f64>, action: &A) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
        A: OptionalAction + Send + Sync,
    {
        if x.len() != self.dim || self.dim != Objective::dim(obj) {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim,
            });
        }
        self.check_manifold(x.len())?;
        let cached = self.same_last_x(x);
        if !cached {
            clip_mut(obj.bounds(), x);
        }
        let (value, raw_grad) = if cached {
            (self.last_value, self.last_grad.clone())
        } else {
            self.gradient_calls += 1;
            obj.value_and_gradient(x.view())
        };
        // CG sees the horizontal gradient. The forward difference
        // needs the raw gradient, which is what `remember` stores on
        // this path.
        let grad = self.horizontal_grad(x, &raw_grad);
        let gnorm = self.stationarity_norm(x, &grad);
        if !gnorm.is_finite() {
            return Err(Error::TrustCollapsed { steps: self.steps });
        }
        if gnorm < self.control.gtol {
            self.remember(x, value, &raw_grad);
            return Ok(self.report(value, x, &grad));
        }
        let (radius, precondition) = match &self.inner {
            Inner::NewtonCg { radius } => (*radius, false),
            Inner::NewtonPcg { radius } => (*radius, true),
            _ => return Err(Error::NeedHvp),
        };
        // Same ceiling as dogleg: Euclidean `maxmove` when one is set,
        // otherwise eight times the current radius, and never below it.
        let rmax = self
            .control
            .maxmove
            .unwrap_or(radius * RADIUS_GROWTH)
            .max(radius);
        let taken = newton_cg_step(
            obj,
            x,
            value,
            &grad,
            &raw_grad,
            &NewtonCgRequest {
                radius,
                rmax,
                atom_cap: self.atom_maxmove,
                euclid_cap: self.control.maxmove,
                project_rigid: self.project_rigid,
                precondition,
                seed: self.steps as u64,
                manifold: (!self.manifold.is_flat()).then_some(self.manifold),
            },
            action,
        );
        self.hvp_calls += taken.hvps;
        self.gradient_calls += taken.trial_gradients;
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
        Ok(self.report(taken.value, x, &report_grad))
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

    fn use_lbfgs_burdakov(&self) -> bool {
        // NaN is not `<= 0`, so a non-finite stored radius stays on, as before.
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
        // An infinite slope also overflows |d|, so tau = 0 would accept x.
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
        } else if let Some(cap) = self.control.maxmove {
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

    /// Line-search-free cap on unconstrained Euclidean L-BFGS. Curved
    /// kinds keep their retraction search. A distance-constraint set and
    /// HiGHS keep their own projections.
    fn use_capped_lbfgs(&self) -> bool {
        let capped = self.lbfgs_step().capped_limit().is_some()
            && matches!(self.manifold, ManifoldKind::Euclidean)
            && self.constraints.is_none();
        #[cfg(feature = "highs")]
        let capped = capped && !self.highs;
        capped
    }

    fn use_geometry_lbfgs(&self) -> bool {
        let on = matches!(self.lbfgs_step(), LbfgsStep::Wolfe)
            && self.atom_maxmove.is_some()
            && matches!(self.manifold, ManifoldKind::Euclidean)
            && matches!(self.inner, Inner::Lbfgs(_));
        #[cfg(feature = "highs")]
        let on = on && !self.highs;
        on
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

    fn shape_step(&self, x: &Array1<f64>, mut direction: Array1<f64>) -> Array1<f64> {
        if !self.project_rigid && matches!(self.manifold, ManifoldKind::Euclidean) {
            return direction;
        }
        if self.project_rigid {
            project_out_rot_trans(&mut direction, x.view());
        }
        self.project_vec(x, &direction)
    }

    /// Search direction of one capped step, after rigid and manifold projection.
    fn capped_search(&mut self, x: &Array1<f64>, g: &Array1<f64>) -> Option<(Array1<f64>, bool)> {
        let (direction, full) = match &mut self.inner {
            Inner::Lbfgs(solver) => solver.capped_direction(g.view())?,
            _ => return None,
        };
        let direction = self.shape_step(x, direction);
        let slope = crate::vecops::dot(direction.view(), g.view());
        if slope.is_finite() && slope < 0.0 {
            return Some((direction, full));
        }
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.forget();
        }
        let direction = self.shape_step(x, g.mapv(|v| -v));
        let slope = crate::vecops::dot(direction.view(), g.view());
        (slope.is_finite() && slope < 0.0).then_some((direction, true))
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
    ) -> Result<(Array1<f64>, f64, Array1<f64>, bool)>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let motion = max_atom_displacement(step);
        if !motion.is_finite() || motion <= 0.0 {
            return Err(Error::LbfgsCap {
                what: "the capped step is zero",
            });
        }
        let trial = clip_owned(obj.bounds(), x + step);
        let taken = max_atom_displacement(&(&trial - x));
        if !taken.is_finite() || taken <= 0.0 {
            return Err(Error::OffManifold {
                kind: self.manifold.as_str(),
            });
        }
        let (value, grad) = obj.value_and_gradient(trial.view());
        if !finite(value, grad.view()) {
            return Err(if steepest {
                non_finite()
            } else {
                Error::Oracle {
                    what: "non-finite quasi-Newton trial",
                }
            });
        }
        Ok((trial, value, grad, steepest))
    }

    fn propose_capped<O>(
        &mut self,
        obj: &O,
        x: &Array1<f64>,
        grad: &Array1<f64>,
        cap: f64,
    ) -> std::result::Result<(Array1<f64>, f64, Array1<f64>), CappedMiss>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let Some((direction, full)) = self.capped_search(x, grad) else {
            return Err(CappedMiss::Stop(Error::LbfgsCap {
                what: "the gradient is not a descent direction",
            }));
        };
        let step = Lbfgs::apply_atom_cap(&direction, cap, full);
        match self.capped_trial(obj, x, &step, full) {
            Ok((trial, value, grad, _)) => Ok((trial, value, grad)),
            Err(Error::Oracle {
                what: "non-finite quasi-Newton trial",
            }) => Err(CappedMiss::Retry),
            Err(err) => Err(CappedMiss::Stop(err)),
        }
    }

    /// One line-search-free L-BFGS step. [`LbfgsStep::Capped`] follows
    /// eOn. [`LbfgsStep::Scaled`] shortens the two-loop step and keeps
    /// the pairs.
    fn capped_lbfgs_step<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        value: f64,
        grad: Array1<f64>,
    ) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if matches!(self.lbfgs_step(), LbfgsStep::Capped { .. }) {
            return self.eon_lbfgs_step(obj, x, value, grad);
        }
        self.scaled_lbfgs_step(obj, x, value, grad)
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
        let probe = clip_owned(obj.bounds(), raw);
        let along = (&probe - x).dot(&u);
        if !(along.is_finite() && along > 0.0) {
            return EonSeed::Scale(EON_INVERSE_CURVATURE);
        }
        let (value, grad) = obj.value_and_gradient(probe.view());
        if !finite(value, grad.view()) {
            return EonSeed::Steepest;
        }
        let curvature = (&grad - g).dot(&u) / along;
        if curvature.is_finite() && curvature > 0.0 {
            EonSeed::Scale(1.0 / curvature)
        } else {
            EonSeed::Steepest
        }
    }

    /// One eOn L-BFGS step. A non-finite quasi-Newton trial is retried
    /// once along `H0` times the force. A second non-finite trial
    /// restores the pairs and returns [`Error::Oracle`].
    fn eon_lbfgs_step<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        _value: f64,
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
            && let Some(previous) = self.last_pos.as_ref()
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
                match self.capped_trial(obj, x, &step, true) {
                    Ok((trial, value, grad, _)) => (trial, value, grad),
                    Err(err) => {
                        if let Inner::Lbfgs(solver) = &mut self.inner {
                            solver.rollback_txn(txn);
                        }
                        return Err(err);
                    }
                }
            }
            Err(CappedMiss::Stop(err)) => {
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.rollback_txn(txn);
                }
                return Err(err);
            }
        };
        let (trial, tval, tgrad) = proposed;
        let old = x.clone();
        let gold = grad;
        *x = trial;
        let grad = self.horizontal_owned(x, tgrad);
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

    /// Euclidean eOn step. The displacement is added to `x` in place.
    #[allow(clippy::too_many_arguments)]
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
        let result = self.capped_trial(obj, x, &step, take.fallback);
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.recycle(step);
        }
        match result {
            Ok((trial, value, grad, _)) => Ok((trial, value, grad)),
            Err(Error::Oracle {
                what: "non-finite quasi-Newton trial",
            }) => Err(CappedMiss::Retry),
            Err(err) => Err(CappedMiss::Stop(err)),
        }
    }

    /// One ASE-style L-BFGS step. The two-loop direction keeps its pairs
    /// when it is shortened to the cap. A non-finite quasi-Newton trial
    /// is retried once along the force. A second non-finite trial restores
    /// the pairs and returns [`Error::Oracle`].
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
        let Some(cap) = self.lbfgs_step().capped_limit() else {
            return Err(Error::LbfgsCap {
                what: "the cap is not positive",
            });
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
                solver.push(s, y);
            }
        }
        let proposed = match self.propose_capped(obj, x, &grad, cap) {
            Ok(point) => point,
            Err(CappedMiss::Retry) => {
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.forget();
                }
                match self.propose_capped(obj, x, &grad, cap) {
                    Ok(point) => point,
                    Err(CappedMiss::Retry) => {
                        if let Inner::Lbfgs(solver) = &mut self.inner {
                            solver.restore(saved);
                        }
                        return Err(non_finite());
                    }
                    Err(CappedMiss::Stop(err)) => {
                        if let Inner::Lbfgs(solver) = &mut self.inner {
                            solver.restore(saved);
                        }
                        return Err(err);
                    }
                }
            }
            Err(CappedMiss::Stop(err)) => {
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
        let grad = self.horizontal_owned(x, tgrad);
        let s = &*x - &old;
        let y = &grad - &gold;
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.push(s, y);
        }
        self.remember(x, tval, &grad);
        self.steps += 1;
        Ok(self.report(tval, x, &grad))
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
        let saved = match &self.inner {
            Inner::Lbfgs(solver) => Some((solver.curvature(), self.geometry_lbfgs_stiff)),
            _ => None,
        };
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
                let probe = clip_owned(obj.bounds(), raw);
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
            if self.project_rigid {
                project_out_rot_trans(&mut direction, x.view());
            }
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
            trial = clip_owned(obj.bounds(), &*x + alpha * &direction);
            let ev = obj.value_and_gradient(trial.view());
            tval = ev.0;
            tgrad = ev.1;
            if finite(tval, tgrad.view()) && tval - reference <= 1.0e-8 {
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
            trial = clip_owned(obj.bounds(), &*x + &sd);
            let ev = obj.value_and_gradient(trial.view());
            tval = ev.0;
            tgrad = ev.1;
            if !finite(tval, tgrad.view()) {
                if let (Inner::Lbfgs(solver), Some((curvature, stiff))) = (&mut self.inner, saved) {
                    solver.restore(curvature);
                    self.geometry_lbfgs_stiff = stiff;
                }
                return Err(non_finite());
            }
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
        grad = self.horizontal_owned(x, tgrad);
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
        Ok(self.report(tval, x, &grad))
    }

    fn saddle_escape_enabled(&self) -> bool {
        #[cfg(feature = "highs")]
        if self.highs {
            return false;
        }
        matches!(self.inner, Inner::Newton { .. } | Inner::Dogleg { .. })
    }

    /// Lowest eigenpair of `hess` when it lies below the stationarity tolerance.
    fn unstable_mode(&self, hess: &Array2<f64>) -> Option<Array1<f64>> {
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
            struct Mat<'a>(&'a Array2<f64>);
            impl ApplyHessian for Mat<'_> {
                fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
                    self.0.dot(&v)
                }
            }
            let mut seed = Array1::zeros(n);
            for i in 0..n {
                seed[i] = ((i % 11) as f64) - 5.0;
            }
            let Ok(found) = lowest_eigenpair(&Mat(hess), seed.view(), seed.view(), 48.min(n))
            else {
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
            if gnorm > self.control.gtol || self.saddle_kicks >= SADDLE_KICKS {
                return;
            }
            let hess = obj.hessian(x.view());
            let Some(mode) = self.unstable_mode(&hess) else {
                return;
            };
            let cap = self
                .atom_maxmove
                .unwrap_or_else(|| self.control.istep.max(RADIUS_START));
            let step = scale_displacement_to_atom_cap(&mode, cap);
            let trial = clip_owned(obj.bounds(), &*x + &step);
            let (v, g) = obj.value_and_gradient(trial.view());
            if !v.is_finite() || g.iter().any(|c| !c.is_finite()) {
                return;
            }
            *x = trial;
            *value = v;
            *grad = self.horizontal_owned(x, g);
            self.saddle_kicks = self.saddle_kicks.saturating_add(1);
            if let Inner::Dogleg { radius } = &mut self.inner {
                *radius = self.control.istep.max(RADIUS_START);
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
        Ok(self.report(value, x, &grad))
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
            clip_mut(obj.bounds(), x);
        }
        let (mut value, mut grad) = if cached {
            (self.last_value, self.last_grad.clone())
        } else {
            obj.value_and_gradient(x.view())
        };
        grad = self.horizontal_owned(x, grad);
        let gnorm = self.stationarity_norm(x, &grad);
        if gnorm < self.control.gtol {
            return self.report_or_escape(obj, x, value, grad);
        }
        let hess = obj.hessian(x.view());
        if matches!(self.inner, Inner::Dogleg { .. }) {
            return self.step_dogleg(obj, x, value, grad, &hess);
        }
        #[cfg(feature = "highs")]
        if self.highs {
            let center = if self.project_rigid && self.dim.is_multiple_of(3) {
                Some((self.dim / 3, 3))
            } else {
                None
            };
            let mut model_hess = hess.clone();
            if newton_kind == Some(NewtonKind::Rfo) {
                // The lowest augmented eigenvalue makes H - lambda I
                // convex and preserves the unconstrained RFO direction.
                let (_, shift) = crate::sella_step::rfo_step_and_shift(&hess, &grad, 0, 1.0);
                for k in 0..self.dim {
                    model_hess[(k, k)] -= shift;
                }
            }
            let dir = crate::lbfgs_qp::highs_feasible_step(
                None,
                Some(&model_hess),
                &grad,
                x.view(),
                self.box_lo.as_deref(),
                self.box_hi.as_deref(),
                self.atom_maxmove,
                self.highs_trust.or(self.control.maxmove),
                center,
                &self.equalities,
                self.highs_solver,
                self.highs_crossover,
                self.highs_callback,
                self.highs_callback_user,
            )?;
            let old = x.clone();
            let gold = grad.clone();
            let (npos, nval, ngrad, moved) = accept_step_with_fallback(
                obj,
                x,
                value,
                &gold,
                &dir,
                Acceptance {
                    control: &self.control,
                    accept: self.accept,
                    e_hist: &mut self.e_hist,
                    atom_maxmove: None,
                    manifold: self.manifold,
                },
                false,
            );
            if moved {
                *x = npos;
                value = nval;
                grad = ngrad;
            }
            grad = self.horizontal_owned(x, grad);
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
            return Ok(self.report(value, x, &grad));
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
        if self.project_rigid {
            project_out_rot_trans(&mut dir, x.view());
        }
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
                manifold: self.manifold,
            },
        );
        if moved {
            *x = npos;
            value = nval;
            grad = ngrad;
        }
        grad = self.horizontal_owned(x, grad);
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
            _ => self.control.istep.max(RADIUS_START),
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
            .maxmove
            .unwrap_or(radius * RADIUS_GROWTH)
            .max(radius);
        // A curved kind steps in the tangent space. The dogleg point is
        // tangent, the trial is the retracted tangent step, and the ratio
        // compares the energy there with the tangent model.
        let curved = !self.manifold.is_flat();
        let mut dir = self.dogleg_point(x, hess, &grad, radius);
        if self.project_rigid {
            project_out_rot_trans(&mut dir, x.view());
        }
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
                self.control.maxmove,
                self.manifold,
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
            if let Some(cap) = self.atom_maxmove {
                scale_step_atom(x, &mut trial, cap);
            } else if let Some(cap) = self.control.maxmove {
                scale_step(x, &mut trial, cap);
            }
            trial = clip_owned(obj.bounds(), trial);
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
        let boundary_norm = if self.atom_maxmove.is_some() || self.control.maxmove.is_some() {
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

    /// Euclidean FIRE with the default preset and no preconditioner.
    fn fire_is_flat(&self) -> bool {
        let Inner::Fire(state) = &self.inner else {
            return false;
        };
        matches!(self.manifold, ManifoldKind::Euclidean)
            && !self.project_rigid
            && self.exp.is_none()
            && !state.abc
            && state.preset() == crate::fire::FirePreset::Default
    }

    /// In-place Euclidean FIRE step. The integrator matches the general arm.
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
        let cached = self.same_last_x(x);
        if !cached {
            clip_mut(obj.bounds(), x);
        }

        if let Inner::Pso { .. } = &self.inner {
            return self.step_pso(obj, x);
        }

        if self.exp.is_none()
            && !self.project_rigid
            && matches!(self.manifold, ManifoldKind::Euclidean)
            && !self.use_lbfgs_burdakov()
        {
            if self.use_capped_lbfgs() && matches!(self.lbfgs_step(), LbfgsStep::Capped { .. }) {
                return self.step_eon_euclidean(obj, x);
            }
            if matches!(self.inner, Inner::Fire(_)) {
                return self.step_fire_euclidean(obj, x);
            }
        }

        let (mut value, mut grad) = if cached {
            if self.brief {
                (
                    self.last_value,
                    std::mem::replace(&mut self.last_grad, Array1::zeros(0)),
                )
            } else {
                (self.last_value, self.last_grad.clone())
            }
        } else {
            obj.value_and_gradient(x.view())
        };
        grad = self.horizontal_owned(x, grad);
        if !finite(value, grad.view()) {
            if self.brief {
                self.last_grad = grad;
            }
            return Err(non_finite());
        }
        let gnorm = self.stationarity_norm(x, &grad);
        if gnorm < self.control.gtol {
            if self.brief {
                self.remember(x, value, &grad);
            }
            return Ok(self.report(value, x, &grad));
        }
        self.refresh_exp(obj, x, &grad)?;

        if self.use_lbfgs_burdakov() {
            if !cached && let Some(previous) = &self.last_pos {
                let (s, y) = self.lbfgs_sy(previous, x, &self.last_grad, &grad);
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.push_pair(s, y, Some(l2(&grad)));
                }
            }
            return self.lbfgs_burdakov_step(obj, x, value, grad);
        }

        if self.use_capped_lbfgs() {
            return self.capped_lbfgs_step(obj, x, value, grad);
        }

        if matches!(self.manifold, ManifoldKind::Sphere | ManifoldKind::Stiefel)
            && matches!(self.inner, Inner::Lbfgs(_))
            && self.accept != Accept::None
        {
            if !cached && let Some(previous) = &self.last_pos {
                let (s, y) = self.lbfgs_sy(previous, x, &self.last_grad, &grad);
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.transport(|v| crate::manifold::Sphere.transport(previous, x, v));
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
                );
                *x = point;
                value = f;
                grad = g;
            }
            self.remember(x, value, &grad);
            self.steps += 1;
            return Ok(self.report(value, x, &grad));
        }

        if self.use_geometry_lbfgs() {
            return self.geometry_lbfgs_step(obj, x, value, grad);
        }

        // An external displacement has a measured gradient at both ends.
        // Its secant updates the retained inverse Hessian at the supplied
        // point, using the same vector transport as accepted solver steps.
        // A refused Accept::None trial takes the pair back: `last_pos`
        // still names the old point, so a retry records the move again.
        let mut before_move = None;
        if !cached && let Some(previous) = &self.last_pos {
            let (s, y) = self.lbfgs_sy(previous, x, &self.last_grad, &grad);
            if let Inner::Lbfgs(solver) = &mut self.inner {
                if self.accept == Accept::None {
                    before_move = Some(solver.curvature());
                }
                solver.push_pair(s, y, Some(l2(&grad)));
            }
        }

        let raw_gradient = match self.inner {
            Inner::Steepest
            | Inner::Bfgs { .. }
            | Inner::Sr1 { .. }
            | Inner::Sr2 { .. }
            | Inner::Bb { .. } => true,
            Inner::Nlcg { .. } => self.exp.is_none(),
            _ => false,
        };
        if self.unit_first_step
            && raw_gradient
            && self.accept != Accept::None
            && self.last_pos.is_none()
        {
            self.istep = self.control.istep.min(1.0 / l2(&grad));
        }

        let start = x.clone();
        // The origin gradient is the secant of an L-BFGS step. Other
        // Euclidean steps do not read it.
        let gold = if matches!(self.inner, Inner::Lbfgs(_)) {
            Some(grad.clone())
        } else {
            None
        };
        let mut refused = false;
        match &mut self.inner {
            Inner::Lbfgs(solver) => {
                // Accept::None: take the clipped two-loop step. The NEB
                // band force is not a gradient of the reported energy, so
                // step_objective (nval < value) never moves.
                // Accept::Energy: Wolfe line search in step_objective.
                if self.accept == Accept::None {
                    let dir = solver.search_direction(x.view(), grad.view());
                    let old = x.clone();
                    let gold = grad.clone();
                    let trial = single_trial(
                        obj,
                        x,
                        &dir,
                        &self.control,
                        &mut self.e_hist,
                        self.atom_maxmove,
                        self.manifold,
                    );
                    let (npos, nval, ngrad) = match trial {
                        Trial::Accepted(npos, nval, ngrad) => (npos, nval, ngrad),
                        Trial::OffManifold => {
                            if let Some(saved) = before_move {
                                solver.restore(saved);
                            }
                            return Err(Error::OffManifold {
                                kind: self.manifold.as_str(),
                            });
                        }
                        Trial::NonFinite => {
                            if let Some(saved) = before_move {
                                solver.restore(saved);
                            }
                            return Err(Error::Oracle {
                                what: "non-finite value or gradient",
                            });
                        }
                    };
                    *x = npos;
                    value = nval;
                    grad = ngrad;
                    solver.push(&*x - &old, &grad - &gold);
                } else {
                    solver.step_objective(
                        obj,
                        x,
                        &mut value,
                        &mut grad,
                        &mut self.istep,
                        self.linesearch,
                        &self.control,
                    );
                }
            }
            Inner::Steepest => {
                let dir = grad.mapv(|g| -g * self.istep);
                let (npos, nval, ngrad, moved) = accept_step(
                    obj,
                    x,
                    value,
                    &grad,
                    &dir,
                    Acceptance {
                        control: &self.control,
                        accept: self.accept,
                        e_hist: &mut self.e_hist,
                        atom_maxmove: self.atom_maxmove,
                        manifold: self.manifold,
                    },
                );
                *x = npos;
                value = nval;
                grad = ngrad;
                refused = !moved && self.accept != Accept::None;
            }
            Inner::Nlcg {
                conjugacy,
                restart,
                dir,
                g_old,
                z_old,
                d_old,
                initialized,
            } => {
                let metric = self.exp.is_some();
                let z_now = if metric {
                    metric_vector(&self.exp, &grad)
                } else {
                    grad.clone()
                };
                if !*initialized {
                    *dir = if metric {
                        z_now.mapv(|z| -z * self.istep)
                    } else {
                        grad.mapv(|g| -g * self.istep)
                    };
                    *g_old = grad.clone();
                    *z_old = z_now;
                    *d_old = dir.clone();
                    *initialized = true;
                }
                let step_dir = dir.clone();
                let (npos, nval, ngrad, moved) = accept_step(
                    obj,
                    x,
                    value,
                    &grad,
                    &step_dir,
                    Acceptance {
                        control: &self.control,
                        accept: self.accept,
                        e_hist: &mut self.e_hist,
                        atom_maxmove: self.atom_maxmove,
                        manifold: self.manifold,
                    },
                );
                *x = npos;
                value = nval;
                grad = ngrad;
                if moved {
                    let ctx = ConjugacyContext {
                        current_gradient: grad.view(),
                        previous_gradient: g_old.view(),
                        previous_direction: d_old.view(),
                    };
                    if metric {
                        let z_new = metric_vector(&self.exp, &grad);
                        let mut beta = conjugacy.beta_metric(&ctx, z_new.view(), z_old.view());
                        if restart.should_restart_metric(
                            grad.view(),
                            z_new.view(),
                            g_old.view(),
                            z_old.view(),
                        ) {
                            beta = 0.0;
                        }
                        *dir = Array1::from_iter(
                            z_new
                                .iter()
                                .zip(d_old.iter())
                                .map(|(z, d)| -z * self.istep + beta * d),
                        );
                        // A non-descent direction in the metric is not a valid
                        // conjugate step. Fall back to the Riesz steepest direction.
                        let gd = crate::vecops::dot(grad.view(), dir.view());
                        if gd.partial_cmp(&0.0) != Some(std::cmp::Ordering::Less) {
                            *dir = z_new.mapv(|z| -z * self.istep);
                        }
                        z_old.assign(&z_new);
                    } else {
                        let mut beta = conjugacy.beta(&ctx);
                        if restart.should_restart(&ctx) {
                            beta = 0.0;
                        }
                        *dir = Array1::from_iter(
                            grad.iter()
                                .zip(d_old.iter())
                                .map(|(g, d)| -g * self.istep + beta * d),
                        );
                        z_old.assign(&grad);
                    }
                    g_old.assign(&grad);
                    d_old.assign(dir);
                } else if self.accept != Accept::None {
                    // The refused direction would be retried from the same
                    // point on every later call.
                    *initialized = false;
                    refused = true;
                }
            }
            Inner::Bfgs { h, fresh } => {
                // The identity carries no length scale, and the energy test
                // allows ten halvings. Take `istep` along it, as L-BFGS does
                // with an empty memory, then rescale it by Nocedal-Wright
                // 6.20 before the first update. SR1 and SR2 do the same.
                let seed = *fresh && self.accept != Accept::None;
                let mut direction = -h.dot(&grad);
                if seed {
                    direction *= self.istep;
                }
                let old = x.clone();
                let gold = grad.clone();
                let (npos, nval, ngrad, moved) = accept_step(
                    obj,
                    x,
                    value,
                    &gold,
                    &direction,
                    Acceptance {
                        control: &self.control,
                        accept: self.accept,
                        e_hist: &mut self.e_hist,
                        atom_maxmove: self.atom_maxmove,
                        manifold: self.manifold,
                    },
                );
                *x = npos;
                value = nval;
                grad = ngrad;
                if moved {
                    let (s, y) = (&*x - &old, &grad - &gold);
                    if !seed {
                        bfgs_inverse_update(h, &s, &y);
                        *fresh = false;
                    } else if let Some(gamma) = secant_scale(&s, &y) {
                        *h = Array2::<f64>::eye(self.dim) * gamma;
                        *fresh = false;
                        bfgs_inverse_update(h, &s, &y);
                    }
                } else if self.accept != Accept::None {
                    // The refused direction would be retried from the same
                    // point on every later call.
                    *h = Array2::<f64>::eye(self.dim);
                    *fresh = true;
                    refused = true;
                }
            }
            Inner::Sr1 { h, fresh } => {
                let seed = *fresh && self.accept != Accept::None;
                let mut direction = -h.dot(&grad);
                if seed {
                    direction *= self.istep;
                }
                let old = x.clone();
                let gold = grad.clone();
                let (npos, nval, ngrad, moved) = accept_step(
                    obj,
                    x,
                    value,
                    &gold,
                    &direction,
                    Acceptance {
                        control: &self.control,
                        accept: self.accept,
                        e_hist: &mut self.e_hist,
                        atom_maxmove: self.atom_maxmove,
                        manifold: self.manifold,
                    },
                );
                *x = npos;
                value = nval;
                grad = ngrad;
                if moved {
                    let (s, y) = (&*x - &old, &grad - &gold);
                    if !seed {
                        sr1_inverse_update(h, &s, &y);
                        *fresh = false;
                    } else if let Some(gamma) = secant_scale(&s, &y) {
                        *h = Array2::<f64>::eye(self.dim) * gamma;
                        *fresh = false;
                        sr1_inverse_update(h, &s, &y);
                    }
                } else if self.accept != Accept::None {
                    // The refused direction would be retried from the same
                    // point on every later call.
                    *h = Array2::<f64>::eye(self.dim);
                    *fresh = true;
                    refused = true;
                }
            }
            Inner::Sr2 { b, fresh } => {
                let seed = *fresh && self.accept != Accept::None;
                let rhs = grad.mapv(|g| -g);
                let mut direction = solve_dense(b, &rhs).unwrap_or(rhs);
                if seed {
                    direction *= self.istep;
                }
                let old = x.clone();
                let gold = grad.clone();
                let (npos, nval, ngrad, moved) = accept_step(
                    obj,
                    x,
                    value,
                    &gold,
                    &direction,
                    Acceptance {
                        control: &self.control,
                        accept: self.accept,
                        e_hist: &mut self.e_hist,
                        atom_maxmove: self.atom_maxmove,
                        manifold: self.manifold,
                    },
                );
                *x = npos;
                value = nval;
                grad = ngrad;
                if moved {
                    let (s, y) = (&*x - &old, &grad - &gold);
                    if !seed {
                        sr2_hessian_update(b, &s, &y);
                        *fresh = false;
                    } else if let Some(gamma) = secant_scale(&s, &y) {
                        *b = Array2::<f64>::eye(self.dim) / gamma;
                        *fresh = false;
                        sr2_hessian_update(b, &s, &y);
                    }
                } else if self.accept != Accept::None {
                    *b = Array2::<f64>::eye(self.dim);
                    *fresh = true;
                    refused = true;
                }
            }
            Inner::Adam {
                m,
                v,
                b1p,
                b2p,
                beta1,
                beta2,
                eps,
            } => {
                let dir = adam_direction(m, v, &grad, *beta1, *beta2, (*b1p, *b2p), *eps);
                let dir = dir.mapv(|d| d * self.istep);
                let (npos, nval, ngrad, _) = accept_step(
                    obj,
                    x,
                    value,
                    &grad,
                    &dir,
                    Acceptance {
                        control: &self.control,
                        accept: self.accept,
                        e_hist: &mut self.e_hist,
                        atom_maxmove: self.atom_maxmove,
                        manifold: self.manifold,
                    },
                );
                *x = npos;
                value = nval;
                grad = ngrad;
                *b1p *= *beta1;
                *b2p *= *beta2;
            }
            Inner::Fire(state) => {
                // A refused trial returns with the integrator as it was, so a
                // retry probes the same point instead of one further along.
                let saved = state.checkpoint();
                let mut raw = grad.mapv(|g| -g);
                project_box_tangent(obj.bounds(), x, &mut raw);
                project_box_tangent(obj.bounds(), x, &mut state.vel);
                let manifold = self.manifold;
                let curved = !manifold.is_flat();
                match fire_prepare_v2(state, &raw) {
                    FirePreparation::Ready => {}
                    FirePreparation::Rollback(correction) => {
                        if curved {
                            let back = manifold.project(x, &correction);
                            let Some((point, _)) =
                                retract_within_cap(obj, x, &back, None, None, manifold)
                            else {
                                state.restore(saved);
                                *x = start;
                                return Err(Error::OffManifold {
                                    kind: manifold.as_str(),
                                });
                            };
                            *x = point;
                        } else {
                            *x = clip_owned(obj.bounds(), &*x + &correction);
                        }
                        let ev = obj.value_and_gradient(x.view());
                        if !finite(ev.0, ev.1.view()) {
                            state.restore(saved);
                            *x = start;
                            return Err(non_finite());
                        }
                        grad = ev.1;
                        raw = grad.mapv(|g| -g);
                        project_box_tangent(obj.bounds(), x, &mut raw);
                    }
                    FirePreparation::Stalled => {
                        return Err(Error::FireStalled { steps: self.steps });
                    }
                }
                let forward_start = x.clone();
                let dx = if self.exp.is_some() {
                    let mut steepest = metric_vector(&self.exp, &raw);
                    project_box_tangent(obj.bounds(), x, &mut steepest);
                    fire_displacement_metric(state, &raw, &steepest, |vel| {
                        fire_mix(&self.exp, vel, &raw, &steepest)
                    })
                } else {
                    fire_displacement(state, &raw)
                };
                let dx = match self.atom_maxmove.or(self.control.maxmove) {
                    Some(cap) if state.abc => fire_clip_components(state, cap),
                    _ => dx,
                };
                // `(x + dx - x) / dt` rounds `v` by about `u |x| / |dx|`. The
                // opt-in settings keep `v` unless a cap, a bound or the
                // retraction moved the trial.
                let plain = state.preset() == FirePreset::Default && !state.abc;
                let free = (!plain).then(|| &*x + &dx);
                let (atom_cap, cap) = if state.abc {
                    (None, None)
                } else {
                    (self.atom_maxmove, self.control.maxmove)
                };
                let trial = if curved {
                    let tangent = manifold.project(x, &dx);
                    let Some((trial, _)) =
                        retract_within_cap(obj, x, &tangent, atom_cap, cap, manifold)
                    else {
                        state.restore(saved);
                        *x = start;
                        return Err(Error::OffManifold {
                            kind: manifold.as_str(),
                        });
                    };
                    trial
                } else {
                    let mut trial = &*x + &dx;
                    if let Some(cap) = atom_cap {
                        scale_step_atom(x, &mut trial, cap);
                    } else if let Some(cap) = cap {
                        scale_step(x, &mut trial, cap);
                    }
                    clip_owned(obj.bounds(), trial)
                };
                let moved = free.is_none_or(|free| trial != free);
                if matches!(state.kind, FireKind::V2) && state.dt > 0.0 && moved {
                    let mut secant = (&trial - &forward_start) / state.dt;
                    // The secant of a curved step has a normal part at the
                    // trial, of order |trial - x| / 2 relative.
                    if curved {
                        manifold.project_in_place(&trial, &mut secant);
                    }
                    state.vel = secant;
                }
                let ev = obj.value_and_gradient(trial.view());
                if !finite(ev.0, ev.1.view()) {
                    state.restore(saved);
                    *x = start;
                    return Err(non_finite());
                }
                *x = trial;
                value = ev.0;
                grad = ev.1;
                project_box_tangent(obj.bounds(), x, &mut state.vel);
                // FIRE 2.0 mixes before the step. The post-step force is
                // read only by FIRE 1.0.
                if matches!(state.kind, FireKind::V1) {
                    let mut raw_new = grad.mapv(|g| -g);
                    project_box_tangent(obj.bounds(), x, &mut raw_new);
                    if self.exp.is_some() {
                        let mut steep_new = metric_vector(&self.exp, &raw_new);
                        project_box_tangent(obj.bounds(), x, &mut steep_new);
                        let mix_new = fire_mix(&self.exp, &state.vel, &raw_new, &steep_new);
                        fire_after_v1_metric(state, &raw_new, &steep_new, mix_new);
                    } else {
                        fire_after_v1(state, &raw_new);
                    }
                }
            }
            Inner::Bb { prev_s, prev_y } => {
                let dir = bb_direction(prev_s.as_ref(), prev_y.as_ref(), &grad, self.istep);
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
                        manifold: self.manifold,
                    },
                );
                *x = npos;
                value = nval;
                grad = ngrad;
                if moved {
                    *prev_s = Some(&*x - &old);
                    *prev_y = Some(&grad - &gold);
                } else if self.accept != Accept::None {
                    // The refused spectral length would be retried from the
                    // same point on every later call.
                    *prev_s = None;
                    *prev_y = None;
                    refused = true;
                }
            }
            Inner::Pso { .. }
            | Inner::Newton { .. }
            | Inner::Dogleg { .. }
            | Inner::NewtonCg { .. }
            | Inner::NewtonPcg { .. } => unreachable!(),
        }
        // Each refusal restarts from steepest descent. A second refusal
        // at the same point refused that restart, and every later call
        // would repeat it.
        if refused && self.refused && cached {
            return Err(Error::AcceptStalled { steps: self.steps });
        }
        self.refused = refused;

        // On a curved kind every arm but the L-BFGS line search evaluated
        // a retracted trial, so the energy in hand is the energy on the set.
        let line_search = matches!(self.inner, Inner::Lbfgs(_)) && self.accept != Accept::None;
        let retracted = !(self.manifold.is_flat() || line_search);
        if !retracted {
            let delta = &*x - &start;
            let y = self.manifold.retract(&start, &delta);
            if y.iter().zip(x.iter()).any(|(a, b)| (*a - *b).abs() > 1e-15) {
                *x = y;
                let ev = obj.value_and_gradient(x.view());
                value = ev.0;
                grad = ev.1;
            }
        }
        grad = self.horizontal_owned(x, grad);

        let pair = if matches!(self.inner, Inner::Lbfgs(_))
            && x.iter().zip(start.iter()).any(|(a, b)| a != b)
        {
            let gold = gold.as_ref().expect("L-BFGS keeps the gradient");
            let (s, y) = self.lbfgs_sy(&start, x, gold, &grad);
            Some((s, y, l2(&grad)))
        } else {
            None
        };
        if let (Inner::Lbfgs(solver), Some((s, y, gn))) = (&mut self.inner, pair) {
            solver.replace_newest(s, y, Some(gn));
        }

        self.remember(x, value, &grad);
        self.steps += 1;
        Ok(self.report(value, x, &grad))
    }

    /// Euclidean eOn step. The gradient at `x` is [`Self::reuse`].`gold`.
    /// A warm call writes the trial into the scratch sized at construction.
    fn step_eon_euclidean<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let cached = self.same_last_x(x);
        if cached {
            crate::workspace::assign_fit(&mut self.reuse.gold, self.last_grad.view());
        } else {
            let ev = obj.value_and_gradient(x.view());
            if !finite(ev.0, ev.1.view()) {
                if self.brief {
                    crate::workspace::absorb(&mut self.last_grad, ev.1);
                    self.last_value = ev.0;
                }
                return Err(non_finite());
            }
            crate::workspace::absorb(&mut self.reuse.gold, ev.1);
            self.last_value = ev.0;
        }
        let value = self.last_value;
        if !finite(value, self.reuse.gold.view()) {
            return Err(non_finite());
        }
        let gnorm = self.stationarity_norm(x, &self.reuse.gold);
        if gnorm < self.control.gtol {
            if self.brief {
                std::mem::swap(&mut self.last_grad, &mut self.reuse.gold);
                self.remember_pos(x, value);
                return Ok(self.report(value, x, &self.last_grad));
            }
            return Ok(self.report(value, x, &self.reuse.gold));
        }
        self.eon_euclidean_move(obj, x)
    }

    /// One eOn move from [`Self::reuse`].`gold`. The accepted gradient lands
    /// in [`Self::last_grad`].
    fn eon_euclidean_move<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        use crate::lbfgs::{EON_INVERSE_CURVATURE, EonSeed};
        if self.lindh_spec.is_some() {
            let xs: Vec<f64> = x.iter().copied().collect();
            let gs: Vec<f64> = self.reuse.gold.iter().copied().collect();
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
            && let Some(previous) = self.last_pos.as_ref()
        {
            let prev = previous.as_slice().unwrap_or(&[]);
            let xs = x.as_slice().unwrap_or(&[]);
            let gold = self.last_grad.as_slice().unwrap_or(&[]);
            let gnew = self.reuse.gold.as_slice().unwrap_or(&[]);
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.store_pos_diff(prev, xs, gold, gnew, &mut txn);
            }
        }
        let wants_fd = match &self.inner {
            Inner::Lbfgs(solver) => solver.eon_wants_fd(),
            _ => false,
        };
        let seed = if wants_fd {
            self.eon_fd_seed(obj, x, &self.reuse.gold)
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
        let proposed = self.eon_trial(obj, x, cap, seed, &mut txn);
        let (tval, steep_retry) = match proposed {
            Ok(value) => (value, false),
            Err(CappedMiss::Retry) => {
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.forget_txn(&mut txn);
                }
                self.force_cap_into(h0, cap);
                match self.trial_of(obj, x, true) {
                    Ok(value) => (value, true),
                    Err(err) => {
                        if let Inner::Lbfgs(solver) = &mut self.inner {
                            solver.rollback_txn(txn);
                        }
                        return Err(err);
                    }
                }
            }
            Err(CappedMiss::Stop(err)) => {
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.rollback_txn(txn);
                }
                return Err(err);
            }
        };
        let _ = steep_retry;
        self.saved_x.assign(x);
        x.assign(&self.reuse.trial);
        {
            let Solver {
                saved_x,
                reuse,
                inner,
                ..
            } = &mut *self;
            if let Inner::Lbfgs(solver) = inner {
                solver.store_pos_diff(
                    saved_x.as_slice().unwrap_or(&[]),
                    x.as_slice().unwrap_or(&[]),
                    reuse.gold.as_slice().unwrap_or(&[]),
                    reuse.grad.as_slice().unwrap_or(&[]),
                    &mut txn,
                );
                solver.commit_txn(txn);
            }
        }
        std::mem::swap(&mut self.last_grad, &mut self.reuse.grad);
        self.remember_pos(x, tval);
        self.steps += 1;
        Ok(self.report(tval, x, &self.last_grad))
    }

    /// `scratch_b = cap(h0 * -gold)`.
    fn force_cap_into(&mut self, h0: f64, cap: f64) {
        let n = self.reuse.gold.len().min(self.scratch_b.len());
        for i in 0..n {
            self.scratch_b[i] = h0 * -self.reuse.gold[i];
        }
        let motion = max_atom_displacement(&self.scratch_b);
        if motion > cap && motion > 0.0 && cap > 0.0 {
            self.scratch_b *= cap / motion;
        }
    }

    /// Trial of the displacement in `scratch_b` when `from_scratch`, otherwise
    /// of the vector `eon_step_from_seed` just produced.
    fn eon_trial<O>(
        &mut self,
        obj: &O,
        x: &Array1<f64>,
        cap: f64,
        seed: crate::lbfgs::EonSeed,
        txn: &mut crate::lbfgs::Txn,
    ) -> std::result::Result<f64, CappedMiss>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        self.scratch.assign(&self.reuse.gold);
        let take = {
            let Solver { scratch, inner, .. } = &mut *self;
            match inner {
                Inner::Lbfgs(solver) => solver.eon_step_from_seed(scratch.view(), cap, seed, txn),
                _ => None,
            }
        };
        let Some(take) = take else {
            return Err(CappedMiss::Stop(Error::LbfgsCap {
                what: "the capped step is zero",
            }));
        };
        let step = self.shape_step(x, take.step);
        let result = self.trial_of_step(obj, x, &step, take.fallback);
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.recycle(step);
        }
        result
    }

    /// Value at `clip(x + scratch_b)`. The point is [`Self::reuse`].`trial`
    /// and the gradient is [`Self::reuse`].`grad`.
    fn trial_of<O>(&mut self, obj: &O, x: &Array1<f64>, steepest: bool) -> Result<f64>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let n = x.len().min(self.scratch_b.len());
        for i in 0..n {
            self.reuse.corr[i] = self.scratch_b[i];
        }
        let step = self.shape_step(x, self.reuse.corr.clone());
        let value = self
            .trial_of_step(obj, x, &step, steepest)
            .map_err(|miss| match miss {
                CappedMiss::Stop(err) => err,
                CappedMiss::Retry => non_finite(),
            })?;
        let _ = step;
        Ok(value)
    }

    /// `clip(x + step)` into [`Self::reuse`].`trial`.
    fn trial_of_step<O>(
        &mut self,
        obj: &O,
        x: &Array1<f64>,
        step: &Array1<f64>,
        steepest: bool,
    ) -> std::result::Result<f64, CappedMiss>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let motion = max_atom_displacement(step);
        if !motion.is_finite() || motion <= 0.0 {
            return Err(CappedMiss::Stop(Error::LbfgsCap {
                what: "the capped step is zero",
            }));
        }
        let n = x.len().min(step.len()).min(self.reuse.trial.len());
        for i in 0..n {
            self.reuse.trial[i] = x[i] + step[i];
        }
        clip_mut(obj.bounds(), &mut self.reuse.trial);
        let taken = max_atom_motion(x, &self.reuse.trial);
        if !taken.is_finite() || taken <= 0.0 {
            return Err(CappedMiss::Stop(Error::OffManifold {
                kind: self.manifold.as_str(),
            }));
        }
        let (value, grad) = obj.value_and_gradient(self.reuse.trial.view());
        if !finite(value, grad.view()) {
            return Err(if steepest {
                CappedMiss::Stop(non_finite())
            } else {
                CappedMiss::Retry
            });
        }
        crate::workspace::absorb(&mut self.reuse.grad, grad);
        Ok(value)
    }

    /// Euclidean FIRE, including Guénolé and ABC. Curved and preconditioned
    /// sessions stay on the allocating arm.
    fn step_fire_euclidean<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let cached = self.same_last_x(x);
        if !cached {
            let ev = obj.value_and_gradient(x.view());
            if !finite(ev.0, ev.1.view()) {
                if self.brief {
                    crate::workspace::absorb(&mut self.last_grad, ev.1);
                    self.last_value = ev.0;
                }
                return Err(non_finite());
            }
            crate::workspace::absorb(&mut self.last_grad, ev.1);
            self.last_value = ev.0;
        }
        let mut value = self.last_value;
        if !finite(value, self.last_grad.view()) {
            return Err(non_finite());
        }
        let gnorm = self.stationarity_norm(x, &self.last_grad);
        if gnorm < self.control.gtol {
            if self.brief {
                self.remember_pos(x, value);
            }
            return Ok(self.report(value, x, &self.last_grad));
        }
        self.saved_x.assign(x);
        self.fire_euclidean_move(obj, x, &mut value)?;
        let drifted = self.saved_x.iter().zip(x.iter()).any(|(s, xi)| {
            let d = *xi - *s;
            (*s + d - *xi).abs() > 1e-15
        });
        if drifted {
            let n = x.len().min(self.saved_x.len());
            for i in 0..n {
                let d = x[i] - self.saved_x[i];
                x[i] = self.saved_x[i] + d;
            }
            let ev = obj.value_and_gradient(x.view());
            value = ev.0;
            crate::workspace::absorb(&mut self.last_grad, ev.1);
        }
        self.remember_pos(x, value);
        self.steps += 1;
        Ok(self.report(value, x, &self.last_grad))
    }

    /// One flat FIRE step. `saved_x` is the point a refusal restores.
    fn fire_euclidean_move<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        value: &mut f64,
    ) -> Result<()>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let n = x.len();
        let steps_now = self.steps;
        for i in 0..n {
            self.scratch[i] = -self.last_grad[i];
        }
        let saved = {
            let Solver {
                scratch,
                reuse,
                inner,
                ..
            } = &mut *self;
            project_box_tangent(obj.bounds(), x, scratch);
            let Inner::Fire(state) = inner else {
                return Err(Error::FireStalled { steps: steps_now });
            };
            project_box_tangent(obj.bounds(), x, &mut state.vel);
            let saved = state.checkpoint();
            let prep = fire_prepare_v2_into(state, scratch, &mut reuse.corr);
            (saved, prep)
        };
        if saved.1 == Prepared::Rollback {
            for i in 0..n {
                x[i] += self.reuse.corr[i];
            }
            clip_mut(obj.bounds(), x);
            let ev = obj.value_and_gradient(x.view());
            if !finite(ev.0, ev.1.view()) {
                x.assign(&self.saved_x);
                if let Inner::Fire(state) = &mut self.inner {
                    state.restore(saved.0);
                }
                return Err(non_finite());
            }
            crate::workspace::absorb(&mut self.reuse.grad, ev.1);
            for i in 0..n {
                self.scratch[i] = -self.reuse.grad[i];
            }
            project_box_tangent(obj.bounds(), x, &mut self.scratch);
        } else if saved.1 == Prepared::Stalled {
            return Err(Error::FireStalled { steps: self.steps });
        }
        if let Inner::Fire(state) = &mut self.inner {
            fire_displacement_into(state, &self.scratch, &mut self.scratch_b);
            if state.abc
                && let Some(cap) = self.atom_maxmove.or(self.control.maxmove)
            {
                fire_clip_velocity(state, cap);
                let dt = state.dt;
                for i in 0..n {
                    self.scratch_b[i] = state.vel[i] * dt;
                }
            }
        }
        self.reuse.fwd.assign(x);
        let (abc, v2, dt, plain) = if let Inner::Fire(state) = &self.inner {
            (
                state.abc,
                matches!(state.kind, FireKind::V2),
                state.dt,
                state.preset() == FirePreset::Default && !state.abc,
            )
        } else {
            (false, false, 0.0, true)
        };
        for i in 0..n {
            self.reuse.trial[i] = x[i] + self.scratch_b[i];
        }
        if !plain {
            self.reuse.free.assign(&self.reuse.trial);
        }
        let (atom_cap, cap) = if abc {
            (None, None)
        } else {
            (self.atom_maxmove, self.control.maxmove)
        };
        if let Some(limit) = atom_cap {
            scale_step_atom(&self.reuse.fwd, &mut self.reuse.trial, limit);
        } else if let Some(limit) = cap {
            scale_step(&self.reuse.fwd, &mut self.reuse.trial, limit);
        }
        clip_mut(obj.bounds(), &mut self.reuse.trial);
        let moved = plain
            || self
                .reuse
                .trial
                .iter()
                .zip(self.reuse.free.iter())
                .any(|(a, b)| a != b);
        if v2
            && dt > 0.0
            && moved
            && let Inner::Fire(state) = &mut self.inner
        {
            for i in 0..n {
                state.vel[i] = (self.reuse.trial[i] - self.reuse.fwd[i]) / dt;
            }
        }
        let ev = obj.value_and_gradient(self.reuse.trial.view());
        if !finite(ev.0, ev.1.view()) {
            x.assign(&self.saved_x);
            if let Inner::Fire(state) = &mut self.inner {
                state.restore(saved.0);
            }
            return Err(non_finite());
        }
        x.assign(&self.reuse.trial);
        *value = ev.0;
        crate::workspace::absorb(&mut self.last_grad, ev.1);
        if let Inner::Fire(state) = &mut self.inner {
            project_box_tangent(obj.bounds(), x, &mut state.vel);
        }
        if matches!(
            self.inner,
            Inner::Fire(ref state) if matches!(state.kind, FireKind::V1)
        ) {
            for i in 0..n {
                self.scratch[i] = -self.last_grad[i];
            }
            project_box_tangent(obj.bounds(), x, &mut self.scratch);
            if let Inner::Fire(state) = &mut self.inner {
                fire_after_v1(state, &self.scratch);
            }
        }
        Ok(())
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
            return Ok(self.report(value, x, &grad));
        }
        unreachable!("PSO slot populated above")
    }
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
    /// Apply [`Control::gtol`] to a newly built L-BFGS session.
    pub fn with_gtol(mut self, gtol: f64) -> Self {
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.gtol = gtol;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use eindir_core::{Bounds, Gradient};
    use ndarray::array;

    use super::*;

    #[test]
    fn the_sphere_dogleg_model_matches_the_projected_basis() {
        let mut solver = Solver::new(Method::Dogleg, Control::default(), 3);
        solver.set_manifold(ManifoldKind::Sphere);
        let x = array![0.3, -1.2, 0.7];
        let hess = array![[2.0, 0.5, -1.0], [0.25, 3.0, 0.4], [-0.6, 1.5, 1.0]];
        solver.fill_tangent_model(&x, &hess);
        let basis = solver.tangent_hessian(&x, &hess);
        let gap = (&solver.tangent.model - &basis)
            .mapv(f64::abs)
            .fold(0.0, |a: f64, &b| a.max(b));
        assert!(
            gap <= 1e-14,
            "rank-one model is {gap:e} from the basis expansion"
        );
    }

    /// The dogleg point of an oblique kind at radii that reach the Newton
    /// point, the dogleg segment and the clipped Cauchy point: its tangent
    /// residual `|s - P s|` and the length of `P s`, the step the session
    /// retracts. At the first radius the point is the Newton point, and its
    /// model gradient `H s + g` must be orthogonal to every column of `P`.
    fn check_dogleg_point(
        kind: ManifoldKind,
        x: &Array1<f64>,
        hess: &Array2<f64>,
        grad: &Array1<f64>,
        radii: [f64; 3],
    ) {
        let mut solver = Solver::new(Method::Dogleg, Control::default(), x.len());
        solver.set_manifold(kind);
        for radius in radii {
            let s = solver.dogleg_point(x, hess, grad, radius);
            let residual = l2(&(&s - &kind.project(x, &s)));
            assert!(
                residual <= 1e-12,
                "{} at radius {radius}: tangent residual {residual:e}",
                kind.as_str()
            );
            let mut step = s;
            kind.project_in_place(x, &mut step);
            let len = l2(&step);
            // A point on the trust sphere meets the radius to rounding.
            assert!(
                len <= radius * (1.0 + 4.0 * f64::EPSILON),
                "{} at radius {radius}: projected step {len:e}",
                kind.as_str()
            );
        }
        let s = solver.dogleg_point(x, hess, grad, radii[0]);
        let slope = hess.dot(&s) + grad;
        let worst = solver
            .projection_columns(x)
            .columns()
            .into_iter()
            .map(|c| c.dot(&slope).abs())
            .fold(0.0, f64::max);
        assert!(
            worst <= 1e-12,
            "{}: the Newton point is {worst:e} from stationary on the tangent space",
            kind.as_str()
        );
    }

    #[test]
    fn multinomial_dogleg_point_is_tangent_within_the_radius() {
        let kind = ManifoldKind::Multinomial;
        let x = array![0.2, 0.3, 0.5];
        let hess = array![[2.0, 0.2, 0.1], [0.2, 3.0, 0.0], [0.1, 0.0, 4.0]];
        let grad = kind.project(&x, &array![1.0, -2.0, 0.5]);
        check_dogleg_point(kind, &x, &hess, &grad, [10.0, 0.86, 0.3]);
    }

    /// A symmetric positive definite 9 by 9 Hessian with a coupled band.
    fn banded_hessian() -> Array2<f64> {
        Array2::from_shape_fn((9, 9), |(i, j)| match i.abs_diff(j) {
            0 => 1.0 + 0.5 * i as f64,
            1 => 0.2,
            2 => -0.1,
            _ => 0.0,
        })
    }

    #[test]
    fn multinomial_ds_dogleg_point_is_tangent_within_the_radius() {
        let kind = ManifoldKind::multinomial_ds(3);
        let x = array![0.5, 0.3, 0.2, 0.2, 0.5, 0.3, 0.3, 0.2, 0.5];
        let v = array![1.0, -2.0, 0.5, 0.3, -0.7, 1.2, -0.4, 0.8, -1.1];
        let grad = kind.project(&x, &v);
        check_dogleg_point(kind, &x, &banded_hessian(), &grad, [10.0, 0.85, 0.3]);
    }

    #[test]
    fn multinomial_sym_dogleg_point_is_tangent_within_the_radius() {
        let kind = ManifoldKind::multinomial_sym(3);
        let x = array![0.5, 0.3, 0.2, 0.3, 0.4, 0.3, 0.2, 0.3, 0.5];
        let v = array![1.0, -2.0, 0.5, 0.3, -0.7, 1.2, -0.4, 0.8, -1.1];
        let grad = kind.project(&x, &v);
        check_dogleg_point(kind, &x, &banded_hessian(), &grad, [10.0, 0.8, 0.3]);
    }

    /// Bound on `|x . v| / |v|`, the bound of the sphere trial tests.
    /// One projection leaves a normal part of a few ulps, and the
    /// secant of a chord of length `l` has a normal part `l / 2`.
    const TANGENT: f64 = 1e-10;

    /// f(x) = -x_3.
    struct Lift(Bounds<f64>);

    impl Objective<f64> for Lift {
        fn dim(&self) -> usize {
            3
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.0
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            -x[2]
        }
    }

    impl Gradient<f64> for Lift {
        fn dim(&self) -> usize {
            3
        }
        fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
            array![0.0, 0.0, -1.0]
        }
    }

    impl DifferentiableObjective<f64> for Lift {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    /// The sphere walk of `tests/manifold_trial_point.rs`: twelve steps
    /// from (1, 0, 0) under a Euclidean cap of 0.1.
    #[test]
    fn fire2_velocity_is_tangent_after_a_curved_step() {
        let steps = 12;
        let control = Control {
            maxiter: steps,
            gtol: 1e-12,
            istep: 1.0,
            maxmove: Some(0.1),
            ..Default::default()
        };
        let lift = Lift(Bounds::new(
            Array1::from_elem(3, -1e6),
            Array1::from_elem(3, 1e6),
            0.0,
        ));
        let mut solver = Solver::new(Method::Fire { kind: FireKind::V2 }, control, 3);
        solver.set_manifold(ManifoldKind::Sphere);
        let mut x = array![1.0, 0.0, 0.0];
        let mut worst = 0.0_f64;
        for _ in 0..steps {
            solver.step(&lift, &mut x).unwrap();
            let Inner::Fire(state) = &solver.inner else {
                unreachable!()
            };
            let speed = l2(&state.vel);
            if speed > 0.0 {
                worst = worst.max(x.dot(&state.vel).abs() / speed);
            }
        }
        assert!(worst <= TANGENT, "|x . v| / |v| reached {worst:.4e}");
    }

    #[test]
    fn the_new_lbfgs_arms_refuse_a_fire_session_and_a_bad_ridge() {
        let mut fire = Solver::new(Method::Fire { kind: FireKind::V1 }, Control::default(), 3);
        assert!(fire.set_lbfgs_model_atom(true).is_err());
        assert!(fire.set_lbfgs_compact_tr(true).is_err());
        assert!(fire.set_lbfgs_skip_bad(true).is_err());
        assert!(fire.set_lindh(None).is_err());
        let mut lbfgs = Solver::new(Method::lbfgs(), Control::default(), 6);
        assert!(
            lbfgs
                .set_lindh(Some(LindhSpec {
                    z: vec![1],
                    c_stab: 0.1,
                    cell: [0.0; 3],
                }))
                .is_err()
        );
        assert!(
            lbfgs
                .set_lindh(Some(LindhSpec {
                    z: vec![1, 1],
                    c_stab: 0.0,
                    cell: [0.0; 3],
                }))
                .is_err()
        );
        lbfgs
            .set_lindh(Some(LindhSpec {
                z: vec![1, 8],
                c_stab: 0.1,
                cell: [0.0; 3],
            }))
            .unwrap();
        lbfgs.set_lbfgs_model_atom(true).unwrap();
        lbfgs.set_lbfgs_compact_tr(true).unwrap();
        lbfgs.set_lbfgs_skip_bad(true).unwrap();
    }
}
