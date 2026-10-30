//! Persistent solver session. One [`Solver::step`] is one outer iteration.

use std::collections::VecDeque;

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[cfg(feature = "highs")]
use crate::accept::accept_step_with_fallback;
use crate::accept::{Accept, accept_step};
use crate::adam::adam_direction;
use crate::bb::bb_direction;
use crate::control::{Control, StopNorm};
use crate::error::{Error, Result};
use crate::fire::{
    FireKind, FireMix, FirePreparation, FireState, fire_after_v1, fire_after_v1_metric,
    fire_displacement, fire_displacement_metric, fire_prepare_v2, project_box_tangent,
};
use crate::hvp::{
    FiniteDifference, HessianVector, NewtonCgRequest, OptionalAction, newton_cg_step,
    sym_eig_jacobi,
};
use crate::lbfgs::{GradNorm, Lbfgs};
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
    cap_displacement_atom, finite, l2, max_atom_displacement, non_finite,
    scale_displacement_to_atom_cap, scale_step, scale_step_atom,
};
use crate::trust::{
    accept_ratio, dogleg_direction, predicted_reduction, reduction_ratio, update_radius,
};

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
    /// Objective gradients. A finite-difference product is an
    /// [`Self::hvp_calls`] entry, not a gradient.
    gradient_calls: usize,
    /// Packwood exponential metric. `None` leaves every method unchanged.
    exp: Option<ExpState>,
    /// Analytic or finite-difference Hessian-vector products.
    hvp_calls: usize,
    /// Negative-curvature kicks already taken in this solve.
    saddle_kicks: u8,
    stop_norm: StopNorm,
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
            gradient_calls: 0,
            hvp_calls: 0,
            exp: None,
            saddle_kicks: 0,
            stop_norm: StopNorm::Two,
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
    /// gradient. [`StopNorm::Two`] is the default.
    pub fn set_stop_norm(&mut self, norm: StopNorm) {
        self.stop_norm = norm;
    }

    /// Norm selected by [`Self::set_stop_norm`].
    pub fn stop_norm(&self) -> StopNorm {
        self.stop_norm
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
    pub fn set_atom_maxmove(&mut self, maxmove: f64) {
        self.atom_maxmove = if maxmove > 0.0 { Some(maxmove) } else { None };
        #[cfg(feature = "highs")]
        self.apply_highs_opts();
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

    /// Embedded manifold for project / retract / transport.
    pub fn set_manifold(&mut self, kind: ManifoldKind) {
        if kind != self.manifold {
            self.forget();
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
            center_axes: if self.project_rigid && self.dim % 3 == 0 {
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
        self.last_pos = Some(x.clone());
        self.last_value = value;
        self.last_grad = grad.clone();
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
                if !x[k].is_finite() || !grad[k].is_finite() || !(lo <= x[k] && x[k] <= hi) {
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
        Report {
            value,
            coords: x.clone(),
            steps: self.steps,
            grad_norm: self.stationarity_norm(x, grad),
            grad: Some(grad.clone()),
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
        self.e_hist.clear();
        self.last_pos = None;
        self.geometry_lbfgs_stiff = None;
    }

    /// Drop method memory. The next step is a cold start from the current `x`.
    pub fn forget(&mut self) {
        self.istep = self.control.istep;
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
                *radius = self.control.istep.max(1e-8);
            }
        }
    }

    /// One outer iteration. `x` is the iterate and is overwritten in place.
    /// A first-order step whose start, or whose FIRE or geometry fallback
    /// trial, answers with a non-finite value or gradient returns
    /// [`Error::Oracle`] with `x` and the solver as they were, so a retry
    /// probes the same trial.
    pub fn step<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
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
            *x = obj.bounds().clip(x.view());
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
        let rmax = self.control.maxmove.unwrap_or(radius * 8.0).max(radius);
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
        if let Inner::Lbfgs(solver) = &self.inner {
            if let Some(pre) = solver.exp_apply(force.view()) {
                return pre;
            }
        }
        h0 * force
    }

    /// `scale * P^{-1} force` when an exponential preconditioner is installed.
    /// Otherwise `scale * force`.
    fn geometry_scaled_force(&self, force: &Array1<f64>, scale: f64) -> Array1<f64> {
        if let Inner::Lbfgs(solver) = &self.inner {
            if let Some(pre) = solver.exp_apply(force.view()) {
                return scale * &pre;
            }
        }
        scale * force
    }

    /// Per-atom cap selects the geometry L-BFGS step. Uncapped sessions keep
    /// the one-oracle two-loop (and Wolfe when acceptance is not [`Accept::None`]).
    fn use_geometry_lbfgs(&self) -> bool {
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
            trial = obj.bounds().clip((&*x + alpha * &direction).view());
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
            trial = obj.bounds().clip((&*x + &sd).view());
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
            let found = lowest_eigenpair(&Mat(hess), seed.view(), seed.view(), 48.min(n));
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
            *x = obj.bounds().clip(x.view());
        }
        let (mut value, mut grad) = if cached {
            (self.last_value, self.last_grad.clone())
        } else {
            obj.value_and_gradient(x.view())
        };
        grad = self.horizontal_grad(x, &grad);
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
            let center = if self.project_rigid && self.dim % 3 == 0 {
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
                &self.control,
                self.accept,
                &mut self.e_hist,
                None,
                self.manifold,
                false,
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
            &self.control,
            self.accept,
            &mut self.e_hist,
            self.atom_maxmove,
            self.manifold,
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
        let rmax = self.control.maxmove.unwrap_or(radius * 8.0).max(radius);
        let mut dir = dogleg_direction(hess, &grad, radius);
        if self.project_rigid {
            project_out_rot_trans(&mut dir, x.view());
        }
        let trust_norm = l2(&dir);
        let mut trial = &*x + &dir;
        if let Some(cap) = self.atom_maxmove {
            scale_step_atom(x, &mut trial, cap);
        } else if let Some(cap) = self.control.maxmove {
            scale_step(x, &mut trial, cap);
        }
        trial = obj.bounds().clip(trial.view());
        let p = &trial - &*x;
        let taken = l2(&p);
        let (ft, gt) = obj.value_and_gradient(trial.view());
        let pred = predicted_reduction(hess, &grad, &p);
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
        if !finite(value, grad.view()) {
            return Err(non_finite());
        }
        let gnorm = self.stationarity_norm(x, &grad);
        if gnorm < self.control.gtol {
            return Ok(self.report(value, x, &grad));
        }
        self.refresh_exp(obj, x, &grad)?;

        if matches!(self.manifold, ManifoldKind::Sphere | ManifoldKind::Stiefel)
            && matches!(self.inner, Inner::Lbfgs(_))
            && self.accept != Accept::None
        {
            if !cached {
                if let Some(previous) = &self.last_pos {
                    let (s, y) = self.lbfgs_sy(previous, x, &self.last_grad, &grad);
                    if let Inner::Lbfgs(solver) = &mut self.inner {
                        solver.transport(|v| crate::manifold::Sphere.transport(previous, x, v));
                        solver.push_pair(s, y, Some(l2(&grad)));
                    }
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
        if !cached {
            if let Some(previous) = &self.last_pos {
                let (s, y) = self.lbfgs_sy(previous, x, &self.last_grad, &grad);
                if let Inner::Lbfgs(solver) = &mut self.inner {
                    solver.push_pair(s, y, Some(l2(&grad)));
                }
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
        let gold = grad.clone();
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
                    let (npos, nval, ngrad, moved) = accept_step(
                        obj,
                        x,
                        value,
                        &gold,
                        &dir,
                        &self.control,
                        self.accept,
                        &mut self.e_hist,
                        self.atom_maxmove,
                        self.manifold,
                    );
                    if !moved {
                        return Err(Error::Oracle {
                            what: "non-finite value or gradient",
                        });
                    }
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
                    &self.control,
                    self.accept,
                    &mut self.e_hist,
                    self.atom_maxmove,
                    self.manifold,
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
                    &self.control,
                    self.accept,
                    &mut self.e_hist,
                    self.atom_maxmove,
                    self.manifold,
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
                    &self.control,
                    self.accept,
                    &mut self.e_hist,
                    self.atom_maxmove,
                    self.manifold,
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
                    &self.control,
                    self.accept,
                    &mut self.e_hist,
                    self.atom_maxmove,
                    self.manifold,
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
                let mut direction = solve_dense(b, &rhs).unwrap_or_else(|| rhs);
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
                    &self.control,
                    self.accept,
                    &mut self.e_hist,
                    self.atom_maxmove,
                    self.manifold,
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
                let dir = adam_direction(m, v, &grad, *beta1, *beta2, *b1p, *b2p, *eps);
                let dir = dir.mapv(|d| d * self.istep);
                let (npos, nval, ngrad, _) = accept_step(
                    obj,
                    x,
                    value,
                    &grad,
                    &dir,
                    &self.control,
                    self.accept,
                    &mut self.e_hist,
                    self.atom_maxmove,
                    self.manifold,
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
                let saved = state.clone();
                let mut raw = grad.mapv(|g| -g);
                project_box_tangent(obj.bounds(), x, &mut raw);
                project_box_tangent(obj.bounds(), x, &mut state.vel);
                match fire_prepare_v2(state, &raw) {
                    FirePreparation::Ready => {}
                    FirePreparation::Rollback(correction) => {
                        *x = obj.bounds().clip((&*x + &correction).view());
                        let ev = obj.value_and_gradient(x.view());
                        if !finite(ev.0, ev.1.view()) {
                            *state = saved;
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
                let mut trial = &*x + &dx;
                if let Some(cap) = self.atom_maxmove {
                    scale_step_atom(x, &mut trial, cap);
                } else if let Some(cap) = self.control.maxmove {
                    scale_step(x, &mut trial, cap);
                }
                trial = obj.bounds().clip(trial.view());
                if matches!(state.kind, FireKind::V2) && state.dt > 0.0 {
                    state.vel = (&trial - &forward_start) / state.dt;
                }
                let ev = obj.value_and_gradient(trial.view());
                if !finite(ev.0, ev.1.view()) {
                    *state = saved;
                    *x = start;
                    return Err(non_finite());
                }
                *x = trial;
                value = ev.0;
                grad = ev.1;
                let mut raw_new = grad.mapv(|g| -g);
                project_box_tangent(obj.bounds(), x, &mut raw_new);
                project_box_tangent(obj.bounds(), x, &mut state.vel);
                if self.exp.is_some() {
                    let mut steep_new = metric_vector(&self.exp, &raw_new);
                    project_box_tangent(obj.bounds(), x, &mut steep_new);
                    let mix_new = fire_mix(&self.exp, &state.vel, &raw_new, &steep_new);
                    fire_after_v1_metric(state, &raw_new, &steep_new, mix_new);
                } else {
                    fire_after_v1(state, &raw_new);
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
                    &self.control,
                    self.accept,
                    &mut self.e_hist,
                    self.atom_maxmove,
                    self.manifold,
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

        let delta = &*x - &start;
        let y = self.manifold.retract(&start, &delta);
        if y.iter().zip(x.iter()).any(|(a, b)| (*a - *b).abs() > 1e-15) {
            *x = y;
            let ev = obj.value_and_gradient(x.view());
            value = ev.0;
            grad = ev.1;
        }
        grad = self.horizontal_grad(x, &grad);

        let pair = if matches!(self.inner, Inner::Lbfgs(_))
            && x.iter().zip(start.iter()).any(|(a, b)| a != b)
        {
            let (s, y) = self.lbfgs_sy(&start, x, &gold, &grad);
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
                *inertia,
                *c1,
                *c2,
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
                radius: istep.max(1e-8),
            },
            Method::NewtonCg => Inner::NewtonCg {
                radius: istep.max(1e-8),
            },
            Method::NewtonPcg => Inner::NewtonPcg {
                radius: istep.max(1e-8),
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
