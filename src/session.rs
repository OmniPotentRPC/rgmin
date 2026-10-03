//! Persistent solver session. One [`Solver::step`] is one outer iteration.

use std::collections::VecDeque;

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::accept::{Accept, accept_step};
use crate::adam::adam_direction;
use crate::bb::bb_direction;
use crate::control::Control;
use crate::error::{Error, Result};
use crate::fire::{
    Fire2Extras, FireState, FireVariant, fire_after_v1, fire_displacement, fire_rescale_velocity,
    fire2_displacement, guenole2020,
};
use crate::lbfgs::{GradNorm, Lbfgs};
use crate::linesearch::LineSearch;
use crate::manifold::{Manifold, ManifoldKind};
use crate::method::Method;
use crate::newton::{HessianObjective, NewtonKind, rfo_direction, shifted_newton};
use crate::nlcg::{Conjugacy, ConjugacyContext, Restart};
use crate::pso::{Particle, RNG_SEED, random_velocity, update_swarm};
use crate::qn::{bfgs_inverse_update, solve_dense, sr1_inverse_update, sr2_hessian_update};
use crate::qn_step::QnStep;
use crate::report::Report;
use crate::rigid::{project_horizontal, project_out_rot_trans};
use crate::step::{l2, next_istep, qn_istep, scale_step, scale_step_atom, take_step};
use crate::trust::{
    accept_ratio, dogleg_direction, predicted_reduction, reduction_ratio, update_radius,
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

/// Long-lived solver. Algorithm memory lives here; `x` stays with the caller.
pub struct Solver {
    dim: usize,
    control: Control,
    linesearch: LineSearch,
    istep: f64,
    steps: usize,
    qn_step: QnStep,
    accept: Accept,
    e_hist: VecDeque<f64>,
    atom_maxmove: Option<f64>,
    project_rigid: bool,
    /// Sella `proj_rot`: false under PBC. Rotation is not a symmetry of the cell.
    periodic: bool,
    manifold: ManifoldKind,
    /// Per-atom masses for [`ManifoldKind::MwRigid`]. Length N, not 3N.
    masses: Option<Array1<f64>>,
    #[cfg(feature = "highs")]
    highs: bool,
    last_pos: Option<Array1<f64>>,
    last_value: f64,
    last_grad: Array1<f64>,
    inner: Inner,
}

#[allow(clippy::large_enum_variant)]
enum Inner {
    Lbfgs(Lbfgs),
    Nlcg {
        conjugacy: Conjugacy,
        restart: Restart,
        dir: Array1<f64>,
        g_old: Array1<f64>,
        d_old: Array1<f64>,
        initialized: bool,
    },
    Bfgs {
        h: Array2<f64>,
    },
    Sr1 {
        h: Array2<f64>,
    },
    Sr2 {
        b: Array2<f64>,
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
    Fire(FireState, Option<Fire2Extras>),
    Bb {
        prev_s: Option<Array1<f64>>,
        prev_y: Option<Array1<f64>>,
    },
    Dogleg {
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
        Self {
            dim,
            control,
            linesearch: SESSION_LINESEARCH,
            istep,
            steps: 0,
            qn_step: QnStep::TwoLoop,
            accept: Accept::None,
            e_hist: VecDeque::new(),
            atom_maxmove: None,
            project_rigid: false,
            periodic: false,
            manifold: ManifoldKind::Euclidean,
            masses: None,
            #[cfg(feature = "highs")]
            highs: false,
            last_pos: None,
            last_value: 0.0,
            last_grad: Array1::zeros(dim),
            inner,
        }
    }

    /// eOn `lbfgs_step`. Newton / RFO need a Hessian on [`Self::step_hess`].
    pub fn set_qn_step(&mut self, step: QnStep) {
        self.qn_step = step;
    }

    /// eOn `lbfgs_accept`. Default is take the clipped step.
    pub fn set_accept(&mut self, accept: Accept) {
        self.accept = accept;
    }

    /// Line search for the line-searched arms (steepest descent, NLCG,
    /// BFGS, SR1, SR2, Adam, and L-BFGS under [`Accept::Energy`]) on the
    /// next [`Self::step`]. Default is strong Wolfe (`c1 = 1e-4`,
    /// `c2 = 0.9`, 20 trials), not [`LineSearch::default`] (Brent).
    pub fn set_linesearch(&mut self, linesearch: LineSearch) {
        self.linesearch = linesearch;
    }

    /// Euclidean cap applied on the next [`Self::step`].
    pub fn set_maxmove(&mut self, maxmove: f64) {
        self.control.maxmove = if maxmove > 0.0 { Some(maxmove) } else { None };
    }

    /// eOn `maxAtomMotionAppliedV`. Preferred over the Euclidean cap.
    pub fn set_atom_maxmove(&mut self, maxmove: f64) {
        self.atom_maxmove = if maxmove > 0.0 { Some(maxmove) } else { None };
    }

    /// eOn `lbfgs_project_rigid`. Isolated clusters only.
    pub fn set_project_rigid(&mut self, enabled: bool) {
        self.project_rigid = enabled;
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

    /// Per-atom masses for [`ManifoldKind::MwRigid`] (Page–McIver / Eckart).
    /// Empty clears them (unit mass).
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

    /// HiGHS feasible-set step. No-op unless this build has `highs`.
    pub fn set_highs(&mut self, enabled: bool) {
        #[cfg(not(feature = "highs"))]
        let _ = enabled;
        #[cfg(feature = "highs")]
        {
            self.highs = enabled;
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.highs = if enabled {
                    Some(crate::HighsStep {
                        trust: self.atom_maxmove.or(self.control.maxmove),
                        lo: None,
                        hi: None,
                        equalities: Vec::new(),
                        center_axes: if self.project_rigid && self.dim.is_multiple_of(3) {
                            Some((self.dim / 3, 3))
                        } else {
                            None
                        },
                    })
                } else {
                    None
                };
            }
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
        let mut g = self.project_vec(x, grad);
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
    /// the published FIRE 2.0 ([`crate::fire::fire2_displacement`]) and its
    /// table 2 parameters, from `dt = Control::istep`;
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
        if let Inner::Fire(state, ext) = &mut self.inner {
            match variant {
                FireVariant::Guenole2020 => {
                    let (s, e) = guenole2020(dim, dt0);
                    *state = s;
                    *ext = Some(e);
                }
                _ => {
                    *state = FireState::new(state.kind, dim, dt0);
                    *ext = None;
                }
            }
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
        self.last_pos = None;
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

    /// Keep method memory and discard the cached point and acceptance window.
    /// The next step evaluates the current objective at the supplied point
    /// with the initial step scale and the retained curvature information.
    pub fn rebase(&mut self) {
        self.istep = self.control.istep;
        self.e_hist.clear();
        self.last_pos = None;
    }

    /// Drop method memory. The next step is a cold start from the current `x`.
    pub fn forget(&mut self) {
        self.istep = self.control.istep;
        self.e_hist.clear();
        self.last_pos = None;
        match &mut self.inner {
            Inner::Lbfgs(solver) => solver.forget(),
            Inner::Nlcg { initialized, .. } => *initialized = false,
            Inner::Bfgs { h } => *h = Array2::<f64>::eye(self.dim),
            Inner::Sr1 { h } => *h = Array2::<f64>::eye(self.dim),
            Inner::Sr2 { b } => *b = Array2::<f64>::eye(self.dim),
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
            Inner::Fire(state, ext) => {
                state.reset();
                if let Some(ext) = ext {
                    ext.steps = 0;
                }
            }
            Inner::Bb { prev_s, prev_y } => {
                *prev_s = None;
                *prev_y = None;
            }
            Inner::Dogleg { radius } => {
                *radius = self.control.istep.max(1e-8);
            }
        }
    }

    /// One outer iteration. `x` is the iterate and is overwritten in place.
    pub fn step<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if matches!(self.inner, Inner::Newton { .. } | Inner::Dogleg { .. }) {
            return Err(Error::NeedHessian);
        }
        self.step_first_order(obj, x)
    }

    /// One Newton / RFO iteration. Hessian is rebuilt at the current `x`.
    pub fn step_hess<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
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
        let gnorm = l2(&grad);
        if gnorm < self.control.gtol {
            return Ok(Report {
                value,
                coords: x.clone(),
                steps: self.steps,
                grad_norm: gnorm,
            });
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
            if let Ok(dir) = crate::lbfgs_qp::highs_feasible_step(
                None,
                Some(&hess),
                &grad,
                self.atom_maxmove,
                self.control.maxmove,
                center,
            ) {
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
                    None,
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
                return Ok(Report {
                    value,
                    coords: x.clone(),
                    steps: self.steps,
                    grad_norm: l2(&grad),
                });
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
        Ok(Report {
            value,
            coords: x.clone(),
            steps: self.steps,
            grad_norm: l2(&grad),
        })
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
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let radius = match &self.inner {
            Inner::Dogleg { radius } => *radius,
            _ => self.control.istep.max(1e-8),
        };
        let rmax = self
            .atom_maxmove
            .or(self.control.maxmove)
            .unwrap_or(radius * 8.0)
            .max(radius);
        let mut dir = dogleg_direction(hess, &grad, radius);
        if self.project_rigid {
            project_out_rot_trans(&mut dir, x.view());
        }
        let mut trial = &*x + &dir;
        if let Some(cap) = self.atom_maxmove {
            scale_step_atom(x, &mut trial, cap);
        } else if let Some(cap) = self.control.maxmove {
            scale_step(x, &mut trial, cap);
        }
        trial = obj.bounds().clip(trial.view());
        let p = &trial - &*x;
        let pnorm = l2(&p);
        let (ft, gt) = obj.value_and_gradient(trial.view());
        let pred = predicted_reduction(hess, &grad, &p);
        let rho = reduction_ratio(value - ft, pred);
        if let Inner::Dogleg { radius } = &mut self.inner {
            *radius = update_radius(*radius, rho, pnorm, rmax);
        }
        if accept_ratio(rho) {
            *x = trial;
            self.remember(x, ft, &gt);
            self.steps += 1;
            Ok(Report {
                value: ft,
                coords: x.clone(),
                steps: self.steps,
                grad_norm: l2(&gt),
            })
        } else {
            self.remember(x, value, &grad);
            self.steps += 1;
            Ok(Report {
                value,
                coords: x.clone(),
                steps: self.steps,
                grad_norm: l2(&grad),
            })
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
        let gnorm = l2(&grad);
        if gnorm < self.control.gtol {
            return Ok(Report {
                value,
                coords: x.clone(),
                steps: self.steps,
                grad_norm: gnorm,
            });
        }

        let start = x.clone();
        let gold = grad.clone();
        // Accept::Step: the two-loop direction goes through accept_step
        // like the BB arm and the Hessian path in step_hess, one oracle
        // call and the clipped step with no energy test. The oracle value
        // of a projected NEB force is not the potential of the gradient it
        // returns, so a decrease test there says nothing. Every other
        // Accept keeps the line-searched step_objective path.
        let lbfgs_direct = match (&self.inner, self.accept) {
            (Inner::Lbfgs(solver), Accept::Step) => Some(solver.direction(grad.view())),
            _ => None,
        };
        let lbfgs_line_searched = lbfgs_direct.is_none() && matches!(self.inner, Inner::Lbfgs(_));
        if let Some(mut dir) = lbfgs_direct {
            if self.project_rigid {
                project_out_rot_trans(&mut dir, x.view());
            }
            dir = self.project_vec(x, &dir);
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
        }
        match &mut self.inner {
            Inner::Lbfgs(_) if !lbfgs_line_searched => {}
            Inner::Lbfgs(solver) => {
                solver.step_objective(
                    obj,
                    x,
                    &mut value,
                    &mut grad,
                    &mut self.istep,
                    self.linesearch,
                    &self.control,
                    self.atom_maxmove,
                );
            }
            Inner::Steepest => {
                let dir = grad.mapv(|g| -g);
                let t = take_step(
                    obj,
                    x,
                    value,
                    &grad,
                    dir.view(),
                    self.istep,
                    self.linesearch,
                    &self.control,
                    self.atom_maxmove,
                );
                *x = t.x;
                value = t.f;
                grad = t.g;
                self.istep = qn_istep(&self.control);
            }
            Inner::Nlcg {
                conjugacy,
                restart,
                dir,
                g_old,
                d_old,
                initialized,
            } => {
                if !*initialized {
                    *dir = grad.mapv(|g| -g);
                    *g_old = grad.clone();
                    *d_old = dir.clone();
                    *initialized = true;
                }
                let t = take_step(
                    obj,
                    x,
                    value,
                    &grad,
                    dir.view(),
                    self.istep,
                    self.linesearch,
                    &self.control,
                    self.atom_maxmove,
                );
                *x = t.x;
                value = t.f;
                grad = t.g;
                let ctx = ConjugacyContext {
                    current_gradient: grad.view(),
                    previous_gradient: g_old.view(),
                    previous_direction: d_old.view(),
                };
                let mut beta = conjugacy.beta(&ctx);
                if restart.should_restart(&ctx) {
                    beta = 0.0;
                }
                *dir = Array1::from_iter(grad.iter().zip(d_old.iter()).map(|(g, d)| -g + beta * d));
                g_old.assign(&grad);
                d_old.assign(dir);
                self.istep = qn_istep(&self.control);
            }
            Inner::Bfgs { h } => {
                let direction = -h.dot(&grad);
                let old = x.clone();
                let gold = grad.clone();
                let t = take_step(
                    obj,
                    x,
                    value,
                    &grad,
                    direction.view(),
                    self.istep,
                    self.linesearch,
                    &self.control,
                    self.atom_maxmove,
                );
                let moved = t.moved;
                *x = t.x;
                value = t.f;
                grad = t.g;
                if moved {
                    bfgs_inverse_update(h, &(&*x - &old), &(&grad - &gold));
                }
                self.istep = qn_istep(&self.control);
            }
            Inner::Sr1 { h } => {
                let direction = -h.dot(&grad);
                let old = x.clone();
                let gold = grad.clone();
                let t = take_step(
                    obj,
                    x,
                    value,
                    &grad,
                    direction.view(),
                    self.istep,
                    self.linesearch,
                    &self.control,
                    self.atom_maxmove,
                );
                let lsstep = t.alpha;
                let moved = t.moved;
                *x = t.x;
                value = t.f;
                grad = t.g;
                if moved {
                    sr1_inverse_update(h, &(&*x - &old), &(&grad - &gold));
                }
                self.istep = next_istep(lsstep, &self.control);
            }
            Inner::Sr2 { b } => {
                let rhs = grad.mapv(|g| -g);
                let direction = solve_dense(b, &rhs).unwrap_or(rhs);
                let old = x.clone();
                let gold = grad.clone();
                let t = take_step(
                    obj,
                    x,
                    value,
                    &grad,
                    direction.view(),
                    self.istep,
                    self.linesearch,
                    &self.control,
                    self.atom_maxmove,
                );
                let lsstep = t.alpha;
                let moved = t.moved;
                *x = t.x;
                value = t.f;
                grad = t.g;
                if moved {
                    sr2_hessian_update(b, &(&*x - &old), &(&grad - &gold));
                }
                self.istep = next_istep(lsstep, &self.control);
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
                let t = take_step(
                    obj,
                    x,
                    value,
                    &grad,
                    dir.view(),
                    self.istep,
                    self.linesearch,
                    &self.control,
                    self.atom_maxmove,
                );
                let lsstep = t.alpha;
                *x = t.x;
                value = t.f;
                grad = t.g;
                *b1p *= *beta1;
                *b2p *= *beta2;
                self.istep = next_istep(lsstep, &self.control);
            }
            Inner::Fire(state, ext) => {
                let force = grad.mapv(|g| -g);
                let dx = match ext {
                    Some(ext) => fire2_displacement(state, ext, &force),
                    None => fire_displacement(state, &force),
                };
                let mut trial = &*x + &dx;
                if let Some(cap) = self.atom_maxmove {
                    scale_step_atom(x, &mut trial, cap);
                } else if let Some(cap) = self.control.maxmove {
                    scale_step(x, &mut trial, cap);
                }
                trial = obj.bounds().clip(trial.view());
                // The clamp and clip shorten the move; the velocity has to
                // describe the move taken, or the power adapt reads an
                // integral of moves that never happened.
                fire_rescale_velocity(state, &dx, &(&trial - &*x));
                *x = trial;
                let ev = obj.value_and_gradient(x.view());
                value = ev.0;
                grad = ev.1;
                let force_new = grad.mapv(|g| -g);
                fire_after_v1(state, &force_new);
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
                }
            }
            Inner::Pso { .. } | Inner::Newton { .. } | Inner::Dogleg { .. } => unreachable!(),
        }

        // A translation retract maps the accepted point to itself; the
        // round trip start + (x - start) only adds rounding (one ulp of
        // a 30 Angstrom coordinate is 3.6e-15) and would buy a second
        // oracle call at the same geometry.
        let y = if self.manifold.retract_is_translation() {
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
            if lbfgs_line_searched {
                // step_objective pushes the Euclidean pair; the transported
                // pair from the retracted point replaces it.
                solver.replace_newest(s, y, Some(gn));
            } else {
                solver.push_pair(s, y, Some(gn));
            }
        }

        self.remember(x, value, &grad);
        self.steps += 1;
        Ok(Report {
            value,
            coords: x.clone(),
            steps: self.steps,
            grad_norm: l2(&grad),
        })
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
            return Ok(Report {
                value,
                coords: x.clone(),
                steps: self.steps,
                grad_norm: l2(&grad),
            });
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
                d_old: Array1::zeros(dim),
                initialized: false,
            },
            Method::Bfgs => Inner::Bfgs {
                h: Array2::<f64>::eye(dim),
            },
            Method::Sr1 => Inner::Sr1 {
                h: Array2::<f64>::eye(dim),
            },
            Method::Sr2 => Inner::Sr2 {
                b: Array2::<f64>::eye(dim),
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
            Method::Fire { kind } => Inner::Fire(FireState::new(*kind, dim, istep), None),
            Method::Bb => Inner::Bb {
                prev_s: None,
                prev_y: None,
            },
            Method::Dogleg => Inner::Dogleg {
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
            },
            2,
        )
    }

    fn fire_state(solver: &Solver) -> &FireState {
        match &solver.inner {
            Inner::Fire(state, _) => state,
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
                Inner::Fire(f, _) => l2(&f.vel),
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
            // the step's trial under Accept::Step.
            assert_eq!(obj.calls.load(Ordering::Relaxed) - before, 2, "{method:?}");
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
