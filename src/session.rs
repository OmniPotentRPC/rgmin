//! Persistent solver session. One [`Solver::step`] is one outer iteration.

use std::collections::VecDeque;

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::accept::{Accept, accept_step};
use crate::adam::adam_direction;
use crate::bb::bb_direction;
use crate::control::Control;
use crate::error::{Error, Result};
use crate::fire::{FireState, fire_after_v1, fire_displacement};
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
use crate::step::{l2, scale_step, scale_step_atom};
use crate::trust::{
    accept_ratio, dogleg_direction, predicted_reduction, reduction_ratio, update_radius,
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
    Fire(FireState),
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
            linesearch: LineSearch::default(),
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
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.highs = if enabled {
                    Some(crate::HighsStep {
                        trust: self.atom_maxmove.or(self.control.maxmove),
                        lo: None,
                        hi: None,
                        equalities: Vec::new(),
                        center_axes: if self.project_rigid && self.dim % 3 == 0 {
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
            Inner::Fire(state) => state.reset(),
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
            let center = if self.project_rigid && self.dim % 3 == 0 {
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
        match &mut self.inner {
            Inner::Lbfgs(solver) => {
                if self.accept == Accept::None {
                    let dir = solver.direction(grad.view());
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
                    *dir = grad.mapv(|g| -g * self.istep);
                    *g_old = grad.clone();
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
                    let mut beta = conjugacy.beta(&ctx);
                    if restart.should_restart(&ctx) {
                        beta = 0.0;
                    }
                    *dir = Array1::from_iter(
                        grad.iter()
                            .zip(d_old.iter())
                            .map(|(g, d)| -g * self.istep + beta * d),
                    );
                    g_old.assign(&grad);
                    d_old.assign(dir);
                }
            }
            Inner::Bfgs { h } => {
                let direction = -h.dot(&grad);
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
                    bfgs_inverse_update(h, &(&*x - &old), &(&grad - &gold));
                }
            }
            Inner::Sr1 { h } => {
                let direction = -h.dot(&grad);
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
                    sr1_inverse_update(h, &(&*x - &old), &(&grad - &gold));
                }
            }
            Inner::Sr2 { b } => {
                let rhs = grad.mapv(|g| -g);
                let direction = solve_dense(b, &rhs).unwrap_or_else(|| rhs);
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
                    sr2_hessian_update(b, &(&*x - &old), &(&grad - &gold));
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
                let force = grad.mapv(|g| -g);
                let dx = fire_displacement(state, &force);
                let mut trial = &*x + &dx;
                if let Some(cap) = self.atom_maxmove {
                    scale_step_atom(x, &mut trial, cap);
                } else if let Some(cap) = self.control.maxmove {
                    scale_step(x, &mut trial, cap);
                }
                trial = obj.bounds().clip(trial.view());
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
            Method::Fire { kind } => Inner::Fire(FireState::new(*kind, dim, istep)),
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
