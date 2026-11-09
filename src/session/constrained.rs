//! Steps on a holonomic distance constraint set.
//!
//! Every method runs in the mass-weighted coordinates `q = M^(1/2) x` of
//! the set's masses. There the inverse-mass projector `Pi` becomes the
//! orthogonal projector `M^(1/2) Pi M^(-1/2)`, a gradient `g` becomes
//! `M^(-1/2) Pi^T g`, and the unconstrained algebra of each method
//! applies as it stands; unit masses leave `q = x`. Stored vectors (the
//! L-BFGS pairs, the CG direction, the FIRE velocity, the BB pair) are
//! moved to each new point by that projector, the projection vector
//! transport of an embedded submanifold (Absil, Mahony and Sepulchre
//! 2008, section 8.1.3). A trial returns to the set along the bonds of
//! the point it leaves, so a line search runs on the curve
//! `R(t) = constrain(x + t d)` with slope `g . R'(t)`.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::{Bounds, DifferentiableObjective, Objective};
use ndarray::{Array1, Array2, ArrayView1, Axis};

use super::{Inner, RADIUS_GROWTH, RADIUS_START, SADDLE_KICKS, Solver};
use crate::accept::{Accept, ENERGY_RISE, FALLBACK_STEP, HALVINGS, push_energy};
use crate::adam::adam_direction;
use crate::bb::bb_direction;
use crate::error::{Error, Result};
use crate::fire::{
    FireKind, FirePreparation, FirePreset, FireState, fire_after_v1, fire_displacement,
    fire_prepare_v2, project_box_tangent,
};
use crate::holonomic::{ConstraintError, DistanceConstraints, Tangent, unsupported};
use crate::hvp::{
    ETA_ACCEPT, FiniteDifference, HessianVector, NYSTROM_RANK, NystromPrecond, OptionalAction,
    RADIUS_FLOOR, forward_difference, steihaug_cg, steihaug_pcg, steps_differ, sym_eig_jacobi,
};
use crate::lbfgs::Curvature;
use crate::linesearch::{LineSearch, Path};
use crate::manifold::ManifoldKind;
use crate::newton::{HessianObjective, NewtonKind, rfo_direction, shifted_newton};
use crate::nlcg::ConjugacyContext;
use crate::qn::{
    bfgs_inverse_update, secant_scale, solve_dense, sr1_inverse_update, sr2_hessian_update,
};
use crate::qn_step::QnStep;
use crate::report::Report;
use crate::rigid::project_horizontal;
use crate::step::{
    finite, l2, max_atom_motion, next_istep, non_finite, scale_displacement_to_atom_cap,
    scale_step, scale_step_atom,
};
use crate::trust::{
    accept_ratio, dogleg_direction, predicted_reduction, reduction_ratio, update_radius,
};

/// Displacement caps an unconstrained trial would take.
#[derive(Clone, Copy)]
struct Caps {
    atom: Option<f64>,
    euclid: Option<f64>,
}

impl Caps {
    fn apply(self, origin: &Array1<f64>, trial: &mut Array1<f64>) {
        if let Some(cap) = self.atom {
            scale_step_atom(origin, trial, cap);
        } else if let Some(cap) = self.euclid {
            scale_step(origin, trial, cap);
        }
    }
}

/// The set at one point, with the mass weights of `q`.
struct Space<'a> {
    tangent: Tangent<'a>,
    masses: Option<&'a [f64]>,
    root_m: Array1<f64>,
    root_w: Array1<f64>,
    /// Coordinates of constrained atoms.
    bonded: Vec<bool>,
    rigid: bool,
}

/// Coordinates of the atoms that appear in a pair.
fn bonded(set: &DistanceConstraints, n: usize) -> Vec<bool> {
    let mut bonded = vec![false; n];
    for a in set.constrained_atoms() {
        bonded[3 * a..3 * a + 3].fill(true);
    }
    bonded
}

/// Clip the free coordinates of `x` to `bounds`.
fn clip_free(bonded: &[bool], bounds: &Bounds<f64>, x: &mut Array1<f64>) {
    for (k, v) in x.iter_mut().enumerate().filter(|(k, _)| !bonded[*k]) {
        *v = v.max(bounds.low[k]).min(bounds.high[k]);
    }
}

/// [`ConstraintError::Bound`] unless every constrained coordinate lies
/// strictly inside `bounds`.
fn inside(bonded: &[bool], bounds: &Bounds<f64>, x: &Array1<f64>) -> Result<()> {
    let out = (0..x.len()).find(|&k| bonded[k] && !(bounds.low[k] < x[k] && x[k] < bounds.high[k]));
    match out {
        Some(k) => Err(ConstraintError::Bound { atom: k / 3 }.into()),
        None => Ok(()),
    }
}

/// `M^(1/2)` and `W^(1/2)` per coordinate of `n`.
fn roots(set: &DistanceConstraints, n: usize) -> (Array1<f64>, Array1<f64>) {
    let masses = set.masses();
    let root_m = Array1::from_shape_fn(n, |k| masses.map_or(1.0, |m| m[k / 3].sqrt()));
    let root_w = root_m.mapv(f64::recip);
    (root_m, root_w)
}

impl<'a> Space<'a> {
    fn at(set: &'a DistanceConstraints, x: ArrayView1<f64>, rigid: bool) -> Result<Self> {
        let tangent = Tangent::new(set, x)?;
        let masses = set.masses();
        let (root_m, root_w) = roots(set, x.len());
        Ok(Self {
            tangent,
            masses,
            root_m,
            root_w,
            bonded: bonded(set, x.len()),
            rigid,
        })
    }

    fn point(&self) -> &Array1<f64> {
        self.tangent.point()
    }

    /// `M^(1/2) v`: a vector of `x` in `q`.
    fn by_root_m(&self, v: &Array1<f64>) -> Array1<f64> {
        v * &self.root_m
    }

    /// `W^(1/2) u`: a vector of `q` in `x`, or a covector of `x` in `q`.
    fn by_root_w(&self, u: &Array1<f64>) -> Array1<f64> {
        u * &self.root_w
    }

    /// `u <- Pi_q u` for a vector or a gradient of `q`. With rigid
    /// projection on, the rigid motions leave as well: they keep every
    /// distance, so the two `M`-orthogonal projectors commute and their
    /// product is the projector onto what both keep.
    fn project(&self, u: &mut Array1<f64>) -> Result<()> {
        let mut v = self.by_root_w(u);
        self.tangent.vector(&mut v)?;
        if self.rigid {
            project_horizontal(&mut v, self.point().view(), self.masses, true);
        }
        *u = self.by_root_m(&v);
        Ok(())
    }

    /// The projected gradient `Pi^T g` of `x`, its image in `q` and the
    /// multipliers of the constraints.
    fn gradient(&self, g: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>, Vec<f64>)> {
        let mut h = g.clone();
        let lambda = self.tangent.covector(&mut h)?;
        let mut q = self.by_root_w(&h);
        if self.rigid {
            let mut v = self.by_root_w(&q);
            project_horizontal(&mut v, self.point().view(), self.masses, true);
            q = self.by_root_m(&v);
            h = self.by_root_m(&q);
        }
        Ok((h, q, lambda))
    }

    /// The trial `x + step`, capped as an unconstrained trial is, its
    /// free coordinates clipped, then returned to the set along the bonds
    /// at `x`. The clip and the return touch different coordinates, and a
    /// constrained atom that ends on or past its box is an error.
    fn place(&self, bounds: &Bounds<f64>, step: &Array1<f64>, caps: Caps) -> Result<Array1<f64>> {
        let x = self.point();
        let mut trial = x + step;
        caps.apply(x, &mut trial);
        self.finish(bounds, &mut trial)?;
        Ok(trial)
    }

    /// Clip the free coordinates of a trial, return it to the set and
    /// check the constrained ones against `bounds`.
    fn finish(&self, bounds: &Bounds<f64>, trial: &mut Array1<f64>) -> Result<()> {
        clip_free(&self.bonded, bounds, trial);
        self.tangent.constrain(trial)?;
        inside(&self.bonded, bounds, trial)
    }

    /// Orthonormal basis of the tangent space of `q`, one column per
    /// direction. The projector has eigenvalues zero and one, and the
    /// midpoint separates them whatever the rounding of the projection.
    fn basis(&self) -> Result<Array2<f64>> {
        let n = self.root_m.len();
        let mut p = Array2::zeros((n, n));
        for k in 0..n {
            let mut e = Array1::zeros(n);
            e[k] = 1.0;
            self.project(&mut e)?;
            p.column_mut(k).assign(&e);
        }
        let p = (&p + &p.t()) * 0.5;
        let (values, vectors) = sym_eig_jacobi(p);
        let keep: Vec<usize> = (0..n).filter(|&k| values[k] > 0.5).collect();
        Ok(vectors.select(Axis(1), &keep))
    }

    /// `Q^T M^(-1/2) H_L M^(-1/2) Q` and `Q^T g_q` for the Lagrangian
    /// Hessian `H_L = H - sum_k lambda_k grad^2 c_k` of
    /// `c_k = |x_i - x_j|^2 / 2`, the second-order model of `f` along the
    /// set for a tangent step.
    fn reduced(
        &self,
        set: &DistanceConstraints,
        hess: &Array2<f64>,
        lambda: &[f64],
        gq: &Array1<f64>,
    ) -> Result<Model> {
        let mut h = hess.clone();
        for (&(i, j), &l) in set.pairs().iter().zip(lambda) {
            for c in 0..3 {
                let (a, b) = (3 * i + c, 3 * j + c);
                h[(a, a)] -= l;
                h[(b, b)] -= l;
                h[(a, b)] += l;
                h[(b, a)] += l;
            }
        }
        let n = h.nrows();
        for r in 0..n {
            for c in 0..n {
                h[(r, c)] *= self.root_w[r] * self.root_w[c];
            }
        }
        let basis = self.basis()?;
        let k = basis.t().dot(&h).dot(&basis);
        Ok(Model {
            k: (&k + &k.t()) * 0.5,
            g: basis.t().dot(gq),
            basis,
        })
    }
}

/// The second-order model of `f` on the tangent space at a point.
struct Model {
    /// The reduced Lagrangian Hessian.
    k: Array2<f64>,
    /// The reduced gradient.
    g: Array1<f64>,
    /// The orthonormal tangent basis `Q` of `q`.
    basis: Array2<f64>,
}

/// A point with its value and gradient.
type Evaluated = (Array1<f64>, f64, Array1<f64>);

/// What an acceptance test made of a direction.
enum Outcome {
    Moved(Array1<f64>, f64, Array1<f64>),
    Refused,
    NonFinite,
}

/// The constrained curve `R(t) = constrain(x + t d)` of a line search.
struct Curve<'s, 'a> {
    space: &'s Space<'a>,
    bounds: &'s Bounds<f64>,
    dir: Array1<f64>,
    failure: Option<Error>,
    placed: bool,
}

impl Path for Curve<'_, '_> {
    fn start(&self) -> Array1<f64> {
        self.space.point().clone()
    }

    fn direction(&self) -> ArrayView1<'_, f64> {
        self.dir.view()
    }

    fn point(&mut self, t: f64) -> Option<Array1<f64>> {
        let mut p = self.space.point() + &(&self.dir * t);
        let met = self.space.tangent.constrain(&mut p);
        match met.and_then(|_| inside(&self.space.bonded, self.bounds, &p)) {
            Ok(()) => {
                self.placed = true;
                Some(p)
            }
            Err(e) => {
                self.failure = Some(e);
                None
            }
        }
    }

    fn slope(&mut self, _t: f64, at: ArrayView1<'_, f64>, g: ArrayView1<'_, f64>) -> f64 {
        match self.space.tangent.curve_velocity(at, &self.dir) {
            Ok(v) => g.dot(&v),
            Err(_) => f64::NAN,
        }
    }
}

/// The start of a constrained step.
struct Start<'a> {
    cached: bool,
    value: f64,
    raw: Array1<f64>,
    space: Space<'a>,
    /// `Pi^T g` in `x`, the reported gradient.
    grad: Array1<f64>,
    /// Its image in `q`.
    gq: Array1<f64>,
    lambda: Vec<f64>,
}

/// The Lagrangian Hessian action on the tangent space of `q`.
struct Reduced<'s, 'a, O: ?Sized, A> {
    obj: &'s O,
    set: &'s DistanceConstraints,
    space: &'s Space<'a>,
    raw: &'s Array1<f64>,
    lambda: &'s [f64],
    action: &'s A,
    hvps: AtomicUsize,
}

impl<O, A> Objective<f64> for Reduced<'_, '_, O, A>
where
    O: DifferentiableObjective<f64> + ?Sized,
    A: OptionalAction + Send + Sync,
{
    fn dim(&self) -> usize {
        Objective::dim(self.obj)
    }
    fn bounds(&self) -> &Bounds<f64> {
        self.obj.bounds()
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.obj.eval(x)
    }
}

impl<O, A> eindir_core::Gradient<f64> for Reduced<'_, '_, O, A>
where
    O: DifferentiableObjective<f64> + ?Sized,
    A: OptionalAction + Send + Sync,
{
    fn dim(&self) -> usize {
        Objective::dim(self.obj)
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.obj.value_and_gradient(x).1
    }
}

impl<O, A> DifferentiableObjective<f64> for Reduced<'_, '_, O, A>
where
    O: DifferentiableObjective<f64> + ?Sized,
    A: OptionalAction + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.obj.value_and_gradient(x)
    }
}

impl<O, A> HessianVector for Reduced<'_, '_, O, A>
where
    O: DifferentiableObjective<f64> + ?Sized,
    A: OptionalAction + Send + Sync,
{
    fn hessian_vector(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let nan = || Array1::from_elem(v.len(), f64::NAN);
        let mut u = v.to_owned();
        if self.space.project(&mut u).is_err() {
            return nan();
        }
        let w = self.space.by_root_w(&u);
        if l2(&w) <= 0.0 {
            return Array1::zeros(v.len());
        }
        let at = self.space.point();
        self.hvps.fetch_add(1, Ordering::Relaxed);
        let mut hv = match self.action.apply(at.view(), w.view()) {
            Some(hv) => hv,
            None => forward_difference(self.obj, self.raw, at.view(), w.view()),
        };
        for (&(i, j), &l) in self.set.pairs().iter().zip(self.lambda) {
            for c in 0..3 {
                let d = l * (w[3 * i + c] - w[3 * j + c]);
                hv[3 * i + c] -= d;
                hv[3 * j + c] += d;
            }
        }
        let mut out = self.space.by_root_w(&hv);
        if self.space.project(&mut out).is_err() {
            return nan();
        }
        out
    }
}

/// The stored state of a method that runs on a set.
enum Memory {
    Lbfgs(Curvature),
    Nlcg([Array1<f64>; 4], bool),
    Matrix(Array2<f64>, bool),
    Adam(Array1<f64>, Array1<f64>, f64, f64),
    Fire(FireState),
    Bb(Option<Array1<f64>>, Option<Array1<f64>>),
    Radius(f64),
    Stateless,
}

impl Inner {
    fn memory(&self) -> Memory {
        match self {
            Inner::Lbfgs(solver) => Memory::Lbfgs(solver.curvature()),
            Inner::Nlcg {
                dir,
                g_old,
                z_old,
                d_old,
                initialized,
                ..
            } => Memory::Nlcg([dir, g_old, z_old, d_old].map(|v| v.clone()), *initialized),
            Inner::Bfgs { h, fresh } | Inner::Sr1 { h, fresh } | Inner::Sr2 { b: h, fresh } => {
                Memory::Matrix(h.clone(), *fresh)
            }
            Inner::Adam { m, v, b1p, b2p, .. } => Memory::Adam(m.clone(), v.clone(), *b1p, *b2p),
            Inner::Fire(state) => Memory::Fire(state.clone()),
            Inner::Bb { prev_s, prev_y } => Memory::Bb(prev_s.clone(), prev_y.clone()),
            Inner::Dogleg { radius } | Inner::NewtonCg { radius } | Inner::NewtonPcg { radius } => {
                Memory::Radius(*radius)
            }
            Inner::Steepest | Inner::Newton { .. } | Inner::Pso { .. } => Memory::Stateless,
        }
    }

    fn restore(&mut self, saved: Memory) {
        match (self, saved) {
            (Inner::Lbfgs(solver), Memory::Lbfgs(saved)) => solver.restore(saved),
            (
                Inner::Nlcg {
                    dir,
                    g_old,
                    z_old,
                    d_old,
                    initialized,
                    ..
                },
                Memory::Nlcg([d, g, z, o], i),
            ) => (*dir, *g_old, *z_old, *d_old, *initialized) = (d, g, z, o, i),
            (
                Inner::Bfgs { h, fresh } | Inner::Sr1 { h, fresh } | Inner::Sr2 { b: h, fresh },
                Memory::Matrix(saved, f),
            ) => (*h, *fresh) = (saved, f),
            (Inner::Adam { m, v, b1p, b2p, .. }, Memory::Adam(sm, sv, p1, p2)) => {
                (*m, *v, *b1p, *b2p) = (sm, sv, p1, p2);
            }
            (Inner::Fire(state), Memory::Fire(saved)) => *state = saved,
            (Inner::Bb { prev_s, prev_y }, Memory::Bb(s, y)) => (*prev_s, *prev_y) = (s, y),
            (
                Inner::Dogleg { radius } | Inner::NewtonCg { radius } | Inner::NewtonPcg { radius },
                Memory::Radius(r),
            ) => *radius = r,
            _ => {}
        }
    }
}

/// What a constrained step writes to the session before it can fail.
/// The gradient and Hessian-vector call counters are not kept: they
/// count calls the failed step made.
struct Checkpoint {
    memory: Memory,
    istep: f64,
    e_hist: VecDeque<f64>,
    refused: bool,
    steps: usize,
    saddle_kicks: u8,
    last_pos: Option<Array1<f64>>,
    last_value: f64,
    last_grad: Array1<f64>,
}

/// `v` with each component of `q` capped at the per-coordinate speed that
/// moves its coordinate of `x` by at most `cap` in one step.
fn clip_components_q(state: &mut FireState, cap: f64, root_m: &Array1<f64>) -> Array1<f64> {
    let dt = state.dt;
    state
        .vel
        .iter_mut()
        .zip(root_m)
        .for_each(|(v, m)| *v = v.max(-cap * m / dt).min(cap * m / dt));
    &state.vel * dt
}

impl Solver {
    fn constraint_set(&self) -> Arc<DistanceConstraints> {
        Arc::clone(
            self.constraints
                .as_ref()
                .expect("a constrained step runs only with a set"),
        )
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            memory: self.inner.memory(),
            istep: self.istep,
            e_hist: self.e_hist.clone(),
            refused: self.refused,
            steps: self.steps,
            saddle_kicks: self.saddle_kicks,
            last_pos: self.last_pos.clone(),
            last_value: self.last_value,
            last_grad: self.last_grad.clone(),
        }
    }

    fn roll_back(&mut self, saved: Checkpoint) {
        self.inner.restore(saved.memory);
        self.istep = saved.istep;
        self.e_hist = saved.e_hist;
        self.refused = saved.refused;
        self.steps = saved.steps;
        self.saddle_kicks = saved.saddle_kicks;
        self.last_pos = saved.last_pos;
        self.last_value = saved.last_value;
        self.last_grad = saved.last_grad;
    }

    /// Run one constrained step on a copy of `x`. On
    /// [`Error::Constraint`] or [`Error::Oracle`] the session goes back
    /// to its checkpoint and `x` stays as the caller passed it. Any other
    /// result hands the copy to `x`, as an unconstrained step leaves `x`
    /// where it stopped.
    pub(super) fn constrained_step<F>(&mut self, x: &mut Array1<f64>, step: F) -> Result<Report>
    where
        F: FnOnce(&mut Self, &mut Array1<f64>) -> Result<Report>,
    {
        let saved = self.checkpoint();
        let mut work = x.clone();
        match step(self, &mut work) {
            Err(e @ (Error::Constraint(_) | Error::Oracle { .. })) => {
                self.roll_back(saved);
                Err(e)
            }
            other => {
                *x = work;
                other
            }
        }
    }

    /// A caller's pair `(s, y)` of `x` in `q`, tangent at the last point
    /// when the session has one.
    pub(super) fn constrained_pair(
        &self,
        s: ArrayView1<f64>,
        y: ArrayView1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let set = self.constraint_set();
        let (root_m, root_w) = roots(&set, self.dim);
        let (mut s, mut y) = (&s * &root_m, &y * &root_w);
        if let Some(at) = &self.last_pos {
            let space = Space::at(&set, at.view(), self.project_rigid)?;
            space.project(&mut s)?;
            space.project(&mut y)?;
        }
        Ok((s, y))
    }

    /// The two-loop direction of a caller's gradient `g`, in `x`.
    pub(super) fn constrained_direction(&self, g: ArrayView1<f64>) -> Result<Array1<f64>> {
        let set = self.constraint_set();
        let (_, root_w) = roots(&set, self.dim);
        let Inner::Lbfgs(solver) = &self.inner else {
            return Ok(g.mapv(|v| -v) * &root_w * &root_w);
        };
        let mut d = solver.direction((&g * &root_w).view());
        if let Some(at) = &self.last_pos {
            let space = Space::at(&set, at.view(), self.project_rigid)?;
            space.project(&mut d)?;
        }
        Ok(d * &root_w)
    }

    /// The settings a constrained step cannot honour, as typed errors.
    pub(super) fn refuse_constraint_settings(&self, set: &DistanceConstraints) -> Result<()> {
        set.fits(self.dim)?;
        if self.manifold != ManifoldKind::Euclidean {
            return Err(unsupported("a manifold other than Euclidean coordinates"));
        }
        if self.exp.is_some() {
            return Err(unsupported("the exponential preconditioner"));
        }
        if self.lindh.is_some() {
            return Err(unsupported("the Lindh preconditioner"));
        }
        #[cfg(feature = "highs")]
        if self.highs {
            return Err(unsupported("the HiGHS feasible-set step"));
        }
        if self.periodic {
            return Err(unsupported(
                "a periodic cell, whose lattice the session does not carry",
            ));
        }
        if matches!(self.inner, Inner::Pso { .. }) {
            return Err(unsupported("particle swarm, which samples the whole box"));
        }
        Ok(())
    }

    fn caps(&self) -> Caps {
        Caps {
            atom: self.atom_maxmove,
            euclid: self.control.maxmove,
        }
    }

    /// Check the settings, put a moved `x` on the set, and evaluate there.
    /// The clip and the return to the set run on a copy, which `x` takes
    /// only when every check here has passed.
    fn enter<'a, O>(
        &mut self,
        set: &'a DistanceConstraints,
        obj: &O,
        x: &mut Array1<f64>,
    ) -> Result<Start<'a>>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if x.len() != self.dim || self.dim != Objective::dim(obj) {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim,
            });
        }
        self.refuse_constraint_settings(set)?;
        let bounds = obj.bounds();
        let bonded = bonded(set, x.len());
        let cached = self.same_last_x(x);
        let mut entered = x.clone();
        if !cached {
            clip_free(&bonded, bounds, &mut entered);
            let excess = set.excess(entered.view());
            if excess.is_nan() || excess > 1.0 {
                Tangent::new(set, entered.view())?.constrain(&mut entered)?;
            }
            inside(&bonded, bounds, &entered)?;
        }
        // The rigid projector moves every atom; at an active bound it and
        // the box projection would not commute.
        let active =
            (0..entered.len()).any(|k| entered[k] <= bounds.low[k] || entered[k] >= bounds.high[k]);
        if self.project_rigid && active {
            return Err(unsupported(
                "rigid projection at an active coordinate bound",
            ));
        }
        let (value, raw) = if cached {
            (self.last_value, self.last_grad.clone())
        } else {
            obj.value_and_gradient(entered.view())
        };
        if !finite(value, raw.view()) {
            return Err(non_finite());
        }
        let space = Space::at(set, entered.view(), self.project_rigid)?;
        let (grad, gq, lambda) = space.gradient(&raw)?;
        *x = entered;
        Ok(Start {
            cached,
            value,
            raw,
            space,
            grad,
            gq,
            lambda,
        })
    }

    /// Move the stored vectors of the method to the tangent space at a new
    /// point.
    fn transport_memory(&mut self, space: &Space<'_>) -> Result<()> {
        let project = |v: &Array1<f64>| -> Result<Array1<f64>> {
            let mut u = v.clone();
            space.project(&mut u)?;
            Ok(u)
        };
        match &mut self.inner {
            Inner::Lbfgs(solver) => solver.try_transport(project)?,
            Inner::Nlcg {
                dir,
                g_old,
                z_old,
                d_old,
                initialized: true,
                ..
            } => {
                *dir = project(dir)?;
                *g_old = project(g_old)?;
                *z_old = g_old.clone();
                *d_old = project(d_old)?;
            }
            Inner::Fire(state) => state.vel = project(&state.vel)?,
            Inner::Bb { prev_s, prev_y } => {
                if let (Some(s), Some(y)) = (prev_s.as_mut(), prev_y.as_mut()) {
                    *s = project(s)?;
                    *y = project(y)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// The pair of a move from `old` (with `q` gradient `gq_old`) to the
    /// point of `space` (with `gq`), both tangent there.
    fn tangent_pair(
        space: &Space<'_>,
        old: &Array1<f64>,
        gq_old: &Array1<f64>,
        gq: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let mut s = space.by_root_m(&(space.point() - old));
        space.project(&mut s)?;
        let mut back = gq_old.clone();
        space.project(&mut back)?;
        Ok((s, gq - &back))
    }

    /// An external move since the last step: carry the memory over and
    /// record its pair. A later failure of the step takes both back
    /// through [`Self::constrained_step`].
    fn arrive_from_outside(&mut self, set: &DistanceConstraints, start: &Start<'_>) -> Result<()> {
        let Some(previous) = self.last_pos.clone() else {
            return Ok(());
        };
        let old = Space::at(set, previous.view(), self.project_rigid)?;
        let (_, gq_old, _) = old.gradient(&self.last_grad)?;
        self.transport_memory(&start.space)?;
        if let Inner::Lbfgs(_) = &self.inner {
            let (s, y) = Self::tangent_pair(&start.space, &previous, &gq_old, &start.gq)?;
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.push_pair(s, y, Some(l2(&start.gq)));
            }
        }
        Ok(())
    }

    /// [`crate::accept::accept_step`] on the set. Trials are placed by
    /// [`Space::place`] and the fallback runs along `-W g`. A trial the
    /// set cannot be met at counts as refused under the energy tests and
    /// is an error under [`Accept::None`], or when no trial was placed.
    fn accept_on<O>(
        &mut self,
        obj: &O,
        space: &Space<'_>,
        value: f64,
        dir: &Array1<f64>,
        steepest: &Array1<f64>,
    ) -> Result<Outcome>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let caps = self.caps();
        let place = |alpha: f64, d: &Array1<f64>| space.place(obj.bounds(), &(d * alpha), caps);
        if self.accept == Accept::None {
            let trial = place(1.0, dir)?;
            let (ft, gt) = obj.value_and_gradient(trial.view());
            if !finite(ft, gt.view()) {
                return Ok(Outcome::NonFinite);
            }
            push_energy(&mut self.e_hist, ft);
            return Ok(Outcome::Moved(trial, ft, gt));
        }
        let mut reference = value;
        if self.accept == Accept::Nonmonotone
            && let Some(m) = self.e_hist.iter().copied().reduce(f64::max)
        {
            reference = m;
        }
        let halvings = (0..HALVINGS).map(|k| (dir, 0.5_f64.powi(k as i32)));
        let mut failure = None;
        let mut placed = false;
        for (d, alpha) in halvings.chain(std::iter::once((steepest, FALLBACK_STEP))) {
            match place(alpha, d) {
                Ok(trial) => {
                    placed = true;
                    let (ft, gt) = obj.value_and_gradient(trial.view());
                    if finite(ft, gt.view()) && ft - reference <= ENERGY_RISE {
                        push_energy(&mut self.e_hist, ft);
                        return Ok(Outcome::Moved(trial, ft, gt));
                    }
                }
                Err(e) => failure = Some(e),
            }
        }
        match failure {
            Some(e) if !placed => Err(e),
            _ => Ok(Outcome::Refused),
        }
    }

    /// The L-BFGS line search on the curve from `space` along `dir`,
    /// started at `t0`, then the post-search clip and cap and the
    /// acceptance of [`crate::step::take_step`].
    fn curve_search<O>(
        &self,
        obj: &O,
        space: &Space<'_>,
        value: f64,
        raw: &Array1<f64>,
        dir: &Array1<f64>,
        t0: f64,
    ) -> Result<(Option<Evaluated>, f64)>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let mut curve = Curve {
            space,
            bounds: obj.bounds(),
            dir: dir.clone(),
            failure: None,
            placed: false,
        };
        let x = space.point();
        let (npos, _, lsstep) =
            self.linesearch
                .search_path(|p| obj.value_and_gradient(p), &mut curve, t0);
        if npos == *x {
            return match curve.failure.take() {
                Some(e) if !curve.placed => Err(e),
                _ => Ok((None, 0.0)),
            };
        }
        let mut trial = npos;
        clip_free(&space.bonded, obj.bounds(), &mut trial);
        let mut t = lsstep;
        let caps = self.caps();
        let reach = match (caps.atom, caps.euclid) {
            (Some(cap), _) => Some((cap, max_atom_motion(x, &trial))),
            (None, Some(cap)) => Some((cap, l2(&(&trial - x)))),
            (None, None) => None,
        };
        if let Some((cap, length)) = reach.filter(|&(cap, length)| length > cap) {
            t *= cap / length;
            trial = x + &(dir * t);
            space.finish(obj.bounds(), &mut trial)?;
        }
        let (nval, grad) = obj.value_and_gradient(trial.view());
        let ok = finite(nval, grad.view());
        // Equal rounded energies: strong curvature along the curve
        // certifies the descent, as on a line.
        let tied = match self.linesearch {
            LineSearch::Wolfe { c1, c2, .. } if ok && nval == value => {
                let initial = t * raw.dot(dir);
                let at = curve.slope(t, trial.view(), grad.view()) * t;
                initial < 0.0
                    && crate::linesearch::conditions::armijo(nval, value, 1.0, initial, c1)
                    && crate::linesearch::conditions::strong_curvature(at, initial, c2)
            }
            _ => false,
        };
        if ok && (nval < value || tied) {
            Ok((Some((trial, nval, grad)), lsstep))
        } else {
            Ok((None, 0.0))
        }
    }

    /// One first-order step on the set.
    pub(super) fn step_constrained<O>(&mut self, obj: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        match self.inner {
            Inner::NewtonCg { .. } | Inner::NewtonPcg { .. } => {
                return self.step_constrained_newton_cg(obj, x, &FiniteDifference);
            }
            Inner::Newton { .. } | Inner::Dogleg { .. } => return Err(Error::NeedHessian),
            _ => {}
        }
        let set = self.constraint_set();
        let start = self.enter(&set, obj, x)?;
        if self.stationarity_norm(x, &start.grad) < self.control.gtol {
            self.remember(x, start.value, &start.raw);
            return Ok(self.report(start.value, x, &start.grad));
        }
        if !start.cached {
            self.arrive_from_outside(&set, &start)?;
        }
        let raw_gradient = matches!(
            self.inner,
            Inner::Steepest
                | Inner::Bfgs { .. }
                | Inner::Sr1 { .. }
                | Inner::Sr2 { .. }
                | Inner::Bb { .. }
                | Inner::Nlcg { .. }
        );
        if self.unit_first_step
            && raw_gradient
            && self.accept != Accept::None
            && self.last_pos.is_none()
        {
            self.istep = self.control.istep.min(1.0 / l2(&start.gq));
        }
        if let Inner::Fire(_) = self.inner {
            return self.step_constrained_fire(&set, obj, x, start);
        }
        if matches!(self.inner, Inner::Lbfgs(_)) && self.accept != Accept::None {
            return self.step_constrained_lbfgs_search(&set, obj, x, start);
        }

        let space = &start.space;
        let gq = &start.gq;
        let istep = self.istep;
        let accept = self.accept;
        let mut dir_q = match &mut self.inner {
            Inner::Lbfgs(solver) => solver.direction(gq.view()),
            Inner::Steepest => gq.mapv(|g| -g * istep),
            Inner::Nlcg {
                dir,
                g_old,
                z_old,
                d_old,
                initialized,
                ..
            } => {
                if !*initialized {
                    *dir = gq.mapv(|g| -g * istep);
                    *g_old = gq.clone();
                    *z_old = gq.clone();
                    *d_old = dir.clone();
                    *initialized = true;
                }
                dir.clone()
            }
            Inner::Bfgs { h, fresh } | Inner::Sr1 { h, fresh } => {
                let mut d = -h.dot(gq);
                if *fresh && accept != Accept::None {
                    d *= istep;
                }
                d
            }
            Inner::Sr2 { b, fresh } => {
                let rhs = gq.mapv(|g| -g);
                let mut d = solve_dense(b, &rhs).unwrap_or(rhs);
                if *fresh && accept != Accept::None {
                    d *= istep;
                }
                d
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
                let d = adam_direction(m, v, gq, *beta1, *beta2, (*b1p, *b2p), *eps);
                *b1p *= *beta1;
                *b2p *= *beta2;
                d.mapv(|d| d * istep)
            }
            Inner::Bb { prev_s, prev_y } => {
                bb_direction(prev_s.as_ref(), prev_y.as_ref(), gq, istep)
            }
            Inner::Fire(_)
            | Inner::Pso { .. }
            | Inner::Newton { .. }
            | Inner::Dogleg { .. }
            | Inner::NewtonCg { .. }
            | Inner::NewtonPcg { .. } => unreachable!(),
        };
        space.project(&mut dir_q)?;
        let dir = space.by_root_w(&dir_q);
        let steepest = space.by_root_w(&gq.mapv(|g| -g));
        let moved = match self.accept_on(obj, space, start.value, &dir, &steepest)? {
            Outcome::Moved(trial, ft, gt) => Some((trial, ft, gt)),
            Outcome::NonFinite if matches!(self.inner, Inner::Lbfgs(_)) => {
                return Err(Error::Oracle {
                    what: "non-finite value or gradient",
                });
            }
            Outcome::NonFinite | Outcome::Refused => None,
        };
        let Some((trial, value, raw)) = moved else {
            let refused = self.accept != Accept::None;
            match &mut self.inner {
                Inner::Nlcg { initialized, .. } if refused => *initialized = false,
                Inner::Bfgs { h, fresh } | Inner::Sr1 { h, fresh } | Inner::Sr2 { b: h, fresh }
                    if refused =>
                {
                    *h = Array2::eye(self.dim);
                    *fresh = true;
                }
                Inner::Bb { prev_s, prev_y } if refused => {
                    *prev_s = None;
                    *prev_y = None;
                }
                _ => {}
            }
            let refused = refused && !matches!(self.inner, Inner::Adam { .. } | Inner::Lbfgs(_));
            if refused && self.refused && start.cached {
                return Err(Error::AcceptStalled { steps: self.steps });
            }
            self.refused = refused;
            self.remember(x, start.value, &start.raw);
            self.steps += 1;
            return Ok(self.report(start.value, x, &start.grad));
        };
        let old = x.clone();
        *x = trial;
        let next = Space::at(&set, x.view(), self.project_rigid)?;
        let (grad, gq_new, _) = next.gradient(&raw)?;
        self.transport_memory(&next)?;
        let (s, y) = Self::tangent_pair(&next, &old, gq, &gq_new)?;
        let istep = self.istep;
        match &mut self.inner {
            Inner::Lbfgs(solver) => {
                solver.push_pair(s, y, Some(l2(&gq_new)));
            }
            Inner::Nlcg {
                conjugacy,
                restart,
                dir,
                g_old,
                z_old,
                d_old,
                ..
            } => {
                let ctx = ConjugacyContext {
                    current_gradient: gq_new.view(),
                    previous_gradient: g_old.view(),
                    previous_direction: d_old.view(),
                };
                let mut beta = conjugacy.beta(&ctx);
                if restart.should_restart(&ctx) {
                    beta = 0.0;
                }
                let mut d = &gq_new.mapv(|g| -g * istep) + &(&*d_old * beta);
                next.project(&mut d)?;
                *dir = d;
                g_old.assign(&gq_new);
                z_old.assign(&gq_new);
                d_old.assign(dir);
            }
            Inner::Bfgs { h, fresh } => {
                Self::seeded_update(h, fresh, accept, &s, &y, false, bfgs_inverse_update);
            }
            Inner::Sr1 { h, fresh } => {
                Self::seeded_update(h, fresh, accept, &s, &y, false, sr1_inverse_update);
            }
            Inner::Sr2 { b, fresh } => {
                Self::seeded_update(b, fresh, accept, &s, &y, true, sr2_hessian_update);
            }
            Inner::Bb { prev_s, prev_y } => {
                *prev_s = Some(s);
                *prev_y = Some(y);
            }
            _ => {}
        }
        self.refused = false;
        self.remember(x, value, &raw);
        self.steps += 1;
        Ok(self.report(value, x, &grad))
    }

    /// Nocedal-Wright 6.20 seed on the first pair of a fresh matrix, then
    /// the update, as the unconstrained quasi-Newton arms do.
    fn seeded_update(
        h: &mut Array2<f64>,
        fresh: &mut bool,
        accept: Accept,
        s: &Array1<f64>,
        y: &Array1<f64>,
        hessian: bool,
        update: fn(&mut Array2<f64>, &Array1<f64>, &Array1<f64>),
    ) {
        let seed = *fresh && accept != Accept::None;
        if !seed {
            update(h, s, y);
            *fresh = false;
        } else if let Some(gamma) = secant_scale(s, y) {
            let n = h.nrows();
            *h = if hessian {
                Array2::eye(n) / gamma
            } else {
                Array2::eye(n) * gamma
            };
            *fresh = false;
            update(h, s, y);
        }
    }

    /// L-BFGS under an energy test: the session line search on the curve.
    fn step_constrained_lbfgs_search<O>(
        &mut self,
        set: &DistanceConstraints,
        obj: &O,
        x: &mut Array1<f64>,
        start: Start<'_>,
    ) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let space = &start.space;
        let Inner::Lbfgs(solver) = &mut self.inner else {
            unreachable!()
        };
        let direction = |solver: &crate::lbfgs::Lbfgs| -> Result<Array1<f64>> {
            let mut d = solver.direction(start.gq.view());
            space.project(&mut d)?;
            Ok(space.by_root_w(&d))
        };
        // Curvature history supplies the quasi-Newton step length.
        let t0 = if solver.is_empty() { self.istep } else { 1.0 };
        let dir = direction(solver)?;
        let (mut found, mut lsstep) =
            self.curve_search(obj, space, start.value, &start.raw, &dir, t0)?;
        if found.is_none() {
            let Inner::Lbfgs(solver) = &mut self.inner else {
                unreachable!()
            };
            if !solver.is_empty() {
                // Retained curvature can describe a different objective
                // scale. Retry with an empty history.
                solver.forget();
                let dir = direction(solver)?;
                (found, lsstep) = self.curve_search(
                    obj,
                    space,
                    start.value,
                    &start.raw,
                    &dir,
                    self.control.istep,
                )?;
            }
        }
        self.istep = next_istep(lsstep, &self.control);
        let Some((trial, value, raw)) = found else {
            self.remember(x, start.value, &start.raw);
            self.steps += 1;
            return Ok(self.report(start.value, x, &start.grad));
        };
        let old = x.clone();
        *x = trial;
        let next = Space::at(set, x.view(), self.project_rigid)?;
        let (grad, gq, _) = next.gradient(&raw)?;
        self.transport_memory(&next)?;
        let (s, y) = Self::tangent_pair(&next, &old, &start.gq, &gq)?;
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.push_pair(s, y, Some(l2(&gq)));
        }
        self.remember(x, value, &raw);
        self.steps += 1;
        Ok(self.report(value, x, &grad))
    }

    /// FIRE and FIRE 2.0 in `q`: unit masses there are the set's masses in
    /// `x`, so the integrator is the unconstrained one. The force and the
    /// velocity pass through the same projector (the set's, then the
    /// active coordinate bounds, which act on other coordinates), and the
    /// velocity is projected again at the new point, the second half of
    /// RATTLE (Andersen 1983).
    fn step_constrained_fire<O>(
        &mut self,
        set: &DistanceConstraints,
        obj: &O,
        x: &mut Array1<f64>,
        start: Start<'_>,
    ) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let bounds = obj.bounds();
        let caps = self.caps();
        let Inner::Fire(state) = &mut self.inner else {
            unreachable!()
        };
        let mut space = start.space;
        let mut raw = start.gq.mapv(|g| -g);
        project_box_tangent(bounds, x, &mut raw);
        space.project(&mut state.vel)?;
        project_box_tangent(bounds, x, &mut state.vel);
        match fire_prepare_v2(state, &raw) {
            FirePreparation::Ready => {}
            FirePreparation::Rollback(correction) => {
                let back = space.place(
                    bounds,
                    &space.by_root_w(&correction),
                    Caps {
                        atom: None,
                        euclid: None,
                    },
                )?;
                let ev = obj.value_and_gradient(back.view());
                if !finite(ev.0, ev.1.view()) {
                    return Err(non_finite());
                }
                *x = back;
                space = Space::at(set, x.view(), self.project_rigid)?;
                let (_, gq, _) = space.gradient(&ev.1)?;
                raw = gq.mapv(|g| -g);
                project_box_tangent(bounds, x, &mut raw);
                space.project(&mut state.vel)?;
            }
            FirePreparation::Stalled => {
                return Err(Error::FireStalled { steps: self.steps });
            }
        }
        let forward_start = x.clone();
        let dx_q = fire_displacement(state, &raw);
        let dx_q = match caps.atom.or(caps.euclid) {
            Some(cap) if state.abc => clip_components_q(state, cap, &space.root_m),
            _ => dx_q,
        };
        let mut trial = &*x + &space.by_root_w(&dx_q);
        let plain = state.preset() == FirePreset::Default && !state.abc;
        let free = (!plain).then(|| trial.clone());
        if !state.abc {
            caps.apply(x, &mut trial);
        }
        clip_free(&space.bonded, bounds, &mut trial);
        let moved = free.is_none_or(|free| trial != free);
        space.tangent.constrain(&mut trial)?;
        inside(&space.bonded, bounds, &trial)?;
        if matches!(state.kind, FireKind::V2) && state.dt > 0.0 && moved {
            state.vel = space.by_root_m(&((&trial - &forward_start) / state.dt));
        }
        let (value, grad) = obj.value_and_gradient(trial.view());
        if !finite(value, grad.view()) {
            return Err(non_finite());
        }
        *x = trial;
        let next = Space::at(set, x.view(), self.project_rigid)?;
        let (report_grad, gq, _) = next.gradient(&grad)?;
        let mut raw_new = gq.mapv(|g| -g);
        project_box_tangent(bounds, x, &mut raw_new);
        next.project(&mut state.vel)?;
        project_box_tangent(bounds, x, &mut state.vel);
        fire_after_v1(state, &raw_new);
        self.remember(x, value, &grad);
        self.steps += 1;
        Ok(self.report(value, x, &report_grad))
    }

    /// One Newton, RFO, L-BFGS-with-Hessian or dogleg step on the set,
    /// with the Lagrangian Hessian reduced to the tangent space.
    pub(super) fn step_constrained_hess<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
    ) -> Result<Report>
    where
        O: HessianObjective + ?Sized,
    {
        let newton_kind = match &self.inner {
            Inner::Newton { kind } => Some(*kind),
            Inner::Lbfgs(_) => match self.qn_step {
                QnStep::Newton => Some(NewtonKind::Shifted),
                QnStep::Rfo => Some(NewtonKind::Rfo),
                QnStep::TwoLoop => None,
            },
            Inner::Dogleg { .. } => None,
            Inner::NewtonCg { .. } | Inner::NewtonPcg { .. } => {
                return self.step_constrained_newton_cg(obj, x, &FiniteDifference);
            }
            _ => return self.step_constrained(obj, x),
        };
        let set = self.constraint_set();
        let start = self.enter(&set, obj, x)?;
        if self.stationarity_norm(x, &start.grad) < self.control.gtol {
            self.remember(x, start.value, &start.raw);
            return self.report_or_escape_constrained(&set, obj, x, start.value, start.raw);
        }
        if !start.cached {
            self.arrive_from_outside(&set, &start)?;
        }
        let hess = obj.hessian(x.view());
        let model = start.space.reduced(&set, &hess, &start.lambda, &start.gq)?;
        if matches!(self.inner, Inner::Dogleg { .. }) {
            return self.step_constrained_dogleg(&set, obj, x, start, &model);
        }
        let Model { k, g: gr, basis } = &model;
        let mut dir_q = match (newton_kind, &self.inner) {
            (Some(NewtonKind::Shifted), _) => basis.dot(&shifted_newton(k, gr)),
            (Some(NewtonKind::Rfo), _) => basis.dot(&rfo_direction(k, gr)),
            (None, Inner::Lbfgs(solver)) => {
                // The reduced Hessian on the tangent space and the identity
                // across it: `H_0^-1` of the two-loop.
                let n = self.dim;
                let p =
                    basis.dot(k).dot(&basis.t()) + Array2::<f64>::eye(n) - basis.dot(&basis.t());
                solver.direction_with_precon(start.gq.view(), Some(&p))
            }
            _ => unreachable!(),
        };
        let space = &start.space;
        space.project(&mut dir_q)?;
        let dir = space.by_root_w(&dir_q);
        let steepest = space.by_root_w(&start.gq.mapv(|g| -g));
        let (mut value, mut raw) = (start.value, start.raw.clone());
        if let Outcome::Moved(trial, ft, gt) = self.accept_on(obj, space, value, &dir, &steepest)? {
            let old = x.clone();
            *x = trial;
            value = ft;
            raw = gt;
            let next = Space::at(&set, x.view(), self.project_rigid)?;
            let (_, gq, _) = next.gradient(&raw)?;
            self.transport_memory(&next)?;
            let (s, y) = Self::tangent_pair(&next, &old, &start.gq, &gq)?;
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.push_pair(s, y, Some(l2(&gq)));
            }
        }
        self.remember(x, value, &raw);
        self.steps += 1;
        self.report_or_escape_constrained(&set, obj, x, value, raw)
    }

    fn step_constrained_dogleg<O>(
        &mut self,
        set: &DistanceConstraints,
        obj: &O,
        x: &mut Array1<f64>,
        start: Start<'_>,
        model: &Model,
    ) -> Result<Report>
    where
        O: HessianObjective + ?Sized,
    {
        let radius = match &self.inner {
            Inner::Dogleg { radius } => *radius,
            _ => self.control.istep.max(RADIUS_START),
        };
        let rmax = self
            .control
            .maxmove
            .unwrap_or(radius * RADIUS_GROWTH)
            .max(radius);
        let space = &start.space;
        let Model { k, g: gr, basis } = model;
        let mut dir_q = basis.dot(&dogleg_direction(k, gr, radius));
        space.project(&mut dir_q)?;
        let trust_norm = l2(&dir_q);
        let caps = self.caps();
        let boundary = |taken: f64| {
            if caps.atom.is_some() || caps.euclid.is_some() {
                trust_norm
            } else {
                taken
            }
        };
        let (rho, taken, trial) = match space.place(obj.bounds(), &space.by_root_w(&dir_q), caps) {
            Ok(trial) => {
                let p = basis.t().dot(&space.by_root_m(&(&trial - &*x)));
                let (ft, gt) = obj.value_and_gradient(trial.view());
                let rho = reduction_ratio(start.value - ft, predicted_reduction(k, gr, &p));
                (rho, l2(&p), Some((trial, ft, gt)))
            }
            // A trial the set cannot be met at is a rejected model step.
            Err(Error::Constraint(_)) => (-1.0, 0.0, None),
            Err(e) => return Err(e),
        };
        if let Inner::Dogleg { radius: r } = &mut self.inner {
            *r = update_radius(*r, rho, boundary(taken), rmax);
        }
        match trial {
            Some((trial, ft, gt)) if accept_ratio(rho) => {
                *x = trial;
                self.remember(x, ft, &gt);
                self.steps += 1;
                self.report_or_escape_constrained(set, obj, x, ft, gt)
            }
            _ => {
                self.remember(x, start.value, &start.raw);
                self.steps += 1;
                self.report_or_escape_constrained(set, obj, x, start.value, start.raw)
            }
        }
    }

    /// [`Solver::report`] at a point of the set, after leaving a saddle of
    /// the reduced Hessian as the unconstrained Newton and dogleg arms do.
    fn report_or_escape_constrained<O>(
        &mut self,
        set: &DistanceConstraints,
        obj: &O,
        x: &mut Array1<f64>,
        mut value: f64,
        mut raw: Array1<f64>,
    ) -> Result<Report>
    where
        O: HessianObjective + ?Sized,
    {
        let mut space = Space::at(set, x.view(), self.project_rigid)?;
        let (mut grad, _, mut lambda) = space.gradient(&raw)?;
        if self.saddle_escape_enabled() {
            while self.stationarity_norm(x, &grad) <= self.control.gtol
                && self.saddle_kicks < SADDLE_KICKS
            {
                let hess = obj.hessian(x.view());
                let (_, gq, _) = space.gradient(&raw)?;
                let Model { k, basis, .. } = space.reduced(set, &hess, &lambda, &gq)?;
                let Some(mode) = self.unstable_mode(&k) else {
                    break;
                };
                let cap = self
                    .atom_maxmove
                    .unwrap_or_else(|| self.control.istep.max(RADIUS_START));
                let step = scale_displacement_to_atom_cap(&space.by_root_w(&basis.dot(&mode)), cap);
                let no_caps = Caps {
                    atom: None,
                    euclid: None,
                };
                let Ok(trial) = space.place(obj.bounds(), &step, no_caps) else {
                    break;
                };
                let (v, g) = obj.value_and_gradient(trial.view());
                if !finite(v, g.view()) {
                    break;
                }
                *x = trial;
                value = v;
                raw = g;
                space = Space::at(set, x.view(), self.project_rigid)?;
                (grad, _, lambda) = space.gradient(&raw)?;
                self.saddle_kicks = self.saddle_kicks.saturating_add(1);
                if let Inner::Dogleg { radius } = &mut self.inner {
                    *radius = self.control.istep.max(RADIUS_START);
                }
                self.remember(x, value, &raw);
            }
        }
        Ok(self.report(value, x, &grad))
    }

    /// One Steihaug-Toint step on the set: CG on the reduced Lagrangian
    /// Hessian action in `q`, the trust region measured in `q`.
    pub(super) fn step_constrained_newton_cg<O, A>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        action: &A,
    ) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
        A: OptionalAction + Send + Sync,
    {
        let set = self.constraint_set();
        let start = self.enter(&set, obj, x)?;
        if !start.cached {
            self.gradient_calls += 1;
        }
        if self.stationarity_norm(x, &start.grad) < self.control.gtol {
            self.remember(x, start.value, &start.raw);
            return Ok(self.report(start.value, x, &start.grad));
        }
        let (radius, precondition) = match &self.inner {
            Inner::NewtonCg { radius } => (*radius, false),
            Inner::NewtonPcg { radius } => (*radius, true),
            _ => return Err(Error::NeedHvp),
        };
        let rmax = self
            .control
            .maxmove
            .unwrap_or(radius * RADIUS_GROWTH)
            .max(radius);
        let space = &start.space;
        let gq = &start.gq;
        let reduced = Reduced {
            obj,
            set: &set,
            space,
            raw: &start.raw,
            lambda: &start.lambda,
            action,
            hvps: AtomicUsize::new(0),
        };
        let n = x.len();
        let rtol = l2(gq).sqrt().min(0.5);
        let (mut p, mut pred) = if precondition {
            let pre = NystromPrecond::build(&reduced, x.view(), NYSTROM_RANK, self.steps as u64);
            steihaug_pcg(&reduced, x.view(), gq, radius, rtol, 2 * n, &pre)
        } else {
            steihaug_cg(&reduced, x.view(), gq, radius, rtol, 2 * n)
        };
        space.project(&mut p)?;
        let pnorm = l2(&p);
        // The projector is orthogonal in q, so this only trims rounding.
        if radius.is_finite() && pnorm > radius {
            p *= radius / pnorm;
        }
        let trust_norm = l2(&p);
        let caps = self.caps();
        let mut trial_gradients = 0;
        let (rho, boundary, found) = if trust_norm <= 0.0 || !(pred.is_finite() && pred > 0.0) {
            (-1.0, 0.0, None)
        } else {
            match space.place(obj.bounds(), &space.by_root_w(&p), caps) {
                Ok(trial) => {
                    let mut step = space.by_root_m(&(&trial - &*x));
                    space.project(&mut step)?;
                    if steps_differ(step.view(), p.view()) {
                        let hs = reduced.hessian_vector(x.view(), step.view());
                        pred = -(gq.dot(&step) + 0.5 * step.dot(&hs));
                    }
                    let (ft, gt) = obj.value_and_gradient(trial.view());
                    trial_gradients = 1;
                    let rho = if ft.is_finite() {
                        reduction_ratio(start.value - ft, pred)
                    } else {
                        -1.0
                    };
                    let boundary = if caps.atom.is_some() || caps.euclid.is_some() {
                        trust_norm
                    } else {
                        l2(&step)
                    };
                    (rho, boundary, Some((trial, ft, gt)))
                }
                Err(Error::Constraint(_)) => (-1.0, 0.0, None),
                Err(e) => return Err(e),
            }
        };
        let hvps = reduced.hvps.load(Ordering::Relaxed);
        self.hvp_calls += hvps;
        self.gradient_calls += trial_gradients;
        let radius = update_radius(radius, rho, boundary, rmax);
        if let Inner::NewtonCg { radius: r } | Inner::NewtonPcg { radius: r } = &mut self.inner {
            *r = radius;
        }
        let (value, raw, grad) = match found {
            Some((trial, ft, gt)) if rho > ETA_ACCEPT && ft.is_finite() => {
                *x = trial;
                let next = Space::at(&set, x.view(), self.project_rigid)?;
                let (grad, _, _) = next.gradient(&gt)?;
                (ft, gt, grad)
            }
            _ => (start.value, start.raw.clone(), start.grad.clone()),
        };
        self.remember(x, value, &raw);
        self.steps += 1;
        if radius < RADIUS_FLOOR {
            return Err(Error::TrustCollapsed { steps: self.steps });
        }
        Ok(self.report(value, x, &grad))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, ArrayView1, array};

    use super::{Inner, Solver, Space, l2};
    use crate::accept::Accept;
    use crate::holonomic::DistanceConstraints;
    use crate::newton::HessianOracle;
    use crate::{Control, Method};

    fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = x.len() / 3;
        let mut e = 0.0;
        let mut g = Array1::zeros(x.len());
        for i in 0..n {
            for j in i + 1..n {
                let d = [0, 1, 2].map(|c| x[3 * i + c] - x[3 * j + c]);
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let s6 = r2.powi(-3);
                e += 4.0 * (s6 * s6 - s6);
                let f = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
                for c in 0..3 {
                    g[3 * i + c] += f * d[c];
                    g[3 * j + c] -= f * d[c];
                }
            }
        }
        (e, g)
    }

    /// The part of `u` off the tangent space of `space`, over `|u|`.
    fn normal_part(space: &Space<'_>, u: &Array1<f64>) -> f64 {
        let mut p = u.clone();
        space.project(&mut p).unwrap();
        l2(&(u - &p)) / l2(u)
    }

    /// Every stored pair is tangent at the point the next step starts
    /// from, after a step of the session and after a move by the caller.
    /// The projection rounds at `U kappa`, far under `sqrt(U)`; the chord
    /// of a step leaves the tangent space by `|s| / (2 d)`, far over it,
    /// so the threshold tells a transported pair from a raw one.
    #[test]
    fn stored_pairs_are_tangent_at_the_current_point() {
        let half = 104.52_f64.to_radians() / 2.0;
        let (s, c) = (0.9572 * half.sin(), 0.9572 * half.cos());
        let start = array![0.0, 0.0, 0.0, s, c, 0.0, -s, c, 0.0];
        let set = DistanceConstraints::at(vec![(0, 1), (0, 2)], start.view())
            .unwrap()
            .with_masses(vec![15.999, 1.008, 1.008])
            .unwrap();
        let obj = HessianOracle::unbounded(9, lj, |_| ndarray::Array2::zeros((9, 9)));
        let threshold = (f64::EPSILON / 2.0).sqrt();
        let control = Control {
            maxiter: 100,
            gtol: 0.0,
            istep: 0.1,
            maxmove: Some(0.2),
            ..Default::default()
        };
        for accept in [Accept::None, Accept::Energy, Accept::Nonmonotone] {
            let mut solver = Solver::new(Method::lbfgs(), control.clone(), 9);
            solver.set_accept(accept);
            solver.set_distance_constraints(set.clone()).unwrap();
            let mut x = start.clone();
            let mut raw_normal = 0.0_f64;
            for k in 0..6 {
                let before = x.clone();
                solver.step(&obj, &mut x).unwrap();
                if k == 3 {
                    let mut moved = x.clone();
                    moved[4] += 0.02;
                    set.constrain(x.view(), &mut moved).unwrap();
                    x = moved;
                    solver.step(&obj, &mut x).unwrap();
                }
                let space = Space::at(&set, x.view(), false).unwrap();
                let chord = space.by_root_m(&(&x - &before));
                if l2(&chord) > 0.0 {
                    raw_normal = raw_normal.max(normal_part(&space, &chord));
                }
                let Inner::Lbfgs(memory) = &solver.inner else {
                    unreachable!()
                };
                for (s, y) in memory.pairs() {
                    assert!(
                        normal_part(&space, s) <= threshold,
                        "{accept:?} step {k}: s"
                    );
                    assert!(
                        normal_part(&space, y) <= threshold,
                        "{accept:?} step {k}: y"
                    );
                }
            }
            assert!(
                raw_normal > threshold,
                "{accept:?}: raw chords {raw_normal}"
            );
        }
    }
}
