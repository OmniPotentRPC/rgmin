//! Riemannian nonlinear conjugate gradient.
//!
//! The retraction, the tangent projection and the vector transport are
//! the ones on [`Manifold`](crate::manifold::Manifold). The session uses
//! those same maps for an embedded kind, retracts a level set through
//! the session, and returns a holonomic bond to its length with
//! [`crate::holonomic`]. This solver does not define a second copy of
//! either map. Chill, Welborn, Terrell, Zhang, Berthet, Pedersen, Jónsson
//! and Henkelman describe the eOn client whose local step the session
//! replaces,
//! <https://doi.org/10.1088/0965-0393/22/5/055002>.
//! Goswami, Masterov, Kamath, Peña-Torres and Jónsson, Goswami and
//! Jónsson, and Goswami, Gunde and Jónsson drive that session as the
//! local step of a Gaussian-process saddle search,
//! <https://doi.org/10.1021/acs.jctc.5c00866>,
//! <https://doi.org/10.1002/cphc.202500730>,
//! <https://doi.org/10.3389/fchem.2026.1807063>.
//! Absil, Mahony and Sepulchre define the retraction and the transport,
//! <https://doi.org/10.1515/9781400830244>.
//!
//! The coefficient is Fletcher-Reeves, Polak-Ribiere-Polyak with a
//! negative value replaced by zero, or Hager-Zhang. Inner products are
//! the manifold metric at the new point, after the previous gradient and
//! the previous direction have been transported there. Sato,
//! *Riemannian Optimization and Its Applications*,
//! <https://doi.org/10.1007/978-3-030-62391-3>.
//! Fletcher and Reeves, <https://doi.org/10.1093/comjnl/7.2.149>.
//! Gilbert and Nocedal, <https://doi.org/10.1137/0802003>.
//! Hager and Zhang, <https://doi.org/10.1137/030601880>.
//!
//! Ring and Wirth need a transport that does not increase length,
//! <https://doi.org/10.1137/11082885X>.
//! Sato and Iwai scale a transported vector back to the length it had
//! when the transport made it longer, and prove global convergence of
//! Fletcher-Reeves under the strong Wolfe conditions with
//! `0 < c1 < c2 < 1/2`,
//! <https://doi.org/10.1080/02331934.2013.836650>.
//! Sato uses that scaled transport for Dai-Yuan under the weak Wolfe
//! conditions, <https://doi.org/10.1007/s10589-015-9801-1>.
//! [`RcgTransport::Identity`] leaves the vector unchanged, which is the
//! Euclidean transport.
//!
//! A direction whose slope is not negative is replaced by the negative
//! Riemannian gradient.

use std::cell::RefCell;

use ndarray::{Array1, ArrayView1};

use crate::error::{Error, Result};
use crate::holonomic::DistanceConstraints;
use crate::linesearch::{LineSearch, Path};
use crate::manifold::Manifold;
use crate::nlcg::div;
use crate::step::Caps;

/// Stop when the Riemannian gradient norm is at most this value.
///
/// Same number as [`crate::Control::default`]'s `gtol`.
pub const RCG_GRAD_TOL: f64 = 1e-5;

/// Iteration cap of [`RcgOptions::default`].
pub const RCG_MAX_ITER: usize = 200;

/// Length of the first trial displacement along the search direction.
///
/// The opening α is this length divided by the norm of the direction.
/// A later trial reuses the previous accepted displacement length.
/// Same number as [`crate::Control::default`]'s `istep`.
pub const RCG_INITIAL_STEP: f64 = 1.0;

/// Armijo constant of the default strong Wolfe search.
///
/// Sato and Iwai require `0 < c1 < c2 < 1/2`.
pub const RCG_WOLFE_C1: f64 = 1e-4;

/// Curvature constant of the default strong Wolfe search.
///
/// Below one half, as the Fletcher-Reeves argument requires.
pub const RCG_WOLFE_C2: f64 = 0.1;

/// Expand and zoom budget of the default strong Wolfe search.
pub const RCG_WOLFE_MAX_ITER: usize = 20;

/// Hager-Zhang lower-bound parameter η.
///
/// The coefficient is at least `-1 / (||d|| min(η, ||g_old||))`.
pub const RCG_HAGER_ZHANG_ETA: f64 = 0.01;

/// Which β multiplies the transported previous direction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RcgBeta {
    /// `||g_k||^2 / ||g_{k-1}||^2`, with the previous norm taken before transport.
    FletcherReeves,
    /// `max(0, <g_k, g_k - T g_{k-1}> / ||g_{k-1}||^2)`.
    #[default]
    PrpPlus,
    /// Hager-Zhang, with the gradient difference and the direction transported.
    HagerZhang,
}

/// How a tangent vector moves to the next point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RcgTransport {
    /// [`Manifold::transport`], shortened to its original length when that
    /// image is longer. Sato and Iwai's `T^{(k)}`.
    #[default]
    Scaled,
    /// Copy the vector. The Euclidean transport.
    Identity,
}

/// Controls for [`minimize_rcg`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RcgOptions {
    /// Coefficient.
    pub beta: RcgBeta,
    /// Vector transport.
    pub transport: RcgTransport,
    /// Line search along the retraction. The default is strong Wolfe with
    /// [`RCG_WOLFE_C1`], [`RCG_WOLFE_C2`] and [`RCG_WOLFE_MAX_ITER`].
    /// [`LineSearch::default`] is Brent, the derivative-free search.
    pub linesearch: LineSearch,
    /// Stop when the Riemannian gradient norm is at most this value.
    /// [`RCG_GRAD_TOL`] in [`Self::default`].
    pub grad_tol: f64,
    /// Iteration cap. [`RCG_MAX_ITER`] in [`Self::default`].
    pub max_iter: usize,
    /// First trial displacement length. [`RCG_INITIAL_STEP`] in [`Self::default`].
    pub initial_step: f64,
}

with_setters!(RcgOptions {
    with_beta => beta: val RcgBeta,
    with_transport => transport: val RcgTransport,
    with_linesearch => linesearch: val LineSearch,
    with_grad_tol => grad_tol: val f64,
    with_max_iter => max_iter: val usize,
    with_initial_step => initial_step: val f64,
});

impl Default for RcgOptions {
    fn default() -> Self {
        Self {
            beta: RcgBeta::default(),
            transport: RcgTransport::default(),
            linesearch: LineSearch::wolfe(RCG_WOLFE_C1, RCG_WOLFE_C2, RCG_WOLFE_MAX_ITER),
            grad_tol: RCG_GRAD_TOL,
            max_iter: RCG_MAX_ITER,
            initial_step: RCG_INITIAL_STEP,
        }
    }
}

impl RcgOptions {
    /// Refuse a tolerance or step no run could read.
    ///
    /// [`Self::grad_tol`] must be finite and not negative, and
    /// [`Self::initial_step`] finite and positive. A NaN tolerance would
    /// compare false against every gradient norm and run to the iteration
    /// cap; a negative one could never be met.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidControl`] names the setting.
    pub fn validate(&self) -> Result<()> {
        if !(self.grad_tol.is_finite() && self.grad_tol >= 0.0) {
            return Err(Error::InvalidControl {
                what: "rcg grad_tol must be finite and not negative",
            });
        }
        if !(self.initial_step.is_finite() && self.initial_step > 0.0) {
            return Err(Error::InvalidControl {
                what: "rcg initial_step must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Point returned by [`minimize_rcg`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RcgMin {
    /// Accepted point, on the manifold.
    pub x: Array1<f64>,
    /// Objective value at [`Self::x`].
    pub value: f64,
    /// Riemannian gradient norm at [`Self::x`].
    pub grad_norm: f64,
    /// Outer iterations. A start that already meets [`RcgOptions::grad_tol`]
    /// reports zero.
    pub iterations: usize,
    /// Times the search direction was replaced by the negative gradient.
    pub restarts: usize,
    /// Objective evaluations, including the line search.
    pub evals: usize,
}

/// Session memory for one Riemannian CG run.
#[derive(Clone, Debug)]
pub(crate) struct RcgMemory {
    beta: RcgBeta,
    transport: RcgTransport,
    initial_step: f64,
    dir: Array1<f64>,
    prev_alpha: f64,
    prev_speed: f64,
    ready: bool,
    restarts: usize,
}

impl RcgMemory {
    pub(crate) fn new(beta: RcgBeta, transport: RcgTransport, initial_step: f64) -> Self {
        Self {
            beta,
            transport,
            initial_step,
            dir: Array1::zeros(0),
            prev_alpha: 0.0,
            prev_speed: 0.0,
            ready: false,
            restarts: 0,
        }
    }

    fn from_options(opts: &RcgOptions) -> Self {
        Self::new(opts.beta, opts.transport, opts.initial_step)
    }

    /// Whether a conjugate direction is held for the next step.
    #[cfg(test)]
    pub(crate) fn holds_direction(&self) -> bool {
        self.ready
    }

    pub(crate) fn reset(&mut self) {
        self.dir = Array1::zeros(0);
        self.prev_alpha = 0.0;
        self.prev_speed = 0.0;
        self.ready = false;
    }
}

/// One accepted step, still in the ambient coordinates.
pub(crate) struct RcgMoved {
    pub x: Array1<f64>,
    pub value: f64,
    pub egrad: Array1<f64>,
    pub evals: usize,
    pub alpha: f64,
}

/// Failures of the maps a [`Manifold`] cannot report.
///
/// [`Manifold`] returns vectors, not results. A manifold that can fail
/// records the first failure and returns the vector it was given, and the
/// step reads the record after the calls that matter.
pub(crate) trait Failures {
    /// The first failure recorded since the last call. The call clears it.
    fn take(&self) -> Option<Error>;
}

/// A manifold whose maps cannot fail.
pub(crate) struct Infallible;

impl Failures for Infallible {
    fn take(&self) -> Option<Error> {
        None
    }
}

/// Alternating projections between the tangent spaces, and between the
/// two sets, stop at this many sweeps.
const MAX_SWEEPS: usize = 200;

/// Relative gap under which two projections of one vector agree.
const SWEEP_TOL: f64 = 1e-13;

/// A manifold cut by a holonomic bond set.
///
/// The tangent space at a point is the intersection of the tangent space
/// of `base` and the null space of the bond Jacobian. The retraction
/// returns a trial to the bonds along those of the point it leaves, then to
/// `base`, until both hold. A flat `base` such as Euclidean space needs no
/// second sweep, so the bond set alone is the one-pass case.
///
/// [`DistanceConstraints::project_vector`] is the bond tangent projection,
/// [`DistanceConstraints::project_gradient`] the bond part of the
/// Riemannian gradient, and [`DistanceConstraints::constrain`] the return
/// to the bonds. A failure in any of them is recorded and read through
/// [`Failures`].
pub(crate) struct BondedManifold<'a, M: ?Sized> {
    base: &'a M,
    constraints: &'a DistanceConstraints,
    failure: RefCell<Option<Error>>,
}

impl<'a, M> BondedManifold<'a, M>
where
    M: Manifold + ?Sized,
{
    pub(crate) fn new(base: &'a M, constraints: &'a DistanceConstraints) -> Self {
        Self {
            base,
            constraints,
            failure: RefCell::new(None),
        }
    }

    fn fail(&self, error: Error) {
        let mut slot = self.failure.borrow_mut();
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    /// The Riemannian gradient `grad` of `base` at `x`, cut by the bond
    /// tangent space. [`Manifold::egrad2rgrad`] is `base`'s map of an
    /// ambient gradient followed by this.
    pub(crate) fn cut_gradient(&self, x: &Array1<f64>, grad: Array1<f64>) -> Array1<f64> {
        self.tangent(x, grad, |set, at, w| set.project_gradient(at, w).map(drop))
    }

    /// `start` moved into both tangent spaces at `x`. `bond` is the bond
    /// projection applied to it.
    fn tangent<B>(&self, x: &Array1<f64>, start: Array1<f64>, bond: B) -> Array1<f64>
    where
        B: Fn(&DistanceConstraints, ArrayView1<'_, f64>, &mut Array1<f64>) -> Result<()>,
    {
        let mut v = start;
        for _ in 0..MAX_SWEEPS {
            let before = v.clone();
            if let Err(e) = bond(self.constraints, x.view(), &mut v) {
                self.fail(e);
                return before;
            }
            let carried = self.base.project(x, &v);
            if agree(&carried, &v) {
                return carried;
            }
            v = carried;
        }
        self.fail(crate::holonomic::unsupported(
            "a manifold whose tangent space does not meet the bond tangent space",
        ));
        v
    }
}

impl<M> Failures for BondedManifold<'_, M> {
    fn take(&self) -> Option<Error> {
        self.failure.borrow_mut().take()
    }
}

fn agree(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    let scale = 1.0 + crate::vecops::nrm2(a.view()).max(crate::vecops::nrm2(b.view()));
    crate::vecops::nrm2_diff(a.view(), b.view()) <= SWEEP_TOL * scale
}

impl<M> Manifold for BondedManifold<'_, M>
where
    M: Manifold + ?Sized,
{
    fn required_dim(&self, n: usize) -> std::result::Result<(), usize> {
        self.base.required_dim(n)
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.tangent(x, v.clone(), |set, at, w| set.project_vector(at, w))
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        self.cut_gradient(x, self.base.egrad2rgrad(x, egrad))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let zero = Array1::zeros(x.len());
        let mut y = self.base.retract(x, v);
        for _ in 0..MAX_SWEEPS {
            if let Err(e) = self.constraints.constrain(x.view(), &mut y) {
                self.fail(e);
                return x.clone();
            }
            let on_base = self.base.retract(&y, &zero);
            if self.constraints.excess(on_base.view()) <= 1.0 {
                return on_base;
            }
            y = on_base;
        }
        self.fail(crate::holonomic::unsupported(
            "a manifold that does not meet the bond set near the trial",
        ));
        x.clone()
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, &self.base.transport(x_from, x_to, v))
    }

    fn inner(&self, x: &Array1<f64>, u: &Array1<f64>, v: &Array1<f64>) -> f64 {
        self.base.inner(x, u, v)
    }
}

/// Transport `v` from `x_from` to `x_to`.
///
/// [`RcgTransport::Scaled`] calls [`Manifold::transport`] and, when the
/// image is longer than `v` and both lengths are finite and positive,
/// multiplies by `||v|| / ||T v||`. A collapsed image is returned as the
/// transport produced it. [`RcgTransport::Identity`] returns a copy.
pub fn scaled_transport<M>(
    manifold: &M,
    x_from: &Array1<f64>,
    x_to: &Array1<f64>,
    v: &Array1<f64>,
    kind: RcgTransport,
) -> Array1<f64>
where
    M: Manifold + ?Sized,
{
    if matches!(kind, RcgTransport::Identity) {
        return v.clone();
    }
    let carried = manifold.transport(x_from, x_to, v);
    let from = manifold.norm(x_from, v);
    let to = manifold.norm(x_to, &carried);
    if from > 0.0 && to > from && from.is_finite() && to.is_finite() {
        let scale = from / to;
        carried.mapv(|c| c * scale)
    } else {
        carried
    }
}

/// Minimize `fg` on `manifold` from `x0`.
///
/// `fg` returns the value and the Euclidean gradient. The Riemannian
/// gradient is [`Manifold::egrad2rgrad`]. The start is retracted by a
/// zero tangent step so it lies on the manifold.
///
/// # Errors
///
/// [`Error::InvalidControl`] from [`RcgOptions::validate`],
/// [`Error::ManifoldDim`] for a start the manifold cannot hold,
/// [`Error::Oracle`] for a non-finite value or gradient, and the typed
/// failure of a projection or retraction that could not be carried out.
pub fn minimize_rcg<M, F>(
    manifold: &M,
    x0: Array1<f64>,
    mut fg: F,
    opts: &RcgOptions,
) -> Result<RcgMin>
where
    M: Manifold + ?Sized,
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    opts.validate()?;
    if manifold.required_dim(x0.len()).is_err() {
        return Err(Error::ManifoldDim {
            kind: "rcg",
            got: x0.len(),
        });
    }
    let zero = Array1::zeros(x0.len());
    let mut x = manifold.retract(&x0, &zero);
    let mut memory = RcgMemory::from_options(opts);
    let mut evals = 0usize;
    let (mut value, mut egrad) = evaluate(&mut fg, x.view(), &mut evals)?;
    let mut grad_norm = rgrad_norm(manifold, &x, &egrad)?;
    let mut iterations = 0usize;
    while iterations < opts.max_iter && grad_norm > opts.grad_tol {
        let moved = rcg_step(
            manifold,
            &Infallible,
            &mut memory,
            &x,
            value,
            &egrad,
            opts.linesearch,
            Caps::default(),
            &mut fg,
        )?;
        evals += moved.evals;
        iterations += 1;
        let stalled = !moved.alpha.is_finite() || moved.alpha <= 0.0;
        x = moved.x;
        value = moved.value;
        egrad = moved.egrad;
        grad_norm = rgrad_norm(manifold, &x, &egrad)?;
        if stalled {
            break;
        }
    }
    Ok(RcgMin {
        x,
        value,
        grad_norm,
        iterations,
        restarts: memory.restarts,
        evals,
    })
}

/// One Riemannian CG step from `x`. `egrad` is the Euclidean gradient there.
///
/// `caps` bound the displacement of every trial, measured on the retracted
/// point, the way the session bounds the step of every other method.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rcg_step<M, F>(
    manifold: &M,
    failures: &dyn Failures,
    memory: &mut RcgMemory,
    x: &Array1<f64>,
    value: f64,
    egrad: &Array1<f64>,
    linesearch: LineSearch,
    caps: Caps,
    mut fg: F,
) -> Result<RcgMoved>
where
    M: Manifold + ?Sized,
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    if egrad.len() != x.len() || egrad.iter().any(|g| !g.is_finite()) || !value.is_finite() {
        return Err(crate::step::non_finite());
    }
    let rgrad = manifold.egrad2rgrad(x, egrad);
    if let Some(failure) = failures.take() {
        return Err(failure);
    }
    if rgrad.len() != x.len() || rgrad.iter().any(|g| !g.is_finite()) {
        return Err(crate::step::non_finite());
    }
    let mut dir = search_direction(manifold, memory, x, &rgrad);
    let mut previous = memory.prev_alpha;
    let mut previous_speed = memory.prev_speed;
    let mut calls = 0usize;
    let mut counted = |p: ArrayView1<'_, f64>| {
        calls += 1;
        fg(p)
    };
    for attempt in 0..2 {
        let speed = manifold.norm(x, &dir);
        let alpha0 = opening_step(speed, memory.initial_step, previous, previous_speed)
            .min(caps.alpha(dir.view()));
        if !alpha0.is_finite() || alpha0 <= 0.0 {
            break;
        }
        let mut path = RetractPath {
            manifold,
            failures,
            origin: x.clone(),
            dir: dir.clone(),
            caps,
            failure: None,
            placed: false,
        };
        let (x_new, _f_new, alpha) = linesearch.search_path(&mut counted, &mut path, alpha0);
        if alpha > 0.0 {
            let (value_new, egrad_new) = counted(x_new.view());
            if !value_new.is_finite()
                || egrad_new.len() != x_new.len()
                || egrad_new.iter().any(|g| !g.is_finite())
            {
                return Err(crate::step::non_finite());
            }
            let rgrad_new = manifold.egrad2rgrad(&x_new, &egrad_new);
            let (next, restarted) = next_direction(
                manifold,
                memory.beta,
                memory.transport,
                x,
                &x_new,
                &rgrad,
                &rgrad_new,
                &dir,
            );
            if let Some(failure) = failures.take() {
                return Err(failure);
            }
            if restarted {
                memory.restarts += 1;
            }
            memory.dir = next;
            memory.prev_alpha = alpha;
            memory.prev_speed = speed;
            memory.ready = true;
            return Ok(RcgMoved {
                x: x_new,
                value: value_new,
                egrad: egrad_new,
                evals: calls,
                alpha,
            });
        }
        if !path.placed
            && let Some(failure) = path.failure.take()
        {
            return Err(failure);
        }
        if attempt == 1 || steepest(manifold, x, &rgrad, &dir) {
            break;
        }
        dir = negated(&rgrad);
        memory.restarts += 1;
        previous = 0.0;
        previous_speed = 0.0;
    }
    memory.ready = false;
    memory.prev_alpha = 0.0;
    memory.prev_speed = 0.0;
    memory.dir = negated(&rgrad);
    Ok(RcgMoved {
        x: x.clone(),
        value,
        egrad: egrad.clone(),
        evals: calls,
        alpha: 0.0,
    })
}

fn search_direction<M>(
    manifold: &M,
    memory: &mut RcgMemory,
    x: &Array1<f64>,
    rgrad: &Array1<f64>,
) -> Array1<f64>
where
    M: Manifold + ?Sized,
{
    if memory.ready && memory.dir.len() == x.len() {
        let dir = memory.dir.clone();
        if dir.iter().all(|v| v.is_finite()) && manifold.inner(x, rgrad, &dir) < 0.0 {
            return dir;
        }
        memory.restarts += 1;
    }
    negated(rgrad)
}

/// Conjugated direction at `x_to`, and whether it was replaced by steepest descent.
fn next_direction<M>(
    manifold: &M,
    beta_kind: RcgBeta,
    transport: RcgTransport,
    x_from: &Array1<f64>,
    x_to: &Array1<f64>,
    g_from: &Array1<f64>,
    g_to: &Array1<f64>,
    d_from: &Array1<f64>,
) -> (Array1<f64>, bool)
where
    M: Manifold + ?Sized,
{
    let g_carried = scaled_transport(manifold, x_from, x_to, g_from, transport);
    let d_carried = scaled_transport(manifold, x_from, x_to, d_from, transport);
    let beta = coefficient(
        manifold, beta_kind, x_from, x_to, g_from, g_to, d_from, &g_carried, &d_carried,
    );
    let mut next = negated(g_to);
    if beta.is_finite() {
        crate::vecops::axpy(beta, d_carried.view(), &mut next);
    }
    next = manifold.project(x_to, &next);
    let slope = manifold.inner(x_to, g_to, &next);
    if !beta.is_finite() || !slope.is_finite() || slope >= 0.0 {
        return (manifold.project(x_to, &negated(g_to)), true);
    }
    (next, false)
}

fn coefficient<M>(
    manifold: &M,
    kind: RcgBeta,
    x_from: &Array1<f64>,
    x_to: &Array1<f64>,
    g_from: &Array1<f64>,
    g_to: &Array1<f64>,
    d_from: &Array1<f64>,
    g_carried: &Array1<f64>,
    d_carried: &Array1<f64>,
) -> f64
where
    M: Manifold + ?Sized,
{
    let old2 = manifold.inner(x_from, g_from, g_from);
    let new2 = manifold.inner(x_to, g_to, g_to);
    match kind {
        RcgBeta::FletcherReeves => div(new2, old2),
        RcgBeta::PrpPlus => {
            let mut y = g_to.clone();
            crate::vecops::axpy(-1.0, g_carried.view(), &mut y);
            div(manifold.inner(x_to, g_to, &y), old2).max(0.0)
        }
        RcgBeta::HagerZhang => hager_zhang(
            manifold, x_from, x_to, g_from, g_to, d_from, g_carried, d_carried,
        ),
    }
}

fn hager_zhang<M>(
    manifold: &M,
    x_from: &Array1<f64>,
    x_to: &Array1<f64>,
    g_from: &Array1<f64>,
    g_to: &Array1<f64>,
    d_from: &Array1<f64>,
    g_carried: &Array1<f64>,
    d_carried: &Array1<f64>,
) -> f64
where
    M: Manifold + ?Sized,
{
    let mut y = g_to.clone();
    crate::vecops::axpy(-1.0, g_carried.view(), &mut y);
    let y_d = manifold.inner(x_to, d_carried, &y);
    let y_g = manifold.inner(x_to, &y, g_to);
    let yy = manifold.inner(x_to, &y, &y);
    let d_g = manifold.inner(x_to, d_carried, g_to);
    let scale = y_g.abs() + yy.abs() + d_g.abs() + 1.0;
    let mut beta = if !y_d.is_finite() || y_d.abs() <= f64::EPSILON * scale {
        0.0
    } else {
        y_g / y_d - 2.0 * yy * d_g / (y_d * y_d)
    };
    let ng = manifold.norm(x_from, g_from);
    let nd = manifold.norm(x_from, d_from);
    let cap = if ng < RCG_HAGER_ZHANG_ETA {
        ng
    } else {
        RCG_HAGER_ZHANG_ETA
    };
    if nd > 0.0 && cap > 0.0 && nd.is_finite() && cap.is_finite() {
        let floor = -1.0 / (nd * cap);
        if floor.is_finite() && beta < floor {
            beta = floor;
        }
    }
    if beta.is_finite() { beta } else { 0.0 }
}

fn opening_step(speed: f64, initial: f64, previous: f64, previous_speed: f64) -> f64 {
    if !speed.is_finite() || speed <= 0.0 {
        return 0.0;
    }
    if previous > 0.0 && previous.is_finite() && previous_speed > 0.0 && previous_speed.is_finite()
    {
        let carried = previous * previous_speed / speed;
        if carried.is_finite() && carried > 0.0 {
            return carried;
        }
    }
    if initial.is_finite() && initial > 0.0 {
        initial / speed
    } else {
        0.0
    }
}

fn steepest<M>(manifold: &M, x: &Array1<f64>, rgrad: &Array1<f64>, dir: &Array1<f64>) -> bool
where
    M: Manifold + ?Sized,
{
    let g2 = manifold.inner(x, rgrad, rgrad);
    let d2 = manifold.inner(x, dir, dir);
    let gd = manifold.inner(x, rgrad, dir);
    let tol = f64::EPSILON * (g2.abs() + 1.0);
    (gd + g2).abs() <= tol && (d2 - g2).abs() <= tol
}

fn negated(v: &Array1<f64>) -> Array1<f64> {
    v.mapv(|c| -c)
}

fn rgrad_norm<M>(manifold: &M, x: &Array1<f64>, egrad: &Array1<f64>) -> Result<f64>
where
    M: Manifold + ?Sized,
{
    if egrad.len() != x.len() || egrad.iter().any(|g| !g.is_finite()) {
        return Err(crate::step::non_finite());
    }
    let g = manifold.egrad2rgrad(x, egrad);
    let n = manifold.norm(x, &g);
    if n.is_finite() {
        Ok(n)
    } else {
        Err(crate::step::non_finite())
    }
}

fn evaluate<F>(fg: &mut F, x: ArrayView1<'_, f64>, evals: &mut usize) -> Result<(f64, Array1<f64>)>
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    *evals += 1;
    let (value, grad) = fg(x);
    if !value.is_finite() || grad.len() != x.len() || grad.iter().any(|g| !g.is_finite()) {
        return Err(crate::step::non_finite());
    }
    Ok((value, grad))
}

/// The curve `t |-> retract(origin, t dir)`, undefined where its point
/// is farther from `origin` than `caps` allow.
///
/// A trial the manifold failed to retract is undefined too. The first such
/// failure is kept, and the step reports it when no trial was ever placed.
struct RetractPath<'a, M: ?Sized> {
    manifold: &'a M,
    failures: &'a dyn Failures,
    origin: Array1<f64>,
    dir: Array1<f64>,
    caps: Caps,
    failure: Option<Error>,
    placed: bool,
}

impl<M: ?Sized> RetractPath<'_, M> {
    /// `true` when the manifold failed since the last call.
    fn failed(&mut self) -> bool {
        match self.failures.take() {
            Some(failure) => {
                self.failure.get_or_insert(failure);
                true
            }
            None => false,
        }
    }
}

impl<M> Path for RetractPath<'_, M>
where
    M: Manifold + ?Sized,
{
    fn start(&self) -> Array1<f64> {
        self.origin.clone()
    }

    fn direction(&self) -> ArrayView1<'_, f64> {
        self.dir.view()
    }

    fn point(&mut self, t: f64) -> Option<Array1<f64>> {
        if !t.is_finite() {
            return None;
        }
        let step = self.dir.mapv(|c| c * t);
        let point = self.manifold.retract(&self.origin, &step);
        if self.failed() || !self.caps.admits(&self.origin, &point) {
            return None;
        }
        self.placed = true;
        Some(point)
    }

    fn slope(&mut self, t: f64, at: ArrayView1<'_, f64>, g: ArrayView1<'_, f64>) -> f64 {
        let at_owned = at.to_owned();
        let vel = retraction_velocity(self.manifold, &self.origin, &self.dir, t, &at_owned);
        let rg = self.manifold.egrad2rgrad(&at_owned, &g.to_owned());
        if self.failed() {
            return f64::NAN;
        }
        self.manifold.inner(&at_owned, &rg, &vel)
    }
}

/// Velocity of `t |-> retract(origin, t dir)`.
///
/// At the foot the velocity is `dir`, because a retraction's derivative
/// there is the identity on the tangent space. Away from the foot the
/// velocity is a central difference of [`Manifold::retract`] when the
/// sample stays positive, and a forward difference otherwise, then
/// [`Manifold::project`] at the arrival point. The spacing is the square
/// root of machine epsilon.
fn retraction_velocity<M>(
    manifold: &M,
    origin: &Array1<f64>,
    dir: &Array1<f64>,
    t: f64,
    at: &Array1<f64>,
) -> Array1<f64>
where
    M: Manifold + ?Sized,
{
    if !t.is_finite() || t <= 0.0 {
        return dir.clone();
    }
    let h = f64::EPSILON.sqrt();
    let ahead = manifold.retract(origin, &dir.mapv(|c| c * (t + h)));
    let vel = if t > h {
        let behind = manifold.retract(origin, &dir.mapv(|c| c * (t - h)));
        let mut diff = ahead;
        crate::vecops::axpy(-1.0, behind.view(), &mut diff);
        diff.mapv(|c| c / (2.0 * h))
    } else {
        let here = manifold.retract(origin, &dir.mapv(|c| c * t));
        let mut diff = ahead;
        crate::vecops::axpy(-1.0, here.view(), &mut diff);
        diff.mapv(|c| c / h)
    };
    manifold.project(at, &vel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{Sphere, StiefelNp};
    use crate::{Control, DistanceConstraints, Method, Oracle, Solver};
    use ndarray::array;

    #[test]
    fn options_refuse_a_tolerance_or_step_no_run_could_read() {
        let bowl = |x: ArrayView1<f64>| (x.dot(&x), x.mapv(|v| 2.0 * v));
        let bad = [
            RcgOptions::default().with_grad_tol(f64::NAN),
            RcgOptions::default().with_grad_tol(f64::INFINITY),
            RcgOptions::default().with_grad_tol(-1e-8),
            RcgOptions::default().with_initial_step(f64::NAN),
            RcgOptions::default().with_initial_step(f64::INFINITY),
            RcgOptions::default().with_initial_step(0.0),
            RcgOptions::default().with_initial_step(-1.0),
        ];
        for opts in &bad {
            assert!(matches!(opts.validate(), Err(Error::InvalidControl { .. })));
            let got = minimize_rcg(&crate::manifold::Euclidean, array![1.0, 2.0], bowl, opts);
            assert!(matches!(got, Err(Error::InvalidControl { .. })), "{opts:?}");
        }
        assert!(RcgOptions::default().validate().is_ok());
        assert!(RcgOptions::default().with_grad_tol(0.0).validate().is_ok());
    }

    fn rayleigh(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let a = [3.0, 1.0, 2.0];
        let mut f = 0.0;
        let mut g = Array1::zeros(3);
        for i in 0..3 {
            f += a[i] * x[i] * x[i];
            g[i] = 2.0 * a[i] * x[i];
        }
        (f, g)
    }

    #[test]
    fn sphere_rayleigh_reaches_the_smallest_eigenvalue() {
        let sphere = Sphere;
        let x = array![0.6, 0.48, 0.64];
        for beta in [
            RcgBeta::FletcherReeves,
            RcgBeta::PrpPlus,
            RcgBeta::HagerZhang,
        ] {
            let out = minimize_rcg(
                &sphere,
                x.clone(),
                rayleigh,
                &RcgOptions::default()
                    .with_beta(beta)
                    .with_grad_tol(1e-6)
                    .with_max_iter(80),
            )
            .unwrap();
            assert!(
                (out.value - 1.0).abs() < 1e-6,
                "{beta:?} f = {} after {}",
                out.value,
                out.iterations
            );
            assert!(out.grad_norm <= 1e-6, "{beta:?} grad {}", out.grad_norm);
            assert!(out.x[1].abs() > 0.99, "{beta:?} x = {}", out.x);
            let n = out.x.dot(&out.x).sqrt();
            assert!((n - 1.0).abs() < 1e-8, "{beta:?} norm {n}");
        }
    }

    #[test]
    fn a_cap_bounds_the_retracted_move_on_a_curved_manifold() {
        let sphere = Sphere;
        let x = array![0.6, 0.48, 0.64];
        let (value, egrad) = rayleigh(x.view());
        let caps = Caps {
            atom: None,
            euclid: Some(0.05),
        };
        let mut memory = RcgMemory::from_options(&RcgOptions::default());
        let moved = rcg_step(
            &sphere,
            &Infallible,
            &mut memory,
            &x,
            value,
            &egrad,
            RcgOptions::default().linesearch,
            caps,
            rayleigh,
        )
        .unwrap();
        let length = crate::vecops::nrm2_diff(moved.x.view(), x.view());
        assert!(moved.alpha > 0.0, "no step");
        assert!(length <= 0.05 * (1.0 + 1e-9), "moved {length}");
        assert!(moved.value < value);
    }

    #[test]
    fn a_bonded_sphere_is_tangent_to_both_and_retracts_onto_both() {
        let x = array![0.6, 0.0, 0.0, 0.0, 0.8, 0.0];
        let set = DistanceConstraints::at(vec![(0, 1)], x.view()).unwrap();
        let bonded = BondedManifold::new(&Sphere, &set);
        let egrad = array![0.3, -0.2, 1.0, 0.1, 0.5, 2.0];
        let rgrad = bonded.egrad2rgrad(&x, &egrad);
        let bond = |v: &Array1<f64>, w: &Array1<f64>| {
            (0..3)
                .map(|k| (v[k] - v[3 + k]) * (w[k] - w[3 + k]))
                .sum::<f64>()
        };
        assert!(x.dot(&rgrad).abs() < 1e-12, "normal part {}", x.dot(&rgrad));
        assert!(
            bond(&x, &rgrad).abs() < 1e-12,
            "bond part {}",
            bond(&x, &rgrad)
        );
        let y = bonded.retract(&x, &rgrad.mapv(|c| -0.3 * c));
        assert!((y.dot(&y).sqrt() - 1.0).abs() < 1e-12);
        assert!(set.excess(y.view()) <= 1.0);
        assert!(bonded.take().is_none());
    }

    fn brockett(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let a = [1.0, 2.0, 3.0, 4.0];
        let nmu = [1.0, 2.0];
        let mut f = 0.0;
        let mut g = Array1::zeros(8);
        for j in 0..2 {
            let mut q = 0.0;
            for i in 0..4 {
                let xi = x[j * 4 + i];
                q += a[i] * xi * xi;
                g[j * 4 + i] = a[i] * nmu[j] * xi;
            }
            f += 0.5 * nmu[j] * q;
        }
        (f, g)
    }

    #[test]
    fn stiefel_brockett_reaches_the_known_minimum() {
        let stiefel = StiefelNp::new(4, 2).unwrap();
        // The cost is the Brockett function from the retained-step test.
        // The minimizer puts the larger weight on the smaller eigenvalue.
        // A short retraction off that frame stays in its basin. The frame
        // used as a cold start in that test is one retraction away from a
        // different critical point.
        let minimizer = array![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let step = stiefel.project(&minimizer, &array![0.0, 0.0, 0.3, 0.1, 0.0, 0.2, 0.0, 0.15]);
        let x = stiefel.retract(&minimizer, &step);
        let (start_value, _) = brockett(x.view());
        assert!(start_value > 2.05, "start f {start_value}");
        let out = minimize_rcg(
            &stiefel,
            x,
            brockett,
            &RcgOptions::default()
                .with_beta(RcgBeta::HagerZhang)
                .with_grad_tol(1e-6)
                .with_max_iter(120),
        )
        .unwrap();
        assert!(
            (out.value - 2.0).abs() < 1e-4,
            "f = {} after {}",
            out.value,
            out.iterations
        );
        assert!(out.grad_norm <= 1e-5, "grad {}", out.grad_norm);
        let mut gram = [[0.0; 2]; 2];
        for j in 0..2 {
            for k in 0..2 {
                let mut dot = 0.0;
                for i in 0..4 {
                    dot += out.x[j * 4 + i] * out.x[k * 4 + i];
                }
                gram[j][k] = dot;
            }
        }
        assert!((gram[0][0] - 1.0).abs() < 1e-6, "{gram:?}");
        assert!((gram[1][1] - 1.0).abs() < 1e-6, "{gram:?}");
        assert!(gram[0][1].abs() < 1e-6, "{gram:?}");
    }

    fn rosenbrock(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let t = x[1] - x[0] * x[0];
        let f = 100.0 * t * t + (1.0 - x[0]) * (1.0 - x[0]);
        let g = array![-400.0 * x[0] * t + 2.0 * (x[0] - 1.0), 200.0 * t];
        (f, g)
    }

    #[test]
    fn euclidean_rosenbrock_with_identity_transport() {
        let out = minimize_rcg(
            &crate::manifold::Euclidean,
            array![-1.2, 1.0],
            rosenbrock,
            &RcgOptions::default()
                .with_transport(RcgTransport::Identity)
                .with_grad_tol(1e-6)
                .with_max_iter(300),
        )
        .unwrap();
        assert!(
            out.value < 1e-8,
            "f = {} after {} restarts {}",
            out.value,
            out.iterations,
            out.restarts
        );
        assert!((out.x[0] - 1.0).abs() < 1e-4, "{}", out.x);
        assert!((out.x[1] - 1.0).abs() < 1e-4, "{}", out.x);
    }

    /// A transport that multiplies every vector by 20.
    ///
    /// Orthogonal projections never lengthen a vector, so the sphere cannot
    /// show the scale. Sato and Iwai scale only when the image is longer.
    struct Inflate;

    const INFLATE: f64 = 20.0;

    impl Manifold for Inflate {
        fn project(&self, _x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
            v.clone()
        }
        fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
            let mut y = x.clone();
            crate::vecops::axpy(1.0, v.view(), &mut y);
            y
        }
        fn transport(
            &self,
            _x_from: &Array1<f64>,
            _x_to: &Array1<f64>,
            v: &Array1<f64>,
        ) -> Array1<f64> {
            v.mapv(|c| c * INFLATE)
        }
    }

    #[test]
    fn scaled_transport_keeps_the_fletcher_reeves_direction_in_descent() {
        // g_old = (1, 0), d = -g_old, g_new = (-0.1, 0). The raw image has
        // length 20. Fletcher-Reeves then gives beta = 0.01, and
        // -g_new + beta * T_raw(d) = (-0.1, 0) has a positive slope.
        // The scaled image keeps the length, and the slope stays negative,
        // so the update is not restarted.
        let from = array![0.0, 0.0];
        let to = array![1.0, 0.0];
        let g_from = array![1.0, 0.0];
        let g_to = array![-0.1, 0.0];
        let d_from = array![-1.0, 0.0];
        let carried = scaled_transport(&Inflate, &from, &to, &d_from, RcgTransport::Scaled);
        let carried_norm = Inflate.norm(&to, &carried);
        let from_norm = Inflate.norm(&from, &d_from);
        assert!(
            carried_norm <= from_norm + 1e-12,
            "scaled length {carried_norm} exceeds {from_norm}"
        );
        let (dir, restarted) = next_direction(
            &Inflate,
            RcgBeta::FletcherReeves,
            RcgTransport::Scaled,
            &from,
            &to,
            &g_from,
            &g_to,
            &d_from,
        );
        assert!(
            !restarted,
            "scaled update was restarted; slope {}",
            Inflate.inner(&to, &g_to, &dir)
        );
        assert!(
            Inflate.inner(&to, &g_to, &dir) < 0.0,
            "slope {}",
            Inflate.inner(&to, &g_to, &dir)
        );
        let raw = Inflate.transport(&from, &to, &d_from);
        let beta = div(g_to.dot(&g_to), g_from.dot(&g_from));
        let mut raw_dir = negated(&g_to);
        crate::vecops::axpy(beta, raw.view(), &mut raw_dir);
        assert!(
            Inflate.inner(&to, &g_to, &raw_dir) > 0.0,
            "the unscaled direction was expected to ascend"
        );
    }

    #[test]
    fn identity_transport_copies_the_vector() {
        let from = array![0.0, 0.0];
        let to = array![1.0, 0.0];
        let v = array![-1.0, 0.0];
        let copied = scaled_transport(&Inflate, &from, &to, &v, RcgTransport::Identity);
        assert_eq!(copied, v);
    }

    #[test]
    fn session_on_a_bond_matches_the_bond_manifold_step() {
        let start = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let constraints = DistanceConstraints::at(vec![(0, 1)], start.view()).unwrap();
        let fg = |x: ArrayView1<f64>| {
            let f = 0.5 * (x[1] * x[1] + (x[4] - 1.0) * (x[4] - 1.0));
            let mut g = Array1::zeros(6);
            g[1] = x[1];
            g[4] = x[4] - 1.0;
            (f, g)
        };
        let mut session_x = start.clone();
        let mut solver = Solver::new(
            Method::rcg(),
            Control::default().with_gtol(1e-12).with_maxiter(4),
            6,
        );
        solver
            .set_distance_constraints(DistanceConstraints::at(vec![(0, 1)], start.view()).unwrap())
            .unwrap();
        let obj = Oracle::unbounded(6, fg);
        let mut direct = start.clone();
        let mut memory = RcgMemory::from_options(&RcgOptions::default());
        let bond = BondedManifold::new(&crate::manifold::Euclidean, &constraints);
        for _ in 0..2 {
            let (value, grad) = fg(direct.view());
            let moved = rcg_step(
                &bond,
                &bond,
                &mut memory,
                &direct,
                value,
                &grad,
                RcgOptions::default().linesearch,
                Caps::default(),
                fg,
            )
            .unwrap();
            direct = moved.x;
            solver.step(&obj, &mut session_x).unwrap();
        }
        for (a, b) in session_x.iter().zip(direct.iter()) {
            assert!((a - b).abs() < 1e-8, "session {session_x} direct {direct}");
        }
        let bond_vec = array![
            session_x[3] - session_x[0],
            session_x[4] - session_x[1],
            session_x[5] - session_x[2],
        ];
        let distance = bond_vec.dot(&bond_vec).sqrt();
        assert!((distance - 1.0).abs() < 1e-8, "distance {distance}");
        let energy = fg(session_x.view()).0;
        assert!(energy < 0.5, "energy {energy}");
    }
}
