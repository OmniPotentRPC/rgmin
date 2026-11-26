//! Cautious limited-memory Riemannian BFGS.
//!
//! Algorithm 2 of Huang, Absil, and Gallivan, *A Riemannian BFGS Method
//! Without Differentiated Retraction for Nonconvex Optimization Problems*,
//! SIAM J. Optim. 28 (2018) 470–495,
//! <https://doi.org/10.1137/17M1127582>. The two-loop map is the
//! limited-memory inverse Hessian of Huang, Gallivan, and Absil,
//! SIAM J. Optim. 25 (2015) 1660–1685,
//! <https://doi.org/10.1137/140955483>. On a Euclidean manifold the
//! transport is the identity and the update is the cautious BFGS method
//! of Li and Fukushima, SIAM J. Optim. 11 (2001) 1054–1064,
//! <https://doi.org/10.1137/S1052623499354242>.
//!
//! The search direction is the two-loop product. The step length is the
//! Armijo–Goldstein backtrack from a unit trial, which does not need a
//! differentiated retraction. A new pair is stored only when it passes
//! the cautious curvature test. The vector transport is the parallel
//! transport of the exponential map when that pair exists, and the
//! retraction transport otherwise.
//!
//! This is not [`crate::minimize_lbfgs`]. The production Euclidean step
//! stays [`crate::LbfgsStep::Capped`] at [`crate::EON_MAX_MOVE`].

use eindir_core::DifferentiableObjective;
use ndarray::Array1;

use crate::control::Control;
use crate::error::{Error, Result};
use crate::manifold::Manifold;
use crate::observer::IterationObserver;
use crate::report::{Report, Status, StopReason};
use crate::step::one_shot_report;
use crate::stop::{Counted, StopRules, counted};
use crate::vecops::{axpy, scale};

/// Cautious slope in `vartheta(t) = 10^{-4} t` (Huang, Absil, Gallivan 2018, section 7).
pub const HUANG_CAUTIOUS_SLOPE: f64 = 1.0 / 10_000.0;
/// Armijo constant `sigma` of the same experiments. The paper requires `sigma < 1/2`.
pub const HUANG_ARMIJO: f64 = 1.0 / 10_000.0;
/// Shrinkage of the unit trial when the bracket parameters agree (section 7).
pub const HUANG_BACKTRACK: f64 = 1.0 / 2.0;
/// Relative gradient stop `||g_f|| / ||g_0|| < 10^{-6}` (section 7.2).
pub const HUANG_GRADIENT_RATIO: f64 = 1.0 / 1_000_000.0;
/// Limited memory `m` used in section 7.
pub const HUANG_MEMORY: usize = 4;
/// Armijo trials from the unit step. Each trial halves the length.
pub const HUANG_LINESEARCH_MAX: usize = 48;
/// Upper bound on the Armijo constant in the paper (`sigma < 1/2`).
const ARMIJO_LIMIT: f64 = 1.0 / 2.0;
/// Initial inverse Hessian is the identity (section 7.2).
const INITIAL_SCALE: f64 = 1.0;

/// Memory, cautious test, and Armijo backtrack for [`minimize_rlbfgs`].
///
/// The defaults are the section 7 choices: memory 4, `vartheta(t) = 10^{-4} t`,
/// Armijo constant `10^{-4}`, backtrack `1/2`, and the relative gradient stop
/// `10^{-6}`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Rlbfgs {
    /// Number of curvature pairs. The oldest is dropped past this length.
    pub memory: usize,
    /// Cautious threshold factor. A pair is kept when `y·s / ||s||^2` is at least
    /// this factor times the gradient norm at the point the step left.
    pub cautious_slope: f64,
    /// Armijo sufficient-decrease factor.
    pub armijo: f64,
    /// Factor applied to a rejected trial length. It lies in `(0, 1)`.
    pub backtrack: f64,
    /// Rejected trials allowed on one iteration, including the unit trial.
    pub linesearch_max: usize,
    /// Stop when `||grad|| / ||grad_0||` falls below this. Zero leaves the
    /// ratio stop off and [`Control::gtol`] decides.
    pub grad_ratio: f64,
}

with_setters!(Rlbfgs {
    with_memory => memory: val usize,
    with_cautious_slope => cautious_slope: val f64,
    with_armijo => armijo: val f64,
    with_backtrack => backtrack: val f64,
    with_linesearch_max => linesearch_max: val usize,
    with_grad_ratio => grad_ratio: val f64,
});

impl Default for Rlbfgs {
    fn default() -> Self {
        Self {
            memory: HUANG_MEMORY,
            cautious_slope: HUANG_CAUTIOUS_SLOPE,
            armijo: HUANG_ARMIJO,
            backtrack: HUANG_BACKTRACK,
            linesearch_max: HUANG_LINESEARCH_MAX,
            grad_ratio: HUANG_GRADIENT_RATIO,
        }
    }
}

impl Rlbfgs {
    /// Refuse a record Algorithm 2 cannot run.
    pub fn check(&self) -> Result<()> {
        if self.memory == 0 {
            return Err(Error::InvalidControl {
                what: "rlbfgs memory must be at least 1",
            });
        }
        if !(self.cautious_slope.is_finite() && self.cautious_slope >= 0.0) {
            return Err(Error::InvalidControl {
                what: "rlbfgs cautious_slope must be finite and non-negative",
            });
        }
        if !(self.armijo.is_finite() && self.armijo > 0.0 && self.armijo < ARMIJO_LIMIT) {
            return Err(Error::InvalidControl {
                what: "rlbfgs armijo must be finite and lie in (0, 1/2)",
            });
        }
        if !(self.backtrack.is_finite() && self.backtrack > 0.0 && self.backtrack < 1.0) {
            return Err(Error::InvalidControl {
                what: "rlbfgs backtrack must be finite and lie in (0, 1)",
            });
        }
        if self.linesearch_max == 0 {
            return Err(Error::InvalidControl {
                what: "rlbfgs linesearch_max must be at least 1",
            });
        }
        if !(self.grad_ratio.is_finite() && self.grad_ratio >= 0.0) {
            return Err(Error::InvalidControl {
                what: "rlbfgs grad_ratio must be finite and non-negative",
            });
        }
        Ok(())
    }
}

struct Pair {
    s: Array1<f64>,
    y: Array1<f64>,
    rho: f64,
}

/// Two-loop inverse Hessian applied to `grad` (Nocedal–Wright 7.4).
///
/// Pairs are oldest first. `gamma` is `(s·y) / (y·y)` of the newest accepted
/// pair, or 1 when the memory is empty. The result is `H grad`, not the
/// search direction.
fn inverse_hessian_times<M>(
    pairs: &[Pair],
    gamma: f64,
    manifold: &M,
    x: &Array1<f64>,
    grad: &Array1<f64>,
) -> Array1<f64>
where
    M: Manifold + ?Sized,
{
    let m = pairs.len();
    let mut alpha = vec![0.0; m];
    let mut q = grad.clone();
    for i in (0..m).rev() {
        alpha[i] = pairs[i].rho * manifold.inner(x, &pairs[i].s, &q);
        axpy(-alpha[i], pairs[i].y.view(), &mut q);
    }
    scale(gamma, &mut q);
    for i in 0..m {
        let beta = pairs[i].rho * manifold.inner(x, &pairs[i].y, &q);
        axpy(alpha[i] - beta, pairs[i].s.view(), &mut q);
    }
    q
}

fn all_finite(v: &Array1<f64>) -> bool {
    v.iter().all(|entry| entry.is_finite())
}

fn ratio_met(gnorm: f64, g0: f64, params: &Rlbfgs) -> bool {
    g0 > 0.0 && gnorm.is_finite() && gnorm / g0 < params.grad_ratio
}

fn trial_scale(eta: &Array1<f64>, control: &Control) -> f64 {
    let Some(cap) = control.maxmove else {
        return INITIAL_SCALE;
    };
    if !(cap.is_finite() && cap > 0.0) {
        return INITIAL_SCALE;
    }
    let length = crate::step::l2(eta);
    if length.is_finite() && length > cap {
        cap / length
    } else {
        INITIAL_SCALE
    }
}

/// Exponential map plus its parallel transport, or the retraction and its
/// vector transport when the geodesic pair is missing.
fn advance<M>(
    manifold: &M,
    x: &Array1<f64>,
    step: &Array1<f64>,
    cargo: &[Array1<f64>],
) -> (Array1<f64>, Vec<Array1<f64>>)
where
    M: Manifold + ?Sized,
{
    if let Some(point) = manifold.exp(x, step)
        && all_finite(&point)
        && point.len() == x.len()
    {
        let mut carried = Vec::with_capacity(cargo.len());
        let mut geodesic = true;
        for vector in cargo {
            match manifold.parallel_transport_along(x, step, vector) {
                Some(transported)
                    if transported.len() == vector.len() && all_finite(&transported) =>
                {
                    carried.push(transported);
                }
                _ => {
                    geodesic = false;
                    break;
                }
            }
        }
        if geodesic {
            return (point, carried);
        }
    }
    let point = manifold.retract(x, step);
    let carried = cargo
        .iter()
        .map(|vector| manifold.transport(x, &point, vector))
        .collect();
    (point, carried)
}

fn report_at(
    control: &Control,
    value: f64,
    pos: Array1<f64>,
    steps: usize,
    grad: &Array1<f64>,
    gnorm: f64,
    status: Status,
) -> Result<Report> {
    control.deliver(one_shot_report(value, pos, steps, gnorm, status).with_grad(Some(grad.clone())))
}

fn stop_after<M, O>(
    stops: &mut StopRules,
    control: &Control,
    obj: &Counted<'_, O>,
    manifold: &M,
    point: &Array1<f64>,
    value: f64,
    moved: bool,
    grad: &Array1<f64>,
) -> Option<Status>
where
    M: Manifold + ?Sized,
    O: ?Sized,
{
    if control.etol.is_none() && control.max_evals.is_none() {
        return None;
    }
    let gradient_met =
        control.classify(manifold.norm(point, grad), grad.len()) != Status::Iterating;
    stops.after(control, value, moved, obj.count(), true, gradient_met)
}

fn search_failed(
    control: &Control,
    value: f64,
    pos: Array1<f64>,
    steps: usize,
    grad: &Array1<f64>,
    gnorm: f64,
) -> Result<Report> {
    control.deliver(
        one_shot_report(value, pos, steps, gnorm, Status::Budget)
            .with_stop_reason(StopReason::LineSearchFailed)
            .with_grad(Some(grad.clone())),
    )
}

struct Accepted {
    point: Array1<f64>,
    value: f64,
    grad: Array1<f64>,
    s: Array1<f64>,
    y: Array1<f64>,
    transported: Vec<Pair>,
}

fn accept_pair<M>(
    params: &Rlbfgs,
    manifold: &M,
    point: &Array1<f64>,
    s: &Array1<f64>,
    y: &Array1<f64>,
    gnorm: f64,
) -> bool
where
    M: Manifold + ?Sized,
{
    let ss = manifold.inner(point, s, s);
    let ys = manifold.inner(point, s, y);
    let yy = manifold.inner(point, y, y);
    ss > 0.0
        && ys > 0.0
        && ys.is_finite()
        && yy.is_finite()
        && yy > 0.0
        && ys / ss >= params.cautious_slope * gnorm
}

fn push_pair(pairs: &mut Vec<Pair>, memory: usize, pair: Pair) {
    pairs.push(pair);
    if pairs.len() > memory {
        let extra = pairs.len() - memory;
        pairs.drain(0..extra);
    }
}

/// Cautious limited-memory Riemannian BFGS (Huang, Absil, Gallivan 2018, Algorithm 2).
///
/// `manifold` supplies the projection, the Riemannian gradient, the
/// retraction or exponential, and the vector transport. `params` NULL is
/// not a Rust concern: pass [`Rlbfgs::default`]. A positive finite
/// [`Control::maxmove`] caps the ambient Euclidean length of the trial
/// step. The default leaves that cap off. [`Control::gtol`] and the
/// paper's relative gradient stop both end a successful run.
pub fn minimize_rlbfgs<M, O>(
    manifold: &M,
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    params: &Rlbfgs,
) -> Result<Report>
where
    M: Manifold + ?Sized,
    O: DifferentiableObjective<f64> + ?Sized,
{
    minimize_rlbfgs_observed(manifold, obj, init, control, params, None)
}

/// [`minimize_rlbfgs`] with an optional per-iteration observer.
pub fn minimize_rlbfgs_observed<M, O>(
    manifold: &M,
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    params: &Rlbfgs,
    observer: Option<&mut dyn IterationObserver>,
) -> Result<Report>
where
    M: Manifold + ?Sized,
    O: DifferentiableObjective<f64> + ?Sized,
{
    params.check()?;
    let init = init.into();
    if eindir_core::Objective::dim(obj) != init.len() {
        return Err(Error::Dim {
            got: init.len(),
            dim: eindir_core::Objective::dim(obj),
        });
    }
    if manifold.required_dim(init.len()).is_err() {
        return Err(Error::ManifoldDim {
            kind: "rlbfgs",
            got: init.len(),
        });
    }
    counted(obj, control, |obj| {
        drive(manifold, obj, init, control, params, observer)
    })
}

fn drive<M, O>(
    manifold: &M,
    obj: &crate::stop::Counted<'_, O>,
    mut pos: Array1<f64>,
    control: &Control,
    params: &Rlbfgs,
    mut observer: Option<&mut dyn IterationObserver>,
) -> Result<Report>
where
    M: Manifold + ?Sized,
    O: DifferentiableObjective<f64> + ?Sized,
{
    let (mut value, egrad) = obj.value_and_gradient(pos.view());
    if !value.is_finite() || egrad.len() != pos.len() || !all_finite(&egrad) {
        return Err(Error::Oracle {
            what: "non-finite value or gradient",
        });
    }
    let mut grad = manifold.egrad2rgrad(&pos, &egrad);
    if grad.len() != pos.len() || !all_finite(&grad) {
        return Err(Error::Oracle {
            what: "non-finite Riemannian gradient",
        });
    }
    let g0 = manifold.norm(&pos, &grad);
    if !g0.is_finite() {
        return Err(Error::Oracle {
            what: "non-finite Riemannian gradient",
        });
    }
    let mut pairs: Vec<Pair> = Vec::new();
    let mut gamma = INITIAL_SCALE;
    let mut stops = StopRules::from_start(value);

    for step in 0..control.maxiter {
        let gnorm = manifold.norm(&pos, &grad);
        if let Some(report) = stop_now(control, value, &pos, step, &grad, gnorm, g0, params)? {
            return Ok(report);
        }
        let accepted = match armijo(
            manifold, obj, control, params, &pos, value, &grad, &pairs, gamma,
        ) {
            Armijo::Stop(status) => {
                return report_at(control, value, pos, step, &grad, gnorm, status);
            }
            Armijo::Failed => {
                if let Some(status) = stop_after(
                    &mut stops, control, obj, manifold, &pos, value, false, &grad,
                ) {
                    return report_at(control, value, pos, step + 1, &grad, gnorm, status);
                }
                return search_failed(control, value, pos, step, &grad, gnorm);
            }
            Armijo::Accepted(accepted) => accepted,
        };
        let step_length = if observer.is_some() {
            crate::observer::separation(accepted.point.view(), pos.view())
        } else {
            0.0
        };
        let left = gnorm;
        pos = accepted.point;
        value = accepted.value;
        grad = accepted.grad;
        if observer.is_some() {
            let gnow = manifold.norm(&pos, &grad);
            if crate::observer::request_stop(
                &mut observer,
                step,
                pos.view(),
                value,
                gnow,
                step_length,
            ) {
                return control.deliver(
                    one_shot_report(value, pos, step + 1, gnow, Status::ObserverStop)
                        .with_stop_reason(StopReason::ObserverRequested)
                        .with_grad(Some(grad)),
                );
            }
        }
        pairs = accepted.transported;
        if accept_pair(params, manifold, &pos, &accepted.s, &accepted.y, left) {
            let ys = manifold.inner(&pos, &accepted.s, &accepted.y);
            let yy = manifold.inner(&pos, &accepted.y, &accepted.y);
            push_pair(
                &mut pairs,
                params.memory,
                Pair {
                    s: accepted.s,
                    y: accepted.y,
                    rho: 1.0 / ys,
                },
            );
            gamma = ys / yy;
        }
        if let Some(status) =
            stop_after(&mut stops, control, obj, manifold, &pos, value, true, &grad)
        {
            let gnorm = manifold.norm(&pos, &grad);
            return report_at(control, value, pos, step + 1, &grad, gnorm, status);
        }
    }
    let gnorm = manifold.norm(&pos, &grad);
    let status = if ratio_met(gnorm, g0, params) {
        Status::Tolerance
    } else {
        control.finish_status(gnorm, grad.len(), true)
    };
    report_at(control, value, pos, control.maxiter, &grad, gnorm, status)
}

fn stop_now(
    control: &Control,
    value: f64,
    pos: &Array1<f64>,
    steps: usize,
    grad: &Array1<f64>,
    gnorm: f64,
    g0: f64,
    params: &Rlbfgs,
) -> Result<Option<Report>> {
    if !gnorm.is_finite() {
        return Err(Error::Oracle {
            what: "non-finite Riemannian gradient",
        });
    }
    if gnorm <= 0.0 || ratio_met(gnorm, g0, params) {
        return report_at(
            control,
            value,
            pos.clone(),
            steps,
            grad,
            gnorm,
            Status::Tolerance,
        )
        .map(Some);
    }
    if let Some(status) = control.halt(gnorm, grad.len()) {
        return report_at(control, value, pos.clone(), steps, grad, gnorm, status).map(Some);
    }
    Ok(None)
}

enum Armijo {
    Accepted(Box<Accepted>),
    Stop(Status),
    Failed,
}

fn armijo<M, O>(
    manifold: &M,
    obj: &crate::stop::Counted<'_, O>,
    control: &Control,
    params: &Rlbfgs,
    pos: &Array1<f64>,
    value: f64,
    grad: &Array1<f64>,
    pairs: &[Pair],
    gamma: f64,
) -> Armijo
where
    M: Manifold + ?Sized,
    O: DifferentiableObjective<f64> + ?Sized,
{
    let hessian_grad = inverse_hessian_times(pairs, gamma, manifold, pos, grad);
    let mut eta = manifold.project(pos, &negated(hessian_grad));
    let mut slope = manifold.inner(pos, grad, &eta);
    if !slope.is_finite() {
        return Armijo::Failed;
    }
    if slope >= 0.0 {
        eta = manifold.project(pos, &negated(grad.clone()));
        slope = manifold.inner(pos, grad, &eta);
        if slope >= 0.0 || !slope.is_finite() {
            return Armijo::Failed;
        }
    }
    let mut alpha = trial_scale(&eta, control);
    let mut best_decrease = 0.0;
    for _ in 0..params.linesearch_max {
        if !(alpha.is_finite() && alpha > 0.0) {
            break;
        }
        let step = scaled(alpha, &eta);
        match try_trial(manifold, obj, pos, grad, pairs, &step) {
            Trial::Bad => {}
            Trial::Oracle => return Armijo::Failed,
            Trial::Ready(ready) => {
                let decrease = value - ready.value;
                if decrease > best_decrease {
                    best_decrease = decrease;
                }
                if ready.value <= value + params.armijo * alpha * slope {
                    return Armijo::Accepted(ready);
                }
            }
        }
        alpha *= params.backtrack;
    }
    if control
        .noise_floor
        .is_some_and(|floor| floor.stalled_on_energy(best_decrease))
    {
        Armijo::Stop(Status::NoiseFloor)
    } else {
        Armijo::Failed
    }
}

enum Trial {
    Ready(Box<Accepted>),
    Bad,
    Oracle,
}

fn try_trial<M, O>(
    manifold: &M,
    obj: &crate::stop::Counted<'_, O>,
    pos: &Array1<f64>,
    grad: &Array1<f64>,
    pairs: &[Pair],
    step: &Array1<f64>,
) -> Trial
where
    M: Manifold + ?Sized,
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut cargo = Vec::with_capacity(2 + pairs.len() * 2);
    cargo.push(step.clone());
    cargo.push(grad.clone());
    for pair in pairs {
        cargo.push(pair.s.clone());
        cargo.push(pair.y.clone());
    }
    let (point, carried) = advance(manifold, pos, step, &cargo);
    if carried.len() != cargo.len() || point.len() != pos.len() || !all_finite(&point) {
        return Trial::Bad;
    }
    let (value_new, egrad) = obj.value_and_gradient(point.view());
    if !value_new.is_finite() || egrad.len() != point.len() || !all_finite(&egrad) {
        return Trial::Bad;
    }
    let rgrad = manifold.egrad2rgrad(&point, &egrad);
    if rgrad.len() != point.len() || !all_finite(&rgrad) {
        return Trial::Oracle;
    }
    let s = manifold.project(&point, &carried[0]);
    let old_grad = manifold.project(&point, &carried[1]);
    if s.len() != point.len() || old_grad.len() != point.len() {
        return Trial::Oracle;
    }
    let y = manifold.project(&point, &(&rgrad - &old_grad));
    let mut transported = Vec::with_capacity(pairs.len());
    for (index, _) in pairs.iter().enumerate() {
        let s_t = manifold.project(&point, &carried[2 + 2 * index]);
        let y_t = manifold.project(&point, &carried[3 + 2 * index]);
        let ys = manifold.inner(&point, &s_t, &y_t);
        if ys.is_finite() && ys > 0.0 {
            transported.push(Pair {
                s: s_t,
                y: y_t,
                rho: 1.0 / ys,
            });
        }
    }
    Trial::Ready(Box::new(Accepted {
        point,
        value: value_new,
        grad: rgrad,
        s,
        y,
        transported,
    }))
}

fn negated(mut vector: Array1<f64>) -> Array1<f64> {
    scale(-INITIAL_SCALE, &mut vector);
    vector
}

fn scaled(alpha: f64, eta: &Array1<f64>) -> Array1<f64> {
    let mut step = eta.clone();
    scale(alpha, &mut step);
    step
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::Euclidean;
    use ndarray::array;

    struct RetractionFallback;

    impl Manifold for RetractionFallback {
        fn project(&self, _x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
            v.clone()
        }

        fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
            x + &(2.0 * v)
        }

        fn transport(
            &self,
            _x_from: &Array1<f64>,
            _x_to: &Array1<f64>,
            v: &Array1<f64>,
        ) -> Array1<f64> {
            v + 10.0
        }

        fn exp(&self, x: &Array1<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
            Some(x + v)
        }
    }

    #[test]
    fn one_accepted_pair_meets_the_secant_equation() {
        let manifold = Euclidean;
        let x = array![0.0, 0.0];
        let s = array![1.0, -2.0];
        let y = array![2.0, -2.0];
        let ys = 6.0;
        let pairs = [Pair {
            s: s.clone(),
            y: y.clone(),
            rho: 1.0 / ys,
        }];
        let hy = inverse_hessian_times(&pairs, ys / 8.0, &manifold, &x, &y);
        let tol = 1.0 / 1_000_000_000_000.0;
        assert!((hy[0] - s[0]).abs() < tol);
        assert!((hy[1] - s[1]).abs() < tol);
    }

    #[test]
    fn zero_curvature_is_rejected_when_the_cautious_slope_is_zero() {
        let params = Rlbfgs::default().with_cautious_slope(0.0);
        let x = array![0.0, 0.0];
        let s = array![1.0, 0.0];
        let y = array![0.0, 1.0];
        assert!(!accept_pair(&params, &Euclidean, &x, &s, &y, 1.0));
    }

    #[test]
    fn cautious_test_skips_positive_curvature_below_its_threshold() {
        let params = Rlbfgs::default().with_cautious_slope(1.0);
        let x = array![0.0, 0.0];
        let s = array![1.0, 0.0];
        let y = array![1.0, 0.0];
        assert!(!accept_pair(&params, &Euclidean, &x, &s, &y, 2.0));
    }

    #[test]
    fn missing_parallel_transport_falls_back_to_retraction_transport() {
        let x = array![1.0, 2.0];
        let step = array![0.5, -0.25];
        let cargo = [array![3.0, 4.0]];
        let (point, carried) = advance(&RetractionFallback, &x, &step, &cargo);
        assert_eq!(point, array![2.0, 1.5]);
        assert_eq!(carried, vec![array![13.0, 14.0]]);
    }
}
