//! A line search that never calls an objective.
//!
//! The caller already has the energy and the gradient. Each
//! [`SuppliedSearch::offer`] consumes that pair and either asks for the
//! next trial or returns the accepted step. The trials are the ones
//! [`super::LineSearch::search`] would make on the same answers.

use ndarray::{Array1, ArrayView1};

use super::{LineSearch, LineSearchOptions};
use crate::step::finite;

/// What the caller does next.
#[derive(Clone, Debug, PartialEq)]
pub enum SearchStep {
    /// Evaluate the objective at `x` (`x0 + alpha * direction`) and pass
    /// that energy and gradient to [`SuppliedSearch::offer`].
    Evaluate {
        /// Trial step length along the direction.
        alpha: f64,
        /// Trial point.
        x: Array1<f64>,
    },
    /// The search is finished. `x` is the accepted point, `value` its
    /// energy, and `alpha` the absolute step length
    /// [`LineSearch::search`] reports.
    Accepted {
        /// Accepted point.
        x: Array1<f64>,
        /// Energy at `x`.
        value: f64,
        /// Absolute step length.
        alpha: f64,
    },
}

/// Line search driven by a caller-supplied energy and gradient.
///
/// Nothing in this type calls an objective. The first [`Self::offer`] is
/// the energy and gradient at the start `x`. Later offers are the energy
/// and gradient at the trial from the previous [`SearchStep::Evaluate`].
///
/// ```
/// use ndarray::{Array1, array};
/// use rgmin::{LineSearch, SearchStep, SuppliedSearch};
///
/// fn quad(x: f64) -> (f64, f64) {
///     let z = x - 1.0;
///     (z * z, 2.0 * z)
/// }
/// let start = array![0.0];
/// let dir = array![1.0];
/// let mut search = SuppliedSearch::new(
///     LineSearch::Backtracking { c: 1e-4, beta: 0.5, maxiter: 8 },
///     start.view(),
///     dir.view(),
///     1.0,
/// );
/// let (f0, g0) = quad(0.0);
/// let mut step = search.offer(f0, array![g0].view());
/// let accepted = loop {
///     match step {
///         SearchStep::Evaluate { x, .. } => {
///             let (f, g) = quad(x[0]);
///             step = search.offer(f, array![g].view());
///         }
///         SearchStep::Accepted { x, .. } => break x,
///     }
/// };
/// assert!((accepted[0] - 1.0).abs() < 1e-12);
/// ```
#[derive(Clone, Debug)]
pub struct SuppliedSearch {
    x0: Array1<f64>,
    dir: Array1<f64>,
    istep: f64,
    kind: LineSearch,
    /// Every answer so far, the start first. The search is deterministic
    /// in its answers, so replaying them reaches the next trial.
    answers: Vec<(f64, Array1<f64>)>,
    done: Option<(Array1<f64>, f64, f64)>,
}

impl SuppliedSearch {
    /// `x` is the start and `dir` the search direction.
    ///
    /// A direction whose length is not `x`'s, or a non-finite direction,
    /// ends on the first offer at the start with step length zero.
    pub fn new(
        kind: LineSearch,
        x: ArrayView1<'_, f64>,
        dir: ArrayView1<'_, f64>,
        istep: f64,
    ) -> Self {
        Self {
            x0: x.to_owned(),
            dir: dir.to_owned(),
            istep,
            kind,
            answers: Vec::new(),
            done: None,
        }
    }

    /// Consume the energy `f` and gradient `g` at the point the search is
    /// waiting on. The first call is the start. Later calls are the trial
    /// from [`SearchStep::Evaluate`].
    pub fn offer(&mut self, f: f64, g: ArrayView1<'_, f64>) -> SearchStep {
        if let Some((x, value, alpha)) = &self.done {
            return SearchStep::Accepted {
                x: x.clone(),
                value: *value,
                alpha: *alpha,
            };
        }
        let n = self.x0.len();
        if self.answers.is_empty() {
            let bad = self.dir.len() != n
                || g.len() != n
                || !finite(f, g)
                || self.dir.iter().any(|v| !v.is_finite());
            if bad {
                return self.finish(self.x0.clone(), f, 0.0);
            }
        }
        let g = if g.len() == n {
            g.to_owned()
        } else {
            Array1::from_elem(n, f64::NAN)
        };
        self.answers.push((f, g));
        self.replay()
    }

    fn replay(&mut self) -> SearchStep {
        let f0 = self.answers[0].0;
        let g0 = &self.answers[0].1;
        let known = &self.answers[1..];
        let mut used = 0;
        let mut ask: Option<(f64, Array1<f64>)> = None;
        let out = self.kind.search_traced(
            |alpha, x| {
                if let Some((f, g)) = known.get(used) {
                    used += 1;
                    return (*f, g.clone());
                }
                if ask.is_none() {
                    ask = Some((alpha, x.to_owned()));
                }
                (f64::NAN, Array1::from_elem(x.len(), f64::NAN))
            },
            self.x0.view(),
            f0,
            g0.view(),
            self.dir.view(),
            self.istep,
            f64::INFINITY,
            LineSearchOptions::default(),
        );
        if let Some((alpha, x)) = ask {
            return SearchStep::Evaluate { alpha, x };
        }
        match out {
            Some(o) => self.finish(o.x, o.f, o.alpha),
            None => self.finish(self.x0.clone(), f0, 0.0),
        }
    }

    fn finish(&mut self, x: Array1<f64>, value: f64, alpha: f64) -> SearchStep {
        self.done = Some((x.clone(), value, alpha));
        SearchStep::Accepted { x, value, alpha }
    }
}
