//! Line search along a direction.

use ndarray::{Array1, ArrayView1};

/// Accept conditions (Armijo, Wolfe, Goldstein).
pub mod conditions;
mod interp;
mod known;
mod zoom;

pub use known::LineOutcome;
pub use zoom::zoom;

/// How to pick α such that `x + α d` decreases `f`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineSearch {
    /// Brent (1973) on a golden-section bracket. Derivative-free in α.
    ///
    /// Brent, *Algorithms for Minimization without Derivatives* (1973).
    Brent {
        /// Bracket / refine iterations.
        maxiter: usize,
        /// Absolute tolerance on α.
        tol: f64,
    },
    /// Armijo backtracking with geometric reduction.
    ///
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Backtracking {
        /// Armijo `c` (default 1e-4).
        c: f64,
        /// Step shrink `β` (default 0.5).
        beta: f64,
        /// Maximum shrinks.
        maxiter: usize,
    },
    /// Goldstein condition (Nocedal-Wright 3.11) with shrink/expand.
    ///
    /// Accepts when `φ(0) + (1-c) α φ'(0) <= φ(α) <= φ(0) + c α φ'(0)`.
    /// Armijo failure shrinks α; a failed lower bound expands α.
    /// `c` belongs in `(0, 0.5)`.
    ///
    /// Goldstein, *Multiplier and gradient methods*,
    /// <https://doi.org/10.1007/BF00927673>.
    Goldstein {
        /// Goldstein / Armijo `c`.
        c: f64,
        /// Step shrink/expand `β` (default 0.5).
        beta: f64,
        /// Maximum shrink or expand trials.
        maxiter: usize,
    },
    /// Strong Wolfe with Nocedal-Wright zoom (algorithms 3.5 and 3.6).
    ///
    /// Wolfe, *Convergence Conditions for Ascent Methods*,
    /// <https://doi.org/10.1137/1011036>.
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Wolfe {
        /// Armijo `c1` (default 1e-4).
        c1: f64,
        /// Strong-curvature `c2` (default 0.9).
        c2: f64,
        /// Expand + zoom iterations.
        maxiter: usize,
    },
}

/// Strong Wolfe for nonlinear CG.
///
/// Nocedal-Wright section 5.2 uses `c2 = 0.1`. `c2 = 0.9` accepts a step
/// whose slope is still nearly the opening slope, so the new gradient
/// stays aligned with the old one. The NJWS angle restart then fires
/// every step and the iteration is steepest descent. A fresh session and
/// a one-shot minimizer that was not given a search both use this value.
/// A caller that passes its own [`LineSearch`] is left alone.
pub const CG_LINESEARCH: LineSearch = LineSearch::Wolfe {
    c1: 1e-4,
    c2: 0.1,
    maxiter: 20,
};

/// How far a Wolfe search extrapolates while the slope is still negative.
///
/// The default is the interior cubic kept inside `[1.1, 4]` times the last
/// increment. [`WolfeExtrapolation::Case3Average`] is opt-in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WolfeExtrapolation {
    /// Interior cubic minimiser, at least 1.1 and at most 4 times the last increment.
    #[default]
    Interior,
    /// Average of that cubic and the More-Thuente case-3 step when the
    /// case-3 step is shorter than three quarters of the interior cubic.
    /// The growth floor is half an increment, and the cap is still four.
    Case3Average,
}

/// Optional policy for objective values whose accuracy is lower than their gradients.
/// Existing line-search entry points use the default four-epsilon window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LineSearchOptions {
    relative_objective_error: Option<f64>,
    extrapolation: WolfeExtrapolation,
}

impl LineSearchOptions {
    /// Permit approximate Wolfe acceptance up to `relative * |f0|` above `f0`.
    ///
    /// This is an acceptance policy, not a certified oracle error bound.
    /// Both slope conditions still apply; gradient stopping is independent.
    /// Negative and nonfinite values are rejected. Zero permits no rise.
    pub fn with_objective_roundoff(relative: f64) -> Option<Self> {
        (relative.is_finite() && relative >= 0.0).then_some(Self {
            relative_objective_error: Some(relative),
            ..Self::default()
        })
    }

    /// Which extrapolation the next Wolfe search uses.
    /// [`WolfeExtrapolation::Interior`] is the default.
    #[must_use]
    pub fn extrapolation(self) -> WolfeExtrapolation {
        self.extrapolation
    }

    /// Set the Wolfe extrapolation. Roundoff acceptance is unchanged.
    #[must_use]
    pub fn with_extrapolation(mut self, extrapolation: WolfeExtrapolation) -> Self {
        self.extrapolation = extrapolation;
        self
    }

    pub(crate) fn accepts_approximate_wolfe(
        self,
        phi: f64,
        phi0: f64,
        dphi: f64,
        dphi0: f64,
        c1: f64,
        c2: f64,
    ) -> bool {
        match self.relative_objective_error {
            Some(relative) => {
                conditions::approximate_strong_wolfe(phi, phi0, dphi, dphi0, c1, c2, relative)
            }
            None => conditions::roundoff_strong_wolfe(phi, phi0, dphi, dphi0, c1, c2),
        }
    }
}

impl Default for LineSearch {
    fn default() -> Self {
        Self::Brent {
            maxiter: 40,
            tol: 1e-10,
        }
    }
}

impl LineSearch {
    /// Returns `(x_new, f_new, |α|)` if the trial beat `f0`, else the start.
    ///
    /// Evaluates the start once, then runs [`Self::search_from`] with no
    /// step cap. A caller that already holds `f` and `g` at `pos` should
    /// call [`Self::search_from`] directly and save that evaluation.
    pub fn search<F>(
        &self,
        mut oracle: F,
        pos: ArrayView1<'_, f64>,
        dir: ArrayView1<'_, f64>,
        istep: f64,
    ) -> (Array1<f64>, f64, f64)
    where
        F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    {
        let (f0, g0) = oracle(pos);
        match self.search_from(&mut oracle, pos, f0, g0.view(), dir, istep, f64::INFINITY) {
            Some(out) => (out.x, out.f, out.alpha),
            None => (pos.to_owned(), f0, 0.0),
        }
    }
}

pub(crate) fn axpy(pos: ArrayView1<'_, f64>, t: f64, dir: ArrayView1<'_, f64>) -> Array1<f64> {
    let mut trial = pos.to_owned();
    crate::vecops::axpy(t, dir, &mut trial);
    trial
}
