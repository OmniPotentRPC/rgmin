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
