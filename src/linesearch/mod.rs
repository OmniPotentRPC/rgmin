//! Line search along a direction.

use ndarray::{Array1, ArrayView1};

/// Accept conditions (Armijo, Wolfe, Goldstein).
pub mod conditions;
mod zoom;

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
    /// A trial answered with a non-finite value or gradient fails every
    /// decrease test. A non-finite start or `dir` returns the start
    /// without probing.
    pub fn search<F>(
        &self,
        oracle: F,
        pos: ArrayView1<'_, f64>,
        dir: ArrayView1<'_, f64>,
        istep: f64,
    ) -> (Array1<f64>, f64, f64)
    where
        F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    {
        self.search_path(oracle, &mut Line { pos, dir }, istep)
    }

    /// [`Self::search`] along any [`Path`]. A point the path cannot
    /// produce scores as a non-finite value.
    pub(crate) fn search_path<F, P>(
        &self,
        mut oracle: F,
        path: &mut P,
        istep: f64,
    ) -> (Array1<f64>, f64, f64)
    where
        F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
        P: Path + ?Sized,
    {
        match *self {
            Self::Brent { maxiter, tol } => brent_search(&mut oracle, path, istep, maxiter, tol),
            Self::Backtracking { c, beta, maxiter } => {
                let rule = Backtrack {
                    c,
                    beta,
                    maxiter,
                    goldstein: false,
                };
                backtrack_search(&mut oracle, path, istep, rule)
            }
            Self::Goldstein { c, beta, maxiter } => {
                let rule = Backtrack {
                    c,
                    beta,
                    maxiter,
                    goldstein: true,
                };
                backtrack_search(&mut oracle, path, istep, rule)
            }
            Self::Wolfe { c1, c2, maxiter } => {
                zoom::wolfe_search(&mut oracle, path, istep, c1, c2, maxiter)
            }
        }
    }
}

/// The curve a line search probes: `x(t)` and the slope of `f` along it.
pub(crate) trait Path {
    /// `x(0)`.
    fn start(&self) -> Array1<f64>;
    /// `x'(0)`.
    fn direction(&self) -> ArrayView1<'_, f64>;
    /// `x(t)`, `None` where the curve is not defined.
    fn point(&mut self, t: f64) -> Option<Array1<f64>>;
    /// `g . x'(t)` for the gradient `g` at `at = x(t)`.
    fn slope(&mut self, t: f64, at: ArrayView1<'_, f64>, g: ArrayView1<'_, f64>) -> f64;
}

/// The straight line `pos + t dir`.
pub(crate) struct Line<'a> {
    pub(crate) pos: ArrayView1<'a, f64>,
    pub(crate) dir: ArrayView1<'a, f64>,
}

impl Path for Line<'_> {
    fn start(&self) -> Array1<f64> {
        self.pos.to_owned()
    }
    fn direction(&self) -> ArrayView1<'_, f64> {
        self.dir.view()
    }
    fn point(&mut self, t: f64) -> Option<Array1<f64>> {
        Some(axpy(self.pos, t, self.dir))
    }
    fn slope(&mut self, _t: f64, _at: ArrayView1<'_, f64>, g: ArrayView1<'_, f64>) -> f64 {
        g.iter().zip(self.dir.iter()).map(|(g, d)| g * d).sum()
    }
}

pub(crate) fn axpy(pos: ArrayView1<'_, f64>, t: f64, dir: ArrayView1<'_, f64>) -> Array1<f64> {
    let mut trial = pos.to_owned();
    crate::vecops::axpy(t, dir, &mut trial);
    trial
}

/// `f`, or `+inf` when the value or gradient is non-finite. NaN fails no
/// comparison and `-inf` passes every decrease test, so either would read
/// a refused answer as a decrease.
pub(crate) fn scored(f: f64, g: ArrayView1<'_, f64>) -> f64 {
    if crate::step::finite(f, g) {
        f
    } else {
        f64::INFINITY
    }
}

/// No line leaves a start whose value, gradient or direction is non-finite.
fn unusable(f0: f64, g0: &Array1<f64>, dir: ArrayView1<'_, f64>) -> bool {
    !crate::step::finite(f0, g0.view()) || dir.iter().any(|v| !v.is_finite())
}

/// `(x(t), f, g)` with `f` scored, `None` where the path is undefined.
fn probe<F, P>(oracle: &mut F, path: &mut P, t: f64) -> Option<(Array1<f64>, f64, Array1<f64>)>
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    P: Path + ?Sized,
{
    let x = path.point(t)?;
    let (f, g) = oracle(x.view());
    let f = scored(f, g.view());
    Some((x, f, g))
}

fn brent_search<F, P>(
    oracle: &mut F,
    path: &mut P,
    istep: f64,
    maxiter: usize,
    tol: f64,
) -> (Array1<f64>, f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    P: Path + ?Sized,
{
    let pos = path.start();
    let (f0, g0) = oracle(pos.view());
    if unusable(f0, &g0, path.direction()) {
        return (pos, f0, 0.0);
    }
    let mut phi = |t: f64| probe(oracle, path, t).map_or(f64::INFINITY, |(_, f, _)| f);
    let (a, b) = bracket(&mut phi, 0.0, istep.max(1e-12), maxiter);
    let (t, ft) = brent(&mut phi, a, b, tol, maxiter.max(20));
    match path.point(t) {
        Some(x) if ft < f0 => (x, ft, t.abs()),
        _ => (pos, f0, 0.0),
    }
}

/// Backtracking constants; `goldstein` adds the Goldstein lower bound.
#[derive(Clone, Copy)]
struct Backtrack {
    c: f64,
    beta: f64,
    maxiter: usize,
    goldstein: bool,
}

fn backtrack_search<F, P>(
    oracle: &mut F,
    path: &mut P,
    istep: f64,
    rule: Backtrack,
) -> (Array1<f64>, f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    P: Path + ?Sized,
{
    let Backtrack {
        c,
        beta,
        maxiter,
        goldstein,
    } = rule;
    let pos = path.start();
    let (f0, g0) = oracle(pos.view());
    if unusable(f0, &g0, path.direction()) {
        return (pos, f0, 0.0);
    }
    let slope: f64 = path.slope(0.0, pos.view(), g0.view());
    let mut alpha = istep.max(1e-16);
    let mut best_x = pos.clone();
    let mut best_f = f0;
    let mut best_alpha = 0.0;
    for _ in 0..maxiter {
        let (x, f) = match probe(oracle, path, alpha) {
            Some((x, f, _)) => (Some(x), f),
            None => (None, f64::INFINITY),
        };
        if let (true, Some(x)) = (f < best_f, &x) {
            best_f = f;
            best_x = x.clone();
            best_alpha = alpha;
        }
        let accept = if goldstein {
            conditions::goldstein(f, f0, alpha, slope, c)
        } else {
            conditions::armijo(f, f0, alpha, slope, c)
        };
        if let (true, Some(x)) = (accept, x) {
            return (x, f, alpha.abs());
        }
        if goldstein && conditions::armijo(f, f0, alpha, slope, c) {
            // Lower bound failed: step is too short (Nocedal-Wright 3.11).
            alpha /= beta;
        } else {
            alpha *= beta;
        }
        let alpha_max = 64.0_f64.max(istep.abs() * 64.0);
        if !alpha.is_finite() || alpha < 1e-16 || alpha > alpha_max {
            break;
        }
    }
    if best_f < f0 {
        (best_x, best_f, best_alpha.abs())
    } else {
        (pos, f0, 0.0)
    }
}

fn bracket(phi: &mut impl FnMut(f64) -> f64, mut a: f64, mut b: f64, maxiter: usize) -> (f64, f64) {
    let gold = 1.618_034;
    let mut fa = phi(a);
    let mut fb = phi(b);
    if fa < fb {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = b + gold * (b - a);
    let mut fc = phi(c);
    let mut it = 0;
    while fb >= fc && it < maxiter {
        let u = b + gold * (c - b);
        let fu = phi(u);
        a = b;
        b = c;
        fb = fc;
        c = u;
        fc = fu;
        it += 1;
        let _ = fa;
        fa = fb;
    }
    if a < c { (a, c) } else { (c, a) }
}

fn brent(
    phi: &mut impl FnMut(f64) -> f64,
    ax: f64,
    cx: f64,
    tol: f64,
    maxiter: usize,
) -> (f64, f64) {
    let cgold = 0.381_966;
    let mut a = ax.min(cx);
    let mut b = ax.max(cx);
    let mut x = 0.5 * (a + b);
    let mut w = x;
    let mut v = x;
    let mut fx = phi(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut e: f64 = 0.0;
    let mut d: f64 = 0.0;
    for _ in 0..maxiter {
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-12;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            return (x, fx);
        }
        let mut u;
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            if p.abs() >= (0.5 * q * etemp).abs() || p <= q * (a - x) || p >= q * (b - x) {
                e = if x >= xm { a - x } else { b - x };
                d = cgold * e;
            } else {
                d = p / q;
                u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = if xm - x >= 0.0 { tol1 } else { -tol1 };
                }
            }
        } else {
            e = if x >= xm { a - x } else { b - x };
            d = cgold * e;
        }
        u = if d.abs() >= tol1 {
            x + d
        } else {
            x + if d >= 0.0 { tol1 } else { -tol1 }
        };
        let fu = phi(u);
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    (x, fx)
}
