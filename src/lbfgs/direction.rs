//! L-BFGS direction from stored pairs and a caller-supplied `H0`.
//!
//! The two-loop is Nocedal and Wright, algorithm 7.4. `H0` is applied
//! to the intermediate vector, and the value returned is the direction
//! `-H g`. Nothing in this module evaluates an objective.
//!
//! [`LbfgsPairs`] stores pairs through the curvature gate of
//! [`Lbfgs::record`] and runs the same recursion as the session step,
//! so a scalar `H0 = γ` equal to the factor [`Lbfgs::two_loop`]
//! multiplies by matches that direction bit for bit, and so does the
//! eOn [`super::LbfgsStep::Capped`] step taken from the same history.
//! [`LbfgsPairs::push_rho`] takes the caller's `rho` for a pair that
//! passes that gate.
//!
//! The C ABI is a thin wrapper over [`LbfgsPairs`]:
//! `rgmin_lbfgs_pairs_create`, `rgmin_lbfgs_pairs_push`,
//! `rgmin_lbfgs_pairs_push_rho` (ABI minor 43) and
//! `rgmin_lbfgs_direction` (ABI minor 38). C++ wraps them in
//! `rgmin::LbfgsPairs`, and Python exposes `rgmin.LbfgsPairs`.
//!
//! Nocedal and Wright, *Numerical Optimization*,
//! <https://doi.org/10.1007/978-0-387-40065-5>.

#![deny(missing_docs)]

use std::fmt;

use ndarray::{Array1, ArrayView1};

use super::{Lbfgs, Pair, stored_pair, two_loop_in_place};
use crate::error::{Error, Result};

/// Initial inverse Hessian for one two-loop direction.
///
/// A diagonal is the vector of diagonal entries. A callback is
/// `v ↦ H0 v`. The callback is invoked once, on the intermediate
/// vector of the backward loop, and is not an objective.
pub enum LbfgsH0<'a> {
    /// `H0 = γ I`.
    Scale(f64),
    /// Diagonal entries of `H0`, one per coordinate.
    Diagonal(ArrayView1<'a, f64>),
    /// `v ↦ H0 v`. A length other than the gradient's is an error.
    Apply(&'a dyn Fn(ArrayView1<'_, f64>) -> Result<Array1<f64>>),
}

impl fmt::Debug for LbfgsPairs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LbfgsPairs")
            .field("n", &self.n)
            .field("pairs", &self.memory.len())
            .finish()
    }
}

impl fmt::Debug for LbfgsH0<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scale(gamma) => f.debug_tuple("Scale").field(gamma).finish(),
            Self::Diagonal(diag) => f.debug_tuple("Diagonal").field(&diag.len()).finish(),
            Self::Apply(_) => f.write_str("Apply(..)"),
        }
    }
}

/// Curvature pairs for [`lbfgs_direction`].
///
/// The handle accumulates `(s, y)` and never evaluates an objective.
/// Pairs are oldest first. A pair whose curvature is not positive is
/// declined and leaves the memory unchanged; [`Self::push`] gives the
/// test. `capacity` 0 keeps ten pairs; a full memory drops the oldest
/// pair.
pub struct LbfgsPairs {
    n: usize,
    max_pairs: usize,
    memory: Vec<Pair>,
}

impl LbfgsPairs {
    /// Pair memory of dimension `n`.
    ///
    /// `capacity` 0 selects 10. A zero dimension is an error.
    pub fn new(n: usize, capacity: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::LbfgsDirection {
                what: "dimension is zero",
            });
        }
        Ok(Self {
            n,
            max_pairs: if capacity == 0 { 10 } else { capacity },
            memory: Vec::new(),
        })
    }

    /// Dimension of every stored vector.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Number of stored pairs.
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// True when no pair is stored.
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Drop every stored pair. The dimension stays.
    pub fn clear(&mut self) {
        self.memory.clear();
    }

    /// Store `(s, y)` when the curvature gate accepts it.
    ///
    /// The gate is the relative-curvature test of [`Lbfgs::record`]: a
    /// non-finite `s · y`, or `s · y <= 1e-8 ||s|| ||y||`, declines the pair. The
    /// stored `rho` is `1 / (s · y)`. `Ok(false)` is a declined pair, not
    /// an error. A length other than [`Self::dim`] is an error.
    pub fn push(&mut self, s: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> Result<bool> {
        self.store(s, y, None)
    }

    /// Store `(s, y)` with the caller's `rho` in place of `1 / (s · y)`.
    ///
    /// The pair passes the gate of [`Self::push`] first, so a pair with
    /// non-positive curvature is declined whatever `rho` is. A `rho`
    /// that is not finite and positive is declined too. Either refusal
    /// is `Ok(false)` and leaves the memory unchanged. A length other
    /// than [`Self::dim`] is an error.
    ///
    /// Each pair enters the two-loop as `(s, y, rho)`, so a `rho` other
    /// than `1 / (s · y)` is the update
    /// `H ← (I - rho s yᵀ) H (I - rho y sᵀ) + rho s sᵀ` with that `rho`.
    ///
    /// ```
    /// use ndarray::array;
    /// use rgmin::{LbfgsH0, LbfgsPairs};
    ///
    /// let (s, y) = (array![1.0, 0.0], array![2.0, 1.0]);
    /// let g = array![1.0, -2.0];
    /// let mut by_gate = LbfgsPairs::new(2, 4).unwrap();
    /// let mut by_rho = LbfgsPairs::new(2, 4).unwrap();
    /// assert!(by_gate.push(s.view(), y.view()).unwrap());
    /// assert!(by_rho.push_rho(s.view(), y.view(), 1.0 / s.dot(&y)).unwrap());
    /// assert_eq!(
    ///     by_gate.direction(g.view(), LbfgsH0::Scale(0.5)).unwrap(),
    ///     by_rho.direction(g.view(), LbfgsH0::Scale(0.5)).unwrap(),
    /// );
    /// assert!(!by_rho.push_rho(s.view(), y.view(), -1.0).unwrap());
    /// assert_eq!(by_rho.len(), 1);
    /// ```
    pub fn push_rho(
        &mut self,
        s: ArrayView1<'_, f64>,
        y: ArrayView1<'_, f64>,
        rho: f64,
    ) -> Result<bool> {
        self.store(s, y, Some(rho))
    }

    fn store(
        &mut self,
        s: ArrayView1<'_, f64>,
        y: ArrayView1<'_, f64>,
        rho: Option<f64>,
    ) -> Result<bool> {
        if s.len() != self.n || y.len() != self.n {
            return Err(Error::LbfgsDirection {
                what: "pair length does not match the dimension",
            });
        }
        if rho.is_some_and(|rho| !(rho.is_finite() && rho > 0.0)) {
            return Ok(false);
        }
        let Some(mut pair) = stored_pair(s.to_owned(), y.to_owned()) else {
            return Ok(false);
        };
        if let Some(rho) = rho {
            pair.rho = rho;
        }
        self.memory.push(pair);
        while self.memory.len() > self.max_pairs {
            self.memory.remove(0);
        }
        Ok(true)
    }

    /// Direction `-H g`. An empty memory returns `-H0 g`.
    ///
    /// Does not evaluate an objective.
    ///
    /// ```
    /// use ndarray::array;
    /// use rgmin::{LbfgsH0, LbfgsPairs};
    ///
    /// let mut pairs = LbfgsPairs::new(2, 4).unwrap();
    /// let g = array![1.0, -2.0];
    /// let empty = pairs.direction(g.view(), LbfgsH0::Scale(0.5)).unwrap();
    /// assert_eq!(empty, array![-0.5, 1.0]);
    /// assert!(pairs.push(array![1.0, 0.0].view(), array![2.0, 1.0].view()).unwrap());
    /// let diag = array![0.5, 0.25];
    /// let from_diag = pairs.direction(g.view(), LbfgsH0::Diagonal(diag.view())).unwrap();
    /// let from_callback = pairs
    ///     .direction(g.view(), LbfgsH0::Apply(&|v| Ok(&diag * &v)))
    ///     .unwrap();
    /// assert_eq!(from_diag, from_callback);
    /// ```
    pub fn direction(&self, gradient: ArrayView1<'_, f64>, h0: LbfgsH0<'_>) -> Result<Array1<f64>> {
        if gradient.len() != self.n {
            return Err(Error::LbfgsDirection {
                what: "gradient length does not match the dimension",
            });
        }
        direction_of(&self.memory, 0, gradient, &h0)
    }
}

impl Lbfgs {
    /// [`Self::two_loop`] with a caller-supplied initial inverse Hessian.
    ///
    /// Does not evaluate an objective, and does not use an installed
    /// preconditioner. Al-Baali replays are the ones [`Self::two_loop`]
    /// uses. [`LbfgsH0::Scale`] equal to the factor that call multiplies
    /// by — `(s · y) / (y · y)` of the newest pair, or the fixed scale
    /// when one is set — matches it bit for bit.
    pub fn two_loop_h0(
        &self,
        gradient: ArrayView1<'_, f64>,
        h0: LbfgsH0<'_>,
    ) -> Result<Array1<f64>> {
        direction_of(&self.memory, self.extra_updates, gradient, &h0)
    }
}

/// Direction `-H g` from explicit pairs, oldest first.
///
/// An empty pair list returns `-H0 g`. A pair the curvature gate
/// would decline is an error. The call does not evaluate an objective.
///
/// ```
/// use ndarray::array;
/// use rgmin::{LbfgsH0, lbfgs_direction};
///
/// let g = array![1.0, -2.0];
/// let d = lbfgs_direction(g.view(), &[], &[], LbfgsH0::Scale(0.5)).unwrap();
/// assert_eq!(d, array![-0.5, 1.0]);
/// ```
pub fn lbfgs_direction(
    gradient: ArrayView1<'_, f64>,
    s: &[ArrayView1<'_, f64>],
    y: &[ArrayView1<'_, f64>],
    h0: LbfgsH0<'_>,
) -> Result<Array1<f64>> {
    if gradient.is_empty() {
        return Err(Error::LbfgsDirection {
            what: "gradient is empty",
        });
    }
    if s.len() != y.len() {
        return Err(Error::LbfgsDirection {
            what: "s and y pair counts differ",
        });
    }
    let mut pairs = LbfgsPairs {
        n: gradient.len(),
        max_pairs: s.len().max(1),
        memory: Vec::new(),
    };
    for (sk, yk) in s.iter().zip(y.iter()) {
        if !pairs.push(*sk, *yk)? {
            return Err(Error::LbfgsDirection {
                what: "curvature pair is not positive",
            });
        }
    }
    pairs.direction(gradient, h0)
}

fn direction_of(
    memory: &[Pair],
    extra: usize,
    gradient: ArrayView1<f64>,
    h0: &LbfgsH0<'_>,
) -> Result<Array1<f64>> {
    if gradient.is_empty() {
        return Err(Error::LbfgsDirection {
            what: "gradient is empty",
        });
    }
    if memory
        .iter()
        .any(|p| p.s.len() != gradient.len() || p.y.len() != gradient.len())
    {
        return Err(Error::LbfgsDirection {
            what: "gradient length does not match the stored pairs",
        });
    }
    if let LbfgsH0::Diagonal(diag) = h0
        && diag.len() != gradient.len()
    {
        return Err(Error::LbfgsDirection {
            what: "H0 diagonal length does not match the gradient",
        });
    }
    let mut q = gradient.to_owned();
    two_loop_in_place(memory, extra, &mut q, |q| apply_h0(q, h0))?;
    Ok(q)
}

fn apply_h0(q: &mut Array1<f64>, h0: &LbfgsH0<'_>) -> Result<()> {
    match h0 {
        LbfgsH0::Scale(gamma) => {
            if !gamma.is_finite() {
                return Err(Error::LbfgsDirection {
                    what: "H0 scale is not finite",
                });
            }
            *q *= *gamma;
        }
        LbfgsH0::Diagonal(diag) => {
            if diag.len() != q.len() {
                return Err(Error::LbfgsDirection {
                    what: "H0 diagonal length does not match the gradient",
                });
            }
            if diag.iter().any(|v| !v.is_finite()) {
                return Err(Error::LbfgsDirection {
                    what: "H0 diagonal is not finite",
                });
            }
            for (qi, di) in q.iter_mut().zip(diag.iter()) {
                *qi *= *di;
            }
        }
        LbfgsH0::Apply(apply) => {
            let out = apply(q.view())?;
            if out.len() != q.len() {
                return Err(Error::LbfgsDirection {
                    what: "H0 callback length does not match the gradient",
                });
            }
            *q = out;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};

    use super::*;

    struct Sample {
        s1: Array1<f64>,
        y1: Array1<f64>,
        s2: Array1<f64>,
        y2: Array1<f64>,
        g: Array1<f64>,
    }

    fn sample() -> Sample {
        Sample {
            s1: array![1.0, 0.5, -0.25],
            y1: array![2.0, 1.0, 0.5],
            s2: array![0.5, -0.25, 1.0],
            y2: array![1.0, 0.5, 2.0],
            g: array![1.0, -2.0, 0.5],
        }
    }

    fn gamma_of(s: &Array1<f64>, y: &Array1<f64>) -> f64 {
        s.dot(y) / y.dot(y)
    }

    #[test]
    fn scalar_h0_matches_the_internal_two_loop() {
        let Sample { s1, y1, s2, y2, g } = sample();
        let mut opt = Lbfgs::with_capacity(4);
        assert!(opt.push(s1.clone(), y1.clone()));
        assert!(opt.push(s2.clone(), y2.clone()));
        let gamma = gamma_of(&s2, &y2);
        let internal = opt.two_loop(g.view());
        let from_solver = opt.two_loop_h0(g.view(), LbfgsH0::Scale(gamma)).unwrap();
        assert_eq!(from_solver, internal);

        let mut pairs = LbfgsPairs::new(3, 4).unwrap();
        assert!(pairs.push(s1.view(), y1.view()).unwrap());
        assert!(pairs.push(s2.view(), y2.view()).unwrap());
        let from_handle = pairs.direction(g.view(), LbfgsH0::Scale(gamma)).unwrap();
        assert_eq!(from_handle, internal);

        let diag = Array1::from_elem(3, gamma);
        let from_diag = pairs
            .direction(g.view(), LbfgsH0::Diagonal(diag.view()))
            .unwrap();
        assert_eq!(from_diag, internal);
    }

    /// `Ax` with a symmetric positive definite `A` whose eigenvalues differ.
    fn quadratic_hessian() -> ndarray::Array2<f64> {
        array![
            [4.0, 1.0, 0.0, 0.5, 0.0, 0.0],
            [1.0, 3.0, 0.5, 0.0, 0.0, 0.25],
            [0.0, 0.5, 2.0, 0.25, 0.0, 0.0],
            [0.5, 0.0, 0.25, 1.5, 0.125, 0.0],
            [0.0, 0.0, 0.0, 0.125, 1.0, 0.0],
            [0.0, 0.25, 0.0, 0.0, 0.0, 0.75],
        ]
    }

    /// `-H g` with `H` the product-form BFGS update of `h0`, one pair at a
    /// time, oldest first: `H ← (I - rho s yᵀ) H (I - rho y sᵀ) + rho s sᵀ`.
    fn dense_direction(
        h0: ndarray::Array2<f64>,
        pairs: &[(Array1<f64>, Array1<f64>, f64)],
        g: &Array1<f64>,
    ) -> Array1<f64> {
        let n = g.len();
        let mut h = h0;
        for (s, y, rho) in pairs {
            let s2 = s.view().insert_axis(ndarray::Axis(1));
            let y2 = y.view().insert_axis(ndarray::Axis(1));
            let v = ndarray::Array2::<f64>::eye(n) - *rho * y2.dot(&s2.t());
            h = v.t().dot(&h).dot(&v) + *rho * s2.dot(&s2.t());
        }
        -h.dot(g)
    }

    fn assert_close(got: &Array1<f64>, want: &Array1<f64>, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            let tol = 1e-12 * (1.0 + a.abs().max(b.abs()));
            assert!((a - b).abs() <= tol, "{what} i={i} two-loop {a} dense {b}");
        }
    }

    #[test]
    fn matches_the_dense_bfgs_inverse_on_a_quadratic() {
        let a = quadratic_hessian();
        let steps = [
            array![1.0, 0.0, -0.5, 0.25, 0.0, 0.5],
            array![0.0, 1.0, 0.25, 0.0, -0.5, 0.0],
            array![0.5, -0.25, 1.0, 0.5, 0.25, -1.0],
            array![-0.25, 0.5, 0.0, 1.0, 0.75, 0.25],
        ];
        let history: Vec<(Array1<f64>, Array1<f64>)> =
            steps.iter().map(|s| (s.clone(), a.dot(s))).collect();
        let g = array![1.0, -2.0, 0.5, 0.25, -1.0, 3.0];
        let mut pairs = LbfgsPairs::new(6, 8).unwrap();
        let mut exact = Vec::new();
        for (s, y) in &history {
            assert!(pairs.push(s.view(), y.view()).unwrap());
            exact.push((s.clone(), y.clone(), 1.0 / s.dot(y)));
        }
        let (s_new, y_new) = history.last().unwrap();
        let gamma = gamma_of(s_new, y_new);
        let diag = array![0.5, 0.25, 2.0, 1.0, 0.75, 1.5];

        let scaled = pairs.direction(g.view(), LbfgsH0::Scale(gamma)).unwrap();
        let want = dense_direction(ndarray::Array2::eye(6) * gamma, &exact, &g);
        assert_close(&scaled, &want, "scalar H0");

        let want = dense_direction(ndarray::Array2::from_diag(&diag), &exact, &g);
        let from_diag = pairs
            .direction(g.view(), LbfgsH0::Diagonal(diag.view()))
            .unwrap();
        assert_close(&from_diag, &want, "diagonal H0");
        let from_callback = pairs
            .direction(g.view(), LbfgsH0::Apply(&|v| Ok(&diag * &v)))
            .unwrap();
        assert_eq!(from_callback, from_diag);

        // The newest pair satisfies the secant equation H y = s.
        let secant = pairs
            .direction(y_new.view(), LbfgsH0::Diagonal(diag.view()))
            .unwrap();
        assert_close(&secant, &-s_new, "secant");

        let mut damped = LbfgsPairs::new(6, 8).unwrap();
        let mut scaled_rho = Vec::new();
        for (s, y) in &history {
            let rho = 0.75 / s.dot(y);
            assert!(damped.push_rho(s.view(), y.view(), rho).unwrap());
            scaled_rho.push((s.clone(), y.clone(), rho));
        }
        let got = damped
            .direction(g.view(), LbfgsH0::Diagonal(diag.view()))
            .unwrap();
        let want = dense_direction(ndarray::Array2::from_diag(&diag), &scaled_rho, &g);
        assert_close(&got, &want, "caller rho");
    }

    /// The eOn step of a capped session is the handle's direction for
    /// the session's pairs and its `H0`, bit for bit.
    #[test]
    fn matches_the_capped_session_step() {
        let a = quadratic_hessian();
        let fg = |x: ArrayView1<f64>| {
            let g = a.dot(&x);
            Some((0.5 * x.dot(&g), g))
        };
        let cap = 10.0;
        let mut opt = Lbfgs::capped(cap);
        let x0 = array![1.0, -1.0, 0.5, 2.0, -0.5, 1.5];
        let (_f, x, _evals) = opt.minimize(x0.view(), 4, fg);
        assert!(opt.len() >= 2, "{} pairs", opt.len());

        let mut pairs = LbfgsPairs::new(6, opt.max_pairs).unwrap();
        for (s, y) in opt.pairs() {
            assert!(pairs.push(s.view(), y.view()).unwrap());
        }
        assert_eq!(pairs.len(), opt.len());
        let g = a.dot(&x);
        let h0 = match opt.eon_seed() {
            crate::lbfgs::EonSeed::Scale(h0) => h0,
            crate::lbfgs::EonSeed::Steepest => panic!("a convex quadratic has positive curvature"),
        };
        let mut txn = opt.begin_txn();
        let take = opt
            .eon_step_from_seed(g.view(), cap, crate::lbfgs::EonSeed::Scale(h0), &mut txn)
            .unwrap();
        assert!(!take.fallback, "the step must be the two-loop, not a reset");
        let from_handle = pairs.direction(g.view(), LbfgsH0::Scale(h0)).unwrap();
        assert_eq!(take.step, from_handle);
        let from_solver = opt.two_loop_h0(g.view(), LbfgsH0::Scale(h0)).unwrap();
        assert_eq!(from_solver, from_handle);
    }

    /// `push` and `push_rho` keep exactly the pairs `Lbfgs::record` keeps,
    /// and the directions agree bit for bit.
    #[test]
    fn push_and_push_rho_follow_the_record_gate() {
        let candidates = [
            (array![1.0, 0.5, -0.25], array![2.0, 1.0, 0.5]),
            (array![1.0, 0.0, 0.0], array![-1.0, 0.0, 0.0]),
            (array![1.0, 0.0, 0.0], array![0.0, 1.0, 0.0]),
            (array![1.0, 0.0, 0.0], array![1e-9, 1.0, 0.0]),
            (array![0.5, -0.25, 1.0], array![f64::NAN, 0.5, 2.0]),
            (array![0.5, -0.25, 1.0], array![1.0, f64::INFINITY, 2.0]),
            (array![0.0, 0.0, 0.0], array![1.0, 0.5, 2.0]),
            (array![0.5, -0.25, 1.0], array![1.0, 0.5, 2.0]),
        ];
        let mut opt = Lbfgs::with_capacity(4);
        let mut pairs = LbfgsPairs::new(3, 4).unwrap();
        let mut with_rho = LbfgsPairs::new(3, 4).unwrap();
        let mut kept = Vec::new();
        for (s, y) in &candidates {
            let recorded = opt.push(s.clone(), y.clone());
            let pushed = pairs.push(s.view(), y.view()).unwrap();
            let sy = s.dot(y);
            let rho = if sy.is_finite() && sy > 0.0 {
                1.0 / sy
            } else {
                1.0
            };
            let pushed_rho = with_rho.push_rho(s.view(), y.view(), rho).unwrap();
            assert_eq!(pushed, recorded, "push vs record for s={s} y={y}");
            assert_eq!(pushed_rho, recorded, "push_rho vs record for s={s} y={y}");
            kept.push(recorded);
        }
        assert_eq!(kept, [true, false, false, false, false, false, false, true]);
        assert_eq!(pairs.len(), opt.len());
        assert_eq!(with_rho.len(), opt.len());

        let g = array![1.0, -2.0, 0.5];
        let gamma = 0.375;
        let internal = opt.two_loop_h0(g.view(), LbfgsH0::Scale(gamma)).unwrap();
        assert_eq!(
            pairs.direction(g.view(), LbfgsH0::Scale(gamma)).unwrap(),
            internal
        );
        assert_eq!(
            with_rho.direction(g.view(), LbfgsH0::Scale(gamma)).unwrap(),
            internal
        );
    }

    #[test]
    fn push_rho_declines_a_bad_rho_and_refuses_a_bad_length() {
        let Sample { s1, y1, s2, y2, g } = sample();
        let mut pairs = LbfgsPairs::new(3, 2).unwrap();
        for rho in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                !pairs.push_rho(s1.view(), y1.view(), rho).unwrap(),
                "rho {rho}"
            );
        }
        assert!(pairs.is_empty());
        let short = pairs.push_rho(array![1.0].view(), y1.view(), 0.5);
        assert!(short.is_err(), "{short:?}");
        let long = pairs.push_rho(s1.view(), array![1.0, 0.0, 0.0, 0.0].view(), 0.5);
        assert!(long.is_err(), "{long:?}");
        assert!(pairs.is_empty());
        assert_eq!(
            pairs.direction(g.view(), LbfgsH0::Scale(0.5)).unwrap(),
            &g * -0.5
        );

        assert!(pairs.push_rho(s1.view(), y1.view(), 0.25).unwrap());
        assert!(pairs.push_rho(s2.view(), y2.view(), 0.25).unwrap());
        assert!(pairs.push_rho(s1.view(), y1.view(), 0.5).unwrap());
        assert_eq!(pairs.len(), 2, "capacity 2 drops the oldest pair");
        let mut newest = LbfgsPairs::new(3, 2).unwrap();
        assert!(newest.push_rho(s2.view(), y2.view(), 0.25).unwrap());
        assert!(newest.push_rho(s1.view(), y1.view(), 0.5).unwrap());
        assert_eq!(
            pairs.direction(g.view(), LbfgsH0::Scale(0.5)).unwrap(),
            newest.direction(g.view(), LbfgsH0::Scale(0.5)).unwrap()
        );
    }

    #[test]
    fn a_diagonal_h0_matches_the_dense_update() {
        let Sample { s1, y1, s2, y2, g } = sample();
        let diag = array![0.5, 0.25, 2.0];
        let mut pairs = LbfgsPairs::new(3, 4).unwrap();
        assert!(pairs.push(s1.view(), y1.view()).unwrap());
        assert!(pairs.push(s2.view(), y2.view()).unwrap());
        let got = pairs
            .direction(g.view(), LbfgsH0::Diagonal(diag.view()))
            .unwrap();
        let want = numpy_dense(
            diag.as_slice().unwrap(),
            &[
                (s1.as_slice().unwrap(), y1.as_slice().unwrap()),
                (s2.as_slice().unwrap(), y2.as_slice().unwrap()),
            ],
            g.as_slice().unwrap(),
        );
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            let tol = 1e-10 * (1.0 + a.abs().max(b.abs()));
            assert!((a - b).abs() <= tol, "i={i} got {a} numpy {b}");
        }
    }

    #[test]
    fn a_callback_h0_matches_the_diagonal() {
        let Sample { s1, y1, g, .. } = sample();
        let diag = array![0.5, 0.25, 2.0];
        let mut pairs = LbfgsPairs::new(3, 4).unwrap();
        assert!(pairs.push(s1.view(), y1.view()).unwrap());
        let from_diag = pairs
            .direction(g.view(), LbfgsH0::Diagonal(diag.view()))
            .unwrap();
        let from_callback = pairs
            .direction(g.view(), LbfgsH0::Apply(&|v| Ok(&diag * &v)))
            .unwrap();
        assert_eq!(from_callback, from_diag);
    }

    #[test]
    fn empty_memory_is_minus_h0_g() {
        let pairs = LbfgsPairs::new(3, 0).unwrap();
        assert!(pairs.is_empty());
        let g = array![1.0, -2.0, 0.5];
        let diag = array![0.5, 0.25, 2.0];
        let from_diag = pairs
            .direction(g.view(), LbfgsH0::Diagonal(diag.view()))
            .unwrap();
        assert_eq!(from_diag, array![-0.5, 0.5, -1.0]);
        let from_callback = pairs
            .direction(g.view(), LbfgsH0::Apply(&|v| Ok(&diag * &v)))
            .unwrap();
        assert_eq!(from_callback, from_diag);
        let from_scale = lbfgs_direction(g.view(), &[], &[], LbfgsH0::Scale(0.5)).unwrap();
        assert_eq!(from_scale, &g * -0.5);
    }

    #[test]
    fn bad_lengths_are_errors() {
        let err = LbfgsPairs::new(0, 4).unwrap_err();
        assert!(err.to_string().contains("dimension"), "{err}");

        let mut pairs = LbfgsPairs::new(3, 2).unwrap();
        let bad = pairs.push(array![1.0].view(), array![1.0, 0.0, 0.0].view());
        assert!(bad.is_err(), "{bad:?}");
        assert!(pairs.is_empty());

        let declined = pairs.push(array![1.0, 0.0, 0.0].view(), array![-1.0, 0.0, 0.0].view());
        assert!(!declined.unwrap());
        assert_eq!(pairs.len(), 0);

        let g = array![1.0, -2.0, 0.5];
        assert!(
            pairs
                .direction(array![1.0].view(), LbfgsH0::Scale(1.0))
                .is_err()
        );
        assert!(
            pairs
                .direction(g.view(), LbfgsH0::Diagonal(array![1.0].view()))
                .is_err()
        );
        assert!(pairs.direction(g.view(), LbfgsH0::Scale(f64::NAN)).is_err());
        let short = pairs.direction(g.view(), LbfgsH0::Apply(&|_v| Ok(array![1.0])));
        assert!(short.is_err(), "{short:?}");

        let s = array![1.0, 0.0, 0.0];
        let mismatch = lbfgs_direction(g.view(), &[s.view()], &[], LbfgsH0::Scale(1.0));
        assert!(mismatch.is_err(), "{mismatch:?}");
        let negative = lbfgs_direction(
            g.view(),
            &[array![1.0, 0.0, 0.0].view()],
            &[array![-1.0, 0.0, 0.0].view()],
            LbfgsH0::Scale(1.0),
        );
        assert!(negative.is_err(), "{negative:?}");
    }

    fn numpy_dense(diag: &[f64], pairs: &[(&[f64], &[f64])], g: &[f64]) -> Vec<f64> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let script = r#"
import json, sys
import numpy as np
req = json.load(sys.stdin)
h = np.diag(np.asarray(req["diag"], dtype=float))
for s, y in req["pairs"]:
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = 1.0 / float(np.dot(s, y))
    b = np.eye(s.size) - rho * np.outer(y, s)
    h = b.T @ h @ b + rho * np.outer(s, s)
json.dump((-h @ np.asarray(req["g"], dtype=float)).tolist(), sys.stdout)
"#;
        let mut body = String::from("{\"diag\":");
        body.push_str(&json_list(diag));
        body.push_str(",\"g\":");
        body.push_str(&json_list(g));
        body.push_str(",\"pairs\":[");
        for (i, (s, y)) in pairs.iter().enumerate() {
            if i > 0 {
                body.push(',');
            }
            body.push('[');
            body.push_str(&json_list(s));
            body.push(',');
            body.push_str(&json_list(y));
            body.push(']');
        }
        body.push_str("]}");
        let py = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(&py)
            .arg("-c")
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|err| panic!("{py} spawn: {err}"));
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(body.as_bytes())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "numpy dense reference failed: {text} {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let inner = text.trim().trim_start_matches('[').trim_end_matches(']');
        inner
            .split(',')
            .map(|s| s.trim().parse::<f64>().expect("numpy entry"))
            .collect()
    }

    fn json_list(values: &[f64]) -> String {
        let mut out = String::from("[");
        for (i, v) in values.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("{v:?}"));
        }
        out.push(']');
        out
    }
}
