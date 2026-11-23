//! Restarted flexible GMRES and the Eisenstat-Walker forcing term.
//!
//! [`fgmres`] solves a nonsymmetric `A x = b` with only the action of
//! `A`, preconditioned on the right. Right preconditioning keeps the
//! minimised quantity the residual `||b - A x||` of the original system,
//! so a caller's stopping test and any merit function built on it read
//! the same norm with or without a preconditioner. The flexible variant
//! stores the preconditioned directions `z_j = M_j^{-1} v_j`, so the
//! preconditioner may change between iterations (an inner iterative
//! solve, a block factor refreshed mid-run).
//!
//! Inside a cycle the residual is the Givens estimate `|g_{j+1}|`. It
//! equals `||b - A x_j||` only in exact arithmetic: rounding, a lost
//! Arnoldi orthogonality or an operator that is not exactly linear
//! separates the two. Every cycle therefore ends by forming its iterate
//! and recomputing `b - A x`, one action of `A`. Only that recomputed
//! residual converges a solve, is reported, and starts the next cycle.
//!
//! Saad, *A flexible inner-outer preconditioned GMRES algorithm*,
//! <https://doi.org/10.1137/0914028>. Saad and Schultz, *GMRES*,
//! <https://doi.org/10.1137/0907058>. Two Gram-Schmidt passes per
//! Arnoldi step: Giraud, Langou and Rozložník, *The loss of
//! orthogonality in the Gram-Schmidt orthogonalization process*,
//! <https://doi.org/10.1016/j.camwa.2005.08.009>.
//!
//! [`EisenstatWalker`] is choice 2 of Eisenstat and Walker, *Choosing the
//! forcing terms in an inexact Newton method*,
//! <https://doi.org/10.1137/0917003>, with the oversolving bound of
//! Kelley, *Iterative Methods for Linear and Nonlinear Equations*, eq.
//! (6.20), <https://doi.org/10.1137/1.9781611970944>.

use std::num::NonZeroUsize;

use ndarray::{Array1, ArrayView1};

use crate::error::{Error, Result};
use crate::vecops::{axpy, dot, nrm2};

/// Controls of one [`fgmres`] solve.
///
/// Non-exhaustive: start from [`Self::default`] and chain the `with_*`
/// setters.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fgmres {
    /// Stop when the recomputed `||b - A x||` is at most `rtol * ||b||`.
    pub rtol: f64,
    /// Arnoldi steps per cycle.
    pub restart: usize,
    /// Actions of `A` in total: the initial residual of a starting
    /// iterate, every Arnoldi step and the closing residual of every
    /// cycle. A cycle needs two, one step and its closing residual. A
    /// restarted run has no dimension bound, so the budget is the
    /// caller's.
    pub max_products: usize,
    /// Stop after this many consecutive cycles whose recomputed residual
    /// is not below the smallest one before them. `None` stops such a run
    /// only at the budget. One cycle that does not decrease the residual
    /// proves nothing about the next: its residual direction differs and a
    /// flexible preconditioner may differ too.
    pub stagnation: Option<NonZeroUsize>,
}

with_setters!(Fgmres {
    with_rtol => rtol: val f64,
    with_restart => restart: val usize,
    with_max_products => max_products: val usize,
    with_stagnation => stagnation: opt NonZeroUsize,
});

impl Default for Fgmres {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            restart: 30,
            max_products: 1000,
            stagnation: None,
        }
    }
}

/// How an [`fgmres`] solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FgmresStatus {
    /// The recomputed residual reached `rtol * ||b||`, or `b` was zero.
    Converged,
    /// The budget cannot pay for another cycle.
    Exhausted,
    /// [`Fgmres::stagnation`] cycles in a row did not lower the residual.
    Stagnated,
    /// The Hessenberg least-squares problem became singular, so the
    /// flexible basis does not determine a correction.
    Breakdown,
}

/// One residual, as the observer sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FgmresStep {
    /// Arnoldi steps so far over all cycles; zero is the initial residual.
    pub iteration: usize,
    /// Actions of `A` so far.
    pub products: usize,
    /// `||b - A x||` of the current iterate.
    pub residual: f64,
    /// `residual` was recomputed from `b - A x`: the initial residual and
    /// the close of every cycle. Otherwise it is the Givens estimate.
    pub exact: bool,
}

/// Result of [`fgmres`].
#[derive(Clone, Debug, PartialEq)]
pub struct FgmresReport {
    /// The iterate with the smallest recomputed residual.
    pub x: Array1<f64>,
    /// How the solve ended.
    pub status: FgmresStatus,
    /// The recomputed `||b - A x||` of `x`.
    pub residual: f64,
    /// `||b||`.
    pub rhs_norm: f64,
    /// Actions of `A`.
    pub products: usize,
    /// Arnoldi steps over all cycles.
    pub iterations: usize,
    /// Cycles closed by a recomputed residual.
    pub cycles: usize,
}

/// Right-preconditioned, restarted flexible GMRES.
///
/// `apply(v)` is `A v` and `precondition(v)` an approximation of `A^{-1}
/// v`; either may fail, which ends the solve with its error. The error
/// type is the caller's, so an operator's own failure passes through
/// unchanged. `x0` is the starting iterate; with `None` the solve starts
/// at zero and spends no product on the initial residual. `observe` sees
/// the initial residual, every Arnoldi step and every closing residual.
///
/// # Errors
///
/// [`Error::Krylov`] for a non-positive or non-finite `rtol`, a zero
/// restart or budget, mismatched lengths or a non-finite right-hand
/// side or starting iterate, [`Error::Oracle`] for a non-finite operator
/// or preconditioner image; otherwise any error of `apply` or
/// `precondition`.
pub fn fgmres<A, M, O, E>(
    mut apply: A,
    mut precondition: M,
    rhs: ArrayView1<f64>,
    x0: Option<ArrayView1<f64>>,
    config: &Fgmres,
    mut observe: O,
) -> std::result::Result<FgmresReport, E>
where
    A: FnMut(ArrayView1<f64>) -> std::result::Result<Array1<f64>, E>,
    M: FnMut(ArrayView1<f64>) -> std::result::Result<Array1<f64>, E>,
    O: FnMut(&FgmresStep),
    E: From<Error>,
{
    let n = rhs.len();
    if !(config.rtol.is_finite() && config.rtol > 0.0) {
        return Err(Error::Krylov {
            what: "rtol must be finite and positive",
        }
        .into());
    }
    if config.restart == 0 || config.max_products == 0 {
        return Err(Error::Krylov {
            what: "restart and max_products must be positive",
        }
        .into());
    }
    if x0.is_some_and(|x| x.len() != n) {
        return Err(Error::Krylov {
            what: "x0 does not match the right-hand side",
        }
        .into());
    }
    if rhs.iter().any(|v| !v.is_finite()) {
        return Err(Error::Krylov {
            what: "the right-hand side is not finite",
        }
        .into());
    }
    if x0.is_some_and(|x| x.iter().any(|v| !v.is_finite())) {
        return Err(Error::Krylov {
            what: "x0 is not finite",
        }
        .into());
    }
    let rhs_norm = nrm2(rhs);
    if !rhs_norm.is_finite() {
        return Err(Error::Krylov {
            what: "the right-hand side is not finite",
        }
        .into());
    }
    let target = config.rtol * rhs_norm;
    if !target.is_finite() {
        return Err(Error::Krylov {
            what: "the residual target overflowed",
        }
        .into());
    }
    let mut x = x0.map_or_else(|| Array1::zeros(n), |x| x.to_owned());
    let mut products = 0usize;
    let mut r = rhs.to_owned();
    if x0.is_some() {
        r -= &checked(apply(x.view())?, n, "A x")?;
        products += 1;
    }
    let mut residual = nrm2(r.view());
    let mut iterations = 0usize;
    let mut cycles = 0usize;
    observe(&FgmresStep {
        iteration: 0,
        products,
        residual,
        exact: true,
    });
    let finish = |x, status, residual, products, iterations, cycles| FgmresReport {
        x,
        status,
        residual,
        rhs_norm,
        products,
        iterations,
        cycles,
    };
    if !residual.is_finite() {
        return Ok(finish(
            x,
            FgmresStatus::Breakdown,
            residual,
            products,
            iterations,
            cycles,
        ));
    }
    if residual <= target {
        return Ok(finish(
            x,
            FgmresStatus::Converged,
            residual,
            products,
            iterations,
            cycles,
        ));
    }
    let mut best = (x.clone(), residual);
    let mut stalled = 0usize;
    loop {
        if config.max_products - products < 2 {
            return Ok(finish(
                best.0,
                FgmresStatus::Exhausted,
                best.1,
                products,
                iterations,
                cycles,
            ));
        }
        let unit = &r / residual;
        if unit.iter().any(|v| !v.is_finite()) {
            return Ok(finish(
                best.0,
                FgmresStatus::Breakdown,
                best.1,
                products,
                iterations,
                cycles,
            ));
        }
        let mut basis = vec![unit];
        let mut directions: Vec<Array1<f64>> = Vec::new();
        let mut columns: Vec<Vec<f64>> = Vec::new();
        let mut rotations: Vec<(f64, f64)> = Vec::new();
        let mut g = vec![residual];
        while directions.len() < config.restart && config.max_products - products >= 2 {
            let j = directions.len();
            let z = checked(precondition(basis[j].view())?, n, "the preconditioner")?;
            let mut w = checked(apply(z.view())?, n, "A z")?;
            products += 1;
            let mut h = vec![0.0; j + 2];
            for _pass in 0..2 {
                for (hi, v) in h.iter_mut().zip(&basis) {
                    let c = dot(w.view(), v.view());
                    axpy(-c, v.view(), &mut w);
                    *hi += c;
                }
            }
            let next = nrm2(w.view());
            if !next.is_finite()
                || w.iter().any(|v| !v.is_finite())
                || h[..=j].iter().any(|v| !v.is_finite())
            {
                return Ok(finish(
                    best.0,
                    FgmresStatus::Breakdown,
                    best.1,
                    products,
                    iterations,
                    cycles,
                ));
            }
            h[j + 1] = next;
            for (i, &(c, s)) in rotations.iter().enumerate() {
                let (a, b) = (h[i], h[i + 1]);
                h[i] = c * a + s * b;
                h[i + 1] = -s * a + c * b;
            }
            let radius = h[j].hypot(h[j + 1]);
            if !radius.is_finite() || h.iter().any(|v| !v.is_finite()) {
                return Ok(finish(
                    best.0,
                    FgmresStatus::Breakdown,
                    best.1,
                    products,
                    iterations,
                    cycles,
                ));
            }
            let (c, s) = if radius == 0.0 {
                (1.0, 0.0)
            } else {
                (h[j] / radius, h[j + 1] / radius)
            };
            h[j] = radius;
            h[j + 1] = 0.0;
            rotations.push((c, s));
            g.push(-s * g[j]);
            g[j] *= c;
            iterations += 1;
            directions.push(z);
            columns.push(h);
            let estimate = if radius == 0.0 {
                g[j].abs()
            } else {
                g[j + 1].abs()
            };
            observe(&FgmresStep {
                iteration: iterations,
                products,
                residual: estimate,
                exact: false,
            });
            if estimate <= target || next == 0.0 {
                break;
            }
            let next_basis = &w / next;
            if next_basis.iter().any(|v| !v.is_finite()) {
                return Ok(finish(
                    best.0,
                    FgmresStatus::Breakdown,
                    best.1,
                    products,
                    iterations,
                    cycles,
                ));
            }
            basis.push(next_basis);
        }
        let k = directions.len();
        let mut y = vec![0.0; k];
        for i in (0..k).rev() {
            let mut s = g[i];
            for (l, yl) in y.iter().enumerate().skip(i + 1) {
                s -= columns[l][i] * yl;
            }
            let diag = columns[i][i];
            if !diag.is_finite() || diag == 0.0 {
                return Ok(finish(
                    best.0,
                    FgmresStatus::Breakdown,
                    best.1,
                    products,
                    iterations,
                    cycles,
                ));
            }
            let yi = s / diag;
            if !yi.is_finite() {
                return Ok(finish(
                    best.0,
                    FgmresStatus::Breakdown,
                    best.1,
                    products,
                    iterations,
                    cycles,
                ));
            }
            y[i] = yi;
        }
        for (z, yi) in directions.iter().zip(&y) {
            axpy(*yi, z.view(), &mut x);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Ok(finish(
                best.0,
                FgmresStatus::Breakdown,
                best.1,
                products,
                iterations,
                cycles,
            ));
        }
        r = &rhs - &checked(apply(x.view())?, n, "A x")?;
        products += 1;
        cycles += 1;
        residual = nrm2(r.view());
        observe(&FgmresStep {
            iteration: iterations,
            products,
            residual,
            exact: true,
        });
        if !residual.is_finite() {
            return Ok(finish(
                best.0,
                FgmresStatus::Breakdown,
                best.1,
                products,
                iterations,
                cycles,
            ));
        }
        if residual <= target {
            return Ok(finish(
                x,
                FgmresStatus::Converged,
                residual,
                products,
                iterations,
                cycles,
            ));
        }
        if residual < best.1 {
            best = (x.clone(), residual);
            stalled = 0;
        } else {
            stalled += 1;
        }
        if config
            .stagnation
            .is_some_and(|limit| stalled >= limit.get())
        {
            return Ok(finish(
                best.0,
                FgmresStatus::Stagnated,
                best.1,
                products,
                iterations,
                cycles,
            ));
        }
    }
}

fn checked(v: Array1<f64>, n: usize, what: &'static str) -> Result<Array1<f64>> {
    if v.len() != n {
        return Err(Error::Krylov {
            what: "an operator returned the wrong length",
        });
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::Oracle { what });
    }
    Ok(v)
}

/// Eisenstat-Walker choice 2 for an inexact Newton iteration.
///
/// `eta_k = gamma (||F_k|| / ||F_{k-1}||)^alpha`, kept at least
/// `gamma eta_{k-1}^alpha` (the paper's safeguard, here without its
/// `0.1` threshold) and at least `target / (2 ||F_k||)` (Kelley's bound,
/// which stops a linear solve below what the outer tolerance `target`
/// needs). Both residuals must belong to one nonlinear system; the ratio
/// across a change of system means nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EisenstatWalker {
    gamma: f64,
    alpha: f64,
}

impl EisenstatWalker {
    /// `gamma` in `(0, 1]` and `alpha` in `(1, 2]`, the range of the
    /// choice-2 convergence theorem.
    ///
    /// # Errors
    ///
    /// [`Error::Krylov`] outside that range.
    pub fn new(gamma: f64, alpha: f64) -> Result<Self> {
        if !(gamma > 0.0 && gamma <= 1.0) {
            return Err(Error::Krylov {
                what: "Eisenstat-Walker gamma must lie in (0, 1]",
            });
        }
        if !(alpha > 1.0 && alpha <= 2.0) {
            return Err(Error::Krylov {
                what: "Eisenstat-Walker alpha must lie in (1, 2]",
            });
        }
        Ok(Self { gamma, alpha })
    }

    /// The coefficient `gamma`.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// The exponent `alpha`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The forcing term at residual `residual`, with `previous` the
    /// `(||F_{k-1}||, eta_{k-1})` of the same system.
    ///
    /// Returns `Ok(None)` on the first step or when the value leaves `(0, 1)`,
    /// where it gives no inexact-Newton contraction (Dembo, Eisenstat and
    /// Steihaug, <https://doi.org/10.1137/0719025>); the caller keeps
    /// its own rule there.
    ///
    /// # Errors
    ///
    /// [`Error::Krylov`] when `residual` or `last` is not finite and
    /// positive, or when `eta_last` or `target` is not finite and
    /// non-negative.
    pub fn forcing(
        &self,
        residual: f64,
        previous: Option<(f64, f64)>,
        target: f64,
    ) -> Result<Option<f64>> {
        if !(residual.is_finite() && residual > 0.0) {
            return Err(Error::Krylov {
                what: "Eisenstat-Walker residual must be finite and positive",
            });
        }
        if !(target.is_finite() && target >= 0.0) {
            return Err(Error::Krylov {
                what: "Eisenstat-Walker target must be finite and non-negative",
            });
        }
        let Some((last, eta_last)) = previous else {
            return Ok(None);
        };
        if !(last.is_finite() && last > 0.0) {
            return Err(Error::Krylov {
                what: "Eisenstat-Walker previous residual must be finite and positive",
            });
        }
        if !(eta_last.is_finite() && eta_last >= 0.0) {
            return Err(Error::Krylov {
                what: "Eisenstat-Walker previous forcing term must be finite and non-negative",
            });
        }
        let eta = (self.gamma * (residual / last).powf(self.alpha))
            .max(self.gamma * eta_last.powf(self.alpha))
            .max(0.5 * target / residual);
        Ok((eta.is_finite() && eta > 0.0 && eta < 1.0).then_some(eta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Nonsymmetric, diagonally dominant, with a spread spectrum.
    fn operator(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j {
                2.0 + i as f64
            } else if j == i + 1 {
                1.0
            } else if i == j + 2 {
                -0.5
            } else {
                0.0
            }
        })
    }

    fn rhs(n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |i| ((i + 1) as f64).sin())
    }

    fn true_residual(a: &Array2<f64>, b: &Array1<f64>, x: &Array1<f64>) -> f64 {
        nrm2((b - &a.dot(x)).view())
    }

    fn config(rtol: f64, restart: usize, max_products: usize) -> Fgmres {
        Fgmres {
            rtol,
            restart,
            max_products,
            stagnation: None,
        }
    }

    /// The reported residual is the recomputed one, so it meets the
    /// tolerance itself rather than within a slack factor.
    #[test]
    fn unpreconditioned_certifies_the_true_residual() -> Result<()> {
        let (a, b) = (operator(40), rhs(40));
        let mut steps = Vec::new();
        let out = fgmres(
            |v| Ok::<_, Error>(a.dot(&v)),
            |v| Ok(v.to_owned()),
            b.view(),
            None,
            &config(1e-10, 40, 41),
            |s| steps.push(*s),
        )?;
        assert_eq!(out.status, FgmresStatus::Converged);
        assert_eq!(out.residual, true_residual(&a, &b, &out.x));
        assert!(out.residual <= 1e-10 * out.rhs_norm);
        assert_eq!(out.products, out.iterations + out.cycles);
        assert!(steps[0].exact && steps[0].iteration == 0);
        let last = steps.last().copied();
        assert_eq!(
            last.map(|s| (s.exact, s.residual)),
            Some((true, out.residual))
        );
        assert_eq!(steps.len(), out.iterations + out.cycles + 1);
        Ok(())
    }

    #[test]
    fn the_exact_inverse_takes_one_step_and_its_check() -> Result<()> {
        let n = 12;
        let a = Array2::from_diag(&Array1::from_shape_fn(n, |i| 1.0 + i as f64));
        let b = rhs(n);
        let out = fgmres(
            |v| Ok::<_, Error>(a.dot(&v)),
            |v| Ok(Array1::from_shape_fn(n, |i| v[i] / (1.0 + i as f64))),
            b.view(),
            None,
            &config(1e-12, n, n + 1),
            |_| {},
        )?;
        assert_eq!(
            (out.status, out.products, out.iterations, out.cycles),
            (FgmresStatus::Converged, 2, 1, 1)
        );
        assert!(true_residual(&a, &b, &out.x) <= 1e-12 * out.rhs_norm);
        Ok(())
    }

    #[test]
    fn restarts_certify_the_true_residual_and_respect_the_budget() -> Result<()> {
        let (a, b) = (operator(60), rhs(60));
        let out = fgmres(
            |v| Ok::<_, Error>(a.dot(&v)),
            |v| Ok(v.to_owned()),
            b.view(),
            None,
            &config(1e-9, 5, 600),
            |_| {},
        )?;
        assert_eq!(out.status, FgmresStatus::Converged);
        assert!(true_residual(&a, &b, &out.x) <= 1e-9 * out.rhs_norm);
        assert!(out.cycles > 1);
        let cut = fgmres(
            |v| Ok::<_, Error>(a.dot(&v)),
            |v| Ok(v.to_owned()),
            b.view(),
            None,
            &config(1e-9, 5, 7),
            |_| {},
        )?;
        assert_eq!(
            (cut.status, cut.products, cut.cycles),
            (FgmresStatus::Exhausted, 6, 1)
        );
        assert_eq!(cut.residual, true_residual(&a, &b, &cut.x));
        assert!(cut.residual < cut.rhs_norm);
        Ok(())
    }

    /// An operator that is not linear: the Givens estimate of the Arnoldi
    /// model falls below the target while `b - A x` does not. The estimate
    /// must not end the solve.
    #[test]
    fn an_estimate_below_the_target_is_checked_against_b_minus_ax() -> Result<()> {
        let n = 6;
        let a = operator(n);
        let b = rhs(n);
        let bent = |v: ArrayView1<f64>| {
            let mut image = a.dot(&v);
            image[0] += 0.05 * dot(v, v);
            Ok::<_, Error>(image)
        };
        let mut steps = Vec::new();
        let out = fgmres(
            bent,
            |v| Ok(v.to_owned()),
            b.view(),
            None,
            &config(1e-10, n, 200),
            |s| steps.push(*s),
        )?;
        let fooled = steps.windows(2).any(|p| {
            !p[0].exact
                && p[0].residual <= 1e-10 * out.rhs_norm
                && p[1].exact
                && p[1].residual > 1e-10 * out.rhs_norm
        });
        assert!(fooled, "{steps:?}");
        assert_eq!(out.status, FgmresStatus::Converged);
        let mut image = a.dot(&out.x);
        image[0] += 0.05 * dot(out.x.view(), out.x.view());
        assert!(nrm2((&b - &image).view()) <= 1e-10 * out.rhs_norm);
        Ok(())
    }

    /// GMRES(1) on a rotation makes no progress with the identity. A
    /// flexible preconditioner that becomes the inverse on its second call
    /// converges in the next cycle, which a one-cycle stagnation rule would
    /// never reach.
    #[test]
    fn one_flat_cycle_is_not_stagnation() -> Result<()> {
        let a = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, -1.0, 0.0])
            .map_err(|_| Error::Krylov { what: "shape" })?;
        let b = Array1::from_vec(vec![1.0, 0.0]);
        let run = |stagnation: Option<NonZeroUsize>| {
            let mut calls = 0usize;
            fgmres(
                |v| Ok::<_, Error>(a.dot(&v)),
                |v: ArrayView1<f64>| {
                    calls += 1;
                    Ok(if calls == 1 {
                        v.to_owned()
                    } else {
                        Array1::from_vec(vec![-v[1], v[0]])
                    })
                },
                b.view(),
                None,
                &Fgmres {
                    rtol: 1e-12,
                    restart: 1,
                    max_products: 8,
                    stagnation,
                },
                |_| {},
            )
        };
        let open = run(None)?;
        assert_eq!((open.status, open.cycles), (FgmresStatus::Converged, 2));
        let patient = run(NonZeroUsize::new(2))?;
        assert_eq!(patient.status, FgmresStatus::Converged);
        let strict = run(NonZeroUsize::new(1))?;
        assert_eq!((strict.status, strict.cycles), (FgmresStatus::Stagnated, 1));
        assert_eq!(strict.residual, 1.0);
        Ok(())
    }

    /// A preconditioner that changes every call: Jacobi sweeps whose count
    /// grows with the iteration, which plain right-preconditioned GMRES
    /// cannot take.
    #[test]
    fn a_varying_preconditioner_converges() -> Result<()> {
        let (a, b) = (operator(50), rhs(50));
        let mut calls = 0usize;
        let jacobi = |v: ArrayView1<f64>| {
            calls += 1;
            let mut z = Array1::<f64>::zeros(v.len());
            for _ in 0..(1 + calls % 3) {
                let az = a.dot(&z);
                for i in 0..v.len() {
                    z[i] += (v[i] - az[i]) / a[[i, i]];
                }
            }
            Ok::<_, Error>(z)
        };
        let out = fgmres(
            |v| Ok(a.dot(&v)),
            jacobi,
            b.view(),
            None,
            &config(1e-10, 50, 51),
            |_| {},
        )?;
        assert_eq!(out.status, FgmresStatus::Converged);
        assert!(true_residual(&a, &b, &out.x) <= 1e-10 * out.rhs_norm);
        let plain = fgmres(
            |v| Ok::<_, Error>(a.dot(&v)),
            |v| Ok(v.to_owned()),
            b.view(),
            None,
            &config(1e-10, 50, 51),
            |_| {},
        )?;
        assert!(
            out.products < plain.products,
            "{} against {}",
            out.products,
            plain.products
        );
        Ok(())
    }

    #[test]
    fn a_starting_iterate_costs_one_product() -> Result<()> {
        let (a, b) = (operator(30), rhs(30));
        let exact = fgmres(
            |v| Ok::<_, Error>(a.dot(&v)),
            |v| Ok(v.to_owned()),
            b.view(),
            None,
            &config(1e-10, 30, 31),
            |_| {},
        )?;
        let warm = fgmres(
            |v| Ok::<_, Error>(a.dot(&v)),
            |v| Ok(v.to_owned()),
            b.view(),
            Some(exact.x.view()),
            &config(1e-10, 30, 31),
            |_| {},
        )?;
        assert_eq!(
            (warm.status, warm.products, warm.iterations, warm.cycles),
            (FgmresStatus::Converged, 1, 0, 0)
        );
        Ok(())
    }

    /// The caller's error passes through `fgmres` unchanged.
    #[test]
    fn an_operator_error_is_the_callers() {
        #[derive(Debug, PartialEq)]
        enum Caller {
            Surface,
            Krylov,
        }
        impl From<Error> for Caller {
            fn from(_: Error) -> Self {
                Caller::Krylov
            }
        }
        let b = rhs(4);
        let out = fgmres(
            |_| Err(Caller::Surface),
            |v| Ok(v.to_owned()),
            b.view(),
            None,
            &config(1e-8, 4, 5),
            |_| {},
        );
        assert_eq!(out.err(), Some(Caller::Surface));
        let refused = fgmres(
            |v: ArrayView1<f64>| Ok::<_, Caller>(v.to_owned()),
            |v| Ok(v.to_owned()),
            b.view(),
            None,
            &config(0.0, 4, 5),
            |_| {},
        );
        assert_eq!(refused.err(), Some(Caller::Krylov));
    }

    #[test]
    fn bad_input_is_refused() {
        let (a, b) = (operator(4), rhs(4));
        let run = |config: Fgmres| {
            fgmres(
                |v| Ok::<_, Error>(a.dot(&v)),
                |v| Ok(v.to_owned()),
                b.view(),
                None,
                &config,
                |_| {},
            )
        };
        assert!(run(config(0.0, 4, 4)).is_err());
        assert!(run(config(1e-8, 0, 4)).is_err());
        assert!(run(config(1e-8, 4, 0)).is_err());
        assert!(matches!(
            run(config(1e-8, 4, 1)),
            Ok(FgmresReport {
                status: FgmresStatus::Exhausted,
                products: 0,
                ..
            })
        ));
        let nan = fgmres(
            |v| Ok::<_, Error>(a.dot(&v)),
            |v| Ok(&v * f64::NAN),
            b.view(),
            None,
            &config(1e-8, 4, 5),
            |_| {},
        );
        assert!(matches!(nan, Err(Error::Oracle { .. })));
        let zero = fgmres(
            |v| Ok::<_, Error>(a.dot(&v)),
            |v| Ok(v.to_owned()),
            Array1::zeros(4).view(),
            None,
            &config(1e-8, 4, 5),
            |_| {},
        );
        assert!(matches!(
            zero,
            Ok(FgmresReport {
                status: FgmresStatus::Converged,
                products: 0,
                ..
            })
        ));
    }

    #[test]
    fn eisenstat_walker_choice_two() -> Result<()> {
        assert!(EisenstatWalker::new(0.0, 2.0).is_err());
        assert!(EisenstatWalker::new(1.1, 2.0).is_err());
        assert!(EisenstatWalker::new(0.9, 1.0).is_err());
        assert!(EisenstatWalker::new(0.9, 2.5).is_err());
        let ew = EisenstatWalker::new(0.9, 2.0)?;
        assert_eq!(ew.forcing(1e-2, None, 1e-10)?, None);
        let eta = ew
            .forcing(1e-2, Some((1e-1, 0.5)), 1e-10)?
            .ok_or(Error::Krylov { what: "no forcing" })?;
        assert_eq!(eta.to_bits(), (0.9f64 * 0.5f64.powf(2.0)).to_bits());
        let fast = ew
            .forcing(1e-4, Some((1e-1, 1e-3)), 1e-10)?
            .ok_or(Error::Krylov { what: "no forcing" })?;
        let expected = (0.9f64 * (1e-4f64 / 1e-1).powf(2.0))
            .max(0.9 * 1e-3f64.powf(2.0))
            .max(0.5 * 1e-10 / 1e-4);
        assert_eq!(fast.to_bits(), expected.to_bits());
        let oversolve = ew
            .forcing(1e-9, Some((1e-1, 1e-6)), 1e-9)?
            .ok_or(Error::Krylov { what: "no forcing" })?;
        assert_eq!(oversolve, 0.5);
        assert_eq!(ew.forcing(1.0, Some((0.5, 0.9)), 1e-10)?, None);
        Ok(())
    }

    #[test]
    fn fgmres_config_defaults_and_setters() {
        let cfg = Fgmres::default()
            .with_rtol(1e-10)
            .with_restart(12)
            .with_max_products(50)
            .with_stagnation(NonZeroUsize::new(3));
        assert_eq!(cfg.rtol, 1e-10);
        assert_eq!(cfg.restart, 12);
        assert_eq!(cfg.max_products, 50);
        assert_eq!(cfg.stagnation, NonZeroUsize::new(3));
        let cleared = cfg.with_stagnation(None);
        assert_eq!(cleared.stagnation, None);
    }

    #[test]
    fn breakdown_reports_true_residual_and_returns_best_iterate() -> Result<()> {
        // j = 0 zero operator: radius == 0.0 must not observe a false 0.0 residual.
        let b = rhs(4);
        let mut steps = Vec::new();
        let zero_op = fgmres(
            |v| Ok::<_, Error>(Array1::zeros(v.len())),
            |v| Ok(v.to_owned()),
            b.view(),
            None,
            &config(1e-8, 4, 10),
            |s| steps.push(*s),
        )?;
        assert_eq!(zero_op.status, FgmresStatus::Breakdown);
        assert_eq!(zero_op.x, Array1::<f64>::zeros(4));
        assert_eq!(zero_op.residual, zero_op.rhs_norm);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].residual, zero_op.rhs_norm);
        assert!(!steps[1].exact);
        assert_eq!(steps[1].residual, zero_op.rhs_norm);

        // j = 1 singular operator: radius == 0.0 mid-cycle reports |g[1]|, not 0.0.
        let sing = Array2::from_diag(&Array1::from_vec(vec![1.0, 0.0]));
        let b2 = Array1::from_vec(vec![3.0, 4.0]);
        let mut mid_steps = Vec::new();
        let mut mid_pre_calls = 0usize;
        let mid = fgmres(
            |v| Ok::<_, Error>(sing.dot(&v)),
            |_| {
                mid_pre_calls += 1;
                Ok(if mid_pre_calls == 1 {
                    Array1::from_vec(vec![1.0, 0.0])
                } else {
                    Array1::from_vec(vec![0.0, 1.0])
                })
            },
            b2.view(),
            None,
            &config(1e-8, 2, 10),
            |s| mid_steps.push(*s),
        )?;
        assert_eq!(mid.status, FgmresStatus::Breakdown);
        assert_eq!(mid.x, Array1::<f64>::zeros(2));
        assert_eq!(mid.residual, mid.rhs_norm);
        assert_eq!(mid_steps.len(), 3);
        assert!(!mid_steps[2].exact);
        assert_eq!(mid_steps[2].residual, 4.0);

        // Back-substitution overflow in y.
        let b_big = Array1::from_vec(vec![1e100, 0.0]);
        let y_overflow = fgmres(
            |v| Ok::<_, Error>(Array1::from_vec(vec![1e-250 * v[0], 0.0])),
            |v| Ok(v.to_owned()),
            b_big.view(),
            None,
            &config(1e-8, 2, 10),
            |_| {},
        )?;
        assert_eq!(y_overflow.status, FgmresStatus::Breakdown);
        assert_eq!(y_overflow.x, Array1::<f64>::zeros(2));
        assert_eq!(y_overflow.residual, y_overflow.rhs_norm);

        // Finite y whose axpy update overflows x.
        let b_mid = Array1::from_vec(vec![1e150, 0.0]);
        let x_overflow = fgmres(
            |v| Ok::<_, Error>(Array1::from_vec(vec![1e-200 * v[0], 0.0])),
            |v| Ok(Array1::from_vec(vec![1e200 * v[0], 0.0])),
            b_mid.view(),
            None,
            &config(1e-8, 2, 10),
            |_| {},
        )?;
        assert_eq!(x_overflow.status, FgmresStatus::Breakdown);
        assert_eq!(x_overflow.x, Array1::<f64>::zeros(2));
        assert_eq!(x_overflow.residual, x_overflow.rhs_norm);

        // nrm2(w) overflow does not poison the basis or emit Error::Oracle.
        let b_unit = Array1::from_vec(vec![1.0, 0.0]);
        let w_overflow = fgmres(
            |v| Ok::<_, Error>(Array1::from_vec(vec![0.0, 1e200 * v[0]])),
            |v| Ok(v.to_owned()),
            b_unit.view(),
            None,
            &config(1e-8, 2, 10),
            |_| {},
        )?;
        assert_eq!(w_overflow.status, FgmresStatus::Breakdown);
        assert_eq!(w_overflow.x, Array1::<f64>::zeros(2));
        assert_eq!(w_overflow.residual, w_overflow.rhs_norm);

        // nrm2(r) overflow at cycle 2 close returns the earlier best iterate from cycle 1.
        let b_two = Array1::from_vec(vec![1.0, 1.0]);
        let mut apply_calls = 0usize;
        let mut pre_calls = 0usize;
        let r_overflow = fgmres(
            |v| {
                apply_calls += 1;
                Ok::<_, Error>(if apply_calls <= 3 {
                    v.to_owned()
                } else {
                    Array1::from_vec(vec![1e200, 0.0])
                })
            },
            |_| {
                pre_calls += 1;
                Ok(if pre_calls == 1 {
                    Array1::from_vec(vec![1.0, 0.0])
                } else {
                    Array1::from_vec(vec![0.0, 0.5])
                })
            },
            b_two.view(),
            None,
            &config(1e-8, 1, 10),
            |_| {},
        )?;
        assert_eq!(r_overflow.status, FgmresStatus::Breakdown);
        assert!((r_overflow.x[0] - 1.0).abs() < 1e-14 && r_overflow.x[1] == 0.0);
        assert!((r_overflow.residual - 1.0).abs() < 1e-14);

        // nrm2(r) overflow on initial x0 ends in Breakdown without poisoning the basis.
        let x0 = Array1::from_vec(vec![1.0, 0.0]);
        let x0_overflow = fgmres(
            |_| Ok::<_, Error>(Array1::from_vec(vec![1e200, 0.0])),
            |v| Ok(v.to_owned()),
            b_unit.view(),
            Some(x0.view()),
            &config(1e-8, 2, 10),
            |_| {},
        )?;
        assert_eq!(x0_overflow.status, FgmresStatus::Breakdown);
        assert_eq!(x0_overflow.x, x0);
        assert!(x0_overflow.residual.is_infinite());
        Ok(())
    }

    #[test]
    fn non_finite_rhs_and_x0_are_refused_before_apply() {
        let valid_rhs = Array1::from_vec(vec![1.0, 2.0]);
        for bad_rhs in [
            Array1::from_vec(vec![f64::NAN, 1.0]),
            Array1::from_vec(vec![f64::INFINITY, 1.0]),
            Array1::from_vec(vec![f64::NEG_INFINITY, 1.0]),
            Array1::from_vec(vec![1e200, 0.0]),
        ] {
            let mut apply_calls = 0usize;
            let mut pre_calls = 0usize;
            let res = fgmres(
                |v| {
                    apply_calls += 1;
                    Ok::<_, Error>(v.to_owned())
                },
                |v| {
                    pre_calls += 1;
                    Ok(v.to_owned())
                },
                bad_rhs.view(),
                None,
                &config(1e-8, 2, 10),
                |_| {},
            );
            assert!(matches!(res, Err(Error::Krylov { .. })));
            assert_eq!((apply_calls, pre_calls), (0, 0));
        }

        for bad_x0 in [
            Array1::from_vec(vec![f64::NAN, 0.0]),
            Array1::from_vec(vec![0.0, f64::INFINITY]),
            Array1::from_vec(vec![0.0, f64::NEG_INFINITY]),
            Array1::from_vec(vec![0.0]),
        ] {
            let mut apply_calls = 0usize;
            let mut pre_calls = 0usize;
            let res = fgmres(
                |v| {
                    apply_calls += 1;
                    Ok::<_, Error>(v.to_owned())
                },
                |v| {
                    pre_calls += 1;
                    Ok(v.to_owned())
                },
                valid_rhs.view(),
                Some(bad_x0.view()),
                &config(1e-8, 2, 10),
                |_| {},
            );
            assert!(matches!(res, Err(Error::Krylov { .. })));
            assert_eq!((apply_calls, pre_calls), (0, 0));
        }
    }

    #[test]
    fn wrong_length_operator_and_preconditioner_are_refused() {
        let b = rhs(3);
        let x0 = Array1::from_vec(vec![1.0, 0.0, 0.0]);

        let bad_x0_apply = fgmres(
            |_| Ok::<_, Error>(Array1::zeros(2)),
            |v| Ok(v.to_owned()),
            b.view(),
            Some(x0.view()),
            &config(1e-8, 3, 10),
            |_| {},
        );
        assert!(matches!(bad_x0_apply, Err(Error::Krylov { .. })));

        let bad_pre = fgmres(
            |v| Ok::<_, Error>(v.to_owned()),
            |_| Ok(Array1::zeros(4)),
            b.view(),
            None,
            &config(1e-8, 3, 10),
            |_| {},
        );
        assert!(matches!(bad_pre, Err(Error::Krylov { .. })));

        let bad_arnoldi_apply = fgmres(
            |_| Ok::<_, Error>(Array1::zeros(4)),
            |v| Ok(v.to_owned()),
            b.view(),
            None,
            &config(1e-8, 3, 10),
            |_| {},
        );
        assert!(matches!(bad_arnoldi_apply, Err(Error::Krylov { .. })));

        let mut calls = 0usize;
        let bad_close_apply = fgmres(
            |v| {
                calls += 1;
                Ok::<_, Error>(if calls == 1 {
                    Array1::from_vec(vec![v[0], 2.0 * v[1], 3.0 * v[2]])
                } else {
                    Array1::zeros(2)
                })
            },
            |v| Ok(v.to_owned()),
            b.view(),
            None,
            &config(1e-8, 1, 10),
            |_| {},
        );
        assert!(matches!(bad_close_apply, Err(Error::Krylov { .. })));
    }

    #[test]
    fn eisenstat_walker_forcing_edge_cases() -> Result<()> {
        for bad in [f64::NAN, -0.5, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(EisenstatWalker::new(bad, 2.0).is_err());
            assert!(EisenstatWalker::new(0.9, bad).is_err());
        }

        let ew = EisenstatWalker::new(0.9, 2.0)?;
        for bad_pos in [f64::NAN, 0.0, -0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                ew.forcing(bad_pos, None, 1e-8),
                Err(Error::Krylov { .. })
            ));
            assert!(matches!(
                ew.forcing(bad_pos, Some((1.0, 0.5)), 1e-8),
                Err(Error::Krylov { .. })
            ));
            assert!(matches!(
                ew.forcing(0.5, Some((bad_pos, 0.5)), 1e-8),
                Err(Error::Krylov { .. })
            ));
        }
        for bad_nonneg in [f64::NAN, -1e-12, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                ew.forcing(0.5, None, bad_nonneg),
                Err(Error::Krylov { .. })
            ));
            assert!(matches!(
                ew.forcing(0.5, Some((1.0, 0.5)), bad_nonneg),
                Err(Error::Krylov { .. })
            ));
            assert!(matches!(
                ew.forcing(0.5, Some((1.0, bad_nonneg)), 1e-8),
                Err(Error::Krylov { .. })
            ));
        }

        assert_eq!(ew.forcing(1e-2, None, 0.0)?, None);
        let zero_floor = ew
            .forcing(1e-2, Some((1e-1, 0.0)), 0.0)?
            .ok_or(Error::Krylov { what: "no forcing" })?;
        assert_eq!(
            zero_floor.to_bits(),
            (0.9f64 * (1e-2f64 / 1e-1).powf(2.0)).to_bits()
        );
        Ok(())
    }

    #[test]
    fn stagnated_returns_the_earlier_best_iterate() -> Result<()> {
        // Cycle 1 reaches x = [1.0, 0.0] with residual 1.0 < sqrt(2).
        // Cycles 2 and 3 move x away along [0.0, 1.0] while the operator
        // produces a worse residual (4.0), triggering Stagnated and returning
        // the cycle-1 best iterate rather than x0 or the final iterate.
        let b = Array1::from_vec(vec![1.0, 1.0]);
        let mut apply_calls = 0usize;
        let mut pre_calls = 0usize;
        let mut steps = Vec::new();
        let out = fgmres(
            |v| {
                apply_calls += 1;
                Ok::<_, Error>(if apply_calls <= 3 || apply_calls == 5 {
                    v.to_owned()
                } else {
                    Array1::from_vec(vec![1.0, 5.0])
                })
            },
            |_| {
                pre_calls += 1;
                Ok(if pre_calls == 1 {
                    Array1::from_vec(vec![1.0, 0.0])
                } else {
                    Array1::from_vec(vec![0.0, 1.0])
                })
            },
            b.view(),
            None,
            &Fgmres::default()
                .with_rtol(1e-12)
                .with_restart(1)
                .with_max_products(12)
                .with_stagnation(NonZeroUsize::new(2)),
            |s| steps.push(*s),
        )?;
        assert_eq!(out.status, FgmresStatus::Stagnated);
        assert_eq!(out.cycles, 3);
        assert!((out.x[0] - 1.0).abs() < 1e-14 && out.x[1] == 0.0);
        assert!((out.residual - 1.0).abs() < 1e-14);
        let exact_residuals: Vec<f64> = steps
            .iter()
            .filter(|s| s.exact)
            .map(|s| s.residual)
            .collect();
        assert_eq!(exact_residuals.len(), 4);
        assert_eq!(out.residual, exact_residuals[1]);
        assert!(exact_residuals[2] > exact_residuals[1]);
        assert!(exact_residuals[3] > exact_residuals[1]);
        Ok(())
    }
}
