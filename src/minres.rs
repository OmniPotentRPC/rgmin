//! Matrix-free preconditioned MINRES with a finite iteration bound and
//! true-residual certification.
//!
//! For a symmetric (possibly indefinite) linear operator $J$ and split
//! preconditioner factors $(L^T, L)$ such that the preconditioned operator
//! $L^T J L$ is symmetric, [`minres_solve_bounded`] runs the Paige-Saunders
//! Lanczos/Givens recurrence on
//!
//! $$
//! (L^T J L)\, y = L^T b, \qquad x = L y,
//! $$
//!
//! up to `iteration_bound` iterations and certifies every candidate solution
//! by recomputing the true preconditioned residual $L^T b - L^T J (L y)$ before
//! accepting. Inner products reuse [`crate::vecops::dot_ordered`].

use std::error::Error as StdError;
use std::fmt;

use ndarray::{Array1, ArrayView1};

use crate::hvp::Preconditioner;
use crate::vecops::dot_ordered;

/// Outcome of a bounded MINRES solve with its verified true
/// preconditioned relative residual $\|L^T b - L^T J L y\|_2 / \|L^T b\|_2$
/// (equivalently $\|b - J x\|_{P^{-1}} / \|b\|_{P^{-1}}$ when $P^{-1} = L L^T$).
#[derive(Clone, Debug, PartialEq)]
pub struct BoundedKrylovSolve {
    /// Approximate physical solution $x = L y$.
    pub x: Vec<f64>,
    /// Krylov iterations performed.
    pub iterations: usize,
    /// Maximum Krylov iterations permitted by the caller's bound.
    pub iteration_bound: usize,
    /// Verified true preconditioned relative residual
    /// $\|L^T b - L^T J L y\|_2 / \|L^T b\|_2 = \|b - J x\|_{P^{-1}} / \|b\|_{P^{-1}}$.
    pub relative_residual: f64,
}

/// Vector source inspected by [`minres_solve_bounded`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MinresVector {
    /// Right-hand side `rhs`.
    RightHandSide,
    /// Left-preconditioned vector `apply_left(...)`.
    LeftPreconditioner,
    /// Right-preconditioned vector `apply_right(...)`.
    RightPreconditioner,
    /// Operator product `matvec(...)`.
    OperatorProduct,
}

impl fmt::Display for MinresVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::RightHandSide => "right-hand side",
            Self::LeftPreconditioner => "left preconditioner product",
            Self::RightPreconditioner => "right preconditioner product",
            Self::OperatorProduct => "operator product",
        };
        f.write_str(label)
    }
}

/// Typed failure of [`minres_solve_bounded`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum MinresError<E> {
    /// The operator callback returned an error.
    Operator(E),
    /// `rhs` is empty.
    EmptySystem,
    /// `rel_tol` is non-positive or non-finite.
    InvalidTolerance {
        /// Refused relative tolerance.
        rel_tol: f64,
    },
    /// `iteration_bound` is zero.
    ZeroIterationBound,
    /// `iteration_bound` or workspace allocation exceeded `Vec` capacity.
    CapacityOverflow {
        /// Requested iteration cap.
        iteration_bound: usize,
    },
    /// A callback (`apply_left`, `apply_right`, or `matvec`) returned a vector
    /// whose length does not match `rhs.len()`.
    DimensionMismatch {
        /// Which callback produced the mismatched vector.
        vector: MinresVector,
        /// Expected length `rhs.len()`.
        expected: usize,
        /// Returned length.
        got: usize,
    },
    /// `rhs` or a callback result contained a non-finite entry.
    NonFinite {
        /// Source of the non-finite vector.
        vector: MinresVector,
    },
    /// Internal norm, Lanczos, Givens, or certification arithmetic overflowed
    /// or produced a non-finite scalar despite finite callback vectors.
    NonFiniteArithmetic,
    /// A nonzero vector mapped to zero under `apply_left`, or `precond.solve`
    /// produced a non-positive inner product $\langle r, P^{-1} r \rangle \le 0$
    /// (singular or non-positive-definite preconditioner).
    SingularPreconditioner,
    /// Lanczos/Givens breakdown before meeting the true preconditioned residual tolerance.
    Breakdown {
        /// Iteration where breakdown occurred.
        iterations: usize,
        /// True preconditioned relative residual at breakdown.
        relative_residual: f64,
    },
    /// Numerically invariant subspace reached without meeting the true
    /// preconditioned residual tolerance.
    InvariantSubspace {
        /// Iteration where the scaled relative threshold detected invariance.
        iterations: usize,
        /// True preconditioned relative residual at termination.
        relative_residual: f64,
    },
    /// `iteration_bound` exhausted without meeting `rel_tol`.
    IterationCapExceeded {
        /// Iteration cap that was exhausted.
        iteration_bound: usize,
    },
}

impl<E> MinresError<E> {
    fn with_iterate(
        self,
        x: Vec<f64>,
        iterations: usize,
        iteration_bound: usize,
        relative_residual: f64,
    ) -> MinresFailure<E> {
        MinresFailure::new(self, x, iterations, iteration_bound, relative_residual)
    }
}

/// A failed bounded solve together with the last finite iterate retained by
/// the solver.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct MinresFailure<E> {
    /// Reason the solve stopped.
    pub error: MinresError<E>,
    /// Last finite iterate and its residual estimate or certification.
    pub last_iterate: BoundedKrylovSolve,
}

impl<E> MinresFailure<E> {
    fn new(
        error: MinresError<E>,
        x: Vec<f64>,
        iterations: usize,
        iteration_bound: usize,
        relative_residual: f64,
    ) -> Self {
        debug_assert!(relative_residual.is_finite());
        Self {
            error,
            last_iterate: BoundedKrylovSolve {
                x,
                iterations,
                iteration_bound,
                relative_residual,
            },
        }
    }

    fn fill_initial_state(mut self, n: usize, iteration_bound: usize) -> Self {
        if self.last_iterate.x.is_empty() && n != 0 {
            self.last_iterate.x.resize(n, 0.0);
        }
        if self.last_iterate.iteration_bound == 0 {
            self.last_iterate.iteration_bound = iteration_bound;
        }
        self
    }
}

impl<E> From<MinresError<E>> for MinresFailure<E> {
    fn from(error: MinresError<E>) -> Self {
        Self::new(error, Vec::new(), 0, 0, 1.0)
    }
}

impl<E: PartialEq> PartialEq<MinresError<E>> for MinresFailure<E> {
    fn eq(&self, other: &MinresError<E>) -> bool {
        &self.error == other
    }
}

impl<E: fmt::Display> fmt::Display for MinresError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operator(err) => write!(f, "{err}"),
            Self::EmptySystem => f.write_str("MINRES was given an empty system"),
            Self::InvalidTolerance { rel_tol } => {
                write!(
                    f,
                    "MINRES relative tolerance must be positive and finite, got {rel_tol}"
                )
            }
            Self::ZeroIterationBound => f.write_str("MINRES iteration bound must be at least 1"),
            Self::CapacityOverflow { iteration_bound } => {
                write!(
                    f,
                    "MINRES workspace reservation overflowed capacity for iteration bound {iteration_bound}"
                )
            }
            Self::DimensionMismatch {
                vector,
                expected,
                got,
            } => write!(
                f,
                "MINRES {vector} length {got} does not match the system dimension {expected}"
            ),
            Self::NonFinite { vector } => {
                write!(f, "MINRES {vector} contains a non-finite entry")
            }
            Self::NonFiniteArithmetic => {
                f.write_str("MINRES recurrence or residual arithmetic produced a non-finite scalar")
            }
            Self::SingularPreconditioner => f.write_str(
                "MINRES preconditioner is singular or not positive definite on a nonzero residual",
            ),
            Self::Breakdown {
                iterations,
                relative_residual,
            } => write!(
                f,
                "MINRES broke down at iteration {iterations} with true relative residual {relative_residual:.3e}"
            ),
            Self::InvariantSubspace {
                iterations,
                relative_residual,
            } => write!(
                f,
                "MINRES invariant subspace stopped at iteration {iterations} with true relative residual {relative_residual:.3e}"
            ),
            Self::IterationCapExceeded { iteration_bound } => {
                write!(f, "MINRES exceeded the iteration cap {iteration_bound}")
            }
        }
    }
}

impl<E: fmt::Display> fmt::Display for MinresFailure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl<E: StdError + 'static> StdError for MinresError<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Operator(err) => Some(err),
            _ => None,
        }
    }
}

impl<E: StdError + 'static> StdError for MinresFailure<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.error.source()
    }
}

/// Preconditioned MINRES for a symmetric (possibly indefinite) operator $J$,
/// using split factors $(L^T, L)$ so the preconditioned operator is the
/// symmetric matrix $L^T J L$, verifying the true preconditioned residual
/// $L^T b - L^T J L y$ before accepting.
pub fn minres_solve_bounded<F, PL, PR, E>(
    matvec: F,
    rhs: &[f64],
    rel_tol: f64,
    apply_left: PL,
    apply_right: PR,
    iteration_bound: usize,
) -> Result<BoundedKrylovSolve, MinresFailure<E>>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, E>,
    PL: FnMut(&[f64]) -> Vec<f64>,
    PR: FnMut(&[f64]) -> Vec<f64>,
{
    minres_solve_bounded_impl(
        matvec,
        rhs,
        rel_tol,
        apply_left,
        apply_right,
        iteration_bound,
    )
    .map_err(|failure| failure.fill_initial_state(rhs.len(), iteration_bound))
}

fn minres_solve_bounded_impl<F, PL, PR, E>(
    mut matvec: F,
    rhs: &[f64],
    rel_tol: f64,
    mut apply_left: PL,
    mut apply_right: PR,
    iteration_bound: usize,
) -> Result<BoundedKrylovSolve, MinresFailure<E>>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, E>,
    PL: FnMut(&[f64]) -> Vec<f64>,
    PR: FnMut(&[f64]) -> Vec<f64>,
{
    let n = rhs.len();
    if n == 0 {
        return Err(MinresError::EmptySystem.with_iterate(Vec::new(), 0, iteration_bound, 0.0));
    }
    if !(rel_tol > 0.0 && rel_tol.is_finite()) {
        return Err(MinresError::InvalidTolerance { rel_tol }.with_iterate(
            vec![0.0; n],
            0,
            iteration_bound,
            1.0,
        ));
    }
    if iteration_bound == 0 {
        return Err(MinresError::ZeroIterationBound.with_iterate(
            vec![0.0; n],
            0,
            iteration_bound,
            1.0,
        ));
    }
    check_finite(rhs, MinresVector::RightHandSide)?;

    let rhs_norm = ScaledNorm::from_slice(rhs)?;
    if rhs_norm.is_zero() {
        return Ok(BoundedKrylovSolve {
            x: try_zeros(n, iteration_bound)?,
            iterations: 0,
            iteration_bound,
            relative_residual: 0.0,
        });
    }
    let rhs_scaled = checked_vec(apply_left(rhs), n, MinresVector::LeftPreconditioner)?;
    let b_scaled_norm = ScaledNorm::from_slice(&rhs_scaled)?;
    if b_scaled_norm.is_zero() {
        return Err(MinresError::SingularPreconditioner.with_iterate(
            vec![0.0; n],
            0,
            iteration_bound,
            1.0,
        ));
    }

    let bnorm = b_scaled_norm.value()?;
    let mut x = try_zeros(n, iteration_bound)?;
    let mut beta = bnorm;
    let mut v_prev = try_zeros(n, iteration_bound)?;
    let mut v = try_zeros(n, iteration_bound)?;
    normalize_slice(&rhs_scaled, b_scaled_norm, bnorm, &mut v)?;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    basis
        .try_reserve_exact(iteration_bound)
        .map_err(|_| MinresError::CapacityOverflow { iteration_bound })?;
    let mut w_prev2 = try_zeros(n, iteration_bound)?;
    let mut w_prev = try_zeros(n, iteration_bound)?;
    let mut cs = -1.0;
    let mut sn = 0.0;
    let mut eps_rot = 0.0;
    let mut dbar = 0.0;
    let mut phibar = beta;

    for iter in 1..=iteration_bound {
        let physical = checked_vec(apply_right(&v), n, MinresVector::RightPreconditioner)?;
        let raw = matvec(&physical).map_err(MinresError::Operator)?;
        let raw = checked_vec(raw, n, MinresVector::OperatorProduct)?;
        let p = checked_vec(apply_left(&raw), n, MinresVector::LeftPreconditioner)?;

        let alpha = dot(&v, &p);
        if !alpha.is_finite() {
            return Err(MinresError::NonFiniteArithmetic.into());
        }
        let mut lanczos = p;
        for i in 0..n {
            lanczos[i] -= alpha * v[i] + beta * v_prev[i];
            if !lanczos[i].is_finite() {
                return Err(MinresError::NonFiniteArithmetic.into());
            }
        }
        let mut v_saved = Vec::new();
        v_saved
            .try_reserve_exact(n)
            .map_err(|_| MinresError::CapacityOverflow { iteration_bound })?;
        v_saved.extend_from_slice(&v);
        basis.push(v_saved);
        for _ in 0..2 {
            for q in &basis {
                let coeff = dot(q, &lanczos);
                if !coeff.is_finite() {
                    return Err(MinresError::NonFiniteArithmetic.into());
                }
                for (slot, &q_i) in lanczos.iter_mut().zip(q) {
                    let next = *slot - coeff * q_i;
                    if !next.is_finite() {
                        return Err(MinresError::NonFiniteArithmetic.into());
                    }
                    *slot = next;
                }
            }
        }

        let beta_next_scaled = ScaledNorm::from_slice(&lanczos)?;
        let beta_next = if beta_next_scaled.is_zero() {
            0.0
        } else {
            beta_next_scaled.value()?
        };
        let old_eps = eps_rot;
        let old_dbar = dbar;
        let delta = cs * dbar + sn * alpha;
        let gbar = sn * dbar - cs * alpha;
        eps_rot = sn * beta_next;
        dbar = -cs * beta_next;
        let gamma_sq = gbar * gbar + beta_next * beta_next;
        let gamma = if gamma_sq.is_finite() && gamma_sq >= f64::MIN_POSITIVE {
            gamma_sq.sqrt()
        } else {
            let g_norm = ScaledNorm::from_slice(&[gbar, beta_next])?;
            if g_norm.is_zero() {
                0.0
            } else {
                g_norm.value()?
            }
        };
        if !(delta.is_finite()
            && gbar.is_finite()
            && eps_rot.is_finite()
            && dbar.is_finite()
            && gamma.is_finite())
        {
            return Err(MinresError::NonFiniteArithmetic.into());
        }
        let breakdown_scale = alpha
            .abs()
            .max(if iter == 1 { 0.0 } else { beta.abs() })
            .max(beta_next.abs())
            .max(old_dbar.abs())
            .max(old_eps.abs());
        if relative_breakdown(gamma, breakdown_scale) {
            let (candidate, true_rel_res) = certify(
                &mut matvec,
                &mut apply_left,
                &mut apply_right,
                &x,
                &rhs_scaled,
                b_scaled_norm,
                n,
            )?;
            if true_rel_res <= rel_tol {
                return Ok(BoundedKrylovSolve {
                    x: candidate,
                    iterations: iter,
                    iteration_bound,
                    relative_residual: true_rel_res,
                });
            }
            return Err(MinresError::Breakdown {
                iterations: iter,
                relative_residual: true_rel_res,
            }
            .with_iterate(candidate, iter, iteration_bound, true_rel_res));
        }

        cs = gbar / gamma;
        sn = beta_next / gamma;
        let phi = cs * phibar;
        phibar *= sn;
        if !(cs.is_finite() && sn.is_finite() && phi.is_finite() && phibar.is_finite()) {
            return Err(MinresError::NonFiniteArithmetic.into());
        }
        let mut w = try_zeros(n, iteration_bound)?;
        for i in 0..n {
            let wi = (v[i] - old_eps * w_prev2[i] - delta * w_prev[i]) / gamma;
            let xi = x[i] + phi * wi;
            if !(wi.is_finite() && xi.is_finite()) {
                return Err(MinresError::NonFiniteArithmetic.into());
            }
            w[i] = wi;
            x[i] = xi;
        }

        let rel_res = phibar.abs() / bnorm;
        if !rel_res.is_finite() {
            return Err(MinresError::NonFiniteArithmetic.into());
        }
        let invariant_subspace = relative_breakdown(beta_next, breakdown_scale);
        if rel_res <= rel_tol || invariant_subspace {
            let (candidate, true_rel_res) = certify(
                &mut matvec,
                &mut apply_left,
                &mut apply_right,
                &x,
                &rhs_scaled,
                b_scaled_norm,
                n,
            )?;
            if true_rel_res <= rel_tol {
                return Ok(BoundedKrylovSolve {
                    x: candidate,
                    iterations: iter,
                    iteration_bound,
                    relative_residual: true_rel_res,
                });
            }
            if invariant_subspace {
                return Err(MinresError::InvariantSubspace {
                    iterations: iter,
                    relative_residual: true_rel_res,
                }
                .with_iterate(candidate, iter, iteration_bound, true_rel_res));
            }
        }

        w_prev2.clone_from(&w_prev);
        w_prev.clone_from(&w);
        v_prev.clone_from(&v);
        normalize_slice(&lanczos, beta_next_scaled, beta_next, &mut v)?;
        beta = beta_next;
    }

    let (candidate, true_rel_res) = certify(
        &mut matvec,
        &mut apply_left,
        &mut apply_right,
        &x,
        &rhs_scaled,
        b_scaled_norm,
        n,
    )?;
    Err(
        MinresError::IterationCapExceeded { iteration_bound }.with_iterate(
            candidate,
            iteration_bound,
            iteration_bound,
            true_rel_res,
        ),
    )
}

/// Preconditioned MINRES using a symmetric positive definite [`Preconditioner`]
/// whose [`Preconditioner::solve`] applies the full inverse $P^{-1} r$.
///
/// If $P^{-1} = L L^T$, this runs Paige-Saunders MINRES in the $P^{-1}$-inner
/// product without forming $L$, generating the exact same iterates $x_k = L y_k$
/// and certified relative residual
/// $\|b - J x_k\|_{P^{-1}} / \|b\|_{P^{-1}} = \|L^T b - L^T J L y_k\|_2 / \|L^T b\|_2$
/// as [`minres_solve_bounded`] with split factors $(L^T, L)$ while invoking
/// `precond.solve` once per Krylov step.
pub fn minres_solve_precond<F, P, E>(
    matvec: F,
    rhs: &[f64],
    rel_tol: f64,
    precond: &P,
    iteration_bound: usize,
) -> Result<BoundedKrylovSolve, MinresFailure<E>>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, E>,
    P: Preconditioner + ?Sized,
{
    minres_solve_precond_impl(matvec, rhs, rel_tol, precond, iteration_bound)
        .map_err(|failure| failure.fill_initial_state(rhs.len(), iteration_bound))
}

fn minres_solve_precond_impl<F, P, E>(
    mut matvec: F,
    rhs: &[f64],
    rel_tol: f64,
    precond: &P,
    iteration_bound: usize,
) -> Result<BoundedKrylovSolve, MinresFailure<E>>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, E>,
    P: Preconditioner + ?Sized,
{
    let n = rhs.len();
    if n == 0 {
        return Err(MinresError::EmptySystem.with_iterate(Vec::new(), 0, iteration_bound, 0.0));
    }
    if !(rel_tol > 0.0 && rel_tol.is_finite()) {
        return Err(MinresError::InvalidTolerance { rel_tol }.with_iterate(
            vec![0.0; n],
            0,
            iteration_bound,
            1.0,
        ));
    }
    if iteration_bound == 0 {
        return Err(MinresError::ZeroIterationBound.with_iterate(
            vec![0.0; n],
            0,
            iteration_bound,
            1.0,
        ));
    }
    check_finite(rhs, MinresVector::RightHandSide)?;

    let b_raw_norm = ScaledNorm::from_slice(rhs)?;
    if b_raw_norm.is_zero() {
        return Ok(BoundedKrylovSolve {
            x: try_zeros(n, iteration_bound)?,
            iterations: 0,
            iteration_bound,
            relative_residual: 0.0,
        });
    }
    let z0 = checked_vec(
        array_to_vec(precond.solve(ArrayView1::from(rhs)), iteration_bound)?,
        n,
        MinresVector::LeftPreconditioner,
    )?;
    let b_pnorm = ScaledNorm::from_precond_pair(rhs, &z0)?;
    if b_pnorm.is_zero() {
        return Err(MinresError::SingularPreconditioner.with_iterate(
            vec![0.0; n],
            0,
            iteration_bound,
            1.0,
        ));
    }
    let bnorm = b_pnorm.value()?;

    let mut r = try_zeros(n, iteration_bound)?;
    let mut z = try_zeros(n, iteration_bound)?;
    normalize_slice(rhs, b_pnorm, bnorm, &mut r)?;
    normalize_slice(&z0, b_pnorm, bnorm, &mut z)?;
    let mut r_prev = try_zeros(n, iteration_bound)?;
    let mut x = try_zeros(n, iteration_bound)?;
    let mut w = try_zeros(n, iteration_bound)?;
    let mut w_prev = try_zeros(n, iteration_bound)?;
    let mut w_prev2 = try_zeros(n, iteration_bound)?;
    let mut basis_r: Vec<Vec<f64>> = Vec::new();
    basis_r
        .try_reserve_exact(iteration_bound)
        .map_err(|_| MinresError::CapacityOverflow { iteration_bound })?;
    let mut basis_z: Vec<Vec<f64>> = Vec::new();
    basis_z
        .try_reserve_exact(iteration_bound)
        .map_err(|_| MinresError::CapacityOverflow { iteration_bound })?;

    let mut beta = 0.0_f64;
    let mut dbar = 0.0_f64;
    let mut eps_rot = 0.0_f64;
    let mut cs = -1.0_f64;
    let mut sn = 0.0_f64;
    let mut phibar = bnorm;

    for iter in 1..=iteration_bound {
        let q_raw = matvec(&z).map_err(MinresError::Operator)?;
        let q_raw = checked_vec(q_raw, n, MinresVector::OperatorProduct)?;
        let alpha = dot(&z, &q_raw);
        if !alpha.is_finite() {
            return Err(MinresError::NonFiniteArithmetic.into());
        }

        let mut lanczos_r = try_zeros(n, iteration_bound)?;
        for i in 0..n {
            let val = q_raw[i] - alpha * r[i] - beta * r_prev[i];
            if !val.is_finite() {
                return Err(MinresError::NonFiniteArithmetic.into());
            }
            lanczos_r[i] = val;
        }
        let mut r_saved = Vec::new();
        r_saved
            .try_reserve_exact(n)
            .map_err(|_| MinresError::CapacityOverflow { iteration_bound })?;
        r_saved.extend_from_slice(&r);
        basis_r.push(r_saved);
        let mut z_saved = Vec::new();
        z_saved
            .try_reserve_exact(n)
            .map_err(|_| MinresError::CapacityOverflow { iteration_bound })?;
        z_saved.extend_from_slice(&z);
        basis_z.push(z_saved);

        for _ in 0..2 {
            for (q_r, q_z) in basis_r.iter().zip(basis_z.iter()) {
                let coeff = dot(&lanczos_r, q_z);
                if !coeff.is_finite() {
                    return Err(MinresError::NonFiniteArithmetic.into());
                }
                for (slot, &qr_i) in lanczos_r.iter_mut().zip(q_r) {
                    let next = *slot - coeff * qr_i;
                    if !next.is_finite() {
                        return Err(MinresError::NonFiniteArithmetic.into());
                    }
                    *slot = next;
                }
            }
        }

        let lanczos_z = checked_vec(
            array_to_vec(precond.solve(ArrayView1::from(&lanczos_r)), iteration_bound)?,
            n,
            MinresVector::LeftPreconditioner,
        )?;
        let beta_next_scaled = ScaledNorm::from_precond_pair(&lanczos_r, &lanczos_z)?;
        let beta_next = beta_next_scaled.value()?;

        let old_eps = eps_rot;
        let old_dbar = dbar;
        let delta = cs * dbar + sn * alpha;
        let gbar = sn * dbar - cs * alpha;
        eps_rot = sn * beta_next;
        dbar = -cs * beta_next;
        if !(delta.is_finite() && gbar.is_finite() && eps_rot.is_finite() && dbar.is_finite()) {
            return Err(MinresError::NonFiniteArithmetic.into());
        }
        let gamma = gbar.hypot(beta_next);
        if !gamma.is_finite() {
            return Err(MinresError::NonFiniteArithmetic.into());
        }
        let breakdown_scale = alpha
            .abs()
            .max(beta.abs())
            .max(beta_next.abs())
            .max(old_dbar.abs())
            .max(old_eps.abs());
        if relative_breakdown(gamma, breakdown_scale) {
            let true_rel_res =
                certify_precond(&mut matvec, precond, &x, rhs, b_pnorm, n, iteration_bound)?;
            if true_rel_res <= rel_tol {
                return Ok(BoundedKrylovSolve {
                    x,
                    iterations: iter,
                    iteration_bound,
                    relative_residual: true_rel_res,
                });
            }
            return Err(MinresError::Breakdown {
                iterations: iter,
                relative_residual: true_rel_res,
            }
            .with_iterate(x, iter, iteration_bound, true_rel_res));
        }

        cs = gbar / gamma;
        sn = beta_next / gamma;
        let phi = cs * phibar;
        phibar *= sn;
        if !(cs.is_finite() && sn.is_finite() && phi.is_finite() && phibar.is_finite()) {
            return Err(MinresError::NonFiniteArithmetic.into());
        }

        for i in 0..n {
            let w_val = (z[i] - old_eps * w_prev2[i] - delta * w_prev[i]) / gamma;
            let x_val = x[i] + phi * w_val;
            if !(w_val.is_finite() && x_val.is_finite()) {
                return Err(MinresError::NonFiniteArithmetic.into());
            }
            w[i] = w_val;
            x[i] = x_val;
        }

        let rel_res = phibar.abs() / bnorm;
        if !rel_res.is_finite() {
            return Err(MinresError::NonFiniteArithmetic.into());
        }
        let invariant_subspace = relative_breakdown(beta_next, breakdown_scale);
        if rel_res <= rel_tol || invariant_subspace {
            let true_rel_res =
                certify_precond(&mut matvec, precond, &x, rhs, b_pnorm, n, iteration_bound)?;
            if true_rel_res <= rel_tol {
                return Ok(BoundedKrylovSolve {
                    x,
                    iterations: iter,
                    iteration_bound,
                    relative_residual: true_rel_res,
                });
            }
            if invariant_subspace {
                return Err(MinresError::InvariantSubspace {
                    iterations: iter,
                    relative_residual: true_rel_res,
                }
                .with_iterate(x, iter, iteration_bound, true_rel_res));
            }
        }

        w_prev2.clone_from(&w_prev);
        w_prev.clone_from(&w);
        r_prev.clone_from(&r);
        normalize_slice(&lanczos_r, beta_next_scaled, beta_next, &mut r)?;
        normalize_slice(&lanczos_z, beta_next_scaled, beta_next, &mut z)?;
        beta = beta_next;
    }

    let true_rel_res = certify_precond(&mut matvec, precond, &x, rhs, b_pnorm, n, iteration_bound)?;
    Err(
        MinresError::IterationCapExceeded { iteration_bound }.with_iterate(
            x,
            iteration_bound,
            iteration_bound,
            true_rel_res,
        ),
    )
}

fn array_to_vec<E>(arr: Array1<f64>, iteration_bound: usize) -> Result<Vec<f64>, MinresError<E>> {
    let mut out = Vec::new();
    out.try_reserve_exact(arr.len())
        .map_err(|_| MinresError::CapacityOverflow { iteration_bound })?;
    out.extend(arr.iter().copied());
    Ok(out)
}

fn try_zeros<E>(len: usize, iteration_bound: usize) -> Result<Vec<f64>, MinresError<E>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| MinresError::CapacityOverflow { iteration_bound })?;
    out.resize(len, 0.0);
    Ok(out)
}

/// Overflow- and underflow-safe Euclidean norm represented as `scale * sqrt(ssq)`.
#[derive(Clone, Copy, Debug)]
struct ScaledNorm {
    scale: f64,
    ssq: f64,
}

impl ScaledNorm {
    fn from_slice<E>(slice: &[f64]) -> Result<Self, MinresError<E>> {
        let mut scale = 0.0;
        let mut ssq = 1.0;
        for &val in slice {
            if !val.is_finite() {
                return Err(MinresError::NonFiniteArithmetic);
            }
            let abs = val.abs();
            if abs == 0.0 {
                continue;
            }
            if scale < abs {
                let ratio = scale / abs;
                ssq = 1.0 + ssq * (ratio * ratio);
                scale = abs;
            } else {
                let ratio = abs / scale;
                ssq += ratio * ratio;
            }
            if !(scale.is_finite() && ssq.is_finite()) {
                return Err(MinresError::NonFiniteArithmetic);
            }
        }
        Ok(Self { scale, ssq })
    }

    fn from_diff<E>(want: &[f64], have: &[f64]) -> Result<Self, MinresError<E>> {
        let mut scale = 0.0;
        let mut ssq = 1.0;
        for (&w, &h) in want.iter().zip(have) {
            let diff = w - h;
            if !diff.is_finite() {
                return Err(MinresError::NonFiniteArithmetic);
            }
            let abs = diff.abs();
            if abs == 0.0 {
                continue;
            }
            if scale < abs {
                let ratio = scale / abs;
                ssq = 1.0 + ssq * (ratio * ratio);
                scale = abs;
            } else {
                let ratio = abs / scale;
                ssq += ratio * ratio;
            }
            if !(scale.is_finite() && ssq.is_finite()) {
                return Err(MinresError::NonFiniteArithmetic);
            }
        }
        Ok(Self { scale, ssq })
    }

    fn from_precond_pair<E>(r: &[f64], z: &[f64]) -> Result<Self, MinresError<E>> {
        let mut scale_r = 0.0_f64;
        for &val in r {
            if !val.is_finite() {
                return Err(MinresError::NonFiniteArithmetic);
            }
            let abs = val.abs();
            if abs > scale_r {
                scale_r = abs;
            }
        }
        let mut scale_z = 0.0_f64;
        for &val in z {
            if !val.is_finite() {
                return Err(MinresError::NonFiniteArithmetic);
            }
            let abs = val.abs();
            if abs > scale_z {
                scale_z = abs;
            }
        }
        if scale_r == 0.0 {
            return Ok(Self {
                scale: 0.0,
                ssq: 1.0,
            });
        }
        if scale_z == 0.0 {
            return Err(MinresError::SingularPreconditioner);
        }
        let mut ssq = 0.0_f64;
        for (&r_i, &z_i) in r.iter().zip(z) {
            let nr = r_i / scale_r;
            let nz = z_i / scale_z;
            let prod = nr * nz;
            ssq += prod;
            if !(nr.is_finite() && nz.is_finite() && prod.is_finite() && ssq.is_finite()) {
                return Err(MinresError::NonFiniteArithmetic);
            }
        }
        if ssq <= 0.0 {
            return Err(MinresError::SingularPreconditioner);
        }
        let scale = scale_r.sqrt() * scale_z.sqrt();
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(MinresError::NonFiniteArithmetic);
        }
        Ok(Self { scale, ssq })
    }

    fn is_zero(self) -> bool {
        self.scale == 0.0
    }

    fn value<E>(self) -> Result<f64, MinresError<E>> {
        if self.scale == 0.0 {
            return Ok(0.0);
        }
        let val = self.scale * self.ssq.sqrt();
        if !(val > 0.0 && val.is_finite()) {
            return Err(MinresError::NonFiniteArithmetic);
        }
        Ok(val)
    }

    fn ratio<E>(self, denom: Self) -> Result<f64, MinresError<E>> {
        if denom.scale == 0.0 {
            return Err(MinresError::NonFiniteArithmetic);
        }
        if self.scale == 0.0 {
            return Ok(0.0);
        }
        let root_ratio = (self.ssq / denom.ssq).sqrt();
        let ratio = (self.scale / denom.scale) * root_ratio;
        if !ratio.is_finite() {
            return Err(MinresError::NonFiniteArithmetic);
        }
        Ok(ratio)
    }
}

fn normalize_slice<E>(
    src: &[f64],
    scaled: ScaledNorm,
    norm_val: f64,
    dst: &mut [f64],
) -> Result<(), MinresError<E>> {
    if norm_val >= f64::MIN_POSITIVE {
        for (slot, &val) in dst.iter_mut().zip(src) {
            let unit = val / norm_val;
            if !unit.is_finite() {
                return Err(MinresError::NonFiniteArithmetic);
            }
            *slot = unit;
        }
        return Ok(());
    }
    let root = scaled.ssq.sqrt();
    if !(scaled.scale > 0.0 && scaled.scale.is_finite() && root > 0.0 && root.is_finite()) {
        return Err(MinresError::NonFiniteArithmetic);
    }
    for (slot, &val) in dst.iter_mut().zip(src) {
        let unit = (val / scaled.scale) / root;
        if !unit.is_finite() {
            return Err(MinresError::NonFiniteArithmetic);
        }
        *slot = unit;
    }
    Ok(())
}

fn certify<F, PL, PR, E>(
    matvec: &mut F,
    apply_left: &mut PL,
    apply_right: &mut PR,
    x: &[f64],
    rhs_scaled: &[f64],
    b_scaled_norm: ScaledNorm,
    n: usize,
) -> Result<(Vec<f64>, f64), MinresError<E>>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, E>,
    PL: FnMut(&[f64]) -> Vec<f64>,
    PR: FnMut(&[f64]) -> Vec<f64>,
{
    let candidate = checked_vec(apply_right(x), n, MinresVector::RightPreconditioner)?;
    let check_raw = matvec(&candidate).map_err(MinresError::Operator)?;
    let check_raw = checked_vec(check_raw, n, MinresVector::OperatorProduct)?;
    let check_scaled = checked_vec(apply_left(&check_raw), n, MinresVector::LeftPreconditioner)?;
    let res_norm = ScaledNorm::from_diff(rhs_scaled, &check_scaled)?;
    let true_rel_res = res_norm.ratio(b_scaled_norm)?;
    Ok((candidate, true_rel_res))
}

fn certify_precond<F, P, E>(
    matvec: &mut F,
    precond: &P,
    x: &[f64],
    rhs: &[f64],
    b_pnorm: ScaledNorm,
    n: usize,
    iteration_bound: usize,
) -> Result<f64, MinresError<E>>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, E>,
    P: Preconditioner + ?Sized,
{
    let ax = matvec(x).map_err(MinresError::Operator)?;
    let ax = checked_vec(ax, n, MinresVector::OperatorProduct)?;
    let mut r_true = Vec::new();
    r_true
        .try_reserve_exact(n)
        .map_err(|_| MinresError::CapacityOverflow { iteration_bound })?;
    for (&b_i, &ax_i) in rhs.iter().zip(&ax) {
        let diff = b_i - ax_i;
        if !diff.is_finite() {
            return Err(MinresError::NonFiniteArithmetic);
        }
        r_true.push(diff);
    }
    let z_true = checked_vec(
        array_to_vec(precond.solve(ArrayView1::from(&r_true)), iteration_bound)?,
        n,
        MinresVector::LeftPreconditioner,
    )?;
    let res_pnorm = ScaledNorm::from_precond_pair(&r_true, &z_true)?;
    res_pnorm.ratio(b_pnorm)
}

fn checked_vec<E>(
    vec: Vec<f64>,
    expected: usize,
    vector: MinresVector,
) -> Result<Vec<f64>, MinresError<E>> {
    if vec.len() != expected {
        return Err(MinresError::DimensionMismatch {
            vector,
            expected,
            got: vec.len(),
        });
    }
    check_finite(&vec, vector)?;
    Ok(vec)
}

fn check_finite<E>(slice: &[f64], vector: MinresVector) -> Result<(), MinresError<E>> {
    if slice.iter().any(|v| !v.is_finite()) {
        return Err(MinresError::NonFinite { vector });
    }
    Ok(())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    dot_ordered(ArrayView1::from(a), ArrayView1::from(b))
}

fn relative_breakdown(value: f64, scale: f64) -> bool {
    value <= 32.0 * f64::EPSILON * scale
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::convert::Infallible;

    use super::*;
    use crate::hvp::IdentityPrecond;

    #[test]
    fn solves_indefinite_system_and_certifies_true_residual() {
        let diag = [4.0, -3.0, 8.0];
        let rhs = [4.0, -6.0, 16.0];
        let solved = minres_solve_bounded(
            |v| Ok::<_, Infallible>(diag.iter().zip(v).map(|(d, x)| d * x).collect::<Vec<f64>>()),
            &rhs,
            1.0e-12,
            |r| r.to_vec(),
            |z| z.to_vec(),
            10,
        )
        .expect("indefinite MINRES solve");

        for (got, want) in solved.x.iter().zip([1.0, 2.0, 2.0]) {
            assert!((got - want).abs() <= 1.0e-12, "{got} != {want}");
        }
        assert!(solved.relative_residual <= 1.0e-12);
    }

    #[test]
    fn split_reorthogonalization_converges_on_ill_conditioned_spd_system() {
        let diag = [1.0, 10.0_f64.powf(1.5), 1.0e3, 10.0_f64.powf(4.5), 1.0e6];
        let rhs = [1.0; 5];
        let matvec =
            |v: &[f64]| Ok::<_, Infallible>(diag.iter().zip(v).map(|(d, x)| d * x).collect());

        let split = minres_solve_bounded(matvec, &rhs, 1.0e-10, |r| r.to_vec(), |z| z.to_vec(), 5)
            .expect("two-pass split MINRES must converge within the exact dimension");
        let precond = minres_solve_precond(matvec, &rhs, 1.0e-10, &IdentityPrecond, 5)
            .expect("preconditioned MINRES must converge within the exact dimension");

        assert_eq!(split.iterations, 5);
        assert_eq!(precond.iterations, 5);
        assert!(split.relative_residual <= 1.0e-10);
        assert!(precond.relative_residual <= 1.0e-10);
        assert!(split.relative_residual <= 2.0 * precond.relative_residual);
    }

    #[test]
    fn split_true_residual_tracks_preconditioned_path_at_high_condition_number() {
        for kappa in [1.0e8_f64, 1.0e9, 1.0e10] {
            let diag = [1.0, kappa.powf(0.25), kappa.sqrt(), kappa.powf(0.75), kappa];
            let rhs = [1.0; 5];
            let matvec =
                |v: &[f64]| Ok::<_, Infallible>(diag.iter().zip(v).map(|(d, x)| d * x).collect());

            let split =
                minres_solve_bounded(matvec, &rhs, 1.0e-5, |r| r.to_vec(), |z| z.to_vec(), 5)
                    .expect("split path must converge in the exact dimension");
            let precond = minres_solve_precond(matvec, &rhs, 1.0e-5, &IdentityPrecond, 5)
                .expect("preconditioned path must converge in the exact dimension");

            let true_residual = |x: &[f64]| {
                rhs.iter()
                    .zip(diag.iter().zip(x))
                    .map(|(b, (d, x_i))| (b - d * x_i).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / (rhs.len() as f64).sqrt()
            };
            let split_residual = true_residual(&split.x);
            let precond_residual = true_residual(&precond.x);
            assert!((split.relative_residual - split_residual).abs() <= 1.0e-15);
            assert!((precond.relative_residual - precond_residual).abs() <= 1.0e-15);
            assert!(
                split_residual <= 2.0 * precond_residual,
                "kappa={kappa:e}: split={split_residual:e}, precond={precond_residual:e}"
            );
        }
    }

    #[test]
    fn rotated_near_breakdown_does_not_divide_by_tiny_gamma() {
        let angle = 0.37_f64;
        let (sin, cos) = angle.sin_cos();
        let matrix = [[cos * cos, cos * sin], [cos * sin, sin * sin]];
        let rhs = [1.0, -1.0];
        let matvec = |v: &[f64]| {
            Ok::<_, Infallible>(vec![
                matrix[0][0] * v[0] + matrix[0][1] * v[1],
                matrix[1][0] * v[0] + matrix[1][1] * v[1],
            ])
        };

        let split = minres_solve_bounded(matvec, &rhs, 1.0e-12, |r| r.to_vec(), |z| z.to_vec(), 4)
            .expect_err("rotated singular split system must break down");
        let precond = minres_solve_precond(matvec, &rhs, 1.0e-12, &IdentityPrecond, 4)
            .expect_err("rotated singular preconditioned system must break down");

        for failure in [split, precond] {
            assert!(matches!(failure.error, MinresError::Breakdown { .. }));
            assert_eq!(failure.last_iterate.x.len(), 2);
            assert!(failure.last_iterate.x.iter().all(|value| value.is_finite()));
            assert!(failure.last_iterate.relative_residual.is_finite());
        }
    }

    #[test]
    fn preconditioner_is_applied_and_collapses_exact_spd_system_to_one_step() {
        struct CountingDiagonalPreconditioner<'a> {
            calls: &'a Cell<usize>,
        }
        impl Preconditioner for CountingDiagonalPreconditioner<'_> {
            fn solve(&self, r: ArrayView1<f64>) -> Array1<f64> {
                self.calls.set(self.calls.get() + 1);
                Array1::from_iter(
                    r.iter()
                        .zip([0.25, 1.0 / 9.0, 0.04])
                        .map(|(&value, inverse)| value * inverse),
                )
            }
        }

        let calls = Cell::new(0);
        let preconditioner = CountingDiagonalPreconditioner { calls: &calls };
        let diag = [4.0, 9.0, 25.0];
        let rhs = [8.0, -27.0, 50.0];
        let solved = minres_solve_precond(
            |v| Ok::<_, Infallible>(diag.iter().zip(v).map(|(d, x)| d * x).collect()),
            &rhs,
            1.0e-12,
            &preconditioner,
            3,
        )
        .expect("exact preconditioner");

        assert_eq!(solved.iterations, 1);
        assert!(calls.get() >= 3);
        for (&got, want) in solved.x.iter().zip([2.0, -3.0, 2.0]) {
            assert!((got - want).abs() <= 1.0e-14);
        }
    }

    #[test]
    fn split_factor_order_is_not_interchangeable() {
        let matrix = [[3.0, 1.0], [1.0, 2.0]];
        let rhs = [1.0, -3.0];
        let apply_l = |z: &[f64]| vec![2.0 * z[0], z[0] + z[1]];
        let apply_lt = |r: &[f64]| vec![2.0 * r[0] + r[1], r[1]];
        let matvec = |v: &[f64]| {
            Ok::<_, Infallible>(vec![
                matrix[0][0] * v[0] + matrix[0][1] * v[1],
                matrix[1][0] * v[0] + matrix[1][1] * v[1],
            ])
        };

        let solved = minres_solve_bounded(matvec, &rhs, 1.0e-12, apply_lt, apply_l, 2)
            .expect("ordered split factors");
        assert!((solved.x[0] - 1.0).abs() <= 1.0e-12);
        assert!((solved.x[1] + 2.0).abs() <= 1.0e-12);
        assert!(solved.relative_residual <= 1.0e-12);
    }

    #[test]
    fn three_dimensional_reorthogonalization_is_not_optional() {
        let q = [
            [2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
            [-2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
        ];
        let eigenvalues = [1.0, 1.0e4, 1.0e8];
        let mut matrix = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                matrix[i][j] = (0..3).map(|k| q[i][k] * eigenvalues[k] * q[j][k]).sum();
            }
        }
        let rhs = [1.0, -1.0, 2.0];
        let matvec = |v: &[f64]| {
            Ok::<_, Infallible>(
                (0..3)
                    .map(|i| (0..3).map(|j| matrix[i][j] * v[j]).sum())
                    .collect(),
            )
        };

        let split = minres_solve_bounded(matvec, &rhs, 1.0e-7, |r| r.to_vec(), |z| z.to_vec(), 3)
            .expect("three-dimensional split solve");
        let precond = minres_solve_precond(matvec, &rhs, 1.0e-7, &IdentityPrecond, 3)
            .expect("three-dimensional preconditioned solve");

        assert!(split.relative_residual <= 1.0e-7);
        assert!(precond.relative_residual <= 1.0e-7);
    }

    #[test]
    fn singular_operator_reports_breakdown_with_last_iterate_in_both_variants() {
        let rhs = [1.0, 1.0];
        let matvec = |v: &[f64]| Ok::<_, Infallible>(vec![v[0], 0.0]);

        let split = minres_solve_bounded(matvec, &rhs, 1.0e-12, |r| r.to_vec(), |z| z.to_vec(), 4)
            .expect_err("singular split system must break down");
        let precond = minres_solve_precond(matvec, &rhs, 1.0e-12, &IdentityPrecond, 4)
            .expect_err("singular preconditioned system must break down");

        for failure in [split, precond] {
            match failure.error {
                MinresError::Breakdown {
                    iterations,
                    relative_residual,
                } => {
                    assert_eq!(iterations, 2);
                    assert!((relative_residual - std::f64::consts::FRAC_1_SQRT_2).abs() <= 1.0e-15);
                    assert_eq!(failure.last_iterate.iterations, iterations);
                    assert_eq!(failure.last_iterate.relative_residual, relative_residual);
                    assert!(failure.last_iterate.x.iter().all(|x| x.is_finite()));
                }
                other => panic!("expected Breakdown, got {other:?}"),
            }
        }
    }

    #[test]
    fn true_residual_certification_rejects_false_recurrence_convergence() {
        let rhs = [1.0];
        let mut split_calls = 0;
        let split = minres_solve_bounded(
            |v| {
                split_calls += 1;
                if split_calls == 1 {
                    Ok::<_, Infallible>(v.to_vec())
                } else {
                    Ok(vec![0.0; v.len()])
                }
            },
            &rhs,
            1.0e-12,
            |r| r.to_vec(),
            |z| z.to_vec(),
            2,
        )
        .expect_err("true residual must reject the split recurrence estimate");
        assert!(split_calls >= 2);
        assert_eq!(
            split.error,
            MinresError::InvariantSubspace {
                iterations: 1,
                relative_residual: 1.0,
            }
        );

        let mut precond_calls = 0;
        let precond = minres_solve_precond(
            |v| {
                precond_calls += 1;
                if precond_calls == 1 {
                    Ok::<_, Infallible>(v.to_vec())
                } else {
                    Ok(vec![0.0; v.len()])
                }
            },
            &rhs,
            1.0e-12,
            &IdentityPrecond,
            2,
        )
        .expect_err("true residual must reject the preconditioned recurrence estimate");
        assert!(precond_calls >= 2);
        assert_eq!(
            precond.error,
            MinresError::InvariantSubspace {
                iterations: 1,
                relative_residual: 1.0,
            }
        );
    }

    #[test]
    fn invalid_tolerance_and_non_finite_arithmetic_retain_finite_state() {
        for rel_tol in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let split = minres_solve_bounded(
                |v| Ok::<_, Infallible>(v.to_vec()),
                &[1.0, 2.0],
                rel_tol,
                |r| r.to_vec(),
                |z| z.to_vec(),
                3,
            )
            .expect_err("invalid split tolerance");
            assert!(matches!(
                split.error,
                MinresError::InvalidTolerance { rel_tol: got }
                    if got.to_bits() == rel_tol.to_bits()
            ));
            assert_eq!(split.last_iterate.x, vec![0.0; 2]);
            assert_eq!(split.last_iterate.relative_residual, 1.0);

            let precond = minres_solve_precond(
                |v| Ok::<_, Infallible>(v.to_vec()),
                &[1.0, 2.0],
                rel_tol,
                &IdentityPrecond,
                3,
            )
            .expect_err("invalid preconditioned tolerance");
            assert!(matches!(
                precond.error,
                MinresError::InvalidTolerance { rel_tol: got }
                    if got.to_bits() == rel_tol.to_bits()
            ));
            assert_eq!(precond.last_iterate.x, vec![0.0; 2]);
            assert_eq!(precond.last_iterate.relative_residual, 1.0);
        }

        let precond_non_finite = minres_solve_precond(
            |_v| Ok::<_, Infallible>(vec![f64::MAX, f64::MAX]),
            &[3.0, 4.0],
            1.0e-12,
            &IdentityPrecond,
            3,
        )
        .expect_err("overflowing preconditioned alpha");
        assert_eq!(precond_non_finite.error, MinresError::NonFiniteArithmetic);
        assert_eq!(precond_non_finite.last_iterate.x, vec![0.0; 2]);
        assert!(
            precond_non_finite
                .last_iterate
                .x
                .iter()
                .all(|x| x.is_finite())
        );
        assert!(
            precond_non_finite
                .last_iterate
                .relative_residual
                .is_finite()
        );
    }

    #[test]
    fn input_and_callback_error_variants_retain_initial_state() {
        let empty = minres_solve_bounded(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &[],
            1.0e-12,
            |r| r.to_vec(),
            |z| z.to_vec(),
            1,
        )
        .expect_err("empty system");
        assert_eq!(empty.error, MinresError::EmptySystem);
        assert!(empty.last_iterate.x.is_empty());
        assert_eq!(empty.last_iterate.relative_residual, 0.0);

        let zero_bound = minres_solve_precond(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &[1.0],
            1.0e-12,
            &IdentityPrecond,
            0,
        )
        .expect_err("zero iteration bound");
        assert_eq!(zero_bound.error, MinresError::ZeroIterationBound);
        assert_eq!(zero_bound.last_iterate.x, vec![0.0]);

        let non_finite = minres_solve_bounded(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &[f64::NAN],
            1.0e-12,
            |r| r.to_vec(),
            |z| z.to_vec(),
            1,
        )
        .expect_err("non-finite right-hand side");
        assert_eq!(
            non_finite.error,
            MinresError::NonFinite {
                vector: MinresVector::RightHandSide,
            }
        );
        assert_eq!(non_finite.last_iterate.x, vec![0.0]);

        let operator = minres_solve_precond(
            |_v| Err::<Vec<f64>, _>("operator refused input"),
            &[1.0],
            1.0e-12,
            &IdentityPrecond,
            1,
        )
        .expect_err("operator callback error");
        assert_eq!(
            operator.error,
            MinresError::Operator("operator refused input")
        );
        assert_eq!(operator.last_iterate.x, vec![0.0]);
        assert_eq!(operator.last_iterate.relative_residual, 1.0);
    }

    #[test]
    fn iteration_cap_carries_the_certified_true_residual() {
        let diag = [1.0, 4.0, 16.0];
        let rhs = [1.0, 1.0, 1.0];
        let matvec =
            |v: &[f64]| Ok::<_, Infallible>(diag.iter().zip(v).map(|(d, x)| d * x).collect());

        let split = minres_solve_bounded(matvec, &rhs, 1.0e-15, |r| r.to_vec(), |z| z.to_vec(), 1)
            .expect_err("one iteration cannot solve a three-eigenvalue system");
        let precond = minres_solve_precond(matvec, &rhs, 1.0e-15, &IdentityPrecond, 1)
            .expect_err("one iteration cannot solve a three-eigenvalue system");

        for failure in [split, precond] {
            assert_eq!(
                failure.error,
                MinresError::IterationCapExceeded { iteration_bound: 1 }
            );
            assert_eq!(failure.last_iterate.iterations, 1);
            assert_eq!(failure.last_iterate.iteration_bound, 1);
            assert!(failure.last_iterate.relative_residual > 1.0e-15);
            assert!(failure.last_iterate.relative_residual < 1.0);
            assert_eq!(failure.last_iterate.x.len(), 3);
        }
    }

    #[test]
    fn rejects_wrong_length_preconditioner_and_operator_outputs_without_panic_or_false_ok() {
        let rhs = [1.0, -2.0, 3.0];

        // Empty left-preconditioner output on a nonzero RHS must return DimensionMismatch, never Ok.
        let err_empty_left = minres_solve_bounded(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &rhs,
            1.0e-12,
            |_r| Vec::new(),
            |z| z.to_vec(),
            10,
        )
        .expect_err("empty left preconditioner output must fail");
        assert_eq!(
            err_empty_left,
            MinresError::DimensionMismatch {
                vector: MinresVector::LeftPreconditioner,
                expected: 3,
                got: 0,
            }
        );

        // Short non-empty left-preconditioner output must return DimensionMismatch, never panic.
        let err_short_left = minres_solve_bounded(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &rhs,
            1.0e-12,
            |_r| vec![1.0, 2.0],
            |z| z.to_vec(),
            10,
        )
        .expect_err("short left preconditioner output must fail");
        assert_eq!(
            err_short_left,
            MinresError::DimensionMismatch {
                vector: MinresVector::LeftPreconditioner,
                expected: 3,
                got: 2,
            }
        );

        // Short right-preconditioner output must return DimensionMismatch, never panic.
        let err_short_right = minres_solve_bounded(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &rhs,
            1.0e-12,
            |r| r.to_vec(),
            |_z| vec![1.0],
            10,
        )
        .expect_err("short right preconditioner output must fail");
        assert_eq!(
            err_short_right,
            MinresError::DimensionMismatch {
                vector: MinresVector::RightPreconditioner,
                expected: 3,
                got: 1,
            }
        );

        // Wrong-length operator product must return DimensionMismatch.
        let err_op = minres_solve_bounded(
            |_v| Ok::<_, Infallible>(vec![1.0, 2.0]),
            &rhs,
            1.0e-12,
            |r| r.to_vec(),
            |z| z.to_vec(),
            10,
        )
        .expect_err("short operator output must fail");
        assert_eq!(
            err_op,
            MinresError::DimensionMismatch {
                vector: MinresVector::OperatorProduct,
                expected: 3,
                got: 2,
            }
        );

        // Zero-mapping left preconditioner on nonzero RHS must return SingularPreconditioner.
        let err_singular = minres_solve_bounded(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &rhs,
            1.0e-12,
            |_r| vec![0.0; 3],
            |z| z.to_vec(),
            10,
        )
        .expect_err("singular left preconditioner on nonzero RHS must fail");
        assert_eq!(err_singular, MinresError::SingularPreconditioner);
    }

    #[test]
    fn precond_trait_adapter_solves_and_handles_zero_rhs() {
        let zero = [0.0, 0.0, 0.0];
        let solved = minres_solve_precond(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &zero,
            1.0e-12,
            &IdentityPrecond,
            5,
        )
        .expect("zero rhs");
        assert_eq!(solved.iterations, 0);
        assert_eq!(solved.x, vec![0.0; 3]);

        // Zero RHS must return x = 0 before invoking the preconditioner or operator.
        struct PanicIfCalledPrecond;
        impl Preconditioner for PanicIfCalledPrecond {
            fn solve(&self, _r: ArrayView1<f64>) -> Array1<f64> {
                panic!("preconditioner must not be called when rhs is identically zero");
            }
        }
        let solved_zero_uncalled = minres_solve_precond(
            |_v| -> Result<Vec<f64>, Infallible> {
                panic!("operator must not be called when rhs is identically zero")
            },
            &zero,
            1.0e-12,
            &PanicIfCalledPrecond,
            5,
        )
        .expect("zero rhs must not invoke preconditioner");
        assert_eq!(solved_zero_uncalled.iterations, 0);
        assert_eq!(solved_zero_uncalled.x, vec![0.0; 3]);
        assert_eq!(solved_zero_uncalled.relative_residual, 0.0);

        let solved_zero_bounded = minres_solve_bounded(
            |_v| -> Result<Vec<f64>, Infallible> {
                panic!("operator must not be called when rhs is identically zero")
            },
            &zero,
            1.0e-12,
            |_r| panic!("left preconditioner must not be called when rhs is identically zero"),
            |_z| panic!("right preconditioner must not be called when rhs is identically zero"),
            5,
        )
        .expect("zero rhs bounded must not invoke preconditioner");
        assert_eq!(solved_zero_bounded.iterations, 0);
        assert_eq!(solved_zero_bounded.x, vec![0.0; 3]);
        assert_eq!(solved_zero_bounded.relative_residual, 0.0);

        // 1. Exact SPD preconditioner P = J (with non-unit diagonal entries) must
        // converge in 1 Krylov step because L^T J L = I when L L^T = P^{-1}.
        // Applying P^{-1} twice (P^{-1} J P^{-1} = P^{-1}) would NOT have unit
        // spectrum and would require 3 iterations.
        struct DiagInvPrecond([f64; 3]);
        impl Preconditioner for DiagInvPrecond {
            fn solve(&self, r: ArrayView1<f64>) -> Array1<f64> {
                Array1::from_iter(r.iter().zip(self.0).map(|(&ri, inv_d)| ri * inv_d))
            }
        }
        let diag_j = [4.0, 9.0, 25.0];
        let precond_exact = DiagInvPrecond([1.0 / 4.0, 1.0 / 9.0, 1.0 / 25.0]);
        let rhs_spd = [8.0, -27.0, 50.0];
        let solved_1step = minres_solve_precond(
            |v| Ok::<_, Infallible>(diag_j.iter().zip(v).map(|(d, x)| d * x).collect()),
            &rhs_spd,
            1.0e-12,
            &precond_exact,
            3,
        )
        .expect("exact SPD preconditioner solve");
        assert_eq!(
            solved_1step.iterations, 1,
            "P = J must converge in 1 step under L^T J L = I"
        );
        for (&got, want) in solved_1step.x.iter().zip([2.0, -3.0, 2.0]) {
            assert!((got - want).abs() <= 1.0e-12, "{got} != {want}");
        }
        assert!(solved_1step.relative_residual <= 1.0e-12);

        // 2. Non-diagonal SPD preconditioner P^{-1} = L L^T on a coupled symmetric
        // indefinite system J x = b: verify that minres_solve_precond(J, b, P)
        // matches minres_solve_bounded(J, b, L^T, L) in iterations, physical
        // solution x, and certified relative residual ||b - J x||_{P^{-1}} / ||b||_{P^{-1}}.
        let l_factor = [[2.0, 0.0, 0.0], [0.5, 1.5, 0.0], [-0.25, 0.5, 1.25]];
        let apply_l = |z: &[f64]| -> Vec<f64> {
            (0..3)
                .map(|i| (0..3).map(|j| l_factor[i][j] * z[j]).sum())
                .collect()
        };
        let apply_lt = |r: &[f64]| -> Vec<f64> {
            (0..3)
                .map(|i| (0..3).map(|j| l_factor[j][i] * r[j]).sum())
                .collect()
        };
        struct FullSpdPrecond {
            l: [[f64; 3]; 3],
        }
        impl Preconditioner for FullSpdPrecond {
            fn solve(&self, r: ArrayView1<f64>) -> Array1<f64> {
                let mut lt_r = [0.0; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        lt_r[i] += self.l[j][i] * r[j];
                    }
                }
                let mut out = [0.0; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        out[i] += self.l[i][j] * lt_r[j];
                    }
                }
                Array1::from_vec(out.to_vec())
            }
        }
        let j_mat = [[3.0, -0.7, 0.2], [-0.7, -2.5, 0.4], [0.2, 0.4, 5.0]];
        let apply_j = |v: &[f64]| -> Result<Vec<f64>, Infallible> {
            Ok((0..3)
                .map(|i| (0..3).map(|j| j_mat[i][j] * v[j]).sum())
                .collect())
        };
        let x_true = [1.5, -2.0, 0.75];
        let rhs_indef = apply_j(&x_true).unwrap();
        let full_precond = FullSpdPrecond { l: l_factor };

        let solved_full = minres_solve_precond(apply_j, &rhs_indef, 1.0e-12, &full_precond, 10)
            .expect("full P^{-1} preconditioned MINRES");
        let solved_split =
            minres_solve_bounded(apply_j, &rhs_indef, 1.0e-12, apply_lt, apply_l, 10)
                .expect("split (L^T, L) preconditioned MINRES");

        assert_eq!(solved_full.iterations, solved_split.iterations);
        for i in 0..3 {
            assert!(
                (solved_full.x[i] - x_true[i]).abs() <= 1.0e-12,
                "x[{i}]: got {} want {}",
                solved_full.x[i],
                x_true[i]
            );
            assert!(
                (solved_full.x[i] - solved_split.x[i]).abs() <= 1.0e-13,
                "full vs split x[{i}]: {} vs {}",
                solved_full.x[i],
                solved_split.x[i]
            );
        }
        // Independent check of certified relative residual ||L^T(b - J x)||_2 / ||L^T b||_2.
        let jx = apply_j(&solved_full.x).unwrap();
        let r_phys: Vec<f64> = rhs_indef.iter().zip(&jx).map(|(b, ax)| b - ax).collect();
        let lt_r = apply_lt(&r_phys);
        let lt_b = apply_lt(&rhs_indef);
        let indep_rel_res = dot(&lt_r, &lt_r).sqrt() / dot(&lt_b, &lt_b).sqrt();
        assert!(
            (solved_full.relative_residual - indep_rel_res).abs() <= 1.0e-15,
            "certified residual {} != independent {}",
            solved_full.relative_residual,
            indep_rel_res
        );

        // 3. Indefinite / non-SPD preconditioner with <r, P^{-1} r> <= 0 must be rejected.
        let bad_precond = DiagInvPrecond([-1.0, 1.0, 1.0]);
        let err_bad = minres_solve_precond(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &[2.0, 0.0, 0.0],
            1.0e-12,
            &bad_precond,
            5,
        )
        .expect_err("non-SPD preconditioner must fail");
        assert_eq!(err_bad, MinresError::SingularPreconditioner);

        // 4. Non-standard-layout (reversed-stride and step-strided with offset) Array1
        // outputs from Preconditioner::solve must be consumed in logical order, not raw
        // backing-buffer order.
        struct ReversedStridePrecond {
            inner: FullSpdPrecond,
        }
        impl Preconditioner for ReversedStridePrecond {
            fn solve(&self, r: ArrayView1<f64>) -> Array1<f64> {
                let logical = self.inner.solve(r);
                let mut rev = Array1::from_iter(logical.iter().rev().copied());
                rev.invert_axis(ndarray::Axis(0));
                rev
            }
        }
        struct OffsetStepStrideIdentityPrecond;
        impl Preconditioner for OffsetStepStrideIdentityPrecond {
            fn solve(&self, r: ArrayView1<f64>) -> Array1<f64> {
                let mut padded = Vec::with_capacity(2 * r.len() + 1);
                padded.push(-999.0);
                for &v in r.iter() {
                    padded.push(v);
                    padded.push(777.0);
                }
                Array1::from_vec(padded).slice_move(ndarray::s![1..;2])
            }
        }
        let solved_offset_stride = minres_solve_precond(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &[1.0, -2.0, 3.0],
            1.0e-12,
            &OffsetStepStrideIdentityPrecond,
            5,
        )
        .expect("offset/step-strided identity preconditioner");
        assert_eq!(solved_offset_stride.iterations, 1);
        for (&got, want) in solved_offset_stride.x.iter().zip([1.0, -2.0, 3.0]) {
            assert!((got - want).abs() <= 1.0e-14, "{got} != {want}");
        }

        let rev_precond = ReversedStridePrecond {
            inner: FullSpdPrecond { l: l_factor },
        };
        let solved_rev = minres_solve_precond(apply_j, &rhs_indef, 1.0e-12, &rev_precond, 10)
            .expect("reversed-stride SPD preconditioner");
        assert_eq!(solved_rev.iterations, solved_full.iterations);
        assert_eq!(solved_rev.x, solved_full.x);
        assert_eq!(solved_rev.relative_residual, solved_full.relative_residual);
    }

    #[test]
    fn min_normal_and_subnormal_rhs_do_not_underflow_to_false_ok() {
        for &tiny in &[f64::MIN_POSITIVE, f64::from_bits(1)] {
            assert!(tiny > 0.0);
            assert_eq!(
                tiny * tiny,
                0.0,
                "naive squaring of {tiny:e} underflows to zero"
            );

            // Identity operator must actually take 1 Krylov step and return x == [tiny].
            let solved = minres_solve_bounded(
                |v| Ok::<_, Infallible>(v.to_vec()),
                &[tiny],
                1.0e-12,
                |r| r.to_vec(),
                |z| z.to_vec(),
                4,
            )
            .expect("tiny nonzero rhs on identity operator must converge");
            assert_eq!(solved.iterations, 1);
            assert_eq!(solved.x, vec![tiny]);
            assert_eq!(solved.relative_residual, 0.0);

            // Zero operator on tiny nonzero RHS must report Breakdown with true_rel_res == 1.0,
            // never a false Ok at iteration 0 or 1.
            let breakdown = minres_solve_bounded(
                |_v| Ok::<_, Infallible>(vec![0.0]),
                &[tiny],
                1.0e-12,
                |r| r.to_vec(),
                |z| z.to_vec(),
                4,
            )
            .expect_err("zero operator on tiny nonzero rhs must fail");
            assert_eq!(
                breakdown,
                MinresError::Breakdown {
                    iterations: 1,
                    relative_residual: 1.0,
                }
            );

            // Rank-deficient operator on 2D tiny RHS must report Breakdown at iteration 2
            // because the uneliminated component has true relative residual 1/sqrt(2).
            let rank_deficient = minres_solve_bounded(
                |v| Ok::<_, Infallible>(vec![v[0], 0.0]),
                &[tiny, tiny],
                1.0e-12,
                |r| r.to_vec(),
                |z| z.to_vec(),
                4,
            )
            .expect_err("rank-deficient operator on tiny 2D rhs must fail certification");
            match rank_deficient.error {
                MinresError::Breakdown {
                    iterations: 2,
                    relative_residual,
                } => {
                    assert!(
                        (relative_residual - std::f64::consts::FRAC_1_SQRT_2).abs() <= 1.0e-15,
                        "unexpected relative residual {relative_residual}"
                    );
                }
                other => panic!("expected Breakdown at iter 2, got {other:?}"),
            }
        }
    }

    #[test]
    fn huge_iteration_bound_returns_capacity_overflow_without_panicking() {
        let err = minres_solve_bounded(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &[1.0, 2.0],
            1.0e-12,
            |r| r.to_vec(),
            |z| z.to_vec(),
            usize::MAX,
        )
        .expect_err("usize::MAX iteration_bound must return CapacityOverflow");
        assert_eq!(
            err,
            MinresError::CapacityOverflow {
                iteration_bound: usize::MAX,
            }
        );
    }

    #[test]
    fn overflowing_norms_and_recurrence_scalars_return_non_finite_arithmetic() {
        // 2D vector [f64::MAX, f64::MAX] has finite components, but its Euclidean norm
        // f64::MAX * sqrt(2) overflows f64::MAX.
        let err_norm = minres_solve_bounded(
            |v| Ok::<_, Infallible>(v.to_vec()),
            &[f64::MAX, f64::MAX],
            1.0e-12,
            |r| r.to_vec(),
            |z| z.to_vec(),
            4,
        )
        .expect_err("overflowing Euclidean norm must fail");
        assert_eq!(err_norm, MinresError::NonFiniteArithmetic);

        // Callback vectors are all finite, but the inner product alpha = dot(v, p)
        // overflows f64::MAX during the Lanczos step.
        let err_alpha = minres_solve_bounded(
            |_v| Ok::<_, Infallible>(vec![f64::MAX, f64::MAX]),
            &[3.0, 4.0],
            1.0e-12,
            |r| r.to_vec(),
            |z| z.to_vec(),
            4,
        )
        .expect_err("overflowing Lanczos alpha must return NonFiniteArithmetic");
        assert_eq!(err_alpha, MinresError::NonFiniteArithmetic);

        // Callback vectors and Lanczos scalars are finite, but the solution update
        // x_i += phi * w_i overflows f64::MAX when dividing by a tiny eigenvalue.
        let err_update = minres_solve_bounded(
            |v| Ok::<_, Infallible>(vec![1.0e-20 * v[0]]),
            &[1.0e300],
            1.0e-12,
            |r| r.to_vec(),
            |z| z.to_vec(),
            4,
        )
        .expect_err("overflowing iterate update must return NonFiniteArithmetic");
        assert_eq!(err_update, MinresError::NonFiniteArithmetic);
    }
}
