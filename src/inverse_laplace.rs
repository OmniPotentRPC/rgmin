//! Generic complex-contour inverse Laplace transform kernels (fixed-Talbot and Weeks/Laguerre),
//! orthonormal Laguerre series evaluation, and complex dense linear algebra.
//!
//! Every floating-point evaluation of a complex Laplace transform `F(beta)` using standard
//! `f64` elementary functions is represented as a typed [`ComplexSample`] estimate, and
//! contour inversions return typed [`TalbotEstimate`] and [`WeeksEstimate`] diagnostics
//! (`is_certified() == false`, `certified_bar() == None`). All non-negative estimate
//! accumulations use outward-directed IEEE-754 `.next_up()` arithmetic on every elementary step.

use core::f64::consts::{FRAC_1_SQRT_2, PI};
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// Asymptotic multiplier `1 / (2 / sqrt(3) - 1) = 3 + 2 sqrt(3) ≈ 6.464101615137755`
/// converting the last quarter `sum_{3N/4 <= n < N} |c_n|` of an `O(n^{-3/2})` Laguerre
/// series into the asymptotic infinite remainder `sum_{n >= N} |c_n|`.
pub const WEEKS_ALGEBRAIC_TAIL_FACTOR: f64 = 6.464_101_615_137_755;

/// Asymptotic multiplier `1 / (1 - 2^{-1/2}) = 2 + sqrt(2) ≈ 3.414213562373095`
/// converting the fine dyadic block `sum_{N <= n < 2N} |c_{2N,n}|` of an `O(n^{-3/2})`
/// Laguerre series into the asymptotic infinite remainder `sum_{n >= N} |c_n|`.
pub const WEEKS_DYADIC_TAIL_FACTOR: f64 = 3.414_213_562_373_095;

/// Maximum permitted dyadic tail ratio `sum_{N <= n < 2N} |a_n| / sum_{N/2 <= n < N} |a_n| <= 2^{-1/2}`
/// for an `O(n^{-3/2})` algebraic tail (`p >= 3/2`, where `2^{1 - p} <= 2^{-1/2}`).
pub const WEEKS_MAX_DYADIC_TAIL_RATIO: f64 = FRAC_1_SQRT_2;

/// Typed refusal conditions for complex-contour inverse Laplace transform kernels.
#[derive(Clone, Debug, PartialEq)]
pub enum InverseLaplaceError {
    /// A configuration or physical parameter was non-finite or out of its domain.
    InvalidParameter {
        /// Parameter name.
        parameter: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// The complex-beta contour entered `Re(beta) <= min_re_beta`.
    ContourCrossesCrossover {
        /// Contour method name.
        method: &'static str,
        /// Sampled `Re(beta)`.
        re_beta: f64,
        /// Minimum admissible `Re(beta)`.
        beta_c: f64,
    },
    /// The complex-beta contour exceeded the maximum admissible `|Im(beta)|`.
    ContourExceedsStrip {
        /// Contour method name.
        method: &'static str,
        /// Sampled `|Im(beta)|`.
        imag_beta: f64,
        /// Maximum admissible `|Im(beta)|`.
        max_imag_beta: f64,
    },
    /// A complex dense linear system encountered a singular or non-finite pivot.
    SingularLinearSystem {
        /// Solver stage name.
        stage: &'static str,
        /// Pivot magnitude at failure.
        pivot_abs: f64,
    },
    /// Refinement across two resolutions failed to contract.
    NonContractingRefinement {
        /// Refinement stage name.
        stage: &'static str,
        /// Coarse discrepancy or block norm.
        coarse_gap: f64,
        /// Fine discrepancy or block norm.
        fine_gap: f64,
    },
    /// An observed dyadic block fails the finite-resolution `n^{-3/2}` model.
    ///
    /// This is one dyadic comparison, not a proof of the eventual decay.
    SlowAlgebraicTail {
        /// Stage where slow algebraic decay was detected.
        stage: &'static str,
        /// Observed dyadic block ratio `sum_{upper} |a_n| / sum_{lower} |a_n|`.
        dyadic_ratio: f64,
        /// Maximum admissible dyadic block ratio `2^{-1/2}` for `O(n^{-3/2})` decay.
        max_dyadic_ratio: f64,
    },
    /// The outward-directed error estimate exceeded the requested tolerance.
    ToleranceNotMet {
        /// Method name.
        method: &'static str,
        /// Outward-directed error estimate. This is not a certified bar.
        estimate: f64,
        /// Caller-requested tolerance.
        tolerance: f64,
    },
    /// A proposed exterior-disk majorant contains the mapped singularity.
    ///
    /// No finite Cauchy majorant exists on a disk that contains a pole.
    SingularityInsideDisk {
        /// `|w_pole|` under `β = σ + b (1+w)/(1-w)`.
        modulus: f64,
        /// Requested disk radius. Certification would have required `modulus > outer_radius > 1`.
        outer_radius: f64,
    },
    /// The integrand callback refused the sample. The caller keeps the detailed error.
    CallbackRefused,
}

impl fmt::Display for InverseLaplaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { parameter, value } => {
                write!(
                    f,
                    "invalid inverse-Laplace parameter {parameter}: {value:.6e}"
                )
            }
            Self::ContourCrossesCrossover {
                method,
                re_beta,
                beta_c,
            } => write!(
                f,
                "{method} contour sampled Re(beta) = {re_beta:.6e} <= beta_c = {beta_c:.6e}"
            ),
            Self::ContourExceedsStrip {
                method,
                imag_beta,
                max_imag_beta,
            } => write!(
                f,
                "{method} contour sampled |Im(beta)| = {imag_beta:.6e} > max_imag_beta = {max_imag_beta:.6e}"
            ),
            Self::SingularLinearSystem { stage, pivot_abs } => write!(
                f,
                "complex linear system singular in {stage}: |pivot| = {pivot_abs:.6e}"
            ),
            Self::NonContractingRefinement {
                stage,
                coarse_gap,
                fine_gap,
            } => write!(
                f,
                "non-contracting refinement in {stage}: coarse {coarse_gap:.6e}, fine {fine_gap:.6e}"
            ),
            Self::SlowAlgebraicTail {
                stage,
                dyadic_ratio,
                max_dyadic_ratio,
            } => write!(
                f,
                "observed Laguerre block in {stage} fails the n^-3/2 model: dyadic ratio {dyadic_ratio:.6e} > {max_dyadic_ratio:.6e}"
            ),
            Self::ToleranceNotMet {
                method,
                estimate,
                tolerance,
            } => write!(
                f,
                "{method} uncertainty estimate {estimate:.6e} exceeds requested tolerance {tolerance:.6e}"
            ),
            Self::SingularityInsideDisk {
                modulus,
                outer_radius,
            } => write!(
                f,
                "Weeks singularity lies inside the disk: |w_pole| = {modulus:.6e} <= R = {outer_radius:.6e}"
            ),
            Self::CallbackRefused => f.write_str("inverse-Laplace callback refused the sample"),
        }
    }
}

fn checked_dim(
    parameter: &'static str,
    value: Option<usize>,
) -> Result<usize, InverseLaplaceError> {
    value.ok_or(InverseLaplaceError::InvalidParameter {
        parameter,
        value: f64::INFINITY,
    })
}

impl std::error::Error for InverseLaplaceError {}

/// Binary64 complex number `re + i im` for complex-beta contour quadrature and dense linear algebra.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex64 {
    /// Construct `re + i im`.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Construct from polar coordinates `r * (cos(theta) + i sin(theta))`.
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Complex conjugate `re - i im`.
    #[inline]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared modulus `re^2 + im^2`.
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Euclidean modulus `hypot(re, im)`.
    #[inline]
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Alias for [`Self::norm`].
    #[inline]
    pub fn abs(self) -> f64 {
        self.norm()
    }

    /// True if both real and imaginary parts are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Complex exponential `exp(re) * (cos(im) + i sin(im))`.
    #[inline]
    pub fn exp(self) -> Self {
        let er = self.re.exp();
        Self {
            re: er * self.im.cos(),
            im: er * self.im.sin(),
        }
    }

    /// Principal complex square root with `Re(sqrt(z)) >= 0`.
    #[inline]
    pub fn sqrt(self) -> Self {
        let r = self.norm();
        if r == 0.0 {
            return Self::new(0.0, 0.0);
        }
        let u = (0.5 * (r + self.re.abs())).sqrt();
        let v = 0.5 * self.im / u;
        if self.re >= 0.0 {
            Self::new(u, v)
        } else {
            Self::new(v.abs(), if self.im >= 0.0 { u } else { -u })
        }
    }

    /// Complex sine `sin(re) cosh(im) + i cos(re) sinh(im)`.
    #[inline]
    pub fn sin(self) -> Self {
        Self {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    /// Complex cosine `cos(re) cosh(im) - i sin(re) sinh(im)`.
    #[inline]
    pub fn cos(self) -> Self {
        Self {
            re: self.re.cos() * self.im.cosh(),
            im: -self.re.sin() * self.im.sinh(),
        }
    }

    /// Complex hyperbolic sine `sinh(re) cos(im) + i cosh(re) sin(im)`.
    #[inline]
    pub fn sinh(self) -> Self {
        Self {
            re: self.re.sinh() * self.im.cos(),
            im: self.re.cosh() * self.im.sin(),
        }
    }
}

impl From<(f64, f64)> for Complex64 {
    #[inline]
    fn from((re, im): (f64, f64)) -> Self {
        Self { re, im }
    }
}

impl From<Complex64> for (f64, f64) {
    #[inline]
    fn from(z: Complex64) -> Self {
        (z.re, z.im)
    }
}

impl Add for Complex64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Neg for Complex64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Mul for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Mul<f64> for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl Mul<Complex64> for f64 {
    type Output = Complex64;
    #[inline]
    fn mul(self, rhs: Complex64) -> Self::Output {
        Complex64 {
            re: self * rhs.re,
            im: self * rhs.im,
        }
    }
}

impl Div for Complex64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let scale = rhs.re.abs() + rhs.im.abs();
        if scale == 0.0 {
            return Self::new(f64::NAN, f64::NAN);
        }
        let xr = self.re / scale;
        let xi = self.im / scale;
        let yr = rhs.re / scale;
        let yi = rhs.im / scale;
        let denom = yr * yr + yi * yi;
        Self {
            re: (xr * yr + xi * yi) / denom,
            im: (xi * yr - xr * yi) / denom,
        }
    }
}

impl Div<f64> for Complex64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

/// Outward-directed sum `(a + b).next_up()` for non-negative finite upper bounds.
#[inline]
pub fn add_up(a: f64, b: f64) -> f64 {
    (a + b).next_up()
}

/// Outward-directed product `(a * b).next_up()` for non-negative finite upper bounds.
#[inline]
pub fn mul_up(a: f64, b: f64) -> f64 {
    (a * b).next_up()
}

/// Outward-directed quotient `(a / b).next_up()` for non-negative finite `a` and positive `b`.
#[inline]
pub fn div_up(a: f64, b: f64) -> f64 {
    (a / b).next_up()
}

/// Outward-directed square root `a.sqrt().next_up()` for non-negative finite `a`.
#[inline]
pub fn sqrt_up(a: f64) -> f64 {
    a.sqrt().next_up()
}

/// Outward-directed modulus `z.norm().next_up()` for a complex value `z`.
#[inline]
pub fn norm_up(z: Complex64) -> f64 {
    z.norm().next_up()
}

/// Outward-directed inverse power `(r^{-n})` computed by step-by-step `mul_up`.
#[inline]
pub fn powi_inv_up(r: f64, n: usize) -> f64 {
    let inv_r = div_up(1.0, r);
    let mut acc = 1.0_f64;
    for _ in 0..n {
        acc = mul_up(acc, inv_r);
    }
    acc
}

/// Outward-directed sum of a slice of non-negative finite components.
pub fn upward_sum(components: &[f64]) -> Result<f64, InverseLaplaceError> {
    let mut total = 0.0_f64;
    for &c in components {
        if !c.is_finite() || c < 0.0 {
            return Err(InverseLaplaceError::InvalidParameter {
                parameter: "error_component",
                value: c,
            });
        }
        if c > 0.0 {
            total = add_up(total, c);
        }
    }
    if !total.is_finite() {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "summed_estimate",
            value: total,
        });
    }
    Ok(total)
}

/// Higham unit-roundoff factor `gamma_n = n u / (1 - n u)` with `u = f64::EPSILON / 2`,
/// computed with outward-directed rounding.
pub fn higham_gamma(n: usize) -> Result<f64, InverseLaplaceError> {
    let u = 0.5 * f64::EPSILON;
    let nu = mul_up(n as f64, u);
    if !nu.is_finite() || nu >= 0.5 {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "higham_gamma_n",
            value: n as f64,
        });
    }
    let denom = (1.0 - nu).next_down();
    Ok(div_up(nu, denom))
}

/// Fail-closed maximum of two finite scalars `acc` and `candidate`.
/// Refuses immediately if either input is `NaN` or infinite.
#[inline]
pub fn max_finite(
    acc: f64,
    candidate: f64,
    parameter: &'static str,
) -> Result<f64, InverseLaplaceError> {
    if !acc.is_finite() {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter,
            value: acc,
        });
    }
    if !candidate.is_finite() {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter,
            value: candidate,
        });
    }
    Ok(if candidate > acc { candidate } else { acc })
}

#[inline]
fn require_positive_finite(name: &'static str, value: f64) -> Result<(), InverseLaplaceError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: name,
            value,
        });
    }
    Ok(())
}

#[inline]
fn require_nonnegative_finite(name: &'static str, value: f64) -> Result<(), InverseLaplaceError> {
    if !value.is_finite() || value < 0.0 {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: name,
            value,
        });
    }
    Ok(())
}

/// Pointwise complex evaluation `F(beta)` paired with a non-negative error estimate `|delta F(beta)|`.
///
/// Evaluations in binary64 arithmetic that call standard library transcendental functions
/// (`sqrt`, `exp`, `sinh`, `sin`, `cos`) are never labeled as proof-level enclosures;
/// [`Self::is_certified`] is always `false` and [`Self::certified_bar`] is always `None`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexSample {
    /// Sampled complex value `F(beta)`.
    pub value: Complex64,
    /// Non-negative pointwise error estimate `|delta F(beta)|`.
    pub estimate: f64,
}

impl ComplexSample {
    /// Construct a pointwise complex estimate.
    #[inline]
    pub const fn estimated(value: Complex64, estimate: f64) -> Self {
        Self { value, estimate }
    }

    /// Sampled complex value `F(beta)`.
    #[inline]
    pub const fn value(self) -> Complex64 {
        self.value
    }

    /// Pointwise error estimate `|delta F(beta)|`.
    #[inline]
    pub const fn uncertainty_estimate(self) -> f64 {
        self.estimate
    }

    /// Returns `false`: binary64 transcendental evaluations are estimates, not proof-level enclosures.
    #[inline]
    pub const fn is_certified(self) -> bool {
        false
    }

    /// Returns `None`: binary64 transcendental evaluations do not carry a proof-level certified bar.
    #[inline]
    pub const fn certified_bar(self) -> Option<f64> {
        None
    }
}

/// Solve a dense `n x n` complex linear system `A x = b` by Gaussian elimination with partial pivoting.
pub fn solve_complex_dense(
    a: &[Complex64],
    b: &[Complex64],
    n: usize,
) -> Result<Vec<Complex64>, InverseLaplaceError> {
    let nn = checked_dim("matrix_dimension", n.checked_mul(n))?;
    if n == 0 || a.len() != nn || b.len() != n {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "linear_system_dimension",
            value: n as f64,
        });
    }
    let mut mat = a.to_vec();
    let mut rhs = b.to_vec();
    let mut scale = 0.0_f64;
    for z in &mat {
        if !z.is_finite() {
            return Err(InverseLaplaceError::SingularLinearSystem {
                stage: "non_finite_matrix_entry",
                pivot_abs: 0.0,
            });
        }
        scale = max_finite(scale, z.norm(), "matrix_entry")?;
    }
    // Relative to the matrix. An absolute floor of `64 ε` rejects a tiny
    // well-conditioned system whose entries are all far below 1.
    let pivot_floor = mul_up(mul_up(64.0, f64::EPSILON), scale);

    for k in 0..n {
        let mut piv_row = k;
        let mut piv_abs = mat[k * n + k].norm();
        for i in (k + 1)..n {
            let cand = mat[i * n + k].norm();
            if cand > piv_abs {
                piv_abs = cand;
                piv_row = i;
            }
        }
        if !piv_abs.is_finite() || piv_abs <= pivot_floor {
            return Err(InverseLaplaceError::SingularLinearSystem {
                stage: "gaussian_elimination",
                pivot_abs: piv_abs,
            });
        }
        if piv_row != k {
            for col in k..n {
                mat.swap(k * n + col, piv_row * n + col);
            }
            rhs.swap(k, piv_row);
        }
        let piv = mat[k * n + k];
        for i in (k + 1)..n {
            let factor = mat[i * n + k] / piv;
            mat[i * n + k] = Complex64::new(0.0, 0.0);
            for col in (k + 1)..n {
                mat[i * n + col] = mat[i * n + col] - factor * mat[k * n + col];
            }
            rhs[i] = rhs[i] - factor * rhs[k];
        }
    }

    let mut x = vec![Complex64::new(0.0, 0.0); n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for col in (i + 1)..n {
            sum = sum - mat[i * n + col] * x[col];
        }
        let z = sum / mat[i * n + i];
        if !z.is_finite() {
            return Err(InverseLaplaceError::SingularLinearSystem {
                stage: "back_substitution",
                pivot_abs: mat[i * n + i].norm(),
            });
        }
        x[i] = z;
    }
    Ok(x)
}

/// Estimate `||A^{-1}||_inf` from binary64 Gaussian elimination.
///
/// The columns are ordinary `f64` solves with no residual correction. Applying
/// `next_up` while summing those computed columns does not upper-bound the
/// norm of the exact inverse.
pub fn complex_matrix_inv_inf_norm(a: &[Complex64], n: usize) -> Result<f64, InverseLaplaceError> {
    let nn = checked_dim("matrix_dimension", n.checked_mul(n))?;
    if n == 0 || a.len() != nn {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "matrix_dimension",
            value: n as f64,
        });
    }
    let mut cols = vec![Complex64::new(0.0, 0.0); nn];
    let mut unit = vec![Complex64::new(0.0, 0.0); n];
    for j in 0..n {
        if j > 0 {
            unit[j - 1] = Complex64::new(0.0, 0.0);
        }
        unit[j] = Complex64::new(1.0, 0.0);
        let col = solve_complex_dense(a, &unit, n)?;
        for i in 0..n {
            cols[i * n + j] = col[i];
        }
    }
    let mut max_row = 0.0_f64;
    for i in 0..n {
        let mut row_sum = 0.0_f64;
        for j in 0..n {
            row_sum = add_up(row_sum, norm_up(cols[i * n + j]));
        }
        max_row = max_finite(max_row, row_sum, "inv_inf_norm_row")?;
    }
    Ok(max_row)
}

/// Evaluate the orthonormal Laguerre function series
/// `sum_{n=0}^{N-1} c_n phi_n(E; b)` where `phi_n(E; b) = sqrt(2b) exp(-b E) L_n(2b E)`.
///
/// Returns `(value, roundoff_estimate)`.
pub fn eval_orthonormal_laguerre_series(
    coeffs: &[f64],
    scale_b: f64,
    energy: f64,
) -> Result<(f64, f64), InverseLaplaceError> {
    require_positive_finite("scale_b", scale_b)?;
    require_nonnegative_finite("energy", energy)?;
    if coeffs.is_empty() {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "coeffs_len",
            value: 0.0,
        });
    }
    for &c in coeffs {
        if !c.is_finite() {
            return Err(InverseLaplaceError::InvalidParameter {
                parameter: "laguerre_coeff",
                value: c,
            });
        }
    }
    let x = 2.0 * scale_b * energy;
    let pref = (2.0 * scale_b).sqrt();
    let pref_up = mul_up(sqrt_up(2.0), sqrt_up(scale_b));
    let half_x = 0.5 * x;

    let mut l_prev = 0.0_f64;
    let mut l_curr = (-half_x).exp();
    let mut sum = coeffs[0] * l_curr;
    let mut abs_sum = mul_up(coeffs[0].abs(), l_curr.abs().next_up());

    for (idx, &c) in coeffs.iter().enumerate().skip(1) {
        let n_minus_1 = idx - 1;
        let nf = n_minus_1 as f64;
        let l_next = ((2.0 * nf + 1.0 - x) * l_curr - nf * l_prev) / ((n_minus_1 + 1) as f64);
        l_prev = l_curr;
        l_curr = l_next;
        sum += c * l_curr;
        abs_sum = add_up(abs_sum, mul_up(c.abs(), l_curr.abs().next_up()));
    }
    let value = pref * sum;
    let gamma = higham_gamma(checked_dim(
        "laguerre_ops",
        coeffs
            .len()
            .checked_mul(4)
            .and_then(|ops| ops.checked_add(6)),
    )?)?;
    let roundoff = mul_up(gamma, mul_up(pref_up, abs_sum));
    Ok((value, roundoff))
}

/// Verify that a sequence of `N` Laguerre coefficients (`N >= 16`) contracts across its last
/// two quarters and satisfies the dyadic algebraic decay bound
/// `sum_{n=N/2}^{N-1} |c_n| / sum_{n=N/4}^{N/2-1} |c_n| <= 2^{-1/2}` whenever the upper half
/// exceeds `roundoff_floor`.
///
/// Returns the observed dyadic ratio `sum_{n=N/2}^{N-1} |c_n| / sum_{n=N/4}^{N/2-1} |c_n|`.
pub fn verify_laguerre_tail_decay(
    coeffs: &[f64],
    roundoff_floor: f64,
) -> Result<f64, InverseLaplaceError> {
    let m = coeffs.len();
    if m < 16 {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "coeffs_len",
            value: m as f64,
        });
    }
    require_nonnegative_finite("roundoff_floor", roundoff_floor)?;
    for &c in coeffs {
        if !c.is_finite() {
            return Err(InverseLaplaceError::InvalidParameter {
                parameter: "laguerre_coeff",
                value: c,
            });
        }
    }
    let q = m / 4;
    let mut q1 = 0.0_f64;
    let mut q2 = 0.0_f64;
    let mut q3 = 0.0_f64;
    for &c in &coeffs[q..2 * q] {
        q1 = add_up(q1, c.abs());
    }
    for &c in &coeffs[2 * q..3 * q] {
        q2 = add_up(q2, c.abs());
    }
    for &c in &coeffs[3 * q..m] {
        q3 = add_up(q3, c.abs());
    }
    if q3 > roundoff_floor && q3 > q2 {
        return Err(InverseLaplaceError::NonContractingRefinement {
            stage: "laguerre_tail_quarters",
            coarse_gap: q2,
            fine_gap: q3,
        });
    }
    let upper_half = add_up(q2, q3);
    let dyadic_ratio = if q1 > 0.0 {
        div_up(upper_half, q1)
    } else {
        0.0
    };
    if upper_half > roundoff_floor && dyadic_ratio > WEEKS_MAX_DYADIC_TAIL_RATIO {
        return Err(InverseLaplaceError::SlowAlgebraicTail {
            stage: "laguerre_dyadic_tail",
            dyadic_ratio,
            max_dyadic_ratio: WEEKS_MAX_DYADIC_TAIL_RATIO,
        });
    }
    Ok(dyadic_ratio)
}

/// Möbius preimage `w = (β - σ - b) / (β - σ + b)` of the Weeks map
/// `β = σ + b (1 + w) / (1 - w)`.
pub fn weeks_mobius_w(
    beta: Complex64,
    sigma: f64,
    scale_b: f64,
) -> Result<Complex64, InverseLaplaceError> {
    require_positive_finite("scale_b", scale_b)?;
    if !sigma.is_finite() || !beta.is_finite() {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "weeks_mobius_beta",
            value: beta.re,
        });
    }
    let num = beta - Complex64::new(sigma + scale_b, 0.0);
    let den = beta - Complex64::new(sigma - scale_b, 0.0);
    let den_abs = den.norm();
    if !den_abs.is_finite() || den_abs == 0.0 {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "weeks_mobius_denominator",
            value: den_abs,
        });
    }
    let w = num / den;
    if !w.is_finite() {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "weeks_mobius_w",
            value: w.re,
        });
    }
    Ok(w)
}

/// `|w_pole|` when it lies strictly outside the disk of radius `outer_radius > 1`.
///
/// Returns [`InverseLaplaceError::SingularityInsideDisk`] when the image of
/// `pole` is inside or on that disk. A disk that contains the singularity
/// has no finite Cauchy majorant, so this function never reports success
/// for the false `R = 3` geometry whose pole sits near `w = -2`.
pub fn exterior_disk_modulus(
    pole: Complex64,
    sigma: f64,
    scale_b: f64,
    outer_radius: f64,
) -> Result<f64, InverseLaplaceError> {
    require_positive_finite("outer_radius", outer_radius)?;
    if outer_radius <= 1.0 {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "outer_radius",
            value: outer_radius,
        });
    }
    let w = weeks_mobius_w(pole, sigma, scale_b)?;
    let modulus = w.norm();
    if !modulus.is_finite() || modulus <= outer_radius {
        return Err(InverseLaplaceError::SingularityInsideDisk {
            modulus,
            outer_radius,
        });
    }
    Ok(modulus)
}

/// Configuration for fixed-Talbot complex-contour inverse Laplace transform.
#[derive(Clone, Copy, Debug)]
pub struct TalbotConfig {
    /// Fine quadrature node count `M` (`M >= 8`, even; compared against coarse `M/2`).
    pub nodes: usize,
    /// Minimum admissible `Re(beta)` before refusing with [`InverseLaplaceError::ContourCrossesCrossover`].
    pub min_re_beta: f64,
    /// Maximum admissible `uncertainty_estimate()`.
    pub tolerance: f64,
}

impl Default for TalbotConfig {
    fn default() -> Self {
        Self {
            nodes: 32,
            min_re_beta: -1.0e6,
            tolerance: 1.0e-6,
        }
    }
}

/// Typed refinement and roundoff estimate for fixed-Talbot contour inversion.
///
/// Because contour samples evaluate binary64 transcendental functions, this is an estimate
/// (`is_certified() == false`, `certified_bar() == None`).
#[derive(Clone, Debug, PartialEq)]
pub struct TalbotEstimate {
    /// Outward-directed sum of `discretization_estimate`, `evaluation_estimate`, and `roundoff_estimate`.
    pub estimate: f64,
    /// Coarse-versus-fine (`M/2` vs `M`) discrepancy plus exponential contour-tail estimate.
    pub discretization_estimate: f64,
    /// Quadrature propagation of pointwise `ComplexSample::estimate` values.
    pub evaluation_estimate: f64,
    /// Outward-directed floating-point summation roundoff estimate.
    pub roundoff_estimate: f64,
}

/// Result of fixed-Talbot complex-contour inverse Laplace transform.
#[derive(Clone, Debug)]
pub struct TalbotResult {
    /// Recovered inverse Laplace value at `energy`.
    pub value: f64,
    /// Typed refinement, evaluation, and roundoff estimate components.
    pub accuracy: TalbotEstimate,
    /// Fine quadrature node count `M`.
    pub nodes_used: usize,
    /// Total evaluations of the complex Laplace transform callback (`3 M / 2`).
    pub evaluations: usize,
    /// Minimum `Re(beta)` sampled on the contour.
    pub min_re_beta_sampled: f64,
    /// Maximum `|Im(beta)|` sampled on the contour.
    pub max_imag_beta_sampled: f64,
}

impl TalbotResult {
    /// Returns `false`: contour inversion over binary64 transcendental callbacks is an estimate.
    #[inline]
    pub const fn is_certified(&self) -> bool {
        false
    }

    /// Returns `None`: contour inversion over binary64 transcendental callbacks has no proof-level bar.
    #[inline]
    pub const fn certified_bar(&self) -> Option<f64> {
        None
    }

    /// Outward-directed total uncertainty estimate.
    #[inline]
    pub const fn uncertainty_estimate(&self) -> f64 {
        self.accuracy.estimate
    }
}

/// Evaluate the fixed-Talbot contour node `(beta(theta), d beta / d theta)` at angle `theta in (0, pi)`
/// with scale `r = M / (5 E)`.
pub fn talbot_node(
    theta: f64,
    nodes_m: usize,
    energy: f64,
) -> Result<(Complex64, Complex64), InverseLaplaceError> {
    require_positive_finite("energy", energy)?;
    if !(0.0 < theta && theta < PI) {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "theta",
            value: theta,
        });
    }
    if nodes_m < 4 {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "nodes",
            value: nodes_m as f64,
        });
    }
    let r = (nodes_m as f64) / (5.0 * energy);
    let sin_t = theta.sin();
    let cos_t = theta.cos();
    let cot_t = cos_t / sin_t;
    let csc2_t = 1.0 / (sin_t * sin_t);
    let beta = Complex64::new(r * theta * cot_t, r * theta);
    let dbeta = Complex64::new(r * (cot_t - theta * csc2_t), r);
    Ok((beta, dbeta))
}

fn talbot_pass<F>(
    energy: f64,
    m: usize,
    min_re_beta: f64,
    eval_f: &mut F,
) -> Result<(f64, f64, f64, f64, f64, f64), InverseLaplaceError>
where
    F: FnMut(Complex64) -> Result<ComplexSample, InverseLaplaceError>,
{
    let h = div_up(PI, m as f64);
    let h_nom = PI / (m as f64);
    let mut sum_re = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut eval_est = 0.0_f64;
    let mut min_re = f64::INFINITY;
    let mut max_im = 0.0_f64;
    let mut endpoint_abs = 0.0_f64;

    for k in 1..=m {
        let theta = ((k as f64) - 0.5) * h_nom;
        let (beta, dbeta) = talbot_node(theta, m, energy)?;
        if beta.re <= min_re_beta {
            return Err(InverseLaplaceError::ContourCrossesCrossover {
                method: "fixed_talbot",
                re_beta: beta.re,
                beta_c: min_re_beta,
            });
        }
        if beta.re < min_re {
            min_re = beta.re;
        }
        max_im = max_finite(max_im, beta.im.abs(), "talbot_max_im")?;
        let sample = eval_f(beta)?;
        let f_val = sample.value();
        let f_err = sample.uncertainty_estimate();
        if !f_val.is_finite() || !f_err.is_finite() || f_err < 0.0 {
            return Err(InverseLaplaceError::InvalidParameter {
                parameter: "talbot_sample",
                value: f_err,
            });
        }
        let exp_eb = (beta * energy).exp();
        let weight = Complex64::new(0.0, -1.0 / PI) * exp_eb * dbeta;
        let term = weight * f_val;
        sum_re += term.re;
        let term_abs = norm_up(term);
        sum_abs = add_up(sum_abs, term_abs);
        let weight_abs = norm_up(weight);
        eval_est = add_up(eval_est, mul_up(weight_abs, f_err));
        if k == m {
            endpoint_abs = norm_up(f_val);
        }
    }

    let value = h_nom * sum_re;
    let eval_est = mul_up(h, eval_est);
    let l1_integral = mul_up(h, sum_abs);
    let gamma = higham_gamma(checked_dim(
        "talbot_ops",
        m.checked_mul(6).and_then(|ops| ops.checked_add(8)),
    )?)?;
    let roundoff = mul_up(gamma, l1_integral);

    let theta_end = ((m as f64) - 0.5) * h_nom;
    let (beta_end, dbeta_end) = talbot_node(theta_end, m, energy)?;
    // One panel of the endpoint integrand, including |F|. Scaling F by a
    // constant then scales this truncation term by the same factor.
    let tail_est = mul_up(
        div_up(h, PI),
        mul_up(
            mul_up((beta_end.re * energy).exp().next_up(), norm_up(dbeta_end)),
            endpoint_abs,
        ),
    );

    Ok((value, eval_est, roundoff, tail_est, min_re, max_im))
}

/// Invert a complex Laplace transform `F(beta)` at `energy > 0` along the fixed-Talbot contour.
pub fn talbot_invert<F>(
    energy: f64,
    config: TalbotConfig,
    mut eval_f: F,
) -> Result<TalbotResult, InverseLaplaceError>
where
    F: FnMut(Complex64) -> Result<ComplexSample, InverseLaplaceError>,
{
    require_positive_finite("energy", energy)?;
    require_positive_finite("tolerance", config.tolerance)?;
    if !config.min_re_beta.is_finite() {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "min_re_beta",
            value: config.min_re_beta,
        });
    }
    if config.nodes < 8 || config.nodes % 2 != 0 {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "nodes",
            value: config.nodes as f64,
        });
    }

    let m_coarse = config.nodes / 2;
    let m_fine = config.nodes;
    let (v_coarse, eval_coarse, round_coarse, _tail_coarse, min_re_c, max_im_c) =
        talbot_pass(energy, m_coarse, config.min_re_beta, &mut eval_f)?;
    let (v_fine, eval_fine, round_fine, tail_fine, min_re_f, max_im_f) =
        talbot_pass(energy, m_fine, config.min_re_beta, &mut eval_f)?;

    let disc_est = add_up((v_fine - v_coarse).abs().next_up(), tail_fine);
    let eval_est = upward_sum(&[eval_coarse, eval_fine])?;
    let roundoff_est = upward_sum(&[round_coarse, round_fine])?;
    let estimate = upward_sum(&[disc_est, eval_est, roundoff_est])?;

    if estimate > config.tolerance {
        return Err(InverseLaplaceError::ToleranceNotMet {
            method: "fixed_talbot",
            estimate,
            tolerance: config.tolerance,
        });
    }

    let min_re_sampled = if min_re_c < min_re_f {
        min_re_c
    } else {
        min_re_f
    };
    let max_imag_sampled = max_finite(max_im_c, max_im_f, "talbot_max_imag_sampled")?;

    Ok(TalbotResult {
        value: v_fine,
        accuracy: TalbotEstimate {
            estimate,
            discretization_estimate: disc_est,
            evaluation_estimate: eval_est,
            roundoff_estimate: roundoff_est,
        },
        nodes_used: m_fine,
        evaluations: m_coarse + m_fine,
        min_re_beta_sampled: min_re_sampled,
        max_imag_beta_sampled: max_imag_sampled,
    })
}

/// Configuration for Weeks (Möbius–Laguerre) complex-contour inverse Laplace transform
/// inside a strip `Re(beta) > min_re_beta`, `|Im(beta)| <= max_imag_beta`.
#[derive(Clone, Copy, Debug)]
pub struct WeeksConfig {
    /// Real Möbius center `sigma > min_re_beta`.
    pub sigma: f64,
    /// Möbius Laguerre scale `b > 0`.
    pub scale_b: f64,
    /// Cauchy circle radius `r in (0, 1]`.
    pub circle_radius: f64,
    /// Number of retained Laguerre modes `M >= 16` (even; verified against `2M`).
    pub laguerre_degree: usize,
    /// Minimum admissible `Re(beta)` (such as `beta_c`).
    pub min_re_beta: f64,
    /// Maximum admissible `|Im(beta)|` in the strip.
    pub max_imag_beta: f64,
    /// Maximum admissible `uncertainty_estimate()`.
    pub tolerance: f64,
}

/// Typed refinement, roundoff, and `O(n^{-3/2})` algebraic-tail estimate for Weeks inversion.
///
/// Because contour samples evaluate binary64 transcendental functions, this is an estimate
/// (`is_certified() == false`, `certified_bar() == None`).
#[derive(Clone, Debug, PartialEq)]
pub struct WeeksEstimate {
    /// Outward-directed sum of `discretization_estimate`, `evaluation_estimate`, `roundoff_estimate`,
    /// and `algebraic_tail_estimate`.
    pub estimate: f64,
    /// Coarse-versus-fine value difference. The coefficient `L1` shift is
    /// [`Self::coefficient_shift`] and is not part of [`Self::estimate`].
    pub discretization_estimate: f64,
    /// Coarse-versus-fine coefficient `L1` shift. A refinement diagnostic, never a bar.
    pub coefficient_shift: f64,
    /// Quadrature propagation of pointwise `ComplexSample::estimate` values amplified by `r^{-n}`.
    pub evaluation_estimate: f64,
    /// Outward-directed floating-point DFT and Clenshaw recurrence roundoff estimate.
    pub roundoff_estimate: f64,
    /// Asymptotic `O(n^{-3/2})` Laguerre truncation estimate.
    pub algebraic_tail_estimate: f64,
    /// Observed fine-grid dyadic tail ratio `sum_{M..2M} |c_{2M,n}| / sum_{M/2..M} |c_{2M,n}| <= 2^{-1/2}`.
    pub observed_dyadic_ratio: f64,
}

/// Result of Weeks (Möbius–Laguerre) complex-contour inverse Laplace transform.
#[derive(Clone, Debug)]
pub struct WeeksResult {
    /// Recovered inverse Laplace value at `energy`.
    pub value: f64,
    /// Typed refinement, evaluation, roundoff, and algebraic-tail estimate components.
    pub accuracy: WeeksEstimate,
    /// Retained orthonormal Laguerre expansion coefficients `c_0, ..., c_{M-1}`.
    pub coefficients: Vec<f64>,
    /// Total evaluations of the complex Laplace transform callback (`3 M`).
    pub evaluations: usize,
    /// Minimum `Re(beta)` sampled on the circle `|w| = r`.
    pub min_re_beta_sampled: f64,
    /// Maximum `|Im(beta)|` sampled on the circle `|w| = r`.
    pub max_imag_beta_sampled: f64,
}

impl WeeksResult {
    /// Returns `false`: contour inversion over binary64 transcendental callbacks is an estimate.
    #[inline]
    pub const fn is_certified(&self) -> bool {
        false
    }

    /// Returns `None`: contour inversion over binary64 transcendental callbacks has no proof-level bar.
    #[inline]
    pub const fn certified_bar(&self) -> Option<f64> {
        None
    }

    /// Outward-directed total uncertainty estimate.
    #[inline]
    pub const fn uncertainty_estimate(&self) -> f64 {
        self.accuracy.estimate
    }
}

struct WeeksCirclePass {
    coeffs: Vec<f64>,
    eval_est: f64,
    dft_roundoff: f64,
    sample_noise_per_mode: f64,
    min_re: f64,
    max_im: f64,
}

fn weeks_dft_coefficients<F>(
    config: &WeeksConfig,
    quad_m: usize,
    num_coeffs: usize,
    eval_f: &mut F,
) -> Result<WeeksCirclePass, InverseLaplaceError>
where
    F: FnMut(Complex64) -> Result<ComplexSample, InverseLaplaceError>,
{
    let r = config.circle_radius;
    let sigma = config.sigma;
    let b = config.scale_b;
    let sqrt_2b = (2.0 * b).sqrt();
    let sqrt_2b_up = mul_up(sqrt_up(2.0), sqrt_up(b));
    let mut phi_samples = Vec::with_capacity(quad_m);
    let mut max_sample_err = 0.0_f64;
    let mut max_phi_abs = 0.0_f64;
    let mut min_re = f64::INFINITY;
    let mut max_im = 0.0_f64;
    let gamma_node = higham_gamma(12)?;

    for j in 0..quad_m {
        let theta = 2.0 * PI * ((j as f64) + 0.5) / (quad_m as f64);
        let w = Complex64::from_polar(r, theta);
        let one = Complex64::new(1.0, 0.0);
        let one_minus_w = one - w;
        let one_plus_w = one + w;
        let beta = Complex64::new(sigma, 0.0) + (one_plus_w / one_minus_w) * b;
        if beta.re <= config.min_re_beta {
            return Err(InverseLaplaceError::ContourCrossesCrossover {
                method: "weeks_contour",
                re_beta: beta.re,
                beta_c: config.min_re_beta,
            });
        }
        if beta.im.abs() > config.max_imag_beta {
            return Err(InverseLaplaceError::ContourExceedsStrip {
                method: "weeks_contour",
                imag_beta: beta.im.abs(),
                max_imag_beta: config.max_imag_beta,
            });
        }
        if beta.re < min_re {
            min_re = beta.re;
        }
        max_im = max_finite(max_im, beta.im.abs(), "weeks_max_im")?;

        let sample = eval_f(beta)?;
        let f_val = sample.value();
        let f_err = sample.uncertainty_estimate();
        if !f_val.is_finite() || !f_err.is_finite() || f_err < 0.0 {
            return Err(InverseLaplaceError::InvalidParameter {
                parameter: "weeks_sample",
                value: f_err,
            });
        }
        let pref = one / one_minus_w * sqrt_2b;
        let phi = pref * f_val;
        let pref_abs = div_up(sqrt_2b_up, norm_up(one_minus_w));
        let phi_abs = norm_up(phi);
        let phi_err = add_up(mul_up(pref_abs, f_err), mul_up(gamma_node, phi_abs));
        max_sample_err = max_finite(max_sample_err, phi_err, "weeks_sample_err")?;
        max_phi_abs = max_finite(max_phi_abs, phi_abs, "weeks_phi_abs")?;
        phi_samples.push((theta, phi));
    }

    let mut coeffs = vec![0.0_f64; num_coeffs];
    let inv_m = 1.0 / (quad_m as f64);
    for (n, slot) in coeffs.iter_mut().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);
        for &(theta, phi) in &phi_samples {
            let phase = Complex64::from_polar(1.0, -(n as f64) * theta);
            sum = sum + phi * phase;
        }
        let r_pow = r.powi(n as i32);
        if !r_pow.is_finite() || r_pow <= 0.0 {
            return Err(InverseLaplaceError::InvalidParameter {
                parameter: "circle_radius_power",
                value: r_pow,
            });
        }
        *slot = (sum.re * inv_m) / r_pow;
    }

    let retain_m = config.laguerre_degree;
    let mut r_inv_sum = 0.0_f64;
    for n in 0..retain_m {
        r_inv_sum = add_up(r_inv_sum, powi_inv_up(r, n));
    }
    let eval_est = mul_up(max_sample_err, r_inv_sum);
    let gamma_dft = higham_gamma(checked_dim(
        "weeks_dft_ops",
        quad_m.checked_mul(6).and_then(|ops| ops.checked_add(8)),
    )?)?;
    let dft_roundoff = mul_up(gamma_dft, mul_up(max_phi_abs, r_inv_sum));
    let sample_noise_per_mode = add_up(max_sample_err, mul_up(gamma_dft, max_phi_abs));

    Ok(WeeksCirclePass {
        coeffs,
        eval_est,
        dft_roundoff,
        sample_noise_per_mode,
        min_re,
        max_im,
    })
}

/// Invert a complex Laplace transform `F(beta)` at `energy > 0` using Weeks's Möbius–Laguerre
/// expansion on the circle `|w| = r <= 1`.
///
/// Refuses with [`InverseLaplaceError::SlowAlgebraicTail`] when an observed dyadic
/// block fails the finite-resolution `n^{-3/2}` model (ratio `> 2^{-1/2}`).
pub fn weeks_invert<F>(
    energy: f64,
    config: WeeksConfig,
    mut eval_f: F,
) -> Result<WeeksResult, InverseLaplaceError>
where
    F: FnMut(Complex64) -> Result<ComplexSample, InverseLaplaceError>,
{
    require_positive_finite("energy", energy)?;
    require_positive_finite("scale_b", config.scale_b)?;
    require_positive_finite("circle_radius", config.circle_radius)?;
    require_positive_finite("max_imag_beta", config.max_imag_beta)?;
    require_positive_finite("tolerance", config.tolerance)?;
    if config.circle_radius > 1.0 {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "circle_radius",
            value: config.circle_radius,
        });
    }
    if !config.sigma.is_finite() || !config.min_re_beta.is_finite() {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "sigma",
            value: config.sigma,
        });
    }
    if config.laguerre_degree < 16 || config.laguerre_degree % 2 != 0 {
        return Err(InverseLaplaceError::InvalidParameter {
            parameter: "laguerre_degree",
            value: config.laguerre_degree as f64,
        });
    }

    let r = config.circle_radius;
    let re_min_analytic = config.sigma + config.scale_b * (1.0 - r) / (1.0 + r);
    if re_min_analytic <= config.min_re_beta {
        return Err(InverseLaplaceError::ContourCrossesCrossover {
            method: "weeks_contour",
            re_beta: re_min_analytic,
            beta_c: config.min_re_beta,
        });
    }
    if r < 1.0 {
        let im_max_analytic = 2.0 * config.scale_b * r / (1.0 - r * r);
        if im_max_analytic > config.max_imag_beta {
            return Err(InverseLaplaceError::ContourExceedsStrip {
                method: "weeks_contour",
                imag_beta: im_max_analytic,
                max_imag_beta: config.max_imag_beta,
            });
        }
    }

    let m = config.laguerre_degree;
    let fine_m = checked_dim("weeks_fine_degree", m.checked_mul(2))?;
    let coarse = weeks_dft_coefficients(&config, m, m, &mut eval_f)?;
    let fine = weeks_dft_coefficients(&config, fine_m, fine_m, &mut eval_f)?;

    let mut shift_l1 = 0.0_f64;
    for n in 0..m {
        shift_l1 = add_up(shift_l1, (fine.coeffs[n] - coarse.coeffs[n]).abs());
    }

    let q = m / 4;
    let mut prev_oct = 0.0_f64;
    let mut last_oct = 0.0_f64;
    for &c in &coarse.coeffs[2 * q..3 * q] {
        prev_oct = add_up(prev_oct, c.abs());
    }
    for &c in &coarse.coeffs[3 * q..m] {
        last_oct = add_up(last_oct, c.abs());
    }
    let coarse_floor = mul_up(16.0, add_up(coarse.dft_roundoff, coarse.eval_est));
    if last_oct > coarse_floor && last_oct > prev_oct {
        return Err(InverseLaplaceError::NonContractingRefinement {
            stage: "weeks_laguerre_tail",
            coarse_gap: prev_oct,
            fine_gap: last_oct,
        });
    }

    // Check dyadic decay rate on the fine 2M-point circle DFT:
    // For |c_n| ~ C n^{-p}, sum_{M..2M} |c_n| / sum_{M/2..M} |c_n| = 2^{1-p}.
    // An O(n^{-3/2}) tail (p >= 3/2) requires this dyadic ratio to be <= 2^{-1/2}.
    let mut fine_lower_dyadic = 0.0_f64;
    let mut fine_upper_dyadic = 0.0_f64;
    let mut fine_upper_noise_floor = 0.0_f64;
    for n in (m / 2)..m {
        fine_lower_dyadic = add_up(fine_lower_dyadic, fine.coeffs[n].abs());
    }
    for n in m..fine_m {
        fine_upper_dyadic = add_up(fine_upper_dyadic, fine.coeffs[n].abs());
        fine_upper_noise_floor = add_up(
            fine_upper_noise_floor,
            mul_up(4.0, mul_up(powi_inv_up(r, n), fine.sample_noise_per_mode)),
        );
    }
    let observed_dyadic_ratio = if fine_lower_dyadic > 0.0 {
        div_up(fine_upper_dyadic, fine_lower_dyadic)
    } else {
        0.0
    };
    if fine_upper_dyadic > fine_upper_noise_floor
        && observed_dyadic_ratio > WEEKS_MAX_DYADIC_TAIL_RATIO
    {
        return Err(InverseLaplaceError::SlowAlgebraicTail {
            stage: "weeks_fine_dyadic_tail",
            dyadic_ratio: observed_dyadic_ratio,
            max_dyadic_ratio: WEEKS_MAX_DYADIC_TAIL_RATIO,
        });
    }
    if r == 1.0 {
        let _ = verify_laguerre_tail_decay(&coarse.coeffs, coarse_floor)?;
    }

    let (series_coarse, series_round_coarse) =
        eval_orthonormal_laguerre_series(&coarse.coeffs, config.scale_b, energy)?;
    let (series_fine, series_round_fine) =
        eval_orthonormal_laguerre_series(&fine.coeffs[..m], config.scale_b, energy)?;

    let exp_sigma = (config.sigma * energy).exp();
    let exp_sigma_up = exp_sigma.next_up();
    let value = exp_sigma * series_fine;
    let value_coarse = exp_sigma * series_coarse;

    let sqrt_2b_up = mul_up(sqrt_up(2.0), sqrt_up(config.scale_b));
    let scale_factor = mul_up(exp_sigma_up, sqrt_2b_up);

    let tail_from_quarter = mul_up(scale_factor, mul_up(WEEKS_ALGEBRAIC_TAIL_FACTOR, last_oct));
    let tail_from_dyadic = if fine_upper_dyadic > fine_upper_noise_floor {
        mul_up(
            scale_factor,
            mul_up(WEEKS_DYADIC_TAIL_FACTOR, fine_upper_dyadic),
        )
    } else {
        0.0
    };
    let algebraic_tail_estimate = max_finite(
        tail_from_quarter,
        tail_from_dyadic,
        "weeks_algebraic_tail_estimate",
    )?;
    // `shift_l1` is a refinement diagnostic. It is stored on the result and
    // is not added to `estimate`, so a heuristic never enters an error envelope.
    let coefficient_shift = shift_l1;
    let discretization_estimate = (value - value_coarse).abs().next_up();
    let evaluation_estimate = mul_up(
        scale_factor,
        max_finite(coarse.eval_est, fine.eval_est, "weeks_eval_est")?,
    );
    let dft_round_max = max_finite(coarse.dft_roundoff, fine.dft_roundoff, "weeks_dft_roundoff")?;
    let series_round_max = max_finite(
        series_round_coarse,
        series_round_fine,
        "weeks_series_roundoff",
    )?;
    let roundoff_estimate = add_up(
        mul_up(scale_factor, dft_round_max),
        mul_up(exp_sigma_up, series_round_max),
    );
    let estimate = upward_sum(&[
        discretization_estimate,
        evaluation_estimate,
        roundoff_estimate,
        algebraic_tail_estimate,
    ])?;

    if estimate > config.tolerance {
        return Err(InverseLaplaceError::ToleranceNotMet {
            method: "weeks_contour",
            estimate,
            tolerance: config.tolerance,
        });
    }

    let min_re_sampled = if coarse.min_re < fine.min_re {
        coarse.min_re
    } else {
        fine.min_re
    };
    let max_imag_sampled = max_finite(coarse.max_im, fine.max_im, "weeks_max_imag_sampled")?;

    Ok(WeeksResult {
        value,
        accuracy: WeeksEstimate {
            estimate,
            discretization_estimate,
            coefficient_shift,
            evaluation_estimate,
            roundoff_estimate,
            algebraic_tail_estimate,
            observed_dyadic_ratio,
        },
        coefficients: fine.coeffs[..m].to_vec(),
        evaluations: checked_dim("weeks_evaluations", m.checked_mul(3))?,
        min_re_beta_sampled: min_re_sampled,
        max_imag_beta_sampled: max_imag_sampled,
    })
}
