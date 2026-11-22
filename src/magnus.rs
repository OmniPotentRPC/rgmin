//! Sixth-order Magnus steps for linear systems `y' = A(t) y`.
//!
//! One step of length `h` samples `A` at the three Gauss-Legendre nodes
//! `t + c_i h`, `c = 1/2 - √15/10, 1/2, 1/2 + √15/10`, and forms
//!
//! ```text
//! α1 = h A2,   α2 = (√15 h / 3)(A3 - A1),   α3 = (10 h / 3)(A3 - 2 A2 + A1)
//! C1 = [α1, α2],   C2 = -(1/60) [α1, 2 α3 + C1]
//! Ω6 = α1 + α3 / 12 + (1/240) [-20 α1 - α3 + C1, α2 + C2]
//! Ω4 = α1 + α3 / 12 - C1 / 12
//! ```
//!
//! (Blanes, Casas and Ros, BIT 40 (2000) 434; Blanes, Casas, Oteo and Ros,
//! Phys. Rep. 470 (2009) 151, section 5). `exp(Ω6)` is the step map to
//! `O(h^7)` locally and `exp(Ω4)` to `O(h^5)`; both use the same three
//! samples, so their difference costs no further evaluations of `A`.
//!
//! The exponential is Higham's scaling and squaring (SIAM J. Matrix Anal.
//! Appl. 26 (2005) 1179): the lowest Padé degree of 3, 5, 7 and 9 whose
//! `θ` bounds the 1-norm, with two to five products and one linear solve;
//! above `θ9`, scale by `2^{-s}` until the 1-norm is at most `θ13`,
//! evaluate the degree-13 approximant with six products and one solve,
//! and square `s` times.

use thiserror::Error;

/// Refusal of a Magnus step or a matrix exponential.
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum MagnusError {
    /// A sample has the wrong length, the dimension is zero, or the step is
    /// not positive and finite.
    #[error("Magnus shape: {0}")]
    Shape(&'static str),
    /// An input or intermediate entry is NaN or infinite.
    #[error("a Magnus entry is not finite")]
    NonFinite,
    /// The Padé denominator has an exactly zero pivot.
    #[error("the Padé denominator is singular")]
    Singular,
}

/// Exponentials of the sixth- and embedded fourth-order Magnus exponents.
///
/// The struct is non-exhaustive, so a field added later is not a breaking
/// change. [`magnus_step`] builds it.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MagnusStep {
    /// `exp(Ω6)`, row-major `m x m`.
    pub sixth: Vec<f64>,
    /// `exp(Ω4)`, row-major `m x m`.
    pub fourth: Vec<f64>,
    /// `m x m` products spent on `exp(Ω6)` and its exponent, with the Padé
    /// solve counted as three products.
    pub sixth_products: u64,
}

/// A matrix exponential and the `m x m` products it spent.
///
/// The struct is non-exhaustive, so a field added later is not a breaking
/// change. [`expm`] builds it.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Exponential {
    /// Row-major `m x m`.
    pub matrix: Vec<f64>,
    /// Padé products, three for the solve, and one per squaring.
    pub products: u64,
}

/// `(θ_d, b)` of the Padé degrees `d = 3, 5, 7, 9`, Higham (2005) table 2.1
/// and the coefficients of `p_d(x) = Σ b_j x^j`.
const LOW_DEGREES: [(f64, &[f64]); 4] = [
    (1.495585217958292e-2, &[120.0, 60.0, 12.0, 1.0]),
    (
        2.53939833006323e-1,
        &[30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0],
    ),
    (
        9.504178996162932e-1,
        &[
            17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0,
        ],
    ),
    (
        2.097847961257068e0,
        &[
            17643225600.0,
            8821612800.0,
            2075673600.0,
            302702400.0,
            30270240.0,
            2162160.0,
            110880.0,
            3960.0,
            90.0,
            1.0,
        ],
    ),
];

/// Unscaled Padé approximant of degree `b.len() - 1`, valid while `‖a‖₁`
/// is within that degree's `θ`.
fn low_degree_pade(a: &[f64], m: usize, b: &[f64]) -> Result<Exponential, MagnusError> {
    let half = b.len() / 2;
    let x2 = multiply(a, a, m);
    let mut powers = vec![identity(m), x2];
    while powers.len() < half {
        let next = multiply(&powers[powers.len() - 1], &powers[1], m);
        powers.push(next);
    }
    let mut odd = vec![0.0; m * m];
    let mut v = vec![0.0; m * m];
    for (j, power) in powers.iter().enumerate() {
        for k in 0..m * m {
            odd[k] += b[2 * j + 1] * power[k];
            v[k] += b[2 * j] * power[k];
        }
    }
    let u = multiply(a, &odd, m);
    let denominator: Vec<f64> = v.iter().zip(&u).map(|(p, q)| p - q).collect();
    let numerator: Vec<f64> = v.iter().zip(&u).map(|(p, q)| p + q).collect();
    let result = solve(denominator, numerator, m)?;
    if result.iter().any(|value| !value.is_finite()) {
        return Err(MagnusError::NonFinite);
    }
    Ok(Exponential {
        matrix: result,
        products: half as u64 + 3,
    })
}

/// Gauss-Legendre fractions `c_i` of a step, in increasing order.
pub fn gauss_legendre_three() -> [f64; 3] {
    let offset = 0.15_f64.sqrt();
    [0.5 - offset, 0.5, 0.5 + offset]
}

/// Exponentials of `Ω6` and `Ω4` from `A` at the three Gauss-Legendre nodes.
///
/// `samples[i]` is `A(t + c_i h)` with `c` from [`gauss_legendre_three`],
/// row-major `m x m`.
pub fn magnus_step(samples: [&[f64]; 3], h: f64, m: usize) -> Result<MagnusStep, MagnusError> {
    if m == 0 {
        return Err(MagnusError::Shape("dimension is zero"));
    }
    if !(h > 0.0 && h.is_finite()) {
        return Err(MagnusError::Shape("the step is not positive and finite"));
    }
    let size = m
        .checked_mul(m)
        .ok_or(MagnusError::Shape("dimension overflows"))?;
    if samples.iter().any(|sample| sample.len() != size) {
        return Err(MagnusError::Shape("a sample is not m x m"));
    }
    if samples
        .iter()
        .any(|sample| sample.iter().any(|v| !v.is_finite()))
    {
        return Err(MagnusError::NonFinite);
    }
    let [a1, a2, a3] = samples;
    let root = 15.0_f64.sqrt();
    let alpha1: Vec<f64> = a2.iter().map(|v| h * v).collect();
    let alpha2: Vec<f64> = a3
        .iter()
        .zip(a1)
        .map(|(late, early)| root * h / 3.0 * (late - early))
        .collect();
    let alpha3: Vec<f64> = (0..size)
        .map(|k| 10.0 * h / 3.0 * (a3[k] - 2.0 * a2[k] + a1[k]))
        .collect();
    let c1 = commutator(&alpha1, &alpha2, m);
    let inner: Vec<f64> = alpha3.iter().zip(&c1).map(|(a, c)| 2.0 * a + c).collect();
    let c2: Vec<f64> = commutator(&alpha1, &inner, m)
        .iter()
        .map(|v| -v / 60.0)
        .collect();
    let left: Vec<f64> = (0..size)
        .map(|k| -20.0 * alpha1[k] - alpha3[k] + c1[k])
        .collect();
    let right: Vec<f64> = alpha2.iter().zip(&c2).map(|(a, c)| a + c).collect();
    let outer = commutator(&left, &right, m);
    let sixth: Vec<f64> = (0..size)
        .map(|k| alpha1[k] + alpha3[k] / 12.0 + outer[k] / 240.0)
        .collect();
    let fourth: Vec<f64> = (0..size)
        .map(|k| alpha1[k] + alpha3[k] / 12.0 - c1[k] / 12.0)
        .collect();
    let sixth = expm(&sixth, m)?;
    let fourth = expm(&fourth, m)?;
    Ok(MagnusStep {
        sixth: sixth.matrix,
        fourth: fourth.matrix,
        sixth_products: sixth.products + 6,
    })
}

/// `exp(a)` for a row-major `m x m` matrix.
pub fn expm(a: &[f64], m: usize) -> Result<Exponential, MagnusError> {
    const THETA13: f64 = 5.371920351148152;
    const B: [f64; 14] = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ];
    if m == 0 {
        return Err(MagnusError::Shape("dimension is zero"));
    }
    if a.len()
        != m.checked_mul(m)
            .ok_or(MagnusError::Shape("dimension overflows"))?
    {
        return Err(MagnusError::Shape("the matrix is not m x m"));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(MagnusError::NonFinite);
    }
    let norm = norm1(a, m);
    if !norm.is_finite() {
        return Err(MagnusError::NonFinite);
    }
    if let Some(&(_, b)) = LOW_DEGREES.iter().find(|(theta, _)| norm <= *theta) {
        return low_degree_pade(a, m, b);
    }
    let mut squarings = 0i32;
    let mut scaled = norm;
    while scaled > THETA13 {
        scaled *= 0.5;
        squarings += 1;
    }
    let factor = 2.0_f64.powi(-squarings);
    let x: Vec<f64> = a.iter().map(|v| v * factor).collect();
    let x2 = multiply(&x, &x, m);
    let x4 = multiply(&x2, &x2, m);
    let x6 = multiply(&x4, &x2, m);
    let eye = identity(m);
    let combine = |w: [f64; 4], terms: [&[f64]; 4]| -> Vec<f64> {
        (0..m * m)
            .map(|k| {
                w[0] * terms[0][k] + w[1] * terms[1][k] + w[2] * terms[2][k] + w[3] * terms[3][k]
            })
            .collect()
    };
    let high_u = combine([B[13], B[11], B[9], 0.0], [&x6, &x4, &x2, &eye]);
    let low_u = combine([B[7], B[5], B[3], B[1]], [&x6, &x4, &x2, &eye]);
    let mut inner_u = multiply(&x6, &high_u, m);
    inner_u
        .iter_mut()
        .zip(&low_u)
        .for_each(|(v, low)| *v += low);
    let u = multiply(&x, &inner_u, m);
    let high_v = combine([B[12], B[10], B[8], 0.0], [&x6, &x4, &x2, &eye]);
    let low_v = combine([B[6], B[4], B[2], B[0]], [&x6, &x4, &x2, &eye]);
    let mut v = multiply(&x6, &high_v, m);
    v.iter_mut()
        .zip(&low_v)
        .for_each(|(value, low)| *value += low);
    let denominator: Vec<f64> = v.iter().zip(&u).map(|(p, q)| p - q).collect();
    let numerator: Vec<f64> = v.iter().zip(&u).map(|(p, q)| p + q).collect();
    let mut result = solve(denominator, numerator, m)?;
    for _ in 0..squarings {
        result = multiply(&result, &result, m);
    }
    if result.iter().any(|v| !v.is_finite()) {
        return Err(MagnusError::NonFinite);
    }
    Ok(Exponential {
        matrix: result,
        products: 9 + squarings as u64,
    })
}

fn commutator(left: &[f64], right: &[f64], m: usize) -> Vec<f64> {
    let forward = multiply(left, right, m);
    let backward = multiply(right, left, m);
    forward.iter().zip(&backward).map(|(f, b)| f - b).collect()
}

fn multiply(left: &[f64], right: &[f64], m: usize) -> Vec<f64> {
    let mut product = vec![0.0; m * m];
    for row in 0..m {
        for inner in 0..m {
            let scale = left[row * m + inner];
            for column in 0..m {
                product[row * m + column] += scale * right[inner * m + column];
            }
        }
    }
    product
}

fn identity(m: usize) -> Vec<f64> {
    let mut eye = vec![0.0; m * m];
    for k in 0..m {
        eye[k * m + k] = 1.0;
    }
    eye
}

fn norm1(a: &[f64], m: usize) -> f64 {
    (0..m)
        .map(|column| (0..m).map(|row| a[row * m + column].abs()).sum::<f64>())
        .fold(
            0.0,
            |acc, col| {
                if col > acc || col.is_nan() { col } else { acc }
            },
        )
}

/// `a^{-1} b` by partial-pivoting LU, both row-major `m x m`.
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, m: usize) -> Result<Vec<f64>, MagnusError> {
    for column in 0..m {
        let mut pivot = column;
        for row in column + 1..m {
            if a[row * m + column].abs() > a[pivot * m + column].abs() {
                pivot = row;
            }
        }
        if a[pivot * m + column] == 0.0 {
            return Err(MagnusError::Singular);
        }
        if pivot != column {
            for k in 0..m {
                a.swap(column * m + k, pivot * m + k);
                b.swap(column * m + k, pivot * m + k);
            }
        }
        let diagonal = a[column * m + column];
        for row in column + 1..m {
            let factor = a[row * m + column] / diagonal;
            if factor == 0.0 {
                continue;
            }
            for k in column..m {
                a[row * m + k] -= factor * a[column * m + k];
            }
            for k in 0..m {
                b[row * m + k] -= factor * b[column * m + k];
            }
        }
    }
    for column in (0..m).rev() {
        let diagonal = a[column * m + column];
        for k in 0..m {
            let mut value = b[column * m + k];
            for inner in column + 1..m {
                value -= a[column * m + inner] * b[inner * m + k];
            }
            b[column * m + k] = value / diagonal;
        }
    }
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_gap(left: &[f64], right: &[f64]) -> f64 {
        assert_eq!(left.len(), right.len(), "max_gap lengths differ");
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(
                0.0,
                |acc, gap| {
                    if gap > acc || gap.is_nan() { gap } else { acc }
                },
            )
    }

    #[test]
    fn the_exponential_matches_closed_forms() {
        for t in [1.0e-3, 0.01, 0.2, 0.7, 0.9, 2.0, 3.0, 5.0, 40.0] {
            let rotation = expm(&[0.0, t, -t, 0.0], 2).expect("rotation");
            let exact = [t.cos(), t.sin(), -t.sin(), t.cos()];
            assert!(max_gap(&rotation.matrix, &exact) <= 64.0 * f64::EPSILON * (1.0 + t));
            let boost = expm(&[0.0, t, t, 0.0], 2).expect("boost");
            let exact = [t.cosh(), t.sinh(), t.sinh(), t.cosh()];
            let gap = max_gap(&boost.matrix, &exact);
            assert!(
                gap <= 64.0 * f64::EPSILON * t.cosh() * (1.0 + t),
                "t {t} gap {gap}"
            );
        }
        let nilpotent = expm(&[0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0], 3).expect("nilpotent");
        let exact = [1.0, 2.0, 3.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0];
        assert!(max_gap(&nilpotent.matrix, &exact) <= 8.0 * f64::EPSILON);
    }

    #[test]
    fn the_pade_degree_follows_the_norm() {
        let products: Vec<u64> = [0.01, 0.2, 0.9, 2.0, 5.0, 20.0]
            .iter()
            .map(|t| expm(&[0.0, *t, -*t, 0.0], 2).expect("rotation").products)
            .collect();
        assert_eq!(products, vec![5, 6, 7, 8, 9, 11]);
    }

    #[test]
    fn a_constant_generator_is_one_exponential() {
        let a = [0.3, 1.0, -0.2, 0.1, 0.0, 0.5, 2.0, -1.0, 0.25];
        let h = 0.4;
        let step = magnus_step([&a, &a, &a], h, 3).expect("step");
        let scaled: Vec<f64> = a.iter().map(|v| h * v).collect();
        let exact = expm(&scaled, 3).expect("exp");
        assert!(max_gap(&step.sixth, &exact.matrix) <= 16.0 * f64::EPSILON);
        assert!(max_gap(&step.fourth, &exact.matrix) <= 16.0 * f64::EPSILON);
    }

    fn mathieu(t: f64) -> [f64; 16] {
        let w = [
            1.0 + 0.6 * t.cos(),
            0.4 * (2.0 * t).sin(),
            0.4 * (2.0 * t).sin(),
            -0.5 + 0.3 * t.sin(),
        ];
        [
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            w[0], w[1], 0.0, 0.0, //
            w[2], w[3], 0.0, 0.0,
        ]
    }

    fn noncommuting_three(t: f64) -> [f64; 9] {
        [
            -0.2 * t.cos(),
            0.9 + 0.5 * t.sin(),
            -0.4 * (2.0 * t).cos(),
            -0.9 + 0.3 * (2.0 * t).cos(),
            0.15 * (3.0 * t).sin(),
            0.7 + 0.6 * t.cos(),
            0.5 * (2.0 * t).sin(),
            -0.7 + 0.4 * t.sin(),
            -0.1 * (3.0 * t).cos(),
        ]
    }

    fn rk4_reference<const N: usize>(
        generator: fn(f64) -> [f64; N],
        m: usize,
        end: f64,
        steps: usize,
    ) -> Vec<f64> {
        assert_eq!(m * m, N);
        let h = end / steps as f64;
        let mut y = identity(m);
        for step in 0..steps {
            let t = step as f64 * h;
            let f = |t: f64, y: &[f64]| multiply(&generator(t), y, m);
            let k1 = f(t, &y);
            let y2: Vec<f64> = y.iter().zip(&k1).map(|(v, k)| v + 0.5 * h * k).collect();
            let k2 = f(t + 0.5 * h, &y2);
            let y3: Vec<f64> = y.iter().zip(&k2).map(|(v, k)| v + 0.5 * h * k).collect();
            let k3 = f(t + 0.5 * h, &y3);
            let y4: Vec<f64> = y.iter().zip(&k3).map(|(v, k)| v + h * k).collect();
            let k4 = f(t + h, &y4);
            for k in 0..m * m {
                y[k] += h / 6.0 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
            }
        }
        y
    }

    fn magnus_flow<const N: usize>(
        generator: fn(f64) -> [f64; N],
        m: usize,
        end: f64,
        steps: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(m * m, N);
        let h = end / steps as f64;
        let c = gauss_legendre_three();
        let mut sixth = identity(m);
        let mut fourth = identity(m);
        for step in 0..steps {
            let t = step as f64 * h;
            let samples = c.map(|ci| generator(t + ci * h));
            let local = magnus_step([&samples[0], &samples[1], &samples[2]], h, m).expect("step");
            sixth = multiply(&local.sixth, &sixth, m);
            fourth = multiply(&local.fourth, &fourth, m);
        }
        (sixth, fourth)
    }

    fn fitted_order(samples: &[(usize, f64)]) -> f64 {
        assert!(samples.len() >= 2);
        let xs: Vec<f64> = samples
            .iter()
            .map(|&(steps, _)| (steps as f64).log2())
            .collect();
        let ys: Vec<f64> = samples
            .iter()
            .map(
                |&(_, error)| {
                    if error > 0.0 { -error.log2() } else { f64::NAN }
                },
            )
            .collect();
        let count = samples.len() as f64;
        let mean_x = xs.iter().sum::<f64>() / count;
        let mean_y = ys.iter().sum::<f64>() / count;
        let numerator: f64 = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let denominator: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
        numerator / denominator
    }

    fn assert_convergence_orders<const N: usize>(
        name: &str,
        generator: fn(f64) -> [f64; N],
        m: usize,
        end: f64,
    ) {
        let reference = rk4_reference(generator, m, end, 1 << 15);
        let mut sixth_errors = Vec::new();
        let mut fourth_errors = Vec::new();
        for steps in [6usize, 12, 24, 48] {
            let (sixth, fourth) = magnus_flow(generator, m, end, steps);
            sixth_errors.push((steps, max_gap(&sixth, &reference)));
            fourth_errors.push((steps, max_gap(&fourth, &reference)));
        }
        let order_six = fitted_order(&sixth_errors);
        let order_four = fitted_order(&fourth_errors);
        assert!(
            (5.8..=6.2).contains(&order_six),
            "{name} sixth-order slope {order_six} from {sixth_errors:?}"
        );
        assert!(
            (3.8..=4.2).contains(&order_four),
            "{name} fourth-order slope {order_four} from {fourth_errors:?}"
        );
    }

    /// Global error slopes over step halvings on the 4 x 4 Hill system and on a
    /// non-commuting 3 x 3 generator where `[α2, α3] ≠ 0`.
    #[test]
    fn the_sixth_and_fourth_order_exponents_converge_at_their_orders() {
        assert_convergence_orders("mathieu", mathieu, 4, 3.0);
        assert_convergence_orders("noncommuting_three", noncommuting_three, 3, 3.0);
    }

    #[test]
    fn max_gap_propagates_nan() {
        assert!(max_gap(&[f64::NAN, 0.0], &[0.0, 1.0]).is_nan());
        assert!(max_gap(&[1.0, 0.0], &[0.0, f64::NAN]).is_nan());
    }

    #[test]
    #[should_panic(expected = "max_gap lengths differ")]
    fn max_gap_rejects_mismatched_lengths() {
        let _ = max_gap(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn malformed_steps_are_refused() {
        let a = [0.0; 4];
        assert_eq!(
            magnus_step([&a, &a, &a[..3]], 0.1, 2).unwrap_err(),
            MagnusError::Shape("a sample is not m x m")
        );
        assert!(matches!(
            magnus_step([&a, &a, &a], 0.0, 2),
            Err(MagnusError::Shape(_))
        ));
        let bad = [f64::NAN, 0.0, 0.0, 0.0];
        assert_eq!(
            magnus_step([&a, &bad, &a], 0.1, 2).unwrap_err(),
            MagnusError::NonFinite
        );
        assert_eq!(expm(&bad, 2).unwrap_err(), MagnusError::NonFinite);
        let huge_nan = [f64::NAN, 0.0, 0.0, 1.0e308];
        assert!(norm1(&huge_nan, 2).is_nan());
        assert!(norm1(&[1.0e308, 0.0, 0.0, f64::NAN], 2).is_nan());
        assert_eq!(expm(&huge_nan, 2).unwrap_err(), MagnusError::NonFinite);
        assert_eq!(
            expm(&[f64::INFINITY, 0.0, 0.0, 0.0], 2).unwrap_err(),
            MagnusError::NonFinite
        );
    }
}
