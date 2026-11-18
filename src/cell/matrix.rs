//! Small dense matrix functions for the cell filter: the exponential,
//! its Frechet derivative, and the principal logarithm.
//!
//! Matrices are `[[f64; N]; N]`, row major. The exponential is the
//! scaling and squaring method with the degree 13 Pade approximant of
//! Higham, SIAM J. Matrix Anal. Appl. 26, 1179 (2005). The Frechet
//! derivative is the upper right block of the exponential of
//! `[[A, E], [0, A]]` (Higham, *Functions of Matrices*, eq. 10.43). The
//! logarithm is inverse scaling and squaring: square roots by the
//! Denman-Beavers iteration until `||X - I||_1` is inside the radius
//! where the 7 point Gauss-Legendre form of `log(I + Y)` is accurate to
//! working precision (Higham, ch. 11; Al-Mohy and Higham, SIAM J.
//! Matrix Anal. Appl. 34, 1 (2012), Table 2.1 gives that radius).

use crate::error::{Error, Result};

pub(crate) type Mat<const N: usize> = [[f64; N]; N];

/// `theta_13` of Higham 2005, Table 2.3 at unit roundoff `2^-53`.
const THETA_13: f64 = 5.371_920_351_148_152;

/// `theta_7` for the logarithm: `||Y||_1` below this keeps the 7 point
/// Gauss-Legendre error under unit roundoff.
const LOG_RADIUS: f64 = 0.264_2;

/// Gauss-Legendre points of the logarithm quadrature.
const LOG_POINTS: usize = 7;

/// Degree 13 Pade coefficients `b_0 .. b_13` (Higham 2005, eq. 2.12).
const PADE13: [f64; 14] = [
    64_764_752_532_480_000.0,
    32_382_376_266_240_000.0,
    7_771_770_303_897_600.0,
    1_187_353_796_428_800.0,
    129_060_195_264_000.0,
    10_559_470_521_600.0,
    670_442_572_800.0,
    33_522_128_640.0,
    1_323_241_920.0,
    40_840_800.0,
    960_960.0,
    16_380.0,
    182.0,
    1.0,
];

pub(crate) fn eye<const N: usize>() -> Mat<N> {
    let mut m = [[0.0; N]; N];
    for i in 0..N {
        m[i][i] = 1.0;
    }
    m
}

pub(crate) fn mul<const N: usize>(a: &Mat<N>, b: &Mat<N>) -> Mat<N> {
    let mut c = [[0.0; N]; N];
    for i in 0..N {
        for k in 0..N {
            let aik = a[i][k];
            for j in 0..N {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

pub(crate) fn transpose<const N: usize>(a: &Mat<N>) -> Mat<N> {
    let mut t = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn add<const N: usize>(a: &Mat<N>, b: &Mat<N>) -> Mat<N> {
    let mut c = *a;
    for i in 0..N {
        for j in 0..N {
            c[i][j] += b[i][j];
        }
    }
    c
}

fn sub<const N: usize>(a: &Mat<N>, b: &Mat<N>) -> Mat<N> {
    let mut c = *a;
    for i in 0..N {
        for j in 0..N {
            c[i][j] -= b[i][j];
        }
    }
    c
}

fn scale<const N: usize>(a: &Mat<N>, s: f64) -> Mat<N> {
    let mut c = *a;
    for row in &mut c {
        for v in row {
            *v *= s;
        }
    }
    c
}

/// Largest absolute column sum.
fn norm1<const N: usize>(a: &Mat<N>) -> f64 {
    let mut best = 0.0_f64;
    for j in 0..N {
        let mut s = 0.0;
        for row in a {
            s += row[j].abs();
        }
        best = best.max(s);
    }
    best
}

fn finite<const N: usize>(a: &Mat<N>) -> bool {
    a.iter().all(|row| row.iter().all(|v| v.is_finite()))
}

/// Solve `A X = B` by LU with partial pivoting. `None` when `A` is
/// singular to working precision.
pub(crate) fn solve<const N: usize>(a: &Mat<N>, b: &Mat<N>) -> Option<Mat<N>> {
    let mut lu = *a;
    let mut x = *b;
    for col in 0..N {
        let mut pivot = col;
        for row in col + 1..N {
            if lu[row][col].abs() > lu[pivot][col].abs() {
                pivot = row;
            }
        }
        if !(lu[pivot][col].is_finite() && lu[pivot][col] != 0.0) {
            return None;
        }
        lu.swap(col, pivot);
        x.swap(col, pivot);
        for row in col + 1..N {
            let f = lu[row][col] / lu[col][col];
            for k in col..N {
                lu[row][k] -= f * lu[col][k];
            }
            for k in 0..N {
                x[row][k] -= f * x[col][k];
            }
        }
    }
    for col in (0..N).rev() {
        for k in 0..N {
            let mut s = x[col][k];
            for j in col + 1..N {
                s -= lu[col][j] * x[j][k];
            }
            x[col][k] = s / lu[col][col];
        }
    }
    finite(&x).then_some(x)
}

pub(crate) fn inverse<const N: usize>(a: &Mat<N>) -> Option<Mat<N>> {
    solve(a, &eye::<N>())
}

pub(crate) fn det3(a: &Mat<3>) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Matrix exponential.
pub(crate) fn expm<const N: usize>(a: &Mat<N>) -> Result<Mat<N>> {
    if !finite(a) {
        return Err(Error::Cell {
            what: "matrix exponential of a non-finite matrix",
        });
    }
    let mut halvings = 0_u32;
    let mut size = norm1(a);
    while size > THETA_13 {
        size *= 0.5;
        halvings += 1;
    }
    let a = scale(a, 0.5_f64.powi(halvings as i32));
    let b = &PADE13;
    let id = eye::<N>();
    let a2 = mul(&a, &a);
    let a4 = mul(&a2, &a2);
    let a6 = mul(&a2, &a4);
    let mut inner_u = scale(&a6, b[13]);
    inner_u = add(&inner_u, &scale(&a4, b[11]));
    inner_u = add(&inner_u, &scale(&a2, b[9]));
    let mut poly_u = mul(&a6, &inner_u);
    poly_u = add(&poly_u, &scale(&a6, b[7]));
    poly_u = add(&poly_u, &scale(&a4, b[5]));
    poly_u = add(&poly_u, &scale(&a2, b[3]));
    poly_u = add(&poly_u, &scale(&id, b[1]));
    let u = mul(&a, &poly_u);
    let mut inner_v = scale(&a6, b[12]);
    inner_v = add(&inner_v, &scale(&a4, b[10]));
    inner_v = add(&inner_v, &scale(&a2, b[8]));
    let mut v = mul(&a6, &inner_v);
    v = add(&v, &scale(&a6, b[6]));
    v = add(&v, &scale(&a4, b[4]));
    v = add(&v, &scale(&a2, b[2]));
    v = add(&v, &scale(&id, b[0]));
    let Some(mut r) = solve(&sub(&v, &u), &add(&v, &u)) else {
        return Err(Error::Cell {
            what: "matrix exponential Pade denominator is singular",
        });
    };
    for _ in 0..halvings {
        r = mul(&r, &r);
    }
    if finite(&r) {
        Ok(r)
    } else {
        Err(Error::Cell {
            what: "matrix exponential overflowed",
        })
    }
}

/// Frechet derivative `L(A, E) = d/dt expm(A + t E)` at `t = 0`.
pub(crate) fn expm_frechet(a: &Mat<3>, e: &Mat<3>) -> Result<Mat<3>> {
    let mut block = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            block[i][j] = a[i][j];
            block[i + 3][j + 3] = a[i][j];
            block[i][j + 3] = e[i][j];
        }
    }
    let r = expm(&block)?;
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = r[i][j + 3];
        }
    }
    Ok(out)
}

/// Principal square root by the Denman-Beavers iteration.
fn sqrtm<const N: usize>(a: &Mat<N>) -> Result<Mat<N>> {
    let mut y = *a;
    let mut z = eye::<N>();
    let tol = 4.0 * f64::EPSILON;
    for _ in 0..DB_ITERATIONS {
        let (Some(yi), Some(zi)) = (inverse(&y), inverse(&z)) else {
            break;
        };
        let next_y = scale(&add(&y, &zi), 0.5);
        let next_z = scale(&add(&z, &yi), 0.5);
        let moved = norm1(&sub(&next_y, &y));
        y = next_y;
        z = next_z;
        if !finite(&y) {
            break;
        }
        if moved <= tol * norm1(&y) {
            return Ok(y);
        }
    }
    Err(Error::Cell {
        what: "square root of the deformation did not converge",
    })
}

/// Denman-Beavers sweeps before the square root is declared divergent.
/// The iteration converges quadratically, so a matrix with no negative
/// real eigenvalue needs a few dozen at most.
const DB_ITERATIONS: usize = 64;

/// Nodes and weights of the `m` point Gauss-Legendre rule on `[0, 1]`.
fn gauss_legendre_unit(m: usize) -> Vec<(f64, f64)> {
    let mut rule = Vec::with_capacity(m);
    for k in 0..m {
        let mut t = (std::f64::consts::PI * (k as f64 + 0.75) / (m as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..DB_ITERATIONS {
            let (mut p0, mut p1) = (1.0, t);
            for n in 2..=m {
                let p2 = ((2 * n - 1) as f64 * t * p1 - (n - 1) as f64 * p0) / n as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = m as f64 * (t * p1 - p0) / (t * t - 1.0);
            let dt = p1 / dp;
            t -= dt;
            if dt.abs() <= f64::EPSILON * t.abs().max(1.0) {
                break;
            }
        }
        let weight = 2.0 / ((1.0 - t * t) * dp * dp);
        rule.push((0.5 * (t + 1.0), 0.5 * weight));
    }
    rule
}

/// Principal matrix logarithm. Fails when `a` has an eigenvalue on the
/// closed negative real axis.
pub(crate) fn logm<const N: usize>(a: &Mat<N>) -> Result<Mat<N>> {
    if !finite(a) {
        return Err(Error::Cell {
            what: "matrix logarithm of a non-finite matrix",
        });
    }
    let id = eye::<N>();
    let mut x = *a;
    let mut roots = 0_u32;
    while norm1(&sub(&x, &id)) > LOG_RADIUS {
        x = sqrtm(&x)?;
        roots += 1;
    }
    let y = sub(&x, &id);
    let mut acc = [[0.0; N]; N];
    for (node, weight) in gauss_legendre_unit(LOG_POINTS) {
        let shifted = add(&id, &scale(&y, node));
        let Some(term) = solve(&shifted, &y) else {
            return Err(Error::Cell {
                what: "matrix logarithm quadrature is singular",
            });
        };
        acc = add(&acc, &scale(&term, weight));
    }
    let log = scale(&acc, 2.0_f64.powi(roots as i32));
    if finite(&log) {
        Ok(log)
    } else {
        Err(Error::Cell {
            what: "matrix logarithm overflowed",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close<const N: usize>(a: &Mat<N>, b: &Mat<N>, tol: f64) -> bool {
        norm1(&sub(a, b)) <= tol * (1.0 + norm1(b))
    }

    #[test]
    fn exponential_of_a_diagonal_matrix_is_elementwise() {
        let a = [[0.3, 0.0, 0.0], [0.0, -1.2, 0.0], [0.0, 0.0, 2.5]];
        let e = expm(&a).unwrap();
        for i in 0..3 {
            assert!((e[i][i] - a[i][i].exp()).abs() <= 8.0 * f64::EPSILON * a[i][i].exp());
        }
        assert_eq!(e[0][1], 0.0);
    }

    #[test]
    fn exponential_of_a_rotation_generator_is_a_rotation() {
        let theta = 0.7_f64;
        let a = [[0.0, -theta, 0.0], [theta, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let e = expm(&a).unwrap();
        let want = [
            [theta.cos(), -theta.sin(), 0.0],
            [theta.sin(), theta.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert!(close(&e, &want, 16.0 * f64::EPSILON));
    }

    #[test]
    fn exponential_scales_and_squares_a_large_matrix() {
        let a = [[7.0, 1.0, 0.0], [0.0, -3.0, 2.0], [0.5, 0.0, 6.0]];
        let half = expm(&scale(&a, 0.5)).unwrap();
        let full = expm(&a).unwrap();
        assert!(close(&mul(&half, &half), &full, 1.0e3 * f64::EPSILON));
    }

    #[test]
    fn logarithm_inverts_the_exponential() {
        let a = [[0.05, 0.02, -0.01], [0.0, -0.03, 0.04], [0.01, 0.0, 0.02]];
        let back = logm(&expm(&a).unwrap()).unwrap();
        assert!(close(&back, &a, 64.0 * f64::EPSILON));
        let big = [[0.9, 0.3, 0.0], [-0.2, 0.5, 0.1], [0.0, 0.1, -0.4]];
        let back = logm(&expm(&big).unwrap()).unwrap();
        assert!(close(&back, &big, 1.0e3 * f64::EPSILON));
    }

    #[test]
    fn logarithm_of_the_identity_is_exactly_zero() {
        assert_eq!(logm(&eye::<3>()).unwrap(), [[0.0; 3]; 3]);
    }

    #[test]
    fn logarithm_refuses_a_matrix_with_negative_eigenvalue() {
        let a = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(logm(&a).is_err());
    }

    #[test]
    fn frechet_derivative_matches_a_central_difference() {
        let a = [[0.1, 0.2, 0.0], [-0.1, 0.05, 0.3], [0.0, 0.1, -0.2]];
        let e = [[0.0, 1.0, 0.0], [0.5, 0.0, -1.0], [0.0, 0.25, 1.0]];
        let l = expm_frechet(&a, &e).unwrap();
        let h = 1.0e-5;
        let plus = expm(&add(&a, &scale(&e, h))).unwrap();
        let minus = expm(&sub(&a, &scale(&e, h))).unwrap();
        let fd = scale(&sub(&plus, &minus), 0.5 / h);
        assert!(close(&l, &fd, 1.0e-8));
    }

    #[test]
    fn frechet_derivative_of_commuting_arguments_is_the_product() {
        let a = [[0.4, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.1]];
        let l = expm_frechet(&a, &a).unwrap();
        let want = mul(&expm(&a).unwrap(), &a);
        assert!(close(&l, &want, 16.0 * f64::EPSILON));
    }

    #[test]
    fn frechet_derivative_is_adjoint_under_transpose() {
        let a = [[0.1, 0.2, 0.0], [-0.1, 0.05, 0.3], [0.0, 0.1, -0.2]];
        let g = [[1.0, -0.5, 0.2], [0.3, 0.7, -1.0], [0.1, 0.0, 0.4]];
        let adj = expm_frechet(&transpose(&a), &g).unwrap();
        for m in 0..3 {
            for n in 0..3 {
                let mut dir = [[0.0; 3]; 3];
                dir[m][n] = 1.0;
                let l = expm_frechet(&a, &dir).unwrap();
                let mut dot = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        dot += l[i][j] * g[i][j];
                    }
                }
                assert!((dot - adj[m][n]).abs() <= 64.0 * f64::EPSILON * (1.0 + dot.abs()));
            }
        }
    }

    #[test]
    fn gauss_legendre_rule_integrates_polynomials_exactly() {
        let rule = gauss_legendre_unit(LOG_POINTS);
        for power in 0..(2 * LOG_POINTS - 1) {
            let got: f64 = rule.iter().map(|&(x, w)| w * x.powi(power as i32)).sum();
            let want = 1.0 / (power as f64 + 1.0);
            assert!((got - want).abs() <= 8.0 * f64::EPSILON, "power {power}");
        }
    }
}
