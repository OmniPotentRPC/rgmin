/// Rotate `mobile` onto `reference` and return the mass-weighted RMSD.
///
/// Any periodic axis skips the rotation and shifts `mobile` by the mean
/// displacement. The host unwraps coordinates before the call. With no
/// periodic axis the rotation is the Horn / Coutsias quaternion. A length
/// mismatch returns NaN.
pub fn align_rigid(
    mobile: &mut [f64],
    reference: &[f64],
    masses: Option<&[f64]>,
    pbc: [bool; 3],
) -> f64 {
    let n = mobile.len();
    if n != reference.len() || n < 3 || n % 3 != 0 {
        return f64::NAN;
    }
    let nat = n / 3;
    let mass_ok = masses.is_some_and(|m| m.len() == nat);
    let mi = |i: usize| -> f64 {
        if mass_ok {
            masses.unwrap()[i].max(0.0)
        } else {
            1.0
        }
    };
    let mut mtot = 0.0;
    let mut cm = [0.0; 3];
    let mut cr = [0.0; 3];
    for i in 0..nat {
        let m = mi(i);
        mtot += m;
        for d in 0..3 {
            cm[d] += m * mobile[3 * i + d];
            cr[d] += m * reference[3 * i + d];
        }
    }
    if mtot > 0.0 {
        for d in 0..3 {
            cm[d] /= mtot;
            cr[d] /= mtot;
        }
    }
    let rotate = !(pbc[0] || pbc[1] || pbc[2]);
    let rot = if rotate {
        let mut cov = [[0.0; 3]; 3];
        for i in 0..nat {
            let m = mi(i);
            for a in 0..3 {
                for b in 0..3 {
                    cov[a][b] += m * (mobile[3 * i + a] - cm[a]) * (reference[3 * i + b] - cr[b]);
                }
            }
        }
        let (eval, evec) = jacobi4(key_matrix(&cov));
        let mut best = 0;
        for i in 1..4 {
            if eval[i] > eval[best] {
                best = i;
            }
        }
        let q = [evec[0][best], evec[1][best], evec[2][best], evec[3][best]];
        quat_to_rot(q)
    } else {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    };
    let mut err = 0.0;
    for i in 0..nat {
        let p = [
            mobile[3 * i] - cm[0],
            mobile[3 * i + 1] - cm[1],
            mobile[3 * i + 2] - cm[2],
        ];
        let mut y = [0.0; 3];
        for a in 0..3 {
            y[a] = rot[a][0] * p[0] + rot[a][1] * p[1] + rot[a][2] * p[2];
            mobile[3 * i + a] = y[a] + cr[a];
        }
        for a in 0..3 {
            let d = y[a] - (reference[3 * i + a] - cr[a]);
            err += mi(i) * d * d;
        }
    }
    if mtot > 0.0 {
        (err / mtot).max(0.0).sqrt()
    } else {
        0.0
    }
}

fn key_matrix(c: &[[f64; 3]; 3]) -> [[f64; 4]; 4] {
    let (xx, xy, xz) = (c[0][0], c[0][1], c[0][2]);
    let (yx, yy, yz) = (c[1][0], c[1][1], c[1][2]);
    let (zx, zy, zz) = (c[2][0], c[2][1], c[2][2]);
    [
        [xx + yy + zz, yz - zy, zx - xz, xy - yx],
        [yz - zy, xx - yy - zz, xy + yx, zx + xz],
        [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
        [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
    ]
}

fn quat_to_rot(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n == 0.0 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let (w, x, y, z) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Largest-eigenvector Jacobi on a symmetric 4x4. Eigenvectors are columns.
fn jacobi4(mut a: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for _ in 0..32 {
        let mut p = 0;
        let mut q = 1;
        let mut big = 0.0;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let aij = a[i][j].abs();
                if aij > big {
                    big = aij;
                    p = i;
                    q = j;
                }
            }
        }
        if big < 1.0e-15 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for i in 0..4 {
            if i == p || i == q {
                continue;
            }
            let aip = a[i][p];
            let aiq = a[i][q];
            a[i][p] = c * aip - s * aiq;
            a[p][i] = a[i][p];
            a[i][q] = s * aip + c * aiq;
            a[q][i] = a[i][q];
        }
        for i in 0..4 {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }
    ([a[0][0], a[1][1], a[2][2], a[3][3]], v)
}
