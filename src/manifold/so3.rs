//! SO(3) as a 9-vector, row-major. manopt_cpp `Rotation<3,1>`.
//!
//! Dimension is exactly 9. A 3N cluster is
//! [`super::RigidQuotient`], not this embedding.

use ndarray::Array1;

use super::Manifold;

/// Rotation matrices packed row-major length 9.
#[derive(Clone, Copy, Debug, Default)]
pub struct So3;

fn unpack(x: &Array1<f64>) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    if x.len() >= 9 {
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = x[3 * i + j];
            }
        }
    }
    r
}

fn pack(r: [[f64; 3]; 3]) -> Array1<f64> {
    Array1::from_shape_vec(9, {
        let mut v = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                v.push(r[i][j]);
            }
        }
        v
    })
    .unwrap()
}

fn mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn skew_of(s: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut k = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            k[i][j] = 0.5 * (s[i][j] - s[j][i]);
        }
    }
    k
}

/// Polar-ish retraction: SVD-free QR with positive diagonal.
fn qr_pos(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Gram-Schmidt columns.
    let mut q = [[0.0; 3]; 3];
    for j in 0..3 {
        let mut v = [a[0][j], a[1][j], a[2][j]];
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..3 {
                dot += q[i][k] * a[i][j];
            }
            for i in 0..3 {
                v[i] -= dot * q[i][k];
            }
        }
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n > 1e-16 {
            for i in 0..3 {
                q[i][j] = v[i] / n;
            }
        }
    }
    let det = q[0][0] * (q[1][1] * q[2][2] - q[1][2] * q[2][1])
        - q[0][1] * (q[1][0] * q[2][2] - q[1][2] * q[2][0])
        + q[0][2] * (q[1][0] * q[2][1] - q[1][1] * q[2][0]);
    if det < 0.0 {
        for i in 0..3 {
            q[i][2] = -q[i][2];
        }
    }
    q
}

impl Manifold for So3 {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n == 9 { Ok(()) } else { Err(9) }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != 9 || v.len() != 9 {
            return v.clone();
        }
        let r = unpack(x);
        let h = unpack(v);
        let rt_h = mul(transpose(r), h);
        let omega = skew_of(rt_h);
        pack(mul(r, omega))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != 9 || v.len() != 9 {
            return x + v;
        }
        let r = unpack(x);
        let u = unpack(v);
        let mut y = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                y[i][j] = r[i][j] + u[i][j];
            }
        }
        pack(qr_pos(y))
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identity_plus_skew_stays_orthogonal() {
        let x = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let v = array![0.0, -0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y = So3.retract(&x, &v);
        let r = unpack(&y);
        let rtr = mul(transpose(r), r);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[i][j] - want).abs() < 1e-12, "{rtr:?}");
            }
        }
    }

    #[test]
    fn wrong_dim_does_not_shrink() {
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = So3.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(So3.project(&x, &v).len(), 114);
        assert!(So3.required_dim(114).is_err());
        assert!(So3.required_dim(9).is_ok());
    }

    /// A length other than 9 is the ambient map on every component.
    /// The vector is not read as a 9-prefix of a longer packing.
    #[test]
    fn wrong_length_is_the_ambient_map() {
        let x = Array1::from_elem(10, 0.2);
        let v = Array1::from_elem(10, -0.05);
        let y = So3.retract(&x, &v);
        let p = So3.project(&x, &v);
        assert_eq!(y.len(), 10);
        assert_eq!(p.len(), 10);
        for i in 0..10 {
            assert!(
                (y[i] - (x[i] + v[i])).abs() < 1e-15,
                "retract component {i}"
            );
            assert!((p[i] - v[i]).abs() < 1e-15, "project component {i}");
        }
        assert_eq!(So3.required_dim(10), Err(9));
    }
}
