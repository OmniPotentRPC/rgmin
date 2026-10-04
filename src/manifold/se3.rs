//! SE(3): row-major SO(3) (9) then translation (3). Length 12.
//!
//! Dimension is exactly 12. A 3N cluster is
//! [`super::RigidQuotient`], not a 12-vector prefix.

use ndarray::Array1;

use super::{Manifold, so3::So3};

/// Rigid motions. Rotation block uses [`So3`]; translation is Euclidean.
#[derive(Clone, Copy, Debug, Default)]
pub struct Se3;

impl Manifold for Se3 {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n == 12 { Ok(()) } else { Err(12) }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != 12 || v.len() != 12 {
            return v.clone();
        }
        let xr = x.slice(ndarray::s![0..9]).to_owned();
        let vr = v.slice(ndarray::s![0..9]).to_owned();
        let pr = So3.project(&xr, &vr);
        let mut out = v.clone();
        for i in 0..9 {
            out[i] = pr[i];
        }
        out
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != 12 || v.len() != 12 {
            return x + v;
        }
        let xr = x.slice(ndarray::s![0..9]).to_owned();
        let vr = v.slice(ndarray::s![0..9]).to_owned();
        let yr = So3.retract(&xr, &vr);
        let mut y = x + v;
        for i in 0..9 {
            y[i] = yr[i];
        }
        y
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::So3;
    use ndarray::array;

    fn eye_t(tx: f64, ty: f64, tz: f64) -> Array1<f64> {
        array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, tx, ty, tz]
    }

    fn rot_block(x: &Array1<f64>) -> [[f64; 3]; 3] {
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = x[3 * i + j];
            }
        }
        r
    }

    #[test]
    fn translation_is_euclidean() {
        let x = eye_t(1.0, 2.0, 3.0);
        let v = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.1, 0.2];
        let y = Se3.retract(&x, &v);
        assert!((y[9] - 1.4).abs() < 1e-15);
        assert!((y[10] - 1.9).abs() < 1e-15);
        assert!((y[11] - 3.2).abs() < 1e-15);
        let r = rot_block(&y);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((r[i][j] - want).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn rotation_block_stays_orthogonal() {
        let x = eye_t(0.0, 0.0, 0.0);
        let mut v = Array1::zeros(12);
        v[1] = -0.2;
        v[3] = 0.2;
        v[9] = 1.0;
        let y = Se3.retract(&x, &v);
        let r = rot_block(&y);
        let mut rtr = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rtr[i][j] = r[0][i] * r[0][j] + r[1][i] * r[1][j] + r[2][i] * r[2][j];
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[i][j] - want).abs() < 1e-12, "{rtr:?}");
            }
        }
        assert!((y[9] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn wrong_dim_is_identity_and_keeps_length() {
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = Se3.retract(&x, &v);
        assert_eq!(y.len(), 114);
        for i in 0..114 {
            assert!((y[i] - (x[i] + v[i])).abs() < 1e-15);
        }
        assert!(Se3.required_dim(114).is_err());
        assert!(Se3.required_dim(12).is_ok());
    }

    /// Length 9 is an SO(3) packing. SE(3) leaves it as `x + v`
    /// instead of retracting a rotation block.
    #[test]
    fn length_nine_is_not_an_so3_block() {
        let x = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let v = array![0.0, -0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y = Se3.retract(&x, &v);
        let p = Se3.project(&x, &v);
        for i in 0..9 {
            assert!(
                (y[i] - (x[i] + v[i])).abs() < 1e-15,
                "retract component {i}"
            );
            assert!((p[i] - v[i]).abs() < 1e-15, "project component {i}");
        }
        let rotated = So3.retract(&x, &v);
        let gap = (0..9)
            .map(|i| {
                let d = y[i] - rotated[i];
                d * d
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            gap > 1e-3,
            "length 9 followed the SO(3) retraction, gap {gap}"
        );
        assert_eq!(Se3.required_dim(9), Err(12));
    }
}
